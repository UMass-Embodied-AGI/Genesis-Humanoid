import sys
import time
from pathlib import Path

import fire
import torch
from gs_env.common.utils.math_utils import (
    quat_apply,
    quat_diff,
    quat_from_angle_axis,
    quat_mul,
    quat_to_rotation_6D,
)
from gs_env.common.utils.motion_utils import build_motion_obs_from_dict
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from gs_env.sim.envs.config.schema import MotionEnvArgs

# Add examples to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from deploy.utils import RedisClient
from examples.utils import yaml_to_config  # type: ignore


def load_checkpoint_and_env_args(
    exp_name: str, num_ckpt: int | None = None, device: str = "cpu"
) -> tuple[torch.jit.ScriptModule, MotionEnvArgs]:
    """Load JIT checkpoint and env_args from deploy/logs directory.

    Args:
        exp_name: Experiment name
        num_ckpt: Checkpoint number. If None, loads the latest checkpoint.

    Returns:
        Tuple of (checkpoint_path, env_args)
    """

    deploy_dir = Path(__file__).parent / "logs" / exp_name
    if not deploy_dir.exists():
        raise FileNotFoundError(f"Deploy directory not found: {deploy_dir}")

    # Load env_args from YAML
    env_args_path = deploy_dir / "env_args.yaml"
    if not env_args_path.exists():
        raise FileNotFoundError(f"env_args.yaml not found: {env_args_path}")

    print(f"Loading env_args from: {env_args_path}")
    env_args = yaml_to_config(env_args_path, MotionEnvArgs)

    # Load checkpoint
    if num_ckpt is not None:
        ckpt_path = deploy_dir / f"checkpoint_{num_ckpt:04d}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        # Find latest checkpoint
        ckpts = list(deploy_dir.glob("checkpoint_*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {deploy_dir}")
        ckpt_path = max(ckpts, key=lambda p: int(p.stem.split("_")[-1]))

    print(f"Loading checkpoint from: {ckpt_path}")
    # Load policy
    policy = torch.jit.load(str(ckpt_path))
    policy.to(device)
    policy.eval()

    return policy, env_args


def main(
    exp_name: str = "bc_motion",
    num_ckpt: int | None = None,
    device: str = "cpu",
    show_viewer: bool = True,
    sim: bool = True,
    view: bool = False,
    action_scale: float = 0.0,  # only for real robot
    redis_url: str = "redis://localhost:6379/0",
    redis_key: str = "motion:ref:latest",
    rate_limit: float = 100.0,
) -> None:
    """Run BC motion policy on either simulation or real robot.

    This script deploys policies trained by run_bc_motion.py.

    Args:
        exp_name: Experiment name (subdirectory in deploy/logs)
        num_ckpt: Checkpoint number. If None, loads latest.
        device: Device for policy inference ('cuda' or 'cpu')
        show_viewer: Show viewer (only for sim mode)
        sim: If True, run in simulation. If False, run on real robot.
        view: If True, view motion reference data instead of running policy
        action_scale: Action scaling factor (only for real robot)
        redis_url: Redis URL for motion reference data
        redis_key: Redis key for motion reference data
    """
    device = "cpu" if not torch.cuda.is_available() else device

    if view:
        policy = None
        env_args = EnvArgsRegistry["g1_motion"]
    else:
        # Load checkpoint and env_args
        policy, env_args = load_checkpoint_and_env_args(exp_name, num_ckpt, device)

    tracking_link_names = getattr(env_args, "tracking_link_names", [])
    num_tracking_links = len(tracking_link_names)

    # Initialize Redis client with tracking links
    redis_client = RedisClient(
        url=redis_url, key=redis_key, device=device, num_tracking_links=num_tracking_links
    )
    motion_elements = list(getattr(env_args, "observed_steps", {}).keys())
    redis_client.set_motion_obs_elements(motion_elements)

    if sim:
        print("Running in SIMULATION mode")
        import gs_env.sim.envs as envs

        envclass = getattr(envs, env_args.env_name)
        env = envclass(
            args=env_args,
            num_envs=1,
            show_viewer=show_viewer,
            device=torch.device(device),
            eval_mode=True,
        )
        env.eval()
        env.reset()

        # base_euler = torch.tensor([0.0, 0.0, 3.14], device=env.device)
        # base_quat = quat_from_euler(base_euler)
        # env.robot.set_state(quat=base_quat)
        # env.update_buffers()

    else:
        if view:
            raise ValueError("View mode is only supported in simulation mode")
        print("Running in REAL ROBOT mode")
        from gs_env.real import UnitreeLeggedEnv

        env = UnitreeLeggedEnv(
            env_args,
            action_scale=action_scale,
            interactive=True,
            device=torch.device(device),
            xml_path="assets/robot/unitree_g1/g1_mocap_29dof.xml",
        )

        print("Press Start button to start the policy")
        while not env.robot.Start:
            time.sleep(0.1)

    if view and sim:
        print("=" * 80)
        print("Starting motion visualization")
        print("Mode: SIMULATION (VIEW)")
        print(f"Device: {device}")
        print(f"Redis URL: {redis_url}")
        print(f"Redis Key: {redis_key}")
        print("=" * 80)
    else:
        print("=" * 80)
        print("Starting policy execution")
        print(f"Mode: {'SIMULATION' if sim else 'REAL ROBOT'}")
        print(f"Device: {device}")
        print("=" * 80)

    def view_loop() -> None:
        """View motion reference data from Redis."""

        nonlocal env, redis_client, tracking_link_names
        assert sim, "View mode only works in simulation"
        assert hasattr(env, "scene"), "Environment must have scene attribute for view mode"

        # Build link_name_to_idx mapping: index in tracking_link_names list
        link_name_to_idx = {link_name: idx for idx, link_name in enumerate(tracking_link_names)}

        last_update_time = time.time()

        start_step_time = time.time()
        step_id = 0
        while True:
            # Control loop timing (50 Hz)
            if time.time() - last_update_time < 0.02:
                time.sleep(0.001)
                continue
            last_update_time = time.time()

            # Update reference values from Redis
            redis_client.update()

            # Transform link positions from local_yaw to global coordinates
            env.robot.set_state(  # type: ignore[attr-defined]
                pos=redis_client.ref_base_pos,
                quat=redis_client.ref_base_quat,
                dof_pos=redis_client.ref_dof_pos,
                dof_vel=redis_client.ref_dof_vel,
                lin_vel=redis_client.ref_base_lin_vel_local,
                ang_vel=redis_client.ref_base_ang_vel_local,
            )
            ref_quat_yaw = quat_from_angle_axis(
                redis_client.ref_base_euler[:, 2],
                torch.tensor([0, 0, 1], device=env.device, dtype=torch.float),
            )
            env.scene.scene.clear_debug_objects()
            for link_name in env.scene.objects.keys():  # type: ignore
                if link_name in link_name_to_idx:
                    link_idx = link_name_to_idx[link_name]
                    if link_idx < redis_client.link_pos_local_yaw.shape[1]:
                        ref_link_pos = redis_client.link_pos_local_yaw[:, link_idx, :]
                        ref_link_quat = redis_client.link_quat_local_yaw[:, link_idx, :]
                        ref_link_pos = quat_apply(ref_quat_yaw, ref_link_pos)
                        ref_link_pos += redis_client.ref_base_pos
                        ref_link_quat = quat_mul(ref_quat_yaw, ref_link_quat)
                        env.scene.set_obj_pose(link_name, pos=ref_link_pos, quat=ref_link_quat)  # type: ignore
                    else:
                        continue
                    if link_name == "left_ankle_roll_link":
                        env.scene.scene.draw_debug_arrow(
                            ref_link_pos,
                            redis_client.ref_foot_contact[0, 0]
                            * torch.tensor([0.0, 0.0, 1.0], device=env.device),
                            radius=0.01,
                            color=(0.0, 0.0, 1.0),
                        )
                    if link_name == "right_ankle_roll_link":
                        env.scene.scene.draw_debug_arrow(
                            ref_link_pos,
                            redis_client.ref_foot_contact[0, 1]
                            * torch.tensor([0.0, 0.0, 1.0], device=env.device),
                            radius=0.01,
                            color=(0.0, 0.0, 1.0),
                        )

            env.scene.scene.step()  # type: ignore
            step_id += 1
            if step_id % 100 == 0 and step_id > 0:
                print(
                    f"Step {step_id}: Average step time: {(time.time() - start_step_time) / 100:.4f}s"
                )
                start_step_time = time.time()

    def deploy_loop() -> None:
        nonlocal env, redis_client, tracking_link_names
        # Initialize tracking variables
        last_action_t = torch.zeros(1, env.action_dim, device=device)
        total_inference_time = 0
        step_id = 0
        action_scale = 0

        # Build link_name_to_idx mapping: index in tracking_link_names list
        link_name_to_idx = {link_name: idx for idx, link_name in enumerate(tracking_link_names)}
        motion_obs_elements = list(getattr(env_args, "observed_steps", {}).keys())

        redis_client.update()
        # redis_client.update_quat(env.base_quat)

        obs_history = None

        rate_limit_dt = 1.0 / rate_limit
        next_step_time = time.time() + max(0.02, rate_limit_dt)
        start_step_time = time.time()
        while True:
            # Check termination condition (only for real robot)
            if not sim and hasattr(env, "is_emergency_stop") and env.is_emergency_stop:  # type: ignore
                print("Emergency stop triggered!")
                break
            if step_id < 50:
                action_scale += 0.02
                action_scale = min(action_scale, 1.0)

            # Update reference values from Redis (zeros if unavailable)
            redis_client.update()

            # Construct observation (matching training observation structure)
            obs_components = []
            for key in env_args.actor_obs_terms:
                if key == "last_action":
                    obs_gt = last_action_t
                elif key.startswith("ref_"):
                    obs_gt = getattr(redis_client, key).reshape(1, -1)
                elif key == "motion_obs":
                    curr_dict = {
                        "base_pos": redis_client.last_ref_base_pos,
                        "base_quat": redis_client.last_ref_base_quat,
                    }
                    future_dict = redis_client.get_future_dict(motion_obs_elements)
                    base_quat = env.base_quat
                    start = time.perf_counter()
                    start_t = time.thread_time()
                    obs_gt = build_motion_obs_from_dict(
                        curr_dict,
                        future_dict,
                        torch.tensor([0], dtype=torch.long, device=device),
                        base_quat=base_quat,
                    )
                    end = time.perf_counter()
                    end_t = time.thread_time()
                    if end - start > 0.05:
                        print(f"motion_obs compute took {end - start:.4f}s")
                        print(f"motion_obs compute thread time: {end_t - start_t:.4f}s")
                elif key == "diff_base_yaw":
                    obs_gt = (redis_client.ref_base_euler[:, 2] - env.base_euler[:, 2]).reshape(
                        1, -1
                    )
                elif key == "diff_base_pos_local_yaw":
                    obs_gt = redis_client.ref_base_lin_vel_local.reshape(1, -1) * 0.0
                elif key == "diff_tracking_link_pos_local_yaw":
                    diff_pos = env.tracking_link_pos_local_yaw - redis_client.link_pos_local_yaw
                    obs_gt = diff_pos.reshape(1, -1)
                elif key == "diff_tracking_link_rotation_6D":
                    diff_quat = quat_diff(
                        redis_client.link_quat_local_yaw,
                        env.tracking_link_quat_local_yaw,
                    )
                    obs_gt = quat_to_rotation_6D(diff_quat).reshape(1, -1)
                else:
                    obs_gt = getattr(env, key)
                obs_gt = obs_gt * env_args.obs_scales.get(key, 1.0)
                obs_components.append(obs_gt)
            obs_t = torch.cat(obs_components, dim=-1)
            if obs_history is None:
                obs_history = torch.zeros_like(obs_t.reshape(-1, 1)).repeat(
                    1, env_args.obs_history_len
                )
            obs_history = torch.cat([obs_history[:, 1:], obs_t.reshape(-1, 1)], dim=1)
            obs_t = obs_history.clone().reshape(1, -1)

            # Get action from policy
            assert policy is not None, "Policy must be loaded for deploy mode"
            with torch.no_grad():
                start_time = time.time()
                action_t = policy(obs_t)
                end_time = time.time()
                total_inference_time += end_time - start_time

            # print(action_t)
            env.apply_action(action_t * action_scale)

            if sim:
                env.time_since_reset[0] = -1.0  # type: ignore
                terminated = env.get_terminated()  # type: ignore
                if terminated[0]:
                    env.reset_idx(torch.IntTensor([0]))  # type: ignore
                ref_quat_yaw = quat_from_angle_axis(
                    redis_client.ref_base_euler[:, 2],
                    torch.tensor([0, 0, 1], device=env.device, dtype=torch.float),
                )
                for link_name in env.scene.objects.keys():  # type: ignore
                    if link_name in link_name_to_idx:
                        link_idx = link_name_to_idx[link_name]
                        if link_idx < redis_client.link_pos_local_yaw.shape[1]:
                            ref_link_pos = redis_client.link_pos_local_yaw[:, link_idx, :]
                            ref_link_quat = redis_client.link_quat_local_yaw[:, link_idx, :]
                            ref_link_pos = quat_apply(ref_quat_yaw, ref_link_pos)
                            ref_link_pos += redis_client.ref_base_pos
                            ref_link_quat = quat_mul(ref_quat_yaw, ref_link_quat)
                            env.scene.set_obj_pose(link_name, pos=ref_link_pos, quat=ref_link_quat)  # type: ignore

            last_action_t = action_t.clone()
            step_id += 1

            # Control loop timing (50 Hz)
            if time.time() < next_step_time:
                time.sleep(max(0, next_step_time - time.time()))
                next_step_time = next_step_time + max(0.02, rate_limit_dt)
            else:
                next_step_time = time.time() + max(0.02, rate_limit_dt)

            if step_id % 100 == 0 and step_id > 0:
                print(f"Step {step_id}: Average inference time: {total_inference_time / 100:.4f}s")
                print(f"Step {step_id}: FPS: {100 / (time.time() - start_step_time):.2f}")
                total_inference_time = 0
                start_step_time = time.time()

    try:
        if view and sim:
            view_loop()
        else:
            deploy_loop()
    except KeyboardInterrupt:
        if not sim:
            env.emergency_stop()
        print("\nKeyboardInterrupt received, stopping...")
    finally:
        if not sim:
            print("Stopping robot handler...")
            # Handler cleanup if needed
        else:
            print("Simulation stopped.")


if __name__ == "__main__":
    fire.Fire(main)
