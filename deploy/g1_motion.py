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
    quat_to_euler,
    quat_to_rotation_6D,
)
from gs_env.common.utils.motion_utils import MotionLib, build_motion_obs_from_dict
from gs_env.sim.envs.config.schema import MotionEnvArgs

# Add examples to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
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
    exp_name: str = "walk",
    num_ckpt: int | None = None,
    device: str = "cpu",
    show_viewer: bool = True,
    sim: bool = True,
    action_scale: float = 0.0,  # only for real robot
    motion_file: str = "./assets/motion/evaluate.pkl",
) -> None:
    """Run policy on either simulation or real robot.

    Args:
        exp_name: Experiment name (subdirectory in deploy/logs)
        num_ckpt: Checkpoint number. If None, loads latest.
        device: Device for policy inference ('cuda' or 'cpu')
        show_viewer: Show viewer (only for sim mode)
        sim: If True, run in simulation. If False, run on real robot.
        num_envs: Number of environments (only for sim mode)
    """
    device = "cpu" if not torch.cuda.is_available() else device
    device_t = torch.device(device)

    # Load checkpoint and env_args
    policy, env_args = load_checkpoint_and_env_args(exp_name, num_ckpt, device)
    env_args = env_args.model_copy(update={"motion_file": motion_file})

    if sim:
        print("Running in SIMULATION mode")
        import gs_env.sim.envs as envs

        envclass = getattr(envs, env_args.env_name)
        env = envclass(
            args=env_args,
            num_envs=1,
            show_viewer=show_viewer,
            device=device_t,
            eval_mode=True,
        )
        env.eval()
        env.reset()

    else:
        print("Running in REAL ROBOT mode")
        from gs_env.real import UnitreeLeggedEnv

        env = UnitreeLeggedEnv(
            env_args,
            action_scale=action_scale,
            interactive=True,
            device=device_t,
            xml_path="assets/robot/unitree_g1/g1_mocap_29dof.xml",
        )

        print("Press Start button to start the policy")
        while not env.robot.Start:
            time.sleep(0.1)

    print("=" * 80)
    print("Starting policy execution")
    print(f"Mode: {'SIMULATION' if sim else 'REAL ROBOT'}")
    print(f"Device: {device}")
    print("=" * 80)

    def deploy_loop() -> None:
        nonlocal env, motion_file

        # Initialize tracking variables
        last_action_t = torch.zeros(1, env.action_dim, device=device_t)
        total_inference_time = 0
        step_id = 0
        action_scale = 0

        # Initialize motion library (direct file playback)
        motion_lib = MotionLib(motion_file=motion_file, device=device_t)
        motion_id_t = torch.tensor(
            [
                0,
            ],
            dtype=torch.long,
            device=device_t,
        )
        t_val = 0.0

        # Initialize motion observation parameters
        motion_obs_steps = motion_lib.get_observed_steps(env_args.observed_steps)
        # Compute tracking_link_idx_local from tracking_link_names
        # Use motion_lib.link_names since it matches the robot structure
        tracking_link_names = env_args.tracking_link_names
        link_names = motion_lib.tracking_link_names
        tracking_link_idx_local = (
            [link_names.index(name) for name in tracking_link_names] if tracking_link_names else []
        )
        envs_idx = torch.tensor([0], dtype=torch.long, device=device_t)

        obs_history = None

        next_step_time = time.time() + 0.02
        start_step_time = time.time()
        while True:
            # Check termination condition (only for real robot)
            if not sim and hasattr(env, "is_emergency_stop") and env.is_emergency_stop:  # type: ignore
                print("Emergency stop triggered!")
                break

            if step_id < 50:
                action_scale += 0.02
                action_scale = min(action_scale, 1.0)

            # Advance motion time and compute reference frame (looping)
            t_val += 0.02
            if t_val > motion_lib.get_motion_length(motion_id_t):
                t_val = 0.0
            motion_time_t = torch.tensor([t_val], dtype=torch.float32, device=device_t)
            (
                ref_base_pos,
                ref_base_quat,
                ref_base_lin_vel,
                ref_base_ang_vel,
                ref_base_ang_vel_local,
                ref_dof_pos,
                ref_dof_vel,
                ref_link_pos_global,
                ref_link_pos_local,
                ref_link_quat_global,
                ref_link_quat_local,
                ref_link_lin_vel,
                ref_link_lin_vel_local,
                ref_link_ang_vel,
                ref_link_ang_vel_local,
                ref_foot_contact,
                ref_foot_contact_weighted,
            ) = motion_lib.get_ref_motion_frame(motion_ids=motion_id_t, motion_times=motion_time_t)

            _ = ref_base_ang_vel_local
            _ = ref_link_pos_global
            _ = ref_link_pos_local
            _ = ref_link_quat_global
            _ = ref_link_quat_local
            _ = ref_link_lin_vel
            _ = ref_link_lin_vel_local
            _ = ref_link_ang_vel
            _ = ref_link_ang_vel_local
            _ = ref_foot_contact
            _ = ref_foot_contact_weighted
            ref_base_euler = quat_to_euler(ref_base_quat)

            # Construct observation (matching training observation structure)
            obs_components = []
            for key in env_args.actor_obs_terms:
                if key == "last_action":
                    obs_gt = last_action_t
                elif key == "motion_obs":
                    # Build motion observation from motion library
                    if len(motion_obs_steps) > 0:
                        curr_motion_obs_dict, future_motion_obs_dict = (
                            motion_lib.get_motion_future_obs(
                                motion_id_t, motion_time_t, motion_obs_steps
                            )
                        )
                        base_quat = env.base_quat
                        obs_gt = build_motion_obs_from_dict(
                            curr_motion_obs_dict,
                            future_motion_obs_dict,
                            envs_idx,
                            tracking_link_idx_local=tracking_link_idx_local,
                            base_quat=base_quat,
                        )
                    else:
                        obs_gt = torch.zeros(1, 0, device=device_t)
                elif key.startswith("ref_"):
                    if key == "ref_base_pos":
                        obs_gt = ref_base_pos
                    elif key == "ref_base_quat":
                        obs_gt = ref_base_quat
                    elif key == "ref_base_lin_vel":
                        obs_gt = ref_base_lin_vel
                    elif key == "ref_base_ang_vel":
                        obs_gt = ref_base_ang_vel
                    elif key == "ref_base_lin_vel_local":
                        obs_gt = ref_base_lin_vel
                    elif key == "ref_base_ang_vel_local":
                        obs_gt = ref_base_ang_vel
                    elif key == "ref_dof_pos":
                        obs_gt = ref_dof_pos
                    elif key == "ref_dof_vel":
                        obs_gt = ref_dof_vel
                    else:
                        # Fallback: try env if it exposes extra ref_* tensors
                        obs_gt = getattr(env, key)
                elif key == "diff_base_yaw":
                    obs_gt = (ref_base_euler[0, 2] - env.base_euler[0, 2]).reshape(1, -1)
                elif key == "diff_base_pos_local_yaw":
                    obs_gt = ref_base_lin_vel * 0.0
                elif key == "diff_tracking_link_pos_local_yaw":
                    diff_pos = env.tracking_link_pos_local_yaw - ref_link_pos_local
                    obs_gt = diff_pos.reshape(1, -1)
                elif key == "diff_tracking_link_rotation_6D":
                    diff_quat = quat_diff(
                        ref_link_quat_local,
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
            with torch.no_grad():
                start_time = time.time()
                action_t = policy(obs_t)
                end_time = time.time()
                total_inference_time += end_time - start_time

            env.apply_action(action_t * action_scale)

            if sim:
                env.time_since_reset[0] = -1.0  # type: ignore
                terminated = env.get_terminated()  # type: ignore
                if terminated[0]:
                    env.reset_idx(torch.IntTensor([0]))  # type: ignore
                    obs_history = None

                ref_quat_yaw = quat_from_angle_axis(
                    ref_base_euler[0, 2],
                    torch.tensor([0, 0, 1], device=env.device, dtype=torch.float),
                )
                link_name_to_idx = {
                    link_name: idx for idx, link_name in enumerate(env.args.tracking_link_names)
                }
                env.scene.scene.clear_debug_objects()  # type: ignore
                for link_name in env.scene.objects.keys():  # type: ignore
                    if link_name in link_name_to_idx:
                        link_idx = link_name_to_idx[link_name]
                        if link_idx < ref_link_pos_local.shape[1]:
                            ref_link_pos = ref_link_pos_local[:, link_idx, :]
                            ref_link_quat = ref_link_quat_local[:, link_idx, :]
                            ref_link_pos = quat_apply(ref_quat_yaw, ref_link_pos)
                            ref_link_pos += ref_base_pos
                            ref_link_quat = quat_mul(ref_quat_yaw, ref_link_quat)
                            env.scene.set_obj_pose(link_name, pos=ref_link_pos, quat=ref_link_quat)  # type: ignore
                        else:
                            continue
                        if link_name == "left_ankle_roll_link":
                            env.scene.scene.draw_debug_arrow(
                                ref_link_pos,
                                ref_foot_contact[0, 0]
                                * torch.tensor([0.0, 0.0, 0.5], device=env.device),
                                radius=0.01,
                                color=(0.0, 0.0, 1.0),
                            )
                        if link_name == "right_ankle_roll_link":
                            env.scene.scene.draw_debug_arrow(
                                ref_link_pos,
                                ref_foot_contact[0, 1]
                                * torch.tensor([0.0, 0.0, 0.5], device=env.device),
                                radius=0.01,
                                color=(0.0, 0.0, 1.0),
                            )
            last_action_t = action_t.clone()
            step_id += 1

            # Control loop timing (50 Hz)
            if time.time() < next_step_time:
                time.sleep(max(0, next_step_time - time.time()))
                next_step_time = next_step_time + 0.02
            else:
                next_step_time = time.time() + 0.02

            if step_id % 100 == 0 and step_id > 0:
                print(f"Step {step_id}: Average inference time: {total_inference_time / 100:.4f}s")
                print(f"Step {step_id}: FPS: {100 / (time.time() - start_step_time):.2f}")
                total_inference_time = 0
                start_step_time = time.time()

    try:
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
