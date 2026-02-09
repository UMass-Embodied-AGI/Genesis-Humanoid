#!/usr/bin/env python3
"""Example: Train PPO on Genesis Walking environment using Genesis RL."""

import glob
import os
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import fire
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend to prevent windows from showing
import gs_env.sim.envs as gs_envs
import torch
from gs_agent.algos.config.registry import PPO_MOTION_MLP, PPOArgs
from gs_agent.algos.ppo import PPO
from gs_agent.runners.config.registry import RUNNER_MOTION_MLP
from gs_agent.runners.config.schema import RunnerArgs
from gs_agent.runners.onpolicy_runner import OnPolicyRunner
from gs_agent.utils.logger import configure as logger_configure
from gs_agent.utils.policy_loader import load_latest_model
from gs_agent.wrappers.gs_env_wrapper import GenesisEnvWrapper
from gs_env.common.utils.math_utils import quat_apply, quat_from_angle_axis, quat_mul
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from gs_env.sim.envs.config.schema import MotionEnvArgs
from gs_env.sim.scenes.config.registry import SceneArgsRegistry
from utils import apply_overrides_generic, config_to_yaml, yaml_to_config


def create_gs_env(
    show_viewer: bool = False,
    num_envs: int = 8192,
    device: str = "cuda",
    args: Any = None,
    eval_mode: bool = False,
) -> gs_envs.MotionEnv:
    """Create gym environment wrapper with optional config overrides."""
    if torch.cuda.is_available() and device == "cuda":
        device_tensor = torch.device("cuda")
    else:
        device_tensor = torch.device("cpu")
    print(f"Using device: {device_tensor}")

    env_class = getattr(gs_envs, args.env_name)

    return env_class(
        args=args,
        num_envs=num_envs,
        show_viewer=show_viewer,
        device=device_tensor,  # type: ignore
        eval_mode=eval_mode,
    )


def create_ppo_runner_from_registry(
    env: gs_envs.MotionEnv,
    exp_name: str | None = None,
    algo_cfg: Any = None,
    runner_args: Any = None,
) -> OnPolicyRunner:
    """Create PPO runner using configuration from the registry."""
    # Environment setup
    wrapped_env = GenesisEnvWrapper(env, device=env.device)

    # Create PPO algorithm
    ppo = PPO(
        env=wrapped_env,
        cfg=algo_cfg,
        device=wrapped_env.device,
    )

    # Create PPO runner
    if exp_name is not None:
        # Avoid mutating a frozen Pydantic model; create a copied config with updated save_path
        runner_args = runner_args.model_copy(update={"save_path": Path(f"./logs/{exp_name}")})
    runner = OnPolicyRunner(
        algorithm=ppo,
        runner_args=runner_args,
        device=wrapped_env.device,
    )
    return runner


def evaluate_policy(
    exp_name: str,
    show_viewer: bool = False,
    num_ckpt: int | None = None,
    device: str = "cuda",
    env_overrides: dict[str, Any] | None = None,
) -> None:
    if env_overrides is None:
        env_overrides = {}
    """Evaluate the policy."""
    print("=" * 80)
    print("EVALUATION MODE: Disabling domain randomization, observation noise, and random push")
    print("=" * 80)

    # Find the experiment directory without creating a new runner
    log_pattern = f"logs/{exp_name}/*"
    log_dirs = glob.glob(log_pattern)
    if not log_dirs:
        raise FileNotFoundError(f"No experiment directories found matching pattern: {log_pattern}")

    log_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    exp_dir = log_dirs[0]
    print(f"Loading policy from experiment: {exp_dir}")

    # Load checkpoint - either specific one or latest
    if num_ckpt is not None:
        ckpt_path = Path(exp_dir) / "checkpoints" / f"checkpoint_{num_ckpt:04d}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")
    else:
        ckpt_path = load_latest_model(Path(exp_dir))
        num_ckpt = int(ckpt_path.stem.split("_")[-1])

    print(f"Loading checkpoint: {ckpt_path}")

    print(f"Loading configs from experiment: {exp_dir}")

    env_args = yaml_to_config(Path(exp_dir) / "configs" / "env_args.yaml", MotionEnvArgs)
    algo_cfg = yaml_to_config(Path(exp_dir) / "configs" / "algo_cfg.yaml", PPOArgs)
    env_args = apply_overrides_generic(env_args, env_overrides, prefixes=("cfgs.", "env."))

    # Disable domain randomization, obs noise, and random push for evaluation
    # Create a copy of env_args with disabled randomization
    env_args = env_args.model_copy(
        update={
            "obs_noises": {},  # Disable observation noise
        }
    )
    env_args = cast(MotionEnvArgs, env_args).model_copy(
        update={"scene_args": SceneArgsRegistry["custom_scene_g1_mocap"]}
    )
    # Update robot args to disable domain randomization
    from gs_env.sim.robots.config.schema import DomainRandomizationArgs

    robot_args = env_args.robot_args.model_copy(
        update={
            "dr_args": DomainRandomizationArgs(
                kp_range=(1.0, 1.0),
                kd_range=(1.0, 1.0),
                motor_strength_range=(1.0, 1.0),
                motor_offset_range=(0.0, 0.0),
                friction_range=(1.0, 1.0),
                mass_range=(0.0, 0.0),
                com_displacement_range=(0.0, 0.0),
            )
        }
    )
    env_args = env_args.model_copy(update={"robot_args": robot_args})

    # Create environment for evaluation
    env = create_gs_env(
        show_viewer=show_viewer,
        num_envs=1,
        device=device,
        args=env_args,
        eval_mode=True,
    )

    wrapped_env = GenesisEnvWrapper(env, device=env.device)

    # Setup GIF recording if not showing viewer
    gif_path = None
    if not show_viewer:
        # Create gif directory structure
        gif_dir = Path("./gif") / exp_name
        gif_dir.mkdir(parents=True, exist_ok=True)

        gif_path = gif_dir / f"{num_ckpt}.gif"
        print(f"Will save GIF to: {gif_path}")

        # Start rendering
        env.start_rendering()  # type: ignore

    # Create PPO algorithm and load checkpoint
    ppo = PPO(
        env=wrapped_env,
        cfg=algo_cfg,
        device=wrapped_env.device,
    )
    ppo.load(ckpt_path, load_optimizer=False)

    # Get inference policy
    inference_policy = ppo.get_inference_policy()

    print("Starting evaluation...")

    def evaluate() -> None:
        nonlocal \
            wrapped_env, \
            inference_policy, \
            gif_path, \
            show_viewer, \
            num_ckpt, \
            exp_name, \
            env_args, \
            env
        if show_viewer:
            print("Running endlessly (press Ctrl+C to stop)")
        else:
            print("Running until all environments are done")

        # Reset environment
        obs, _ = wrapped_env.get_observations()  # Unpack actor and critic obs, use actor for policy

        # Create a wrapper that always uses deterministic=True
        class DeterministicWrapper(torch.nn.Module):
            def __init__(self, policy: Any) -> None:
                super().__init__()
                self.policy = policy

            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                action, _ = self.policy(obs, deterministic=True)
                return action

        # Wrap and trace the policy with deterministic=True baked in
        wrapped_policy = DeterministicWrapper(inference_policy)
        inference_policy = torch.jit.trace(wrapped_policy, obs)

        # Save the JIT-traced policy and env_args to deploy folder
        print("Saving JIT-traced policy and env_args for deployment...")

        # Create deploy directory structure
        deploy_dir = Path("./deploy/logs") / exp_name
        deploy_dir.mkdir(parents=True, exist_ok=True)

        # Save JIT-traced policy
        jit_policy_path = deploy_dir / f"checkpoint_{num_ckpt}.pt"
        torch.jit.save(inference_policy, str(jit_policy_path))
        print(f"JIT-traced policy saved to: {jit_policy_path}")

        # Save env_args as YAML
        env_args_path = deploy_dir / "env_args.yaml"
        config_to_yaml(env_args, env_args_path)
        print(f"Environment args saved to: {env_args_path}")

        motion_id = 0
        step_count = 0

        link_name_to_idx = {}
        for link_name in env.scene.objects.keys():
            link_name_to_idx[link_name] = env_args.tracking_link_names.index(link_name)
        foot_link_tracking_idx = [
            env_args.tracking_link_names.index(name) for name in env_args.robot_args.foot_link_names
        ]

        while True:
            env.time_since_reset[0] = 0.0
            env.hard_reset_motion(
                torch.IntTensor(
                    [
                        0,
                    ]
                ),
                motion_id,
            )
            env.hard_sync_motion(torch.IntTensor([0]))
            obs = wrapped_env.obs
            while (
                env.motion_times[0]
                < env.motion_lib.get_motion_length(torch.IntTensor([motion_id])) - 0.02
            ):
                # Get action from policy
                with torch.no_grad():
                    action = inference_policy(obs)  # type: ignore[misc]

                # Step environment
                env.apply_action(action)
                terminated = env.get_terminated()
                if terminated[0]:
                    env.hard_sync_motion(torch.IntTensor([0]))
                env.update_history()
                wrapped_env.update_obs_history()
                obs = wrapped_env.obs

                ref_quat_yaw = quat_from_angle_axis(
                    env.ref_base_euler[:, 2],
                    torch.tensor([0, 0, 1], device=env.device, dtype=torch.float),
                )
                for link_name in env.scene.objects.keys():
                    ref_link_pos = env.ref_tracking_link_pos_local_yaw[
                        :, link_name_to_idx[link_name]
                    ]
                    ref_link_quat = env.ref_tracking_link_quat_local_yaw[
                        :, link_name_to_idx[link_name]
                    ]
                    ref_link_pos = quat_apply(ref_quat_yaw, ref_link_pos)
                    ref_link_pos += env.ref_base_pos
                    ref_link_quat = quat_mul(ref_quat_yaw, ref_link_quat)
                    env.scene.set_obj_pose(link_name, pos=ref_link_pos, quat=ref_link_quat)
                env.scene.scene.clear_debug_objects()
                for i in range(len(env.robot.foot_links_idx)):
                    env.scene.scene.draw_debug_arrow(
                        env.link_positions[0, env.robot.foot_links_idx[i]],
                        env.foot_contact_weighted[0, i]
                        * torch.tensor([0.0, 0.0, 1.0], device=env.device),
                        radius=0.01,
                        color=(0.0, 1.0, 0.0),
                    )
                    foot_link_pos = env.ref_tracking_link_pos_local_yaw[
                        :, foot_link_tracking_idx[i]
                    ]
                    foot_link_pos = quat_apply(ref_quat_yaw, foot_link_pos)
                    foot_link_pos += env.ref_base_pos
                    env.scene.scene.draw_debug_arrow(
                        foot_link_pos,
                        env.ref_foot_contact[0, i]
                        * torch.tensor([0.0, 0.0, 0.5], device=env.device),
                        radius=0.01,
                        color=(0.0, 0.0, 1.0),
                    )

                step_count += 1

                # Print progress
                if step_count % 50 == 0:
                    print(f"Step {step_count}")

                # Stop rendering and save GIF if recording
                if not show_viewer and gif_path is not None and step_count >= 500:
                    print("Stopping rendering and saving GIF...")
                    env.stop_rendering(save_gif=True, gif_path=str(gif_path))  # type: ignore
                    print(f"GIF saved to: {gif_path}")
                    exit()

            env.time_since_reset[0] = 0.0
            while True:
                action = input(
                    "Enter n to play next motion, q to quit, r to replay current motion, p to play previous motion, id to play specific motion\n"
                )
                if action == "n":
                    motion_id = (motion_id + 1) % env.motion_lib.num_motions
                    break
                elif action == "q":
                    return
                elif action == "r":
                    break
                elif action == "p":
                    motion_id = (motion_id - 1) % env.motion_lib.num_motions
                    break
                elif action.isdigit():
                    motion_id = int(action)
                    break
                else:
                    print("Invalid action")
                    return

    try:
        evaluate()
    except KeyboardInterrupt:
        pass


def resume_training(
    exp_name: str,
    num_ckpt: int | None = None,
    show_viewer: bool = False,
    num_envs: int = 8192,
    device: str = "cuda",
    use_wandb: bool = True,
    use_stored_config: bool = False,
    env_args: Any = None,
    algo_cfg: Any = None,
    runner_args: Any = None,
    env_overrides: dict[str, Any] | None = None,
    algo_overrides: dict[str, Any] | None = None,
    runner_overrides: dict[str, Any] | None = None,
) -> None:
    """Resume training from a checkpoint.

    Args:
        exp_name: Name of the experiment to resume
        num_ckpt: Checkpoint number to resume from. If None, uses latest.
        show_viewer: Whether to show viewer
        num_envs: Number of parallel environments
        device: Device to use ("cuda" or "cpu")
        use_wandb: Whether to use wandb logging
        use_stored_config: If True, use stored algorithm config (resume original training).
                          If False, use PPO config (load checkpoint and train like train_policy).
        env_args: Environment config
        algo_cfg: Algorithm config
        runner_args: Runner config
    """
    print("=" * 80)
    print("RESUME MODE: Loading checkpoint and continuing training")
    print("=" * 80)

    # Find the experiment directory
    log_pattern = f"logs/{exp_name}/*"
    log_dirs = glob.glob(log_pattern)
    if not log_dirs:
        raise FileNotFoundError(f"No experiment directories found matching pattern: {log_pattern}")

    log_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    exp_dir = log_dirs[0]
    print(f"Resuming from experiment: {exp_dir}")

    # Load checkpoint
    if num_ckpt is not None:
        ckpt_path = Path(exp_dir) / "checkpoints" / f"checkpoint_{num_ckpt:04d}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")
    else:
        ckpt_path = load_latest_model(Path(exp_dir))
        num_ckpt = int(ckpt_path.stem.split("_")[-1])

    print(f"Resuming from checkpoint: {ckpt_path}")

    # Determine algorithm config based on use_stored_config flag
    if use_stored_config:
        # Load configs
        print(f"Loading configs from experiment: {exp_dir}")
        env_args = yaml_to_config(Path(exp_dir) / "configs" / "env_args.yaml", MotionEnvArgs)
        algo_cfg = yaml_to_config(Path(exp_dir) / "configs" / "algo_cfg.yaml", PPOArgs)
        runner_args = yaml_to_config(Path(exp_dir) / "configs" / "runner_args.yaml", RunnerArgs)
    if env_overrides is not None:
        env_args = apply_overrides_generic(env_args, env_overrides, prefixes=("cfgs.", "env."))
    if algo_overrides is not None:
        algo_cfg = apply_overrides_generic(algo_cfg, algo_overrides, prefixes=("cfgs.", "algo."))
    if runner_overrides is not None:
        runner_args = apply_overrides_generic(
            runner_args, runner_overrides, prefixes=("cfgs.", "runner.")
        )

    env_args = cast(MotionEnvArgs, env_args).model_copy(
        update={"scene_args": SceneArgsRegistry["flat_scene_legged"]}
    )

    # Create environment
    env = create_gs_env(
        show_viewer=show_viewer,
        num_envs=num_envs,
        device=device,
        args=env_args,
    )

    wrapped_env = GenesisEnvWrapper(env, device=env.device)

    algorithm = PPO(env=wrapped_env, cfg=algo_cfg, device=wrapped_env.device)

    # Load checkpoint
    try:
        algorithm.load(ckpt_path, load_optimizer=True)
        start_iteration = algorithm.current_iter
        print(f"Successfully loaded checkpoint. Resuming from iteration: {start_iteration}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

    # Update runner args to continue from where we left off
    if use_stored_config:
        remaining_iterations = runner_args.total_iterations - algorithm.current_iter
    else:
        remaining_iterations = runner_args.total_iterations
    if remaining_iterations <= 0:
        print("Training already completed!")
        return

    print(f"Continuing training for {remaining_iterations} iterations")

    # Create new experiment name for resumed training
    resume_exp_name = f"{exp_name}_resume"
    resume_save_path = Path(f"./logs/{resume_exp_name}")

    # Create runner with new save path for resumed training
    runner = OnPolicyRunner(
        algorithm=algorithm,
        runner_args=runner_args.model_copy(
            update={"save_path": resume_save_path, "total_iterations": remaining_iterations}
        ),
        device=wrapped_env.device,
    )

    # Set up logging for new experiment
    logger = logger_configure(
        folder=str(runner.save_dir),
        format_strings=["stdout", "csv", "wandb"],
        entity=None,
        project=None,
        exp_name=resume_exp_name,
        mode="online" if use_wandb and (not show_viewer) else "disabled",
    )

    # Save configuration files to YAML in new experiment directory
    print("Saving configuration files to YAML...")
    config_dir = runner.save_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_to_yaml(env_args, config_dir / "env_args.yaml")
    config_to_yaml(algo_cfg, config_dir / "algo_cfg.yaml")
    config_to_yaml(runner_args, config_dir / "runner_args.yaml")

    def train() -> None:
        nonlocal runner, logger
        train_summary_info = runner.train(metric_logger=logger)
        print("Training completed successfully!")
        print(f"Training completed in {train_summary_info['total_time']:.2f} seconds.")
        print(f"Total iterations: {train_summary_info['total_iterations']}.")
        print(f"Total steps: {train_summary_info['total_steps']}.")
        print(f"Final reward: {train_summary_info['final_reward']:.2f}.")

    try:
        train()
    except KeyboardInterrupt:
        pass


def train_policy(
    exp_name: str | None = None,
    show_viewer: bool = False,
    num_envs: int = 8192,
    device: str = "cuda",
    use_wandb: bool = True,
    env_args: Any = None,
    algo_cfg: Any = None,
    runner_args: Any = None,
) -> None:
    """Train the policy using PPO."""

    env_args = cast(MotionEnvArgs, env_args).model_copy(
        update={"scene_args": SceneArgsRegistry["flat_scene_legged"]}
    )

    # Create environment
    env = create_gs_env(
        show_viewer=show_viewer,
        num_envs=num_envs,
        device=device,
        args=env_args,
    )

    print(algo_cfg)
    # Get configuration and runner from registry
    runner = create_ppo_runner_from_registry(
        env,
        exp_name=exp_name,
        algo_cfg=algo_cfg,
        runner_args=runner_args,
    )

    # Set up logging with proper configuration
    if exp_name is not None:
        save_path = Path(f"./logs/{exp_name}")
    else:
        save_path = runner_args.save_path
    logger_folder = save_path / datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logger_configure(
        folder=str(logger_folder),
        format_strings=["stdout", "csv", "wandb"],
        entity=None,
        project=None,
        exp_name=exp_name,
        mode="online" if use_wandb and (not show_viewer) else "disabled",
    )

    # Save configuration files to YAML
    print("Saving configuration files to YAML...")
    config_dir = logger_folder / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_to_yaml(env_args, config_dir / "env_args.yaml")
    config_to_yaml(algo_cfg, config_dir / "algo_cfg.yaml")
    config_to_yaml(runner_args, config_dir / "runner_args.yaml")

    # Train using Runner
    print("Starting training...")

    def train() -> None:
        nonlocal runner, logger
        train_summary_info = runner.train(metric_logger=logger)
        print("Training completed successfully!")
        print(f"Training completed in {train_summary_info['total_time']:.2f} seconds.")
        print(f"Total iterations: {train_summary_info['total_iterations']}.")
        print(f"Total steps: {train_summary_info['total_steps']}.")
        print(f"Total reward: {train_summary_info['final_reward']:.2f}.")

    try:
        train()
    except KeyboardInterrupt:
        pass


def main(
    num_envs: int = 8192,
    show_viewer: bool = False,
    device: str = "cuda",
    eval: bool = False,
    resume: bool = False,
    use_stored_config: bool = False,
    exp_name: str | None = None,
    num_ckpt: int | None = None,
    use_wandb: bool = True,
    env_name: str = "g1_motion_teacher",
    **cfg_overrides: Any,
) -> None:
    """Entry point.

    You can override configs via dot-notation, e.g.:
    --env.reward_args.AngVelZReward 5
    --reward_args.G1BaseHeightPenalty 50
    --algo.lr 1e-4
    --runner.total_iterations 2000
    """
    # Bucket overrides into env / algo / runner
    env_overrides: dict[str, Any] = {}
    algo_overrides: dict[str, Any] = {}
    runner_overrides: dict[str, Any] = {}

    for k, v in cfg_overrides.items():
        if k.startswith("cfgs.env.") or k.startswith("env.") or k.startswith("reward_args."):
            env_overrides[k] = v
            continue
        if k.startswith("cfgs.algo.") or k.startswith("algo."):
            algo_overrides[k] = v
            continue
        if k.startswith("cfgs.runner.") or k.startswith("runner."):
            runner_overrides[k] = v
            continue

    env_args = EnvArgsRegistry[env_name]
    env_args = apply_overrides_generic(env_args, env_overrides, prefixes=("cfgs.", "env."))
    algo_cfg = apply_overrides_generic(PPO_MOTION_MLP, algo_overrides, prefixes=("cfgs.", "algo."))
    runner_args = apply_overrides_generic(
        RUNNER_MOTION_MLP, runner_overrides, prefixes=("cfgs.", "runner.")
    )

    if resume:
        # Resume mode
        assert exp_name is not None, "exp_name is required for resume"
        resume_training(
            exp_name=exp_name,
            num_ckpt=num_ckpt,
            show_viewer=show_viewer,
            num_envs=num_envs,
            device=device,
            use_wandb=use_wandb,
            use_stored_config=use_stored_config,
            env_args=env_args,
            algo_cfg=algo_cfg,
            runner_args=runner_args,
            env_overrides=env_overrides,
            algo_overrides=algo_overrides,
            runner_overrides=runner_overrides,
        )
        return
    elif eval:
        # Evaluation mode - don't create runner to avoid creating empty log dir
        print("Evaluation mode: Loading trained policy")
        assert exp_name is not None, "exp_name is required for evaluation"
        evaluate_policy(
            exp_name=exp_name,
            show_viewer=show_viewer,
            num_ckpt=num_ckpt,
            device=device,
            env_overrides=env_overrides,
        )
    else:
        # Training mode
        print("Training mode: Starting policy training")
        train_policy(
            exp_name=exp_name,
            show_viewer=show_viewer,
            num_envs=num_envs,
            device=device,
            use_wandb=use_wandb,
            env_args=env_args,
            algo_cfg=algo_cfg,
            runner_args=runner_args,
        )


if __name__ == "__main__":
    fire.Fire(main)
