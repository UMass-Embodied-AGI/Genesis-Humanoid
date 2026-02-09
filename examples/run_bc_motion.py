#!/usr/bin/env python3
"""Example: Train BC to distill policy from g1_motion_teacher to g1_motion."""

import glob
import os
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import fire
import gs_env.sim.envs as gs_envs
import torch
from gs_agent.algos.bc import BC
from gs_agent.algos.config.registry import BC_MOTION_MLP
from gs_agent.algos.config.schema import BCArgs, PPOArgs
from gs_agent.runners.config.registry import RUNNER_BC_MOTION_MLP
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
    """Create Genesis Motion environment with optional config overrides."""
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


def create_bc_runner_from_registry(
    env: gs_envs.MotionEnv,
    teacher_exp_name: str,
    exp_name: str | None = None,
    teacher_ckpt: int | None = None,
    algo_cfg: Any = None,
    runner_args: Any = None,
) -> tuple[OnPolicyRunner, Path]:
    """Create BC runner to distill from teacher to student.

    Args:
        env: Student environment (g1_motion)
        teacher_exp_name: Name of teacher experiment directory (e.g., "g1_motion_teacher")
        exp_name: Name of BC experiment directory
        teacher_ckpt: Checkpoint number to load from teacher. If None, loads latest.
        algo_cfg: BC algorithm configuration (from registry)
        runner_args: Runner configuration (from registry)
    """
    # Environment setup
    wrapped_env = GenesisEnvWrapper(env, device=env.device)

    # Find teacher experiment directory
    log_pattern = f"logs/{teacher_exp_name}/*"
    log_dirs = glob.glob(log_pattern)
    if not log_dirs:
        raise FileNotFoundError(
            f"No teacher experiment directories found matching pattern: {log_pattern}"
        )

    log_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    teacher_exp_dir = log_dirs[0]
    print(f"Loading teacher from experiment: {teacher_exp_dir}")

    # Load teacher checkpoint
    if teacher_ckpt is not None:
        teacher_ckpt_path = (
            Path(teacher_exp_dir) / "checkpoints" / f"checkpoint_{teacher_ckpt:04d}.pt"
        )
        if not teacher_ckpt_path.exists():
            raise FileNotFoundError(f"Teacher checkpoint {teacher_ckpt_path} not found")
    else:
        teacher_ckpt_path = load_latest_model(Path(teacher_exp_dir))
        teacher_ckpt = int(teacher_ckpt_path.stem.split("_")[-1])

    print(f"Loading teacher checkpoint: {teacher_ckpt_path}")

    # Load teacher environment config
    teacher_config_dir = Path(teacher_exp_dir) / "configs"
    if not teacher_config_dir.exists():
        raise FileNotFoundError(
            f"Teacher config not found at {teacher_config_dir}. "
            "Make sure the teacher experiment has configs/env_args.yaml"
        )
    print(f"Loading teacher config from: {teacher_config_dir}")

    # Update algo_cfg with teacher path and config
    teacher_algo_cfg = yaml_to_config(teacher_config_dir / "algo_cfg.yaml", PPOArgs)
    algo_cfg = algo_cfg.model_copy(
        update={
            "teacher_path": teacher_ckpt_path,
            "teacher_config_path": teacher_config_dir / "env_args.yaml",
            "teacher_backbone": teacher_algo_cfg.policy_backbone,
        }
    )

    # Create BC algorithm
    bc = BC(
        env=wrapped_env,
        cfg=algo_cfg,
        device=wrapped_env.device,
    )

    # Create BC runner
    if exp_name is not None:
        # Avoid mutating a frozen Pydantic model; create a copied config with updated save_path
        runner_args = runner_args.model_copy(update={"save_path": Path(f"./logs/{exp_name}")})
    runner = OnPolicyRunner(
        algorithm=bc,
        runner_args=runner_args,
        device=wrapped_env.device,
    )

    return runner, teacher_config_dir


def evaluate_policy(
    exp_name: str,
    show_viewer: bool = False,
    num_ckpt: int | None = None,
    device: str = "cuda",
    env_overrides: dict[str, Any] | None = None,
) -> None:
    if env_overrides is None:
        env_overrides = {}
    """Evaluate a trained BC policy."""
    print("=" * 80)
    print("EVALUATION MODE: Disabling observation noise")
    print("=" * 80)

    # Locate experiment directory
    log_pattern = f"logs/{exp_name}/*"
    log_dirs = glob.glob(log_pattern)
    if not log_dirs:
        raise FileNotFoundError(f"No experiment directories found matching pattern: {log_pattern}")
    log_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    exp_dir = log_dirs[0]
    print(f"Loading policy from experiment: {exp_dir}")

    # Resolve checkpoint
    if num_ckpt is not None:
        ckpt_path = Path(exp_dir) / "checkpoints" / f"checkpoint_{num_ckpt:04d}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")
    else:
        ckpt_path = load_latest_model(Path(exp_dir))
        num_ckpt = int(ckpt_path.stem.split("_")[-1])
    print(f"Loading checkpoint: {ckpt_path}")

    # Load configs
    print(f"Loading configs from experiment: {exp_dir}")
    env_args = yaml_to_config(Path(exp_dir) / "configs" / "env_args.yaml", MotionEnvArgs)
    algo_cfg = yaml_to_config(Path(exp_dir) / "configs" / "algo_cfg.yaml", BCArgs)

    env_args = apply_overrides_generic(env_args, env_overrides, prefixes=("cfgs.", "env."))

    # Make a copy for eval visualization if desired
    env_args = env_args.model_copy(
        update={
            "obs_noises": {},  # disable obs noise during eval
        }
    )
    env_args = cast(MotionEnvArgs, env_args).model_copy(
        update={"scene_args": SceneArgsRegistry["custom_scene_g1_mocap"]}
    )

    # Build eval environment
    env = create_gs_env(
        show_viewer=show_viewer,
        num_envs=1,
        device=device,
        args=env_args,
        eval_mode=True,
    )
    wrapped_env = GenesisEnvWrapper(env, device=env.device)

    # Setup GIF if headless
    gif_path = None
    if not show_viewer:
        gif_dir = Path("./gif") / exp_name
        gif_dir.mkdir(parents=True, exist_ok=True)
        gif_path = gif_dir / f"{num_ckpt}.gif"
        print(f"Will save GIF to: {gif_path}")
        env.start_rendering()  # type: ignore

    # Recreate BC algo and load weights
    bc = BC(env=wrapped_env, cfg=algo_cfg, device=wrapped_env.device)
    bc.load(ckpt_path, load_optimizer=False)
    inference_policy = bc.get_inference_policy()

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
            print("Running until GIF is saved (500 steps)")

        # Get an example observation and trace the policy
        obs, _ = wrapped_env.get_observations()
        traced_policy = torch.jit.trace(inference_policy, obs)

        # Save JIT policy and env_args for deployment
        print("Saving JIT-traced policy and env_args for deployment...")
        deploy_dir = Path("./deploy/logs") / exp_name
        deploy_dir.mkdir(parents=True, exist_ok=True)
        jit_policy_path = deploy_dir / f"checkpoint_{num_ckpt}.pt"
        torch.jit.save(traced_policy, str(jit_policy_path))
        print(f"JIT-traced policy saved to: {jit_policy_path}")
        env_args_path = deploy_dir / "env_args.yaml"
        config_to_yaml(env_args, env_args_path)
        print(f"Environment args saved to: {env_args_path}")

        motion_id = 0
        step_count = 0

        link_name_to_idx: dict[str, int] = {}
        for link_name in env.scene.objects.keys():
            link_name_to_idx[link_name] = env_args.tracking_link_names.index(link_name)

        while True:
            env.time_since_reset[0] = 0.0
            env.hard_reset_motion(torch.IntTensor([0]), motion_id)
            env.hard_sync_motion(torch.IntTensor([0]))
            obs = wrapped_env.obs
            while (
                env.motion_times[0]
                < env.motion_lib.get_motion_length(torch.IntTensor([motion_id])) - 0.02
            ):
                with torch.no_grad():
                    action = traced_policy(obs)  # type: ignore[misc]
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

                step_count += 1
                if step_count % 50 == 0:
                    print(f"Step {step_count}")

                if not show_viewer and gif_path is not None and step_count >= 500:
                    print("Stopping rendering and saving GIF...")
                    env.stop_rendering(save_gif=True, gif_path=str(gif_path))  # type: ignore
                    print(f"GIF saved to: {gif_path}")
                    return

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


def train_policy(
    teacher_exp_name: str = "g1_motion_teacher",
    exp_name: str | None = None,
    show_viewer: bool = False,
    num_envs: int = 8192,
    device: str = "cuda",
    use_wandb: bool = True,
    teacher_ckpt: int | None = None,
    env_args: Any = None,
    algo_cfg: Any = None,
    runner_args: Any = None,
) -> None:
    """Train BC policy to distill from teacher to student.

    Args:
        teacher_exp_name: Name of teacher experiment directory
        exp_name: Name of BC experiment directory. If None, uses "bc_motion"
        show_viewer: Whether to show viewer
        num_envs: Number of parallel environments
        device: Device to use ("cuda" or "cpu")
        use_wandb: Whether to use wandb logging
        teacher_ckpt: Checkpoint number to load from teacher. If None, loads latest.
        env_args: Student environment configuration
        algo_cfg: BC algorithm configuration
        runner_args: Runner configuration
    """
    env_args = cast(MotionEnvArgs, env_args).model_copy(
        update={"scene_args": SceneArgsRegistry["flat_scene_legged"]}
    )

    # Create student environment
    env = create_gs_env(
        show_viewer=show_viewer,
        num_envs=num_envs,
        device=device,
        args=env_args,
    )

    # Create BC runner
    runner, teacher_config_dir = create_bc_runner_from_registry(
        env=env,
        teacher_exp_name=teacher_exp_name,
        exp_name=exp_name,
        teacher_ckpt=teacher_ckpt,
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
        exp_name=exp_name or "bc_motion",
        mode="online" if use_wandb and (not show_viewer) else "disabled",
    )

    # Save configuration files to YAML
    print("Saving configuration files to YAML...")
    config_dir = logger_folder / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_to_yaml(env_args, config_dir / "env_args.yaml")
    config_to_yaml(algo_cfg, config_dir / "algo_cfg.yaml")
    config_to_yaml(runner_args, config_dir / "runner_args.yaml")

    # Copy teacher config for reference
    import shutil

    shutil.copy(teacher_config_dir / "env_args.yaml", config_dir / "teacher_env_args.yaml")
    print(f"Saved teacher config to: {config_dir / 'teacher_env_args.yaml'}")
    print("Note: BC uses teacher observations (from teacher config) for teacher policy.")
    print("Student uses student observations. Teacher provides action labels.")

    # Train using Runner
    print("Starting BC training...")
    print("Student environment: g1_motion")
    print(f"Teacher environment: g1_motion_teacher (from {teacher_exp_name})")
    print(f"Student observation dim: {env.actor_obs_dim}")

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


def main(
    teacher_exp_name: str = "g1_motion_teacher",
    exp_name: str | None = None,
    show_viewer: bool = False,
    num_envs: int = 8192,
    device: str = "cuda",
    use_wandb: bool = True,
    teacher_ckpt: int | None = None,
    eval: bool = False,
    num_ckpt: int | None = None,
    env_name: str = "g1_motion",
    **cfg_overrides: Any,
) -> None:
    """Entry point for BC training and evaluation.

    You can override configs via dot-notation, e.g.:
    --env.reward_args.AngVelZReward=5
    --algo.lr=1e-4
    --runner.total_iterations=2000
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
    algo_cfg = apply_overrides_generic(BC_MOTION_MLP, algo_overrides, prefixes=("cfgs.", "algo."))
    runner_args = apply_overrides_generic(
        RUNNER_BC_MOTION_MLP, runner_overrides, prefixes=("cfgs.", "runner.")
    )

    if eval:
        print("Evaluation mode: Loading trained BC policy")
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
        print("Training mode: Starting BC policy distillation")
        train_policy(
            teacher_exp_name=teacher_exp_name,
            exp_name=exp_name,
            show_viewer=show_viewer,
            num_envs=num_envs,
            device=device,
            use_wandb=use_wandb,
            teacher_ckpt=teacher_ckpt,
            env_args=env_args,
            algo_cfg=algo_cfg,
            runner_args=runner_args,
        )


if __name__ == "__main__":
    fire.Fire(main)
