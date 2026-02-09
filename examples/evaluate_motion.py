#!/usr/bin/env python3
"""Evaluate BC or PPO policies by traversing all motions in the motion library."""

import glob
import json
import os
import re
from ast import literal_eval
from pathlib import Path
from typing import Any, cast

import fire
import gs_env.sim.envs as gs_envs
import torch
from gs_agent.algos.bc import BC
from gs_agent.algos.config.schema import BCArgs, PPOArgs
from gs_agent.algos.ppo import PPO
from gs_agent.utils.policy_loader import load_latest_model
from gs_agent.wrappers.gs_env_wrapper import GenesisEnvWrapper
from gs_env.common.utils.math_utils import quat_to_angle_axis
from gs_env.sim.envs.config.schema import MotionEnvArgs
from gs_env.sim.scenes.config.registry import SceneArgsRegistry
from utils import apply_overrides_generic, yaml_to_config

EXP_NAMES: list[str] = []
MOTION_FILE: str | None = None


def create_gs_env(
    show_viewer: bool = False,
    num_envs: int = 1,
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


def evaluate_policies(
    exp_names: list[str],
    policy_type: str = "auto",  # "auto", "bc", or "ppo"
    num_ckpt: int | None = None,
    device: str = "cuda",
    env_overrides: dict[str, Any] | None = None,
    show_viewer: bool = False,
    motion_file: str | None = None,
    xlsx_path: str = "logs/eval_metrics.xlsx",
) -> None:
    """Evaluate multiple trained BC/PPO policies in parallel (one policy per env).

    - num_envs == number of policies
    - all envs traverse all motions in the shared motion library
    """
    if env_overrides is None:
        env_overrides = {}

    if len(exp_names) == 0:
        raise ValueError("exp_names is empty. Provide at least one experiment name.")

    print("=" * 80)
    print("EVALUATION MODE: Disabling observation noise and domain randomization")
    print("=" * 80)

    # Locate experiment directories (pick latest run per exp_name)
    exp_dirs: list[str] = []
    for name in exp_names:
        log_pattern = f"logs/{name}/*"
        log_dirs = glob.glob(log_pattern)
        if not log_dirs:
            raise FileNotFoundError(
                f"No experiment directories found matching pattern: {log_pattern}"
            )
        log_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        exp_dirs.append(log_dirs[0])

    print("Loading policies from experiments:")
    for name, d in zip(exp_names, exp_dirs, strict=False):
        print(f"  - {name}: {d}")

    # Resolve checkpoints (latest per exp unless a specific num_ckpt is requested)
    ckpt_paths: list[Path] = []
    for exp_dir in exp_dirs:
        if num_ckpt is not None:
            ckpt_path = Path(exp_dir) / "checkpoints" / f"checkpoint_{num_ckpt:04d}.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")
        else:
            ckpt_path = load_latest_model(Path(exp_dir))
        ckpt_paths.append(ckpt_path)

    print("Loading checkpoints:")
    for name, ckpt in zip(exp_names, ckpt_paths, strict=False):
        print(f"  - {name}: {ckpt}")

    # Load env config from FIRST experiment (shared env across all policies)
    base_exp_dir = exp_dirs[0]
    print(f"Loading shared env config from experiment: {base_exp_dir}")
    env_args = yaml_to_config(Path(base_exp_dir) / "configs" / "env_args.yaml", MotionEnvArgs)
    env_args = apply_overrides_generic(env_args, env_overrides, prefixes=("cfgs.", "env."))

    # Override motion_file if provided
    if motion_file is not None:
        env_args = env_args.model_copy(update={"motion_file": motion_file})
        print(f"Using motion file: {motion_file}")

    # Column key: base name of motion_file (or env_args.motion_file if motion_file wasn't provided)
    motion_file_key_src = motion_file if motion_file is not None else str(env_args.motion_file)
    motion_key = Path(motion_file_key_src).stem

    # Determine policy type per experiment (supports mixing, though usually they match)
    policy_types: list[str] = []
    for exp_dir in exp_dirs:
        if policy_type != "auto":
            policy_types.append(policy_type)
            continue
        algo_cfg_path = Path(exp_dir) / "configs" / "algo_cfg.yaml"
        if algo_cfg_path.exists():
            algo_cfg_dict = yaml_to_config(algo_cfg_path, None)
            if "teacher_path" in algo_cfg_dict:
                policy_types.append("bc")
            else:
                policy_types.append("ppo")
        else:
            policy_types.append("ppo")

    print("Policy types:")
    for name, ptype in zip(exp_names, policy_types, strict=False):
        print(f"  - {name}: {ptype}")

    # Disable observation noise and domain randomization for evaluation
    env_args = env_args.model_copy(update={"obs_noises": {}})
    env_args = cast(MotionEnvArgs, env_args).model_copy(
        update={"scene_args": SceneArgsRegistry["flat_scene_legged"]}
    )

    from gs_env.sim.robots.config.registry import DRArgsRegistry

    robot_args = env_args.robot_args.model_copy(
        update={"dr_args": DRArgsRegistry["no_randomization"]}
    )
    env_args = env_args.model_copy(update={"robot_args": robot_args})

    # Build eval environment
    env = create_gs_env(
        show_viewer=show_viewer,
        num_envs=len(exp_names),
        device=device,
        args=env_args,
        eval_mode=True,
    )
    wrapped_env = GenesisEnvWrapper(env, device=env.device)

    # Create algorithms + load weights (one per policy), then trace each on a single-env obs
    obs_batch, _ = wrapped_env.get_observations()
    sample_obs = obs_batch[:1]  # shape [1, obs_dim]

    # torch.jit.trace() can return different traced module types depending on torch version;
    # we only need something callable that maps obs -> action.
    traced_policies: list[Any] = []

    class DeterministicWrapper(torch.nn.Module):
        def __init__(self, policy: Any) -> None:
            super().__init__()
            self.policy = policy

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            action, _ = self.policy(obs, deterministic=True)
            return action

    for exp_dir, ckpt_path, ptype in zip(exp_dirs, ckpt_paths, policy_types, strict=False):
        if ptype == "bc":
            algo_cfg = yaml_to_config(Path(exp_dir) / "configs" / "algo_cfg.yaml", BCArgs)
            algorithm = BC(env=wrapped_env, cfg=algo_cfg, device=wrapped_env.device)
            algorithm.load(ckpt_path, load_optimizer=False)
            inference_policy = algorithm.get_inference_policy()
            traced = torch.jit.trace(inference_policy, sample_obs)
        else:
            algo_cfg = yaml_to_config(Path(exp_dir) / "configs" / "algo_cfg.yaml", PPOArgs)
            algorithm = PPO(env=wrapped_env, cfg=algo_cfg, device=wrapped_env.device)
            algorithm.load(ckpt_path, load_optimizer=False)
            inference_policy = algorithm.get_inference_policy()
            traced = torch.jit.trace(DeterministicWrapper(inference_policy), sample_obs)

        traced_policies.append(traced)

    print("Starting evaluation across all motions...")
    print(f"Total motions in library: {env.motion_lib.num_motions}")

    # Optional progress bar (no hard dependency)
    def _get_tqdm() -> Any | None:
        try:
            import importlib

            return importlib.import_module("tqdm").tqdm
        except Exception:
            return None

    tqdm_fn = _get_tqdm()

    # Statistics collection
    motion_results: list[dict[str, Any]] = []
    total_steps_per_policy = [0 for _ in exp_names]
    total_terminations_per_policy = [0 for _ in exp_names]

    envs_idx = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    # Global (across all motions) metric accumulators for XLSX export
    global_base_pos_err_sum = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    global_tracking_pos_err_sum = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    global_tracking_quat_err_sum = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    global_step_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    # Iterate through all motions
    for motion_id in range(env.motion_lib.num_motions):
        motion_name = env.motion_lib.motion_names[motion_id]
        motion_length = env.motion_lib.get_motion_length(
            torch.tensor([motion_id], device=env.device, dtype=torch.long)
        ).item()

        print(f"\nEvaluating motion {motion_id}/{env.motion_lib.num_motions - 1}: {motion_name}")
        print(f"  Motion length: {motion_length:.2f}s")

        # Reset environment for this motion
        env.time_since_reset[:] = 0.0
        env.hard_reset_motion(envs_idx, motion_id)
        obs, _ = wrapped_env.get_observations()

        # Track metrics for this motion
        motion_steps = 0
        motion_terminated = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        # Per-policy accumulators (mean over time; and over tracking links for link metrics)
        base_pos_err_sum = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        tracking_pos_err_sum = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        tracking_quat_err_sum = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        step_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

        # Per-motion rolling progress bar (steps ~= motion_length / dt)
        dt = float(getattr(env, "dt", 0.02))
        total_est_steps = max(int((motion_length - 0.02) / max(dt, 1e-6)), 1)
        pbar = (
            tqdm_fn(
                total=total_est_steps,
                desc=f"motion {motion_id}/{env.motion_lib.num_motions - 1}",
                leave=False,
                dynamic_ncols=True,
            )
            if tqdm_fn is not None
            else None
        )

        # Run until motion completes or terminates
        while env.motion_times[0] < motion_length - 0.02:
            prev_terminated = motion_terminated.clone()
            active = ~prev_terminated
            with torch.no_grad():
                # One policy per env: action[i] = policy_i(obs[i])
                actions: list[torch.Tensor] = []
                for i, pol in enumerate(traced_policies):
                    if prev_terminated[i].item():
                        actions.append(
                            torch.zeros(1, env.action_dim, device=env.device, dtype=torch.float32)
                        )
                    else:
                        actions.append(pol(obs[i : i + 1]))  # shape [1, act_dim]
                action = torch.cat(actions, dim=0)  # shape [num_envs, act_dim]

            obs = wrapped_env.step(action)[0]

            # Collect errors for envs that were active at the start of this step.
            # - base pos: L2 norm of diff_base_pos_local_yaw (mean over time)
            # - tracking pos: mean (over links) L2 norm of diff_tracking_link_pos_local_yaw (then mean over time)
            # - tracking quat: mean (over links) angle error in radians from diff_tracking_link_quat_local_yaw
            if active.any():
                base_pos_err = torch.linalg.norm(env.diff_base_pos_local_yaw, dim=-1)  # (N,)
                tracking_pos_err = torch.linalg.norm(
                    env.diff_tracking_link_pos_local_yaw, dim=-1
                ).mean(dim=-1)  # (N,)
                tracking_quat_err = torch.linalg.norm(
                    quat_to_angle_axis(env.diff_tracking_link_quat_local_yaw), dim=-1
                ).mean(dim=-1)  # (N,)

                base_pos_err_sum[active] += base_pos_err[active]
                tracking_pos_err_sum[active] += tracking_pos_err[active]
                tracking_quat_err_sum[active] += tracking_quat_err[active]
                step_count[active] += 1

            motion_steps += 1
            if pbar is not None:
                pbar.update(1)
                if motion_steps % 10 == 0:
                    pbar.set_postfix(
                        terminated=int(motion_terminated.sum().item()),
                        refresh=False,
                    )

        if pbar is not None:
            pbar.close()

        # Store results per policy for this motion
        for i, name in enumerate(exp_names):
            denom = float(max(int(step_count[i].item()), 1))
            mean_base_pos_err = float((base_pos_err_sum[i] / denom).item())
            mean_tracking_pos_err = float((tracking_pos_err_sum[i] / denom).item())
            mean_tracking_quat_err = float((tracking_quat_err_sum[i] / denom).item())
            motion_results.append(
                {
                    "policy": name,
                    "policy_idx": i,
                    "motion_id": motion_id,
                    "motion_name": motion_name,
                    "motion_length": motion_length,
                    "steps": motion_steps,
                    "terminated": bool(motion_terminated[i].item()),
                    "mean_diff_base_pos_local_yaw_dist": mean_base_pos_err,
                    "mean_diff_tracking_link_pos_local_yaw_dist": mean_tracking_pos_err,
                    "mean_diff_tracking_link_quat_local_yaw_angle": mean_tracking_quat_err,
                }
            )
            total_steps_per_policy[i] += motion_steps
            if motion_terminated[i].item():
                total_terminations_per_policy[i] += 1

        # Accumulate globals (weighted by active steps)
        global_base_pos_err_sum += base_pos_err_sum
        global_tracking_pos_err_sum += tracking_pos_err_sum
        global_tracking_quat_err_sum += tracking_quat_err_sum
        global_step_count += step_count

        term_cnt = int(motion_terminated.sum().item())
        print(f"  Steps: {motion_steps}, Terminated envs: {term_cnt}/{env.num_envs}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Policies evaluated: {len(exp_names)}")
    print(f"Motions evaluated per policy: {env.motion_lib.num_motions}")
    for i, name in enumerate(exp_names):
        terms = total_terminations_per_policy[i]
        total = env.motion_lib.num_motions
        print("-" * 80)
        print(f"Policy: {name}")
        print(f"  Total steps: {total_steps_per_policy[i]}")
        print(f"  Terminations: {terms} ({100.0 * terms / total:.1f}%)")
        print(f"  Success rate: {100.0 * (total - terms) / total:.1f}%")

    # Per-motion statistics
    print("\nPer-motion statistics:")
    print(
        f"{'Policy':<18} {'Motion ID':<10} {'Name':<30} {'Steps':<8} {'Status':<10} "
        f"{'base_pos(m)':<12} {'trk_pos(m)':<12} {'trk_ang(rad)':<12}"
    )
    print("-" * 90)
    for result in motion_results:
        status = "TERMINATED" if result["terminated"] else "COMPLETED"
        print(
            f"{result['policy'][:16]:<18} "
            f"{result['motion_id']:<10} "
            f"{result['motion_name'][:28]:<30} "
            f"{result['steps']:<8} "
            f"{status:<10} "
            f"{result['mean_diff_base_pos_local_yaw_dist']:<12.4f} "
            f"{result['mean_diff_tracking_link_pos_local_yaw_dist']:<12.4f} "
            f"{result['mean_diff_tracking_link_quat_local_yaw_angle']:<12.4f}"
        )

    print("=" * 80)

    # ----------------
    # Write XLSX export
    # ----------------
    # We write 3 sheets, each with:
    # - row names = exp_names (Column A)
    # - column names = motion_key (Row 1)
    # Each cell is the global mean over time (and over links where applicable) across all motions.
    def _xlsx_upsert_metric(
        workbook_path: str,
        sheet_name: str,
        row_names: list[str],
        col_name: str,
        values: list[float],
    ) -> None:
        try:
            import importlib

            openpyxl = importlib.import_module("openpyxl")
            Workbook = openpyxl.Workbook
            load_workbook = openpyxl.load_workbook
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Excel export requires 'openpyxl'. Install it (e.g. `pip install openpyxl`)."
            ) from e

        xlsx_p = Path(workbook_path)
        xlsx_p.parent.mkdir(parents=True, exist_ok=True)

        if xlsx_p.exists():
            wb = load_workbook(xlsx_p)
        else:
            wb = Workbook()
            # Remove default sheet if present and unused
            if "Sheet" in wb.sheetnames and len(wb.sheetnames) == 1:
                ws0 = wb["Sheet"]
                if ws0.max_row == 1 and ws0.max_column == 1 and ws0["A1"].value is None:
                    wb.remove(ws0)

        if sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
        else:
            ws = wb.create_sheet(sheet_name)

        # Ensure header row/col exist
        ws.cell(row=1, column=1, value="exp_name")

        # Find or create column for col_name
        col_idx = None
        for c in range(2, ws.max_column + 1):
            if ws.cell(row=1, column=c).value == col_name:
                col_idx = c
                break
        if col_idx is None:
            col_idx = ws.max_column + 1 if ws.max_column >= 2 else 2
            ws.cell(row=1, column=col_idx, value=col_name)

        # Build a map of existing exp_name -> row index
        row_map: dict[str, int] = {}
        for r in range(2, ws.max_row + 1):
            v = ws.cell(row=r, column=1).value
            if isinstance(v, str) and v != "":
                row_map[v] = r

        # Upsert rows and write values
        for name, val in zip(row_names, values, strict=False):
            if name in row_map:
                r = row_map[name]
            else:
                r = ws.max_row + 1 if ws.max_row >= 2 else 2
                ws.cell(row=r, column=1, value=name)
                row_map[name] = r

            cell = ws.cell(row=r, column=col_idx)
            existing = cell.value
            new_v = float(val)

            # If there's an existing value, turn it into a list and append; if it's already a list, append.
            # Stored format: JSON list string, e.g. "[0.12, 0.34]".
            values_list: list[float]
            if existing is None or existing == "":
                values_list = [new_v]
            elif isinstance(existing, int | float):
                values_list = [float(existing), new_v]
            elif isinstance(existing, str):
                s = existing.strip()
                parsed: Any = None
                # Try JSON list first
                try:
                    parsed = json.loads(s)
                except Exception:
                    parsed = None
                # Try python literal (e.g. "[1.0, 2.0]") if not JSON
                if parsed is None:
                    try:
                        parsed = literal_eval(s)
                    except Exception:
                        parsed = None
                if isinstance(parsed, list):
                    # best-effort float conversion
                    tmp: list[float] = []
                    for x in parsed:
                        try:
                            tmp.append(float(x))
                        except Exception:
                            pass
                    values_list = tmp + [new_v]
                else:
                    # last resort: extract the first float from the string, if any
                    m = re.search(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", s)
                    if m is not None:
                        values_list = [float(m.group(0)), new_v]
                    else:
                        values_list = [new_v]
            else:
                # Unknown cell type; overwrite into a list
                values_list = [new_v]

            cell.value = json.dumps(values_list)

        wb.save(xlsx_p)

    # Compute global means per policy (avoid div-by-zero)
    global_denoms = torch.clamp(global_step_count.to(torch.float32), min=1.0)
    global_base_mean = (global_base_pos_err_sum / global_denoms).detach().cpu().tolist()
    global_trk_pos_mean = (global_tracking_pos_err_sum / global_denoms).detach().cpu().tolist()
    global_trk_ang_mean = (global_tracking_quat_err_sum / global_denoms).detach().cpu().tolist()

    _xlsx_upsert_metric(
        xlsx_path, "base_pos_m", exp_names, motion_key, cast(list[float], global_base_mean)
    )
    _xlsx_upsert_metric(
        xlsx_path, "trk_pos_m", exp_names, motion_key, cast(list[float], global_trk_pos_mean)
    )
    _xlsx_upsert_metric(
        xlsx_path, "trk_ang_rad", exp_names, motion_key, cast(list[float], global_trk_ang_mean)
    )
    print(f"Saved XLSX metrics to: {xlsx_path} (column: {motion_key})")


def main(
    exp_name: str | None = None,
    exp_names: list[str] | None = None,
    policy_type: str = "auto",
    num_ckpt: int | None = None,
    device: str = "cpu",
    show_viewer: bool = False,
    motion_file: str | None = None,
    **cfg_overrides: Any,
) -> None:
    """Entry point for motion evaluation.

    Args:
        exp_name: Name of the experiment directory
        policy_type: Type of policy ("auto", "bc", or "ppo")
        num_ckpt: Checkpoint number to load. If None, loads latest.
        device: Device to use ("cuda" or "cpu")
        show_viewer: Whether to show viewer
        motion_file: Optional motion file path to override the one from experiment config
        **cfg_overrides: Optional config overrides (e.g., --env.reward_args.AngVelZReward=5)
    """
    # Bucket overrides into env
    env_overrides: dict[str, Any] = {}

    for k, v in cfg_overrides.items():
        if k.startswith("cfgs.env.") or k.startswith("env.") or k.startswith("reward_args."):
            env_overrides[k] = v

    # Resolve exp_names from CLI args or file-level defaults
    if exp_names is None:
        if exp_name is not None:
            exp_names = [exp_name]
        else:
            exp_names = EXP_NAMES

    if motion_file is None:
        motion_file = MOTION_FILE

    evaluate_policies(
        exp_names=exp_names,
        policy_type=policy_type,
        num_ckpt=num_ckpt,
        device=device,
        env_overrides=env_overrides,
        show_viewer=show_viewer,
        motion_file=motion_file,
    )


if __name__ == "__main__":
    fire.Fire(main)
