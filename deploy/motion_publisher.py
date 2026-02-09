import json

# Add examples to path to import utils
import sys
import time
from pathlib import Path

import fire
import redis
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from gs_env.common.utils.motion_utils import MotionLib
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from gs_env.sim.envs.config.schema import MotionEnvArgs
from utils import yaml_to_config  # type: ignore


def _to_list(t: torch.Tensor) -> list[float]:
    return t.detach().cpu().flatten().tolist()


def load_motion_file_from_exp(exp_name: str) -> str:
    deploy_dir = Path(__file__).parent / "logs" / exp_name
    env_args_path = deploy_dir / "env_args.yaml"
    if not env_args_path.exists():
        raise FileNotFoundError(f"env_args.yaml not found: {env_args_path}")
    env_args = yaml_to_config(env_args_path, MotionEnvArgs)
    return env_args.motion_file


def load_env_args_from_exp(exp_name: str) -> MotionEnvArgs:
    deploy_dir = Path(__file__).parent / "logs" / exp_name
    env_args_path = deploy_dir / "env_args.yaml"
    if not env_args_path.exists():
        raise FileNotFoundError(f"env_args.yaml not found: {env_args_path}")
    return yaml_to_config(env_args_path, MotionEnvArgs)


def publish_motion(
    motion_file: str,
    redis_url: str = "redis://localhost:6379/0",
    key: str = "ref:",
    motion_id: int = 0,
    motion_freq: float = 50.0,
    publish_rate: float = 50.0,
    device: str = "cpu",
    tracking_link_names: list[str] | None = None,
) -> None:
    """Publish reference motion frames to Redis at a fixed rate.

    The publisher writes each field as a separate Redis key:
      - {key}:motion:base_pos [3]
      - {key}:motion:base_quat [4] (w, x, y, z)
      - {key}:motion:base_lin_vel [3]
      - {key}:motion:base_ang_vel [3]
      - {key}:motion:base_ang_vel_local [3]
      - {key}:motion:dof_pos [D]
      - {key}:motion:dof_vel [D]
      - {key}:motion:link_pos_local [N*3] (filtered to tracking links if specified)
      - {key}:motion:link_quat_local [N*4] (filtered to tracking links if specified)
      - {key}:motion:foot_contact [F]
      - {key}:timestamp:base_pos [1]
      - {key}:timestamp:base_quat [1]
      - {key}:timestamp:base_lin_vel [1]
      - {key}:timestamp:base_ang_vel [1]
      - {key}:timestamp:base_ang_vel_local [1]
      - {key}:timestamp:dof_pos [1]
      - {key}:timestamp:dof_vel [1]
      - {key}:timestamp:link_pos_local [1]
      - {key}:timestamp:link_quat_local [1]
      - {key}:timestamp:foot_contact [1]
    """
    r = redis.from_url(redis_url)

    device_t = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    motion_lib = MotionLib(motion_file=motion_file, device=device_t, target_fps=motion_freq)

    # Pre-compute tracking link indices if names are provided
    link_indices: list[int] | None = None
    if tracking_link_names is not None and len(tracking_link_names) > 0:
        name_to_idx = {name: i for i, name in enumerate(motion_lib.tracking_link_names)}
        link_indices = []
        missing: list[str] = []
        for name in tracking_link_names:
            if name in name_to_idx:
                link_indices.append(name_to_idx[name])
            else:
                missing.append(name)
        if len(missing) > 0:
            print(f"[motion_publisher] Warning: tracking links not found in motion_lib: {missing}")
        if len(link_indices) == 0:
            print("[motion_publisher] No valid tracking links found; publishing all links instead.")
            link_indices = None

    motion_id_t = torch.tensor([motion_id], dtype=torch.long, device=device_t)

    motion_dt = 1.0 / motion_freq
    publish_dt = 1.0 / publish_rate
    t_val = 0.0

    print("=" * 80)
    print("Motion Publisher started")
    print(f"Redis: {redis_url}")
    print(f"Key: {key}")
    print(f"Motion file: {motion_file}")
    print(f"Motion id: {motion_id}")
    print(f"Publish rate: {1.0 / publish_dt:.2f} Hz")
    if link_indices is not None:
        selected_names = [motion_lib.tracking_link_names[i] for i in link_indices]
        print(f"Publishing {len(selected_names)} tracking links only: {selected_names}")
    print("=" * 80)

    timestamp = 0
    next_publish_time = time.time() + publish_dt

    try:
        while True:
            if t_val > motion_lib.get_motion_length(motion_id_t):
                t_val = 0.0
                motion_id_t += 1
                if motion_id_t >= motion_lib.num_motions:
                    motion_id_t = torch.tensor([0], dtype=torch.long, device=device_t)
            motion_time_t = torch.tensor([t_val], dtype=torch.float32, device=device_t)
            (
                base_pos,
                base_quat,
                base_lin_vel,
                base_ang_vel,
                base_ang_vel_local,
                dof_pos,
                dof_vel,
                link_pos_global,
                link_pos_local,
                link_quat_global,
                link_quat_local,
                link_lin_vel,
                link_lin_vel_local,
                link_ang_vel,
                link_ang_vel_local,
                foot_contact,
                foot_contact_weighted,
            ) = motion_lib.get_ref_motion_frame(motion_ids=motion_id_t, motion_times=motion_time_t)

            _ = foot_contact_weighted
            _ = link_pos_global
            _ = link_quat_global

            # Filter link states to tracking links if requested
            if link_indices is not None:
                link_pos_local = link_pos_local[0, link_indices]
                link_quat_local = link_quat_local[0, link_indices]

            # Publish each field as a separate Redis key
            r.set(f"{key}:motion:base_pos", json.dumps(_to_list(base_pos)))
            r.set(f"{key}:motion:base_quat", json.dumps(_to_list(base_quat)))  # (w, x, y, z)
            r.set(f"{key}:motion:base_lin_vel", json.dumps(_to_list(base_lin_vel)))
            r.set(f"{key}:motion:base_ang_vel", json.dumps(_to_list(base_ang_vel)))
            r.set(f"{key}:motion:base_ang_vel_local", json.dumps(_to_list(base_ang_vel_local)))
            r.set(f"{key}:motion:dof_pos", json.dumps(_to_list(dof_pos)))
            r.set(f"{key}:motion:dof_vel", json.dumps(_to_list(dof_vel)))
            r.set(f"{key}:motion:link_pos_local", json.dumps(_to_list(link_pos_local)))
            r.set(f"{key}:motion:link_quat_local", json.dumps(_to_list(link_quat_local)))
            r.set(f"{key}:motion:link_lin_vel", json.dumps(_to_list(link_lin_vel)))
            r.set(f"{key}:motion:link_ang_vel", json.dumps(_to_list(link_ang_vel)))
            r.set(f"{key}:motion:link_lin_vel_local", json.dumps(_to_list(link_lin_vel_local)))
            r.set(f"{key}:motion:link_ang_vel_local", json.dumps(_to_list(link_ang_vel_local)))
            r.set(f"{key}:motion:foot_contact", json.dumps(_to_list(foot_contact)))
            r.set(f"{key}:timestamp:base_pos", timestamp)
            r.set(f"{key}:timestamp:base_quat", timestamp)
            r.set(f"{key}:timestamp:base_lin_vel", timestamp)
            r.set(f"{key}:timestamp:base_ang_vel", timestamp)
            r.set(f"{key}:timestamp:base_ang_vel_local", timestamp)
            r.set(f"{key}:timestamp:dof_pos", timestamp)
            r.set(f"{key}:timestamp:dof_vel", timestamp)
            r.set(f"{key}:timestamp:link_pos_local", timestamp)
            r.set(f"{key}:timestamp:link_quat_local", timestamp)
            r.set(f"{key}:timestamp:link_lin_vel", timestamp)
            r.set(f"{key}:timestamp:link_ang_vel", timestamp)
            r.set(f"{key}:timestamp:link_lin_vel_local", timestamp)
            r.set(f"{key}:timestamp:link_ang_vel_local", timestamp)
            r.set(f"{key}:timestamp:foot_contact", timestamp)
            timestamp += 1

            # Advance time, loop by motion length
            t_val += motion_dt
            t_now = time.time()
            if t_now < next_publish_time:
                time.sleep(max(0, next_publish_time - t_now))
                next_publish_time = next_publish_time + publish_dt
            else:
                next_publish_time = time.time() + publish_dt

    except KeyboardInterrupt:
        print("\nStopping motion publisher...")


def main(
    exp_name: str | None = None,
    motion_file: str | None = None,
    redis_url: str = "redis://localhost:6379/0",
    key: str = "motion:ref:latest",
    motion_id: int = 0,
    motion_freq: float = 50.0,
    publish_rate: float = 50.0,
    device: str = "cpu",
) -> None:
    # Resolve motion_file and tracking_link_names
    tracking_link_names: list[str] = []
    if motion_file is None:
        if exp_name is None:
            raise ValueError("Either exp_name or motion_file must be provided")
        env_args = load_env_args_from_exp(exp_name)
        motion_file = env_args.motion_file
        if motion_file is None:
            raise ValueError("motion_file missing in env_args for provided exp_name")
        tracking_link_names = getattr(env_args, "tracking_link_names", []) or []
    else:
        # No exp given: fall back to registry default for tracking links
        # Prefer 'g1_motion' config if available; otherwise try 'g1_motion_teacher'
        default_names: list[str] = []
        if "g1_motion" in EnvArgsRegistry:
            default_names = getattr(EnvArgsRegistry["g1_motion"], "tracking_link_names", []) or []
        elif "g1_motion_teacher" in EnvArgsRegistry:
            default_names = (
                getattr(EnvArgsRegistry["g1_motion_teacher"], "tracking_link_names", []) or []
            )
        tracking_link_names = default_names

    publish_motion(
        motion_file=motion_file,
        redis_url=redis_url,
        key=key,
        motion_id=motion_id,
        motion_freq=motion_freq,
        publish_rate=publish_rate,
        device=device,
        tracking_link_names=tracking_link_names,
    )


if __name__ == "__main__":
    fire.Fire(main)
