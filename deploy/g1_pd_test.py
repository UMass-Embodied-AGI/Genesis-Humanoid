import os
import sys
import time
from pathlib import Path
from typing import Any

import fire
import matplotlib
import yaml

matplotlib.use("Agg")  # Use non-interactive backend to prevent windows from showing
import matplotlib.pyplot as plt
import numpy as np
import torch
from gs_env.sim.envs.config.registry import EnvArgsRegistry

# Add examples to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from utils import (
    compute_SD,
    compute_SRD,
    cross_correlation,
    plot_metric_on_axis,
)  # type: ignore

WAIT_FOR_NEXT_CTRL = True


def run_single_dof_wave_diagnosis(
    env: Any,
    dof_idx: int = 0,
    num_dofs: int = 29,
    wave_type: str = "SIN",
    period: int = 1,
    amplitude: float = 1.0,
    offset: float = 0.0,
) -> dict[str, Any]:
    """Run single DoF diagnosis."""
    print(f"Running DoF diagnosis for {wave_type}")
    NUM_TOTAL_PERIODS = 3

    def linear_phase(x: int) -> float:
        return x / period * 2 * np.pi

    def quadratic_phase(x: int) -> float:
        return (x / period) ** 2 * 2 * np.pi / NUM_TOTAL_PERIODS * 2

    def linear_amp(x: int) -> float:
        return (10 * (x - 1) / period) // NUM_TOTAL_PERIODS * amplitude / 10

    if wave_type == "SIN":

        def wave_func(x: int) -> float:
            return np.sin(linear_phase(x)) * amplitude
    elif wave_type == "FM-SIN":

        def wave_func(x: int) -> float:
            return np.sin(quadratic_phase(x)) * amplitude
    elif wave_type == "SQ":

        def wave_func(x: int) -> float:
            return np.sign(np.sin(linear_phase(x))) * linear_amp(x)
    else:
        raise ValueError(f"Invalid wave type: {wave_type}")

    env.reset()
    action = torch.zeros((1, num_dofs), device=env.device)
    action[:, dof_idx] = offset / env.action_scale[dof_idx]

    next_ctrl = time.time() + env.dt

    current_dof_pos = env.dof_pos[0] - env.default_dof_pos
    TOTAL_RESET_STEPS = 50
    for i in range(TOTAL_RESET_STEPS):
        if WAIT_FOR_NEXT_CTRL:
            if time.time() >= next_ctrl:
                next_ctrl = time.time() + env.dt
            else:
                time.sleep(max(0, next_ctrl - time.time()))
            next_ctrl += env.dt
        env.apply_action(
            current_dof_pos / env.action_scale * (1 - i / TOTAL_RESET_STEPS)
            + action * (i / TOTAL_RESET_STEPS)
        )

    env.robot.start_logging()

    for i in range(period * NUM_TOTAL_PERIODS):
        if WAIT_FOR_NEXT_CTRL:
            if time.time() >= next_ctrl:
                next_ctrl = time.time() + env.dt
            else:
                time.sleep(max(0, next_ctrl - time.time()))
            next_ctrl += env.dt

        target_dof_pos = wave_func(i) + offset
        action[:, dof_idx] = target_dof_pos / env.action_scale[dof_idx]
        env.apply_action(action)

        if i % period == 0:
            print(f"Step {i} of {period * NUM_TOTAL_PERIODS}")

    log = env.robot.stop_logging()
    for key in log.keys():
        if key != "time_stamp":
            log[key] = log[key][..., dof_idx]
        if isinstance(log[key], torch.Tensor):
            log[key] = log[key].squeeze().cpu().numpy()

    return log


def run_single_dof_diagnosis(
    env: Any,
    dof_idx: int = 0,
    dof_name: str = "DoF",
    num_dofs: int = 29,
    period: int = 100,
    amplitude: float = 1.0,
    offset: float = 0.0,
    log_dir: Path = Path(__file__).parent / "logs" / "pd_test",
    sim: bool = True,
    kp: float | None = None,
    kd: float | None = None,
    feed_forward_ratio: float | None = None,
    low_pass_alpha: float | None = None,
) -> dict[str, float]:
    if sim:
        if kp is not None:
            batched_kp = env.robot.batched_dof_kp.clone()
            batched_kp[:, dof_idx] = kp
            env.robot.set_batched_dof_kp(batched_kp)
        if kd is not None:
            batched_kd = env.robot.batched_dof_kd.clone()
            batched_kd[:, dof_idx] = kd
            env.robot.set_batched_dof_kd(batched_kd)
    else:
        if kp is not None:
            env.robot.kp[dof_idx] = kp
        if kd is not None:
            env.robot.kd[dof_idx] = kd
    if kp is None:
        kp = env.robot.dof_kp[dof_idx].item()
    if kd is None:
        kd = env.robot.dof_kd[dof_idx].item()
    if feed_forward_ratio is not None:
        env.robot.feed_forward_ratio = feed_forward_ratio
    else:
        feed_forward_ratio = env.robot.feed_forward_ratio
    if low_pass_alpha is not None:
        env.low_pass_alpha = low_pass_alpha
    else:
        low_pass_alpha = env.low_pass_alpha
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    wave_types = [
        "SIN",
    ]
    for i, wave_type in enumerate(wave_types):
        log = run_single_dof_wave_diagnosis(
            env,
            dof_idx=dof_idx,
            num_dofs=num_dofs,
            wave_type=wave_type,
            period=period,
            amplitude=amplitude,
            offset=offset,
        )
        time_stamp_raw = log["time_stamp"]
        time_stamp_raw -= time_stamp_raw[0]
        interpolate_num = int(time_stamp_raw[-1] / env.dt * env.decimation) + 1
        time_stamp = np.arange(interpolate_num) * env.dt / env.decimation
        target_dof_pos_raw = log["target_dof_pos"]
        dof_pos_raw = log["dof_pos"]
        target_dof_pos = np.interp(time_stamp, time_stamp_raw, target_dof_pos_raw)
        dof_pos = np.interp(time_stamp, time_stamp_raw, dof_pos_raw)
        dof_vel_raw = log["dof_vel"]

        dof_pos_lag = (
            cross_correlation(
                target_dof_pos[:: env.decimation],
                dof_pos[:: env.decimation],
            )
            * env.dt
        )
        print("=" * 40)
        print(f"Lag: {dof_pos_lag:.4f}")
        print("=" * 40)
        target_dof_pos_SRD = compute_SRD(target_dof_pos)
        target_dof_pos_SD = compute_SD(target_dof_pos)
        target_dof_pos_SRD_decimated = compute_SRD(target_dof_pos[:: env.decimation])
        target_dof_pos_SD_decimated = compute_SD(target_dof_pos[:: env.decimation])
        print(f"target_dof_pos SRD: {target_dof_pos_SRD:.4f}, SD: {target_dof_pos_SD:.4f}")
        print(
            f"target_dof_pos / {env.decimation} SRD: {target_dof_pos_SRD_decimated:.4f}, SD: {target_dof_pos_SD_decimated:.4f}"
        )
        dof_pos_SRD = compute_SRD(dof_pos)
        dof_pos_SD = compute_SD(dof_pos)
        dof_pos_SRD_decimated = compute_SRD(dof_pos[:: env.decimation])
        dof_pos_SD_decimated = compute_SD(dof_pos[:: env.decimation])
        print(f"dof_pos SRD: {dof_pos_SRD:.4f}, SD: {dof_pos_SD:.4f}")
        print(
            f"dof_pos / {env.decimation} SRD: {dof_pos_SRD_decimated:.4f}, SD: {dof_pos_SD_decimated:.4f}"
        )
        if sim:
            data_lists = [target_dof_pos.tolist(), dof_pos.tolist()]
            labels = ["Target", "Dof Pos"]
        else:
            dof_pos_raw_raw = log["dof_pos_raw"]
            dof_pos_raw = np.interp(time_stamp, time_stamp_raw, dof_pos_raw_raw)
            print("=" * 40)
            dof_pos_lag = (
                cross_correlation(
                    target_dof_pos[:: env.decimation],
                    dof_pos_raw[:: env.decimation],
                )
                * env.dt
            )
            print(f"Raw Lag: {dof_pos_lag:.4f}")
            print("=" * 40)
            dof_pos_raw_SRD = compute_SRD(dof_pos_raw)
            dof_pos_raw_SD = compute_SD(dof_pos_raw)
            dof_pos_raw_SRD_decimated = compute_SRD(dof_pos_raw[:: env.decimation])
            dof_pos_raw_SD_decimated = compute_SD(dof_pos_raw[:: env.decimation])
            print(f"dof_pos_raw SRD: {dof_pos_raw_SRD:.4f}, SD: {dof_pos_raw_SD:.4f}")
            print(
                f"dof_pos_raw / {env.decimation} SRD: {dof_pos_raw_SRD_decimated:.4f}, SD: {dof_pos_raw_SD_decimated:.4f}"
            )
            data_lists = [target_dof_pos.tolist(), dof_pos.tolist(), dof_pos_raw.tolist()]
            labels = ["Target", "Dof Pos", "Dof Pos Raw"]

        plot_metric_on_axis(
            axes[i * 2],
            time_stamp,
            data_lists,
            labels,
            "Dof Pos",
            wave_type,
            yscale="linear",
        )

        print("=" * 40)
        dof_vel_SRD = compute_SRD(dof_vel_raw)
        dof_vel_SD = compute_SD(dof_vel_raw)
        dof_vel_SRD_decimated = compute_SRD(dof_vel_raw[:: env.decimation])
        dof_vel_SD_decimated = compute_SD(dof_vel_raw[:: env.decimation])
        print(f"dof_vel SRD: {dof_vel_SRD:.4f}, SD: {dof_vel_SD:.4f}")
        print(
            f"dof_vel / {env.decimation} SRD: {dof_vel_SRD_decimated:.4f}, SD: {dof_vel_SD_decimated:.4f}"
        )

        if sim:
            data_lists = [dof_vel_raw.tolist()]
            labels = ["Dof Vel"]
        else:
            dof_vel_raw_raw = log["dof_vel_raw"]
            dof_vel_raw_SRD = compute_SRD(dof_vel_raw_raw)
            dof_vel_raw_SD = compute_SD(dof_vel_raw_raw)
            dof_vel_raw_SRD_decimated = compute_SRD(dof_vel_raw_raw[:: env.decimation])
            dof_vel_raw_SD_decimated = compute_SD(dof_vel_raw_raw[:: env.decimation])
            low_pass_lag = (
                cross_correlation(
                    dof_vel_raw_raw,
                    dof_vel_raw,
                )
                * env.dt
                / env.decimation
            )
            print("=" * 40)
            print(f"Low Pass Lag: {low_pass_lag:.4f}")
            print("=" * 40)
            print(f"dof_vel_raw SRD: {dof_vel_raw_SRD:.4f}, SD: {dof_vel_raw_SD:.4f}")
            print(
                f"dof_vel_raw / {env.decimation} SRD: {dof_vel_raw_SRD_decimated:.4f}, SD: {dof_vel_raw_SD_decimated:.4f}"
            )
            data_lists = [dof_vel_raw.tolist(), dof_vel_raw_raw.tolist()]
            labels = ["Dof Vel", "Dof Vel Raw"]

        plot_metric_on_axis(
            axes[i * 2 + 1],
            time_stamp_raw,
            data_lists,
            labels,
            "Dof Vel",
            wave_type,
            yscale="linear",
        )

        print("=" * 40)

    # Save plot
    log_dir.mkdir(parents=True, exist_ok=True)

    sim_suffix = "sim" if sim else "real"
    plot_path = log_dir / f"{dof_name}_{sim_suffix}.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"Plot saved to: {plot_path}")

    result = {
        "dof_name": dof_name,
        "sim": sim,
        "period": period * env.dt,
        "kp": kp,
        "kd": kd,
        "feed_forward_ratio": feed_forward_ratio,
        "low_pass_alpha": low_pass_alpha,
        "dof_pos_lag": dof_pos_lag,
        "dof_vel_SRD": dof_vel_SRD,
        "dof_vel_SD": dof_vel_SD,
        "dof_vel_SRD_decimated": dof_vel_SRD_decimated,
        "dof_vel_SD_decimated": dof_vel_SD_decimated,
    }

    if not sim:
        result["low_pass_lag"] = low_pass_lag
        result["dof_vel_raw_SRD"] = dof_vel_raw_SRD
        result["dof_vel_raw_SD"] = dof_vel_raw_SD
        result["dof_vel_raw_SRD_decimated"] = dof_vel_raw_SRD_decimated
        result["dof_vel_raw_SD_decimated"] = dof_vel_raw_SD_decimated

    os.makedirs(log_dir / "yaml", exist_ok=True)
    with open(
        log_dir
        / "yaml"
        / f"{dof_name}_{sim_suffix}_{period * env.dt:.1f}_{kp:.1f}_{kd:.1f}_{feed_forward_ratio:.1f}_{low_pass_alpha:.1f}.yaml",
        "w",
    ) as f:
        yaml.dump(result, f)

    return result


def main(
    show_viewer: bool = False,
    device: str = "cpu",
    sim: bool = True,
) -> None:
    # Load checkpoint and env_args
    env_args = EnvArgsRegistry["g1_fixed"]

    robot_args = env_args.robot_args.model_copy(
        update={
            "decimation": 4,
            "low_pass_alpha": 0.3,
            "feed_forward_ratio": 0.5,
        }
    )
    env_args = env_args.model_copy(
        update={
            "robot_args": robot_args,
            "action_latency": 0,
        }
    )

    if sim and not show_viewer:
        global WAIT_FOR_NEXT_CTRL
        WAIT_FOR_NEXT_CTRL = False

    if sim:
        print("Running in SIMULATION mode")
        from gs_env.sim.envs.locomotion.leggedrobot_env import LeggedRobotEnv

        env = LeggedRobotEnv(
            args=env_args,
            num_envs=1,
            show_viewer=show_viewer,
            device=torch.device(device),
            eval_mode=True,
        )
        env.eval()
        env.reset()

    else:
        print("Running in REAL ROBOT mode")
        from gs_env.real import UnitreeLeggedEnv

        env = UnitreeLeggedEnv(
            env_args, action_scale=1.0, interactive=True, device=torch.device(device)
        )

        print("Press Start button to start the test")
        while not env.robot.Start:
            time.sleep(0.1)

    # DoF names, lower bound, upper bound
    test_dofs = [
        # "hip_roll": [0.0, 0.8],
        # "hip_pitch": [-0.5, 0.5],
        # "hip_yaw": [-0.5, 0.5],
        # "knee": [0.0, 1.0],
        # "ankle_roll": [-0.2, 0.2],
        # "ankle_pitch": [-0.5, 0.5],
        # "waist_yaw": [-1.0, 1.0],
        # "waist_roll": [-0.3, 0.3],
        # "waist_pitch": [-0.3, 0.3],
        # "shoulder_roll": [0.0, 1.0],
        # "shoulder_pitch": [-0.5, 0.5],
        # "shoulder_yaw": [0.0, 1.0],
        ["elbow", [0.0, 1.0]],
        # "wrist_roll": [-1.0, 1.0],
        # "wrist_pitch": [-1.0, 1.0],
        # "wrist_yaw": [0.0, 1.0],
    ]

    for i, name in enumerate(env.dof_names):
        if test_dofs[0][0] in name:
            dof_idx = i
            break
    if sim:
        kp = env.robot.dof_kp[dof_idx].item()
        kd = env.robot.dof_kd[dof_idx].item()
    else:
        kp = env.robot.kp[dof_idx]
        kd = env.robot.kd[dof_idx]
    # periods = [1, 2, 4]
    periods = [
        2,
    ]
    pds = [[kp / 2, kd / 2], [kp / 2, kd], [kp, kd], [kp, kd * 2], [kp * 2, kd * 2]]
    # feed_forward_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    feed_forward_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    # low_pass_alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    low_pass_alphas = [1.0]

    def run_dof_diagnosis_fixed() -> None:
        nonlocal env, test_dofs, pds, feed_forward_ratios, low_pass_alphas, periods
        dof_names = env.dof_names

        log_dir = Path(__file__).parent / "logs" / "pd_test"

        for kp, kd in pds:
            for feed_forward_ratio in feed_forward_ratios:
                for low_pass_alpha in low_pass_alphas:
                    for period in periods:
                        dof_name, (lower_bound, upper_bound) = test_dofs[0]
                        dof_idx = -1
                        for i, name in enumerate(dof_names):
                            if dof_name in name:
                                dof_idx = i
                                break
                        if dof_idx == -1:
                            print(f"Dof {dof_name} not found")
                            continue
                        amplitude = (upper_bound - lower_bound) / 2
                        offset = (upper_bound + lower_bound) / 2
                        run_single_dof_diagnosis(
                            env,
                            dof_idx=dof_idx,
                            dof_name=dof_name,
                            num_dofs=env.action_dim,
                            period=int(period / env.dt),
                            amplitude=amplitude,
                            offset=offset,
                            log_dir=log_dir,
                            sim=sim,
                            kp=kp,
                            kd=kd,
                            feed_forward_ratio=feed_forward_ratio,
                            low_pass_alpha=low_pass_alpha,
                        )

    try:
        run_dof_diagnosis_fixed()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    fire.Fire(main)
