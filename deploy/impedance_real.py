import sys
import time
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend to prevent windows from showing
import matplotlib.pyplot as plt
import numpy as np
import torch
from gs_env.real.leggedrobot_env import UnitreeLeggedEnv
from gs_env.sim.envs.config.registry import EnvArgsRegistry

# Add examples to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from utils import plot_metric_on_axis  # type: ignore


def get_period(y: np.ndarray) -> float:
    """
    y : array of shape (B, N)
        Each row is a signal y(x) sampled at x = 0, 1, ..., N-1

    Returns:
        period : array of shape (B,)
            Average gap between detected local maxima for each batch (in samples)
    """

    num_samples = y.shape[0]

    if num_samples < 3:
        return num_samples

    # Identify local maxima by comparing each point to its neighbors
    mid = y[1:-1]
    prev_vals = y[:-2]
    next_vals = y[2:]
    is_local_max = (mid > prev_vals) & (mid >= next_vals)

    peak_indices = np.nonzero(is_local_max)[0] + 1  # offset by 1 due to slicing
    if len(peak_indices) >= 2:
        period = np.diff(peak_indices).mean()
    else:
        # Fallback to total duration when not enough peaks are detected
        period = num_samples

    return period


def resonate_dof_test(
    env: UnitreeLeggedEnv,
    dof_idx: int = 0,
) -> float:
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))

    dt = env.robot.logging_interval

    action = torch.zeros((1, env.action_dim), device=env.device)
    action[0, dof_idx] = 0.1 / env.action_scale[dof_idx]

    last_update_time = time.time()

    TOTAL_RESET_STEPS = 50
    for i in range(TOTAL_RESET_STEPS):
        while time.time() - last_update_time < 0.02:
            time.sleep(0.001)
        last_update_time = time.time()
        env.apply_action(action * i / TOTAL_RESET_STEPS)
    TOTAL_RESET_STEPS = 50
    for _ in range(TOTAL_RESET_STEPS):
        while time.time() - last_update_time < 0.02:
            time.sleep(0.001)
        last_update_time = time.time()
        env.apply_action(action)

    action *= 0.0

    env.robot.start_logging()
    print("Started logging")

    TOTAL_OSCILLATION_STEPS = 100
    for _ in range(TOTAL_OSCILLATION_STEPS):
        while time.time() - last_update_time < 0.02:
            time.sleep(0.001)
        last_update_time = time.time()
        env.apply_action(action)

    dof_pos_history = env.robot.stop_logging()["dof_pos"][:, dof_idx]

    natural_period = get_period(dof_pos_history) * dt

    plot_metric_on_axis(
        axes[0],
        np.arange(dof_pos_history.shape[0]) * dt,
        [
            dof_pos_history.tolist(),
        ],
        [
            "Dof Pos",
        ],
        "Dof Pos",
        "Resonance Test",
        yscale="linear",
    )

    plot_path = "./debug.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    return natural_period


def main(
    show_viewer: bool = False,
    device: str = "cpu",
) -> None:
    # Load checkpoint and env_args
    env_args = EnvArgsRegistry["g1_fixed"]

    robot_args = env_args.robot_args.model_copy(
        update={
            "decimation": 4,
        }
    )
    env_args = env_args.model_copy(update={"robot_args": robot_args})

    print("Running in REAL ROBOT mode")

    from gs_env.real import UnitreeLeggedEnv

    env = UnitreeLeggedEnv(
        env_args, action_scale=1.0, interactive=True, device=torch.device(device)
    )

    print("Press Start button to start the test")
    while not env.robot.Start:
        time.sleep(0.1)

    def run_pd_test() -> None:
        nonlocal env
        dof_names = env.dof_names
        test_dof_names = [
            # "hip_roll",
            "hip_pitch",
            # "hip_yaw",
            # "knee",
            # "ankle_roll",
            # "ankle_pitch",
            # "waist_yaw",
            # "waist_roll",
            # "waist_pitch",
            # "shoulder_roll",
            # "shoulder_pitch",
            # "shoulder_yaw",
            # "elbow",
            # "wrist_roll",
            # "wrist_pitch",
            # "wrist_yaw",
        ]
        for test_dof_name in test_dof_names:
            dof_idx = -1
            for i, dof_name in enumerate(dof_names):
                if test_dof_name in dof_name:
                    dof_idx = i
                    break

            if dof_idx == -1:
                continue

            dof_kp = env.robot.kp[dof_idx]
            dof_kd = env.robot.kd[dof_idx]
            print(f"Dof {test_dof_name}: {dof_kp:.2f}, {dof_kd:.2f}")
            env.robot.kd[dof_idx] = 0.0

            batched_natural_period = resonate_dof_test(
                env,
                dof_idx=dof_idx,
            )

            natural_frequency = 1.0 / batched_natural_period * 2 * np.pi
            impedance = dof_kp / (natural_frequency**2)
            print(f"{test_dof_name}: {impedance:.2f}")

    run_pd_test()


if __name__ == "__main__":
    fire.Fire(main)
