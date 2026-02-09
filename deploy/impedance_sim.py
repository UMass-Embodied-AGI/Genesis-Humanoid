import sys
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend to prevent windows from showing
import matplotlib.pyplot as plt
import numpy as np
import torch
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from gs_env.sim.envs.locomotion.leggedrobot_env import LeggedRobotEnv

# Add examples to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from utils import plot_metric_on_axis  # type: ignore


def get_batched_period(y: np.ndarray) -> np.ndarray:
    """
    y : array of shape (B, N)
        Each row is a signal y(x) sampled at x = 0, 1, ..., N-1

    Returns:
        period : array of shape (B,)
            Average gap between detected local maxima for each batch (in samples)
    """

    if y.ndim != 2:
        raise ValueError("Expected y to have shape (B, N)")

    batch_size, num_samples = y.shape
    periods = np.full(batch_size, np.nan, dtype=np.float64)

    if num_samples < 3:
        return np.full(batch_size, num_samples, dtype=np.float64)

    # Identify local maxima by comparing each point to its neighbors
    mid = y[:, 1:-1]
    prev_vals = y[:, :-2]
    next_vals = y[:, 2:]
    is_local_max = (mid > prev_vals) & (mid >= next_vals)

    for batch_idx in range(batch_size):
        peak_indices = np.nonzero(is_local_max[batch_idx])[0] + 1  # offset by 1 due to slicing
        if peak_indices.size >= 2:
            periods[batch_idx] = np.diff(peak_indices).mean()
        else:
            # Fallback to total duration when not enough peaks are detected
            periods[batch_idx] = num_samples

    return periods


def resonate_dof_test(
    env: LeggedRobotEnv,
    dof_idx: int = 0,
) -> torch.Tensor:
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))

    dt = env.dt / env.decimation

    env.reset()
    action = torch.zeros((env.num_envs, env.action_dim), device=env.device)

    dof_pos = env.default_dof_pos[None, :].repeat(env.num_envs, 1)
    env.robot.set_state(dof_pos=dof_pos)

    TOTAL_RESET_STEPS = 10
    for _ in range(TOTAL_RESET_STEPS):
        env.apply_action(action)

    dof_pos[:, dof_idx] += 0.1
    env.robot.set_state(dof_pos=dof_pos)

    env.robot.start_logging()

    TOTAL_OSCILLATION_STEPS = 500
    for _ in range(TOTAL_OSCILLATION_STEPS):
        env.apply_action(action)

    dof_pos_history = env.robot.stop_logging()["dof_pos"][:, :, dof_idx]

    natural_period = get_batched_period(dof_pos_history.cpu().numpy()) * dt

    plot_metric_on_axis(
        axes[0],
        np.arange(dof_pos_history.shape[1]) * dt,
        [
            dof_pos_history[0].tolist(),
        ],
        [
            "Dof Pos",
        ],
        "Dof Pos",
        "Resonance Test",
        yscale="linear",
    )

    plot_metric_on_axis(
        axes[1],
        np.arange(dof_pos_history.shape[1]) * dt,
        [
            dof_pos_history[1].tolist(),
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

    print("Running in SIMULATION mode")

    env = LeggedRobotEnv(
        args=env_args,
        num_envs=10,
        show_viewer=show_viewer,
        device=torch.device(device),
        eval_mode=True,
    )
    env.eval()
    env.reset()

    extra_impedance = {
        "knee": 20 * 0.2**2,
        "hip_yaw": 40 * 0.15**2,
        "waist_pitch": 10 * 0.2**2,
        "waist_yaw": 10 * 0.2**2,
    }

    def run_pd_test() -> None:
        NUM_EPOCHS = 1
        FREQ_N = 10
        DAMP_RATIO = 1.0
        nonlocal env, extra_impedance
        post_order_dof_names = [joint.name for joint in reversed(env.robot.joints)]
        # post_order_dof_names = ["left_ankle_pitch_joint"]
        dof_names = env.dof_names
        dof_kp = env.robot.dof_kp
        dof_kd = env.robot.dof_kd

        for _ in range(NUM_EPOCHS):
            for dof_name in post_order_dof_names:
                dof_idx = dof_names.index(dof_name)

                tmp_dof_kp = dof_kp[None, :].repeat(env.num_envs, 1)
                tmp_dof_kd = dof_kd[None, :].repeat(env.num_envs, 1)
                tmp_dof_kp[:, dof_idx] *= torch.rand(env.num_envs, device=env.device)
                tmp_dof_kd[:, dof_idx] = 0.01
                env.robot.set_batched_dof_kp(tmp_dof_kp)
                env.robot.set_batched_dof_kd(tmp_dof_kd)

                batched_natural_period = resonate_dof_test(
                    env,
                    dof_idx=dof_idx,
                )

                natural_frequency = 1.0 / batched_natural_period * 2 * np.pi
                nonzero_mask = natural_frequency > 0.0
                batched_impedance = tmp_dof_kp[:, dof_idx][nonzero_mask] / (
                    natural_frequency[nonzero_mask] ** 2
                )
                impedance = batched_impedance.mean().item()
                print(
                    f"{dof_name}: {batched_impedance.mean().item():.2f} ± {batched_impedance.std().item():.2f}"
                )

                for key, value in extra_impedance.items():
                    if key in dof_name:
                        impedance += value
                dof_kp[dof_idx] = impedance * FREQ_N**2
                dof_kd[dof_idx] = 2 * DAMP_RATIO * impedance * FREQ_N
                print(f"PD Gains: {dof_kp[dof_idx]:.2f}, {dof_kd[dof_idx]:.2f}")

    try:
        run_pd_test()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    fire.Fire(main)
