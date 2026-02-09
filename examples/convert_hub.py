import os
import pickle
import time
from pathlib import Path
from typing import Any, cast

import gs_env.sim.envs as gs_envs
import joblib
import torch
import yaml
from gs_env.common.utils.math_utils import (
    quat_from_euler,
    quat_mul,
    quat_to_euler,
)
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from gs_env.sim.envs.config.schema import MotionEnvArgs
from gs_env.sim.scenes.config.registry import SceneArgsRegistry


def hub_to_motion_data(
    env: gs_envs.MotionEnv, data: dict[str, Any], show_viewer: bool = False
) -> None | dict[str, Any]:
    """
    Convert TWIST data to motion data
    If show_viewer is True, the environment will be rendered in a separate thread.
    Returns:
        motion_data: The motion data
        None: If show_viewer is True, the environment will be rendered in a separate thread.
    """

    links = env.robot.robot.links
    link_names = [link.name for link in links]
    dof_names = env.dof_names

    lafan_order = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    dof_index = []
    for dof_name in lafan_order:
        dof_index.append(dof_names.index(dof_name))
    motion_data = {}
    motion_data["fps"] = data["fps"]
    motion_data["link_names"] = link_names
    motion_data["dof_names"] = dof_names
    # save foot link indices and names for downstream usage
    foot_links_idx = env.robot.foot_links_idx
    motion_data["foot_link_indices"] = foot_links_idx
    pos_list = []
    quat_list = []
    dof_pos_list = []
    link_pos_list = []
    link_quat_list = []
    foot_contact_list = []

    def run() -> dict[str, Any]:
        nonlocal env, data, motion_data, show_viewer, dof_index
        last_update_time = time.time()
        foot_links_idx = env.robot.foot_links_idx
        quat_offset = quat_from_euler(torch.tensor([0.0, 0.0, 0.0]))
        for i in range(len(data["root_trans_offset"])):
            pos = torch.tensor(data["root_trans_offset"][i], dtype=torch.float32)
            quat = torch.tensor(data["root_rot"][i], dtype=torch.float32)[[3, 0, 1, 2]]
            quat = quat_mul(quat, quat_offset)
            dof_pos = torch.zeros(29, dtype=torch.float32)
            dof_pos[dof_index] = torch.tensor(data["dof"][i], dtype=torch.float32)
            env.robot.set_state(pos=pos, quat=quat, dof_pos=dof_pos)
            env.update_buffers()
            pos_list.append(env.base_pos[0].clone())
            quat_list.append(env.base_quat[0].clone())
            dof_pos_list.append(env.dof_pos[0].clone())
            link_pos_list.append(env.link_positions[0].clone())
            link_quat_list.append(env.link_quaternions[0].clone())

            # compute foot contact
            foot_pos = env.link_positions[0, foot_links_idx, :]
            foot_quat = env.link_quaternions[0, foot_links_idx, :]
            foot_euler = quat_to_euler(foot_quat)
            foot_tilt = torch.clamp(
                (torch.abs(foot_euler[:, 0]) + torch.abs(foot_euler[:, 1]) - 0.4) / 0.4, 0.0, 1.0
            )
            foot_lift = torch.clamp((foot_pos[:, 2] - 0.15) / 0.15, 0.0, 1.0)
            if i == 0:
                foot_last_pos = foot_pos.clone()
            foot_vel = torch.clamp(
                (torch.norm((foot_pos[..., :2] - foot_last_pos[..., :2]) / env.dt, dim=-1) - 0.1)
                / 0.1,
                0.0,
                1.0,
            )
            foot_last_pos = foot_pos.clone()
            foot_not_contact = ((foot_tilt + foot_lift + foot_vel) / 1.5).clamp(0.0, 1.0)
            foot_contact = 1 - foot_not_contact
            foot_contact_list.append(foot_contact)
            if show_viewer:
                env.scene.scene.clear_debug_objects()
                for i in range(len(foot_links_idx)):
                    env.scene.scene.draw_debug_arrow(
                        foot_pos[i],
                        foot_contact[i] * torch.tensor([0.0, 0.0, 0.5]),
                        radius=0.01,
                        color=(0.0, 0.0, 1.0),
                    )

            if show_viewer:
                env.scene.scene.step()
                while time.time() - last_update_time < 1 / motion_data["fps"]:
                    time.sleep(0.01)
                last_update_time = time.time()

        motion_data["pos"] = torch.stack(pos_list).numpy()
        motion_data["quat"] = torch.stack(quat_list).numpy()
        motion_data["dof_pos"] = torch.stack(dof_pos_list).numpy()
        motion_data["link_pos"] = torch.stack(link_pos_list).numpy()
        motion_data["link_quat"] = torch.stack(link_quat_list).numpy()
        motion_data["foot_contact"] = torch.stack(foot_contact_list).numpy()
        return motion_data

    try:
        return run()
    except KeyboardInterrupt:
        return None


if __name__ == "__main__":
    show_viewer = False

    # add files in directory assets/HuB
    pkl_files = [f for f in Path("./assets/HuB").glob("*.pkl")]
    # pkl_files = ["assets/HuB/squat.pkl",]

    log_dir = Path("./assets/motion/hub")
    os.makedirs(log_dir, exist_ok=True)

    env_args = cast(MotionEnvArgs, EnvArgsRegistry["g1_motion"]).model_copy(
        update={"scene_args": SceneArgsRegistry["flat_scene_legged"]}
    )
    env = gs_envs.MotionEnv(
        args=env_args,
        num_envs=1,
        show_viewer=show_viewer,
        device=torch.device("cpu"),
        eval_mode=True,
    )
    env.reset()

    dataset_yaml = {
        "root_path": str(log_dir),
        "motions": [],
    }

    for pkl_file in pkl_files:
        with open(pkl_file, "rb") as f:
            dict_data = joblib.load(f)
        for motion_name in dict_data.keys():
            data = dict_data[motion_name]
            motion_file = motion_name + ".pkl"
            motion_path = log_dir / motion_file
            motion_data = hub_to_motion_data(env=env, data=data, show_viewer=show_viewer)
            if motion_data is not None:
                print(f"Saving motion data to {motion_path}")
                with open(motion_path, "wb") as f:
                    pickle.dump(motion_data, f)
                dataset_yaml["motions"].append(
                    {
                        "file": motion_file,
                        "weight": 1.0,
                    }
                )
            else:
                print(f"Skipping motion data for {motion_name} in {pkl_file}")

    dataset_yaml["motions"].sort(key=lambda x: x["file"])
    with open(log_dir / "hub.yaml", "w") as f:
        yaml.dump(dataset_yaml, f)
