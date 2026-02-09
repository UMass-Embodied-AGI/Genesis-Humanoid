import pickle
import time
from pathlib import Path
from typing import Any

import gs_env.sim.envs as gs_envs
import numpy as np
import smplx
import torch
import yaml
from gs_env.common.utils.motion_utils import (
    GeneralMotionRetargeting,
    load_smplx_data_frames,
    load_smplx_file,
)
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from scipy.spatial.transform import Rotation as R

EXCLUDE_KEY_WORDS = ["bmlrub", "ekut", "crawl", "_lie", "upstairs", "downstairs"]


HUMAN_TO_ROBOT_TRACKING_DICT = {
    "pelvis": "pelvis",
    "spine3": "torso_link",
    "left_foot": "left_ankle_roll_link",
    "right_foot": "right_ankle_roll_link",
    "left_wrist": "left_wrist_yaw_link",
    "right_wrist": "right_wrist_yaw_link",
}


def retarget_smplx(
    smplx_data: list[dict[str, Any]],
    fps: int,
    actual_human_height: float,
    env: gs_envs.MotionEnv,
    show_viewer: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Retarget SMPLX data to the robot motion data
    """
    # Initialize the retargeting system
    retargeter = GeneralMotionRetargeting(
        robot_xml_file="assets/robot/unitree_g1/g1_mocap_29dof.xml",
        ik_config_file="assets/robot/unitree_g1/smplx_to_g1.json",
        actual_human_height=actual_human_height,
        aligned_fps=fps,
    )

    raw_tracking_link_names = [robot_name for _, robot_name in HUMAN_TO_ROBOT_TRACKING_DICT.items()]
    raw_tracking_link_pos_global = []
    raw_tracking_link_quat_global = []
    raw_pos_list = []
    raw_quat_list = []
    foot_links_idx = (
        raw_tracking_link_names.index("left_ankle_roll_link"),
        raw_tracking_link_names.index("right_ankle_roll_link"),
    )
    raw_motion_data = {
        "fps": fps,
        "link_names": raw_tracking_link_names,
        "dof_names": env.dof_names,
        "foot_link_indices": foot_links_idx,
    }
    base_idx = raw_tracking_link_names.index("pelvis")
    retargeted_tracking_link_names = [link.name for link in env.robot.robot.links]
    retargeted_tracking_link_pos_global = []
    retargeted_tracking_link_quat_global = []
    tracking_link_pos = torch.zeros_like(env.tracking_link_pos_global)[0]
    tracking_link_quat = torch.zeros_like(env.tracking_link_quat_local_yaw)[0]
    pos_list = []
    quat_list = []
    dof_pos_list = []
    foot_contact_list = []
    foot_last_pos = None
    foot_contact = torch.ones(2, dtype=torch.float32)
    retargeted_motion_data = {
        "fps": fps,
        "link_names": retargeted_tracking_link_names,
        "dof_names": env.dof_names,
        "foot_link_indices": env.robot.foot_links_idx,
    }

    frame_idx = 0
    frame_counter = 0
    retarget_start_time = time.time()
    speed_measurement_interval = 2.0

    while True:
        # Advance frame index
        if show_viewer:
            frame_idx = (frame_idx + 1) % len(smplx_data)
            # FPS measurements
            frame_counter += 1
            current_time = time.time()
            if current_time - retarget_start_time >= speed_measurement_interval:
                actual_fps = frame_counter / (current_time - retarget_start_time)
                print(f"Actual retargeting FPS: {actual_fps:.2f}")
                frame_counter = 0
                retarget_start_time = current_time

        else:
            frame_idx += 1
            if frame_idx >= len(smplx_data):
                break

        # Current SMPLX frame
        smplx_frame = smplx_data[frame_idx]

        # Retarget
        scaled_human_data = retargeter.process_human_data(smplx_frame)
        qpos = retargeter.retarget(scaled_human_data)
        qpos_t = torch.tensor(qpos, device=env.device, dtype=torch.float32)

        for j, (human_name, robot_name) in enumerate(HUMAN_TO_ROBOT_TRACKING_DICT.items()):
            if human_name in scaled_human_data.keys():
                pos, quat = scaled_human_data[human_name]
                pos_t = torch.tensor(pos, device=env.device, dtype=torch.float32)
                quat_t = torch.tensor(quat, device=env.device, dtype=torch.float32)
                if "ankle" in robot_name:
                    offset = torch.tensor([-0.1, 0, 0.02], device=env.device, dtype=torch.float32)
                    pos_t += R.from_quat(quat_t, scalar_first=True).apply(offset)
                if "torso" in robot_name:
                    offset = np.array([-0.0039635, 0.0, 0.044], dtype=float)
                    pos_t = torch.tensor(scaled_human_data["pelvis"][0]) + R.from_quat(
                        quat_t, scalar_first=True
                    ).apply(offset)
                tracking_link_pos[j] = pos_t
                tracking_link_quat[j] = quat_t

        foot_pos = tracking_link_pos[foot_links_idx, :]
        if foot_last_pos is not None:
            foot_vel = torch.clamp(
                (torch.norm((foot_pos[..., :2] - foot_last_pos[..., :2]) * fps, dim=-1) - 0.2)
                / 0.2,
                0.0,
                1.0,
            )
            foot_lift = torch.clamp((foot_pos[:, 2] - 0.2) / 0.2, 0.0, 1.0)
            foot_not_contact = (foot_lift + foot_vel).clamp(0.0, 1.0)
            foot_contact = 1 - foot_not_contact
        foot_last_pos = foot_pos.clone()
        foot_contact_list.append(foot_contact.clone())

        raw_tracking_link_pos_global.append(tracking_link_pos.clone())
        raw_tracking_link_quat_global.append(tracking_link_quat.clone())
        raw_pos_list.append(tracking_link_pos[base_idx].clone())
        raw_quat_list.append(tracking_link_quat[base_idx].clone())

        env.robot.set_state(
            pos=qpos_t[:3],
            quat=qpos_t[3:7],
            dof_pos=qpos_t[7:],
        )
        env.update_buffers()

        pos_list.append(qpos_t[:3].clone())
        quat_list.append(qpos_t[3:7].clone())
        dof_pos_list.append(qpos_t[7:].clone())
        retargeted_tracking_link_pos_global.append(env.link_positions[0].clone())
        retargeted_tracking_link_quat_global.append(env.link_quaternions[0].clone())

        if show_viewer:
            env.scene.scene.clear_debug_objects()
            for j, link_name in enumerate(raw_tracking_link_names):
                pos = tracking_link_pos[j]
                quat = tracking_link_quat[j]
                env.scene.set_obj_pose(link_name, pos=pos[None, :], quat=quat[None, :])  # type: ignore
            for i in range(len(foot_links_idx)):
                env.scene.scene.draw_debug_arrow(
                    foot_pos[i],
                    foot_contact[i] * torch.tensor([0.0, 0.0, 0.5]),
                    radius=0.01,
                    color=(0.0, 0.0, 1.0),
                )
            env.scene.scene.step()

    raw_motion_data["pos"] = torch.stack(raw_pos_list).numpy()
    raw_motion_data["quat"] = torch.stack(raw_quat_list).numpy()
    raw_motion_data["dof_pos"] = torch.stack(dof_pos_list).numpy()
    raw_motion_data["link_pos"] = torch.stack(raw_tracking_link_pos_global).numpy()
    raw_motion_data["link_quat"] = torch.stack(raw_tracking_link_quat_global).numpy()
    raw_motion_data["foot_contact"] = torch.stack(foot_contact_list).numpy()

    retargeted_motion_data["pos"] = torch.stack(pos_list).numpy()
    retargeted_motion_data["quat"] = torch.stack(quat_list).numpy()
    retargeted_motion_data["dof_pos"] = torch.stack(dof_pos_list).numpy()
    retargeted_motion_data["link_pos"] = torch.stack(retargeted_tracking_link_pos_global).numpy()
    retargeted_motion_data["link_quat"] = torch.stack(retargeted_tracking_link_quat_global).numpy()
    retargeted_motion_data["foot_contact"] = torch.stack(foot_contact_list).numpy()

    return raw_motion_data, retargeted_motion_data


def amass_to_motion_data(
    env: gs_envs.MotionEnv, body_models: Any, smplx_path: Path, show_viewer: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        smplx_path, body_models
    )

    smplx_data, fps = load_smplx_data_frames(smplx_data, body_model, smplx_output, tgt_fps=50)

    raw_motion_data, retargeted_motion_data = retarget_smplx(
        smplx_data, fps, actual_human_height, env, show_viewer
    )

    return raw_motion_data, retargeted_motion_data


def recursive_convert_amass(
    amass_path: Path,
    raw_path: Path,
    retargeted_path: Path,
    env: Any,
    body_models: Any,
    show_viewer: bool,
) -> str | None:
    """
    Recursively convert AMASS data to motion data
    """
    dataset_yaml = {
        "motions": [],
    }

    print(f"Converting {amass_path} to {raw_path} and {retargeted_path}")

    npz_files = list(amass_path.glob("*.npz"))
    for npz_file in npz_files:
        motion_name = npz_file.name.replace(".npz", "")
        if any(keyword in str(npz_file.absolute()).lower() for keyword in EXCLUDE_KEY_WORDS):
            continue
        raw_file = raw_path / f"{motion_name}.pkl"
        retargeted_file = retargeted_path / f"{motion_name}.pkl"
        if not raw_file.exists() or not retargeted_file.exists():
            raw_motion_data, retargeted_motion_data = amass_to_motion_data(
                env, body_models, npz_file, show_viewer
            )
            raw_path.mkdir(parents=True, exist_ok=True)
            retargeted_path.mkdir(parents=True, exist_ok=True)
            with open(raw_file, "wb") as f:
                pickle.dump(raw_motion_data, f)
            with open(retargeted_file, "wb") as f:
                pickle.dump(retargeted_motion_data, f)
            dataset_yaml["motions"].append({"file": f"{motion_name}.pkl", "weight": 1.0})

    for subfolder in amass_path.iterdir():
        if subfolder.is_dir() and not subfolder.name.startswith("."):
            sub_folder_name = subfolder.relative_to(amass_path)
            sub_folder_result = recursive_convert_amass(
                subfolder,
                raw_path / sub_folder_name,
                retargeted_path / sub_folder_name,
                env,
                body_models,
                show_viewer,
            )
            if sub_folder_result is not None:
                dataset_yaml["motions"].append(
                    {"file": str(sub_folder_name / sub_folder_result), "weight": 1.0}
                )

    if len(dataset_yaml["motions"]):
        raw_path.mkdir(parents=True, exist_ok=True)
        retargeted_path.mkdir(parents=True, exist_ok=True)

        yaml_file = f"{amass_path.name}.yaml"
        raw_dataset_yaml = dataset_yaml.copy()
        raw_dataset_yaml["root_path"] = str(raw_path)
        with open(raw_path / yaml_file, "w") as f:
            yaml.dump(raw_dataset_yaml, f)

        retargeted_dataset_yaml = dataset_yaml.copy()
        retargeted_dataset_yaml["root_path"] = str(retargeted_path)
        with open(retargeted_path / yaml_file, "w") as f:
            yaml.dump(retargeted_dataset_yaml, f)

        return yaml_file

    return None


if __name__ == "__main__":
    show_viewer = False
    AMASS_dir = "assets/AMASS"
    SMPLX_FOLDER = "assets/body_models"

    body_models = {
        "neutral": smplx.create(
            SMPLX_FOLDER,
            "smplx",
            gender="neutral",
            use_pca=False,
        ),
        "male": smplx.create(
            SMPLX_FOLDER,
            "smplx",
            gender="male",
            use_pca=False,
        ),
        "female": smplx.create(
            SMPLX_FOLDER,
            "smplx",
            gender="female",
            use_pca=False,
        ),
    }

    env_args = EnvArgsRegistry["g1_motion"]
    envclass = getattr(gs_envs, env_args.env_name)
    env = envclass(
        args=env_args,
        num_envs=1,
        show_viewer=show_viewer,
        device=torch.device("cpu"),
        eval_mode=True,
    )
    env.reset()

    recursive_convert_amass(
        Path(AMASS_dir),
        Path("./assets/motion/AMASS_raw"),
        Path("./assets/motion/AMASS"),
        env,
        body_models,
        show_viewer,
    )
