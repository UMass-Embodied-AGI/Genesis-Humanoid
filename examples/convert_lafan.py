import os
import pickle
import time
from pathlib import Path
from typing import Any, cast

import gs_env.sim.envs as gs_envs
import numpy as np
import torch
import yaml
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from gs_env.sim.envs.config.schema import MotionEnvArgs
from gs_env.sim.scenes.config.registry import SceneArgsRegistry


def lafan_to_motion_data(
    env: gs_envs.MotionEnv, data: torch.Tensor, show_viewer: bool = False
) -> None | dict[str, Any]:
    """
    Convert LAFAN data to motion data
    If show_viewer is True, the environment will be rendered in a separate thread.
    Returns:
        motion_data: The motion data
        None: If show_viewer is True, the environment will be rendered in a separate thread.
    """

    links = env.robot.robot.links
    link_names = [link.name for link in links]
    dof_names = env.dof_names

    motion_data = {}
    motion_data["fps"] = 30
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
        nonlocal env, data, motion_data, show_viewer
        last_update_time = time.time()
        foot_links_idx = env.robot.foot_links_idx
        for i in range(len(data)):
            sliced_data = data[i]
            pos = sliced_data[:3]
            quat = sliced_data[[6, 3, 4, 5]]
            dof_pos = sliced_data[7:]
            # dof_pos[[19, 20, 21, 26, 27, 28]] = 0.0
            env.robot.set_state(pos=pos, quat=quat, dof_pos=dof_pos)
            env.update_buffers()
            pos_list.append(env.base_pos[0].clone())
            quat_list.append(env.base_quat[0].clone())
            dof_pos_list.append(env.dof_pos[0].clone())
            link_pos_list.append(env.link_positions[0].clone())
            link_quat_list.append(env.link_quaternions[0].clone())

            # compute foot contact
            foot_pos = env.link_positions[0, foot_links_idx, :]
            foot_lift = torch.clamp((foot_pos[:, 2] - 0.2) / 0.2, 0.0, 1.0)
            if i == 0:
                foot_last_pos = foot_pos.clone()
            foot_vel = torch.clamp(
                (torch.norm((foot_pos[..., :2] - foot_last_pos[..., :2]) / env.dt, dim=-1) - 0.2)
                / 0.2,
                0.0,
                1.0,
            )
            foot_last_pos = foot_pos.clone()
            foot_not_contact = (foot_lift + foot_vel).clamp(0.0, 1.0)
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

    # add files in directory assets/lafan
    csv_files = [f for f in Path("./assets/lafan").glob("*.csv")]
    # csv_files = ["./assets/lafan/run2_subject4.csv"]

    log_dir = Path("./assets/motion/lafan")
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

    for csv_file in csv_files:
        data = np.genfromtxt(csv_file, delimiter=",")
        data = torch.from_numpy(data).to(torch.float32)
        motion_name = os.path.basename(csv_file).split(".")[0]
        motion_file = motion_name + ".pkl"
        motion_path = log_dir / motion_file
        motion_data = lafan_to_motion_data(env=env, data=data, show_viewer=show_viewer)
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
            print(f"Skipping motion data for {csv_file}")

    dataset_yaml["motions"].sort(key=lambda x: x["file"])
    with open(log_dir / "lafan.yaml", "w") as f:
        yaml.dump(dataset_yaml, f)
