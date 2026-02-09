import time

import fire
import torch
from gs_env.real.leggedrobot_env import UnitreeLeggedEnv
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from gs_env.sim.envs.config.schema import MotionEnvArgs
from gs_env.sim.envs.locomotion.motion_env import MotionEnv

XML_PATH = "assets/robot/unitree_g1/g1_mocap_29dof.xml"


def main(
    device: str = "cpu",
) -> None:
    device = "cpu" if not torch.cuda.is_available() else device
    device = torch.device(device)  # type: ignore[arg-type]

    env_args: MotionEnvArgs = EnvArgsRegistry["g1_motion"]  # type: ignore
    env_args = env_args.model_copy(update={"motion_file": "assets/motion/evaluate.pkl"})
    sim_env = MotionEnv(args=env_args, num_envs=1, show_viewer=True, device=device)  # type: ignore[arg-type]

    env_args: MotionEnvArgs = EnvArgsRegistry["g1_motion"]  # type: ignore
    real_env = UnitreeLeggedEnv(args=env_args, interactive=False, device=device, xml_path=XML_PATH)  # type: ignore[arg-type]

    print("=" * 80)
    print("Starting visualization")
    print(f"Device: {device}")
    print("=" * 80)

    try:
        last_update_time = time.time()
        while True:
            # Control loop timing (50 Hz)
            if time.time() - last_update_time < 0.02:
                time.sleep(0.005)
                continue
            last_update_time = time.time()

            sim_env.set_dof_pos(real_env.dof_pos[0])
            # sim_env.robot.set_state(quat=real_env.quat[0])
            link_idx_local = sim_env.get_link_idx_local_by_name("pelvis")
            base_pos = torch.tensor([0.0, 0.0, 1.0])
            sim_env.set_link_pose(link_idx_local, pos=base_pos, quat=real_env.base_quat[0])

            tracking_link_pos = real_env.tracking_link_pos_local_yaw
            tracking_link_quat = real_env.tracking_link_quat_local_yaw
            # quat_yaw = quat_from_angle_axis(
            #     real_env.base_euler[0, 2], torch.tensor([0, 0, 1], device=device, dtype=torch.float)
            # )
            # tracking_link_pos = quat_apply(quat_yaw, tracking_link_pos)
            # tracking_link_quat = quat_mul(quat_yaw, tracking_link_quat)
            for link_name in sim_env.scene.objects.keys():  # type: ignore
                if link_name in env_args.tracking_link_names:
                    link_idx = env_args.tracking_link_names.index(link_name)
                    link_pos = tracking_link_pos[:, link_idx, :]
                    link_pos += base_pos
                    link_quat = tracking_link_quat[:, link_idx, :]
                    sim_env.scene.set_obj_pose(link_name, pos=link_pos, quat=link_quat)  # type: ignore

            sim_env.step_visualizer()

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    fire.Fire(main)
