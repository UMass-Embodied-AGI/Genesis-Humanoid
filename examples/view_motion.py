from __future__ import annotations

from typing import Any

import fire
import gs_env.sim.envs as gs_envs
import torch
from gs_env.sim.envs.config.registry import EnvArgsRegistry


def view_motion(env_args: Any) -> None:
    """View reference motion playback (no policy)."""
    # Create environment for evaluation
    env = gs_envs.MotionEnv(
        args=env_args,
        num_envs=1,
        show_viewer=True,
        device=torch.device("cpu"),
        eval_mode=True,
    )
    import time

    link_name_to_idx: dict[str, int] = {}
    for link_name in env.scene.objects.keys():
        link_name_to_idx[link_name] = env.motion_lib.tracking_link_names.index(link_name)

    def run() -> None:
        nonlocal env
        motion_id = 0
        while True:
            env.time_since_reset[0] = 0.0
            env.hard_reset_motion(torch.IntTensor([0]), motion_id)
            env.hard_sync_motion(torch.IntTensor([0]))
            last_update_time = time.time()
            while env.motion_times[0] + 0.02 < env.motion_lib.get_motion_length(
                torch.IntTensor([motion_id])
            ):
                env.scene.scene.step()
                env.time_since_reset[0] += 0.02
                env.hard_sync_motion(torch.IntTensor([0]))
                env.update_buffers()
                for link_name in env.scene.objects.keys():
                    link_pos = env.ref_tracking_link_pos_global[:, link_name_to_idx[link_name]]
                    link_quat = env.ref_tracking_link_quat_global[:, link_name_to_idx[link_name]]
                    env.scene.set_obj_pose(link_name, pos=link_pos, quat=link_quat)
                env.scene.scene.clear_debug_objects()
                for i in range(len(env.robot.foot_links_idx)):
                    env.scene.scene.draw_debug_arrow(
                        env.link_positions[0, env.robot.foot_links_idx[i]],
                        env.ref_foot_contact_weighted[0, i]
                        * torch.tensor([0.0, 0.0, 0.5], device=env.device),
                        radius=0.01,
                        color=(0.0, 0.0, 1.0),
                    )

                while time.time() - last_update_time < 0.02:
                    time.sleep(0.01)
                last_update_time = time.time()
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
        run()
    except KeyboardInterrupt:
        pass


def main(
    motion_file: str = "assets/motion/evaluate.pkl",
) -> None:
    env_args = EnvArgsRegistry["g1_motion"]
    env_args = env_args.model_copy(update={"motion_file": motion_file})
    view_motion(env_args)


if __name__ == "__main__":
    fire.Fire(main)
