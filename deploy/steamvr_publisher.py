import argparse
import json
import sys
import termios
import time
import tty

import redis
import torch
from gs_env.common.utils.math_utils import (
    quat_apply,
    quat_from_euler,
    quat_mul,
    quat_to_euler,
    rotmat_to_quat,
)
from gs_env.common.utils.motion_utils import G1Retargeter
from gs_env.real.steamvr.SteamVRClient import SteamVRClient


def _to_list(t: torch.Tensor) -> list[float]:
    return t.detach().cpu().flatten().tolist()


def getch() -> str:
    """Non-blocking single-key input (Linux/macOS)."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def _on_button(button_state: tuple[int, int], button: str) -> bool:
    lb_map = {
        "LX": 1 << 0,
        "LY": 1 << 1,
        "LTrigger": 1 << 2,
        "LGrip": 1 << 3,
        "LClick": 1 << 4,
    }
    rb_map = {
        "RA": 1 << 0,
        "RB": 1 << 1,
        "RTrigger": 1 << 2,
        "RGrip": 1 << 3,
        "RClick": 1 << 4,
    }
    if button in lb_map:
        return (button_state[0] & lb_map[button]) != 0
    elif button in rb_map:
        return (button_state[1] & rb_map[button]) != 0
    else:
        return False


class SteamVRReceiver(SteamVRClient):
    def __init__(
        self, udp_host: str = "0.0.0.0", udp_port: int = 5005, device: str = "cpu"
    ) -> None:
        super().__init__(udp_host, udp_port, device)

        # x_t = -z_o, y_t = -x_o, z_t = y_o
        # SteamVR uses right-handed Y-up coordinate system
        # Target uses right-handed Z-up coordinate system
        A = torch.tensor(
            [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        self.global_rot = rotmat_to_quat(A).view(1, 4)
        self.global_rot_inv = rotmat_to_quat(A.T).view(1, 4)
        # x_t = z_o, y_t = -y_o, z_t = x_o
        A = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        self.base_local_rot = rotmat_to_quat(A)
        # x_t = -x_o, y_t = -y_o, z_t = z_o
        A = torch.tensor(
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        self.foot_local_rot = rotmat_to_quat(A)
        # tracker x axis to body center
        self.x_shift = torch.tensor([-0.10, 0.0, 0.0], dtype=torch.float32)

    def manual_shift_x(self, pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        euler = quat_to_euler(quat)
        euler[:, 0] = 0.0
        euler[:, 1] = 0.0
        quat_xy = quat_from_euler(euler)
        shifted_pos = pos + quat_apply(quat_xy, self.x_shift.view(1, 3).repeat(euler.shape[0], 1))
        return shifted_pos

    def get_links(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        tracked_pos, tracked_quat, frame_id = super().get_links()
        tracked_pos_norm = tracked_pos.norm(dim=-1)
        if tracked_pos_norm.min() < 1e-4:
            raise ValueError("all zero link pose detected!")
        global_rot = self.global_rot.repeat(6, 1)
        global_rot_inv = self.global_rot_inv.repeat(6, 1)
        tracked_pos = quat_apply(global_rot, tracked_pos)
        tracked_quat = quat_mul(quat_mul(global_rot, tracked_quat), global_rot_inv)
        tracked_quat[[0, 1]] = quat_mul(tracked_quat[[0, 1]], self.foot_local_rot)
        tracked_quat[5] = quat_mul(tracked_quat[5], self.base_local_rot)
        shift_idxs = [0, 1, 4, 5]
        tracked_pos[shift_idxs] = self.manual_shift_x(
            tracked_pos[shift_idxs], tracked_quat[shift_idxs]
        )
        return tracked_pos, tracked_quat, frame_id


class RedisMotionPublisher:
    """
    Orchestrates OptiTrack receiving + retargeting, and publishes results into Redis.

    Redis keys:
        - {key}:motion:base_pos [3]
        - {key}:motion:base_quat [4] (w, x, y, z)
        - {key}:motion:base_lin_vel [3]
        - {key}:motion:base_ang_vel [3]
        - {key}:motion:base_ang_vel_local [3]
        - {key}:motion:link_pos_local [N*3]
        - {key}:motion:link_quat_local [N*4]
        - {key}:motion:link_lin_vel [N*3]
        - {key}:motion:link_ang_vel [N*3]
        - {key}:motion:foot_contact [F]
        - {key}:timestamp:* [1]
    """

    def __init__(
        self,
        redis_url: str,
        key_prefix: str,
        udp_host: str,
        udp_port: int,
        freq_hz: float,
        save: bool,
        save_dir: str,
    ) -> None:
        self._redis = redis.from_url(redis_url)
        self.key_prefix = key_prefix
        self.freq_hz = freq_hz
        self.save = save
        self.save_dir = save_dir

        self.receiver = SteamVRReceiver(
            udp_host=udp_host,
            udp_port=udp_port,
        )
        self.retargeter = G1Retargeter()
        self.retargeter.estimate_torso_quat = True

        self.save_data = {
            "fps": int(self.freq_hz),
            "pos": [],
            "quat": [],
            "frame_id": [],
            "foot_contact": [],
        }

        # Foot contact calibration/state (computed from raw 51-link positions)
        self._foot_contact_height_thresh = 0.04
        self._foot_contact_velocity_thresh = 0.5
        self._foot_initial_height = None
        self._foot_indices = [0, 1]
        self._foot_last_pos = torch.zeros((2, 3), dtype=torch.float32)

    def publish(self, key: str, value: torch.Tensor, frame_id: int) -> None:
        self._redis.set(f"{self.key_prefix}:motion:{key}", json.dumps(_to_list(value)))
        self._redis.set(f"{self.key_prefix}:timestamp:{key}", frame_id)

    def _get_foot_contact(self, all_link_pos: torch.Tensor) -> torch.Tensor:
        foot_pos = all_link_pos[self._foot_indices, :]
        if self._foot_initial_height is None:
            self._foot_initial_height = foot_pos[:, 2]
            self._foot_last_pos = foot_pos.clone()
        foot_height = foot_pos[:, 2]
        foot_not_contact_height = (
            (foot_height - self._foot_initial_height) / self._foot_contact_height_thresh
        ).clamp(0.0, 1.0)
        foot_velocity = (foot_pos - self._foot_last_pos) * self.freq_hz
        self._foot_last_pos = foot_pos.clone()
        foot_not_contact_velocity = (
            torch.norm(foot_velocity[..., :2], dim=-1) / self._foot_contact_velocity_thresh
        ).clamp(0.0, 1.0)
        foot_contact = 1 - (foot_not_contact_height + foot_not_contact_velocity).clamp(0.0, 1.0)
        return foot_contact

    def close(self) -> None:
        self.receiver.shutdown()

    def run(self) -> None:
        print("=" * 80)
        print("[steamvr_publisher] Started")
        print(f"Redis key prefix: {self.key_prefix}")
        print(f"UDP host: {self.receiver.udp_host}")
        print(f"UDP port: {self.receiver.udp_port}")
        print("=" * 80)

        self.receiver.start()
        self.receiver.get_links()
        print("[steamvr_publisher] Successfully received data from SteamVR server.")
        print("[steamvr_publisher] Press RT + A on controller to calibrate...")

        try:
            next_publish_time = time.time() + 1.0 / self.freq_hz
            while True:
                tracked_pos, tracked_quat, frame_id = self.receiver.get_links()

                foot_contact = self._get_foot_contact(tracked_pos)

                if self.save:
                    self.save_data["pos"].append(tracked_pos.detach().cpu().clone())
                    self.save_data["quat"].append(tracked_quat.detach().cpu().clone())
                    self.save_data["foot_contact"].append(foot_contact.detach().cpu().clone())
                    self.save_data["frame_id"].append(frame_id)

                if not self.retargeter.calibrated:
                    button_states = self.receiver.get_button_states()
                    if _on_button(button_states, "RTrigger") and _on_button(button_states, "RA"):
                        self.retargeter.calibrate(
                            tracked_pos=tracked_pos,
                            tracked_quat=tracked_quat,
                        )
                        print("[steamvr_publisher] Calibrated.")
                    continue

                retargeted = self.retargeter.step(
                    tracked_pos=tracked_pos,
                    tracked_quat=tracked_quat,
                    frame_id=frame_id,
                )
                retargeted["foot_contact"] = foot_contact
                for k, v in retargeted.items():
                    self.publish(k, v, frame_id)

                if time.time() >= next_publish_time:
                    # print("SteamVR is lagging behind")
                    next_publish_time = time.time() + 1.0 / self.freq_hz
                    continue
                time.sleep(max(0.0, next_publish_time - time.time()))
                next_publish_time += 1.0 / self.freq_hz
        except KeyboardInterrupt:
            print("\n[steamvr_publisher] Stopped by user.")
        finally:
            if self.save:
                filename = (
                    f"steamvr_{self.save_data['frame_id'][0]}_{self.save_data['frame_id'][-1]}"
                )
                self.save_data["pos"] = torch.stack(self.save_data["pos"], dim=0)
                self.save_data["quat"] = torch.stack(self.save_data["quat"], dim=0)
                self.save_data["foot_contact"] = torch.stack(self.save_data["foot_contact"], dim=0)
                self.save_data["frame_id"] = torch.tensor(
                    self.save_data["frame_id"], dtype=torch.int64
                )
                import os
                import pickle

                os.makedirs(self.save_dir, exist_ok=True)
                with open(os.path.join(self.save_dir, filename + ".pkl"), "wb") as f:
                    pickle.dump(self.save_data, f)
                print(f"Saved data to {os.path.join(self.save_dir, filename + '.pkl')}")
            self.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--redis_url", type=str, default="redis://localhost:6379/0")
    parser.add_argument("--key", type=str, default="motion:ref:latest")
    parser.add_argument("--udp_host", type=str, default="0.0.0.0")
    parser.add_argument("--udp_port", type=int, default=5005)
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, default="assets/steamvr")
    args = parser.parse_args()

    RedisMotionPublisher(
        redis_url=args.redis_url,
        key_prefix=args.key,
        udp_host=args.udp_host,
        udp_port=args.udp_port,
        freq_hz=60.0,
        save=args.save,
        save_dir=args.save_dir,
    ).run()
