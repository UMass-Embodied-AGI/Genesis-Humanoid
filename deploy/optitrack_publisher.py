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
    quat_inv,
    quat_mul,
)
from gs_env.common.utils.motion_utils import G1Retargeter
from gs_env.real.config.registry import EnvArgsRegistry
from gs_env.real.config.schema import OptitrackEnvArgs
from gs_env.real.optitrack.NatNetClient import setup_optitrack
from gs_env.real.optitrack.optitrack_config import RIGID_BODY_ID_MAP, track_id_offset


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


class OptitrackReceiver:
    """Connects to OptiTrack (NatNet) and provides the latest skeleton frame."""

    SKELETON_ORDER = [RIGID_BODY_ID_MAP[i + track_id_offset] for i in range(1, 52)]

    def __init__(self, server_ip: str, client_ip: str, use_multicast: bool) -> None:
        optitrack_env_args = EnvArgsRegistry["g1_links_tracking"]
        assert isinstance(optitrack_env_args, OptitrackEnvArgs)

        self.server_ip = server_ip if server_ip != "0.0.0.0" else optitrack_env_args.server_ip
        self.client_ip = client_ip if client_ip != "0.0.0.0" else optitrack_env_args.client_ip

        self._client = setup_optitrack(
            server_address=self.server_ip,
            client_address=self.client_ip,
            use_multicast=use_multicast,
        )

        self.name_to_idx: dict[str, int] = {n: i for i, n in enumerate(self.SKELETON_ORDER)}
        self.link_names: list[str] = list(self.SKELETON_ORDER)

        self.global_rot = quat_from_euler(torch.tensor([0.0, 0.0, 1.0]) * torch.pi / 2.0).view(1, 4)
        self.global_rot_inv = quat_inv(self.global_rot)

    def start(self) -> None:
        self._client.run()

    def shutdown(self) -> None:
        try:
            self._client.shutdown()
        except Exception:
            pass

    def get_frame(self) -> dict[int, list[list[float]]]:
        return self._client.get_frame()

    def get_frame_number(self) -> int:
        return self._client.get_frame_number()

    def _parse_frame(
        self, frame: dict[int, list[list[float]]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_link_pos = torch.zeros((51, 3), dtype=torch.float32)
        all_link_quat = torch.zeros((51, 4), dtype=torch.float32)
        global_rot = self.global_rot.repeat(51, 1)
        global_rot_inv = self.global_rot_inv.repeat(51, 1)
        all_link_quat[:, 0] = 1.0
        for rb_id, (p, q) in frame.items():
            rb_id = rb_id % 65536
            if rb_id not in RIGID_BODY_ID_MAP:
                # raise ValueError(f"Unmapped RB ID {rb_id}!! Please check RIGID_BODY_ID_MAP.")
                continue
            name = RIGID_BODY_ID_MAP[rb_id]
            if name not in self.name_to_idx:
                continue
            idx = self.name_to_idx[name]
            all_link_pos[idx] = torch.tensor(p, dtype=torch.float32)
            all_link_quat[idx] = torch.roll(torch.tensor(q, dtype=torch.float32), 1)
        # Axis conversion -y to +x, along z
        all_link_pos = quat_apply(global_rot, all_link_pos)
        all_link_quat = quat_mul(quat_mul(global_rot, all_link_quat), global_rot_inv)

        return all_link_pos, all_link_quat

    def get_links(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Returns (all_link_pos [51,3], all_link_quat [51,4], frame_id)."""
        frame = self.get_frame()
        frame_id = self.get_frame_number()
        all_link_pos, all_link_quat = self._parse_frame(frame)
        return all_link_pos, all_link_quat, frame_id


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
        server_ip: str,
        client_ip: str,
        use_multicast: bool,
        freq_hz: float,
        save: bool,
        save_dir: str,
    ) -> None:
        self._redis = redis.from_url(redis_url)
        self.key_prefix = key_prefix
        self.freq_hz = freq_hz
        self.save = save
        self.save_dir = save_dir

        self.receiver = OptitrackReceiver(
            server_ip=server_ip,
            client_ip=client_ip,
            use_multicast=use_multicast,
        )
        self.retargeter = G1Retargeter(joint_space_retarget=False)

        # Fixed tracked-link order fed into retargeter
        self.tracked_link_names = (
            "LeftFoot",
            "RightFoot",
            "LeftHand",
            "RightHand",
            "Spine1",
            "Hips",
        )
        self._name_to_idx_51 = {n: i for i, n in enumerate(self.receiver.link_names)}
        self._tracked_indices_51 = [self._name_to_idx_51[n] for n in self.tracked_link_names]
        self.save_data = {
            "fps": int(self.freq_hz),
            "link_names": self.tracked_link_names,
            "pos": [],
            "quat": [],
            "frame_id": [],
            "foot_contact": [],
        }

        # Foot contact calibration/state (computed from raw 51-link positions)
        self._foot_contact_height_thresh = 0.04
        self._foot_contact_velocity_thresh = 0.5
        self._foot_initial_height = None
        self._foot_indices_51 = [
            self._name_to_idx_51["LeftFoot"],
            self._name_to_idx_51["RightFoot"],
        ]
        self._foot_last_pos = torch.zeros((2, 3), dtype=torch.float32)

    def publish(self, key: str, value: torch.Tensor, frame_id: int) -> None:
        self._redis.set(f"{self.key_prefix}:motion:{key}", json.dumps(_to_list(value)))
        self._redis.set(f"{self.key_prefix}:timestamp:{key}", frame_id)

    def _get_foot_contact(self, all_link_pos: torch.Tensor) -> torch.Tensor:
        foot_pos = all_link_pos[self._foot_indices_51, :]
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
        print("[optitrack_publisher] Started")
        print(f"Redis key prefix: {self.key_prefix}")
        print(f"OptiTrack server IP: {self.receiver.server_ip}")
        print(f"OptiTrack client IP: {self.receiver.client_ip}")
        print("=" * 80)

        self.receiver.start()
        self.receiver.get_links()
        print("[optitrack_publisher] Successfully received data from OptiTrack server.")
        print("[optitrack_publisher] Press any key to calibrate and start publishing...")
        getch()

        try:
            next_publish_time = time.time() + 1.0 / self.freq_hz
            while True:
                all_link_pos, all_link_quat, frame_id = self.receiver.get_links()

                tracked_pos = all_link_pos[self._tracked_indices_51, :]
                tracked_quat = all_link_quat[self._tracked_indices_51, :]
                foot_contact = self._get_foot_contact(all_link_pos)

                if self.save:
                    self.save_data["pos"].append(tracked_pos.detach().cpu().clone())
                    self.save_data["quat"].append(tracked_quat.detach().cpu().clone())
                    self.save_data["foot_contact"].append(foot_contact.detach().cpu().clone())
                    self.save_data["frame_id"].append(frame_id)

                if not self.retargeter.calibrated:
                    self.retargeter.calibrate(
                        tracked_pos=tracked_pos,
                        tracked_quat=tracked_quat,
                    )
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
                    # print("Optitrack is lagging behind")
                    next_publish_time = time.time() + 1.0 / self.freq_hz
                    continue
                time.sleep(max(0.0, next_publish_time - time.time()))
                next_publish_time += 1.0 / self.freq_hz
        except KeyboardInterrupt:
            print("\n[optitrack_publisher] Stopped by user.")
        finally:
            if self.save:
                filename = (
                    f"optitrack_{self.save_data['frame_id'][0]}_{self.save_data['frame_id'][-1]}"
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
    parser.add_argument("--server_ip", type=str, default="0.0.0.0")
    parser.add_argument("--client_ip", type=str, default="0.0.0.0")
    parser.add_argument("--use_multicast", action="store_true", default=False)
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, default="assets/optitrack")
    args = parser.parse_args()

    RedisMotionPublisher(
        redis_url=args.redis_url,
        key_prefix=args.key,
        server_ip=args.server_ip,
        client_ip=args.client_ip,
        use_multicast=args.use_multicast,
        freq_hz=120.0,
        save=args.save,
        save_dir=args.save_dir,
    ).run()
