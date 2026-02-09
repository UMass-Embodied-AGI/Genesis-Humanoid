import socket
import threading
import time
from typing import Any

import torch


class SteamVRClient:
    """
    - Incomming socket format:
        FRAME:          int (1),
        HMD:            float (7),
        LEFTHAND:       float (7),
        LEFTBUTTON:     int (1),
        RIGHTHAND:      float (7),
        RIGHTBUTTON:    int (1),
        WAIST:          float (7),
        LEFTFOOT:       float (7),
        RIGHTFOOT:      float (7),
    - Outgoing format:
        link_pos:       torch.Tensor, shape (N_links, 3)
        link_quat:      torch.Tensor, shape (N_links, 4)
        frame_id:       int (1)
        button_states:  int (2)
        link order:
            left_ankle_roll_link,
            right_ankle_roll_link,
            left_wrist_yaw_link,
            right_wrist_yaw_link,
            torso_link,
            pelvis,
    """

    def __init__(
        self, udp_host: str = "0.0.0.0", udp_port: int = 5005, device: str = "cpu"
    ) -> None:
        self.udp_host = udp_host
        self.udp_port = udp_port

        self._device_str = device
        self.device = torch.device(device)

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((udp_host, udp_port))

        self._lock = threading.Lock()
        self._has_frame = threading.Event()
        self._latest_raw: dict[str, Any] = {}

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except Exception:
            pass

    def get_links(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        self._has_frame.wait()
        with self._lock:
            raw = self._latest_raw

        return raw["link_pos"], raw["link_quat"], raw["frame_id"]

    def get_button_states(self) -> tuple[int, int]:
        self._has_frame.wait()
        with self._lock:
            raw = self._latest_raw

        return raw["button_states"]

    # ---------------- internal ----------------

    def _recv_loop(self) -> None:
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(8192)  # blocking recv
            except OSError:
                break

            recv_time = time.time()
            msg = data.decode("utf-8", errors="ignore").strip()

            parsed = self._parse_frame_packet_to_lists(msg, recv_time)
            if parsed is None:
                continue

            with self._lock:
                self._latest_raw = parsed
            self._has_frame.set()

    def _parse_frame_packet_to_lists(self, msg: str, recv_time: float) -> dict[str, Any] | None:
        # parse frame id
        parts = msg.split(",")
        if len(parts) < 2 or parts[0] != "FRAME":
            return None

        try:
            frame_id = int(parts[1])
        except ValueError:
            return None

        # parse blocks
        hmd_pose = self._parse_pose_block(parts, "HMD")
        left_hand_pose = self._parse_pose_block(parts, "LEFTHAND")
        right_hand_pose = self._parse_pose_block(parts, "RIGHTHAND")
        waist_pose = self._parse_pose_block(parts, "WAIST")
        left_foot_pose = self._parse_pose_block(parts, "LEFTFOOT")
        right_foot_pose = self._parse_pose_block(parts, "RIGHTFOOT")

        if (
            hmd_pose is None
            or left_hand_pose is None
            or right_hand_pose is None
            or waist_pose is None
            or left_foot_pose is None
            or right_foot_pose is None
        ):
            return None

        link_pos = torch.stack(
            [
                left_foot_pose[:3],
                right_foot_pose[:3],
                left_hand_pose[:3],
                right_hand_pose[:3],
                hmd_pose[:3],
                waist_pose[:3],
            ],
            dim=0,
        )
        link_quat = torch.stack(
            [
                left_foot_pose[3:],
                right_foot_pose[3:],
                left_hand_pose[3:],
                right_hand_pose[3:],
                hmd_pose[3:],
                waist_pose[3:],
            ],
            dim=0,
        ).roll(1, dims=-1)  # (w, x, y, z)

        left_button = self._parse_button_block(parts, "LEFTBUTTON")
        right_button = self._parse_button_block(parts, "RIGHTBUTTON")
        button_states = (
            left_button if left_button is not None else 0,
            right_button if right_button is not None else 0,
        )

        return {
            "link_pos": link_pos,
            "link_quat": link_quat,
            "button_states": button_states,
            "frame_id": frame_id,
        }

    def _parse_pose_block(self, parts: list[str], label: str) -> torch.Tensor | None:
        try:
            i = parts.index(label)
        except ValueError:
            return None
        if i + 7 >= len(parts):
            return None
        try:
            px = float(parts[i + 1])
            py = float(parts[i + 2])
            pz = float(parts[i + 3])
            qx = float(parts[i + 4])
            qy = float(parts[i + 5])
            qz = float(parts[i + 6])
            qw = float(parts[i + 7])
        except ValueError:
            return None
        return torch.tensor([px, py, pz, qx, qy, qz, qw], dtype=torch.float32, device=self.device)

    def _parse_button_block(self, parts: list[str], label: str) -> int | None:
        try:
            i = parts.index(label)
        except ValueError:
            return None
        if i + 1 >= len(parts):
            return None
        try:
            b = int(parts[i + 1])
        except ValueError:
            return None
        return b
