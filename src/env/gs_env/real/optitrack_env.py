import threading
from typing import Any

import numpy as np
import torch
import yaml

from gs_env.common.bases.base_env import BaseEnv
from gs_env.common.utils.math_utils import np_pose_mul
from gs_env.real.config.schema import OptitrackEnvArgs

from .optitrack.NatNetClient import setup_optitrack
from .optitrack.optitrack_config import RIGID_BODY_ID_MAP

_DEFAULT_DEVICE = torch.device("cpu")


class OptitrackEnv(BaseEnv):
    def __init__(
        self,
        num_envs: int,
        args: OptitrackEnvArgs,
        device: torch.device = _DEFAULT_DEVICE,
    ) -> None:
        super().__init__(device=device)

        if num_envs != 1:
            raise ValueError("OptitrackEnv only supports num_envs=1")

        self._num_envs = num_envs
        self._args = args
        self._device = device

        self._client = setup_optitrack(
            server_address=self._args.server_ip,
            client_address=self._args.client_ip,
            use_multicast=self._args.use_multicast,
        )
        thread = threading.Thread(target=self._client.run)
        thread.start()
        if not self._client:
            print("Failed to setup OptiTrack client")
            exit(1)

        self.rigid_body_id_map: dict[int, str] = RIGID_BODY_ID_MAP
        self.robot_link_offsets = {}
        with open(self._args.offset_config) as f:
            off = yaml.safe_load(f)
        for name in self._args.tracked_link_names:
            data = off.get(name, {})
            self.robot_link_offsets[name] = {
                "pos": np.array(data.get("pos", [0.0, 0.0, 0.0]), dtype=np.float32),
                "quat": np.array(data.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32),
            }

    def __del__(self) -> None:
        self._client.shutdown()

    def _calculate_tracked_link_by_name(
        self, name: str, pos: np.typing.NDArray[np.float32], quat: np.typing.NDArray[np.float32]
    ) -> tuple[np.typing.NDArray[np.float32], np.typing.NDArray[np.float32]]:
        """
        Calculate a single tracked link by name.
        """
        if name not in self.robot_link_offsets:
            raise ValueError(f"Tracked link {name} not found!")
        aligned_quat, aligned_pos = np_pose_mul(
            pos,
            quat,
            self.robot_link_offsets[name]["pos"],
            self.robot_link_offsets[name]["quat"],
        )
        return aligned_pos, aligned_quat

    def get_tracked_links(
        self,
    ) -> dict[str, tuple[np.typing.NDArray[np.float32], np.typing.NDArray[np.float32]]]:
        """
        Get all tracked links. Will always get the latest frame.
        """
        aligned_poses = {}
        frame = self._client.get_frame()
        for rb_id, (pos, quat) in frame.items():
            if rb_id not in self.rigid_body_id_map:
                raise ValueError(f"Unmapped RB ID {rb_id}!! Please check RIGID_BODY_ID_MAP.")
            name = self.rigid_body_id_map[rb_id]
            if name in self._args.tracked_link_names:
                pos_array = np.array(pos, dtype=np.float32)
                quat_array = np.roll(np.array(quat, dtype=np.float32), 1)
                new_pos, new_quat = self._calculate_tracked_link_by_name(
                    name, pos_array, quat_array
                )
                aligned_poses[name] = (new_pos, new_quat)
        return aligned_poses

    # all the abstract methods below should not be used in real envs
    def reset_idx(self, envs_idx: torch.IntTensor) -> None:
        pass

    def apply_action(self, action: torch.Tensor) -> None:
        pass

    def get_observations(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros((self._num_envs, 0), device=self._device), torch.zeros(
            (self._num_envs, 0), device=self._device
        )

    def get_extra_infos(self) -> dict[str, Any]:
        return {}

    def get_terminated(self) -> torch.Tensor:
        return torch.zeros((self._num_envs,), device=self._device)

    def get_truncated(self) -> torch.Tensor:
        return torch.zeros((self._num_envs,), device=self._device)

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return torch.zeros((self._num_envs,), device=self._device), {}

    @property
    def num_envs(self) -> int:
        return self._num_envs
