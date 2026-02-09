from abc import ABC, abstractmethod

import torch

from gs_env.common.utils.math_utils import quat_apply


class RewardTerm(ABC):
    """
    A reward term that *declares* the names of the tensors it needs.
    Nothing is inferred – you can read `.required_keys` and know exactly
    what the caller must supply.

    Return:
        A tensor of shape (B,) where B is the batch size.
    """

    #: Ordered list of keys the term expects to find in the `state` dict.
    required_keys: tuple[str, ...] = ()

    def __init__(self, scale: float = 1.0, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__
        self.scale = float(scale)

    def __call__(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        tensors = [state[k] for k in self.required_keys]
        return self.scale * self._compute(*tensors)

    @abstractmethod
    def _compute(self, *tensors: torch.Tensor) -> torch.Tensor:
        """Actual reward computation.  Signature is fixed by `required_keys`."""


### ---- Reward Terms ---- ###
class ActionL2Penalty(RewardTerm):
    """
    Penalize the action by its squared L2 norm.

    Args:
        action: Action tensor of shape (B, D) where B is the batch size and D is the action dimension.
    """

    required_keys = ("action",)

    def _compute(self, action: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(action**2, dim=-1)


class PositionL2Penalty(RewardTerm):
    """
    Penalize the squared L2 distance between two positions.

    Args:
        pos_a: Current position tensor of shape (B, 3).
        pos_b: Goal position tensor of shape (B, 3).
    """

    required_keys = ("pos_a", "pos_b")

    def _compute(self, pos_a: torch.Tensor, pos_b: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.norm(pos_a - pos_b, p=2, dim=-1)


class KeypointsAlign(RewardTerm):
    """
    Penalize the squared L2 distance between two sets of keypoints.

    Args:
        pose_a: Current pose tensor of shape (B, 7) where B is the batch size.
        pose_b: Goal pose tensor of shape (B, 7).
        key_offsets: Keypoints offsets tensor of shape (B, K, 3) where K is the number of keypoints. (local coordinates)
    """

    required_keys = ("pose_a", "pose_b", "key_offsets")

    def _compute(  # type: ignore
        self,
        pose_a: torch.Tensor,  # (B, 7)
        pose_b: torch.Tensor,  # (B, 7)
        key_offsets: torch.Tensor,  # (B, K, 3)
    ) -> torch.Tensor:
        b_pos, b_quat = pose_a[:, :3], pose_a[:, 3:]
        g_pos, g_quat = pose_b[:, :3], pose_b[:, 3:]

        cur_kp = self._to_world(b_pos, b_quat, key_offsets)
        goal_kp = self._to_world(g_pos, g_quat, key_offsets)

        dist = torch.norm(cur_kp - goal_kp, p=2, dim=-1).sum(-1)
        return torch.exp(-dist)

    @staticmethod
    def _to_world(pos: torch.Tensor, quat: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        world = torch.zeros_like(offsets)
        for k in range(offsets.size(1)):
            world[:, k] = pos + quat_apply(quat, offsets[:, k])
        return world


class PoseDist(RewardTerm):
    """
    Penalize the squared L2 distance between two poses.

    Args:
        pose_a: Current pose tensor of shape (B, 7) where B is the batch size.
        pose_b: Goal pose tensor of shape (B, 7).

        The pose is represented as (x, y, z, qw, qx, qy, qz).
    """

    required_keys = ("pose_a", "pose_b")

    def _compute(self, pose_a: torch.Tensor, pose_b: torch.Tensor) -> torch.Tensor:  # type: ignore
        ...

    @staticmethod
    def _quat_err(quat_a: torch.Tensor, quat_b: torch.Tensor) -> torch.Tensor:
        """
        Compute the quaternion error between two quaternions.
        """
        ...
