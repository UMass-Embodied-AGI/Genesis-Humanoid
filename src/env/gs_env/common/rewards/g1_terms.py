import torch

from gs_env.common.utils.math_utils import quat_diff, quat_to_angle_axis

from .leggedrobot_terms import (
    ActionLimitPenalty,  # noqa
    ActionRatePenalty,  # noqa
    AngVelXYPenalty,  # noqa
    AngVelZReward,  # noqa
    BaseHeightPenalty,
    BodyAngVelXYPenalty,  # noqa
    DofPosLimitPenalty,  # noqa
    DofVelPenalty,  # noqa
    FeetAirTimePenalty,  # noqa
    FeetAirTimeReward,  # noqa
    FeetHeightPenalty,
    FeetSlidePenalty,
    FeetZVelocityPenalty,  # noqa
    LinVelXYReward,  # noqa
    LinVelZPenalty,  # noqa
    OrientationPenalty,  # noqa
    StandStillFeetContactPenalty,  # noqa
    StandStillReward,  # noqa
    TorquePenalty,  # noqa
)
from .reward_terms import RewardTerm


### ---- Reward Terms ---- ###
class G1BaseHeightPenalty(BaseHeightPenalty):
    target_height = 0.75


class UpperBodyActionPenalty(RewardTerm):
    """
    Penalize the upper body action position.

    Args:
        action: Action tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
    """

    required_keys = ("action",)

    def _compute(self, action: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(action[:, 15:]), dim=-1)


class MotionFeetAirTimePenalty(RewardTerm):
    """
    Penalize the feet air time.

    Args:
        feet_air_time: Feet air time tensor of shape (B, 2) where B is the batch size.
        feet_first_contact: Feet first contact tensor of shape (B, 2) where B is the batch size.
    """

    required_keys = ("feet_first_contact", "feet_air_time")
    target_feet_air_time = 0.4

    def _compute(
        self, feet_first_contact: torch.Tensor, feet_air_time: torch.Tensor
    ) -> torch.Tensor:  # type: ignore
        pen_air_time = torch.sum(
            torch.clamp(self.target_feet_air_time - feet_air_time, min=0.0) * feet_first_contact,
            dim=1,
        )
        return -pen_air_time


class MotionStandStillFeetContactPenalty(RewardTerm):
    """
    Penalize the feet contact when standing still.

    Args:
        foot_contact_weighted: Weighted feet contact force tensor of shape (B, N) where B is the batch size and N is the number of feet.
        ref_foot_contact: Reference foot contact tensor of shape (B, N) where B is the batch size and N is the number of feet.
    """

    required_keys = ("foot_contact_weighted", "ref_foot_contact")

    def _compute(
        self, foot_contact_weighted: torch.Tensor, ref_foot_contact: torch.Tensor
    ) -> torch.Tensor:  # type: ignore
        stand_still = torch.all(ref_foot_contact > 0.6, dim=-1)
        insufficient_contact = (0.25 - foot_contact_weighted).clamp(min=0.0).square().sum(dim=-1)
        return -insufficient_contact * stand_still


class MotionStandStillAnkleVelPenalty(RewardTerm):
    """
    Penalize the ankle joint velocity when standing still.

    Args:
        dof_vel: DoF velocity tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
    """

    required_keys = ("dof_vel", "ref_dof_vel", "foot_contact_weighted")

    def _compute(
        self, dof_vel: torch.Tensor, ref_dof_vel: torch.Tensor, foot_contact_weighted: torch.Tensor
    ) -> torch.Tensor:  # type: ignore
        stand_still = torch.all(foot_contact_weighted > 0.4, dim=-1)
        ankle_vel = dof_vel[:, [4, 5, 10, 11]]
        ref_ankle_vel = ref_dof_vel[:, [4, 5, 10, 11]]
        ankle_vel_error = (ankle_vel - ref_ankle_vel).abs().sum(dim=-1)
        return -ankle_vel_error * stand_still


class WaistRollPenalty(RewardTerm):
    """
    Penalize the waist DoF position.

    Args:
        dof_pos: DoF position tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
    """

    required_keys = ("dof_pos",)

    def _compute(self, dof_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.abs(dof_pos[:, 13])


class HipYawPenalty(RewardTerm):
    """
    Penalize the hip yaw DoF position.

    Args:
        dof_pos: DoF position tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
    """

    required_keys = ("dof_pos",)

    def _compute(self, dof_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.abs(dof_pos[:, [2, 8]]), dim=-1)


class HipRollPenalty(RewardTerm):
    """
    Penalize the hip roll DoF position.

    Args:
        dof_pos: DoF position tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
    """

    required_keys = ("dof_pos",)

    def _compute(self, dof_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.abs(dof_pos[:, [1, 7]]), dim=-1)


class WaistVelPenalty(RewardTerm):
    """
    Penalize the waist DoF velocity.

    Args:
        dof_vel: DoF velocity tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
    """

    required_keys = ("dof_vel",)

    def _compute(self, dof_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(dof_vel[:, [12, 13, 14]]), dim=-1)


class BodyRollPenalty(RewardTerm):
    """
    Penalize the body roll.

    Args:
        body_euler: Body Euler tensor of shape (B, 3) where B is the batch size.
    """

    required_keys = ("body_euler",)

    def _compute(self, body_euler: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.square(body_euler[:, 0])


class AnkleTorquePenalty(RewardTerm):
    """
    Penalize the ankle torque.

    Args:
        torque: Torque tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
    """

    required_keys = ("torque",)

    def _compute(self, torque: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(torque[:, [4, 5, 10, 11]]), dim=-1)


class G1FeetSlidePenalty(FeetSlidePenalty):
    feet_slide_height_threshold = 0.2


class G1FeetHeightPenalty(FeetHeightPenalty):
    target_height = 0.2


class FeetOrientationPenalty(RewardTerm):
    """
    Penalize the feet orientation.

    Args:
        feet_orientation: Feet orientation tensor of shape (B, 2, 3) where B is the batch size.
    """

    required_keys = ("feet_orientation",)

    def _compute(self, feet_orientation: torch.Tensor) -> torch.Tensor:  # type: ignore
        feet_orientation_deviation = feet_orientation[:, :, :2].square().sum(dim=-1)
        return -feet_orientation_deviation.sum(dim=-1)


class LinVelYPenalty(RewardTerm):
    """
    Penalize the linear velocity in the Y direction.

    Args:
        base_lin_vel: Linear velocity tensor of shape (B, 3) where B is the batch size.
    """

    required_keys = ("base_lin_vel",)

    def _compute(self, base_lin_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.square(base_lin_vel[:, 1])


class DofPosReward(RewardTerm):
    """
    Reward the DoF position.

    Args:
        dof_pos_error_weighted: DoF position tensor of shape (B,) where B is the batch size.
        deviation_buf: Deviation buffer tensor of shape (B,) where B is the batch size.
    """

    required_keys = ("dof_pos_error_weighted", "deviation_buf")

    def _compute(
        self, dof_pos_error_weighted: torch.Tensor, deviation_buf: torch.Tensor
    ) -> torch.Tensor:  # type: ignore
        # return torch.exp(-dof_pos_error_weighted * 0.15)
        return -dof_pos_error_weighted * deviation_buf


class DofVelReward(RewardTerm):
    """
    Reward the DoF velocity.

    Args:
        dof_vel_error_weighted: DoF velocity tensor of shape (B,) where B is the batch size .
        deviation_buf: Deviation buffer tensor of shape (B,) where B is the batch size.
    """

    required_keys = ("dof_vel_error_weighted", "deviation_buf")

    def _compute(
        self, dof_vel_error_weighted: torch.Tensor, deviation_buf: torch.Tensor
    ) -> torch.Tensor:  # type: ignore
        return -dof_vel_error_weighted * deviation_buf


class BaseHeightReward(RewardTerm):
    """
    Reward the base height.

    Args:
        base_height: Base height tensor of shape (B, 1) where B is the batch size.
        ref_base_height: Reference base height tensor of shape (B, 1) where B is the batch size.
    """

    required_keys = ("base_pos", "ref_base_pos")

    def _compute(self, base_pos: torch.Tensor, ref_base_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        base_height_error = torch.square(base_pos[:, 2] - ref_base_pos[:, 2])
        return -base_height_error


class BasePosReward(RewardTerm):
    """
    Reward the base position.

    Args:
        base_pos: Base position tensor of shape (B, 3) where B is the batch size.
        ref_base_pos: Reference base position tensor of shape (B, 3) where B is the batch size.
    """

    required_keys = ("base_pos", "ref_base_pos")

    def _compute(self, base_pos: torch.Tensor, ref_base_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        base_pos_error = torch.square(base_pos - ref_base_pos).sum(dim=-1)
        return -base_pos_error


class BaseQuatReward(RewardTerm):
    """
    Reward the base quaternion.

    Args:
        base_quat: Base quaternion tensor of shape (B, 4) where B is the batch size.
        ref_base_quat: Reference base quaternion tensor of shape (B, 4) where B is the batch size.
    """

    required_keys = ("base_quat", "ref_base_quat")

    def _compute(self, base_quat: torch.Tensor, ref_base_quat: torch.Tensor) -> torch.Tensor:  # type: ignore
        base_quat_error = quat_to_angle_axis(quat_diff(base_quat, ref_base_quat)).norm(dim=-1)
        return -(base_quat_error**2)


class BaseLinVelReward(RewardTerm):
    """
    Reward the base linear velocity.

    Args:
        base_lin_vel: Base linear velocity tensor of shape (B, 3) where B is the batch size.
        ref_base_lin_vel: Reference base linear velocity tensor of shape (B, 3) where B is the batch size.
    """

    required_keys = ("base_lin_vel", "ref_base_lin_vel")

    def _compute(self, base_lin_vel: torch.Tensor, ref_base_lin_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        base_lin_vel_error = torch.square(base_lin_vel - ref_base_lin_vel).sum(dim=-1)
        return -base_lin_vel_error


class BaseAngVelReward(RewardTerm):
    """
    Reward the base angular velocity.

    Args:
        base_ang_vel: Base angular velocity tensor of shape (B, 3) where B is the batch size.
        ref_base_ang_vel: Reference base angular velocity tensor of shape (B, 3) where B is the batch size.
    """

    required_keys = ("base_ang_vel", "ref_base_ang_vel")

    def _compute(self, base_ang_vel: torch.Tensor, ref_base_ang_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        base_ang_vel_error = torch.square(base_ang_vel - ref_base_ang_vel).sum(dim=-1)
        return -base_ang_vel_error


class TrackingLinkPosGlobalReward(RewardTerm):
    """
    Reward the tracking link global position.

    Args:
        tracking_link_pos_global: Tracking link position tensor of shape (B, N, 3) where B is the batch size and N is the number of tracking links.
        ref_tracking_link_pos_global: Reference tracking link position tensor of shape (B, N, 3) where B is the batch size and N is the number of tracking links.
        tracking_link_pos_global_weights: Tracking link global position weights tensor of shape (N,) where N is the number of tracking links.
    """

    required_keys = (
        "tracking_link_pos_global",
        "ref_tracking_link_pos_global",
        "tracking_link_pos_global_weights",
    )

    def _compute(
        self,
        tracking_link_pos_global: torch.Tensor,
        ref_tracking_link_pos_global: torch.Tensor,
        tracking_link_pos_global_weights: torch.Tensor,
    ) -> torch.Tensor:  # type: ignore
        tracking_link_pos_error = (
            torch.square(tracking_link_pos_global - ref_tracking_link_pos_global).sum(dim=-1)
            * tracking_link_pos_global_weights[None, :]
        ).sum(dim=-1)
        return -tracking_link_pos_error


class TrackingLinkPosLocalReward(RewardTerm):
    """
    Reward the tracking link position.

    Args:
        tracking_link_pos_local_yaw: Tracking link position tensor of shape (B, N, 3) where B is the batch size and N is the number of tracking links.
        ref_tracking_link_pos_local_yaw: Reference tracking link position tensor of shape (B, N, 3) where B is the batch size and N is the number of tracking links.
        tracking_link_pos_local_weights: Tracking link local position weights tensor of shape (N,) where N is the number of tracking links.
        deviation_buf: Deviation buffer tensor of shape (B,) where B is the batch size.
    """

    required_keys = (
        "tracking_link_pos_local_yaw",
        "ref_tracking_link_pos_local_yaw",
        "tracking_link_pos_local_weights",
        "deviation_buf",
    )

    def _compute(
        self,
        tracking_link_pos_local_yaw: torch.Tensor,
        ref_tracking_link_pos_local_yaw: torch.Tensor,
        tracking_link_pos_local_weights: torch.Tensor,
        deviation_buf: torch.Tensor,
    ) -> torch.Tensor:  # type: ignore
        tracking_link_pos_error = (
            torch.square(tracking_link_pos_local_yaw - ref_tracking_link_pos_local_yaw).sum(dim=-1)
            * tracking_link_pos_local_weights[None, :]
        ).sum(dim=-1)
        return -tracking_link_pos_error * deviation_buf


class TrackingLinkQuatGlobalReward(RewardTerm):
    """
    Reward the tracking link quaternion.

    Args:
        tracking_link_quat_global: Tracking link quaternion tensor of shape (B, N, 4) where B is the batch size and N is the number of tracking links.
        ref_tracking_link_quat_global: Reference tracking link quaternion tensor of shape (B, N, 4) where B is the batch size and N is the number of tracking links.
        tracking_link_quat_weights: Tracking link quaternion weights tensor of shape (N,) where N is the number of tracking links.
        deviation_buf: Deviation buffer tensor of shape (B,) where B is the batch size.
    """

    required_keys = (
        "tracking_link_quat_global",
        "ref_tracking_link_quat_global",
        "tracking_link_quat_weights",
        "deviation_buf",
    )

    def _compute(
        self,
        tracking_link_quat_global: torch.Tensor,
        ref_tracking_link_quat_global: torch.Tensor,
        tracking_link_quat_weights: torch.Tensor,
        deviation_buf: torch.Tensor,
    ) -> torch.Tensor:  # type: ignore
        tracking_link_quat_error = (
            quat_to_angle_axis(
                quat_diff(tracking_link_quat_global, ref_tracking_link_quat_global)
            ).norm(dim=-1)
            * tracking_link_quat_weights[None, :]
        ).sum(dim=-1)
        return -tracking_link_quat_error * deviation_buf


class TrackingLinkQuatReward(RewardTerm):
    """
    Reward the tracking link quaternion.

    Args:
        tracking_link_quat_local_yaw: Tracking link quaternion tensor of shape (B, N, 4) where B is the batch size and N is the number of tracking links.
        ref_tracking_link_quat_local_yaw: Reference tracking link quaternion tensor of shape (B, N, 4) where B is the batch size and N is the number of tracking links.
        tracking_link_quat_weights: Tracking link quaternion weights tensor of shape (N,) where N is the number of tracking links.
    """

    required_keys = (
        "tracking_link_quat_local_yaw",
        "ref_tracking_link_quat_local_yaw",
        "tracking_link_quat_weights",
        "deviation_buf",
    )

    def _compute(
        self,
        tracking_link_quat_local_yaw: torch.Tensor,
        ref_tracking_link_quat_local_yaw: torch.Tensor,
        tracking_link_quat_weights: torch.Tensor,
        deviation_buf: torch.Tensor,
    ) -> torch.Tensor:  # type: ignore
        tracking_link_quat_error = (
            quat_to_angle_axis(
                quat_diff(tracking_link_quat_local_yaw, ref_tracking_link_quat_local_yaw)
            ).norm(dim=-1)
            * tracking_link_quat_weights[None, :]
        ).sum(dim=-1)
        return -tracking_link_quat_error * deviation_buf


class TrackingLinkLinVelReward(RewardTerm):
    """
    Reward the tracking link linear velocity.
    """

    required_keys = (
        "tracking_link_lin_vel_global",
        "ref_tracking_link_lin_vel_global",
        "deviation_buf",
    )

    def _compute(
        self,
        tracking_link_lin_vel_global: torch.Tensor,
        ref_tracking_link_lin_vel_global: torch.Tensor,
        deviation_buf: torch.Tensor,
    ) -> torch.Tensor:  # type: ignore
        tracking_link_lin_vel_error = torch.square(
            tracking_link_lin_vel_global - ref_tracking_link_lin_vel_global
        ).sum(dim=[-1, -2])
        return -tracking_link_lin_vel_error * deviation_buf


class TrackingLinkAngVelReward(RewardTerm):
    """
    Reward the tracking link angular velocity.
    """

    required_keys = (
        "tracking_link_ang_vel_global",
        "ref_tracking_link_ang_vel_global",
        "deviation_buf",
    )

    def _compute(
        self,
        tracking_link_ang_vel_global: torch.Tensor,
        ref_tracking_link_ang_vel_global: torch.Tensor,
        deviation_buf: torch.Tensor,
    ) -> torch.Tensor:  # type: ignore
        tracking_link_ang_vel_error = torch.square(
            tracking_link_ang_vel_global - ref_tracking_link_ang_vel_global
        ).sum(dim=[-1, -2])
        return -tracking_link_ang_vel_error * deviation_buf


class FootContactReward(RewardTerm):
    """
    Reward the foot contact.

    Args:
        foot_contact_weighted: Foot contact weighted tensor of shape (B, N) where B is the batch size and N is the number of feet.
        ref_foot_contact_weighted: Reference foot contact weighted tensor of shape (B, N) where B is the batch size and N is the number of feet.
    """

    required_keys = ("foot_contact_weighted", "ref_foot_contact_weighted")

    def _compute(
        self,
        foot_contact_weighted: torch.Tensor,
        ref_foot_contact_weighted: torch.Tensor,
    ) -> torch.Tensor:  # type: ignore
        # reward
        foot_contact_weighted_error = (ref_foot_contact_weighted - foot_contact_weighted).abs()
        return -torch.clamp(foot_contact_weighted_error - 0.2, min=0.0).square().sum(dim=-1)


class FootContactPenalty(RewardTerm):
    """
    Penalize the foot contact.

    Args:
        foot_contact_weighted: Weighted foot contact force tensor of shape (B, N) where B is the batch size and N is the number of feet.
        ref_foot_contact: Reference foot contact tensor of shape (B, N) where B is the batch size and N is the number of feet.
    """

    required_keys = ("foot_contact_weighted", "ref_foot_contact")

    def _compute(
        self, foot_contact_weighted: torch.Tensor, ref_foot_contact: torch.Tensor
    ) -> torch.Tensor:  # type: ignore
        contact_force = foot_contact_weighted * (ref_foot_contact < 0.2)
        return -torch.square(contact_force).sum(dim=-1)


class FeetContactForceLimitPenalty(RewardTerm):
    """
    Penalize the feet contact force limit.

    Args:
        foot_contact_weighted: Weighted foot contact force tensor of shape (B, N) where B is the batch size and N is the number of feet.
    """

    required_keys = ("foot_contact_weighted",)

    def _compute(self, foot_contact_weighted: torch.Tensor) -> torch.Tensor:  # type: ignore
        contact_above_limit = (foot_contact_weighted - 1.0).clamp(min=0.0)
        return -torch.square(contact_above_limit).sum(dim=-1)
