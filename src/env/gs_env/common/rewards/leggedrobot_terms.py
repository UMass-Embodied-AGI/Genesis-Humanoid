import torch

from .reward_terms import RewardTerm


### ---- Reward Terms ---- ###
class LinVelXYReward(RewardTerm):
    """
    Reward the linear velocity in the X and Y directions.

    Args:
        base_lin_vel: Linear velocity tensor of shape (B, 3) where B is the batch size.
    """

    required_keys = ("base_lin_vel", "commands")

    def _compute(self, base_lin_vel: torch.Tensor, commands: torch.Tensor) -> torch.Tensor:  # type: ignore
        return torch.exp(
            -torch.sum(torch.square(base_lin_vel[:, :2] - commands[:, :2]) / 0.25, dim=-1)
        )


class AngVelZReward(RewardTerm):
    """
    Reward the angular velocity in the Z direction.

    Args:
        base_ang_vel: Angular velocity tensor of shape (B, 3) where B is the batch size.
        command_ang_vel: Commanded angular velocity tensor of shape (B,) representing desired yaw rate.
    """

    required_keys = ("base_ang_vel", "commands")

    def _compute(self, base_ang_vel: torch.Tensor, commands: torch.Tensor) -> torch.Tensor:  # type: ignore
        return torch.exp(-torch.square(base_ang_vel[:, 2] - commands[:, 2]) / 0.25)


class LinVelZPenalty(RewardTerm):
    """
    Penalize the linear velocity in the Z direction.

    Args:
        base_lin_vel: Linear velocity tensor of shape (B, 3) where B is the batch size.
    """

    required_keys = ("base_lin_vel",)

    def _compute(self, base_lin_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.square(base_lin_vel[:, 2])


class AngVelXYPenalty(RewardTerm):
    """
    Penalize the angular velocity in the X and Y directions.

    Args:
        base_ang_vel: Angular velocity tensor of shape (B, 3) where B is the batch size.
    """

    required_keys = ("base_ang_vel",)

    def _compute(self, base_ang_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(base_ang_vel[:, :2]), dim=-1)


class BodyLinVelZPenalty(RewardTerm):
    """
    Penalize the linear velocity in the Z direction.

    Args:
        body_lin_vel: Linear velocity tensor of shape (B, 3) where B is the batch size.
    """

    required_keys = ("body_lin_vel",)

    def _compute(self, body_lin_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.square(body_lin_vel[:, 2])


class BodyAngVelXYPenalty(RewardTerm):
    """
    Penalize the angular velocity in the X and Y directions.

    Args:
        body_ang_vel: Angular velocity tensor of shape (B, 3) where B is the batch size.
    """

    required_keys = ("body_ang_vel",)

    def _compute(self, body_ang_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(body_ang_vel[:, :2]), dim=-1)


class OrientationPenalty(RewardTerm):
    """
    Penalize the orientation deviation from upright.

    Args:
        root_quat: Root orientation quaternion tensor of shape (B, 4) where B is the batch size.
    """

    required_keys = ("projected_gravity",)

    def _compute(self, projected_gravity: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(projected_gravity[:, :2]), dim=-1)


class BaseHeightPenalty(RewardTerm):
    """
    Penalize the deviation of the base height from a target height.

    Args:
        base_height: Base height tensor of shape (B,) where B is the batch size.
    """

    required_keys = ("base_pos",)
    target_height: float = 1.0

    def _compute(self, base_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.square(base_pos[:, 2] - self.target_height)


class ActionRatePenalty(RewardTerm):
    """
    Penalize the action rate by its squared L2 norm.

    Args:
        action: Action tensor of shape (B, D) where B is the batch size and D is the action dimension.
        last_action: Last action tensor of shape (B, D).
    """

    required_keys = ("action", "last_action")

    def _compute(self, action: torch.Tensor, last_action: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum((action - last_action) ** 2, dim=-1)


class StandStillActionRatePenalty(RewardTerm):
    """
    Penalize the action rate by its squared L2 norm.

    Args:
        action: Action tensor of shape (B, D) where B is the batch size and D is the action dimension.
        last_action: Last action tensor of shape (B, D).
        commands: Commands tensor of shape (B, 3) where B is the batch size.
    """

    required_keys = ("action", "last_action", "commands")

    def _compute(
        self, action: torch.Tensor, last_action: torch.Tensor, commands: torch.Tensor
    ) -> torch.Tensor:  # type: ignore
        return -torch.sum((action - last_action) ** 2, dim=-1) * (torch.norm(commands, dim=1) < 0.1)


class TorquePenalty(RewardTerm):
    """
    Penalize the torque by its squared L2 norm.

    Args:
        joint_torques: Joint torque tensor of shape (B, D) where B is the batch size and D is the number of joints.
    """

    required_keys = ("torque",)

    def _compute(self, torque: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torque**2, dim=-1)


class DofPosLimitPenalty(RewardTerm):
    """
    Penalize the degree of freedom (DoF) position limit violations.

    Args:
        dof_pos: DoF position tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
        dof_pos_limits: DoF position limits tensor of shape (D, 2) where each row contains [min, max] limits for a DoF.
    """

    required_keys = ("dof_pos", "dof_pos_limits")

    def _compute(self, dof_pos: torch.Tensor, dof_pos_limits: torch.Tensor) -> torch.Tensor:  # type: ignore
        out_of_limits = -(dof_pos - dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
        out_of_limits += (dof_pos - dof_pos_limits[:, 1]).clip(min=0.0)  # upper limit
        return -torch.sum(out_of_limits, dim=1)


class ActionLimitPenalty(RewardTerm):
    """
    Penalize the action limit violations.

    Args:
        action: Action tensor of shape (B, D) where B is the batch size and D is the action dimension.
        action_limits: Action limits tensor of shape (D, 2) where each row contains [min, max] limits for an action dimension.
    """

    required_keys = ("action",)

    def _compute(self, action: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(torch.abs(action).clip(min=12) - 12), dim=1)


class FeetAirTimeReward(RewardTerm):
    """
    Reward the feet air time.

    Args:
        feet_air_time: Feet air time tensor of shape (B, 2) where B is the batch size.
        feet_first_contact: Feet first contact tensor of shape (B, 2) where B is the batch size.
        commands: Commands tensor of shape (B, 3) where B is the batch size.
    """

    required_keys = ("feet_first_contact", "feet_air_time", "commands")

    def _compute(
        self, feet_first_contact: torch.Tensor, feet_air_time: torch.Tensor, commands: torch.Tensor
    ) -> torch.Tensor:  # type: ignore
        rew_air_time = torch.sum(feet_air_time * feet_first_contact, dim=1).clip(max=0.5)
        rew_air_time *= torch.norm(commands, dim=1) > 0.1
        return rew_air_time


class FeetAirTimePenalty(RewardTerm):
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
            torch.abs(feet_air_time - self.target_feet_air_time) * feet_first_contact, dim=1
        )
        return -pen_air_time


class FeetHeightPenalty(RewardTerm):
    """
    Penalize the feet height.

    Args:
        feet_height: Feet height tensor of shape (B, N) where B is the batch size and N is the number of feet.
        commands: Commands tensor of shape (B, 3) where B is the batch size.
    """

    required_keys = ("feet_height", "commands")
    target_height: float = 0.0

    def _compute(self, feet_height: torch.Tensor, commands: torch.Tensor) -> torch.Tensor:  # type: ignore
        feet_height = feet_height.max(dim=1)[0] - self.target_height
        feet_height = feet_height.clip(max=0.0)
        feet_height *= torch.norm(commands, dim=1) > 0.1
        return feet_height


class FeetZVelocityPenalty(RewardTerm):
    """
    Penalize the feet vertical velocity.

    Args:
        feet_velocity: Feet velocity tensor of shape (B, N, 3) where B is the batch size and N is the number of feet.
    """

    required_keys = ("feet_velocity",)

    def _compute(self, feet_velocity: torch.Tensor) -> torch.Tensor:  # type: ignore
        feet_z_velocity = feet_velocity[:, :, 2]
        feet_z_velocity = torch.square(feet_z_velocity).sum(dim=1)
        return -feet_z_velocity


class StandStillFeetContactPenalty(RewardTerm):
    """
    Penalize the uneven feet contact force when stand still.

    Args:
        foot_contact_weighted: Weighted feet contact force tensor of shape (B, N) where B is the batch size and N is the number of feet.
        commands: Commands tensor of shape (B, 3) where B is the batch size.
    """

    required_keys = ("foot_contact_weighted", "commands")

    def _compute(self, foot_contact_weighted: torch.Tensor, commands: torch.Tensor) -> torch.Tensor:  # type: ignore
        contact_force_diff = foot_contact_weighted - foot_contact_weighted.mean(dim=1, keepdim=True)
        contact_force_diff = torch.square(contact_force_diff).sum(dim=1)
        contact_force_diff *= torch.norm(commands, dim=1) < 0.1
        return -contact_force_diff


class DofVelPenalty(RewardTerm):
    """
    Penalize the dof velocities.

    Args:
        dof_vel: dof_vel tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
    """

    required_keys = ("dof_vel",)

    def _compute(self, dof_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(dof_vel), dim=-1)


class StandStillReward(RewardTerm):
    """
    Reward standing still by low joint torques.

    Args:
        default_dof_pos: default_dof_pos tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
        dof_pos: dof_pos tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
        commands: Commands tensor of shape (B, 3) where B is the batch size.
    """

    required_keys = ("default_dof_pos", "dof_pos", "commands")

    def _compute(
        self, default_dof_pos: torch.Tensor, dof_pos: torch.Tensor, commands: torch.Tensor
    ) -> torch.Tensor:  # type: ignore
        dof_error = torch.norm(dof_pos - default_dof_pos, dim=1)
        rew = torch.exp(-dof_error * 2)
        rew[commands.norm(dim=1) > 0.1] = 0.0
        return rew


class FeetSlidePenalty(RewardTerm):
    """
    Penalize the feet slide.

    Args:
        feet_height: Feet height tensor of shape (B, 2) where B is the batch size.
        foot_contact: Feet contact tensor of shape (B, 2) where B is the batch size.
        feet_velocity: Feet velocity tensor of shape (B, 2, 3) where B is the batch size.
    """

    required_keys = ("feet_height", "foot_contact", "feet_velocity")
    feet_slide_height_threshold = 0.1

    def _compute(
        self, feet_height: torch.Tensor, foot_contact: torch.Tensor, feet_velocity: torch.Tensor
    ) -> torch.Tensor:  # type: ignore
        foot_contact_mask = foot_contact + (feet_height < self.feet_slide_height_threshold).float()
        feet_vel_xy = torch.square(feet_velocity[:, :, :2]).sum(dim=-1)
        return -torch.sum(feet_vel_xy * foot_contact_mask, dim=-1)
