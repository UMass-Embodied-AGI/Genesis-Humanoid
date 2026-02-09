import torch

#
from gs_env.common.utils.math_utils import (
    quat_apply,
    quat_inv,
    quat_to_euler,
)
from gs_env.sim.envs.config.schema import WalkingEnvArgs
from gs_env.sim.envs.locomotion.leggedrobot_env import LeggedRobotEnv

_DEFAULT_DEVICE = torch.device("cpu")


class WalkingEnv(LeggedRobotEnv):
    """
    Walking Environment for Legged Robots.
    """

    def __init__(
        self,
        args: WalkingEnvArgs,
        num_envs: int,
        show_viewer: bool = False,
        device: torch.device = _DEFAULT_DEVICE,
        eval_mode: bool = False,
        debug: bool = False,
    ) -> None:
        # Initialize base legged-robot environment
        self._args = args
        super().__init__(
            args=args,
            num_envs=num_envs,
            show_viewer=show_viewer,
            device=device,
            eval_mode=eval_mode,
            debug=debug,
        )

    def _init(self) -> None:
        # Pre-parent: allocate feet-related buffers required by observation terms
        self.body_euler = torch.zeros(self.num_envs, 3, device=self._device)
        self.feet_position = torch.zeros(
            (self.num_envs, len(self._robot.foot_links_idx), 3),
            device=self._device,
            dtype=torch.float32,
        )
        self.feet_height = torch.zeros(
            (self.num_envs, len(self._robot.foot_links_idx)),
            device=self._device,
            dtype=torch.float32,
        )
        self.feet_orientation = torch.zeros(
            (self.num_envs, len(self._robot.foot_links_idx), 3),
            device=self._device,
            dtype=torch.float32,
        )
        self.feet_velocity = torch.zeros(
            (self.num_envs, len(self._robot.foot_links_idx), 3),
            device=self._device,
            dtype=torch.float32,
        )
        self.foot_contact = torch.zeros(
            (self.num_envs, len(self._robot.foot_links_idx)),
            device=self._device,
            dtype=torch.float32,
        )
        self.foot_contact_force = torch.zeros(
            (self.num_envs, len(self._robot.foot_links_idx)),
            device=self._device,
            dtype=torch.float32,
        )
        self.foot_contact_weighted = torch.zeros(
            (self.num_envs, len(self._robot.foot_links_idx)),
            device=self._device,
            dtype=torch.float32,
        )
        self.feet_first_contact = torch.zeros(
            (self.num_envs, len(self._robot.foot_links_idx)),
            device=self._device,
            dtype=torch.float32,
        )
        self.feet_air_time = torch.zeros(
            (self.num_envs, len(self._robot.foot_links_idx)),
            device=self._device,
            dtype=torch.float32,
        )

        # Additional buffers for walking environment
        self.commands_range = self._args.commands_range
        self.commands = torch.zeros(
            (self.num_envs, len(self.commands_range)), device=self._device, dtype=torch.float32
        )

        # Let base class set up common buffers, spaces, and rendering
        super()._init()

        # Additional timers specific to this environment
        self._command_resample_time = self._args.command_resample_time  # seconds
        self.time_since_resample = torch.zeros(self.num_envs, device=self._device)

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        super().reset_idx(envs_idx=envs_idx)
        self.feet_air_time[envs_idx] = 0.0
        self._resample_commands(envs_idx=envs_idx)
        self.time_since_resample[envs_idx] = 0.0

    def apply_action(self, action: torch.Tensor) -> None:
        super().apply_action(action=action)

        self.feet_first_contact[:] = (self.feet_air_time > 0.0) * self.foot_contact
        self.feet_air_time += self.dt

    def _pre_step(self) -> None:
        super()._pre_step()
        self.time_since_resample += self._scene.scene.dt

    def update_history(self) -> None:
        super().update_history()
        self.feet_air_time *= 1 - self.foot_contact

        resample_env_ids = torch.nonzero(
            self.time_since_resample > self._command_resample_time, as_tuple=False
        ).squeeze(-1)
        self._resample_commands(envs_idx=resample_env_ids)
        self.time_since_resample[resample_env_ids] = 0.0

    def update_buffers(self) -> None:
        super().update_buffers()
        self.foot_contact_force[:] = self.link_contact_forces[:, self._robot.foot_links_idx, 2]
        self.foot_contact[:] = self.foot_contact_force > 1.0
        self.foot_contact_weighted[:] = self.foot_contact_force / (
            self.robot.mass * self.scene.gravity
        )
        self.feet_position[:] = self.link_positions[:, self._robot.foot_links_idx]
        self.feet_height[:] = self.feet_position[:, :, 2]
        self.feet_velocity[:] = self._robot.link_lin_velocities[:, self._robot.foot_links_idx]
        feet_quaternions = self.link_quaternions[:, self._robot.foot_links_idx].reshape(-1, 4)
        self.feet_orientation[:] = quat_apply(
            quat_inv(feet_quaternions), self.global_gravity.repeat(2, 1)
        ).reshape(self.num_envs, len(self._robot.foot_links_idx), 3)
        body_quat = self._robot.link_quaternions[:, self._robot.body_link_idx]
        self.body_euler[:] = quat_to_euler(body_quat)

    def _resample_commands(self, envs_idx: torch.Tensor) -> None:
        for i in range(len(self.commands_range)):
            self.commands[envs_idx, i] = (
                torch.rand(len(envs_idx), device=self._device)
                * (self.commands_range[i][1] - self.commands_range[i][0])
                + self.commands_range[i][0]
            )
        self.commands[envs_idx, :2] *= (
            torch.norm(self.commands[envs_idx, :2], dim=-1) > self._args.command_lin_vel_clip
        )[:, None]
        self.commands[envs_idx, 2] *= (
            torch.abs(self.commands[envs_idx, 2]) > self._args.command_ang_vel_clip
        )
        if_go_straight = (
            torch.rand((len(envs_idx)), device=self.device) < self._args.go_straight_ratio
        )
        if_go_nowhere = (
            torch.rand((len(envs_idx), 1), device=self.device) < self._args.go_nowhere_ratio
        )
        self.commands[envs_idx, 2] *= ~if_go_straight
        self.commands[envs_idx, :2] *= ~if_go_nowhere
