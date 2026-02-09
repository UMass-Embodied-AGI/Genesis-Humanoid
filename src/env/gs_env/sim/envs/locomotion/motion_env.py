from typing import Any

import torch

from gs_env.common.utils.math_utils import (
    quat_apply,
    quat_diff,
    quat_error_magnitude,
    quat_from_angle_axis,
    quat_from_euler,
    quat_inv,
    quat_mul,
    quat_to_euler,
    quat_to_rotation_6D,
)

#
from gs_env.common.utils.motion_utils import MotionLib, build_motion_obs_from_dict
from gs_env.sim.envs.config.schema import MotionEnvArgs
from gs_env.sim.envs.locomotion.leggedrobot_env import LeggedRobotEnv

_DEFAULT_DEVICE = torch.device("cpu")


class MotionEnv(LeggedRobotEnv):
    """
    Motion imitation environment using reference motions.

    Exposes the following additional tensors so they can be used in
    observations and rewards via `actor_obs_terms` / `critic_obs_terms`:
      - ref_base_pos: (B, 3)
      - ref_base_quat: (B, 4)
      - ref_base_lin_vel: (B, 3)
      - ref_base_ang_vel: (B, 3)
      - ref_dof_pos: (B, D)
      - ref_dof_vel: (B, D)
      - ref_body_pos: (B, L, 3) global positions of key bodies in the ref
      - mimic_obs: (B, M) compact imitation observation (current frame)
    """

    def __init__(
        self,
        args: MotionEnvArgs,
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
        # Pre-parent: allocate feet-related buffers optionally used by rewards
        self.feet_height = torch.zeros(
            (self.num_envs, len(self._robot.foot_links_idx)),
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
        self.foot_contact_weighted = torch.zeros(
            (self.num_envs, len(self._robot.foot_links_idx)),
            device=self._device,
            dtype=torch.float32,
        )
        self.base_height = torch.zeros(self.num_envs, device=self._device, dtype=torch.float32)
        self.base_rotation_6D = torch.zeros(
            self.num_envs, 6, device=self._device, dtype=torch.float32
        )
        self.base_lin_vel_local = torch.zeros(self.num_envs, 3, device=self._device)
        self.base_ang_vel_local = torch.zeros(self.num_envs, 3, device=self._device)
        self.link_pos_local_yaw = torch.zeros(
            self.num_envs, self._robot.n_links, 3, device=self._device
        )
        self.link_quat_local_yaw = torch.zeros(
            self.num_envs, self._robot.n_links, 4, device=self._device
        )
        self.body_lin_vel = torch.zeros(self.num_envs, 3, device=self._device)
        self.body_ang_vel = torch.zeros(self.num_envs, 3, device=self._device)
        self.body_euler = torch.zeros(self.num_envs, 3, device=self._device)

        # reference trajectories (current frame)
        self.ref_base_pos = torch.zeros(self.num_envs, 3, device=self._device)
        self.ref_base_quat = torch.zeros(self.num_envs, 4, device=self._device)
        self.ref_base_height = torch.zeros(self.num_envs, device=self._device, dtype=torch.float32)
        self.ref_base_rotation_6D = torch.zeros(
            self.num_envs, 6, device=self._device, dtype=torch.float32
        )
        self.ref_base_euler = torch.zeros(
            self.num_envs, 3, device=self._device, dtype=torch.float32
        )
        self.ref_base_lin_vel = torch.zeros(self.num_envs, 3, device=self._device)
        self.ref_base_ang_vel = torch.zeros(self.num_envs, 3, device=self._device)
        self.ref_base_lin_vel_local = torch.zeros(self.num_envs, 3, device=self._device)
        self.ref_base_ang_vel_local = torch.zeros(self.num_envs, 3, device=self._device)
        self.ref_dof_pos = torch.zeros(self.num_envs, self._robot.dof_dim, device=self._device)
        self.ref_dof_vel = torch.zeros(self.num_envs, self._robot.dof_dim, device=self._device)

        if self._args.dof_weights is not None:
            dof_weights_list = []
            for dof_name in self.robot.dof_names:
                dof_weights_list.append(1.0)
                for key in self._args.dof_weights.keys():
                    if key in dof_name:
                        dof_weights_list[-1] = self._args.dof_weights[key]
                        break
            self.dof_weights = torch.tensor(
                dof_weights_list, device=self._device, dtype=torch.float32
            )
        else:
            self.dof_weights = torch.ones(
                self._robot.dof_dim, device=self._device, dtype=torch.float32
            )
        self.dof_pos_error_weighted = torch.zeros(self.num_envs, device=self._device)
        self.dof_vel_error_weighted = torch.zeros(self.num_envs, device=self._device)

        # Motion library and reference buffers
        self._motion_lib = MotionLib(
            motion_file=self._args.motion_file,
            device=self._device,
            tracking_link_names=self._args.tracking_link_names,
        )
        if self._args.motion_file is not None:
            self.ref_joint_idx_local = [
                self._motion_lib.get_joint_idx_by_name(name) for name in self.robot.dof_names
            ]
        observed_steps = self._args.observed_steps
        self._motion_obs_steps = self.motion_lib.get_observed_steps(observed_steps)
        self.deviation_thresholds = self._args.deviation_thresholds

        # tracking link indices
        tracking_link_names = self._args.tracking_link_names
        self.tracking_link_idx_local = [
            self._robot.link_names.index(name) for name in tracking_link_names
        ]
        if self._args.link_pos_global_weights is not None:
            tracking_link_weights_list = []
            for link_name in tracking_link_names:
                tracking_link_weights_list.append(1.0)
                for key in self._args.link_pos_global_weights.keys():
                    if key in link_name:
                        tracking_link_weights_list[-1] = self._args.link_pos_global_weights[key]
                        break
            self.tracking_link_pos_global_weights = torch.tensor(
                tracking_link_weights_list, device=self._device, dtype=torch.float32
            )
        else:
            self.tracking_link_pos_global_weights = torch.ones(
                len(tracking_link_names), device=self._device, dtype=torch.float32
            )
        if self._args.link_pos_local_weights is not None:
            tracking_link_weights_list = []
            for link_name in tracking_link_names:
                tracking_link_weights_list.append(1.0)
                for key in self._args.link_pos_local_weights.keys():
                    if key in link_name:
                        tracking_link_weights_list[-1] = self._args.link_pos_local_weights[key]
                        break
            self.tracking_link_pos_local_weights = torch.tensor(
                tracking_link_weights_list, device=self._device, dtype=torch.float32
            )
        else:
            self.tracking_link_pos_local_weights = torch.ones(
                len(tracking_link_names), device=self._device, dtype=torch.float32
            )
        if self._args.link_quat_weights is not None:
            tracking_link_weights_list = []
            for link_name in tracking_link_names:
                tracking_link_weights_list.append(1.0)
                for key in self._args.link_quat_weights.keys():
                    if key in link_name:
                        tracking_link_weights_list[-1] = self._args.link_quat_weights[key]
                        break
            self.tracking_link_quat_weights = torch.tensor(
                tracking_link_weights_list, device=self._device, dtype=torch.float32
            )
        else:
            self.tracking_link_quat_weights = torch.ones(
                len(tracking_link_names), device=self._device, dtype=torch.float32
            )

        # Get NUM_MOTION_OBS by using motionlib and _build_motion_obs_from_dict
        if len(self._motion_obs_steps) > 0 and self._args.motion_file is not None:
            # Create dummy data for size calculation
            dummy_motion_ids = torch.zeros(1, device=self._device, dtype=torch.long)
            dummy_motion_times = torch.zeros(1, device=self._device, dtype=torch.float32)
            dummy_envs_idx = torch.zeros(1, device=self._device, dtype=torch.long)

            curr_motion_obs_dict, future_motion_obs_dict = self.motion_lib.get_motion_future_obs(
                dummy_motion_ids, dummy_motion_times, self._motion_obs_steps
            )

            self.base_quat = torch.zeros(1, 4, device=self._device, dtype=torch.float32)
            self.base_quat[:, 0] = 1.0  # identity quaternion

            # motion_obs_sample = self._build_motion_obs_from_dict(
            #     curr_motion_obs_dict, future_motion_obs_dict, dummy_envs_idx
            # )
            motion_obs_sample = build_motion_obs_from_dict(
                curr_motion_obs_dict,
                future_motion_obs_dict,
                dummy_envs_idx,
                base_quat=self.base_quat,
            )
            NUM_MOTION_OBS = motion_obs_sample.shape[1]
        else:
            NUM_MOTION_OBS = 0
        # motion observation buffer (future-step observations after post-processing)
        self.motion_obs = torch.zeros(self.num_envs, NUM_MOTION_OBS, device=self._device)

        self.tracking_link_pos_global = torch.zeros(
            self.num_envs, len(self.tracking_link_idx_local), 3, device=self._device
        )
        self.tracking_link_quat_global = torch.zeros(
            self.num_envs, len(self.tracking_link_idx_local), 4, device=self._device
        )
        self.tracking_link_pos_local_yaw = torch.zeros(
            self.num_envs, len(self.tracking_link_idx_local), 3, device=self._device
        )
        self.tracking_link_quat_local_yaw = torch.zeros(
            self.num_envs, len(self.tracking_link_idx_local), 4, device=self._device
        )
        self.ref_tracking_link_pos_global = torch.zeros(
            self.num_envs, len(self.tracking_link_idx_local), 3, device=self._device
        )
        self.ref_tracking_link_quat_global = torch.zeros(
            self.num_envs, len(self.tracking_link_idx_local), 4, device=self._device
        )
        self.ref_tracking_link_pos_local_yaw = torch.zeros(
            self.num_envs, len(self.tracking_link_idx_local), 3, device=self._device
        )
        self.ref_tracking_link_quat_local_yaw = torch.zeros(
            self.num_envs, len(self.tracking_link_idx_local), 4, device=self._device
        )
        self.tracking_link_lin_vel_global = torch.zeros(
            self.num_envs, len(self.tracking_link_idx_local), 3, device=self._device
        )
        self.tracking_link_ang_vel_global = torch.zeros(
            self.num_envs, len(self.tracking_link_idx_local), 3, device=self._device
        )
        self.tracking_link_lin_vel_local_yaw = torch.zeros(
            self.num_envs, len(self.tracking_link_idx_local), 3, device=self._device
        )
        self.tracking_link_ang_vel_local_yaw = torch.zeros(
            self.num_envs, len(self.tracking_link_idx_local), 3, device=self._device
        )
        self.ref_tracking_link_lin_vel_global = torch.zeros(
            self.num_envs, len(self.tracking_link_idx_local), 3, device=self._device
        )
        self.ref_tracking_link_ang_vel_global = torch.zeros(
            self.num_envs, len(self.tracking_link_idx_local), 3, device=self._device
        )
        self.ref_foot_contact = torch.zeros(
            self.num_envs, len(self._robot.foot_links_idx), device=self._device
        )
        self.ref_foot_contact_weighted = torch.zeros(
            self.num_envs, len(self._robot.foot_links_idx), device=self._device
        )

        # differences to reference (used for observations)
        self.diff_base_pos_local_yaw = torch.zeros(self.num_envs, 3, device=self._device)
        self.diff_base_rotation_6D = torch.zeros(
            self.num_envs, 6, device=self._device, dtype=torch.float32
        )
        self.diff_dof_pos = torch.zeros(self.num_envs, self._robot.dof_dim, device=self._device)
        self.diff_dof_vel = torch.zeros(self.num_envs, self._robot.dof_dim, device=self._device)
        self.diff_base_euler = torch.zeros(
            self.num_envs, 3, device=self._device, dtype=torch.float32
        )
        self.diff_base_yaw = torch.zeros(self.num_envs, 1, device=self._device, dtype=torch.float32)
        self.diff_base_lin_vel_local = torch.zeros(self.num_envs, 3, device=self._device)
        self.diff_base_ang_vel_local = torch.zeros(self.num_envs, 3, device=self._device)
        self.diff_tracking_link_pos_global_local_yaw = torch.zeros(
            self.num_envs, len(self.tracking_link_idx_local), 3, device=self._device
        )
        self.diff_tracking_link_pos_local_yaw = torch.zeros(
            self.num_envs, len(self.tracking_link_idx_local), 3, device=self._device
        )
        self.diff_tracking_link_quat_local_yaw = torch.zeros(
            self.num_envs, len(self.tracking_link_idx_local), 4, device=self._device
        )
        self.diff_tracking_link_rotation_6D = torch.zeros(
            self.num_envs, len(self.tracking_link_idx_local), 6, device=self._device
        )
        self.diff_tracking_link_lin_vel_local_yaw = torch.zeros(
            self.num_envs, len(self.tracking_link_idx_local), 3, device=self._device
        )
        self.diff_tracking_link_ang_vel_local_yaw = torch.zeros(
            self.num_envs, len(self.tracking_link_idx_local), 3, device=self._device
        )

        # per-env motion selection and time offset
        self._motion_ids = torch.zeros(self.num_envs, device=self._device, dtype=torch.long)
        self._motion_time_offsets = torch.zeros(
            self.num_envs, device=self._device, dtype=torch.float32
        )
        self._motion_lengths = torch.zeros(self.num_envs, device=self._device, dtype=torch.float32)

        # termination
        error_list = [
            "base_pos_error",
            "base_height_error",
            "base_quat_error",
            "dof_pos_error",
            "tracking_link_pos_error",
        ]
        self._terminate_after_error = {}
        self._min_terminate_after_error = {}
        self._max_terminate_after_error = {}
        self._num_step_since_update_terminate_error = 0
        self._error_mask_buffer = {}
        for error_name in error_list:
            if error_name not in self._args.terminate_after_error.keys():
                continue
            terminate_after_error = self._args.terminate_after_error[error_name][0]
            min_terminate_after_error, max_terminate_after_error = self._args.terminate_after_error[  # pyright: ignore[reportGeneralTypeIssues]
                error_name
            ][1]
            if terminate_after_error:
                self._min_terminate_after_error[error_name] = min_terminate_after_error
                self._max_terminate_after_error[error_name] = max_terminate_after_error
                self._terminate_after_error[error_name] = self._args.terminate_after_error[
                    error_name
                ][0]
                self._error_mask_buffer[error_name] = []
        self.deviation_buf = torch.ones(self.num_envs, device=self._device, dtype=torch.float32)

        # randomize base yaw offset
        self.base_yaw_offset = torch.zeros(self.num_envs, device=self._device, dtype=torch.float32)
        self.base_yaw_offset_quat = torch.zeros(
            self.num_envs, 4, device=self._device, dtype=torch.float32
        )
        self.base_yaw_offset_quat[:, 0] = 1.0
        self.base_pos_offset = torch.zeros(
            self.num_envs, 3, device=self._device, dtype=torch.float32
        )

        # Let base class set up common buffers, spaces, and rendering
        super()._init()

        # initialize once
        envs_idx = torch.arange(self.num_envs, device=self._device, dtype=torch.long)
        self.reset_idx(envs_idx)

    def _reset_buffers(self, envs_idx: torch.Tensor) -> None:
        super()._reset_buffers(envs_idx=envs_idx)
        self.feet_air_time[envs_idx] = 0.0

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        if self._eval_mode:
            self.reset_to_default_pos(envs_idx)
            return

        # set reference motion first
        self.time_since_reset[envs_idx] = 0.0
        self._reset_motion(envs_idx=envs_idx)

        num_selected = int(self._args.reset_to_default_pose_ratio * len(envs_idx))
        if num_selected > 0:
            local_idx = torch.randperm(len(envs_idx), device=envs_idx.device)[:num_selected]
            envs_selected = envs_idx[local_idx]

            # Set motion time to 0.0 by zeroing time_since_reset for selected
            self._motion_time_offsets[envs_selected] = 0.0

            # Update reference at t=0 for these envs
            self._update_ref_motion(
                envs_idx=envs_selected,
                motion_ids=self._motion_ids[envs_selected],
                motion_times=self._motion_time_offsets[envs_selected],
            )

            self.reset_to_default_pos(envs_selected)

        self._reset_buffers(envs_idx=envs_idx)

    def reset_to_default_pos(self, envs_idx: torch.Tensor) -> None:
        # different from super().reset_idx(), considering the facing direction
        default_pos = self._robot.default_pos[None, :].repeat(len(envs_idx), 1)
        default_quat = self._robot.default_quat[None, :].repeat(len(envs_idx), 1)
        default_dof_pos = self._robot.default_dof_pos[None, :].repeat(len(envs_idx), 1)

        base_pos = self.ref_base_pos[envs_idx]
        base_pos[:, 2] = default_pos[:, 2]
        random_euler = torch.zeros(len(envs_idx), 3, device=self._device)
        random_euler[:, 0] = (
            torch.rand(len(envs_idx), device=self._device)
            * (self._args.reset_pitch_range[1] - self._args.reset_pitch_range[0])
            + self._args.reset_pitch_range[0]
        )
        random_euler[:, 1] = (
            torch.rand(len(envs_idx), device=self._device)
            * (self._args.reset_roll_range[1] - self._args.reset_roll_range[0])
            + self._args.reset_roll_range[0]
        )
        random_euler[:, 2] = self.ref_base_euler[envs_idx, 2]
        random_euler[:, 2] += (
            torch.rand(len(envs_idx), device=self._device)
            * (self._args.reset_yaw_range[1] - self._args.reset_yaw_range[0])
            + self._args.reset_yaw_range[0]
        )
        random_quat = quat_from_euler(random_euler)
        base_quat = quat_mul(default_quat, random_quat)
        random_dof_pos = (
            torch.rand(len(envs_idx), self._robot.dof_dim, device=self._device)
            * (self._args.reset_dof_pos_range[1] - self._args.reset_dof_pos_range[0])
            + self._args.reset_dof_pos_range[0]
        )
        dof_pos = default_dof_pos + random_dof_pos

        # Apply state to robot for selected envs
        self._robot.set_state(pos=base_pos, quat=base_quat, dof_pos=dof_pos, envs_idx=envs_idx)

    def get_terminated(self) -> torch.Tensor:
        # termination dictionary for extra info
        termination_dict = {}

        time_out_buf = self.get_truncated()
        reset_buf = time_out_buf.clone()

        # motion_end is not timeout (causes value function explosion)
        motion_end_buf = self.motion_times + self.dt > self._motion_lengths
        reset_buf |= motion_end_buf

        contact_force_mask = torch.any(
            torch.norm(self.link_contact_forces[:, self._terminate_link_idx_local, :], dim=-1)
            > 1.0,
            dim=-1,
        )
        reset_buf |= contact_force_mask

        # Only enable error-based termination after a certain motion time, if specified
        terminate_by_error = self.motion_times > self._args.no_terminate_before_motion_time
        terminate_by_error |= self.time_since_reset > self._args.no_terminate_after_reset_time
        terminate_by_error |= (
            self.time_since_random_push > self._args.no_terminate_after_random_push_time
        )

        error_dict = self._compute_error()
        error_mask = {}
        for error_name in self._terminate_after_error.keys():
            error_mask[error_name] = (
                error_dict[error_name] > self._terminate_after_error[error_name]
            )
            termination_dict[f"{error_name}"] = error_mask[error_name].clone()
        if len(self._terminate_after_error.keys()) > 0:
            terminate_by_error &= torch.any(torch.stack(list(error_mask.values())), dim=0)

        if not self._eval_mode:
            reset_buf |= terminate_by_error

        if self._args.adaptive_termination_ratio is not None:
            self._update_terminate_error(error_mask)
            for key in self._terminate_after_error.keys():
                self._extra_info["info"][f"threshold_{key}"] = self._terminate_after_error[key]

        # for error_name in self._terminate_after_error.keys():
        #     if error_mask[error_name][0]:
        #         print(f"terminate by {error_name}")

        self.reset_buf[:] = reset_buf

        termination_dict["time_out"] = time_out_buf.clone()
        termination_dict["motion_end"] = motion_end_buf.clone()
        # termination_dict["terminate_by_error"] = terminate_by_error.clone()
        termination_dict["contact_force"] = contact_force_mask.clone()
        # termination_dict["any"] = reset_buf.clone()
        self._extra_info["termination"] = termination_dict

        return reset_buf

    def _compute_error(self) -> dict[str, float]:
        base_pos_error = torch.norm(self.base_pos - self.ref_base_pos, dim=-1)
        base_height_error = torch.abs(self.base_height - self.ref_base_height)
        base_quat_error = quat_error_magnitude(self.base_quat, self.ref_base_quat)
        base_lin_vel_error = torch.norm(self.base_lin_vel - self.ref_base_lin_vel, dim=-1)
        base_ang_vel_error = torch.norm(self.base_ang_vel - self.ref_base_ang_vel, dim=-1)
        dof_pos_error = torch.sum(
            self.dof_weights * torch.abs(self.dof_pos - self.ref_dof_pos), dim=-1
        )
        tracking_link_pos_error = torch.norm(
            self.tracking_link_pos_local_yaw - self.ref_tracking_link_pos_local_yaw, dim=-1
        ).mean(dim=-1)
        foot_contact_weighted_error = (
            self.foot_contact_weighted * (1 - self.ref_foot_contact)
        ).sum(dim=-1)

        error_dict = {}
        error_dict["base_pos_error"] = base_pos_error.clone()
        error_dict["base_height_error"] = base_height_error.clone()
        error_dict["base_quat_error"] = base_quat_error.clone()
        error_dict["base_lin_vel_error"] = base_lin_vel_error.clone()
        error_dict["base_ang_vel_error"] = base_ang_vel_error.clone()
        error_dict["dof_pos_error"] = dof_pos_error.clone()
        error_dict["tracking_link_pos_error"] = tracking_link_pos_error.clone()
        error_dict["foot_contact_weighted_error"] = foot_contact_weighted_error.clone()

        self.deviation_buf = torch.ones_like(self.deviation_buf)
        for key in self.deviation_thresholds.keys():
            self.deviation_buf *= (2 - error_dict[key] / self.deviation_thresholds[key]).clamp(
                max=1.0, min=0.0
            )

        error_mean_dict = {}
        for key, value in error_dict.items():
            error_mean_dict[key] = value.mean().item()
        self._extra_info["info"].update(error_mean_dict)

        return error_dict

    def _update_terminate_error(self, error_mask: dict[str, torch.Tensor]) -> None:
        for error_name in self._terminate_after_error.keys():
            self._error_mask_buffer[error_name].append(error_mask[error_name].float())
        self._num_step_since_update_terminate_error += 1
        if self._num_step_since_update_terminate_error >= 24:
            self._num_step_since_update_terminate_error = 0
            for error_name in self._terminate_after_error.keys():
                terminate_by_error_ratio = torch.mean(
                    torch.stack(self._error_mask_buffer[error_name])
                ).item()
                if terminate_by_error_ratio > 3.0 * self._args.adaptive_termination_ratio:  # type: ignore
                    self._terminate_after_error[error_name] *= 1.2
                    self._terminate_after_error[error_name] = min(
                        self._max_terminate_after_error[error_name],
                        self._terminate_after_error[error_name],
                    )
                elif terminate_by_error_ratio < 0.3 * self._args.adaptive_termination_ratio:  # type: ignore
                    self._terminate_after_error[error_name] /= 1.2
                    self._terminate_after_error[error_name] = max(
                        self._min_terminate_after_error[error_name],
                        self._terminate_after_error[error_name],
                    )
                self._error_mask_buffer[error_name] = []

    def apply_action(self, action: torch.Tensor) -> None:
        super().apply_action(action=action)
        self.feet_first_contact[:] = (self.feet_air_time > 0.0) * self.foot_contact
        self.feet_air_time += self.dt

    def _pre_step(self) -> None:
        super()._pre_step()

    def update_buffers(self) -> None:
        self.base_pos[:] = self._robot.base_pos
        self.base_quat[:] = self._robot.base_quat
        self.base_height[:] = self.base_pos[:, 2]
        self.base_rotation_6D[:] = quat_to_rotation_6D(self.base_quat)
        self.base_euler[:] = quat_to_euler(self.base_quat)
        self.projected_gravity[:] = quat_apply(quat_inv(self.base_quat), self.global_gravity)

        self.link_contact_forces[:] = self._robot.link_contact_forces
        self.link_positions[:] = self._robot.link_positions
        self.link_quaternions[:] = self._robot.link_quaternions
        self.link_lin_velocities[:] = self._robot.link_lin_velocities
        self.link_ang_velocities[:] = self._robot.link_ang_velocities

        self.base_lin_vel[:] = self.link_lin_velocities[:, 0]
        self.base_ang_vel[:] = self.link_ang_velocities[:, 0]
        self.base_lin_vel_local[:] = self.global_to_local(self.base_lin_vel)
        self.base_ang_vel_local[:] = self.global_to_local(self.base_ang_vel)
        self.body_lin_vel[:] = self.link_lin_velocities[:, self._robot.body_link_idx]
        self.body_ang_vel[:] = self.link_ang_velocities[:, self._robot.body_link_idx]
        body_quat = self._robot.link_quaternions[:, self._robot.body_link_idx]
        self.body_euler[:] = quat_to_euler(body_quat)

        self.tracking_link_pos_global[:] = self._robot.link_positions[
            :, self.tracking_link_idx_local
        ].clone()
        self.tracking_link_quat_global[:] = self._robot.link_quaternions[
            :, self.tracking_link_idx_local
        ].clone()
        link_pos_local_yaw = self._robot.link_positions
        link_pos_local_yaw[:, :, :] -= self.base_pos[:, None, :]
        quat_yaw = quat_from_angle_axis(
            self.base_euler[:, 2], torch.tensor([0, 0, 1], device=self._device, dtype=torch.float)
        )
        link_pos_local_yaw = self.batched_global_to_local(quat_yaw, link_pos_local_yaw)
        self.link_pos_local_yaw[:] = link_pos_local_yaw
        link_quat_local_yaw = self.batched_global_to_local(quat_yaw, self._robot.link_quaternions)
        self.link_quat_local_yaw[:] = link_quat_local_yaw
        self.tracking_link_pos_local_yaw[:] = self.link_pos_local_yaw[
            :, self.tracking_link_idx_local
        ]
        self.tracking_link_quat_local_yaw[:] = self.link_quat_local_yaw[
            :, self.tracking_link_idx_local
        ]
        self.tracking_link_lin_vel_global[:] = self.link_lin_velocities[
            :, self.tracking_link_idx_local
        ]
        self.tracking_link_ang_vel_global[:] = self.link_ang_velocities[
            :, self.tracking_link_idx_local
        ]
        self.tracking_link_lin_vel_local_yaw[:] = self.batched_global_to_local(
            quat_yaw, self.tracking_link_lin_vel_global
        )
        self.tracking_link_ang_vel_local_yaw[:] = self.batched_global_to_local(
            quat_yaw, self.tracking_link_ang_vel_global
        )

        self.dof_pos_error_weighted[:] = torch.sum(
            self.dof_weights * torch.square(self.dof_pos - self.ref_dof_pos), dim=-1
        )
        self.dof_vel_error_weighted[:] = torch.sum(
            self.dof_weights * torch.square(self.dof_vel - self.ref_dof_vel), dim=-1
        )

        # contacts
        self.feet_height[:] = self.link_positions[:, self._robot.foot_links_idx, 2]
        self.feet_velocity[:] = self.link_lin_velocities[:, self._robot.foot_links_idx]
        self.foot_contact_force[:] = self.link_contact_forces[:, self._robot.foot_links_idx, 2]
        self.foot_contact[:] = self.foot_contact_force > 1.0
        self.foot_contact_weighted[:] = self.foot_contact_force / (
            self.robot.mass * self.scene.gravity
        )

    def update_history(self) -> None:
        super().update_history()
        self.feet_air_time *= 1 - self.foot_contact
        # update reference motion after calculating rewards
        self._update_ref_motion()

    # ---------- Motion utilities ----------
    def _reset_motion(self, envs_idx: torch.Tensor) -> None:
        """
        Reset the motion ids and times for the given environments.
        """
        assert self._motion_lib is not None
        n = len(envs_idx)
        motion_ids = self._motion_lib.sample_motion_ids(n)
        motion_times = self._motion_lib.sample_motion_times(motion_ids)
        self._motion_ids[envs_idx] = motion_ids
        self._motion_time_offsets[envs_idx] = motion_times
        self._motion_lengths[envs_idx] = self._motion_lib.get_motion_length(motion_ids)
        self.base_yaw_offset[envs_idx] = (
            torch.rand(len(envs_idx), device=self._device, dtype=torch.float32) * 2 * torch.pi
        )
        self.base_yaw_offset_quat[envs_idx] = quat_from_angle_axis(
            self.base_yaw_offset[envs_idx],
            torch.tensor([0, 0, 1], device=self._device, dtype=torch.float),
        )
        self.base_pos_offset[envs_idx, :2] = (
            torch.rand(len(envs_idx), 2, device=self._device, dtype=torch.float32) - 0.5
        )
        self.hard_sync_motion(envs_idx=envs_idx)

    def hard_reset_motion(self, envs_idx: torch.Tensor, motion_id: int) -> None:
        """
        Reset motion ids to given motion id and reset the motion time to 0.0.
        Hard sync the robot state to current motion frame.
        """
        if motion_id > self.motion_lib.num_motions:
            print(
                f"Motion ID {motion_id} is out of range. Valid range is 0 to {self.motion_lib.num_motions - 1}"
            )
            return
        print(f"Hard resetting motion to {self.motion_lib.motion_names[motion_id]}")
        self._motion_ids[envs_idx] = motion_id
        self._motion_time_offsets[envs_idx] = 0.0
        self._motion_lengths[envs_idx] = self._motion_lib.get_motion_length(
            self._motion_ids[envs_idx]
        )
        self.hard_sync_motion(envs_idx=envs_idx)

    def _update_ref_motion(
        self,
        envs_idx: torch.Tensor | None = None,
        motion_ids: torch.Tensor | None = None,
        motion_times: torch.Tensor | None = None,
    ) -> None:
        """
        Update the reference motion (next motion frame) and motion_obs for the given environments.
        """
        if envs_idx is None:
            envs_idx = torch.arange(self.num_envs, device=self._device, dtype=torch.long)
        if motion_ids is None:
            motion_ids = self.motion_ids[envs_idx]
        if motion_times is None:
            motion_times = self.motion_times[envs_idx]
        (
            base_pos,
            base_quat,
            base_lin_vel,
            base_ang_vel,
            base_ang_vel_local,
            dof_pos,
            dof_vel,
            link_pos_global,
            link_pos_local,
            link_quat_global,
            link_quat_local,
            link_lin_vel_global,
            link_lin_vel_local,
            link_ang_vel_global,
            link_ang_vel_local,
            foot_contact,
            foot_contact_weighted,
        ) = self._motion_lib.get_ref_motion_frame(motion_ids, motion_times)

        _ = link_lin_vel_local
        _ = link_ang_vel_local

        curr_motion_obs_dict, future_motion_obs_dict = self.motion_lib.get_motion_future_obs(
            motion_ids, motion_times, self._motion_obs_steps
        )
        # self.motion_obs[envs_idx] = self._build_motion_obs_from_dict(
        #     curr_motion_obs_dict, future_motion_obs_dict, envs_idx
        # )
        self.motion_obs[envs_idx] = build_motion_obs_from_dict(
            curr_motion_obs_dict,
            future_motion_obs_dict,
            envs_idx,
            base_quat=self.base_quat[envs_idx],
        )

        base_pos = (
            quat_apply(self.base_yaw_offset_quat[envs_idx], base_pos)
            + self.base_pos_offset[envs_idx]
        )
        link_pos_global = quat_apply(
            self.base_yaw_offset_quat[envs_idx, None, :].repeat(1, link_pos_global.shape[1], 1),
            link_pos_global,
        ) + self.base_pos_offset[envs_idx, None, :].repeat(1, link_pos_global.shape[1], 1)
        base_quat = quat_mul(self.base_yaw_offset_quat[envs_idx], base_quat)
        base_lin_vel = quat_apply(self.base_yaw_offset_quat[envs_idx], base_lin_vel)
        base_ang_vel = quat_apply(self.base_yaw_offset_quat[envs_idx], base_ang_vel)

        self.ref_base_pos[envs_idx] = base_pos
        self.ref_base_quat[envs_idx] = base_quat
        self.ref_base_height[envs_idx] = self.ref_base_pos[envs_idx, 2]
        self.ref_base_rotation_6D[envs_idx] = quat_to_rotation_6D(base_quat)
        self.ref_base_euler[envs_idx] = quat_to_euler(base_quat)
        self.ref_base_lin_vel[envs_idx] = base_lin_vel
        self.ref_base_ang_vel[envs_idx] = base_ang_vel
        self.ref_base_lin_vel_local[envs_idx] = self.batched_global_to_local(
            self.ref_base_quat[envs_idx], base_lin_vel
        )
        self.ref_base_ang_vel_local[envs_idx] = base_ang_vel_local
        self.ref_dof_pos[envs_idx] = dof_pos[:, self.ref_joint_idx_local]
        self.ref_dof_vel[envs_idx] = dof_vel[:, self.ref_joint_idx_local]
        self.ref_tracking_link_pos_global[envs_idx] = link_pos_global
        self.ref_tracking_link_pos_local_yaw[envs_idx] = link_pos_local
        self.ref_tracking_link_quat_local_yaw[envs_idx] = link_quat_local
        self.ref_tracking_link_quat_global[envs_idx] = link_quat_global
        self.ref_tracking_link_lin_vel_global[envs_idx] = link_lin_vel_global
        self.ref_tracking_link_ang_vel_global[envs_idx] = link_ang_vel_global
        self.ref_foot_contact[envs_idx] = foot_contact
        self.ref_foot_contact_weighted[envs_idx] = foot_contact_weighted

        pos_diff = self.ref_base_pos[envs_idx] - self.base_pos[envs_idx]
        quat_yaw = quat_from_angle_axis(
            self.base_euler[envs_idx, -1],
            torch.tensor([0, 0, 1], device=self._device, dtype=torch.float),
        )
        self.diff_base_pos_local_yaw[envs_idx] = self.batched_global_to_local(quat_yaw, pos_diff)
        base_quat_diff = quat_diff(self.base_quat[envs_idx], self.ref_base_quat[envs_idx])
        self.diff_base_rotation_6D[envs_idx] = quat_to_rotation_6D(base_quat_diff)
        self.diff_base_euler[envs_idx] = self.ref_base_euler[envs_idx] - self.base_euler[envs_idx]
        self.diff_base_yaw[envs_idx] = (
            self.ref_base_euler[envs_idx, 2:3] - self.base_euler[envs_idx, 2:3]
        )
        self.diff_dof_pos[envs_idx] = self.ref_dof_pos[envs_idx] - self.dof_pos[envs_idx]
        self.diff_dof_vel[envs_idx] = self.ref_dof_vel[envs_idx] - self.dof_vel[envs_idx]
        self.diff_base_lin_vel_local[envs_idx] = (
            self.ref_base_lin_vel_local[envs_idx] - self.base_lin_vel_local[envs_idx]
        )
        self.diff_base_ang_vel_local[envs_idx] = (
            self.ref_base_ang_vel_local[envs_idx] - self.base_ang_vel_local[envs_idx]
        )
        if len(self.tracking_link_idx_local) > 0:
            diff_tracking_link_pos_global = (
                self.ref_tracking_link_pos_global[envs_idx]
                - self.tracking_link_pos_global[envs_idx]
            )
            self.diff_tracking_link_pos_global_local_yaw[envs_idx] = self.batched_global_to_local(
                quat_yaw, diff_tracking_link_pos_global
            )
            self.diff_tracking_link_pos_local_yaw[envs_idx] = (
                self.tracking_link_pos_local_yaw[envs_idx]
                - self.ref_tracking_link_pos_local_yaw[envs_idx]
            )
            self.diff_tracking_link_quat_local_yaw[envs_idx] = quat_diff(
                self.ref_tracking_link_quat_local_yaw[envs_idx],
                self.tracking_link_quat_local_yaw[envs_idx],
            )
            self.diff_tracking_link_rotation_6D[envs_idx] = quat_to_rotation_6D(
                self.diff_tracking_link_quat_local_yaw[envs_idx]
            )
            diff_tracking_link_lin_vel_global = (
                self.ref_tracking_link_lin_vel_global[envs_idx]
                - self.tracking_link_lin_vel_global[envs_idx]
            )
            diff_tracking_link_ang_vel_global = (
                self.ref_tracking_link_ang_vel_global[envs_idx]
                - self.tracking_link_ang_vel_global[envs_idx]
            )
            self.diff_tracking_link_lin_vel_local_yaw[envs_idx] = self.batched_global_to_local(
                quat_yaw, diff_tracking_link_lin_vel_global
            )
            self.diff_tracking_link_ang_vel_local_yaw[envs_idx] = self.batched_global_to_local(
                quat_yaw, diff_tracking_link_ang_vel_global
            )

    def get_observations(self, obs_args: Any = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Override to support teacher-specific motion_obs differing from student's config."""
        self.update_buffers()
        args_to_use = obs_args if obs_args is not None else self._args

        obs_components = []
        for key in args_to_use.actor_obs_terms:
            if key == "motion_obs" and obs_args is not None:
                # Build a teacher-specific motion_obs on the fly
                steps_map = self.motion_lib.get_observed_steps(args_to_use.observed_steps)
                if len(steps_map) > 0:
                    envs_idx = torch.arange(self.num_envs, device=self._device, dtype=torch.long)
                    motion_ids = self.motion_ids[envs_idx]
                    motion_times = self.motion_times[envs_idx]
                    curr_d, future_d = self.motion_lib.get_motion_future_obs(
                        motion_ids, motion_times, steps_map
                    )
                    # obs_gt = self._build_motion_obs_from_dict(curr_d, future_d, envs_idx)
                    obs_gt = build_motion_obs_from_dict(
                        curr_d,
                        future_d,
                        envs_idx,
                        base_quat=self.base_quat[envs_idx],
                    )
                else:
                    obs_gt = torch.zeros(self.num_envs, 0, device=self._device)
                obs_gt = obs_gt * args_to_use.obs_scales.get(key, 1.0)
            else:
                obs_gt = getattr(self, key) * args_to_use.obs_scales.get(key, 1.0)
                if len(obs_gt.shape) > 2:
                    obs_gt = obs_gt.view(self.num_envs, -1)
            obs_noise = torch.randn_like(obs_gt) * args_to_use.obs_noises.get(key, 0.0)
            if self._eval_mode:
                obs_noise *= 0
            obs_components.append(obs_gt + obs_noise)
        actor_obs = torch.cat(obs_components, dim=-1)

        obs_components = []
        for key in args_to_use.critic_obs_terms:
            obs_gt = getattr(self, key) * args_to_use.obs_scales.get(key, 1.0)
            if len(obs_gt.shape) > 2:
                obs_gt = obs_gt.view(self.num_envs, -1)
            obs_components.append(obs_gt)
        critic_obs = torch.cat(obs_components, dim=-1)
        return actor_obs, critic_obs

    def hard_sync_motion(self, envs_idx: torch.Tensor) -> None:
        """
        Hard sync the robot state to current motion frame for the given environments.
        """
        motion_ids = self._motion_ids[envs_idx]
        motion_times = self.motion_times[envs_idx]
        self._update_ref_motion(envs_idx=envs_idx, motion_ids=motion_ids, motion_times=motion_times)
        (
            base_pos,
            base_quat,
            base_lin_vel,
            base_ang_vel,
            dof_pos,
            dof_vel,
        ) = self._motion_lib.get_motion_frame(motion_ids, motion_times)
        base_pos = (
            quat_apply(self.base_yaw_offset_quat[envs_idx], base_pos)
            + self.base_pos_offset[envs_idx]
        )
        base_pos[:, 2] += 0.03
        base_quat = quat_mul(self.base_yaw_offset_quat[envs_idx], base_quat)
        base_lin_vel = quat_apply(self.base_yaw_offset_quat[envs_idx], base_lin_vel)
        base_ang_vel = quat_apply(self.base_yaw_offset_quat[envs_idx], base_ang_vel)
        self._robot.set_state(
            pos=base_pos,
            quat=base_quat,
            dof_pos=dof_pos,
            envs_idx=envs_idx,
            lin_vel=base_lin_vel,
            ang_vel=base_ang_vel,
            dof_vel=dof_vel,
        )

    def update_motion_lib(self, motion_lib: MotionLib) -> None:
        print("Warning: motion_lib is being replaced.")
        self._motion_lib = motion_lib

    @property
    def motion_lib(self) -> MotionLib:
        return self._motion_lib

    @property
    def motion_ids(self) -> torch.Tensor:
        return self._motion_ids

    @property
    def motion_times(self) -> torch.Tensor:
        return self.time_since_reset + self._motion_time_offsets
