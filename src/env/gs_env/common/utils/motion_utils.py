import json
import os
import pickle
import statistics
import time
from typing import Any

import mink
import mujoco as mj
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

#
from gs_env.common.utils.math_utils import (
    pose_diff_quat,
    pose_mul_quat,
    quat_apply,
    quat_diff,
    quat_from_angle_axis,
    quat_from_euler,
    quat_inv,
    quat_mul,
    quat_to_angle_axis,
    quat_to_euler,
    quat_to_rotation_6D,
    slerp,
)

_DEFAULT_DEVICE = torch.device("cpu")


class MotionLib:
    def __init__(
        self,
        motion_file: str | None = None,
        device: torch.device = _DEFAULT_DEVICE,
        target_fps: float = 50.0,
        tracking_link_names: list[str] | None = None,
    ) -> None:
        self._device = device
        self._target_fps = target_fps
        self._tracking_link_names = tracking_link_names
        self._motion_obs_steps = None
        foot_contac_weights = torch.tensor(
            [225 - (i - 15) ** 2 for i in range(31)], dtype=torch.float, device=self._device
        )
        self._foot_contac_weights = foot_contac_weights / foot_contac_weights.sum()
        if motion_file is not None:
            self._load_motions(motion_file)

    def _load_motions(self, motion_file: str) -> None:
        self._motion_names = []
        self._motion_files = []
        self._dof_names = []

        motion_weights = []
        motion_num_frames = []
        motion_lengths = []

        motion_base_pos = []
        motion_base_quat = []
        motion_base_lin_vel = []
        motion_base_ang_vel = []
        motion_base_ang_vel_local = []
        motion_dof_pos = []
        motion_dof_vel = []
        motion_link_pos_global = []
        motion_link_quat_global = []
        motion_link_pos_local = []
        motion_link_quat_local = []
        motion_link_lin_vel = []
        motion_link_lin_vel_local = []
        motion_link_ang_vel = []
        motion_link_ang_vel_local = []
        motion_foot_contact = []
        motion_foot_contact_weighted = []

        full_motion_files, full_motion_weights = self._fetch_motion_files(motion_file)
        num_motion_files = len(full_motion_files)

        for i in tqdm(range(num_motion_files), desc="[MotionLib] Loading motions"):
            curr_file = full_motion_files[i]
            try:
                with open(curr_file, "rb") as f:
                    motion_data = pickle.load(f)
            except Exception as e:
                print(f"Error loading motion file {curr_file}: {e}")
                continue

            if len(self._dof_names) == 0:
                self._dof_names = motion_data["dof_names"]

            link_names = motion_data["link_names"]
            tracking_link_indices = []
            if self._tracking_link_names is None:
                self._tracking_link_names = link_names
            for name in self._tracking_link_names:
                if name in link_names:
                    tracking_link_indices.append(link_names.index(name))
                else:
                    raise ValueError(
                        f"Tracking link name '{name}' not found in motion data link names"
                    )

            base_pos = torch.tensor(motion_data["pos"], dtype=torch.float, device=self._device)
            base_quat = torch.tensor(motion_data["quat"], dtype=torch.float, device=self._device)

            fps = motion_data["fps"]
            dt = 1.0 / fps
            num_frames = base_pos.shape[0]
            length = dt * (num_frames - 1)

            base_lin_vel = torch.zeros_like(base_pos)
            base_lin_vel[:-1, :] = fps * (base_pos[1:, :] - base_pos[:-1, :])
            base_lin_vel[-1, :] = base_lin_vel[-2, :]
            base_lin_vel = self.smooth(base_lin_vel, 19, device=self._device)

            base_ang_vel = torch.zeros_like(base_pos)  # (num_frames, 3)
            base_dquat = quat_diff(base_quat[:-1], base_quat[1:])
            base_ang_vel[:-1, :] = fps * quat_to_angle_axis(base_dquat)
            base_ang_vel[-1, :] = base_ang_vel[-2, :]
            base_ang_vel = self.smooth(base_ang_vel, 19, device=self._device)

            dof_pos = torch.tensor(motion_data["dof_pos"], dtype=torch.float, device=self._device)
            dof_vel = torch.zeros_like(dof_pos)  # (num_frames, num_dof)
            dof_vel[:-1, :] = fps * (dof_pos[1:, :] - dof_pos[:-1, :])
            dof_vel[-1, :] = dof_vel[-2, :]
            dof_vel = self.smooth(dof_vel, 19, device=self._device)

            link_pos_global = torch.tensor(
                motion_data["link_pos"], dtype=torch.float, device=self._device
            )
            link_quat_global = torch.tensor(
                motion_data["link_quat"], dtype=torch.float, device=self._device
            )
            link_pos_global = link_pos_global[:, tracking_link_indices, :]
            link_quat_global = link_quat_global[:, tracking_link_indices, :]

            foot_contact = torch.tensor(
                motion_data["foot_contact"], dtype=torch.float, device=self._device
            )

            # Resample to target FPS if requested
            target_fps_curr = float(self._target_fps)

            if abs(target_fps_curr - fps) > 1e-6:
                # time length stays the same
                new_num_frames = int(round(length * target_fps_curr)) + 1
                t = torch.linspace(0.0, length, steps=new_num_frames, device=self._device)
                # compute blend weights against original frames
                phase = torch.clip(t / length, 0.0, 1.0)
                idx0 = (phase * (num_frames - 1)).long()
                idx1 = torch.min(idx0 + 1, torch.tensor(num_frames - 1, device=self._device))
                blend = phase * (num_frames - 1) - idx0.float()
                blend_u = blend.unsqueeze(-1)

                # positions, dof: linear
                base_pos = (1.0 - blend_u) * base_pos[idx0] + blend_u * base_pos[idx1]
                dof_pos = (1.0 - blend_u) * dof_pos[idx0] + blend_u * dof_pos[idx1]
                foot_contact = 1 - (1 - foot_contact[idx0]) * (1 - foot_contact[idx1])
                link_pos_global = (1.0 - blend_u.unsqueeze(1)) * link_pos_global[
                    idx0
                ] + blend_u.unsqueeze(1) * link_pos_global[idx1]

                # quaternions: slerp
                base_quat = slerp(base_quat[idx0], base_quat[idx1], blend)
                link_quat_global = slerp(
                    link_quat_global[idx0],
                    link_quat_global[idx1],
                    blend[:, None].repeat(1, link_quat_global.shape[1]),
                )

                # update meta based on resampled length
                fps = target_fps_curr
                dt = 1.0 / fps
                num_frames = base_pos.shape[0]
                length = dt * (num_frames - 1)
            else:
                # ensure library fps is set
                fps = target_fps_curr
                dt = 1.0 / fps

            # recompute velocities at current fps
            base_lin_vel = torch.zeros_like(base_pos)
            base_lin_vel[:-1, :] = fps * (base_pos[1:, :] - base_pos[:-1, :])
            base_lin_vel[-1, :] = base_lin_vel[-2, :]
            base_lin_vel = self.smooth(base_lin_vel, 19, device=self._device)

            base_ang_vel = torch.zeros_like(base_pos)  # (num_frames, 3)
            base_dquat = quat_diff(base_quat[:-1], base_quat[1:])
            base_ang_vel[:-1, :] = fps * quat_to_angle_axis(base_dquat)
            base_ang_vel[-1, :] = base_ang_vel[-2, :]
            base_ang_vel = self.smooth(base_ang_vel, 19, device=self._device)

            base_ang_vel_local = quat_apply(quat_inv(base_quat), base_ang_vel)

            dof_vel = torch.zeros_like(dof_pos)  # (num_frames, num_dof)
            dof_vel[:-1, :] = fps * (dof_pos[1:, :] - dof_pos[:-1, :])
            dof_vel[-1, :] = dof_vel[-2, :]
            dof_vel = self.smooth(dof_vel, 19, device=self._device)

            # recompute local link transforms with yaw-only removal from base
            relative_link_pos_global = link_pos_global.clone()
            relative_link_pos_global[:, :, :] -= base_pos[:, None, :]
            base_euler = quat_to_euler(base_quat)
            base_euler[:, :2] = 0.0
            batched_inv_quat_yaw = quat_from_euler(
                -base_euler[:, None, :].repeat(1, link_pos_global.shape[1], 1)
            )
            link_pos_local = quat_apply(batched_inv_quat_yaw, relative_link_pos_global)
            link_quat_local = quat_mul(batched_inv_quat_yaw, link_quat_global)

            # compute link velocities (global)
            link_lin_vel = torch.zeros_like(link_pos_global)  # (num_frames, num_links, 3)
            link_lin_vel[:-1, :, :] = fps * (link_pos_global[1:, :, :] - link_pos_global[:-1, :, :])
            link_lin_vel[-1, :, :] = link_lin_vel[-2, :, :]
            # Smooth each link separately across frames
            link_lin_vel_flat = link_lin_vel.reshape(
                link_lin_vel.shape[0], -1
            )  # (num_frames, num_links * 3)
            link_lin_vel_flat = self.smooth(link_lin_vel_flat, 19, device=self._device)
            link_lin_vel = link_lin_vel_flat.reshape(link_pos_global.shape)

            link_ang_vel = torch.zeros_like(link_pos_global)  # (num_frames, num_links, 3)
            link_dquat_global = quat_diff(
                link_quat_global[:-1], link_quat_global[1:]
            )  # (num_frames-1, num_links, 4)
            link_ang_vel[:-1, :, :] = fps * quat_to_angle_axis(link_dquat_global)
            link_ang_vel[-1, :, :] = link_ang_vel[-2, :, :]
            # Smooth each link separately across frames
            link_ang_vel_flat = link_ang_vel.reshape(
                link_ang_vel.shape[0], -1
            )  # (num_frames, num_links * 3)
            link_ang_vel_flat = self.smooth(link_ang_vel_flat, 19, device=self._device)
            link_ang_vel = link_ang_vel_flat.reshape(link_pos_global.shape)

            link_lin_vel_local = quat_apply(batched_inv_quat_yaw, link_lin_vel)
            link_ang_vel_local = quat_apply(batched_inv_quat_yaw, link_ang_vel)

            contact_clip_threshold = 0.6
            foot_contact_clipped = (foot_contact > contact_clip_threshold).float()
            foot_contact_sum = foot_contact_clipped.sum(dim=-1)  # (num_frames,)=
            windowed_sum = torch.nn.functional.conv1d(
                foot_contact_sum.view(1, 1, -1),
                self._foot_contac_weights.view(1, 1, -1),
                padding="same",
            )[0, 0, :]
            foot_contact_weighted = foot_contact_clipped / (windowed_sum[:, None] + 1e-8)

            self._motion_names.append(os.path.basename(curr_file))
            self._motion_files.append(curr_file)

            motion_weights.append(full_motion_weights[i])
            motion_num_frames.append(num_frames)
            motion_lengths.append(length)

            motion_base_pos.append(base_pos)
            motion_base_quat.append(base_quat)
            motion_base_lin_vel.append(base_lin_vel)
            motion_base_ang_vel.append(base_ang_vel)
            motion_base_ang_vel_local.append(base_ang_vel_local)
            motion_dof_pos.append(dof_pos)
            motion_dof_vel.append(dof_vel)
            motion_link_pos_global.append(link_pos_global)
            motion_link_quat_global.append(link_quat_global)
            motion_link_pos_local.append(link_pos_local)
            motion_link_quat_local.append(link_quat_local)
            motion_link_lin_vel.append(link_lin_vel)
            motion_link_ang_vel.append(link_ang_vel)
            motion_link_lin_vel_local.append(link_lin_vel_local)
            motion_link_ang_vel_local.append(link_ang_vel_local)
            motion_foot_contact.append(foot_contact)
            motion_foot_contact_weighted.append(foot_contact_weighted)

        assert len(self._dof_names) > 0, "Dof names list is empty"

        motion_weights = torch.tensor(motion_weights, dtype=torch.float, device=self._device)
        self._motion_weights = motion_weights / torch.sum(motion_weights)
        self._motion_num_frames = torch.tensor(
            motion_num_frames, dtype=torch.long, device=self._device
        )
        self._motion_lengths = torch.tensor(motion_lengths, dtype=torch.float, device=self._device)

        self._motion_base_pos = torch.cat(motion_base_pos, dim=0)
        self._motion_base_quat = torch.cat(motion_base_quat, dim=0)
        self._motion_base_lin_vel = torch.cat(motion_base_lin_vel, dim=0)
        self._motion_base_ang_vel = torch.cat(motion_base_ang_vel, dim=0)
        self._motion_base_ang_vel_local = torch.cat(motion_base_ang_vel_local, dim=0)
        self._motion_dof_pos = torch.cat(motion_dof_pos, dim=0)
        self._motion_dof_vel = torch.cat(motion_dof_vel, dim=0)
        self._motion_link_pos_global = torch.cat(motion_link_pos_global, dim=0)
        self._motion_link_quat_global = torch.cat(motion_link_quat_global, dim=0)
        self._motion_link_pos_local = torch.cat(motion_link_pos_local, dim=0)
        self._motion_link_quat_local = torch.cat(motion_link_quat_local, dim=0)
        self._motion_link_lin_vel = torch.cat(motion_link_lin_vel, dim=0)
        self._motion_link_lin_vel_local = torch.cat(motion_link_lin_vel_local, dim=0)
        self._motion_link_ang_vel = torch.cat(motion_link_ang_vel, dim=0)
        self._motion_link_ang_vel_local = torch.cat(motion_link_ang_vel_local, dim=0)
        self._motion_foot_contact = torch.cat(motion_foot_contact, dim=0)
        self._motion_foot_contact_weighted = torch.cat(motion_foot_contact_weighted, dim=0)

        lengths_shifted = self._motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        self._motion_start_idx = lengths_shifted.cumsum(0)  # prefix sum of num frames

        self._motion_ids = torch.arange(self.num_motions, dtype=torch.long, device=self._device)

        print(
            f"Loaded {self.num_motions:d} motions with a total length of {self.total_length:.3f}s."
        )

    def sample_motion_ids(
        self, n: int, motion_difficulty: torch.Tensor | None = None
    ) -> torch.Tensor:
        if motion_difficulty is not None:
            motion_prob = self._motion_weights * motion_difficulty
        else:
            motion_prob = self._motion_weights
        motion_ids = torch.multinomial(motion_prob, num_samples=n, replacement=True)
        return motion_ids

    def sample_motion_times(self, motion_ids: torch.Tensor) -> torch.Tensor:
        # Sample integer steps uniformly and convert to times by dividing by fps
        n_steps = self._motion_num_frames[motion_ids] - 1 - int(self.fps)
        phase = torch.rand(motion_ids.shape, device=self._device)
        steps = torch.round(phase * n_steps.float()).long()
        steps = torch.clamp(steps, min=0)  # safety
        motion_times = steps.float() / float(self.fps)
        return motion_times

    def _fetch_motion_files(
        self, motion_file: str, motion_weight: float = 1.0
    ) -> tuple[list[str], list[float]]:
        # Recursively expand YAML motion manifests into flat file and weight lists.
        if motion_file.endswith(".yaml"):
            all_files: list[str] = []
            all_weights: list[float] = []
            try:
                with open(motion_file) as f:
                    motion_config = yaml.load(f, Loader=yaml.SafeLoader)
            except Exception:
                return [], []
            motion_base_path = motion_config["root_path"]
            motion_list = motion_config["motions"]
            for motion_entry in motion_list:
                curr_file = os.path.join(motion_base_path, motion_entry["file"])
                curr_weight = float(motion_entry.get("weight", 1.0))
                assert curr_weight >= 0
                sub_files, sub_weights = self._fetch_motion_files(
                    curr_file, curr_weight * motion_weight
                )
                all_files.extend(sub_files)
                all_weights.extend(sub_weights)
            return all_files, all_weights
        else:
            return [motion_file], [motion_weight]

    def get_observed_steps(self, observed_steps: dict[str, list[int]]) -> dict[str, torch.Tensor]:
        """Convert observed steps lists into tensors on the correct device."""
        steps_map: dict[str, torch.Tensor] = {}
        for term in observed_steps.keys():
            steps_map[term] = torch.tensor(
                observed_steps[term], dtype=torch.long, device=self._device
            )
        return steps_map

    def get_motion_frame(
        self,
        motion_ids: torch.Tensor,
        motion_times: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        assert motion_times.min() >= 0.0, "motion_times must be non-negative"
        # snap to discrete frame grid using unified fps and clamp within motion length
        fps = self.fps
        motion_len = self._motion_lengths[motion_ids] - 1.0 / fps
        motion_times = torch.min(motion_times, motion_len)
        steps = torch.round(motion_times * fps).long()

        frame_start_idx = self._motion_start_idx[motion_ids]
        frame_idx = frame_start_idx + steps + 1

        base_pos = self._motion_base_pos[frame_idx]
        base_quat = self._motion_base_quat[frame_idx]
        base_lin_vel = self._motion_base_lin_vel[frame_idx]
        base_ang_vel = self._motion_base_ang_vel[frame_idx]
        dof_pos = self._motion_dof_pos[frame_idx]
        dof_vel = self._motion_dof_vel[frame_idx]

        return (
            base_pos,
            base_quat,
            base_lin_vel,
            base_ang_vel,
            dof_pos,
            dof_vel,
        )

    def get_ref_motion_frame(
        self,
        motion_ids: torch.Tensor,
        motion_times: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        assert motion_times.min() >= 0.0, "motion_times must be non-negative"
        # snap to discrete frame grid using unified fps and clamp within motion length
        fps = self.fps
        motion_len = self._motion_lengths[motion_ids] - 1.0 / fps
        motion_times = torch.min(motion_times, motion_len)
        steps = torch.round(motion_times * fps).long()

        frame_start_idx = self._motion_start_idx[motion_ids]
        frame_idx = frame_start_idx + steps + 1

        base_pos = self._motion_base_pos[frame_idx]
        base_quat = self._motion_base_quat[frame_idx]
        base_lin_vel = self._motion_base_lin_vel[frame_idx]
        base_ang_vel = self._motion_base_ang_vel[frame_idx]
        base_ang_vel_local = self._motion_base_ang_vel_local[frame_idx]
        dof_pos = self._motion_dof_pos[frame_idx]
        dof_vel = self._motion_dof_vel[frame_idx]
        link_pos_global = self._motion_link_pos_global[frame_idx]
        link_pos_local = self._motion_link_pos_local[frame_idx]
        link_quat_global = self._motion_link_quat_global[frame_idx]
        link_quat_local = self._motion_link_quat_local[frame_idx]
        link_lin_vel = self._motion_link_lin_vel[frame_idx]
        link_lin_vel_local = self._motion_link_lin_vel_local[frame_idx]
        link_ang_vel = self._motion_link_ang_vel[frame_idx]
        link_ang_vel_local = self._motion_link_ang_vel_local[frame_idx]
        foot_contact = self._motion_foot_contact[frame_idx]
        foot_contact_weighted = self._motion_foot_contact_weighted[frame_idx]

        return (
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
            link_lin_vel,
            link_lin_vel_local,
            link_ang_vel,
            link_ang_vel_local,
            foot_contact,
            foot_contact_weighted,
        )

    def get_motion_future_obs(
        self,
        motion_ids: torch.Tensor,
        motion_times: torch.Tensor,
        observed_steps: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Compute current-frame and future-step motion observations.

        Returns:
            (curr_obs_dict, future_obs_dict)
        """
        if len(observed_steps) == 0:
            return {}, {}
        assert motion_times.min() >= 0.0, "motion_times must be non-negative"
        fps = self.fps
        motion_len = self._motion_lengths[motion_ids] - 1.0 / fps
        motion_times = torch.min(motion_times, motion_len)
        steps = torch.round(motion_times * fps).long()

        frame_start_idx = self._motion_start_idx[motion_ids]
        max_steps = self._motion_num_frames[motion_ids] - 1

        # current frame (step = 0)
        curr_idx = frame_start_idx + steps
        curr_obs: dict[str, torch.Tensor] = {}
        for key in observed_steps.keys():
            tensor = getattr(self, f"_motion_{key}")
            curr_obs[key] = tensor[curr_idx]

        def gather(term: str) -> torch.Tensor:
            steps_tensor = observed_steps[term]
            future_steps = steps[:, None] + steps_tensor[None, :]
            future_steps = torch.minimum(future_steps, max_steps[:, None])
            future_idx_local = frame_start_idx[:, None] + future_steps  # (B, K)
            B, K = future_idx_local.shape
            tensor = getattr(self, f"_motion_{term}")
            flat = tensor[future_idx_local.reshape(-1)]
            return flat.reshape(B, K, *tensor.shape[1:])

        future_obs_dict: dict[str, torch.Tensor] = {}
        for key in observed_steps.keys():
            future_obs_dict[key] = gather(key)
        return curr_obs, future_obs_dict

    def get_joint_idx_by_name(self, name: str) -> int:
        return self._dof_names.index(name)

    def get_motion_length(self, motion_ids: torch.Tensor) -> torch.Tensor:
        return self._motion_lengths[motion_ids]

    def get_motion_num_frames(self, motion_ids: torch.Tensor) -> torch.Tensor:
        return self._motion_num_frames[motion_ids]

    def get_motion_weights(self, motion_ids: torch.Tensor) -> torch.Tensor:
        return self._motion_weights[motion_ids]

    @staticmethod
    def smooth(x: torch.Tensor, box_pts: int, device: torch.device) -> torch.Tensor:
        box = torch.ones(box_pts, device=device) / box_pts
        num_channels = x.shape[1]
        x_reshaped = x.T.unsqueeze(0)
        smoothed = torch.nn.functional.conv1d(
            x_reshaped,
            box.view(1, 1, -1).expand(num_channels, 1, -1),
            groups=num_channels,
            padding="same",
        )
        return smoothed.squeeze(0).T

    @property
    def tracking_link_names(self) -> list[str]:
        assert self._tracking_link_names is not None
        return self._tracking_link_names

    @property
    def dof_names(self) -> list[str]:
        return self._dof_names

    @property
    def num_motions(self) -> int:
        return self._motion_weights.shape[0]

    @property
    def motion_names(self) -> list[str]:
        return self._motion_names

    @property
    def total_length(self) -> float:
        return torch.sum(self._motion_lengths).item()

    @property
    def fps(self) -> float:
        return self._target_fps


def batched_global_to_local(base_quat: torch.Tensor, global_vec: torch.Tensor) -> torch.Tensor:
    assert base_quat.shape[0] == global_vec.shape[0]
    global_vec_shape = global_vec.shape
    global_vec = global_vec.reshape(global_vec_shape[0], -1, global_vec_shape[-1])
    B, L, D = global_vec.shape
    global_flat = global_vec.reshape(B * L, D)
    quat_rep = base_quat[:, None, :].repeat(1, L, 1).reshape(B * L, 4)
    if D == 3:
        local_flat = quat_apply(quat_inv(quat_rep), global_flat)
    elif D == 4:
        local_flat = quat_mul(quat_inv(quat_rep), global_flat)
    else:
        raise ValueError(
            f"Global vector shape must be (B, L, 3) or (B, L, 4), but got {global_flat.shape}"
        )
    return local_flat.reshape(global_vec_shape)


def build_motion_obs_from_dict(
    curr_obs: dict[str, torch.Tensor],
    future_obs: dict[str, torch.Tensor],
    envs_idx: torch.Tensor,
    tracking_link_idx_local: list[int] | None = None,
    base_quat: torch.Tensor | None = None,
) -> torch.Tensor:
    """Transform motion observations into local-yaw space and 6D rotations.

    Returns processed (curr_dict, future_dict), without touching self.ref_ variables.
    """
    B = envs_idx.shape[0]
    motion_obs_list: list[torch.Tensor] = []

    # Compute yaw quaternion from current base quat if available
    quat_yaw = quat_from_angle_axis(
        quat_to_euler(curr_obs["base_quat"])[:, -1],
        torch.tensor([0, 0, 1], device=curr_obs["base_quat"].device, dtype=torch.float),
    )

    if "base_pos" in future_obs:
        pos_diff = future_obs["base_pos"] - curr_obs["base_pos"][:, None, :]
        motion_obs_list.append(batched_global_to_local(quat_yaw, pos_diff).reshape(B, -1))
    if "base_quat" in future_obs:
        qy = quat_yaw[:, None, :].repeat(1, future_obs["base_quat"].shape[1], 1)
        base_quat_local = quat_mul(quat_inv(qy), future_obs["base_quat"])
        if base_quat is None:
            base_quat = curr_obs["base_quat"][:, None, :]
        else:
            base_quat = base_quat[:, None, :].repeat(1, future_obs["base_quat"].shape[1], 1)
        base_quat_diff = quat_mul(quat_inv(base_quat), future_obs["base_quat"])
        motion_obs_list.append(quat_to_rotation_6D(base_quat_local).reshape(B, -1))
        motion_obs_list.append(quat_to_rotation_6D(base_quat_diff).reshape(B, -1))
    if "base_lin_vel" in future_obs:
        motion_obs_list.append(
            batched_global_to_local(quat_yaw, future_obs["base_lin_vel"]).reshape(B, -1)
        )
    if "base_ang_vel" in future_obs:
        motion_obs_list.append(
            batched_global_to_local(quat_yaw, future_obs["base_ang_vel"]).reshape(B, -1)
        )
    if "base_ang_vel_local" in future_obs:
        motion_obs_list.append(future_obs["base_ang_vel_local"].reshape(B, -1))
    if "dof_pos" in future_obs:
        motion_obs_list.append(future_obs["dof_pos"].reshape(B, -1))
    if "dof_vel" in future_obs:
        motion_obs_list.append(0.1 * future_obs["dof_vel"].reshape(B, -1))
    if "link_pos_local" in future_obs:
        if tracking_link_idx_local is not None:
            tracking_link_pos_local = future_obs["link_pos_local"][:, :, tracking_link_idx_local, :]
        else:
            tracking_link_pos_local = future_obs["link_pos_local"]
        motion_obs_list.append(tracking_link_pos_local.reshape(B, -1))
    if "link_quat_local" in future_obs:
        if tracking_link_idx_local is not None:
            tracking_link_quat_local = future_obs["link_quat_local"][
                :, :, tracking_link_idx_local, :
            ]
        else:
            tracking_link_quat_local = future_obs["link_quat_local"]
        motion_obs_list.append(quat_to_rotation_6D(tracking_link_quat_local).reshape(B, -1))
    if "link_lin_vel" in future_obs:
        if tracking_link_idx_local is not None:
            tracking_link_lin_vel = future_obs["link_lin_vel"][:, :, tracking_link_idx_local, :]
        else:
            tracking_link_lin_vel = future_obs["link_lin_vel"]
        motion_obs_list.append(
            batched_global_to_local(quat_yaw, tracking_link_lin_vel).reshape(B, -1)
        )
    if "link_ang_vel" in future_obs:
        if tracking_link_idx_local is not None:
            tracking_link_ang_vel = future_obs["link_ang_vel"][:, :, tracking_link_idx_local, :]
        else:
            tracking_link_ang_vel = future_obs["link_ang_vel"]
        motion_obs_list.append(
            batched_global_to_local(quat_yaw, tracking_link_ang_vel).reshape(B, -1)
        )
    if "link_lin_vel_local_yaw" in future_obs:
        if tracking_link_idx_local is not None:
            tracking_link_lin_vel_local_yaw = future_obs["link_lin_vel_local_yaw"][
                :, :, tracking_link_idx_local, :
            ]
        else:
            tracking_link_lin_vel_local_yaw = future_obs["link_lin_vel_local_yaw"]
        motion_obs_list.append(tracking_link_lin_vel_local_yaw.reshape(B, -1))
    if "link_ang_vel_local_yaw" in future_obs:
        if tracking_link_idx_local is not None:
            tracking_link_ang_vel_local_yaw = future_obs["link_ang_vel_local_yaw"][
                :, :, tracking_link_idx_local, :
            ]
        else:
            tracking_link_ang_vel_local_yaw = future_obs["link_ang_vel_local_yaw"]
        motion_obs_list.append(tracking_link_ang_vel_local_yaw.reshape(B, -1))
    if "foot_contact" in future_obs:
        motion_obs_list.append(future_obs["foot_contact"].reshape(B, -1))

    return torch.cat(motion_obs_list, dim=-1)


def load_smplx_file(
    smplx_file: str, body_models: Any | None = None, smplx_body_model_path: str | None = None
) -> tuple[dict[str, Any], Any, Any, float]:
    import smplx

    smplx_data = np.load(smplx_file, allow_pickle=True)

    if body_models is None:
        if smplx_body_model_path is None:
            raise ValueError("smplx_body_model_path is required if body_model is not provided")
        body_model = smplx.create(
            smplx_body_model_path,
            "smplx",
            gender=str(smplx_data["gender"]),
            use_pca=False,
        )
    else:
        body_model = body_models[str(smplx_data["gender"])]

    num_frames = smplx_data["pose_body"].shape[0]

    # betas: repeat single vector to batch size
    betas = torch.tensor(smplx_data["betas"]).float().view(1, -1).repeat(num_frames, 1)  # (N, 10)

    # expression: use from file if available, otherwise zeros
    if "expression" in smplx_data:
        expression = torch.tensor(smplx_data["expression"]).float()  # (N, 10)
    else:
        expression = torch.zeros(num_frames, 10).float()

    smplx_output = body_model(
        betas=betas,  # (N, 10)
        global_orient=torch.tensor(smplx_data["root_orient"]).float(),  # (N, 3)
        body_pose=torch.tensor(smplx_data["pose_body"]).float(),  # (N, 63)
        transl=torch.tensor(smplx_data["trans"]).float(),  # (N, 3)
        left_hand_pose=torch.zeros(num_frames, 45).float(),
        right_hand_pose=torch.zeros(num_frames, 45).float(),
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        expression=expression,  # (N, 10)
        return_full_pose=True,
    )

    if len(smplx_data["betas"].shape) == 1:
        human_height = 1.66 + 0.1 * smplx_data["betas"][0]
    else:
        human_height = 1.66 + 0.1 * smplx_data["betas"][0, 0]

    return smplx_data, body_model, smplx_output, human_height


def load_smplx_data_frames(
    smplx_data: dict[str, Any],
    body_model: Any,
    smplx_output: Any,
    tgt_fps: int = 30,
) -> tuple[list[dict[str, Any]], int]:
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    from scipy.interpolate import interp1d
    from smplx.joint_names import JOINT_NAMES

    src_fps = smplx_data["mocap_frame_rate"].item()
    frame_skip = int(src_fps / tgt_fps)
    num_frames = smplx_data["pose_body"].shape[0]
    global_orient = smplx_output.global_orient.squeeze()
    full_body_pose = smplx_output.full_pose.reshape(num_frames, -1, 3)
    joints = smplx_output.joints.detach().numpy().squeeze()
    joint_names = JOINT_NAMES[: len(body_model.parents)]
    parents = body_model.parents

    if tgt_fps < src_fps:
        # perform fps alignment with proper interpolation
        new_num_frames = num_frames // frame_skip

        # Create time points for interpolation
        original_time = np.arange(num_frames)
        target_time = np.linspace(0, num_frames - 1, new_num_frames)

        # Interpolate global orientation using SLERP
        global_orient_interp = []
        for i in range(len(target_time)):
            t = target_time[i]
            idx1 = int(np.floor(t))
            idx2 = min(idx1 + 1, num_frames - 1)
            alpha = t - idx1

            q1 = R.from_rotvec(global_orient[idx1]).as_quat(scalar_first=True)
            q2 = R.from_rotvec(global_orient[idx2]).as_quat(scalar_first=True)
            q = slerp(torch.tensor(q1), torch.tensor(q2), torch.tensor(alpha))
            interp_rot = R.from_quat(q, scalar_first=True)
            global_orient_interp.append(interp_rot.as_rotvec())
        global_orient = np.stack(global_orient_interp, axis=0)

        # Interpolate full body pose using SLERP
        full_body_pose_interp = []
        for i in range(full_body_pose.shape[1]):  # For each joint
            joint_rots = []
            for j in range(len(target_time)):
                t = target_time[j]
                idx1 = int(np.floor(t))
                idx2 = min(idx1 + 1, num_frames - 1)
                alpha = t - idx1

                rot1 = R.from_rotvec(full_body_pose[idx1, i])
                rot2 = R.from_rotvec(full_body_pose[idx2, i])
                q1 = rot1.as_quat(scalar_first=True)
                q2 = rot2.as_quat(scalar_first=True)
                q = slerp(torch.tensor(q1), torch.tensor(q2), torch.tensor(alpha))
                interp_rot = R.from_quat(q, scalar_first=True)
                joint_rots.append(interp_rot.as_rotvec())
            full_body_pose_interp.append(np.stack(joint_rots, axis=0))
        full_body_pose = np.stack(full_body_pose_interp, axis=1)

        # Interpolate joint positions using linear interpolation
        joints_interp = []
        for i in range(joints.shape[1]):  # For each joint
            for j in range(3):  # For each coordinate
                interp_func = interp1d(original_time, joints[:, i, j], kind="linear")
                joints_interp.append(interp_func(target_time))
        joints = np.stack(joints_interp, axis=1).reshape(new_num_frames, -1, 3)

        aligned_fps = len(global_orient) / num_frames * src_fps
    else:
        aligned_fps = tgt_fps

    smplx_data_frames = []
    for curr_frame in range(len(global_orient)):
        result = {}
        single_global_orient = global_orient[curr_frame]
        single_full_body_pose = full_body_pose[curr_frame]
        single_joints = joints[curr_frame]
        joint_orientations = []
        for i, joint_name in enumerate(joint_names):
            if i == 0:
                rot = R.from_rotvec(single_global_orient)
            else:
                rot = joint_orientations[parents[i]] * R.from_rotvec(
                    single_full_body_pose[i].squeeze()
                )
            joint_orientations.append(rot)
            result[joint_name] = (single_joints[i], rot.as_quat(scalar_first=True))

        smplx_data_frames.append(result)

    return smplx_data_frames, aligned_fps


class GeneralMotionRetargeting:
    """General Motion Retargeting (GMR)."""

    from rich import print

    def __init__(
        self,
        robot_xml_file: str,
        ik_config_file: str,
        actual_human_height: float = 1.7,
        solver: str = "daqp",  # change from "quadprog" to "daqp".
        damping: float = 5e-1,  # change from 1e-1 to 1e-2.
        verbose: bool = True,
        use_velocity_limit: bool = False,
        aligned_fps: float = 30,
        contact_filter: bool = False,
    ) -> None:
        # load the robot model
        self.model = mj.MjModel.from_xml_path(robot_xml_file)
        if verbose:
            print("Use robot model: ", robot_xml_file)

        # Load the IK config
        with open(ik_config_file) as f:
            ik_config = json.load(f)
        if verbose:
            print("Use IK config: ", ik_config_file)

        # Print DoF names in order
        print("[GMR] Robot Degrees of Freedom (DoF) names and their order:")
        self.robot_dof_names = {}
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            self.robot_dof_names[dof_name] = i
            if verbose:
                print(f"DoF {i}: {dof_name}")

        print("[GMR] Robot Body names and their IDs:")
        self.robot_body_names = {}
        for i in range(self.model.nbody):  # 'nbody' is the number of bodies
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
            if verbose:
                print(f"Body ID {i}: {body_name}")

        print("[GMR] Robot Motor (Actuator) names and their IDs:")
        self.robot_motor_names = {}
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i
            if verbose:
                print(f"Motor ID {i}: {motor_name}")

        # compute the scale ratio based on given human height and the assumption in the IK config
        ratio = actual_human_height / ik_config["human_height_assumption"]

        # adjust the human scale table
        for key in ik_config["human_scale_table"].keys():
            ik_config["human_scale_table"][key] = ik_config["human_scale_table"][key] * ratio

        # used for retargeting
        self.ik_match_tables = ik_config["ik_match_tables"]
        self.num_ik_match_tables = len(self.ik_match_tables)
        self.human_root_name = ik_config["human_root_name"]
        self.robot_root_name = ik_config["robot_root_name"]
        self.human_scale_table = ik_config["human_scale_table"]
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])

        self.max_iter = 20

        self.solver = solver
        self.damping = damping

        self.human_body_to_tasks = [{} for _ in range(self.num_ik_match_tables)]
        self.pos_offsets = [{} for _ in range(self.num_ik_match_tables)]
        self.rot_offsets = [{} for _ in range(self.num_ik_match_tables)]
        self.task_errors = [{} for _ in range(self.num_ik_match_tables)]
        self.all_final_errors = [[] for _ in range(self.num_ik_match_tables)]

        self.foot_last_pos = None
        self.foot_contact_list = []
        self.robot_foot_last_pos = None
        self.dt = 1.0 / aligned_fps

        self.contact_filter = contact_filter

        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        if use_velocity_limit:
            VELOCITY_LIMITS = {k: 3 * np.pi for k in self.robot_motor_names.keys()}
            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS))

        self.g1_hack = False
        self.setup_retarget_configuration()

        self.ground_offset = 0.0

    def setup_retarget_configuration(self) -> None:
        self.configuration = mink.Configuration(self.model)

        if self.g1_hack:
            # hacked initial pose for g1
            q = self.configuration.q
            q[7 + 0] = -0.5
            q[7 + 6] = -0.5
            q[7 + 3] = 0.5
            q[7 + 9] = 0.5
            self.configuration.update(q=q)

        self.tasks = [[] for _ in range(self.num_ik_match_tables)]
        self.task_errors = [{} for _ in range(self.num_ik_match_tables)]

        for i in range(self.num_ik_match_tables):
            for frame_name, entry in self.ik_match_tables[i].items():
                body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
                if pos_weight != 0 or rot_weight != 0:
                    task = mink.FrameTask(
                        frame_name=frame_name,
                        frame_type="body",
                        position_cost=pos_weight,
                        orientation_cost=rot_weight,
                        lm_damping=1,
                    )
                    self.human_body_to_tasks[i][body_name] = task
                    self.pos_offsets[i][body_name] = np.array(pos_offset) - self.ground
                    self.rot_offsets[i][body_name] = R.from_quat(rot_offset, scalar_first=True)
                    self.tasks[i].append(task)
                    self.task_errors[i][task] = []

    def process_human_data(
        self, human_data: dict[str, Any], offset_to_ground: bool = False
    ) -> dict[str, Any]:
        # scale human data in local frame
        human_data = self.to_numpy(human_data)
        human_data = self.scale_human_data(human_data, self.human_root_name, self.human_scale_table)
        human_data = self.offset_human_data(human_data, self.pos_offsets[0], self.rot_offsets[0])
        human_data = self.apply_ground_offset(human_data)

        return human_data

    def retarget(self, scaled_human_data: dict[str, Any]) -> np.ndarray:
        if self.g1_hack:
            # hacked initial pose for g1
            q = self.configuration.q
            # should yaw
            q[7 + 17] = 0.0
            q[7 + 24] = 0.0
            # elbow
            q[7 + 18] = 0.5
            q[7 + 25] = 0.5
            # wrist pitch
            q[7 + 20] = 0.0
            q[7 + 27] = 0.0
            self.configuration.update(q=q)

        for i in range(self.num_ik_match_tables):
            for body_name in self.human_body_to_tasks[i].keys():
                task = self.human_body_to_tasks[i][body_name]
                pos, rot = scaled_human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))

        for i in range(self.num_ik_match_tables):
            curr_error = self.error(i)
            next_error = curr_error - 0.1
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                dt = self.configuration.model.opt.timestep
                vel = mink.solve_ik(
                    self.configuration, self.tasks[i], dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel, dt)
                self.clamp_q_to_limits()
                next_error = self.error(i)
                num_iter += 1
            self.all_final_errors[i].append(next_error)

        return self.configuration.data.qpos.copy()

    def clamp_q_to_limits(self) -> None:
        q = self.configuration.q

        lb = self.model.jnt_range[1:, 0]
        ub = self.model.jnt_range[1:, 1]
        jnt_pos = q[7:]
        jnt_pos = np.clip(jnt_pos, lb, ub)
        q[7:] = jnt_pos

        self.configuration.update(q=q)

    def reset_error_tracking(self) -> None:
        """Reset all error tracking lists."""
        self.all_final_errors = [[] for _ in range(self.num_ik_match_tables)]

    def error(self, i: int) -> float:
        return np.linalg.norm(
            np.concatenate([task.compute_error(self.configuration) for task in self.tasks[i]])
        )

    def to_numpy(self, human_data: dict[str, Any]) -> dict[str, Any]:
        for body_name in human_data.keys():
            human_data[body_name] = [
                np.asarray(human_data[body_name][0]),
                np.asarray(human_data[body_name][1]),
            ]
        return human_data

    @staticmethod
    def scale_human_data(
        human_data: dict[str, Any], human_root_name: str, human_scale_table: dict[str, Any]
    ) -> dict[str, Any]:
        human_data_local = {}
        root_pos, root_quat = human_data[human_root_name]

        # scale root
        scaled_root_pos = human_scale_table[human_root_name] * root_pos

        # scale other body parts in local frame
        for body_name in human_data.keys():
            if body_name not in human_scale_table:
                continue
            if body_name == human_root_name:
                continue
            else:
                # transform to local frame (only position)
                human_data_local[body_name] = (
                    human_data[body_name][0] - root_pos
                ) * human_scale_table[body_name]

        # transform the human data back to the global frame
        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (
                human_data_local[body_name] + scaled_root_pos,
                human_data[body_name][1],
            )

        return human_data_global

    @staticmethod
    def offset_human_data(
        human_data: dict[str, Any], pos_offsets: dict[str, Any], rot_offsets: dict[str, Any]
    ) -> dict[str, Any]:
        """the pos offsets are applied in the local frame"""
        offset_human_data = {}
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            # apply rotation offset first
            updated_quat = (R.from_quat(quat, scalar_first=True) * rot_offsets[body_name]).as_quat(
                scalar_first=True
            )
            offset_human_data[body_name][1] = updated_quat

            local_offset = pos_offsets[body_name]
            # compute the global position offset using the updated rotation
            global_pos_offset = R.from_quat(updated_quat, scalar_first=True).apply(local_offset)

            offset_human_data[body_name][0] = pos + global_pos_offset

        return offset_human_data

    def set_ground_offset(self, ground_offset: float) -> None:
        self.ground_offset = ground_offset

    def apply_ground_offset(self, human_data: dict[str, Any]) -> dict[str, Any]:
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            human_data[body_name][0] = pos - np.array([0, 0, self.ground_offset])
        return human_data


class G1Retargeter:
    """Stateful retargeter: calibrates once, then retargets frames into robot-space motion."""

    LINK_ORDER = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "torso_link",
        "pelvis",
    )

    def __init__(self, joint_space_retarget: bool = False) -> None:
        self.frame_rate = 120.0

        # Indices into the 6-link tensors (fixed order above)
        self.l_foot_idx = 0
        self.r_foot_idx = 1
        self.l_hand_idx = 2
        self.r_hand_idx = 3
        self.torso_idx = 4
        self.base_idx = 5

        # Local rotation & global xy-plane transform
        self.motion_quat_inv = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(6, 1)
        self.global_yaw_inv = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.global_xy = torch.tensor([0.0, 0.0])

        # G1 physical parameters
        arm_scale = 0.95
        leg_scale = 0.97
        self.g1_shoulder_y = 0.100
        self.g1_arm_length = 0.419 * arm_scale
        self.g1_pelvis_shoulder_z = 1.082 - 0.793
        self.g1_pelvis_torso_z = 0.837 - 0.793
        self.g1_pelvis_z = 0.793 * leg_scale
        self.g1_shoulder_anchor = torch.tensor(
            [
                [0.0, self.g1_shoulder_y, self.g1_pelvis_shoulder_z],
                [0.0, -self.g1_shoulder_y, self.g1_pelvis_shoulder_z],
            ],
            dtype=torch.float32,
        )
        self.g1_leg_y = 0.1185
        self.g1_leg_z = 0.05

        # Calibrated physical parameters
        self.aug_pelvis_z = self.g1_pelvis_z * 1.0
        self.aug_arm_length = torch.tensor([self.g1_arm_length, self.g1_arm_length]) * 1.0
        self.aug_shoulder_anchor = self.g1_shoulder_anchor.clone()
        self.foot_offset = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        self._calibrated = False

        # Estimate torso orientation (vr approach only)
        self.estimate_torso_quat = False

        # Velocity
        self.vel_ema_alpha = 0.25

        self.prev_frame_id: int = -1
        self.prev_tracked_pos: torch.Tensor
        self.prev_tracked_quat: torch.Tensor

        self.ema_base_lin_vel = torch.zeros(3)
        self.ema_base_ang_vel = torch.zeros(3)
        self.ema_base_ang_vel_local = torch.zeros(3)
        self.ema_link_lin_vel = torch.zeros(6, 3)
        self.ema_link_ang_vel = torch.zeros(6, 3)

        # GMR
        self.joint_space_retarget = joint_space_retarget
        if self.joint_space_retarget:
            from gs_env.common.utils.motion_utils import GeneralMotionRetargeting

            self.joint_space_retargeter = GeneralMotionRetargeting(
                robot_xml_file="assets/robot/unitree_g1/g1_mocap_29dof.xml",
                ik_config_file="assets/robot/unitree_g1/optitrack_to_g1.json",
                aligned_fps=60.0,
            )
            self.joint_space_retargeter.g1_hack = True

        self._enable_timer = False
        self._timer_records: dict[str, list[float]] = {}

    def start_timer(self, reset: bool = True) -> None:
        """Enable internal timing records for key sub-steps.

        Args:
            reset: If True, clear previous timing records.
        """
        self._enable_timer = True
        if reset:
            self._timer_records = {}

    def _record_timer(self, name: str, duration_s: float) -> None:
        if not self._enable_timer:
            return
        self._timer_records.setdefault(name, []).append(float(duration_s))

    def stop_timer(self) -> dict[str, dict[str, float | int]]:
        """Disable timing and return aggregated statistics.

        Returns:
            Dict keyed by timer name (plus 'overall'), each containing:
            {'avg': float, 'var': float, 'max': float, 'p99': float, 'p95': float,
             'p90': float, 'p50': float, 'count': int}
        """
        self._enable_timer = False

        def _percentile(sorted_xs: list[float], q: float) -> float:
            """Compute percentile with linear interpolation (like numpy default)."""
            n = len(sorted_xs)
            if n == 0:
                return 0.0
            if n == 1:
                return float(sorted_xs[0])
            q = max(0.0, min(100.0, float(q)))
            # Type-7 method: h = (n - 1) * q + 1, 1-indexed
            # Equivalent 0-indexed position:
            pos = (n - 1) * (q / 100.0)
            lo = int(pos)
            hi = min(lo + 1, n - 1)
            w = pos - lo
            return float(sorted_xs[lo] * (1.0 - w) + sorted_xs[hi] * w)

        def _stats(xs: list[float]) -> dict[str, float | int]:
            if len(xs) == 0:
                return {
                    "avg": 0.0,
                    "var": 0.0,
                    "max": 0.0,
                    "p99": 0.0,
                    "p95": 0.0,
                    "p90": 0.0,
                    "p50": 0.0,
                    "count": 0,
                }
            xs_sorted = sorted(xs)
            avg = float(statistics.fmean(xs))
            var = float(statistics.pvariance(xs)) if len(xs) > 1 else 0.0
            mx = float(xs_sorted[-1])
            return {
                "avg": avg,
                "var": var,
                "max": mx,
                "p99": _percentile(xs_sorted, 99.0),
                "p95": _percentile(xs_sorted, 95.0),
                "p90": _percentile(xs_sorted, 90.0),
                "p50": _percentile(xs_sorted, 50.0),
                "count": len(xs_sorted),
            }

        out: dict[str, dict[str, float | int]] = {}
        all_xs: list[float] = []
        for k, xs in self._timer_records.items():
            out[k] = _stats(xs)
            all_xs.extend(xs)
        out["overall"] = _stats(all_xs)
        return out

    def _ema(self, prev: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        a = self.vel_ema_alpha
        return (1.0 - a) * prev + a * x

    def _reorient_quat(self, quat_local: torch.Tensor, idxs: list[int]) -> torch.Tensor:
        quat_local[idxs] = quat_mul(quat_local[idxs], self.motion_quat_inv[idxs])
        return quat_local

    def _apply_yaw_inv(
        self, pos: torch.Tensor, quat: torch.Tensor, yaw_inv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        yaw_inv = yaw_inv.view(1, 4).repeat(pos.shape[0], 1)
        link_pos_global = quat_apply(yaw_inv, pos)
        link_quat_global = quat_mul(yaw_inv, quat)
        return link_pos_global, link_quat_global

    def _localize(
        self, pos: torch.Tensor, quat: torch.Tensor, keep_height: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        link_pos_global = pos
        link_quat_global = quat
        base_pos = link_pos_global[self.base_idx, :]
        base_quat = link_quat_global[self.base_idx, :]
        relative_link_pos_global = link_pos_global.clone()
        if keep_height:
            relative_link_pos_global[:, :2] -= base_pos[:2]
        else:
            relative_link_pos_global -= base_pos
        base_euler = quat_to_euler(base_quat)
        base_euler[0] = 0.0
        base_euler[1] = 0.0
        inv_yaw = quat_from_euler(-base_euler)
        link_pos_local, link_quat_local = self._apply_yaw_inv(
            relative_link_pos_global, link_quat_global, inv_yaw
        )
        return link_pos_local, link_quat_local

    def calibrate(
        self,
        tracked_pos: torch.Tensor,
        tracked_quat: torch.Tensor,
    ) -> None:
        ### Global
        base_quat = tracked_quat[self.base_idx]
        base_euler = quat_to_euler(base_quat)
        base_euler[0] = 0.0
        base_euler[1] = 0.0
        yaw = quat_from_euler(base_euler)
        self.global_yaw_inv = quat_inv(yaw)
        self.global_xy = tracked_pos[self.base_idx, :2].clone()
        ### Local
        tracked_pos, tracked_quat = self._localize(tracked_pos, tracked_quat)
        self.motion_quat_inv = quat_inv(tracked_quat)
        ### Scale
        left_hand_pos = tracked_pos[self.l_hand_idx]
        right_hand_pos = tracked_pos[self.r_hand_idx]
        self.aug_arm_length = torch.stack([left_hand_pos[0], right_hand_pos[0]])
        aug_l_shoulder_y = left_hand_pos[1].item()
        aug_r_shoulder_y = right_hand_pos[1].item()
        aug_l_shoulder_z = left_hand_pos[2].item()
        aug_r_shoulder_z = right_hand_pos[2].item()
        self.aug_pelvis_z = tracked_pos[self.base_idx, 2].item()
        self.aug_shoulder_anchor = torch.tensor(
            [
                [0.0, aug_l_shoulder_y, aug_l_shoulder_z - self.aug_pelvis_z],
                [0.0, aug_r_shoulder_y, aug_r_shoulder_z - self.aug_pelvis_z],
            ],
            dtype=torch.float32,
        )
        left_leg_pos = tracked_pos[self.l_foot_idx]
        right_leg_pos = tracked_pos[self.r_foot_idx]
        left_leg_pos_scaled = left_leg_pos * (self.g1_pelvis_z / self.aug_pelvis_z)
        right_leg_pos_scaled = right_leg_pos * (self.g1_pelvis_z / self.aug_pelvis_z)
        self.foot_offset[0, :] = (
            torch.tensor([0.0, self.g1_leg_y, self.g1_leg_z]) - left_leg_pos_scaled[:]
        )
        self.foot_offset[1, :] = (
            torch.tensor([0.0, -self.g1_leg_y, self.g1_leg_z]) - right_leg_pos_scaled[:]
        )
        self._calibrated = True
        print("Calibration result:")
        print(f"  - Pelvis Z: {self.aug_pelvis_z:.3f}")
        print(f"  - Arm Length (Mean): {self.aug_arm_length.mean().item():.3f}")
        print(f"  - Shoulder Y (Mean): {self.aug_shoulder_anchor[:, 1].abs().mean().item():.3f}")
        print(f"  - Shoulder Z (Mean): {(self.aug_shoulder_anchor[:, 2].mean().item()):.3f}")

    def cartesian_mapping(
        self, tracked_pos: torch.Tensor, tracked_quat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Local re-orientation
        tracked_quat = self._reorient_quat(tracked_quat, list(range(6)))
        # Global
        tracked_pos[:, :2] = tracked_pos[:, :2] - self.global_xy
        tracked_pos, tracked_quat = self._apply_yaw_inv(
            tracked_pos, tracked_quat, self.global_yaw_inv
        )
        # Torso quat estimation
        if self.estimate_torso_quat:
            p = quat_apply(
                quat_inv(tracked_quat[self.base_idx]),
                tracked_pos[self.torso_idx] - tracked_pos[self.base_idx],
            )
            d = p / (torch.linalg.norm(p) + 1e-6)
            yaw = torch.tensor(0.0)
            pitch = torch.atan2(d[0], d[2])
            roll = -torch.atan2(d[1], torch.sqrt(d[0] ** 2 + d[2] ** 2))
            # intrinsic ypr == extrinsic rpy
            q = quat_from_euler(torch.stack([roll, pitch, yaw]))
            tracked_quat[self.torso_idx] = quat_mul(tracked_quat[self.base_idx], q)
        # EE scaling (arm use base frame + torso rotation)
        _ee_idxs = [self.l_hand_idx, self.r_hand_idx]
        _ee_base_pos_idxs = [self.base_idx] * 2
        _ee_base_quat_idxs = [self.torso_idx] * 2
        p, q = pose_diff_quat(
            tracked_pos[_ee_base_pos_idxs + [self.base_idx]],
            tracked_quat[_ee_base_quat_idxs + [self.base_idx]],
            tracked_pos[_ee_idxs + [self.torso_idx]],
            tracked_quat[_ee_idxs + [self.torso_idx]],
        )
        hand_pos_local, hand_quat_local = p[0:2], q[0:2]
        hand_pos_local = (hand_pos_local - self.aug_shoulder_anchor) * (
            self.g1_arm_length / self.aug_arm_length.view(2, 1)
        )
        # hand height scaling
        hand_pos_local[:, 2] *= 0.85
        hand_pos_local = self.g1_shoulder_anchor + hand_pos_local
        # Base/foot scaling
        _lower_idxs = [self.base_idx, self.l_foot_idx, self.r_foot_idx]
        tracked_pos[_lower_idxs] = tracked_pos[_lower_idxs] * (
            self.g1_pelvis_z / (self.aug_pelvis_z + 1e-6)
        )
        # Torso replacing
        torso_pos_local = torch.tensor(
            [[0.0, 0.0, self.g1_pelvis_torso_z]],
            dtype=torch.float32,
        )
        torso_quat_local = q[2:3]
        # Update back
        p, q = pose_mul_quat(
            tracked_pos[_ee_base_pos_idxs + [self.base_idx]],
            tracked_quat[_ee_base_quat_idxs + [self.base_idx]],
            torch.cat(
                [
                    hand_pos_local,
                    torso_pos_local,
                ],
                dim=0,
            ),
            torch.cat(
                [
                    hand_quat_local,
                    torso_quat_local,
                ],
                dim=0,
            ),
        )
        tracked_pos[_ee_idxs + [self.torso_idx]] = p[:]
        tracked_quat[_ee_idxs + [self.torso_idx]] = q[:]
        # Foot xy
        p = tracked_pos[[self.l_foot_idx, self.r_foot_idx]] + quat_apply(
            tracked_quat[[self.l_foot_idx, self.r_foot_idx]],
            self.foot_offset,
        )
        tracked_pos[self.l_foot_idx] = p[0]
        tracked_pos[self.r_foot_idx] = p[1]
        return tracked_pos, tracked_quat

    def step(
        self, tracked_pos: torch.Tensor, tracked_quat: torch.Tensor, frame_id: int
    ) -> dict[str, torch.Tensor]:
        if not self._calibrated:
            self.calibrate(tracked_pos, tracked_quat)

        if self._enable_timer:
            start_time = time.perf_counter()
            tracked_pos, tracked_quat = self.cartesian_mapping(tracked_pos, tracked_quat)
            self._record_timer("cartesian_mapping", time.perf_counter() - start_time)
        else:
            tracked_pos, tracked_quat = self.cartesian_mapping(tracked_pos, tracked_quat)

        base_pos = tracked_pos[self.base_idx]
        base_quat = tracked_quat[self.base_idx]

        dof_pos = torch.zeros(29, device=tracked_pos.device)
        if self.joint_space_retarget:
            human_data = {}
            for i, link_name in enumerate(self.LINK_ORDER):
                link_pos = tracked_pos[i].detach().cpu().numpy()
                link_quat = tracked_quat[i].detach().cpu().numpy()
                human_data[link_name] = (link_pos, link_quat)

            if self._enable_timer:
                start_time = time.perf_counter()
                qpos = self.joint_space_retargeter.retarget(human_data)
                self._record_timer("joint_space_retargeting", time.perf_counter() - start_time)
            else:
                qpos = self.joint_space_retargeter.retarget(human_data)
            qpos_t = torch.from_numpy(qpos).to(tracked_pos.device).float()
            base_pos = qpos_t[0:3]
            base_quat = qpos_t[3:7]
            dof_pos = qpos_t[7:]

        # Localize
        tracked_pos_local, tracked_quat_local = self._localize(
            tracked_pos, tracked_quat, keep_height=False
        )

        if self.prev_frame_id != -1:
            df = frame_id - self.prev_frame_id
            if df > 0:
                dt = df / self.frame_rate
                link_lin_vel_raw = (tracked_pos - self.prev_tracked_pos) / dt
                base_lin_vel_raw = link_lin_vel_raw[self.base_idx]
                q_delta = quat_diff(tracked_quat, self.prev_tracked_quat)
                axis_angle = quat_to_angle_axis(q_delta)
                link_ang_vel_raw = axis_angle / dt
                base_ang_vel_raw = link_ang_vel_raw[self.base_idx]

                base_q = tracked_quat[self.base_idx]
                base_ang_vel_local_raw = quat_apply(quat_inv(base_q), base_ang_vel_raw)

                self.ema_link_lin_vel = self._ema(self.ema_link_lin_vel, link_lin_vel_raw)
                self.ema_base_lin_vel = self._ema(self.ema_base_lin_vel, base_lin_vel_raw)
                self.ema_link_ang_vel = self._ema(self.ema_link_ang_vel, link_ang_vel_raw)
                self.ema_base_ang_vel = self._ema(self.ema_base_ang_vel, base_ang_vel_raw)
                self.ema_base_ang_vel_local = self._ema(
                    self.ema_base_ang_vel_local, base_ang_vel_local_raw
                )

        self.prev_tracked_pos = tracked_pos.detach().clone()
        self.prev_tracked_quat = tracked_quat.detach().clone()
        self.prev_frame_id = frame_id

        return {
            "base_pos": base_pos,
            "base_quat": base_quat,
            "link_pos_local": tracked_pos_local,
            "link_quat_local": tracked_quat_local,
            "base_lin_vel": self.ema_base_lin_vel,
            "base_ang_vel": self.ema_base_ang_vel,
            "base_ang_vel_local": self.ema_base_ang_vel_local,
            "link_lin_vel": self.ema_link_lin_vel,
            "link_ang_vel": self.ema_link_ang_vel,
            "dof_pos": dof_pos,
        }

    @property
    def calibrated(self) -> bool:
        return self._calibrated
