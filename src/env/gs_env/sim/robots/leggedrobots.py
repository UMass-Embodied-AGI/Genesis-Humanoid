from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import genesis as gs
import numpy as np
import torch
from gymnasium import spaces

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidEntity
    from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver

from gs_env.common.bases.base_robot import BaseGymRobot
from gs_env.common.utils.asset_utils import resolve_file_path
from gs_env.common.utils.math_utils import quat_from_euler
from gs_env.sim.robots.config.schema import (
    BaseAction,
    CtrlType,
    DRJointPosAction,
    DRJointPosVelAction,
    HumanoidRobotArgs,
    LeggedRobotArgs,
    QuadrupedRobotArgs,
)


class LeggedRobotBase(BaseGymRobot):
    """
    Base class for legged robots
    """

    def __init__(
        self,
        num_envs: int,
        scene: gs.Scene,
        args: LeggedRobotArgs,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        # == set members ==
        self._device = device
        self._scene = scene
        self._num_envs = num_envs
        self._args = args

        # == Genesis configurations ==
        material = gs.materials.Rigid(**args.material_args.model_dump())
        morph_args = args.morph_args.model_copy(
            update={"file": resolve_file_path(args.morph_args.file)}
        )
        morph = gs.morphs.URDF(**morph_args.model_dump())
        self._robot: RigidEntity = scene.add_entity(  # type: ignore
            material=material,
            morph=morph,
            visualize_contact=args.visualize_contact,
            vis_mode=args.vis_mode,
        )

        # == action space ==
        n_dof = len(args.dof_names)
        self._action_space = spaces.Box(shape=(n_dof,), low=-np.inf, high=np.inf)
        assert self._action_space is not None, "Action space cannot be None"

        # == some buffer initialization ==
        self._init()

    def _init(self) -> None:
        self._dof_dim = len(self._args.dof_names)  # total number of joints

        #
        self._dof_idx = torch.arange(self._dof_dim, device=self._device)

        #
        self.body_link = self._robot.get_link(self._args.body_link_name)
        self.body_link_idx = self.body_link.idx_local
        self.foot_links = [self._robot.get_link(name) for name in self._args.foot_link_names]
        self.foot_links_idx = [link.idx_local for link in self.foot_links]

        #
        self._dofs_idx_local = [
            self._robot.get_joint(name).dofs_idx_local[0] for name in self.dof_names
        ]
        dof_kp, dof_kd = [], []
        for dof_name in self.dof_names:
            for key in self._args.dof_kp.keys():
                if key in dof_name:
                    dof_kp.append(self._args.dof_kp[key])
                    dof_kd.append(self._args.dof_kd[key])
        dof_feed_forward_ratio = []
        if isinstance(self._args.feed_forward_ratio, dict):
            for dof_name in self.dof_names:
                for key in self._args.feed_forward_ratio.keys():
                    if key in dof_name:
                        dof_feed_forward_ratio.append(self._args.feed_forward_ratio[key])
        else:
            dof_feed_forward_ratio = [self._args.feed_forward_ratio] * self._dof_dim
        for joint_name in self._args.indirect_drive_joint_names:
            for i, dof_name in enumerate(self.dof_names):
                if joint_name in dof_name:
                    dof_feed_forward_ratio[i] = 0.0
        self._dof_kp = torch.tensor(dof_kp, device=self._device)
        self._dof_kd = torch.tensor(dof_kd, device=self._device)
        self._dof_feed_forward_ratio = torch.tensor(dof_feed_forward_ratio, device=self._device)[
            None, :
        ]
        self._batched_dof_kp = self._dof_kp[None, :].repeat(self._num_envs, 1)
        self._batched_dof_kd = self._dof_kd[None, :].repeat(self._num_envs, 1)
        self._torque = torch.zeros((self._num_envs, self._dof_dim), device=self._device)
        self._dof_armature = None
        if self._args.dof_armature is not None:
            dof_armature = []
            for dof_name in self.dof_names:
                for key in self._args.dof_armature.keys():
                    if key in dof_name:
                        dof_armature.append(self._args.dof_armature[key])
            self._dof_armature = torch.tensor(dof_armature, device=self._device)

        #
        self.direct_drive_mask = torch.ones((self._dof_dim,), device=self._device)

        #
        self._kp_ratio = torch.ones(self._num_envs, self._dof_dim, device=self._device)
        self._kd_ratio = torch.ones(self._num_envs, self._dof_dim, device=self._device)
        self._motor_strength = torch.ones(
            (self._num_envs, self._dof_dim), device=self._device
        )  # motor strength scaling factor
        self._motor_offset = torch.zeros(
            (self._num_envs, self._dof_dim), device=self._device
        )  # motor offset
        self._friction_ratio = torch.ones(self._num_envs, 1, device=self._device)
        self._added_mass = torch.zeros(self._num_envs, 1, device=self._device)
        self._com_displacement = torch.zeros(self._num_envs, 3, device=self._device)

        #
        if self._args.external_force_links_idx is not None:
            self._external_force_links_idx = [
                self._robot.get_link(name).idx for name in self._args.external_force_links_idx
            ]
        else:
            self._external_force_links_idx = []
        num_external_force_links = len(self._external_force_links_idx)
        if self._args.dr_args.external_force_range is not None:
            self._external_force_range = torch.tensor(
                self._args.dr_args.external_force_range, device=self._device
            )[None, None, :].repeat(self._num_envs, num_external_force_links, 1)
        else:
            self._external_force_range = torch.zeros(
                self._num_envs, num_external_force_links, 3, device=self._device
            )
        if self._args.dr_args.external_torque_range is not None:
            self._external_torque_range = torch.tensor(
                self._args.dr_args.external_torque_range, device=self._device
            )[None, None, :].repeat(self._num_envs, num_external_force_links, 1)
        else:
            self._external_torque_range = torch.zeros(
                self._num_envs, num_external_force_links, 3, device=self._device
            )
        self._external_force = torch.zeros(
            self._num_envs, len(self._external_force_links_idx), 3, device=self._device
        )
        self._external_torque = torch.zeros(
            self._num_envs, len(self._external_force_links_idx), 3, device=self._device
        )
        self._steps_since_randomize_external_force = 0
        self._steps_to_randomize_external_force = 800

        # default states
        self._default_pos = torch.tensor(
            self._args.morph_args.pos, dtype=torch.float32, device=self._device
        )
        self._default_euler = torch.tensor(
            self._args.morph_args.euler, dtype=torch.float32, device=self._device
        )
        self._default_quat = quat_from_euler(self._default_euler)
        default_joint_angles = [self._args.default_dof_pos[name] for name in self.dof_names]
        self._default_dof_pos = torch.tensor(
            default_joint_angles, dtype=torch.float32, device=self._device
        )
        # buffers
        self._dof_pos = torch.zeros(
            (self._num_envs, self._dof_dim), dtype=torch.float32, device=self._device
        )
        self._dof_vel = torch.zeros(
            (self._num_envs, self._dof_dim), dtype=torch.float32, device=self._device
        )

        # == set up control dispatch ==
        self._dispatch: dict[CtrlType, Callable[[BaseAction], None]] = {  # type: ignore
            CtrlType.DR_JOINT_POSITION.value: self._apply_dr_joint_pos,
            CtrlType.DR_JOINT_POSITION_VELOCITY.value: self._apply_dr_joint_pos_vel,
        }
        self.feed_forward_ratio = self._args.feed_forward_ratio

        # == set up dof pos logger ==
        self._target_dof_pos_history = []
        self._dof_pos_history = []
        self._dof_vel_history = []
        self._logging_time_stamp = []
        self._time_stamp = 0.0
        self._logging = False

        self._steps_since_randomize_pds = 0
        self._steps_to_randomize_pds = 40

    def post_build_init(self, eval_mode: bool = False) -> None:
        self._mass = self._robot.get_mass()

        if not eval_mode:
            self._init_domain_randomization()

        if self._dof_armature is not None:
            self._robot.set_dofs_armature(
                self._dof_armature,
                dofs_idx_local=self._dofs_idx_local,
            )

        # limits
        self._dof_pos_limits = torch.stack(self._robot.get_dofs_limit(self._dofs_idx_local), dim=1)
        for i in range(self._dof_pos_limits.shape[0]):
            # soft limits
            m = (self._dof_pos_limits[i, 0] + self._dof_pos_limits[i, 1]) / 2
            r = self._dof_pos_limits[i, 1] - self._dof_pos_limits[i, 0]
            self._dof_pos_limits[i, 0] = m - 0.5 * r * self._args.soft_dof_pos_range
            self._dof_pos_limits[i, 1] = m + 0.5 * r * self._args.soft_dof_pos_range
        self._torque_limits = self._robot.get_dofs_force_range(self._dofs_idx_local)[1]
        if self._args.dof_vel_limit is not None:
            dof_vel_limit = []
            for dof_name in self._args.dof_names:
                for key in self._args.dof_vel_limit.keys():
                    if key in dof_name:
                        dof_vel_limit.append(self._args.dof_vel_limit[key])
            self._dof_vel_limit = torch.tensor(dof_vel_limit, device=self._device)

    def _init_domain_randomization(self) -> None:
        envs_idx: torch.Tensor = torch.arange(0, self._num_envs, device=self._device)
        self._randomize_rigids(envs_idx)
        self._randomize_controls(envs_idx)
        self._steps_since_randomize_pds = 0

    def _randomize_rigids(self, envs_idx: torch.Tensor) -> None:
        # friction
        min_friction, max_friction = self._args.dr_args.friction_range
        solver: RigidSolver = self._robot.solver
        ratios = (
            torch.rand(len(envs_idx), 1).repeat(1, solver.n_geoms) * (max_friction - min_friction)
            + min_friction
        )
        solver.set_geoms_friction_ratio(ratios, torch.arange(0, solver.n_geoms), envs_idx)
        self._friction_ratio[envs_idx, 0] = ratios[:, 0]
        # mass
        min_mass, max_mass = self._args.dr_args.mass_range
        added_mass = torch.rand(len(envs_idx), 1) * (max_mass - min_mass) + min_mass
        self._robot.set_mass_shift(
            added_mass,
            [
                self.body_link_idx,
            ],
            envs_idx,
        )
        self._added_mass[envs_idx] = added_mass
        # com displacement
        min_com, max_com = self._args.dr_args.com_displacement_range
        displacement = (torch.rand(len(envs_idx), 1, 3) - 0.5) * (max_com - min_com) + min_com
        self._robot.set_COM_shift(
            displacement,
            [
                self.body_link_idx,
            ],
            envs_idx,
        )
        self._com_displacement[envs_idx] = displacement[:, 0, :]

    def _randomize_controls(self, envs_idx: torch.Tensor) -> None:
        # kp
        min_kp, max_kp = self._args.dr_args.kp_range
        ratios = torch.rand(len(envs_idx), self._dof_dim) * (max_kp - min_kp) + min_kp
        self._batched_dof_kp[envs_idx] = ratios * self._dof_kp[None, :]
        self._kp_ratio[envs_idx] = ratios
        # self._robot.set_dofs_kp(
        #     self._batched_dof_kp[envs_idx], dofs_idx_local=self._dofs_idx_local, envs_idx=envs_idx
        # )
        # kd
        min_kd, max_kd = self._args.dr_args.kd_range
        ratios = torch.rand(len(envs_idx), self._dof_dim) * (max_kd - min_kd) + min_kd
        self._batched_dof_kd[envs_idx] = ratios * self._dof_kd[None, :]
        self._kd_ratio[envs_idx] = ratios
        # self._robot.set_dofs_kv(
        #     self._batched_dof_kd[envs_idx], dofs_idx_local=self._dofs_idx_local, envs_idx=envs_idx
        # )
        # motor strength
        min_strength, max_strength = self._args.dr_args.motor_strength_range
        self._motor_strength[envs_idx] = (
            torch.rand(len(envs_idx), self._dof_dim) * (max_strength - min_strength) + min_strength
        )
        # motor offset
        min_offset, max_offset = self._args.dr_args.motor_offset_range
        self._motor_offset[envs_idx] = (
            torch.rand(len(envs_idx), self._dof_dim) * (max_offset - min_offset) + min_offset
        )

    def _randomize_pds(self) -> None:
        # kp
        min_kp, max_kp = self._args.dr_args.kp_range
        ratios = torch.rand(self._num_envs, self._dof_dim) * (max_kp - min_kp) + min_kp
        self._batched_dof_kp[:] = ratios * self._dof_kp[None, :]
        self._kp_ratio[:] = ratios
        # kd
        min_kd, max_kd = self._args.dr_args.kd_range
        ratios = torch.rand(self._num_envs, self._dof_dim) * (max_kd - min_kd) + min_kd
        self._batched_dof_kd[:] = ratios * self._dof_kd[None, :]
        self._kd_ratio[:] = ratios

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = torch.arange(self._num_envs, device=self._device)
        if len(envs_idx) == 0:
            return
        self.reset_idx(envs_idx)

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        self._robot.set_pos(
            self._default_pos[None].repeat(len(envs_idx), 1),
            envs_idx=envs_idx,
            zero_velocity=True,
        )
        self._robot.set_quat(
            self._default_quat[None].repeat(len(envs_idx), 1),
            envs_idx=envs_idx,
            zero_velocity=True,
        )
        self._robot.set_dofs_position(
            self._default_dof_pos[None].repeat(len(envs_idx), 1),
            envs_idx=envs_idx,
            dofs_idx_local=self._dofs_idx_local,
            zero_velocity=True,
        )
        self._dof_pos[envs_idx] = self._default_dof_pos[None].repeat(len(envs_idx), 1)
        self._dof_vel[envs_idx] = 0.0

    def set_state(
        self,
        envs_idx: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        quat: torch.Tensor | None = None,
        dof_pos: torch.Tensor | None = None,
        lin_vel: torch.Tensor | None = None,
        ang_vel: torch.Tensor | None = None,
        dof_vel: torch.Tensor | None = None,
    ) -> None:
        if envs_idx is None:
            envs_idx = torch.arange(self._num_envs, device=self._device)
        if pos is not None:
            self._robot.set_pos(pos, envs_idx=envs_idx)
        if quat is not None:
            self._robot.set_quat(quat, envs_idx=envs_idx)
        if dof_pos is not None:
            dof_pos = torch.clamp(dof_pos, self._dof_pos_limits[:, 0], self._dof_pos_limits[:, 1])
            self._robot.set_dofs_position(
                dof_pos, envs_idx=envs_idx, dofs_idx_local=self._dofs_idx_local
            )
            self._dof_pos[envs_idx] = dof_pos.clone()
        if lin_vel is not None:
            self._robot.set_dofs_velocity(lin_vel, envs_idx=envs_idx, dofs_idx_local=[0, 1, 2])
        else:
            self._robot.set_dofs_velocity(
                torch.zeros((len(envs_idx), 3), device=self._device),
                envs_idx=envs_idx,
                dofs_idx_local=[0, 1, 2],
            )
        if ang_vel is not None:
            self._robot.set_dofs_velocity(ang_vel, envs_idx=envs_idx, dofs_idx_local=[3, 4, 5])
        else:
            self._robot.set_dofs_velocity(
                torch.zeros((len(envs_idx), 3), device=self._device),
                envs_idx=envs_idx,
                dofs_idx_local=[3, 4, 5],
            )
        if dof_vel is not None:
            self._robot.set_dofs_velocity(
                dof_vel, envs_idx=envs_idx, dofs_idx_local=self._dofs_idx_local
            )
            self._dof_vel[envs_idx] = dof_vel.clone()
        else:
            self._robot.set_dofs_velocity(
                torch.zeros((len(envs_idx), self._dof_dim), device=self._device),
                envs_idx=envs_idx,
                dofs_idx_local=self._dofs_idx_local,
            )
            self._dof_vel[envs_idx] = 0.0

    def apply_action(self, action: BaseAction | torch.Tensor) -> None:
        """
        Apply the action to the robot.
        """
        if self._logging:
            self._dof_pos_history.append(self._dof_pos.clone())
            self._dof_vel_history.append(self._dof_vel.clone())
            self._target_dof_pos_history.append(
                action[:, : self._dof_dim].clone() + self._default_dof_pos
            )
            self._logging_time_stamp.append(self._time_stamp)
            self._time_stamp += 1.0 / self.ctrl_freq / self.decimation
        if isinstance(action, torch.Tensor):
            match self.ctrl_type:
                case CtrlType.DR_JOINT_POSITION:
                    action = DRJointPosAction(joint_pos=action)
                case CtrlType.DR_JOINT_POSITION_VELOCITY:
                    joint_pos = action[:, : self._dof_dim]
                    joint_vel = action[:, self._dof_dim :]
                    action = DRJointPosVelAction(joint_pos=joint_pos, joint_vel=joint_vel)
                case _:
                    raise ValueError(f"Unsupported control type: {self.ctrl_type}")
        self._dispatch[self._args.ctrl_type](action)
        self._dof_pos[:] = self._robot.get_dofs_position(self._dofs_idx_local)
        self._dof_vel[:] = self._robot.get_dofs_velocity(self._dofs_idx_local)

        self._steps_since_randomize_external_force += 1
        if self._steps_since_randomize_external_force >= self._steps_to_randomize_external_force:
            self._randomize_external_force()
            self._steps_since_randomize_external_force = 0
        self._apply_external_force()

        self._steps_since_randomize_pds += 1
        if self._steps_since_randomize_pds >= self._steps_to_randomize_pds:
            self._randomize_pds()
            self._steps_since_randomize_pds = 0

    def _apply_dr_joint_pos(self, act: DRJointPosAction) -> None:
        """
        Apply noised joint position control to the robot.
        """
        assert act.joint_pos.shape == (
            self._num_envs,
            self._dof_dim,
        ), "Joint position action must match the number of joints."
        q_force = (
            self._batched_dof_kp
            * (act.joint_pos + self._default_dof_pos - self._dof_pos + self._motor_offset)
            - self._batched_dof_kd * self._dof_vel
        )
        # logging.info(f"q_des mean: {torch.mean(act.joint_pos + self._default_dof_pos - self._dof_pos + self._motor_offset)}")
        # logging.info(f"qd_des mean: {torch.mean(self._dof_vel)}")
        q_force = q_force * self._motor_strength
        q_force = torch.clamp(q_force, -self._torque_limits, self._torque_limits)
        self._torque[:] = q_force
        self._robot.control_dofs_force(force=q_force, dofs_idx_local=self._dofs_idx_local)

    def _apply_dr_joint_pos_vel(self, act: DRJointPosVelAction) -> None:
        """
        Apply noised joint position and velocity control to the robot.
        """
        assert act.joint_pos.shape == (
            self._num_envs,
            self._dof_dim,
        ), "Joint position action must match the number of joints."

        q_force = self._batched_dof_kp * (
            act.joint_pos + self._default_dof_pos - self._dof_pos + self._motor_offset
        ) + self._batched_dof_kd * (-self._dof_vel + act.joint_vel * self._dof_feed_forward_ratio)
        q_force = q_force * self._motor_strength
        q_force = torch.clamp(q_force, -self._torque_limits, self._torque_limits)
        self._torque[:] = q_force
        self._robot.control_dofs_force(force=q_force, dofs_idx_local=self._dofs_idx_local)

    def _randomize_external_force(self) -> None:
        """
        Randomize external force to the robot.
        """
        self._external_force = (
            torch.rand(self._num_envs, len(self._external_force_links_idx), 3, device=self._device)
            * 2.0
            - 1.0
        )
        self._external_force *= self._external_force_range
        self._external_torque = (
            torch.rand(self._num_envs, len(self._external_force_links_idx), 3, device=self._device)
            * 2.0
            - 1.0
        )
        self._external_torque *= self._external_torque_range
        zero_force = (
            torch.rand(self._num_envs, len(self._external_force_links_idx), device=self._device)
            < 0.3
        )
        zero_torque = (
            torch.rand(self._num_envs, len(self._external_force_links_idx), device=self._device)
            < 0.3
        )
        self._external_force[zero_force] = 0.0
        self._external_torque[zero_torque] = 0.0

    def _apply_external_force(self) -> None:
        """
        Apply external force to the robot.
        """
        if self._args.dr_args.external_force_range is not None:
            self._robot.solver.apply_links_external_force(
                force=self._external_force,
                links_idx=self._external_force_links_idx,
                envs_idx=torch.arange(self._num_envs, device=self._device),
                ref="link_com",
                local=False,
            )
        if self._args.dr_args.external_torque_range is not None:
            self._robot.solver.apply_links_external_torque(
                torque=self._external_torque,
                links_idx=self._external_force_links_idx,
                envs_idx=torch.arange(self._num_envs, device=self._device),
                ref="link_com",
                local=False,
            )

    def get_link_idx_local_by_name(self, name: str) -> int:
        return self._robot.get_link(name).idx_local

    def start_logging(self) -> None:
        self._logging = True
        self._dof_pos_history = []
        self._dof_vel_history = []
        self._target_dof_pos_history = []
        self._logging_time_stamp = []
        self._time_stamp = 0.0

    def stop_logging(self) -> dict[str, torch.Tensor]:
        self._logging = False
        pos_history = torch.stack(self._dof_pos_history, dim=1).cpu()
        vel_history = torch.stack(self._dof_vel_history, dim=1).cpu()
        target_pos_history = torch.stack(self._target_dof_pos_history, dim=1).cpu()
        time_stamp = torch.tensor(self._logging_time_stamp, device=self._device)
        self._dof_pos_history = []
        self._dof_vel_history = []
        self._target_dof_pos_history = []
        self._logging_time_stamp = []
        self._time_stamp = 0.0
        return {
            "dof_pos": pos_history,
            "dof_vel": vel_history,
            "target_dof_pos": target_pos_history,
            "time_stamp": time_stamp,
        }

    @property
    def dof_kp(self) -> torch.Tensor:
        return self._dof_kp

    @property
    def dof_kd(self) -> torch.Tensor:
        return self._dof_kd

    @property
    def batched_dof_kp(self) -> torch.Tensor:
        return self._batched_dof_kp

    @property
    def batched_dof_kd(self) -> torch.Tensor:
        return self._batched_dof_kd

    def set_batched_dof_kp(self, batched_dof_kp: torch.Tensor) -> None:
        self._batched_dof_kp = batched_dof_kp

    def set_batched_dof_kd(self, batched_dof_kd: torch.Tensor) -> None:
        self._batched_dof_kd = batched_dof_kd

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    @property
    def robot(self) -> RigidEntity:
        return self._robot

    @property
    def mass(self) -> float:
        return self._mass

    @property
    def n_links(self) -> int:
        return self._robot.n_links

    @property
    def n_joints(self) -> int:
        return self._robot.n_joints

    @property
    def dof_dim(self) -> int:
        return self._dof_dim

    @property
    def dof_names(self) -> list[str]:
        return self._args.dof_names

    @property
    def link_names(self) -> list[str]:
        return [link.name for link in self._robot.links]

    @property
    def default_pos(self) -> torch.Tensor:
        return self._default_pos

    @property
    def default_quat(self) -> torch.Tensor:
        return self._default_quat

    @property
    def default_dof_pos(self) -> torch.Tensor:
        return self._default_dof_pos

    @property
    def base_pos(self) -> torch.Tensor:
        return self._robot.get_pos()

    @property
    def base_quat(self) -> torch.Tensor:
        return self._robot.get_quat()

    @property
    def dof_pos(self) -> torch.Tensor:
        return self._dof_pos

    @property
    def dof_vel(self) -> torch.Tensor:
        return self._dof_vel

    @property
    def torque(self) -> torch.Tensor:
        return self._torque

    @property
    def link_contact_forces(self) -> torch.Tensor:
        return self._robot.get_links_net_contact_force()

    @property
    def link_positions(self) -> torch.Tensor:
        return self._robot.get_links_pos()

    @property
    def link_quaternions(self) -> torch.Tensor:
        return self._robot.get_links_quat()

    @property
    def link_lin_velocities(self) -> torch.Tensor:
        return self._robot.get_links_vel()

    @property
    def link_ang_velocities(self) -> torch.Tensor:
        return self._robot.get_links_ang()

    @property
    def dof_pos_limits(self) -> torch.Tensor:
        return self._dof_pos_limits

    @property
    def dr_obs(self) -> torch.Tensor:
        return torch.cat(
            [
                self._friction_ratio,
                self._added_mass,
                self._com_displacement,
                self._kp_ratio,
                self._kd_ratio,
                self._motor_strength,
                self._motor_offset,
            ],
            dim=-1,
        )

    def __getattr__(self, item: str) -> Any:
        if hasattr(self._robot, item):
            return getattr(self._robot, item)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    @property
    def ctrl_type(self) -> CtrlType:
        return self._args.ctrl_type

    @property
    def ctrl_freq(self) -> float:
        return self._args.ctrl_freq

    @property
    def decimation(self) -> int:
        return self._args.decimation


class HumanoidRobotBase(LeggedRobotBase):
    def __init__(
        self,
        num_envs: int,
        scene: gs.Scene,
        args: QuadrupedRobotArgs | HumanoidRobotArgs,
        device: torch.device,
    ) -> None:
        super().__init__(num_envs, scene, args, device)  # type: ignore


class G1Robot(HumanoidRobotBase):
    def __init__(
        self,
        num_envs: int,
        scene: gs.Scene,
        args: QuadrupedRobotArgs | HumanoidRobotArgs,
        device: torch.device,
    ) -> None:
        super().__init__(num_envs, scene=scene, args=args, device=device)  # type: ignore
