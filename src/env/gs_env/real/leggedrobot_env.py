import mujoco
import numpy as np
import torch
from gymnasium import spaces
from transforms3d import quaternions

from gs_env.common.bases.base_robot import BaseGymRobot
from gs_env.common.utils.math_utils import quat_from_euler, quat_to_euler, quat_to_rotation_6D
from gs_env.real.unitree.utils.low_state_controller import LowStateCmdHandler
from gs_env.real.unitree.utils.low_state_handler import LowStateMsgHandler
from gs_env.sim.envs.config.schema import LeggedRobotEnvArgs

_DEFAULT_DEVICE = torch.device("cpu")


class UnitreeLeggedEnv(BaseGymRobot):
    def __init__(
        self,
        args: LeggedRobotEnvArgs,
        action_scale: float = 0.0,
        interactive: bool = False,
        device: torch.device = _DEFAULT_DEVICE,
        xml_path: str | None = None,
    ) -> None:
        super().__init__()
        self._args = args
        if interactive:
            self._robot = LowStateCmdHandler(args.robot_args)
            self._robot.init()
            self._robot.start()
        else:
            self._robot = LowStateMsgHandler(args.robot_args)
            self._robot.init()
        self.real_time_fk = False
        if xml_path is not None:
            self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
            self.mj_data = mujoco.MjData(self.mj_model)
            self.qpos_adr = np.zeros((len(args.robot_args.dof_names),), dtype=np.int32)
            for i, joint_name in enumerate(args.robot_args.dof_names):
                self.qpos_adr[i] = int(self.mj_model.joint(joint_name).qposadr)
            self.tracking_link_idx = np.zeros((len(args.tracking_link_names),), dtype=np.int32)
            for i, link_name in enumerate(args.tracking_link_names):
                self.tracking_link_idx[i] = int(
                    mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
                )
            self.real_time_fk = True
        self._device = device
        self._action_space = spaces.Box(shape=(self.action_dim,), low=-np.inf, high=np.inf)
        self._action_scale = np.ones((self.action_dim,), dtype=np.float32)
        self._hard_action_scale = action_scale
        if self._args.robot_args.adaptive_action_scale:
            assert self._args.robot_args.dof_torque_limit is not None, (
                "Adaptive action scaling requires dof_torque_limit to be set."
            )
            dof_torque_limit = self._args.robot_args.dof_torque_limit
            dof_kp = self._args.robot_args.dof_kp
            for i, dof_name in enumerate(self.dof_names):
                for key in dof_torque_limit.keys():
                    if key in dof_name:
                        self._action_scale[i] = dof_torque_limit[key] / dof_kp[key]
                        break
        self._action_scale *= self._args.robot_args.action_scale
        if isinstance(self.robot, LowStateCmdHandler):
            self.prev_target_pos = np.array(self.robot.default_dof_pos, dtype=np.float32)
            self.target_vel_low_pass = np.zeros_like(self.robot.default_dof_pos, dtype=np.float32)
        self.low_pass_alpha = self._args.robot_args.low_pass_alpha

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        # TODO: implement reset to reset_pos
        pass

    def apply_action(self, action: torch.Tensor) -> None:
        if not isinstance(self.robot, LowStateCmdHandler):
            raise RuntimeError("apply_action is only available in interactive mode.")
        action_np = action[0].cpu().numpy()
        target_pos = (
            self.robot.default_dof_pos + action_np * self._action_scale * self._hard_action_scale
        )
        target_vel = (target_pos - self.prev_target_pos) / self.dt * self._hard_action_scale
        self.target_vel_low_pass += self.low_pass_alpha * (target_vel - self.target_vel_low_pass)
        self.robot.target_pos = target_pos
        self.robot.target_vel = target_vel
        self.prev_target_pos = target_pos.copy()

    def emergency_stop(self) -> None:
        self.robot.emergency_stop()

    @property
    def is_emergency_stop(self) -> bool:
        return self.robot.is_emergency_stop

    @property
    def robot(self) -> LowStateCmdHandler | LowStateMsgHandler:
        return self._robot  # type: ignore

    @property
    def action_dim(self) -> int:
        return self._robot.num_dof

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    @property
    def action_scale(self) -> torch.Tensor:
        return torch.tensor(self._action_scale, device=self._device, dtype=torch.float32)

    @property
    def dt(self) -> float:
        return 1.0 / self.ctrl_freq

    @property
    def dof_names(self) -> list[str]:
        return self.robot.dof_names

    @property
    def dof_kp(self) -> torch.Tensor:
        return torch.tensor(self.robot.kp, device=self._device, dtype=torch.float32)

    @property
    def dof_kd(self) -> torch.Tensor:
        return torch.tensor(self.robot.kd, device=self._device, dtype=torch.float32)

    @property
    def default_dof_pos(self) -> torch.Tensor:
        return torch.tensor(self.robot.default_dof_pos, device=self._device, dtype=torch.float32)[
            None, :
        ]

    @property
    def dof_pos(self) -> torch.Tensor:
        return torch.tensor(self.robot.joint_pos, device=self._device, dtype=torch.float32)[None, :]

    @property
    def dof_vel(self) -> torch.Tensor:
        return torch.tensor(self.robot.joint_vel, device=self._device, dtype=torch.float32)[None, :]

    @property
    def base_quat(self) -> torch.Tensor:
        return torch.tensor(self.robot.quat, device=self._device, dtype=torch.float32)[None, :]

    @property
    def base_euler(self) -> torch.Tensor:
        return quat_to_euler(self.base_quat)

    @property
    def base_rotation_6D(self) -> torch.Tensor:
        return quat_to_rotation_6D(self.base_quat)

    @property
    def projected_gravity(self) -> torch.Tensor:
        projected_gravity = quaternions.rotate_vector(
            v=np.array([0, 0, -1]),
            q=quaternions.qinverse(self.robot.quat),
        )
        return torch.tensor(projected_gravity, device=self._device, dtype=torch.float32)[None, :]

    # @property
    # def base_ang_vel(self) -> torch.Tensor:
    #     return torch.tensor(self.robot.ang_vel, device=self._device, dtype=torch.float32)[None, :]
    # SHOULD BE GLOBAL ANGULAR VELOCITY, NOT IMPLEMENTED YET

    @property
    def base_ang_vel_local(self) -> torch.Tensor:
        return torch.tensor(self.robot.ang_vel, device=self._device, dtype=torch.float32)[None, :]

    @property
    def tracking_link_pos_local_yaw(self) -> torch.Tensor:
        if not self.real_time_fk:
            raise RuntimeError("Real-time FK is not enabled.")
        self.mj_data.qpos[0:3] = 0.0
        local_euler = self.base_euler.cpu().clone()
        local_euler[:, 2] = 0.0
        quat = quat_from_euler(local_euler)
        self.mj_data.qpos[3:7] = quat.numpy()[0]
        self.mj_data.qpos[self.qpos_adr] = self.dof_pos.cpu().numpy()
        mujoco.mj_forward(self.mj_model, self.mj_data)
        tracking_link_pos_local_yaw = torch.tensor(
            self.mj_data.xpos[self.tracking_link_idx], device=self._device, dtype=torch.float32
        )[None, :]
        return tracking_link_pos_local_yaw

    @property
    def tracking_link_quat_local_yaw(self) -> torch.Tensor:
        if not self.real_time_fk:
            raise RuntimeError("Real-time FK is not enabled.")
        self.mj_data.qpos[0:3] = 0.0
        local_euler = self.base_euler.cpu().clone()
        local_euler[:, 2] = 0.0
        quat = quat_from_euler(local_euler)
        self.mj_data.qpos[3:7] = quat.numpy()[0]
        self.mj_data.qpos[self.qpos_adr] = self.dof_pos.cpu().numpy()
        mujoco.mj_forward(self.mj_model, self.mj_data)
        tracking_link_quat_local_yaw = torch.tensor(
            self.mj_data.xquat[self.tracking_link_idx], device=self._device, dtype=torch.float32
        )[None, :]
        return tracking_link_quat_local_yaw

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def ctrl_type(self) -> int:
        return self._args.robot_args.ctrl_type

    @property
    def ctrl_freq(self) -> int:
        return self._args.robot_args.ctrl_freq

    @property
    def decimation(self) -> int:
        return self._args.robot_args.decimation
