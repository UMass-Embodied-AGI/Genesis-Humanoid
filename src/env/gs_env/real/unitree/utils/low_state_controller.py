import threading
import time

import numpy as np
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.default import (
    std_msgs_msg_dds__String_,
    unitree_go_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowCmd_,
)
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmd_go
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmd_hg
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

from gs_env.sim.robots.config.schema import HumanoidRobotArgs

from .low_state_handler import LowStateMsgHandler


class LowStateCmdHandler(LowStateMsgHandler):
    def __init__(self, cfg: HumanoidRobotArgs, freq: int = 1000) -> None:
        super().__init__(cfg, freq)

        kp_groups = self.cfg.dof_kp
        self.kp = [
            kp_groups[self.group_from_name(name, kp_groups.keys())] for name in self.dof_names
        ]

        kd_groups = self.cfg.dof_kd
        self.kd = [
            kd_groups[self.group_from_name(name, kd_groups.keys())] for name in self.dof_names
        ]

        ffr_groups = self.cfg.feed_forward_ratio
        if isinstance(ffr_groups, dict):
            self.feed_forward_ratio = [
                ffr_groups[self.group_from_name(name, ffr_groups.keys())] for name in self.dof_names
            ]
        else:
            self.feed_forward_ratio = [ffr_groups for _ in self.dof_names]
        for i, name in enumerate(self.dof_names):
            for dof_name in self.cfg.indirect_drive_joint_names:
                if dof_name in name:
                    self.feed_forward_ratio[i] = 0.0

        self.default_dof_pos = np.array([self.cfg.default_dof_pos[name] for name in self.dof_names])

        reset_joint_angles = getattr(self.cfg, "reset_joint_angles", None)
        if reset_joint_angles is not None:
            reset_dof_pos = [reset_joint_angles[name] for name in self.dof_names]
            self.reset_dof_pos = np.array(reset_dof_pos)
        else:
            default_pos = [self.cfg.default_dof_pos[name] for name in self.dof_names]
            self.reset_dof_pos = np.array(default_pos)
        self.target_pos = self.reset_dof_pos
        self.target_vel = np.zeros_like(self.target_pos)

        self.full_default_dof_pos = np.zeros(self.num_full_dof)
        for i in range(self.num_dof):
            self.full_default_dof_pos[self.dof_index[i]] = self.default_dof_pos[i]
            self.full_joint_pos[self.dof_index[i]] = self.reset_dof_pos[i]

        if self.robot_name == "go2":
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
        elif self.robot_name == "g1":
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self._emergency_stop = False

        # thread handling
        self.lowCmdWriteThreadPtr = None

        self._target_pos_history = []

        self.crc = CRC()

    # Public methods
    def init(self) -> None:
        super().init()

        if self.robot_name == "go2":
            self.lidar_switch_publisher = ChannelPublisher("rt/utlidar/switch", String_)
            self.lidar_switch_publisher.Init()
            self.lidar_switch = std_msgs_msg_dds__String_()
            self.lidar_switch.data = "OFF"
            self.lidar_switch_publisher.Write(self.lidar_switch)

        # create publisher #
        if self.robot_name == "go2":
            self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_go)
            self.lowcmd_publisher.Init()
        if self.robot_name == "g1":
            self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_hg)
            self.lowcmd_publisher.Init()

        # self.sc = SportClient()
        # self.sc.SetTimeout(5.0)
        # self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

    def start(self) -> None:
        self.msc.ReleaseMode()

        self.full_initial_dof_pos = self.full_joint_pos.copy()
        self.full_initial_dof_pos[-1] += self._right_wrist_offset
        self.initial_stage = 0.0

        self.init_low_cmd()

        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=self.update_interval, target=self.LowCmdWrite, name="writebasiccmd"
        )
        self.lowCmdWriteThreadPtr.Start()

    def recover(self) -> None:
        status, result = self.msc.CheckMode()
        while result["name"] != "normal":
            self.msc.SelectMode("normal")
            status, result = self.msc.CheckMode()
            time.sleep(1)

    def init_low_cmd(self) -> None:
        if self.robot_name == "g1":
            Kp = [
                # Left Leg
                60,
                60,
                60,
                100,
                40,
                40,
                # Right Leg
                60,
                60,
                60,
                100,
                40,
                40,
                # Waist
                400,
                400,
                150,
                # Left Arm
                40,
                40,
                40,
                40,
                40,
                40,
                40,
                # Right Arm
                40,
                40,
                40,
                40,
                40,
                40,
                40,
            ]
            Kd = [
                # Left Leg
                1,
                1,
                1,
                2,
                1,
                1,
                # Right Leg
                1,
                1,
                1,
                2,
                1,
                1,
                # Waist
                6,
                6,
                4,
                # Left Arm
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                # Right Arm
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ]
            self.low_cmd.mode_pr = 0  # 0 for pitch roll, 1 for A B
            self.low_cmd.mode_machine = self.msg.mode_machine
            for i in range(29):
                self.low_cmd.motor_cmd[i].mode = 1  # 1:Enable, 0:Disable
                self.low_cmd.motor_cmd[i].q = self.full_initial_dof_pos[i]
                self.low_cmd.motor_cmd[i].kp = Kp[i]
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kd = Kd[i]
                self.low_cmd.motor_cmd[i].tau = 0.0
        elif self.robot_name == "go2":
            # self.low_cmd.head[0]=0xFE
            # self.low_cmd.head[1]=0xEF
            # self.low_cmd.level_flag = 0xFF
            # self.low_cmd.gpio = 0
            for i in range(12):
                self.low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
                self.low_cmd.motor_cmd[i].q = self.full_initial_dof_pos[i]
                self.low_cmd.motor_cmd[i].kp = 30
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kd = 1.5
                self.low_cmd.motor_cmd[i].tau = 0.0

    def set_stop_cmd(self) -> None:
        if self.robot_name == "go2":
            for i in range(20):
                self.low_cmd.motor_cmd[i].mode = 0
                self.low_cmd.motor_cmd[i].q = 2.146e9
                self.low_cmd.motor_cmd[i].kp = 0
                self.low_cmd.motor_cmd[i].dq = 16000.0
                self.low_cmd.motor_cmd[i].kd = 5
                self.low_cmd.motor_cmd[i].tau = 0
        elif self.robot_name == "g1":
            for i in range(29):
                self.low_cmd.motor_cmd[i].mode = 0
                self.low_cmd.motor_cmd[i].q = 0
                self.low_cmd.motor_cmd[i].kp = 0
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kd = 5
                self.low_cmd.motor_cmd[i].tau = 0

    def set_cmd(self) -> None:
        for i in range(self.num_dof):
            self.low_cmd.motor_cmd[self.dof_index[i]].q = self.target_pos[i]
            self.low_cmd.motor_cmd[self.dof_index[i]].dq = (
                self.target_vel[i] * self.feed_forward_ratio[i]
            )
            self.low_cmd.motor_cmd[self.dof_index[i]].kp = self.kp[i]
            self.low_cmd.motor_cmd[self.dof_index[i]].kd = self.kd[i]
            self.low_cmd.motor_cmd[self.dof_index[i]].tau = 0
        # TEMP
        self.low_cmd.motor_cmd[self.dof_index[-1]].q += self._right_wrist_offset

    def LowCmdWrite(self) -> None:
        if self.L2 and self.R2:  # if both L2 and R2 are pressed, emergency stop
            self.emergency_stop()

        if self._emergency_stop:
            self.set_stop_cmd()
        elif self.initial_stage < 1.0:
            target_pos = (
                self.full_initial_dof_pos
                + (self.full_default_dof_pos - self.full_initial_dof_pos) * self.initial_stage
            )
            for i in range(self.num_full_dof):
                self.low_cmd.motor_cmd[i].q = target_pos[i]
            self.low_cmd.motor_cmd[self.dof_index[-1]].q += self._right_wrist_offset
            self.initial_stage += 0.001
        else:
            self.set_cmd()

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)

    def emergency_stop(self) -> None:
        self._emergency_stop = True

    def group_from_name(self, joint_name: str, groups: dict[str, float]) -> str:
        for g in sorted(groups, key=len, reverse=True):
            if joint_name.endswith(g + "_joint") or joint_name.endswith(g):
                return g

    @property
    def is_emergency_stop(self) -> bool:
        return self._emergency_stop

    # =========================
    # Logging API (threaded)
    # =========================
    def _logging_loop(self) -> None:
        next_log_time = time.time() + self.logging_interval
        while self._logging:
            # Copy current measurements
            self._joint_pos_history.append(self.joint_pos.copy())
            self._joint_pos_raw_history.append(self.joint_pos_raw.copy())
            self._joint_vel_history.append(self.joint_vel.copy())
            self._joint_vel_raw_history.append(self.joint_vel_raw.copy())
            self._target_pos_history.append(self.target_pos.copy())
            self._logging_time_stamp.append(time.time() - self._logging_start_time)
            time.sleep(max(0, next_log_time - time.time()))
            next_log_time += self.logging_interval

    def start_logging(self) -> None:
        self._logging = True
        self._joint_pos_history = []
        self._joint_pos_raw_history = []
        self._joint_vel_history = []
        self._joint_vel_raw_history = []
        self._target_pos_history = []
        self._logging_time_stamp = []
        self._logging_thread = threading.Thread(target=self._logging_loop, daemon=True)
        self._logging_start_time = time.time()
        self._logging_thread.start()

    def stop_logging(self) -> dict[str, np.typing.NDArray]:
        self._logging = False
        if self._logging_thread is not None:
            self._logging_thread.join(timeout=1.0)
            self._logging_thread = None
        pos_history = np.stack(self._joint_pos_history, axis=0)
        pos_raw_history = np.stack(self._joint_pos_raw_history, axis=0)
        vel_history = np.stack(self._joint_vel_history, axis=0)
        vel_raw_history = np.stack(self._joint_vel_raw_history, axis=0)
        target_pos_history = np.stack(self._target_pos_history, axis=0)
        time_stamp = np.array(self._logging_time_stamp)
        return {
            "dof_pos": pos_history,
            "dof_pos_raw": pos_raw_history,
            "dof_vel": vel_history,
            "dof_vel_raw": vel_raw_history,
            "target_dof_pos": target_pos_history,
            "time_stamp": time_stamp,
        }
