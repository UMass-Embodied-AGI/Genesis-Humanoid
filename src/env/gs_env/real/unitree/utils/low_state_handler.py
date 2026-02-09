import struct
import threading
import time
from typing import Any

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowState_go
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowState_hg

from gs_env.sim.robots.config.schema import HumanoidRobotArgs

JointID = {
    "go2": {
        "FR_hip_joint": 0,
        "FR_thigh_joint": 1,
        "FR_calf_joint": 2,
        "FL_hip_joint": 3,
        "FL_thigh_joint": 4,
        "FL_calf_joint": 5,
        "RR_hip_joint": 6,
        "RR_thigh_joint": 7,
        "RR_calf_joint": 8,
        "RL_hip_joint": 9,
        "RL_thigh_joint": 10,
        "RL_calf_joint": 11,
    },
    "g1": {
        # Left Lower body 0:6
        "left_hip_pitch_joint": 0,
        "left_hip_roll_joint": 1,
        "left_hip_yaw_joint": 2,
        "left_knee_joint": 3,
        "left_ankle_pitch_joint": 4,
        "left_ankle_roll_joint": 5,
        # Right Lower body 6:12
        "right_hip_pitch_joint": 6,
        "right_hip_roll_joint": 7,
        "right_hip_yaw_joint": 8,
        "right_knee_joint": 9,
        "right_ankle_pitch_joint": 10,
        "right_ankle_roll_joint": 11,
        # Waist 12:15
        "waist_yaw_joint": 12,
        "waist_roll_joint": 13,
        "waist_pitch_joint": 14,
        # Left Upper body 12:19
        "left_shoulder_pitch_joint": 15,
        "left_shoulder_roll_joint": 16,
        "left_shoulder_yaw_joint": 17,
        "left_elbow_joint": 18,
        "left_wrist_roll_joint": 19,
        "left_wrist_pitch_joint": 20,
        "left_wrist_yaw_joint": 21,
        # Right Upper body 19:26
        "right_shoulder_pitch_joint": 22,
        "right_shoulder_roll_joint": 23,
        "right_shoulder_yaw_joint": 24,
        "right_elbow_joint": 25,
        "right_wrist_roll_joint": 26,
        "right_wrist_pitch_joint": 27,
        "right_wrist_yaw_joint": 28,
        # "LeftHipPitch": 0,
        # "LeftHipRoll": 1,
        # "LeftHipYaw": 2,
        # "LeftKnee": 3,
        # "LeftAnklePitch": 4,
        # "LeftAnkleB": 4,
        # "LeftAnkleRoll": 5,
        # "LeftAnkleA": 5,
        # "RightHipPitch": 6,
        # "RightHipRoll": 7,
        # "RightHipYaw": 8,
        # "RightKnee": 9,
        # "RightAnklePitch": 10,
        # "RightAnkleB": 10,
        # "RightAnkleRoll": 11,
        # "RightAnkleA": 11,
        # "WaistYaw": 12,
        # "WaistRoll": 13,        # NOTE: INVALID for g1 23dof/29dof with waist locked
        # "WaistA": 13,           # NOTE: INVALID for g1 23dof/29dof with waist locked
        # "WaistPitch": 14,       # NOTE: INVALID for g1 23dof/29dof with waist locked
        # "WaistB": 14,           # NOTE: INVALID for g1 23dof/29dof with waist locked
        # "LeftShoulderPitch": 15,
        # "LeftShoulderRoll": 16,
        # "LeftShoulderYaw": 17,
        # "LeftElbow": 18,
        # "LeftWristRoll": 19,
        # "LeftWristPitch": 20,   # NOTE: INVALID for g1 23dof
        # "LeftWristYaw": 21,     # NOTE: INVALID for g1 23dof
        # "RightShoulderPitch": 22,
        # "RightShoulderRoll": 23,
        # "RightShoulderYaw": 24,
        # "RightElbow": 25,
        # "RightWristRoll": 26,
        # "RightWristPitch": 27,  # NOTE: INVALID for g1 23dof
        # "RightWristYaw": 28,    # NOTE: INVALID for g1 23dof
    },
}


class LowStateMsgHandler:
    def __init__(self, cfg: HumanoidRobotArgs, freq: int = 1000) -> None:
        self.cfg = cfg
        self.update_interval = 1.0 / freq
        if "g1" in cfg.morph_args.file:
            self.robot_name = "g1"
        elif "go2" in cfg.morph_args.file:
            self.robot_name = "go2"
        else:
            raise ValueError(f"Not supported robot: {cfg.morph_args.file}")
        self.dof_names = cfg.dof_names
        self.num_dof = len(self.dof_names)
        self.dof_index = [JointID[self.robot_name][name] for name in self.dof_names]

        self.msg = None
        self.msg_received = False

        # robot
        self.quat = np.zeros(4)
        self.ang_vel = np.zeros(3)
        self.joint_pos = np.zeros(self.num_dof)
        self.joint_pos_raw = np.zeros(self.num_dof)
        self.joint_vel = np.zeros(self.num_dof)
        self.joint_vel_raw = np.zeros(self.num_dof)
        self.torque = np.zeros(self.num_dof)
        self.temperature = np.zeros(self.num_dof)
        if self.robot_name == "go2":
            self.num_full_dof = 12
            self.full_joint_pos = np.zeros(12)
        if self.robot_name == "g1":
            self.num_full_dof = 29
            self.full_joint_pos = np.zeros(29)

        # button
        self.L1 = 0
        self.L2 = 0
        self.R1 = 0
        self.R2 = 0
        self.A = 0
        self.B = 0
        self.X = 0
        self.Y = 0
        self.Up = 0
        self.Down = 0
        self.Left = 0
        self.Right = 0
        self.Select = 0
        self.F1 = 0
        self.F3 = 0
        self.Start = 0

        # Low Pass Filter
        self.low_pass_alpha = 0.1

        # Logging (runs in a separate thread)
        self._logging = False
        self._joint_pos_history = []
        self._joint_pos_raw_history = []
        self._joint_vel_history = []
        self._joint_vel_raw_history = []
        self._logging_time_stamp = []
        # Use control frequency times decimation for logging frequency
        self.logging_interval = 1.0 / (self.cfg.ctrl_freq * self.cfg.decimation)
        self._logging_thread: threading.Thread | None = None

        # Create a thread for the main loop
        self.main_thread = threading.Thread(target=self.main_loop, daemon=True)

        # TEMP
        self._right_wrist_offset = -0.0

    def init(self) -> None:
        try:
            ChannelFactoryInitialize(0, "enx2c16dbaafd43")  # MANUAL SET NETWORK INTERFACE
        except Exception:
            pass

        if self.robot_name == "go2":
            self.robot_lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_go)
            self.robot_lowstate_subscriber.Init(self.LowStateHandler_go, 10)
        elif self.robot_name == "g1":
            self.robot_lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_hg)
            self.robot_lowstate_subscriber.Init(self.LowStateHandler_hg, 10)
        while not self.msg_received:
            print("Waiting for Low State Message...")
            time.sleep(0.5)
        print("Low State Message Received!!!")

        self.main_thread.start()

    def LowStateHandler_go(self, msg: LowState_go) -> None:
        self.msg = msg
        self.msg_received = True

    def LowStateHandler_hg(self, msg: LowState_hg) -> None:
        self.msg = msg
        self.msg_received = True

    def main_loop(self) -> None:
        total_publish_cnt = 0  # noqa: F841
        start_time = time.time()  # noqa: F841
        while True:
            update_start_time = time.time()

            # Process raw message
            self.parse_imu(self.msg.imu_state)
            self.parse_motor_state(self.msg.motor_state)
            self.parse_wireless_remote(self.msg.wireless_remote)

            cur_time = time.time()
            if cur_time - update_start_time < self.update_interval:
                time.sleep(self.update_interval - (cur_time - update_start_time))

            # Print publishing rate
            # total_publish_cnt += 1
            # if total_publish_cnt == 1000:
            #     end_time = time.time()
            #     print("-" * 50)
            #     print(f"LowStateMsg Receiving Rate: {total_publish_cnt / (end_time - start_time)}")
            #     start_time = end_time
            #     total_publish_cnt = 0

    def parse_imu(self, imu_state: Any) -> None:
        self.quat = np.array(imu_state.quaternion)  # w, x, y, z
        self.ang_vel = np.array(imu_state.gyroscope)

    def parse_motor_state(self, motor_state: Any) -> None:
        for i in range(self.num_dof):
            self.joint_pos_raw[i] = motor_state[self.dof_index[i]].q
            self.joint_pos[i] += self.low_pass_alpha * (self.joint_pos_raw[i] - self.joint_pos[i])
            self.joint_vel_raw[i] = motor_state[self.dof_index[i]].dq
            self.joint_vel[i] += self.low_pass_alpha * (self.joint_vel_raw[i] - self.joint_vel[i])
            self.torque[i] = motor_state[self.dof_index[i]].tau_est
            # self.temperature[i] = motor_state[self.dof_index[i]].temperature
            error_code = motor_state[self.dof_index[i]].reserve[0]
            if error_code != 0:
                print(f"Joint {self.dof_index[i]} Error Code: {error_code}")
        for i in range(self.num_full_dof):
            self.full_joint_pos[i] = motor_state[i].q
        # print("low_state_big_flag", self.robot_low_state.bit_flag)
        self.full_joint_pos[-1] = motor_state[-1].q - self._right_wrist_offset

    def parse_botton(self, data1: int, data2: int) -> None:
        self.R1 = (data1 >> 0) & 1
        self.L1 = (data1 >> 1) & 1
        self.Start = (data1 >> 2) & 1
        self.Select = (data1 >> 3) & 1
        self.R2 = (data1 >> 4) & 1
        self.L2 = (data1 >> 5) & 1
        self.F1 = (data1 >> 6) & 1
        self.F3 = (data1 >> 7) & 1
        self.A = (data2 >> 0) & 1
        self.B = (data2 >> 1) & 1
        self.X = (data2 >> 2) & 1
        self.Y = (data2 >> 3) & 1
        self.Up = (data2 >> 4) & 1
        self.Right = (data2 >> 5) & 1
        self.Down = (data2 >> 6) & 1
        self.Left = (data2 >> 7) & 1

    def parse_key(self, data: bytes) -> None:
        lx_offset = 4
        self.Lx = struct.unpack("<f", data[lx_offset : lx_offset + 4])[0]
        rx_offset = 8
        self.Rx = struct.unpack("<f", data[rx_offset : rx_offset + 4])[0]
        ry_offset = 12
        self.Ry = struct.unpack("<f", data[ry_offset : ry_offset + 4])[0]
        # L2_offset = 16
        # L2 = struct.unpack("<f", data[L2_offset : L2_offset + 4])[0]  # Placeholder，unused
        ly_offset = 20
        self.Ly = struct.unpack("<f", data[ly_offset : ly_offset + 4])[0]

    def parse_wireless_remote(self, remoteData: bytes) -> None:
        self.parse_key(remoteData)
        self.parse_botton(remoteData[2], remoteData[3])

        # print("Lx:", self.Lx)
        # print("Rx:", self.Rx)
        # print("Ry:", self.Ry)
        # print("Ly:", self.Ly)

        # print("L1:", self.L1)
        # print("L2:", self.L2)
        # print("R1:", self.R1)
        # print("R2:", self.R2)
        # print("A:", self.A)
        # print("B:", self.B)
        # print("X:", self.X)
        # print("Y:", self.Y)
        # print("Up:", self.Up)
        # print("Down:", self.Down)
        # print("Left:", self.Left)
        # print("Right:", self.Right)
        # print("Select:", self.Select)
        # print("F1:", self.F1)
        # print("F3:", self.F3)
        # print("Start:", self.Start)

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
            self._logging_time_stamp.append(time.time() - self._logging_start_time)
            time.sleep(max(0, next_log_time - time.time()))
            next_log_time += self.logging_interval

    def start_logging(self) -> None:
        self._logging = True
        self._joint_pos_history = []
        self._joint_pos_raw_history = []
        self._joint_vel_history = []
        self._joint_vel_raw_history = []
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
        time_stamp = np.array(self._logging_time_stamp)
        return {
            "dof_pos": pos_history,
            "dof_pos_raw": pos_raw_history,
            "dof_vel": vel_history,
            "dof_vel_raw": vel_raw_history,
            "time_stamp": time_stamp,
        }
