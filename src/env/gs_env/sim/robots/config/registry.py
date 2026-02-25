from typing import TypeAlias

from gs_env.sim.robots.config.schema import (
    CtrlType,
    DomainRandomizationArgs,
    HumanoidRobotArgs,
    MJCFMorphArgs,
    QuadrupedRobotArgs,
    RigidMaterialArgs,
    URDFMorphArgs,
)

# ------------------------------------------------------------
# Material
# ------------------------------------------------------------

MaterialArgs: TypeAlias = RigidMaterialArgs


MaterialArgsRegistry: dict[str, MaterialArgs] = {}


MaterialArgsRegistry["default"] = RigidMaterialArgs(
    rho=200.0,
    friction=None,
    needs_coup=True,
    coup_friction=0.1,
    coup_softness=0.002,
    coup_restitution=0.0,
    sdf_cell_size=0.005,
    sdf_min_res=32,
    sdf_max_res=128,
    gravity_compensation=1,
)


MaterialArgsRegistry["g1_default"] = RigidMaterialArgs(
    rho=200.0,
    friction=None,
    needs_coup=True,
    coup_friction=0.1,
    coup_softness=0.002,
    coup_restitution=0.0,
    sdf_cell_size=0.005,
    sdf_min_res=32,
    sdf_max_res=128,
    gravity_compensation=0,
)


MaterialArgsRegistry["g1_fixed"] = RigidMaterialArgs(
    rho=200.0,
    friction=None,
    needs_coup=True,
    coup_friction=0.1,
    coup_softness=0.002,
    coup_restitution=0.0,
    sdf_cell_size=0.005,
    sdf_min_res=32,
    sdf_max_res=128,
    gravity_compensation=1,
)


# ------------------------------------------------------------
# Morph
# ------------------------------------------------------------

MorphArgs: TypeAlias = URDFMorphArgs | MJCFMorphArgs


MorphArgsRegistry: dict[str, MorphArgs] = {}


MorphArgsRegistry["franka_default"] = MJCFMorphArgs(
    pos=(0.0, 0.0, 0.0),
    quat=(1.0, 0.0, 0.0, 0.0),
    file="xml/franka_emika_panda/panda.xml",
)


MorphArgsRegistry["g1_default"] = URDFMorphArgs(
    pos=(0.0, 0.0, 0.8),
    euler=(0, 0, 0),
    quat=None,
    visualization=True,
    collision=True,
    requires_jac_and_IK=True,
    is_free=True,
    file="assets/robot/unitree_g1/g1_custom_collision_29dof.urdf",
    scale=1.0,
    convexify=True,
    recompute_inertia=False,
    fixed=False,
    prioritize_urdf_material=False,
    merge_fixed_links=True,
    links_to_keep=[],
    decimate=True,
)


MorphArgsRegistry["g1_fixed"] = URDFMorphArgs(
    pos=(0.0, 0.0, 1.0),
    euler=(0, 0, 0),
    quat=None,
    visualization=True,
    collision=False,
    requires_jac_and_IK=True,
    is_free=True,
    file="assets/robot/unitree_g1/g1_custom_collision_29dof.urdf",
    scale=1.0,
    convexify=True,
    recompute_inertia=False,
    fixed=True,
    prioritize_urdf_material=False,
    merge_fixed_links=False,
    links_to_keep=[],
    decimate=True,
)


# ------------------------------------------------------------
# Domain Randomization
# ------------------------------------------------------------

DRArgs: TypeAlias = DomainRandomizationArgs


DRArgsRegistry: dict[str, DRArgs] = {}


DRArgsRegistry["default"] = DomainRandomizationArgs(
    kp_range=(0.8, 1.2),
    kd_range=(0.8, 1.2),
    motor_strength_range=(0.8, 1.2),
    motor_offset_range=(-0.1, 0.1),
    friction_range=(0.3, 1.0),
    mass_range=(-2.0, 5.0),
    com_displacement_range=(-0.1, 0.1),
    # external_force_range=(5.0, 5.0, 10.0),
    # external_torque_range=(1.0, 1.0, 1.0),
)


DRArgsRegistry["no_randomization"] = DomainRandomizationArgs(
    kp_range=(1.0, 1.0),
    kd_range=(1.0, 1.0),
    motor_strength_range=(1.0, 1.0),
    motor_offset_range=(0.0, 0.0),
    friction_range=(1.0, 1.0),
    mass_range=(0.0, 0.0),
    com_displacement_range=(0.0, 0.0),
)


# ------------------------------------------------------------
# Robot
# ------------------------------------------------------------


RobotArgsRegistry: dict[str, QuadrupedRobotArgs | HumanoidRobotArgs] = {}

# ------------------------------------------------------------
# G1 Configuration
# ------------------------------------------------------------

G1_dof_names: list[str] = [
    # Left Lower body 0:6
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # Right Lower body 6:12
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # Waist 12:15
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # Left Upper body 15:22
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # Right Upper body 22:29
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
G1_no_waist_dof_names: list[str] = [
    # Left Lower body 0:6
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # Right Lower body 6:12
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # Left Upper body 12:19
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # Right Upper body 19:26
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
G1_default_dof_pos: dict[str, float] = {
    "left_hip_pitch_joint": -0.2,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.4,
    "left_ankle_pitch_joint": -0.2,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.2,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.4,
    "right_ankle_pitch_joint": -0.2,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 1.0,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 1.0,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}
G1_kp_dict: dict[str, float] = {
    "hip_roll": 100.0,
    "hip_pitch": 100.0,
    "hip_yaw": 100.0,
    "knee": 100.0,
    "ankle_roll": 30.0,
    "ankle_pitch": 30.0,
    "waist_roll": 150.0,
    "waist_pitch": 150.0,
    "waist_yaw": 100.0,
    "shoulder_roll": 30.0,
    "shoulder_pitch": 30.0,
    "shoulder_yaw": 11.0,
    "elbow": 15.0,
    "wrist_roll": 10.0,
    "wrist_pitch": 13.0,
    "wrist_yaw": 12.0,
}
G1_kd_dict: dict[str, float] = {
    "hip_roll": 20.0,
    "hip_pitch": 20.0,
    "hip_yaw": 20.0,
    "knee": 20.0,
    "ankle_roll": 6.0,
    "ankle_pitch": 6.0,
    "waist_roll": 10.0,
    "waist_pitch": 10.0,
    "waist_yaw": 10.0,
    "shoulder_roll": 6.0,
    "shoulder_pitch": 6.0,
    "shoulder_yaw": 2.2,
    "elbow": 3.0,
    "wrist_roll": 2.0,
    "wrist_pitch": 2.6,
    "wrist_yaw": 2.4,
}
UPPER_BODY_FFR = 0.9
LOWER_BODY_FFR = 0.9
G1_feed_forward_ratio_dict: dict[str, float] = {
    "hip_roll": LOWER_BODY_FFR,
    "hip_pitch": LOWER_BODY_FFR,
    "hip_yaw": LOWER_BODY_FFR,
    "knee": LOWER_BODY_FFR,
    "ankle_roll": LOWER_BODY_FFR,
    "ankle_pitch": LOWER_BODY_FFR,
    "waist_roll": LOWER_BODY_FFR,
    "waist_pitch": LOWER_BODY_FFR,
    "waist_yaw": LOWER_BODY_FFR,
    "shoulder_roll": UPPER_BODY_FFR,
    "shoulder_pitch": UPPER_BODY_FFR,
    "shoulder_yaw": UPPER_BODY_FFR,
    "elbow": UPPER_BODY_FFR,
    "wrist_roll": UPPER_BODY_FFR,
    "wrist_pitch": UPPER_BODY_FFR,
    "wrist_yaw": UPPER_BODY_FFR,
}
G1_vel_limit_dict: dict[str, float] = {
    "hip_roll": 20.0,
    "hip_pitch": 32.0,
    "hip_yaw": 32.0,
    "knee": 20.0,
    "ankle_roll": 37.0,
    "ankle_pitch": 37.0,
    "waist_roll": 37.0,
    "waist_pitch": 37.0,
    "waist_yaw": 32.0,
    "shoulder_roll": 37.0,
    "shoulder_pitch": 37.0,
    "shoulder_yaw": 37.0,
    "elbow": 37.0,
    "wrist_roll": 37.0,
    "wrist_pitch": 22.0,
    "wrist_yaw": 22.0,
}
G1_torque_limit_dict: dict[str, float] = {
    "hip_roll": 139.0,
    "hip_pitch": 88.0,
    "hip_yaw": 88.0,
    "knee": 139.0,
    "ankle_roll": 50.0,
    "ankle_pitch": 50.0,
    "waist_roll": 50.0,
    "waist_pitch": 50.0,
    "waist_yaw": 88.0,
    "shoulder_roll": 25.0,
    "shoulder_pitch": 25.0,
    "shoulder_yaw": 25.0,
    "elbow": 25.0,
    "wrist_roll": 25.0,
    "wrist_pitch": 5.0,
    "wrist_yaw": 5.0,
}
G1_indirect_drive_joints: list[str] = [
    "ankle",
    "waist_pitch",
    "waist_roll",
]

RobotArgsRegistry["g1_default"] = HumanoidRobotArgs(
    material_args=MaterialArgsRegistry["g1_default"],
    morph_args=MorphArgsRegistry["g1_default"],
    dr_args=DRArgsRegistry["default"],
    visualize_contact=False,
    vis_mode="visual",
    ctrl_type=CtrlType.DR_JOINT_POSITION_VELOCITY,
    body_link_name="torso_link",
    foot_link_names=[
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ],
    external_force_links_idx=[
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ],
    show_target=True,
    dof_names=G1_dof_names,
    default_dof_pos=G1_default_dof_pos,
    soft_dof_pos_range=0.9,
    dof_kp=G1_kp_dict,
    dof_kd=G1_kd_dict,
    dof_vel_limit=G1_vel_limit_dict,
    dof_torque_limit=G1_torque_limit_dict,
    action_scale=0.15,
    ctrl_freq=50,
    decimation=4,
    adaptive_action_scale=False,
    feed_forward_ratio=G1_feed_forward_ratio_dict,
    indirect_drive_joint_names=[
        "ankle",
        "waist_pitch",
        "waist_roll",
    ],
    low_pass_alpha=0.5,
)


RobotArgsRegistry["g1_no_dr"] = HumanoidRobotArgs(
    material_args=MaterialArgsRegistry["g1_default"],
    morph_args=MorphArgsRegistry["g1_default"],
    dr_args=DRArgsRegistry["no_randomization"],
    visualize_contact=False,
    vis_mode="visual",
    ctrl_type=CtrlType.DR_JOINT_POSITION,
    body_link_name="torso_link",
    foot_link_names=[
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ],
    show_target=True,
    dof_names=G1_dof_names,
    default_dof_pos=G1_default_dof_pos,
    soft_dof_pos_range=0.9,
    dof_kp=G1_kp_dict,
    dof_kd=G1_kd_dict,
    dof_vel_limit=G1_vel_limit_dict,
    dof_torque_limit=G1_torque_limit_dict,
    action_scale=0.15,
    ctrl_freq=50,
    decimation=4,
    feed_forward_ratio=G1_feed_forward_ratio_dict,
    indirect_drive_joint_names=[
        "ankle",
        "waist_pitch",
        "waist_roll",
    ],
    adaptive_action_scale=False,
)


RobotArgsRegistry["g1_fixed"] = HumanoidRobotArgs(
    material_args=MaterialArgsRegistry["g1_fixed"],
    morph_args=MorphArgsRegistry["g1_fixed"],
    dr_args=DRArgsRegistry["default"],
    visualize_contact=False,
    vis_mode="visual",
    ctrl_type=CtrlType.DR_JOINT_POSITION_VELOCITY,
    body_link_name="torso_link",
    foot_link_names=[
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ],
    show_target=True,
    dof_names=G1_dof_names,
    default_dof_pos=G1_default_dof_pos,
    soft_dof_pos_range=0.9,
    dof_kp=G1_kp_dict,
    dof_kd=G1_kd_dict,
    dof_vel_limit=G1_vel_limit_dict,
    dof_torque_limit=G1_torque_limit_dict,
    action_scale=0.15,
    feed_forward_ratio=G1_feed_forward_ratio_dict,
    indirect_drive_joint_names=[
        "ankle",
        "waist_pitch",
        "waist_roll",
    ],
    adaptive_action_scale=False,
    ctrl_freq=50,
    decimation=4,
)
