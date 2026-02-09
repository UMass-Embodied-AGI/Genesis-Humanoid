from gs_env.sim.sensors.config.schema import (
    OakCameraArgs,
    ProprioceptiveSensorArgs,
    ProprioceptiveSensorType,
    SensorArgs,
)

SensorArgsRegistry: dict[str, SensorArgs] = {}


SensorArgsRegistry["oak_camera_default"] = OakCameraArgs(
    name="front_view",
    silent=True,
    resolution=(480, 270),
    fps=30,
    exposure=None,
    white_balance=None,
    pos=(0.7366, 0.0, 0.4826),
    lookat=(0.7366 * 0.4, 0, 0.0),
    fov=42,
    GUI=False,
)


SensorArgsRegistry["ee_link_pos"] = ProprioceptiveSensorArgs(
    sensor_type=ProprioceptiveSensorType.EE_LINK_POS,
)


SensorArgsRegistry["ee_link_quat"] = ProprioceptiveSensorArgs(
    sensor_type=ProprioceptiveSensorType.EE_LINK_QUAT,
)


SensorArgsRegistry["joint_angles"] = ProprioceptiveSensorArgs(
    sensor_type=ProprioceptiveSensorType.JOINT_ANGLES,
)


SensorArgsRegistry["gripper_width"] = ProprioceptiveSensorArgs(
    sensor_type=ProprioceptiveSensorType.GRIPPER_WIDTH,
)
