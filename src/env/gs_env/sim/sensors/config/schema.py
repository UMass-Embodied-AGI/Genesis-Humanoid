from typing import TypeAlias

from gs_schemas.base_types import GenesisEnum, genesis_pydantic_config
from pydantic import BaseModel

# ------------------------------------------------------------
# Camera
# ------------------------------------------------------------


class OakCameraArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)
    # aligned with real
    silent: bool
    resolution: tuple[int, int]
    fps: int
    exposure: float | None
    white_balance: float | None

    # sim-specific
    name: str
    pos: tuple[float, float, float]
    lookat: tuple[float, float, float]
    fov: int  # TODO: cannot set hfov
    # TODO: no intrinsic (maybe focus_dist but not sure about the conversion)
    GUI: bool


class RealSenseCameraArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)


class ZEDCameraArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)


CameraArgs: TypeAlias = OakCameraArgs | RealSenseCameraArgs | ZEDCameraArgs


# ------------------------------------------------------------
# Proprioceptive
# ------------------------------------------------------------


class ProprioceptiveSensorType(GenesisEnum):
    EE_LINK_POS = "EE_LINK_POS"
    EE_LINK_QUAT = "EE_LINK_QUAT"
    JOINT_ANGLES = "JOINT_ANGLES"
    GRIPPER_WIDTH = "GRIPPER_WIDTH"


class ProprioceptiveSensorArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)
    sensor_type: ProprioceptiveSensorType


SensorArgs: TypeAlias = CameraArgs | ProprioceptiveSensorArgs
