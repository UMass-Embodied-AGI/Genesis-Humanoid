from typing import TypeAlias

import genesis as gs
from gs_schemas.base_types import genesis_pydantic_config
from pydantic import BaseModel


class RigidMaterialArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)
    rho: float
    friction: float | None
    needs_coup: bool
    coup_friction: float
    coup_softness: float
    coup_restitution: float
    sdf_cell_size: float
    sdf_min_res: int
    sdf_max_res: int
    gravity_compensation: float


class PrimitiveMorphArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)
    pos: tuple[float, float, float]
    euler: tuple[int, int, int]
    quat: tuple[float, float, float, float] | None
    visualization: bool
    collision: bool
    requires_jac_and_IK: bool
    is_free: bool

    fixed: bool


class BoxMorphArgs(PrimitiveMorphArgs):
    model_config = genesis_pydantic_config(frozen=True, arbitrary_types_allowed=True)
    lower: tuple[float, float, float] | None
    upper: tuple[float, float, float] | None
    size: tuple[float, float, float] | None


class CylinderMorphArgs(PrimitiveMorphArgs):
    model_config = genesis_pydantic_config(frozen=True)
    height: float
    radius: float


class SphereMorphArgs(PrimitiveMorphArgs):
    model_config = genesis_pydantic_config(frozen=True)
    radius: float


class PrimitiveObjectArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True, arbitrary_types_allowed=True)
    material_args: RigidMaterialArgs
    morph_args: PrimitiveMorphArgs
    visualize_contact: bool
    vis_mode: str


class MeshObjectArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True, arbitrary_types_allowed=True)
    file: str
    up: tuple[int, int, int]
    front: tuple[int, int, int]
    scale: float
    coacd_options: gs.options.misc.CoacdOptions


class PartNetMobilityObjectArgs(MeshObjectArgs):
    model_config = genesis_pydantic_config(frozen=True)


class ObjaverseObjectArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)


class BlenderkitObjectArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)


ObjectArgs: TypeAlias = (
    PrimitiveObjectArgs
    | MeshObjectArgs
    | PartNetMobilityObjectArgs
    | ObjaverseObjectArgs
    | BlenderkitObjectArgs
)
