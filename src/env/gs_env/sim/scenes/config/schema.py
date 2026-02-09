from typing import Any, TypeAlias

import genesis as gs
from gs_schemas.base_types import genesis_pydantic_config
from pydantic import BaseModel


class BaseSceneArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)
    center_envs_at_origin: bool
    compile_kernels: bool

    scene_type: str

    sim_options: gs.options.SimOptions
    tool_options: gs.options.ToolOptions
    rigid_options: gs.options.RigidOptions
    mpm_options: gs.options.MPMOptions
    fem_options: gs.options.FEMOptions
    sf_options: gs.options.SFOptions
    vis_options: gs.options.VisOptions
    viewer_options: gs.options.ViewerOptions

    show_viewer: bool
    show_FPS: bool


class FlatSceneArgs(BaseSceneArgs):
    model_config = genesis_pydantic_config(frozen=True)
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)


class CustomSceneArgs(BaseSceneArgs):
    model_config = genesis_pydantic_config(frozen=True)
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    remove_ground: bool = False
    objects: list[dict[Any, Any]] = []


SceneArgs: TypeAlias = FlatSceneArgs | CustomSceneArgs
