from typing import Any

import genesis as gs
import torch

from gs_env.common.bases.base_scene import BaseSimScene
from gs_env.sim.scenes.config.schema import FlatSceneArgs


class FlatScene(BaseSimScene):
    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        args: FlatSceneArgs,
        show_viewer: bool = False,
        show_fps: bool = False,
        n_envs_per_row: int | None = None,
        env_spacing: tuple[float, float] = (1.0, 1.0),
        img_resolution: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self._device = device
        #
        # _renderer = (
        #     gs.options.renderers.BatchRenderer(
        #         use_rasterizer=False,
        #     )
        # )
        self._scene = gs.Scene(
            sim_options=args.sim_options,
            tool_options=args.tool_options,
            rigid_options=args.rigid_options,
            mpm_options=args.mpm_options,
            fem_options=args.fem_options,
            sf_options=args.sf_options,
            vis_options=args.vis_options,
            viewer_options=args.viewer_options,
            show_FPS=show_fps,
            show_viewer=show_viewer,
            # renderer=_renderer,
        )
        self._gravity = abs(self._scene.gravity[2].item())
        #
        self._plane = self._scene.add_entity(
            gs.morphs.Plane(normal=args.normal),
        )
        #
        self._num_envs = num_envs
        self._env_spacing = env_spacing
        self._n_envs_per_row = n_envs_per_row
        self._center_envs_at_origin = args.center_envs_at_origin
        self._compile_kernels = args.compile_kernels

    def reset(self, envs_idx: torch.Tensor) -> None:
        self._scene.reset(envs_idx=envs_idx)

    def __getattr__(self, item: str) -> Any:
        if hasattr(self._scene, item):
            return getattr(self._scene, item)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    @property
    def scene(self) -> gs.Scene:
        """Returns the underlying genesis scene."""
        return self._scene

    @property
    def gravity(self) -> float:
        return self._gravity

    @property
    def env_spacing(self) -> tuple[float, float]:
        """Returns the spacing between environments."""
        return self._env_spacing

    @property
    def n_envs_per_row(self) -> int | None:
        return self._n_envs_per_row

    @property
    def center_envs_at_origin(self) -> bool:
        return self._center_envs_at_origin

    @property
    def num_envs(self) -> int:
        return self._num_envs
