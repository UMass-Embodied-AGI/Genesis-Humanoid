from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import genesis as gs
from genesis.engine.materials.base import Material
from genesis.options.morphs import Morph
from genesis.options.surfaces import Surface

if TYPE_CHECKING:
    from genesis.engine.entities.base_entity import Entity


class BaseSimScene(abc.ABC):
    """
    Base class for all simulated scenes. Acts as a wrapper of gs.Scene with additional
    randomization or generative components, e.g., randomizing terrains or envmap.
    """

    _scene: gs.Scene
    _num_envs: int
    _env_spacing: tuple[float, float]
    _n_envs_per_row: int | None
    _center_envs_at_origin: bool
    _compile_kernels: bool

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def add_entity(
        self,
        morph: Morph,
        material: Material | None = None,
        surface: Surface | None = None,
        visualize_contact: bool = False,
        vis_mode: str | None = None,
    ) -> Entity:
        return self._scene.add_entity(morph, material, surface, visualize_contact, vis_mode)

    def build(self) -> None:
        self._scene.build(
            n_envs=self._num_envs,
            env_spacing=self._env_spacing,
            n_envs_per_row=self._n_envs_per_row,
            center_envs_at_origin=self._center_envs_at_origin,
            compile_kernels=self._compile_kernels,
        )

    def step(self, update_visualizer: bool = True, refresh_visualizer: bool = True) -> None:
        self._scene.step(update_visualizer, refresh_visualizer)

    @property
    def scene(self) -> gs.Scene:
        return self._scene

    @property
    def num_envs(self) -> int:
        return self._num_envs
