from typing import Any

import genesis as gs
import torch

from gs_env.common.bases.base_scene import BaseSimScene
from gs_env.common.utils.asset_utils import resolve_file_path
from gs_env.common.utils.math_utils import quat_apply, quat_from_euler, quat_mul
from gs_env.sim.scenes.config.schema import CustomSceneArgs
from gs_env.sim.scenes.flat_scene import FlatScene


class CustomScene(FlatScene):
    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        args: CustomSceneArgs,
        show_viewer: bool = False,
        show_fps: bool = False,
        n_envs_per_row: int | None = None,
        env_spacing: tuple[float, float] = (1.0, 1.0),
        img_resolution: tuple[int, int] | None = None,
    ) -> None:
        BaseSimScene.__init__(self)
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
        if not args.remove_ground:
            self._plane = self._scene.add_entity(
                gs.morphs.Plane(normal=args.normal),
            )
        self._objects = {}
        self._objects_offset = {}
        for object in args.objects:
            obj_type: str = object.get("type", "")
            if obj_type.lower() in ["obj", "stl", "ply"]:
                obj = self._scene.add_entity(
                    gs.morphs.Mesh(
                        file=resolve_file_path(object["path"]),
                        scale=object.get("scale", 1.0),
                        pos=object["position"],
                        euler=object["orientation"],
                        fixed=object.get("fixed", True),
                        visualization=object.get("visualization", True),
                        collision=object.get("collision", True),
                    ),
                    surface=gs.surfaces.Plastic(color=object.get("color", (0.5, 0.5, 0.5))),
                )
            elif obj_type.lower() == "urdf":
                obj = self._scene.add_entity(
                    gs.morphs.URDF(
                        file=resolve_file_path(object["path"]),
                        pos=object["position"],
                        euler=object["orientation"],
                        fixed=object.get("fixed", True),
                        visualization=object.get("visualization", True),
                        collision=object.get("collision", True),
                    )
                )
            elif obj_type.lower() == "box":
                obj = self._scene.add_entity(
                    gs.morphs.Box(
                        size=object["size"],
                        pos=object["position"],
                        euler=object["orientation"],
                        fixed=object.get("fixed", True),
                        visualization=object.get("visualization", True),
                        collision=object.get("collision", True),
                    )
                )
            elif obj_type.lower() == "cylinder":
                obj = self._scene.add_entity(
                    gs.morphs.Cylinder(
                        radius=object["radius"],
                        height=object["height"],
                        pos=object["position"],
                        euler=object["orientation"],
                        fixed=object.get("fixed", True),
                        visualization=object.get("visualization", True),
                        collision=object.get("collision", True),
                    )
                )
            elif obj_type.lower() == "sphere":
                obj = self._scene.add_entity(
                    gs.morphs.Sphere(
                        radius=object["radius"],
                        pos=object["position"],
                        fixed=object.get("fixed", True),
                        visualization=object.get("visualization", True),
                        collision=object.get("collision", True),
                    )
                )
            else:
                raise ValueError(f"Unsupported object type: {obj_type}")
            self._objects[object["name"]] = obj
            pos_offset = object.get("pos_offset", (0.0, 0.0, 0.0))
            rot_offset = torch.tensor(
                object.get("rot_offset", (0.0, 0.0, 0.0)), device=self._device
            )
            quat_offset = quat_from_euler(rot_offset)
            self._objects_offset[object["name"]] = (
                torch.tensor(pos_offset, device=self._device)[None, :].repeat(num_envs, 1),
                quat_offset[None, :].repeat(num_envs, 1),
            )
            print(f"Added object: {object['name']}")

        #
        self._num_envs = num_envs
        self._env_spacing = env_spacing
        self._n_envs_per_row = n_envs_per_row
        self._center_envs_at_origin = args.center_envs_at_origin
        self._compile_kernels = args.compile_kernels

    def set_obj_pose(
        self,
        name: str,
        envs_idx: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        quat: torch.Tensor | None = None,
    ) -> None:
        assert name in self._objects, f"Object {name} not found in scene"
        if envs_idx is None:
            envs_idx = torch.arange(self.num_envs, device=self._device)
        if pos is not None:
            assert pos.shape == (len(envs_idx), 3), (
                "Position must be a tensor of shape (num_envs, 3)"
            )
        if quat is not None:
            assert quat.shape == (len(envs_idx), 4), (
                "Quaternion must be a tensor of shape (num_envs, 4)"
            )
        obj = self._objects[name]
        pos_offset = quat_apply(quat, self._objects_offset[name][0][envs_idx])
        quat_offset = quat_mul(quat, self._objects_offset[name][1][envs_idx])
        if pos is not None:
            obj.set_pos(pos + pos_offset, envs_idx=envs_idx)
        if quat is not None:
            obj.set_quat(quat_offset, envs_idx=envs_idx)

    @property
    def objects(self) -> dict[str, Any]:
        return self._objects
