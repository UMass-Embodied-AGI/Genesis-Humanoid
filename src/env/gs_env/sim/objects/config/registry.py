from typing import TypeAlias

import genesis as gs

from gs_env.sim.objects.config.schema import (
    BoxMorphArgs,
    MeshObjectArgs,
    ObjectArgs,
    PrimitiveMorphArgs,
    PrimitiveObjectArgs,
    RigidMaterialArgs,
)

# ------------------------------------------------------------
# CoACD
# ------------------------------------------------------------

CoACDArgsRegistry: dict[str, gs.options.misc.CoacdOptions] = {}

CoACDArgsRegistry["default"] = gs.options.misc.CoacdOptions(
    threshold=0.1,
    max_convex_hull=-1,
    preprocess_mode="auto",
    preprocess_resolution=30,
    resolution=1000,
    mcts_nodes=20,
    mcts_iterations=100,
    mcts_max_depth=3,
    pca=False,
    merge=True,
    decimate=False,
    max_ch_vertex=256,
    extrude=False,
    extrude_margin=0.1,
    apx_mode="ch",
    seed=0,
)

CoACDArgsRegistry["hq"] = gs.options.misc.CoacdOptions(
    threshold=0.1,
    max_convex_hull=-1,
    preprocess_mode="auto",
    preprocess_resolution=150,
    resolution=1000,
    mcts_nodes=20,
    mcts_iterations=100,
    mcts_max_depth=3,
    pca=False,
    merge=True,
    decimate=False,
    max_ch_vertex=256,
    extrude=False,
    extrude_margin=0.1,
    apx_mode="ch",
    seed=0,
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


# ------------------------------------------------------------
# Morph
# ------------------------------------------------------------

MorphArgs: TypeAlias = PrimitiveMorphArgs


MorphArgsRegistry: dict[str, MorphArgs] = {}


MorphArgsRegistry["box_default"] = BoxMorphArgs(
    pos=(0.3, 0.0, 0.021),
    euler=(0, 0, 0),
    quat=None,
    visualization=True,
    collision=True,
    requires_jac_and_IK=True,
    is_free=True,
    fixed=False,
    lower=None,
    upper=None,
    size=(0.04, 0.04, 0.04),
)


# ------------------------------------------------------------
# Objects
# ------------------------------------------------------------

ObjectArgsRegistry: dict[str, ObjectArgs] = {}


ObjectArgsRegistry["box_default"] = PrimitiveObjectArgs(
    material_args=MaterialArgsRegistry["default"],
    morph_args=MorphArgsRegistry["box_default"],
    # TODO: surface
    visualize_contact=False,
    vis_mode="visual",
)


ObjectArgsRegistry["mesh_default"] = MeshObjectArgs(
    file="", up=(0, 0, 1), front=(1, 0, 0), scale=1.0, coacd_options=CoACDArgsRegistry["default"]
)
