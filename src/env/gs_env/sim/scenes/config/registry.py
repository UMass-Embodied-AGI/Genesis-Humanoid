import genesis as gs

from gs_env.sim.scenes.config.schema import (
    CustomSceneArgs,
    FlatSceneArgs,
    SceneArgs,
)

# NOTE: break dependencies on default values from Genesis to avoid silent bugs
# due to any changes in the Genesis repo.


# ------------------------------------------------------------
# Sim
# ------------------------------------------------------------

SimArgsRegistry: dict[str, gs.options.SimOptions] = {}

SimArgsRegistry["default"] = gs.options.SimOptions(
    dt=1e-2,
    substeps=1,
    substeps_local=None,
    gravity=(0.0, 0.0, -9.81),
    floor_height=0.0,
    requires_grad=False,
)


SimArgsRegistry["legged"] = gs.options.SimOptions(
    dt=5e-3,
    substeps=1,
    substeps_local=None,
    gravity=(0.0, 0.0, -9.81),
    floor_height=0.0,
    requires_grad=False,
)


# ------------------------------------------------------------
# Tool
# ------------------------------------------------------------

ToolArgsRegistry: dict[str, gs.options.ToolOptions] = {}

ToolArgsRegistry["default"] = gs.options.ToolOptions(
    dt=None,
    floor_height=0.0,
)


# ------------------------------------------------------------
# Rigid
# ------------------------------------------------------------

RigidArgsRegistry: dict[str, gs.options.RigidOptions] = {}

RigidArgsRegistry["default"] = gs.options.RigidOptions(
    dt=None,
    gravity=None,
    enable_collision=True,
    enable_joint_limit=True,
    enable_self_collision=False,
    enable_adjacent_collision=False,
    max_collision_pairs=100,
    integrator=gs.integrator.approximate_implicitfast,
    IK_max_targets=6,
    constraint_solver=gs.constraint_solver.CG,
    iterations=100,
    tolerance=1e-5,
    ls_iterations=50,
    ls_tolerance=1e-2,
    sparse_solve=False,
    contact_resolve_time=None,
    use_contact_island=False,
    use_hibernation=False,
    hibernation_thresh_vel=1e-3,
    hibernation_thresh_acc=1e-2,
    box_box_detection=False,
)


# ------------------------------------------------------------
# MPM
# ------------------------------------------------------------

MPMArgsRegistry: dict[str, gs.options.MPMOptions] = {}

MPMArgsRegistry["default"] = gs.options.MPMOptions(
    dt=None,
    gravity=None,
    particle_size=None,
    grid_density=64,
    enable_CPIC=False,
    lower_bound=(-1.0, -1.0, 0.0),
    upper_bound=(1.0, 1.0, 1.0),
    use_sparse_grid=False,
    leaf_block_size=8,
)


# ------------------------------------------------------------
# FEM
# ------------------------------------------------------------

FEMArgsRegistry: dict[str, gs.options.FEMOptions] = {}

FEMArgsRegistry["default"] = gs.options.FEMOptions(
    dt=None,
    gravity=None,
    damping=0.0,
    floor_height=0.0,
)


# ------------------------------------------------------------
# SF
# ------------------------------------------------------------

SFArgsRegistry: dict[str, gs.options.SFOptions] = {}

SFArgsRegistry["default"] = gs.options.SFOptions(
    dt=None,
    res=128,
    solver_iters=500,
    decay=0.99,
    T_low=1.0,
    T_high=0.0,
    inlet_pos=(0, 0, 0),
    inlet_vel=(0, 0, 1),
    inlet_quat=(1, 0, 0, 0),
    inlet_s=400.0,
)


# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------

VisArgsRegistry: dict[str, gs.options.VisOptions] = {}

VisArgsRegistry["default"] = gs.options.VisOptions(
    show_world_frame=True,
    world_frame_size=1.0,
    show_link_frame=False,
    link_frame_size=0.2,
    show_cameras=False,
    shadow=True,
    plane_reflection=False,
    env_separate_rigid=False,
    background_color=(0.04, 0.08, 0.12),
    ambient_light=(0.1, 0.1, 0.1),
    visualize_mpm_boundary=False,
    visualize_sph_boundary=False,
    visualize_pbd_boundary=False,
    segmentation_level="link",
    render_particle_as="sphere",
    particle_size_scale=1.0,
    contact_force_scale=0.01,
    n_support_neighbors=12,
    n_rendered_envs=None,
    lights=[
        {"type": "directional", "dir": (-1, -1, -1), "color": (1.0, 1.0, 1.0), "intensity": 5.0},
    ],
)


# ------------------------------------------------------------
# Viewer
# ------------------------------------------------------------

ViewerArgsRegistry: dict[str, gs.options.ViewerOptions] = {}

ViewerArgsRegistry["default"] = gs.options.ViewerOptions(
    res=None,
    refresh_rate=60,
    max_FPS=60,
    camera_pos=(3.5, 0.5, 2.5),
    camera_lookat=(0.0, 0.0, 0.5),
    camera_up=(0.0, 0.0, 1.0),
    camera_fov=40,
)


# ------------------------------------------------------------
# Scene
# ------------------------------------------------------------


SceneArgsRegistry: dict[str, SceneArgs] = {}


SceneArgsRegistry["flat_scene_default"] = FlatSceneArgs(
    scene_type="FlatScene",
    show_viewer=False,
    show_FPS=False,
    center_envs_at_origin=True,
    compile_kernels=True,
    sim_options=SimArgsRegistry["default"],
    tool_options=ToolArgsRegistry["default"],
    rigid_options=gs.options.RigidOptions(
        enable_joint_limit=True,
        enable_collision=True,
        gravity=(0, 0, -9.8),
        box_box_detection=True,
    ),
    mpm_options=MPMArgsRegistry["default"],
    fem_options=FEMArgsRegistry["default"],
    sf_options=SFArgsRegistry["default"],
    vis_options=VisArgsRegistry["default"],
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(-0.6, 0.0, 0.7),
        camera_lookat=(0.2, 0.0, 0.1),
        camera_fov=50,
        max_FPS=60,
    ),
    normal=(0.0, 0.0, 1.0),
)


SceneArgsRegistry["flat_scene_legged"] = FlatSceneArgs(
    scene_type="FlatScene",
    show_viewer=False,
    show_FPS=False,
    center_envs_at_origin=True,
    compile_kernels=True,
    sim_options=SimArgsRegistry["legged"],
    tool_options=ToolArgsRegistry["default"],
    rigid_options=gs.options.RigidOptions(
        enable_joint_limit=True,
        enable_collision=True,
        gravity=(0, 0, -9.8),
        box_box_detection=True,
    ),
    mpm_options=MPMArgsRegistry["default"],
    fem_options=FEMArgsRegistry["default"],
    sf_options=SFArgsRegistry["default"],
    vis_options=VisArgsRegistry["default"],
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(-2.0, 0.0, 0.6),
        camera_lookat=(0.0, 0.0, 0.6),
        camera_fov=50,
        max_FPS=60,
    ),
    normal=(0.0, 0.0, 1.0),
)


SceneArgsRegistry["custom_scene_desk"] = CustomSceneArgs(
    scene_type="CustomScene",
    show_viewer=False,
    show_FPS=False,
    center_envs_at_origin=True,
    compile_kernels=True,
    sim_options=SimArgsRegistry["legged"],
    tool_options=ToolArgsRegistry["default"],
    rigid_options=gs.options.RigidOptions(
        enable_joint_limit=True,
        enable_collision=True,
        gravity=(0, 0, -9.8),
        box_box_detection=True,
    ),
    mpm_options=MPMArgsRegistry["default"],
    fem_options=FEMArgsRegistry["default"],
    sf_options=SFArgsRegistry["default"],
    vis_options=VisArgsRegistry["default"],
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(-2.0, 0.0, 0.6),
        camera_lookat=(0.0, 0.0, 0.6),
        camera_fov=50,
        max_FPS=60,
    ),
    normal=(0.0, 0.0, 1.0),
    remove_ground=False,
    objects=[
        {
            "type": "obj",
            "path": "assets/scene/desk/desk.obj",
            "scale": 1.0,
            "position": (0.0, 0.0, 0.0),
            "orientation": (90.0, 0.0, -90.0),
            "fixed": True,
            "visualization": True,
            "collision": False,
        },
        # {
        #     "type": "box",
        #     "size": (0.4, 0.4, 0.4),
        #     "position": (2.0, 0.0, 0.0),
        #     "orientation": (0.0, 0.0, 0.0),
        #     "fixed": True,
        #     "visualization": True,
        #     "collision": True,
        # },
    ],
)


SceneArgsRegistry["custom_scene_g1_mocap"] = CustomSceneArgs(
    scene_type="CustomScene",
    show_viewer=False,
    show_FPS=False,
    center_envs_at_origin=True,
    compile_kernels=True,
    sim_options=SimArgsRegistry["legged"],
    tool_options=ToolArgsRegistry["default"],
    rigid_options=gs.options.RigidOptions(
        enable_joint_limit=True,
        enable_collision=True,
        gravity=(0, 0, -9.8),
        box_box_detection=True,
        # noslip_iterations=5,
    ),
    mpm_options=MPMArgsRegistry["default"],
    fem_options=FEMArgsRegistry["default"],
    sf_options=SFArgsRegistry["default"],
    vis_options=VisArgsRegistry["default"],
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(-2.0, 0.0, 0.6),
        camera_lookat=(0.0, 0.0, 0.6),
        camera_fov=50,
        max_FPS=60,
    ),
    normal=(0.0, 0.0, 1.0),
    remove_ground=False,
    objects=[
        {
            "name": "left_ankle_roll_link",
            "type": "obj",
            "path": "assets/robot/unitree_g1/meshes/left_ankle_roll_link.STL",
            "scale": 1.0,
            "position": (0.0, 0.0, 0.0),
            "orientation": (0.0, 0.0, 0.0),
            "fixed": True,
            "visualization": True,
            "collision": False,
            "color": (0.0, 0.0, 1.0),
        },
        {
            "name": "right_ankle_roll_link",
            "type": "obj",
            "path": "assets/robot/unitree_g1/meshes/right_ankle_roll_link.STL",
            "scale": 1.0,
            "position": (0.0, 0.0, 0.0),
            "orientation": (0.0, 0.0, 0.0),
            "fixed": True,
            "visualization": True,
            "collision": False,
            "color": (0.0, 0.0, 1.0),
        },
        {
            "name": "pelvis",
            "type": "obj",
            "path": "assets/robot/unitree_g1/meshes/pelvis_contour_link.STL",
            "scale": 1.0,
            "position": (0.0, 0.0, 0.0),
            "orientation": (0.0, 0.0, 0.0),
            "fixed": True,
            "visualization": True,
            "collision": False,
            "color": (0.0, 0.0, 1.0),
        },
        {
            "name": "torso_link",
            "type": "obj",
            "path": "assets/robot/unitree_g1/meshes/torso_link_rev_1_0.STL",
            "scale": 1.0,
            "position": (0.0, 0.0, 0.0),
            "orientation": (0.0, 0.0, 0.0),
            "fixed": True,
            "visualization": True,
            "collision": False,
            "color": (0.0, 0.0, 1.0),
        },
        {
            "name": "left_wrist_yaw_link",
            "type": "obj",
            "path": "assets/robot/unitree_g1/meshes/left_rubber_hand.STL",
            "scale": 1.0,
            "position": (0.0, 0.0, 0.0),
            "orientation": (0.0, 0.0, 0.0),
            "pos_offset": (0.0415, 0.003, 0.0),
            "fixed": True,
            "visualization": True,
            "collision": False,
            "color": (0.0, 0.0, 1.0),
        },
        {
            "name": "right_wrist_yaw_link",
            "type": "obj",
            "path": "assets/robot/unitree_g1/meshes/right_rubber_hand.STL",
            "scale": 1.0,
            "position": (0.0, 0.0, 0.0),
            "orientation": (0.0, 0.0, 0.0),
            "pos_offset": (0.0415, -0.003, 0.0),
            "fixed": True,
            "visualization": True,
            "collision": False,
            "color": (0.0, 0.0, 1.0),
        },
    ],
)
