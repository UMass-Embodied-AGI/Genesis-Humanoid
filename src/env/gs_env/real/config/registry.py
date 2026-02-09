from gs_env.real.config.schema import EnvArgs, OptitrackEnvArgs

# ------------------------------------------------------------
# Manipulation
# ------------------------------------------------------------


EnvArgsRegistry: dict[str, EnvArgs] = {}

EnvArgsRegistry["g1_links_tracking"] = OptitrackEnvArgs(
    server_ip="192.168.0.232",
    client_ip="192.168.0.129",
    use_multicast=False,
    offset_config="./config/optitrack_offset/calibrated.yaml",
    tracked_link_names=[
        "pelvis_contour_link",
        "torso_link",
        "left_rubber_hand",
        "right_rubber_hand",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ],
)
