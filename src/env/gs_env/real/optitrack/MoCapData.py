# =============================================================================
# Copyright © 2025 NaturalPoint, Inc. All Rights Reserved.
#
# THIS SOFTWARE IS GOVERNED BY THE OPTITRACK PLUGINS EULA AVAILABLE AT
# https://www.optitrack.com/about/legal/eula.html AND/OR FOR DOWNLOAD WITH THE
# APPLICABLE SOFTWARE FILE(S) (“PLUGINS EULA”). BY DOWNLOADING, INSTALLING,
# ACTIVATING AND/OR OTHERWISE USING THE SOFTWARE, YOU ARE AGREEING THAT YOU
# HAVE READ, AND THAT YOU AGREE TO COMPLY WITH AND ARE BOUND BY, THE PLUGINS
# EULA AND ALL APPLICABLE LAWS AND REGULATIONS. IF YOU DO NOT AGREE TO BE
# BOUND BY THE PLUGINS EULA, THEN YOU MAY NOT DOWNLOAD, INSTALL, ACTIVATE OR
# OTHERWISE USE THE SOFTWARE AND YOU MUST PROMPTLY DELETE OR RETURN IT. IF YOU
# ARE DOWNLOADING, INSTALLING, ACTIVATING AND/OR OTHERWISE USING THE SOFTWARE
# ON BEHALF OF AN ENTITY, THEN BY DOING SO YOU REPRESENT AND WARRANT THAT YOU
# HAVE THE APPROPRIATE AUTHORITY TO ACCEPT THE PLUGINS EULA ON BEHALF OF SUCH
# ENTITY.
# See license file in root directory for additional governing terms and
# information.
# =============================================================================


# OptiTrack NatNet direct depacketization sample for Python 3.x
#


# Uses the Python NatNetClient.py library to establish a connection
# (by creating a NatNetClient),and receive data via a NatNet connection
# and decode it using the NatNetClient library.

# Utility functions

import copy
import hashlib
import random

K_SKIP = [0, 0, 1]
K_FAIL = [0, 1, 0]
K_PASS = [1, 0, 0]


# MoCap Frame Classes
class FramePrefixData:
    def __init__(self, frame_number: int) -> None:
        self.frame_number: int = frame_number

    def get_as_string(self, tab_str: str = "  ", level: int = 0) -> str:
        out_tab_str = get_tab_str(tab_str, level)
        out_str = f"{out_tab_str}Frame #: {self.frame_number:3d}\n"
        return out_str


class MarkerData:
    def __init__(self) -> None:
        self.model_name: str | bytes = ""
        self.marker_pos_list: list[list[float] | tuple[float, float, float]] = []

    def set_model_name(self, model_name: str | bytes) -> None:
        self.model_name = model_name

    def add_pos(self, pos: list[float] | tuple[float, float, float]) -> int:
        self.marker_pos_list.append(copy.deepcopy(pos))
        return len(self.marker_pos_list)

    def get_num_points(self) -> int:
        return len(self.marker_pos_list)

    def get_as_string(self, tab_str: str = "  ", level: int = 0) -> str:
        out_tab_str = get_tab_str(tab_str, level)
        out_tab_str2 = get_tab_str(tab_str, level + 1)
        out_str = ""
        out_str += f"{out_tab_str}MarkerData:\n"
        if self.model_name != "":
            out_str += f"{out_tab_str}Model Name : {get_as_string(self.model_name)}\n"
        marker_count = len(self.marker_pos_list)
        out_str += f"{out_tab_str}Marker Count :{marker_count:3d}\n"
        for i in range(marker_count):
            pos = self.marker_pos_list[i]
            out_str += f"{out_tab_str2}Marker {i:3d} pos : [x={pos[0]:3.2f},y={pos[1]:3.2f},z={pos[2]:3.2f}]\n"
        return out_str


class MarkerSetData:
    def __init__(self) -> None:
        self.marker_data_list: list[MarkerData] = []
        self.unlabeled_markers: MarkerData = MarkerData()
        self.unlabeled_markers.set_model_name("")

    def add_marker_data(self, marker_data: MarkerData) -> int:
        self.marker_data_list.append(copy.deepcopy(marker_data))
        return len(self.marker_data_list)

    def add_unlabeled_marker(self, pos: list[float]) -> int:
        return self.unlabeled_markers.add_pos(pos)

    def get_marker_set_count(self) -> int:
        return len(self.marker_data_list)

    def get_unlabeled_marker_count(self) -> int:
        return self.unlabeled_markers.get_num_points()

    def get_as_string(self, tab_str: str = "  ", level: int = 0) -> str:
        out_tab_str = get_tab_str(tab_str, level)

        out_str = ""

        # Labeled markers count
        marker_data_count = len(self.marker_data_list)
        out_str += f"{out_tab_str}Markerset Count:{marker_data_count:3d}\n"
        for marker_data in self.marker_data_list:
            out_str += marker_data.get_as_string(tab_str, level + 1)

        # Unlabeled markers count (4 bytes)
        unlabeled_markers_count = self.unlabeled_markers.get_num_points()
        out_str += f"{out_tab_str}Unlabeled Marker Count:{unlabeled_markers_count:3d}\n"
        out_str += self.unlabeled_markers.get_as_string(tab_str, level + 1)
        return out_str


class LegacyMarkerData:
    def __init__(self) -> None:
        self.marker_pos_list: list[list[float] | tuple[float, float, float]] = []

    def add_pos(self, pos: tuple[float, float, float]) -> int:
        self.marker_pos_list.append(copy.deepcopy(pos))
        return len(self.marker_pos_list)

    def get_marker_count(self) -> int:
        return len(self.marker_pos_list)

    def get_as_string(self, tab_str: str = "  ", level: int = 0) -> str:
        out_tab_str = get_tab_str(tab_str, level)
        out_tab_str2 = get_tab_str(tab_str, level + 1)
        out_str = ""
        marker_count = len(self.marker_pos_list)
        out_str += f"{out_tab_str}Legacy Marker Count :{marker_count:3d}\n"
        for i in range(marker_count):
            pos = self.marker_pos_list[i]
            out_str += f"{out_tab_str2}Marker {i:3d} pos : [x={pos[0]:3.2f},y={pos[1]:3.2f},z={pos[2]:3.2f}]\n"
        return out_str


class RigidBodyMarker:
    def __init__(self) -> None:
        self.pos: list[float] = [0.0, 0.0, 0.0]
        self.id_num: int = 0
        self.size: int = 0
        self.error: float = 0.0
        self.marker_num: int = -1

    def get_as_string(self, tab_str: str = "  ", level: int = 0) -> str:
        out_tab_str = get_tab_str(tab_str, level)
        out_str = ""
        out_str += f"{out_tab_str}RBMarker:"
        if self.marker_num > -1:
            out_str += f" {self.marker_num:3d}"
        out_str += "\n"

        out_str += (
            f"{out_tab_str}Position: [{self.pos[0]:3.2f} {self.pos[1]:3.2f} {self.pos[2]:3.2f}]\n"
        )
        out_str += f"{out_tab_str}ID      : {self.id_num:3d}\n"
        out_str += f"{out_tab_str}Size    : {self.size:3d}\n"
        return out_str


class RigidBody:
    def __init__(
        self,
        new_id: int,
        pos: list[float] | tuple[float, float, float],
        rot: list[float] | tuple[float, float, float, float],
    ) -> None:
        self.id_num: int = new_id
        self.pos: list[float] | tuple[float, float, float] = pos
        self.rot: list[float] | tuple[float, float, float, float] = rot
        self.rb_marker_list: list[RigidBodyMarker] = []
        self.tracking_valid: bool = False
        self.error: float = 0.0
        self.marker_num: int = -1

    def add_rigid_body_marker(self, rigid_body_marker: RigidBodyMarker) -> int:
        self.rb_marker_list.append(copy.deepcopy(rigid_body_marker))
        return len(self.rb_marker_list)

    def get_as_string(self, tab_str: str = "  ", level: int = 0) -> str:
        out_tab_str = get_tab_str(tab_str, level)

        out_str = ""

        # header
        out_str += f"{out_tab_str}Rigid Body    :"
        if self.marker_num > -1:
            out_str += f" {self.marker_num:3d}"
        out_str += "\n"
        print(self.id_num)
        out_str += f"{out_tab_str}  ID            : {self.id_num:3d}\n"
        # Position and orientation
        out_str += f"{out_tab_str}  Position      : [{self.pos[0]:3.2f}, {self.pos[1]:3.2f}, {self.pos[2]:3.2f}]\n"
        out_str += f"{out_tab_str}  Orientation   : [{self.rot[0]:3.2f}, {self.rot[1]:3.2f}, {self.rot[2]:3.2f}, {self.rot[3]:3.2f}]\n"

        marker_count = len(self.rb_marker_list)
        marker_count_range = range(0, marker_count)

        # Marker Data
        if marker_count > 0:
            out_str += f"{out_tab_str}  Marker Count  : {marker_count:3d}\n"
            for i in marker_count_range:
                rbmarker = self.rb_marker_list[i]
                rbmarker.marker_num = i
                out_str += rbmarker.get_as_string(tab_str, level + 2)

        out_str += f"{out_tab_str}  Marker Error  : {self.error:3.2f}\n"

        # Valid Tracking
        tf_string = "False"
        if self.tracking_valid:
            tf_string = "True"
        out_str += f"{out_tab_str}Tracking Valid: {tf_string}\n"

        return out_str


class RigidBodyData:
    def __init__(self) -> None:
        self.rigid_body_list: list[RigidBody] = []

    def add_rigid_body(self, rigid_body: RigidBody) -> int:
        self.rigid_body_list.append(copy.deepcopy(rigid_body))
        return len(self.rigid_body_list)

    def get_rigid_body_count(self) -> int:
        return len(self.rigid_body_list)

    def get_as_string(self, tab_str: str = "  ", level: int = 0) -> str:
        out_tab_str = get_tab_str(tab_str, level)
        out_str = ""
        rigid_body_count = len(self.rigid_body_list)
        out_str += f"{out_tab_str}Rigid Body Count: {rigid_body_count:3d}\n"
        rb_num = 0
        for rigid_body in self.rigid_body_list:
            rigid_body.marker_num = rb_num
            out_str += rigid_body.get_as_string(tab_str, level + 1)
            rb_num += 1
        return out_str


class Skeleton:
    def __init__(self, new_id: int = 0) -> None:
        self.id_num: int = new_id
        self.rigid_body_list: list[RigidBody] = []

    def add_rigid_body(self, rigid_body: RigidBody) -> int:
        self.rigid_body_list.append(copy.deepcopy(rigid_body))
        return len(self.rigid_body_list)

    def get_as_string(self, tab_str: str = "  ", level: int = 0) -> str:
        out_tab_str = get_tab_str(tab_str, level)
        out_str = " "
        out_str += f"{out_tab_str}ID: {self.id_num:3d}\n"
        rigid_body_count = len(self.rigid_body_list)
        out_str += f"{out_tab_str}Rigid Body Count: {rigid_body_count:3d}\n"
        for rb_num in range(rigid_body_count):
            self.rigid_body_list[rb_num].marker_num = rb_num
            out_str += self.rigid_body_list[rb_num].get_as_string(tab_str, level + 2)
        return out_str


class SkeletonData:
    def __init__(self) -> None:
        self.skeleton_list: list[Skeleton] = []

    def add_skeleton(self, new_skeleton: Skeleton) -> None:
        self.skeleton_list.append(copy.deepcopy(new_skeleton))

    def get_skeleton_count(self) -> int:
        return len(self.skeleton_list)

    def get_as_string(self, tab_str: str = "  ", level: int = 0) -> str:
        out_tab_str = get_tab_str(tab_str, level)
        out_tab_str2 = get_tab_str(tab_str, level + 1)

        out_str = ""
        skeleton_count = len(self.skeleton_list)
        out_str += f"{out_tab_str}Skeleton Count: {skeleton_count:3d}\n"
        for skeleton_num in range(skeleton_count):
            out_str += f"{out_tab_str2}Skeleton {skeleton_num:3d}\n"
            out_str += self.skeleton_list[skeleton_num].get_as_string(tab_str, level + 2)
        return out_str


class AssetMarkerData:
    def __init__(
        self,
        marker_id: int,
        pos: tuple[float, float, float],
        marker_size: float = 0.0,
        marker_params: int = 0,
        residual: float = 0.0,
        marker_num: int = -1,
    ) -> None:
        self.marker_id: int = marker_id
        self.pos: tuple[float, float, float] = pos
        self.marker_size: float = marker_size
        self.marker_params: int = marker_params
        self.residual: float = residual
        self.marker_num: int = marker_num

    def get_as_string(self, tab_str: str = "  ", level: int = 0) -> str:
        out_tab_str = get_tab_str(tab_str, level)
        out_str = ""
        out_str += f"{out_tab_str}"
        if self.marker_num > -1:
            out_str += f"{self.marker_num:3d} "
        else:
            out_str += "    "
        out_str += f"Marker {self.marker_id:7d}"
        out_str += f" pos : [{self.pos[0]:3.2f}, {self.pos[1]:3.2f}, {self.pos[2]:3.2f}] "
        out_str += f"       size={self.marker_size:3.2f}"
        out_str += f"       err={self.residual:3.2f}"
        out_str += f"        params={self.marker_params:d}"
        out_str += "\n"

        return out_str


class AssetRigidBodyData:
    def __init__(
        self,
        new_id: int,
        pos: tuple[float, float, float],
        rot: tuple[float, float, float, float],
        mean_error: float = 0.0,
        param: int = 0,
    ) -> None:
        self.id_num: int = new_id
        self.pos: tuple[float, float, float] = pos
        self.rot: tuple[float, float, float, float] = rot
        self.mean_error: float = mean_error
        self.param: int = param
        self.rb_num: int = -1

    def get_as_string(self, tab_str: str = "  ", level: int = 0) -> str:
        out_tab_str = get_tab_str(tab_str, level)
        out_str = ""
        out_str += f"{out_tab_str}Rigid Body :"
        if self.rb_num > -1:
            out_str += f"{self.rb_num:3d}"
        out_str += "\n"
        out_str += f"{out_tab_str}ID          : {get_as_string(self.id_num)}\n"
        out_str += f"{out_tab_str}Position    : [{self.pos[0]:3.2f}, {self.pos[1]:3.2f}, {self.pos[2]:3.2f}]\n"
        out_str += f"{out_tab_str}Orientation : [{self.rot[0]:3.2f}, {self.rot[1]:3.2f}, {self.rot[2]:3.2f}, {self.rot[3]:3.2f}]\n"
        out_str += f"{out_tab_str}Mean Error  : {self.mean_error:3.2f}\n"
        out_str += f"{out_tab_str}Params      : {self.param:3d}\n"

        return out_str


class Asset:
    def __init__(self) -> None:
        self.asset_id: int = 0
        self.rigid_body_list: list[AssetRigidBodyData] = []
        self.marker_list: list[AssetMarkerData] = []

    def set_id(self, new_id: int) -> None:
        self.asset_id = new_id

    def add_rigid_body(self, rigid_body: AssetRigidBodyData) -> int:
        self.rigid_body_list.append(copy.deepcopy(rigid_body))
        return len(self.rigid_body_list)

    def add_marker(self, marker: AssetMarkerData) -> int:
        self.marker_list.append(copy.deepcopy(marker))
        return len(self.marker_list)

    def get_rigid_body_count(self) -> int:
        return len(self.rigid_body_list)

    def get_marker_count(self) -> int:
        return len(self.marker_list)

    def get_as_string(self, tab_str: str = "  ", level: int = 0) -> str:
        out_tab_str = get_tab_str(tab_str, level)

        out_str = ""
        out_str += f"{out_tab_str}Asset ID        : {self.asset_id}\n"
        rigid_body_count = len(self.rigid_body_list)
        out_str += f"{out_tab_str}Rigid Body Count: {rigid_body_count}\n"
        rb_num = 0
        for rigid_body in self.rigid_body_list:
            rigid_body.rb_num = rb_num
            out_str += rigid_body.get_as_string(tab_str, level + 1)
            rb_num += 1

        marker_count = len(self.marker_list)
        out_str += f"{out_tab_str}Marker Count: {marker_count}\n"
        marker_num = 0
        for marker in self.marker_list:
            marker.marker_num = marker_num
            out_str += marker.get_as_string(tab_str, level + 1)
            marker_num += 1
        return out_str


class AssetData:
    def __init__(self) -> None:
        self.asset_list: list[Asset] = []

    def add_asset(self, new_asset: Asset) -> None:
        self.asset_list.append(copy.deepcopy(new_asset))

    def get_asset_count(self) -> int:
        return len(self.asset_list)

    def get_as_string(self, tab_str: str = "  ", level: int = 0) -> str:
        out_tab_str = get_tab_str(tab_str, level)
        out_tab_str2 = get_tab_str(tab_str, level + 1)

        out_str = ""
        asset_count = self.get_asset_count()
        out_str += f"{out_tab_str}Asset Count: {asset_count}\n"
        for asset_num in range(asset_count):
            out_str += f"{out_tab_str2}Asset {asset_num:3d}\n"
            out_str += self.asset_list[asset_num].get_as_string(tab_str, level + 2)
        return out_str


class LabeledMarker:
    def __init__(
        self,
        new_id: int,
        pos: list[float] | tuple[float, float, float],
        size: tuple[float, float, float] | float = 0.0,
        param: int = 0,
        residual: float = 0.0,
    ) -> None:
        self.id_num: int = new_id
        self.pos: list[float] | tuple[float, float, float] = pos
        self.size: tuple[float, float, float] | float = size
        self.param: int = param
        self.residual: float = residual
        self.marker_num: int = -1
        if type(size) is tuple:
            self.size = size[0]

    def __decode_marker_id(self) -> tuple[int, int]:
        model_id = self.id_num >> 16
        marker_id = self.id_num & 0x0000FFFF
        return model_id, marker_id

    def __decode_param(self) -> tuple[bool, bool, bool]:
        occluded = (self.param & 0x01) != 0
        point_cloud_solved = (self.param & 0x02) != 0
        model_solved = (self.param & 0x04) != 0
        return occluded, point_cloud_solved, model_solved

    def get_as_string(self, tab_str: str, level: int) -> str:
        out_tab_str = get_tab_str(tab_str, level)
        model_id, marker_id = self.__decode_marker_id()
        out_str = ""
        out_str += f"{out_tab_str}Labeled Marker"
        if self.marker_num > -1:
            out_str += f" {self.marker_num}"
        out_str += ":\n"
        out_str += f"{out_tab_str}ID                 : [MarkerID: {marker_id:3d}] [ModelID: {model_id:3d}]\n"
        out_str += f"{out_tab_str}pos                : [{self.pos[0]:3.2f}, {self.pos[1]:3.2f}, {self.pos[2]:3.2f}]\n"
        out_str += f"{out_tab_str}size               : [{self.size:3.2f}]\n"
        out_str += f"{out_tab_str}err                : [{self.residual:3.2f}]\n"

        occluded, point_cloud_solved, model_solved = self.__decode_param()
        out_str += f"{out_tab_str}occluded           : [{occluded:3d}]\n"
        out_str += f"{out_tab_str}point_cloud_solved : [{point_cloud_solved:3d}]\n"
        out_str += f"{out_tab_str}model_solved       : [{model_solved:3d}]\n"

        return out_str


class LabeledMarkerData:
    def __init__(self) -> None:
        self.labeled_marker_list: list[LabeledMarker] = []

    def add_labeled_marker(self, labeled_marker: LabeledMarker) -> int:
        self.labeled_marker_list.append(copy.deepcopy(labeled_marker))
        return len(self.labeled_marker_list)

    def get_labeled_marker_count(self) -> int:
        return len(self.labeled_marker_list)

    def get_as_string(self, tab_str: str = "  ", level: int = 0) -> str:
        out_tab_str = get_tab_str(tab_str, level)
        out_str = ""

        labeled_marker_count = len(self.labeled_marker_list)
        out_str += f"{out_tab_str}Labeled Marker Count: {labeled_marker_count:3d}\n"
        for i in range(0, labeled_marker_count):
            labeled_marker = self.labeled_marker_list[i]
            labeled_marker.marker_num = i
            out_str += labeled_marker.get_as_string(tab_str, level + 2)
        return out_str


class ForcePlateChannelData:
    def __init__(self) -> None:
        # list of floats
        self.frame_list: list[float] = []

    def add_frame_entry(self, frame_entry: float) -> int:
        self.frame_list.append(copy.deepcopy(frame_entry))
        return len(self.frame_list)

    def get_as_string(self, tab_str: str, level: int, channel_num: int = -1) -> str:
        fc_max = 4
        out_tab_str = get_tab_str(tab_str, level)

        out_str = ""
        frame_count = len(self.frame_list)
        fc_show = min(frame_count, fc_max)
        out_str += f"{out_tab_str}"
        if channel_num >= 0:
            out_str += f"Channel {channel_num:3d}: "
        out_str += f"{frame_count:3d} Frames - Frame Data: "
        for i in range(fc_show):
            out_str += f"{self.frame_list[i]:3.2f} "
        if fc_show < frame_count:
            out_str += f" - Showing {fc_show:3d} of {frame_count:3d} frames"
        out_str += "\n"
        return out_str


class ForcePlate:
    def __init__(self, new_id: int = 0) -> None:
        self.id_num: int = new_id
        self.channel_data_list: list[ForcePlateChannelData] = []

    def add_channel_data(self, channel_data: ForcePlateChannelData) -> int:
        self.channel_data_list.append(copy.deepcopy(channel_data))
        return len(self.channel_data_list)

    def get_as_string(self, tab_str: str, level: int) -> str:
        out_tab_str = get_tab_str(tab_str, level)
        out_str = ""

        out_str += f"{out_tab_str}ID           : {self.id_num:3d}\n"
        num_channels = len(self.channel_data_list)
        out_str += f"{out_tab_str}  Channel Count: {num_channels:3d}\n"
        for i in range(num_channels):
            out_str += self.channel_data_list[i].get_as_string(tab_str, level + 1, i)
        return out_str


class ForcePlateData:
    def __init__(self) -> None:
        self.force_plate_list: list[ForcePlate] = []

    def add_force_plate(self, force_plate: ForcePlate) -> int:
        self.force_plate_list.append(copy.deepcopy(force_plate))
        return len(self.force_plate_list)

    def get_force_plate_count(self) -> int:
        return len(self.force_plate_list)

    def get_as_string(self, tab_str: str = "  ", level: int = 0) -> str:
        out_tab_str = get_tab_str(tab_str, level)
        out_tab_str2 = get_tab_str(tab_str, level + 1)
        out_str = ""

        force_plate_count = len(self.force_plate_list)
        out_str += f"{out_tab_str}Force Plate Count: {force_plate_count:3d}\n"
        for i in range(force_plate_count):
            out_str += f"{out_tab_str2}Force Plate {i:3d}\n"
            out_str += self.force_plate_list[i].get_as_string(tab_str, level + 2)

        return out_str


class DeviceChannelData:
    def __init__(self) -> None:
        # list of floats
        self.frame_list: list[float] = []

    def add_frame_entry(self, frame_entry: float) -> int:
        self.frame_list.append(copy.deepcopy(frame_entry))
        return len(self.frame_list)

    def get_as_string(self, tab_str: str, level: int, channel_num: int = -1) -> str:
        fc_max = 4
        out_tab_str = get_tab_str(tab_str, level)

        out_str = ""
        frame_count = len(self.frame_list)
        fc_show = min(frame_count, fc_max)
        out_str += f"{out_tab_str}"
        if channel_num >= 0:
            out_str += f"Channel {channel_num:3d}: "
        out_str += f"{frame_count:3d} Frames - Frame Data: "
        for i in range(fc_show):
            out_str += f"{self.frame_list[i]:3.2f} "
        if fc_show < frame_count:
            out_str += f" - Showing {fc_show:3d} of {frame_count:3d} frames"
        out_str += "\n"
        return out_str


class Device:
    def __init__(self, new_id: int) -> None:
        self.id_num: int = new_id
        self.channel_data_list: list[DeviceChannelData] = []

    def add_channel_data(self, channel_data: DeviceChannelData) -> int:
        self.channel_data_list.append(copy.deepcopy(channel_data))
        return len(self.channel_data_list)

    def get_as_string(self, tab_str: str, level: int, device_num: int) -> str:
        out_tab_str = get_tab_str(tab_str, level)

        out_str = ""

        num_channels = len(self.channel_data_list)
        out_str += f"{out_tab_str}Device {device_num:3d}      ID: {self.id_num:3d} Num Channels: {num_channels:3d}\n"
        for i in range(num_channels):
            out_str += self.channel_data_list[i].get_as_string(tab_str, level + 1, i)

        return out_str


class DeviceData:
    def __init__(self) -> None:
        self.device_list: list[Device] = []

    def add_device(self, device: Device) -> int:
        self.device_list.append(copy.deepcopy(device))
        return len(self.device_list)

    def get_device_count(self) -> int:
        return len(self.device_list)

    def get_as_string(self, tab_str: str = "  ", level: int = 0) -> str:
        out_tab_str = get_tab_str(tab_str, level)

        out_str = ""

        device_count = len(self.device_list)
        out_str += f"{out_tab_str}Device Count: {device_count:3d}\n"
        for i in range(device_count):
            out_str += self.device_list[i].get_as_string(tab_str, level + 1, i)
        return out_str


class FrameSuffixData:
    def __init__(self) -> None:
        self.timecode: int | str = -1
        self.timecode_sub: int = -1
        self.timestamp: float = -1.0
        self.stamp_camera_mid_exposure: int = -1
        self.stamp_data_received: int = -1
        self.stamp_transmit: int = -1
        self.prec_timestamp_secs: int = -1
        self.prec_timestamp_frac_secs: int = -1
        self.param: int = 0
        self.is_recording: bool = False
        self.tracked_models_changed: bool = True

    def get_as_string(self, tab_str: str = "  ", level: int = 0) -> str:
        out_tab_str = get_tab_str(tab_str, level)

        if not self.timecode == -1 and not self.timecode_sub == -1:
            self.timecode = stringify_timecode(self.timecode, self.timecode_sub)

        out_str = ""
        if not self.timecode == -1:
            out_str += f"{out_tab_str}Timecode: {self.timecode}\n"
        if not self.timestamp == -1:
            out_str += f"{out_tab_str}Timestamp                      : {self.timestamp:3.3f}\n"
        if not self.stamp_camera_mid_exposure == -1:
            out_str += f"{out_tab_str}Mid-exposure timestamp         : {self.stamp_camera_mid_exposure:3d}\n"
        if not self.stamp_data_received == -1:
            out_str += (
                f"{out_tab_str}Camera data received timestamp : {self.stamp_data_received:3d}\n"
            )
        if not self.stamp_transmit == -1:
            out_str += f"{out_tab_str}Transmit timestamp             : {self.stamp_transmit:3d}\n"
        if not self.prec_timestamp_secs == -1:
            # hours = int(self.prec_timestamp_secs/3600)
            # minutes=int(self.prec_timestamp_secs/60)%60
            # seconds=self.prec_timestamp_secs%60
            # hms_string = """%sPrecision timestamp (hh:mm:ss) : %2d:%2d:
            # %2d\n""" % (out_tab_str, hours, minutes, seconds)
            # out_str += hms_string
            out_str += (
                f"{out_tab_str}Precision timestamp (seconds)  : {self.prec_timestamp_secs:3d}\n"
            )
            if not self.prec_timestamp_frac_secs == -1:
                out_str += f"{out_tab_str}Precision timestamp (fractional seconds) : {self.prec_timestamp_frac_secs:3d}\n"

        return out_str


class MoCapData:
    def __init__(self) -> None:
        # Packet Parts
        self.prefix_data: FramePrefixData | None = None
        self.marker_set_data: MarkerSetData | None = None
        self.legacy_other_markers: LegacyMarkerData | None = None
        self.rigid_body_data: RigidBodyData | None = None
        self.asset_data: AssetData | None = None
        self.skeleton_data: SkeletonData | None = None
        self.labeled_marker_data: LabeledMarkerData | None = None
        self.force_plate_data: ForcePlateData | None = None
        self.device_data: DeviceData | None = None
        self.suffix_data: FrameSuffixData | None = None

    def set_prefix_data(self, new_prefix_data: FramePrefixData) -> None:
        self.prefix_data = new_prefix_data

    def set_marker_set_data(self, new_marker_set_data: MarkerSetData) -> None:
        self.marker_set_data = new_marker_set_data

    def set_legacy_other_markers(self, new_marker_set_data: LegacyMarkerData) -> None:
        self.legacy_other_markers = new_marker_set_data

    def set_rigid_body_data(self, new_rigid_body_data: RigidBodyData) -> None:
        self.rigid_body_data = new_rigid_body_data

    def set_skeleton_data(self, new_skeleton_data: SkeletonData) -> None:
        self.skeleton_data = new_skeleton_data

    def set_asset_data(self, new_asset_data: AssetData) -> None:
        self.asset_data = new_asset_data

    def set_labeled_marker_data(self, new_labeled_marker_data: LabeledMarkerData) -> None:
        self.labeled_marker_data = new_labeled_marker_data

    def set_force_plate_data(self, new_force_plate_data: ForcePlateData) -> None:
        self.force_plate_data = new_force_plate_data

    def set_device_data(self, new_device_data: DeviceData) -> None:
        self.device_data = new_device_data

    def set_suffix_data(self, new_suffix_data: FrameSuffixData) -> None:
        self.suffix_data = new_suffix_data

    def get_as_string(self, tab_str: str = "  ", level: int = 0) -> str:
        out_tab_str = get_tab_str(tab_str, level)

        out_str = ""
        out_str += f"{out_tab_str}MoCap Frame Begin\n{out_tab_str}-----------------\n"
        if self.prefix_data is not None:
            out_str += self.prefix_data.get_as_string()
        else:
            out_str += f"{out_tab_str}No Prefix Data Set\n"

        if self.marker_set_data is not None:
            out_str += self.marker_set_data.get_as_string(tab_str, level + 1)
        else:
            out_str += f"{out_tab_str}No Markerset Data Set\n"

        if self.rigid_body_data is not None:
            out_str += self.rigid_body_data.get_as_string(tab_str, level + 1)
        else:
            out_str += f"{out_tab_str}No Rigid Body Data Set\n"

        if self.skeleton_data is not None:
            out_str += self.skeleton_data.get_as_string(tab_str, level + 1)
        else:
            out_str += f"{out_tab_str}No Skeleton Data Set\n"

        if self.asset_data is not None:
            out_str += self.asset_data.get_as_string(tab_str, level + 1)
        else:
            out_str += f"{out_tab_str}No Asset Data Set\n"

        if self.labeled_marker_data is not None:
            out_str += self.labeled_marker_data.get_as_string(tab_str, level + 1)
        else:
            out_str += f"{out_tab_str}No Labeled Marker Data Set\n"

        if self.force_plate_data is not None:
            out_str += self.force_plate_data.get_as_string(tab_str, level + 1)
        else:
            out_str += f"{out_tab_str}No Force Plate Data Set\n"

        if self.device_data is not None:
            out_str += self.device_data.get_as_string(tab_str, level + 1)
        else:
            out_str += f"{out_tab_str}No Device Data Set\n"

        if self.suffix_data is not None:
            out_str += self.suffix_data.get_as_string(tab_str, level + 1)
        else:
            out_str += f"{out_tab_str}No Suffix Data Set\n"

        out_str += f"{out_tab_str}MoCap Frame End\n{out_tab_str}-----------------\n"

        return out_str


# get_tab_str
# generate a string that takes the nesting level into account
def get_tab_str(tab_str: str, level: int) -> str:
    out_tab_str = ""
    loop_range = range(0, level)
    for _ in loop_range:
        out_tab_str += tab_str
    return out_tab_str


def add_lists(totals: list[int], totals_tmp: list[int]) -> list[int]:
    totals[0] += totals_tmp[0]
    totals[1] += totals_tmp[1]
    totals[2] += totals_tmp[2]
    return totals


def test_hash(
    test_name: str,
    test_hash_str: str,
    test_object: FramePrefixData
    | MarkerSetData
    | LegacyMarkerData
    | RigidBodyData
    | SkeletonData
    | AssetData
    | LabeledMarkerData
    | ForcePlateData
    | DeviceData
    | FrameSuffixData
    | MoCapData,
) -> bool:
    out_str = test_object.get_as_string()
    out_hash_str = hashlib.sha1(out_str.encode()).hexdigest()
    ret_value = True
    if test_hash_str == out_hash_str:
        print(f"[PASS]: {test_name}")
    else:
        print(f"[FAIL]: {test_name} test_hash_str != out_hash_str")
        print(f"test_hash_str={test_hash_str}")
        print(f"out_hash_str={out_hash_str}")
        print(f"out_str =\n{out_str}")
        ret_value = False
    return ret_value


def test_hash2(
    test_name: str,
    test_hash_str: str,
    test_object: FramePrefixData
    | MarkerSetData
    | LegacyMarkerData
    | RigidBodyData
    | SkeletonData
    | AssetData
    | LabeledMarkerData
    | ForcePlateData
    | DeviceData
    | FrameSuffixData
    | MoCapData
    | None,
    generator_string: str,
    run_test: bool,
) -> list[int]:
    ret_value = K_FAIL
    out_str = "FAIL"
    out_str2 = ""
    indent_string = "       "
    if not run_test:
        ret_value = K_SKIP
        out_str = "SKIP"
    elif test_object is None:
        out_str = "FAIL"
        ret_value = K_FAIL
        out_str2 = f"{indent_string}ERROR: test_object was None"
    else:
        obj_out_hash_str = ""
        obj_out_str = ""
        if str(type(test_object)) != "NoneType":
            obj_out_str = test_object.get_as_string()
            obj_out_hash_str = hashlib.sha1(obj_out_str.encode()).hexdigest()

        if test_hash_str == obj_out_hash_str:
            out_str = "PASS"
            ret_value = K_PASS
        else:
            out_str2 += f"{indent_string}{test_name} test_hash_str != out_hash_str\n"
            out_str2 += f"{indent_string}test_hash_str={test_hash_str}\n"
            out_str2 += f"{indent_string}obj_out_hash_str={obj_out_hash_str}\n"
            out_str2 += f"{indent_string}Updated Test Entry:\n"
            out_str2 += f'{indent_string}["{test_name}", "{obj_out_hash_str}", "{generator_string}", True],\n'
            out_str2 += f"{indent_string}obj_out_str =\n{obj_out_str}"

            ret_value = K_FAIL
    print(f"[{out_str}]:{test_name}")

    if len(out_str2):
        print(f"{out_str2}")
    return ret_value


def get_as_string(input_str: str | None | bytes | int) -> str:
    if type(input_str) is str:
        return input_str
    elif type(input_str) is type(None):
        return ""
    elif type(input_str) is bytes:
        return input_str.decode("utf-8")
    elif type(input_str) is int:
        return str(input_str)
    else:
        print(f"type_input_str = {type(input_str)} NOT HANDLED")
        return "<unknown>"


# Timecode Decoding Functions
def decode_timecode(in_timecode: int, in_subframe_timecode: int) -> tuple[int, int, int, int, int]:
    """Takes in timecode and decodes it"""
    hour = (in_timecode >> 24) & 255
    minute = (in_timecode >> 16) & 255
    second = (in_timecode >> 8) & 255
    frame = in_timecode & 255
    subframe = in_subframe_timecode

    return hour, minute, second, frame, subframe


def stringify_timecode(timecode: int | str, timecode_sub: int) -> str:
    """prints out timecode"""
    if type(timecode) is str:
        return timecode
    hour, minute, second, frame, subframe = decode_timecode(timecode, timecode_sub)  # type: ignore
    timecode_string = f"{hour:02}:{minute:02}:{second:02}:{frame:02}:{subframe:02}"  # type: ignore
    return timecode_string


# test program


def generate_prefix_data(frame_num: int = 0) -> FramePrefixData:
    frame_prefix_data = FramePrefixData(frame_num)
    return frame_prefix_data


def generate_label(label_base: str = "label", label_num: int = 0) -> str:
    out_label = f"{label_base}_{label_num:3d}"
    return out_label


def generate_position_srand(pos_num: int = 0, frame_num: int = 0) -> list[float]:
    random.seed(pos_num + (frame_num * 1000))
    position = [(random.random() * 100), (random.random() * 100), (random.random() * 100)]
    return position


def generate_marker_data(label_base: str | None, label_num: int, num_points: int = 1) -> MarkerData:
    if (label_base is None) or (label_base == ""):
        label = ""
    else:
        label = generate_label(label_base, label_num)
    marker_data = MarkerData()
    marker_data.set_model_name(label)
    start_num = label_num * 10000
    end_num = start_num + num_points
    for point_num in range(start_num, end_num):
        position = generate_position_srand(point_num)
        marker_data.add_pos(position)

    return marker_data


def generate_marker_set_data(frame_num: int = 0, marker_set_num: int = 0) -> MarkerSetData:
    marker_set_data = MarkerSetData()
    # add labeled markers
    marker_set_data.add_marker_data(generate_marker_data("marker", 0, 3))
    marker_set_data.add_marker_data(generate_marker_data("marker", 1, 6))
    marker_set_data.add_marker_data(generate_marker_data("marker", 2, 5))
    # add unlabeled markers
    num_points = 5
    start_num = (frame_num * 100000) + (10000 + marker_set_num)
    end_num = start_num + num_points
    for point_num in range(start_num, end_num):
        position = generate_position_srand(point_num)
        marker_set_data.add_unlabeled_marker(position)
    return marker_set_data


def generate_rigid_body_marker_srand(marker_num: int = 0, frame_num: int = 0) -> RigidBodyMarker:
    rigid_body_marker = RigidBodyMarker()
    rbm_num = 11000 + marker_num
    random.seed(rbm_num)
    rigid_body_marker.pos = generate_position_srand(rbm_num, frame_num)
    rigid_body_marker.id_num = marker_num
    rigid_body_marker.size = 1
    rigid_body_marker.error = random.random()

    return rigid_body_marker


def generate_rigid_body(body_num: int = 0, frame_num: int = 0) -> RigidBody:
    pos = generate_position_srand(10000 + body_num, frame_num)
    rot = [1.0, 0.0, 0.0, 0.0]
    rigid_body = RigidBody(body_num, pos, rot)
    rigid_body.add_rigid_body_marker(generate_rigid_body_marker_srand(0, frame_num))
    rigid_body.add_rigid_body_marker(generate_rigid_body_marker_srand(1, frame_num))
    rigid_body.add_rigid_body_marker(generate_rigid_body_marker_srand(2))
    return rigid_body


def generate_rigid_body_data(frame_num: int = 0) -> RigidBodyData:
    rigid_body_data = RigidBodyData()
    # add rigid bodies
    rigid_body_data.add_rigid_body(generate_rigid_body(0, frame_num))
    rigid_body_data.add_rigid_body(generate_rigid_body(1, frame_num))
    rigid_body_data.add_rigid_body(generate_rigid_body(2, frame_num))
    return rigid_body_data


def generate_skeleton(frame_num: int = 0, skeleton_num: int = 0, num_rbs: int = 1) -> Skeleton:
    skeleton = Skeleton(skeleton_num)
    # add rigid bodies
    rb_seed_start = skeleton_num * 165
    rb_seed_end = rb_seed_start + num_rbs
    for rb_num in range(rb_seed_start, rb_seed_end):
        skeleton.add_rigid_body(generate_rigid_body(rb_num, frame_num))
    return skeleton


def generate_skeleton_data(frame_num: int = 0) -> SkeletonData:
    skeleton_data = SkeletonData()
    skeleton_data.add_skeleton(generate_skeleton(frame_num, 0, 2))
    skeleton_data.add_skeleton(generate_skeleton(frame_num, 1, 6))
    skeleton_data.add_skeleton(generate_skeleton(frame_num, 2, 3))
    return skeleton_data


def generate_labeled_marker(frame_num: int = 0, marker_num: int = 0) -> LabeledMarker:
    point_num = (frame_num * 2000) + marker_num
    pos = generate_position_srand(point_num)
    size = 1
    param = 0
    # occluded 0x01
    param += 0x01 * 0
    # point_cloud_solved 0x02
    param += 0x02 * 0
    # model_solved 0x04
    param += 0x04 * 1
    residual = 0.01
    return LabeledMarker(marker_num, pos, size, param, residual)


def generate_labeled_marker_data(frame_num: int = 0) -> LabeledMarkerData:
    labeled_marker_data = LabeledMarkerData()
    # add labeled marker
    labeled_marker_data.add_labeled_marker(generate_labeled_marker(frame_num, 0))
    labeled_marker_data.add_labeled_marker(generate_labeled_marker(frame_num, 1))
    labeled_marker_data.add_labeled_marker(generate_labeled_marker(frame_num, 2))

    return labeled_marker_data


def generate_fp_channel_data(
    frame_num: int = 0, fp_num: int = 0, channel_num: int = 0, num_frames: int = 1
) -> ForcePlateChannelData:
    rseed = (frame_num * 100000) + (fp_num * 10000) + (channel_num * 1000)
    random.seed(rseed)
    fp_channel_data = ForcePlateChannelData()
    for _ in range(num_frames):
        fp_channel_data.add_frame_entry(100.0 * random.random())
    return fp_channel_data


def generate_force_plate(frame_num: int = 0, fp_num: int = 0, num_channels: int = 1) -> ForcePlate:
    force_plate = ForcePlate(fp_num)
    # add channel_data
    for i in range(num_channels):
        force_plate.add_channel_data(generate_fp_channel_data(frame_num, fp_num, i, 10))
    return force_plate


def generate_force_plate_data(frame_num: int = 0) -> ForcePlateData:
    force_plate_data = ForcePlateData()
    # add force plates
    force_plate_data.add_force_plate(generate_force_plate(frame_num, 0, 3))
    force_plate_data.add_force_plate(generate_force_plate(frame_num, 1, 4))
    force_plate_data.add_force_plate(generate_force_plate(frame_num, 2, 2))
    return force_plate_data


def generate_device_channel_data(
    frame_num: int = 0, device_num: int = 0, channel_num: int = 0, num_frames: int = 1
) -> DeviceChannelData:
    rseed = (frame_num * 100000) + (device_num * 10000) + (channel_num * 1000)
    random.seed(rseed)
    device_channel_data = DeviceChannelData()
    for _ in range(num_frames):
        device_channel_data.add_frame_entry(100.0 * random.random())
    return device_channel_data


def generate_device(frame_num: int = 0, device_num: int = 0) -> Device:
    device = Device(device_num)
    device.add_channel_data(generate_device_channel_data(frame_num, device_num, 1, 4))
    device.add_channel_data(generate_device_channel_data(frame_num, device_num, 3, 2))
    device.add_channel_data(generate_device_channel_data(frame_num, device_num, 7, 6))
    return device


def generate_device_data(frame_num: int = 0) -> DeviceData:
    device_data = DeviceData()
    device_data.add_device(generate_device(frame_num, 0))
    device_data.add_device(generate_device(frame_num, 2))
    return device_data


def generate_suffix_data(frame_num: int = 0) -> FrameSuffixData:
    frame_suffix_data = FrameSuffixData()
    frame_suffix_data.stamp_camera_mid_exposure = 5844402979291 + frame_num
    frame_suffix_data.stamp_data_received = 0
    frame_suffix_data.stamp_transmit = 5844403268753 + frame_num
    frame_suffix_data.prec_timestamp_secs = 0
    frame_suffix_data.prec_timestamp_frac_secs = 0
    frame_suffix_data.timecode = 0
    frame_suffix_data.timecode_sub = 0
    frame_suffix_data.timestamp = 762.63
    return frame_suffix_data


def generate_mocap_data(frame_num: int = 0) -> MoCapData:
    mocap_data = MoCapData()

    mocap_data.set_prefix_data(generate_prefix_data(frame_num))
    mocap_data.set_marker_set_data(generate_marker_set_data(frame_num))
    mocap_data.set_rigid_body_data(generate_rigid_body_data(frame_num))
    mocap_data.set_skeleton_data(generate_skeleton_data(frame_num))
    mocap_data.set_labeled_marker_data(generate_labeled_marker_data(frame_num))
    mocap_data.set_force_plate_data(generate_force_plate_data(frame_num))
    mocap_data.set_device_data(generate_device_data(frame_num))
    mocap_data.set_suffix_data(generate_suffix_data(frame_num))

    return mocap_data


def test_all(run_test: bool = True) -> list[int]:
    totals = [0, 0, 0]
    if run_test is True:
        test_cases = [
            [
                "Test Prefix Data 0",
                "bffba016d02cf2167780df31aee697e1ec746b4c",
                "generate_prefix_data(0)",
                True,
            ],
            [
                "Test Markerset Data 0",
                "e56eb605b7b583252f644ca67118aafb7642f49f",
                "generate_marker_set_data(0)",
                True,
            ],
            [
                "Test Rigid Body Data 0",
                "5357b7146719aca7df226dab585b15d1d6096e35",
                "generate_rigid_body_data(0)",
                True,
            ],
            [
                "Test Skeleton Data 0",
                "19b6b8e2f4b4c68d5c67f353bea0b09d10343074",
                "generate_skeleton_data(0)",
                True,
            ],
            [
                "Test Labeled Marker Data 0",
                "e0dd01035424e8e927a4956c21819a1f0ed18355",
                "generate_labeled_marker_data(0)",
                True,
            ],
            [
                "Test Force Plate Data 0",
                "2bb1000049a98b3c4ff8c48c7560af94dcdd32b3",
                "generate_force_plate_data(0)",
                True,
            ],
            [
                "Test Device Data 0",
                "be10f0b93a7ba3858dce976b7868c1f79fd719c3",
                "generate_device_data(0)",
                True,
            ],
            [
                "Test Suffix Data 0",
                "005a1b3e1f9e7530255ca75f34e4786cef29fcdb",
                "generate_suffix_data(0)",
                True,
            ],
            [
                "Test MoCap Data 0",
                "1f85afac1eb790d431a4f5936b44a8555a316122",
                "generate_mocap_data(0)",
                True,
            ],
        ]
        num_tests = len(test_cases)
        for i in range(num_tests):
            data = eval(test_cases[i][2])
            totals_tmp = test_hash2(
                test_cases[i][0], test_cases[i][1], data, test_cases[i][2], test_cases[i][3]
            )
            totals = add_lists(totals, totals_tmp)

    print("--------------------")
    print(f"[PASS] Count = {totals[0]:3d}")
    print(f"[FAIL] Count = {totals[1]:3d}")
    print(f"[SKIP] Count = {totals[2]:3d}")

    return totals


if __name__ == "__main__":
    test_all(True)
