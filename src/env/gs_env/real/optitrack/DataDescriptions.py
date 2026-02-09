# =============================================================================
# Copyright © 2025 NaturalPoint, Inc. All Rights Reserved.
#
# THIS SOFTWARE IS GOVERNED BY THE OPTITRACK PLUGINS EULA AVAILABLE AT
#  https://www.optitrack.com/about/legal/eula.html AND/OR FOR DOWNLOAD WITH
# THE APPLICABLE SOFTWARE FILE(S) (“PLUGINS EULA”). BY DOWNLOADING, INSTALLING,
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


# Uses the Python NatNetClient.py library to establish a connection (by
# creating a NatNetClient), and receive data via a NatNet connection and
# decode it using the NatNetClient library.


import copy
import hashlib
import random

K_SKIP = [0, 0, 1]
K_FAIL = [0, 1, 0]
K_PASS = [1, 0, 0]


# cMarkerSetDescription
class MarkerSetDescription:
    def __init__(self) -> None:
        self.marker_set_name = "Not Set"
        self.marker_names_list = []

    def set_name(self, new_name: str | bytes) -> None:
        self.marker_set_name = new_name

    def get_num_markers(self) -> int:
        return len(self.marker_names_list)

    def add_marker_name(self, marker_name: str | bytes) -> int:
        self.marker_names_list.append(copy.copy(marker_name))
        return self.get_num_markers()

    def get_as_string(self, tab_str: str = "  ", level: int = 0) -> str:
        out_tab_str = get_tab_str(tab_str, level)
        out_tab_str2 = get_tab_str(tab_str, level + 1)
        out_tab_str3 = get_tab_str(tab_str, level + 2)
        out_string = ""
        out_string += f"{out_tab_str}Markerset Name: {get_as_string(self.marker_set_name)}\n"
        num_markers = len(self.marker_names_list)
        out_string += f"{out_tab_str2}Marker Count   : {num_markers}\n"
        for i in range(num_markers):
            out_string += (
                f"{out_tab_str3}{i:3.1d} Marker Name: {get_as_string(self.marker_names_list[i])}\n"
            )
        return out_string


class RBMarker:
    def __init__(
        self,
        marker_name: str = "",
        active_label: int = 0,
        pos: list[float] | tuple[float, float, float] | None = None,
    ) -> None:
        if pos is None:
            pos = [0.0, 0.0, 0.0]
        self.marker_name = marker_name
        self.active_label = active_label
        self.pos = pos

    def get_as_string(self, tab_str: str = "  ", level: int = 0) -> str:
        out_tab_str = get_tab_str(tab_str, level)
        out_string = ""
        out_string += f"{out_tab_str}Marker Label: {self.active_label} Position: [{self.pos[0]:3.2f} {self.pos[1]:3.2f} {self.pos[2]:3.2f}] {self.marker_name}\n"
        return out_string


class RigidBodyDescription:
    def __init__(
        self,
        sz_name: str = "",
        new_id: int = 0,
        parent_id: int = 0,
        pos: list[float] | None = None,
    ) -> None:
        if pos is None:
            pos = [0.0, 0.0, 0.0]
        self.sz_name = sz_name
        self.id_num = new_id
        self.parent_id = parent_id
        self.pos = pos
        self.rb_marker_list = []
        self.rb_num = -1

    def set_name(self, new_name: str | bytes) -> None:
        self.sz_name = new_name

    def set_id(self, new_id: int) -> None:
        self.id_num = new_id

    def set_parent_id(self, parent_id: int) -> None:
        self.parent_id = parent_id

    def set_pos(self, p_x: float, p_y: float, p_z: float) -> None:
        self.pos = [p_x, p_y, p_z]

    def get_num_markers(self) -> int:
        return len(self.rb_marker_list)

    def add_rb_marker(self, new_rb_marker: RBMarker) -> int:
        self.rb_marker_list.append(copy.deepcopy(new_rb_marker))
        return self.get_num_markers()

    def get_as_string(self, tab_str: str = "  ", level: int = 0) -> str:
        out_tab_str = get_tab_str(tab_str, level)
        out_tab_str2 = get_tab_str(tab_str, level + 1)
        out_string = ""
        out_string += f"{out_tab_str}Rigid Body        :"
        if self.rb_num > -1:
            out_string += f" {self.rb_num}\n"
        out_string += "\n"
        out_string += f"{out_tab_str}Rigid Body Name   : {get_as_string(self.sz_name)}\n"
        out_string += f"{out_tab_str}Rigid Body ID     : {self.id_num}\n"
        out_string += f"{out_tab_str}Parent ID         : {self.parent_id}\n"
        out_string += f"{out_tab_str}Position          : [{self.pos[0]:3.2f}, {self.pos[1]:3.2f}, {self.pos[2]:3.2f}]\n"
        num_markers = len(self.rb_marker_list)
        out_string += f"{out_tab_str}Number of Markers : {num_markers}\n"
        # loop over markers
        for i in range(num_markers):
            out_string += f"{out_tab_str2}{i} {self.rb_marker_list[i].get_as_string(tab_str, 0)}"
        return out_string


class SkeletonDescription:
    def __init__(self, name: str = "", new_id: int = 0) -> None:
        self.name = name
        self.id_num = new_id
        self.rigid_body_description_list = []

    def set_name(self, new_name: str | bytes) -> None:
        self.name = new_name

    def set_id(self, new_id: int) -> None:
        self.id_num = new_id

    def add_rigid_body_description(self, rigid_body_description: RigidBodyDescription) -> int:
        self.rigid_body_description_list.append(copy.deepcopy(rigid_body_description))
        return len(self.rigid_body_description_list)

    def get_as_string(self, tab_str: str = "  ", level: int = 0) -> str:
        out_tab_str = get_tab_str(tab_str, level)
        out_tab_str2 = get_tab_str(tab_str, level + 1)
        out_string = ""
        out_string += f"{out_tab_str}Name                    : {get_as_string(self.name)}\n"
        out_string += f"{out_tab_str}ID                      : {self.id_num}\n"
        num_bones = len(self.rigid_body_description_list)
        out_string += f"{out_tab_str}Rigid Body (Bone) Count : {num_bones}\n"
        for i in range(num_bones):
            out_string += f"{out_tab_str2}Rigid Body (Bone) {i}\n"
            out_string += self.rigid_body_description_list[i].get_as_string(tab_str, level + 2)
        return out_string


class ForcePlateDescription:
    def __init__(self, new_id: int = 0, serial_number: str = "") -> None:
        self.id_num = new_id
        self.serial_number = serial_number
        self.width = 0
        self.length = 0
        self.position = [0.0, 0.0, 0.0]
        self.cal_matrix = [[0.0 for col in range(12)] for row in range(12)]
        self.corners = [[0.0 for col in range(3)] for row in range(4)]
        self.plate_type = 0
        self.channel_data_type = 0
        self.channel_list = []

    def set_id(self, new_id: int) -> None:
        self.id_num = new_id

    def set_serial_number(self, serial_number: str | bytes) -> None:
        self.serial_number = serial_number

    def set_dimensions(self, width: float, length: float) -> None:
        self.width = width
        self.length = length

    def set_origin(self, p_x: float, p_y: float, p_z: float) -> None:
        self.position = [p_x, p_y, p_z]

    def set_cal_matrix(self, cal_matrix: list[list[float]]) -> None:
        self.cal_matrix = cal_matrix

    def set_corners(self, corners: list[list[float]]) -> None:
        self.corners = corners

    def set_plate_type(self, plate_type: int) -> None:
        self.plate_type = plate_type

    def set_channel_data_type(self, channel_data_type: int) -> None:
        self.channel_data_type = channel_data_type

    def add_channel_name(self, channel_name: str | bytes) -> int:
        self.channel_list.append(copy.deepcopy(channel_name))
        return len(self.channel_list)

    def get_cal_matrix_as_string(self, tab_str: str = "", level: int = 0) -> str:
        """Get force plate calibration matrix as string"""
        out_tab_str = get_tab_str(tab_str, level)
        out_tab_str2 = get_tab_str(tab_str, level + 1)
        out_string = ""
        out_string += f"{out_tab_str}Cal Matrix:\n"
        for i in range(0, 12):
            out_string += f"{out_tab_str2}{i:2.1d} "
            out_string += f"{self.cal_matrix[i][0]:3.3e} "
            out_string += f"{self.cal_matrix[i][1]:3.3e} "
            out_string += f"{self.cal_matrix[i][2]:3.3e} "
            out_string += f"{self.cal_matrix[i][3]:3.3e} "
            out_string += f"{self.cal_matrix[i][4]:3.3e} "
            out_string += f"{self.cal_matrix[i][5]:3.3e} "
            out_string += f"{self.cal_matrix[i][6]:3.3e} "
            out_string += f"{self.cal_matrix[i][7]:3.3e} "
            out_string += f"{self.cal_matrix[i][8]:3.3e} "
            out_string += f"{self.cal_matrix[i][9]:3.3e} "
            out_string += f"{self.cal_matrix[i][10]:3.3e} "
            out_string += f"{self.cal_matrix[i][11]:3.3e}\n"
        return out_string

    def get_corners_as_string(self, tab_str: str = "", level: int = 0) -> str:
        """Get force plate corner positions as a string"""
        # Corners 4x3 floats
        out_tab_str = get_tab_str(tab_str, level)
        out_tab_str2 = get_tab_str(tab_str, level + 1)
        out_string = ""
        out_string += f"{out_tab_str}Corners:\n"
        for i in range(0, 4):
            out_string += f"{out_tab_str2}{i:2.1d} {self.corners[i][0]:3.3e} {self.corners[i][1]:3.3e} {self.corners[i][2]:3.3e}\n"
        return out_string

    def get_as_string(self, tab_str: str = "  ", level: int = 0) -> str:
        """Get force plate description as a class"""
        out_tab_str = get_tab_str(tab_str, level)
        out_string = ""
        out_string += f"{out_tab_str}ID                      : {self.id_num}\n"
        out_string += (
            f"{out_tab_str}Serial Number           : {get_as_string(self.serial_number)}\n"
        )
        out_string += f"{out_tab_str}Width                   : {self.width:3.2f}\n"
        out_string += f"{out_tab_str}Length                  : {self.length:3.2f}\n"
        out_string += f"{out_tab_str}Origin                  : [{self.position[0]:3.2f}, {self.position[1]:3.2f}, {self.position[2]:3.2f}]\n"
        out_string += self.get_cal_matrix_as_string(tab_str, level)
        out_string += self.get_corners_as_string(tab_str, level)

        out_string += f"{out_tab_str}Plate Type                : {self.plate_type}\n"
        out_string += f"{out_tab_str}Channel Data Type         : {self.channel_data_type}\n"
        num_channels = len(self.channel_list)
        out_string += f"{out_tab_str}Number of Channels        : {num_channels}\n"
        # Channel Names list of NoC strings
        out_tab_str2 = get_tab_str(tab_str, level + 1)
        for channel_num in range(num_channels):
            out_string += f"{out_tab_str2}Channel Name {channel_num}: {get_as_string(self.channel_list[channel_num])}\n"

        return out_string


class DeviceDescription:
    """Device Description class"""

    def __init__(
        self,
        new_id: int,
        name: str | bytes,
        serial_number: str | bytes,
        device_type: int,
        channel_data_type: int,
    ) -> None:
        self.id_num = new_id
        self.name = name
        self.serial_number = serial_number
        self.device_type = device_type
        self.channel_data_type = channel_data_type
        self.channel_list = []

    def set_id(self, new_id: int) -> None:
        """Set the device id"""
        self.id_num = new_id

    def set_name(self, name: str) -> None:
        """Set the Device name"""
        self.name = name

    def add_channel_name(self, channel_name: str | bytes) -> int:
        """Add channel name to channel_list"""
        self.channel_list.append(channel_name)
        return len(self.channel_list)

    def get_as_string(self, tab_str: str = "  ", level: int = 0) -> str:
        """Get Device Description as string"""
        out_tab_str = get_tab_str(tab_str, level)
        out_tab_str2 = get_tab_str(tab_str, level + 1)
        out_string = ""
        out_string += f"{out_tab_str}ID                 : {self.id_num:5.1d}\n"
        out_string += f"{out_tab_str}Name               : {get_as_string(self.name)}\n"
        out_string += f"{out_tab_str}Serial Number      : {get_as_string(self.serial_number)}\n"
        out_string += f"{out_tab_str}Device Type        : {self.device_type}\n"
        out_string += f"{out_tab_str}Channel Data Type  : {self.channel_data_type}\n"
        num_channels = len(self.channel_list)
        out_string += f"{out_tab_str}Number of Channels : {num_channels}\n"
        for i in range(num_channels):
            out_string += (
                f"{out_tab_str2}Channel {i:2.1d} Name : {get_as_string(self.channel_list[i])}\n"
            )
        return out_string


class CameraDescription:
    """Camera Description class"""

    def __init__(
        self,
        name: str | bytes,
        position_vec3: list[float] | tuple[float, float, float],
        orientation_quat: list[float] | tuple[float, float, float, float],
    ) -> None:
        self.name = name
        self.position = position_vec3
        self.orientation = orientation_quat

    def get_as_string(self, tab_str: str = "..", level: int = 0) -> str:
        """Get Camera Description as a string"""
        out_tab_str = get_tab_str(tab_str, level)
        out_string = ""
        out_string += f"{out_tab_str}Name        : {get_as_string(self.name)}\n"
        out_string += f"{out_tab_str}Position    : [{self.position[0]:3.2f}, {self.position[1]:3.2f}, {self.position[2]:3.2f}]\n"
        out_string += f"{out_tab_str}Orientation : [{self.orientation[0]:3.2f}, {self.orientation[1]:3.2f}, {self.orientation[2]:3.2f}, {self.orientation[3]:3.2f}]\n"
        return out_string


class MarkerDescription:
    """Marker Description class"""

    def __init__(
        self,
        name: str | bytes,
        marker_id: int,
        position: list[float] | tuple[float, float, float],
        marker_size: float,
        marker_params: int,
    ) -> None:
        self.name = name
        self.marker_id = marker_id
        self.position = position
        self.marker_size = marker_size
        self.marker_params = marker_params

    def get_as_string(self, tab_str: str = "..", level: int = 0) -> str:
        """Get Marker Description as a string"""
        out_tab_str = get_tab_str(tab_str, level)
        out_string = ""
        out_string += f"{out_tab_str}Name        : {get_as_string(self.name)}\n"
        out_string += f"{out_tab_str}ID          : {self.marker_id}\n"
        out_string += f"{out_tab_str}Position    : [{self.position[0]:3.2f}, {self.position[1]:3.2f}, {self.position[2]:3.2f}]\n"
        out_string += f"{out_tab_str}Size       : {self.marker_size:3.2f}\n"
        out_string += f"{out_tab_str}Params        : {self.marker_params}\n"

        return out_string


class AssetDescription:
    """Asset Description class"""

    def __init__(
        self,
        name: str | bytes,
        assetType: int,
        assetID: int,
        rigidbodyArray: list[RigidBodyDescription],
        markerArray: list[MarkerDescription],
    ) -> None:
        self.name = name
        self.assetType = assetType
        self.assetID = assetID
        self.rigidbodyArray = rigidbodyArray
        self.markerArray = markerArray

    def get_as_string(self, tab_str: str = "..", level: int = 0) -> str:
        """Get Asset Description as a string"""
        out_tab_str = get_tab_str(tab_str, level)
        out_string = ""
        # out_string += "Asset Description\n"
        out_string += f"{out_tab_str}Name       : {get_as_string(self.name)}\n"
        out_string += f"{out_tab_str}Type       : {self.assetType}\n"
        out_string += f"{out_tab_str}ID         : {self.assetID}\n"

        rbCount = 0
        out_string += f"{out_tab_str}RigidBody (Bone) Count : {len(self.rigidbodyArray)}\n"
        for rigidbody in self.rigidbodyArray:
            out_string += f"{out_tab_str}RigidBody (Bone) {rbCount}:\n"
            out_string += rigidbody.get_as_string(tab_str, level + 1)
            rbCount += 1

        markerCount = 0
        out_string += f"{out_tab_str}Marker Count : {len(self.markerArray)}\n"
        for marker in self.markerArray:
            out_string += f"{out_tab_str}Marker {markerCount}:\n"
            out_string += marker.get_as_string(tab_str, level + 1)
            markerCount += 1

        return out_string


# cDataDescriptions
# Full data descriptions
class DataDescriptions:
    """Data Descriptions class"""

    order_num = 0

    def __init__(self) -> None:
        self.data_order_dict = {}
        self.marker_set_list = []
        self.rigid_body_list = []
        self.skeleton_list = []
        self.asset_list = []
        self.force_plate_list = []
        self.device_list = []
        self.camera_list = []

    def generate_order_name(self) -> str:
        """Generate the name for the order list based on the current length of
        the list"""
        # should be a one up counter instead of based on length of
        # data_order_dict
        order_name = f"data_{self.order_num:3.3d}"
        self.order_num += 1
        return order_name

    # Add Markerset
    def add_marker_set(self, new_marker_set: MarkerSetDescription) -> None:
        """Add a Markerset"""
        order_name = self.generate_order_name()

        # generate order entry
        pos = len(self.marker_set_list)
        self.data_order_dict[order_name] = ("marker_set_list", pos)
        self.marker_set_list.append(copy.deepcopy(new_marker_set))

    # Add Rigid Body
    def add_rigid_body(self, new_rigid_body: RigidBodyDescription) -> None:
        """Add a rigid body"""
        order_name = self.generate_order_name()

        # generate order entry
        pos = len(self.rigid_body_list)
        self.data_order_dict[order_name] = ("rigid_body_list", pos)
        self.rigid_body_list.append(copy.deepcopy(new_rigid_body))

    # Add a skeleton
    def add_skeleton(self, new_skeleton: SkeletonDescription) -> None:
        """Add a skeleton"""
        order_name = self.generate_order_name()

        # generate order entry
        pos = len(self.skeleton_list)
        self.data_order_dict[order_name] = ("skeleton_list", pos)
        self.skeleton_list.append(copy.deepcopy(new_skeleton))

    # Add an asset
    def add_asset(self, new_asset: AssetDescription) -> None:
        """Add an asset"""
        order_name = self.generate_order_name()

        # generate order entry
        pos = len(self.asset_list)
        self.data_order_dict[order_name] = ("asset_list", pos)
        self.asset_list.append(copy.deepcopy(new_asset))

    # Add a force plate
    def add_force_plate(self, new_force_plate: ForcePlateDescription) -> None:
        """Add a force plate"""
        order_name = self.generate_order_name()

        # generate order entry
        pos = len(self.force_plate_list)
        self.data_order_dict[order_name] = ("force_plate_list", pos)
        self.force_plate_list.append(copy.deepcopy(new_force_plate))

    def add_device(self, newdevice: DeviceDescription) -> None:
        """add_device - Add a device"""
        order_name = self.generate_order_name()

        # generate order entry
        pos = len(self.device_list)
        self.data_order_dict[order_name] = ("device_list", pos)
        self.device_list.append(copy.deepcopy(newdevice))

    def add_camera(self, newcamera: CameraDescription) -> None:
        """Add a new camera"""
        order_name = self.generate_order_name()

        # generate order entry
        pos = len(self.camera_list)
        self.data_order_dict[order_name] = ("camera_list", pos)
        self.camera_list.append(copy.deepcopy(newcamera))

    def add_data(
        self,
        new_data: MarkerSetDescription
        | RigidBodyDescription
        | SkeletonDescription
        | ForcePlateDescription
        | DeviceDescription
        | CameraDescription
        | MarkerDescription
        | AssetDescription
        | None,
    ) -> None:
        """Add data based on data type"""
        if type(new_data) is MarkerSetDescription:
            self.add_marker_set(new_data)
        elif type(new_data) is RigidBodyDescription:
            self.add_rigid_body(new_data)
        elif type(new_data) is SkeletonDescription:
            self.add_skeleton(new_data)
        elif type(new_data) is ForcePlateDescription:
            self.add_force_plate(new_data)
        elif type(new_data) is DeviceDescription:
            self.add_device(new_data)
        elif type(new_data) is CameraDescription:
            self.add_camera(new_data)
        elif type(new_data) is AssetDescription:
            self.add_asset(new_data)
        elif type(new_data) is type(None):
            pass
        else:
            print(f"ERROR: Type {str(type(new_data))} unknown")

    def get_object_from_list(
        self, list_name: str, pos_num: int
    ) -> (
        MarkerSetDescription
        | RigidBodyDescription
        | SkeletonDescription
        | ForcePlateDescription
        | DeviceDescription
        | CameraDescription
        | MarkerDescription
        | AssetDescription
        | None
    ):
        """Determine list name and position of the object"""
        ret_value = None
        if (list_name == "marker_set_list") and (pos_num < len(self.marker_set_list)):
            ret_value = self.marker_set_list[pos_num]

        elif (list_name == "rigid_body_list") and (pos_num < len(self.rigid_body_list)):
            ret_value = self.rigid_body_list[pos_num]

        elif (list_name == "skeleton_list") and (pos_num < len(self.skeleton_list)):
            ret_value = self.skeleton_list[pos_num]

        elif (list_name == "asset_list") and (pos_num < len(self.asset_list)):
            ret_value = self.asset_list[pos_num]

        elif (list_name == "force_plate_list") and (pos_num < len(self.force_plate_list)):
            ret_value = self.force_plate_list[pos_num]

        elif (list_name == "device_list") and (pos_num < len(self.device_list)):
            ret_value = self.device_list[pos_num]

        elif (list_name == "camera_list") and (pos_num < len(self.camera_list)):
            ret_value = self.camera_list[pos_num]

        else:
            ret_value = None

        return ret_value

    def get_as_string(self, tab_str: str = "  ", level: int = 0) -> str:
        """Ensure data comes back as a string"""
        out_tab_str = get_tab_str(tab_str, level)
        out_tab_str2 = get_tab_str(tab_str, level + 1)
        out_tab_str3 = get_tab_str(tab_str, level + 2)
        out_string = ""
        num_data_sets = len(self.data_order_dict)
        out_string += f"{out_tab_str}Dataset Count: {num_data_sets}\n"
        i = 0
        for tmp_key, tmp_value in self.data_order_dict.items():
            # tmp_name,tmp_num=self.data_order_dict[data_set]
            tmp_name = tmp_value[0]
            tmp_num = tmp_value[1]
            tmp_object = self.get_object_from_list(tmp_name, tmp_num)
            out_string += f"{out_tab_str2}Dataset {i:3.1d}\n"
            tmp_string = get_data_sub_packet_type(tmp_object)
            if tmp_string != "":
                out_string += f"{out_tab_str2}{tmp_string}"
            # outputs keys for looking up objects
            # out_string += "%s%s %s %d\n" % (
            #    out_tab_str2, data_set, tmp_name, tmp_num)
            if tmp_object is not None:
                out_string += tmp_object.get_as_string(tab_str, level + 2)
            else:
                out_string += f"{out_tab_str3}{tmp_key} {tmp_name} {tmp_num} not found\n"
            out_string += "\n"
            i += 1

        return out_string


# cDataDescriptions END


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
    test_object: MarkerSetDescription
    | RigidBodyDescription
    | SkeletonDescription
    | ForcePlateDescription
    | DeviceDescription
    | CameraDescription
    | MarkerDescription
    | AssetDescription
    | DataDescriptions,
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
    test_object: MarkerSetDescription
    | RigidBodyDescription
    | SkeletonDescription
    | ForcePlateDescription
    | DeviceDescription
    | CameraDescription
    | MarkerDescription
    | AssetDescription
    | DataDescriptions
    | None,
    generator_string: str,
    run_test: bool = True,
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


def get_as_string(input_str: str | bytes) -> str:
    if type(input_str) is str:
        return input_str
    elif type(input_str) is bytes:
        return input_str.decode("utf-8")
    else:
        return "<unknown>"


def get_data_sub_packet_type(
    new_data: MarkerSetDescription
    | RigidBodyDescription
    | SkeletonDescription
    | ForcePlateDescription
    | DeviceDescription
    | CameraDescription
    | MarkerDescription
    | AssetDescription
    | DataDescriptions
    | None,
) -> str:
    out_string = ""
    data_type = type(new_data)
    if data_type == MarkerSetDescription:
        out_string = "Type: 0 Markerset\n"
    elif data_type == RigidBodyDescription:
        out_string = "Type: 1 Rigid Body\n"
    elif data_type == SkeletonDescription:
        out_string = "Type: 2 Skeleton\n"
    elif data_type == ForcePlateDescription:
        out_string = "Type: 3 Force Plate\n"
    elif data_type == DeviceDescription:
        out_string = "Type: 4 Device\n"
    elif data_type == CameraDescription:
        out_string = "Type: 5 Camera\n"
    elif data_type == AssetDescription:
        out_string = "Type: 6 Asset\n"
    elif data_type is type(None):
        out_string = "Type: None\n"
    else:
        out_string = f"Type: Unknown {str(data_type)}\n"
    return out_string


def generate_marker_set_description(set_num: int = 0) -> MarkerSetDescription:
    """generate_marker_set_description - Testing functions"""
    marker_set_description = MarkerSetDescription()
    marker_set_description.set_name(f"MarkerSetName{set_num:3.3d}")
    marker_set_description.add_marker_name(f"MarkerName{set_num:3.3d}_0")
    marker_set_description.add_marker_name(f"MarkerName{set_num:3.3d}_1")
    marker_set_description.add_marker_name(f"MarkerName{set_num:3.3d}_2")
    marker_set_description.add_marker_name(f"MarkerName{set_num:3.3d}_3")
    return marker_set_description


def generate_rb_marker(marker_num: int = 0) -> RBMarker:
    """generate_rb_marker - Generate rigid body marker based on marker
    number"""
    marker_num_mod = marker_num % 4
    marker_name = f"RBMarker_{marker_num:3.3d}"
    marker_active_label = marker_num + 10000
    marker_pos = [1.0, 4.0, 9.0]
    if marker_num_mod == 1:
        marker_pos = [1.0, 8.0, 27.0]
    elif marker_num_mod == 2:
        marker_pos = [3.1, 4.1, 5.9]
    elif marker_num_mod == 3:
        marker_pos = [1.0, 3.0, 6.0]

    return RBMarker(marker_name, marker_active_label, marker_pos)


def generate_rigid_body_description(rbd_num: int = 0) -> RigidBodyDescription:
    """generate_rigid_body_description - Generate Rigid Body Description
    Data"""
    rbd = RigidBodyDescription()
    rbd.set_name(f"rigidBodyDescription_{rbd_num:3.3d}")
    rbd.set_id(3141)
    rbd.set_parent_id(314)
    rbd.set_pos(1, 4, 9)
    rbd.add_rb_marker(generate_rb_marker(0))
    rbd.add_rb_marker(generate_rb_marker(1))
    rbd.add_rb_marker(generate_rb_marker(2))

    return rbd


def generate_skeleton_description(skeleton_num: int = 0) -> SkeletonDescription:
    """generate_skeleton_description -Generate Test SkeletonDescription Data"""
    skel_desc = SkeletonDescription(f"SkeletonDescription_{skeleton_num:3.3d}", skeleton_num)
    # generate some rigid bodies to add
    skel_desc.add_rigid_body_description(generate_rigid_body_description(0))
    skel_desc.add_rigid_body_description(generate_rigid_body_description(1))
    skel_desc.add_rigid_body_description(generate_rigid_body_description(2))
    skel_desc.add_rigid_body_description(generate_rigid_body_description(3))
    skel_desc.add_rigid_body_description(generate_rigid_body_description(5))
    skel_desc.add_rigid_body_description(generate_rigid_body_description(7))
    return skel_desc


def generate_force_plate_description(force_plate_num: int = 0) -> ForcePlateDescription:
    """generate_force_plate_description - Generate Test ForcePlateDescription
    Data"""
    fp_id = force_plate_num
    random.seed(force_plate_num)

    serial_number = f"S/N_{random.randint(0, 99999):5.5d}"
    width = random.random() * 10
    length = random.random() * 10
    origin = [(random.random() * 100), (random.random() * 100), (random.random() * 100)]
    corners = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]

    fp_desc = ForcePlateDescription(fp_id, serial_number)
    fp_desc.set_dimensions(width, length)
    fp_desc.set_origin(origin[0], origin[1], origin[2])
    # fp_desc.set_cal_matrix(cal_matrix)
    fp_desc.set_corners(corners)
    for i in range(3):
        fp_desc.add_channel_name(f"channel_{i:3.3d}")
    return fp_desc


def generate_device_description(dev_num: int = 0) -> DeviceDescription:
    """generate_device_description- Generate Test DeviceDescription Data"""
    new_id = 0
    name = f"Device_{dev_num:3.3d}"
    serial_number = f"SerialNumber_{dev_num:3.3d}"
    device_type = dev_num % 4
    channel_data_type = dev_num % 5
    dev_desc = DeviceDescription(new_id, name, serial_number, device_type, channel_data_type)
    for i in range(channel_data_type + 3):
        dev_desc.add_channel_name(f"channel_name_{i:2.2d}")
    return dev_desc


def generate_camera_description(cam_num: int = 0) -> CameraDescription:
    """generate_camera_description - Generate Test CameraDescription data"""
    pos_vec3 = [1.0, 2.0, 3.0]
    orientation_quat = [1.0, 2.0, 3.0, 4.0]
    return CameraDescription(f"Camera_{cam_num:3.3d}", pos_vec3, orientation_quat)


# generate_data_descriptions - Generate Test DataDescriptions
def generate_data_descriptions(data_desc_num: int = 0) -> DataDescriptions:
    """Generate data descriptions"""
    data_descs = DataDescriptions()

    data_descs.add_data(generate_marker_set_description(data_desc_num + 0))
    data_descs.add_data(generate_marker_set_description(data_desc_num + 1))

    data_descs.add_data(generate_rigid_body_description(data_desc_num + 0))
    data_descs.add_data(generate_rigid_body_description(data_desc_num + 1))

    data_descs.add_skeleton(generate_skeleton_description(data_desc_num + 3))
    data_descs.add_skeleton(generate_skeleton_description(data_desc_num + 9))
    data_descs.add_skeleton(generate_skeleton_description(data_desc_num + 27))

    data_descs.add_force_plate(generate_force_plate_description(data_desc_num + 123))
    data_descs.add_force_plate(generate_force_plate_description(data_desc_num + 87))
    data_descs.add_force_plate(generate_force_plate_description(data_desc_num + 21))

    data_descs.add_device(generate_device_description(data_desc_num + 0))
    data_descs.add_device(generate_device_description(data_desc_num + 2))
    data_descs.add_device(generate_device_description(data_desc_num + 4))

    data_descs.add_camera(generate_camera_description(data_desc_num + 0))
    data_descs.add_camera(generate_camera_description(data_desc_num + 10))
    data_descs.add_camera(generate_camera_description(data_desc_num + 3))
    data_descs.add_camera(generate_camera_description(data_desc_num + 7))
    return data_descs


# test_all - Test all the major classes
def test_all(run_test: bool = True) -> list[int]:
    """Test all the Data Description classes"""
    totals = [0, 0, 0]
    if run_test is True:
        test_cases = [
            [
                "Test Markerset Description 0",
                "d918228cc347bd0dac69dd02b1a5375a4421364f",
                "generate_marker_set_description(0)",
                True,
            ],
            [
                "Test RB Marker 0",
                "df582ca7b764d889041b59ceb6a43251b68ca3be",
                "generate_rb_marker(0)",
                True,
            ],
            [
                "Test Rigid Body Description 0",
                "0ea7085657c391efe2fd349cc03f242247efbbe4",
                "generate_rigid_body_description(0)",
                True,
            ],
            [
                "Test Skeleton Description 0",
                "fa2a59e76f31c1d884f6554fe13e5cfcf31e703c",
                "generate_skeleton_description(0)",
                True,
            ],
            [
                "Test Force Plate Description 0",
                "798793a2fed302bc472b2636beff959901214be2",
                "generate_force_plate_description(0)",
                True,
            ],
            [
                "Test Device Description 0",
                "39b4fdda402bc73c0b1cd5c7f61599476aa9a926",
                "generate_device_description(0)",
                True,
            ],
            [
                "Test Camera Description 0",
                "614602c5d290bda3b288138d5e25516dd1e1e85a",
                "generate_camera_description(0)",
                True,
            ],
            [
                "Test Data Description 0",
                "b2fcffb251ae526e91ec9f65f5f2137f0d74db49",
                "generate_data_descriptions(0)",
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
    print(f"[PASS] Count = {totals[0]:3.1d}")
    print(f"[FAIL] Count = {totals[1]:3.1d}")
    print(f"[SKIP] Count = {totals[2]:3.1d}")

    return totals


if __name__ == "__main__":
    test_all(True)
