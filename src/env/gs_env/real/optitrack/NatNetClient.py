# =============================================================================
# Copyright © 2025 NaturalPoint, Inc. All Rights Reserved.
#
# THIS SOFTWARE IS GOVERNED BY THE OPTITRACK PLUGINS EULA AVAILABLE AT
# https://www.optitrack.com/about/legal/eula.html  AND/OR FOR DOWNLOAD WITH
# THE APPLICABLE SOFTWARE FILE(S) (“PLUGINS EULA”). BY DOWNLOADING,
# INSTALLING, ACTIVATING AND/OR OTHERWISE USING THE SOFTWARE, YOU ARE AGREEING
# THAT YOU HAVE READ, AND THAT YOU AGREE TO COMPLY WITH AND ARE BOUND BY, THE
# PLUGINS EULA AND ALL APPLICABLE LAWS AND REGULATIONS. IF YOU DO NOT AGREE TO
# BE BOUND BY THE PLUGINS EULA, THEN YOU MAY NOT DOWNLOAD, INSTALL, ACTIVATE
# OR OTHERWISE USE THE SOFTWARE AND YOU MUST PROMPTLY DELETE OR RETURN IT. IF
# YOU ARE DOWNLOADING, INSTALLING, ACTIVATING AND/OR OTHERWISE USING THE
# SOFTWARE ON BEHALF OF AN ENTITY, THEN BY DOING SO YOU REPRESENT AND WARRANT
# THAT YOU HAVE THE APPROPRIATE AUTHORITY TO ACCEPT THE PLUGINS EULA ON BEHALF
# OF SUCH ENTITY.
# See license file in root directory for additional governing terms and
# information.
# =============================================================================

# OptiTrack NatNet direct depacketization library for Python 3.x

# The workflow is altered to always only save the latest frame of data

import sys  # noqa F401
import socket
import threading
import struct
from threading import Thread
import copy
import time
from . import DataDescriptions as DataDescriptions
from . import MoCapData as MoCapData
from collections.abc import Callable


def trace(*args) -> None:
    # uncomment the one you want to use
    # print("".join(map(str,args)))
    pass


# Used for Data Description functions
def trace_dd(*args) -> None:
    # uncomment the one you want to use

    # print("".join(map(str,args)))
    pass


# Used for MoCap Frame Data functions
def trace_mf(*args) -> None:
    # uncomment the one you want to use
    # print("".join(map(str,args)))
    pass


def get_message_id(data: bytes | bytearray) -> int:
    message_id = int.from_bytes(data[0:2], byteorder="little", signed=True)
    return message_id


# Create structs for reading various object types to speed up parsing.
Vector2 = struct.Struct("<ff")
Vector3 = struct.Struct("<fff")
Quaternion = struct.Struct("<ffff")
FloatValue = struct.Struct("<f")
DoubleValue = struct.Struct("<d")
NNIntValue = struct.Struct("<I")
FPCalMatrixRow = struct.Struct("<ffffffffffff")
FPCorners = struct.Struct("<ffffffffffff")


class NatNetClient:
    # print_level = 0 off
    # print_level = 1 on
    # print_level = >1 on / print every nth mocap frame
    print_level = 20

    def __init__(self) -> None:
        # Change this value to the IP address of the NatNet server.
        self.server_ip_address = "127.0.0.1"

        # Change this value to the IP address of your local network interface
        self.local_ip_address = "127.0.0.1"

        # Should match multicast address listed in Motive's streaming settings.
        self.multicast_address = "239.255.42.99"

        # NatNet Command channel
        self.command_port = 1510

        # NatNet Data channel
        self.data_port = 1511

        self.use_multicast = None

        # Set this to a callback method of your choice.
        # Allows receiving per-rigid-body data at each frame.
        self.rigid_body_listener = None
        self.new_frame_listener = None
        self.new_frame_with_data_listener = None
        self.data_description_listener = None

        # Set Application Name
        self.__application_name = "Not Set"

        # NatNet stream version server is capable of.
        # Updated during initialization only.
        self.__nat_net_stream_version_server = [3, 0, 3, 0]

        # NatNet stream version.
        # Will be updated to the actual version the server is using at runtime.
        self.__nat_net_requested_version = [3, 0, 3, 0]

        # server stream version.
        # Will be updated to the actual version the server is using at init..
        self.__server_version = [3, 0, 3, 0]

        # Lock values once run is called
        self.__is_locked = False

        # Server has the ability to change bitstream version
        self.__can_change_bitstream_version = False

        self.command_thread = None
        self.data_thread = None
        self.command_socket = None
        self.data_socket = None

        self.stop_threads = False

        # Latest MoCap data
        self._latest_mocap_lock = threading.Lock()
        self._latest_mocap_event = threading.Event()
        self._latest_mocap_data: MoCapData.MoCapData = MoCapData.MoCapData()

    # Client/server message ids
    NAT_CONNECT = 0
    NAT_SERVERINFO = 1
    NAT_REQUEST = 2
    NAT_RESPONSE = 3
    NAT_REQUEST_MODELDEF = 4
    NAT_MODELDEF = 5
    NAT_REQUEST_FRAMEOFDATA = 6
    NAT_FRAMEOFDATA = 7
    NAT_MESSAGESTRING = 8
    NAT_DISCONNECT = 9
    NAT_KEEPALIVE = 10
    NAT_UNRECOGNIZED_REQUEST = 100
    NAT_UNDEFINED = 999999.9999

    def set_client_address(self, local_ip_address: str) -> None:
        if not self.__is_locked:
            self.local_ip_address = local_ip_address

    def get_client_address(self) -> str:
        return self.local_ip_address

    def set_server_address(self, server_ip_address: str) -> None:
        if not self.__is_locked:
            self.server_ip_address = server_ip_address

    def get_server_address(self) -> str:
        return self.server_ip_address

    def set_use_multicast(self, use_multicast: bool) -> None:
        if not self.__is_locked:
            self.use_multicast = use_multicast

    def can_change_bitstream_version(self) -> bool:
        return self.__can_change_bitstream_version

    def set_nat_net_version(self, major: int, minor: int) -> int:
        """checks to see if stream version can change, then
        changes it with position reset"""
        return_code = -1
        if self.__can_change_bitstream_version and (
            (major != self.__nat_net_requested_version[0])
            or (minor != self.__nat_net_requested_version[1])
        ):
            sz_command = f"Bitstream,{major:1d}.{minor:1d}"
            return_code = self.send_command(sz_command)
            if return_code >= 0:
                self.__nat_net_requested_version[0] = major
                self.__nat_net_requested_version[1] = minor
                self.__nat_net_requested_version[2] = 0
                self.__nat_net_requested_version[3] = 0
                print("changing bitstream MAIN")
                # get original output state
                # print_results = self.get_print_results()

                # turn off output
                # self.set_print_results(False)

                # force frame send and play reset
                self.send_command("TimelinePlay")
                time.sleep(0.1)
                tmpCommands = [
                    "TimelinePlay",
                    "TimelineStop",
                    "SetPlaybackCurrentFrame,0",
                    "TimelineStop",
                ]
                self.send_commands(tmpCommands, False)
                time.sleep(2)

                # reset to original output state
                # self.set_print_results(print_results)

            else:
                print("Bitstream change request failed")
        return return_code

    def get_major(self) -> int:
        return self.__nat_net_requested_version[0]

    def get_minor(self) -> int:
        return self.__nat_net_requested_version[1]

    def set_print_level(self, print_level: int = 0) -> int:
        if print_level >= 0:
            self.print_level = print_level
        return self.print_level

    def get_print_level(self) -> int:
        return self.print_level

    def connected(self) -> bool:
        ret_value = True
        # check sockets
        if self.command_socket is None:
            ret_value = False
        elif self.data_socket is None:
            ret_value = False
        # check versions
        elif self.get_application_name() == "Not Set":
            ret_value = False
        elif (
            (self.__server_version[0] == 0)
            and (self.__server_version[1] == 0)
            and (self.__server_version[2] == 0)
            and (self.__server_version[3] == 0)
        ):
            ret_value = False
        return ret_value

    # Create a command socket to attach to the NatNet stream
    def __create_command_socket(self) -> socket.socket | None:
        result = None
        if self.use_multicast:
            result = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
            result.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                if self.server_ip_address == self.local_ip_address:
                    # used, as ip/port issues arise when using the same
                    # address as server and client
                    result.bind(("", 0))
                else:
                    result.bind((self.local_ip_address, self.command_port))
            except OSError as e:
                print(f"Socket error: {e}")
            # set to broadcast mode
            result.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            # set timeout to allow for keep alive messages
            result.settimeout(2.0)
        else:
            result = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            try:
                result.bind((self.local_ip_address, 0))
            except OSError as e:
                print(f"Socket error: {e}")
        return result

    # Create a data socket to attach to the NatNet stream
    def __create_data_socket(self) -> socket.socket | None:
        result = None
        if self.use_multicast:
            # Multicast case
            result = socket.socket(
                socket.AF_INET,  # Internet
                socket.SOCK_DGRAM,
                0,
            )  # UDP
            result.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            result.setsockopt(
                socket.IPPROTO_IP,
                socket.IP_ADD_MEMBERSHIP,
                socket.inet_aton(self.multicast_address) + socket.inet_aton(self.local_ip_address),
            )
            try:
                # Use bind in data socket due to the nature of UDP
                result.bind((self.local_ip_address, self.data_port))
            except OSError as e:
                print(f"Multicast Error: {e}")
                sys.exit(1)
        else:
            # Unicast case
            self.use_multicast = False
            result = socket.socket(
                socket.AF_INET,  # Internet
                socket.SOCK_DGRAM,
                socket.IPPROTO_UDP,
            )
            result.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                # result.bind((self.local_ip_address, 0))
                result.bind(("0.0.0.0", 1511))  # Broadcast port 1511
            except OSError as e:
                print(f"Unicast Socket Error: {e}")
                sys.exit(1)
        return result

    def __unpack_rigid_body_3_and_above(
        self, data: bytes | memoryview, rb_num: int
    ) -> tuple[int, MoCapData.RigidBody]:
        """Calculates offset for NatNet 3 and above for rigid body
        unpacking"""
        offset = 0

        # ID (4 bytes)
        new_id = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4

        trace_mf(f"RB: {rb_num:3d} ID: {new_id:3d}")

        # Position and orientation
        pos = Vector3.unpack(data[offset : offset + 12])
        offset += 12
        trace_mf(f"\tPosition   : [{pos[0]:3.2f}, {pos[1]:3.2f}, {pos[2]:3.2f}]")

        rot = Quaternion.unpack(data[offset : offset + 16])
        offset += 16
        trace_mf(f"\tOrientation: [{rot[0]:3.2f}, {rot[1]:3.2f}, {rot[2]:3.2f}, {rot[3]:3.2f}]")

        rigid_body = MoCapData.RigidBody(new_id, pos, rot)

        # Send information to any listener.
        if self.rigid_body_listener is not None:
            self.rigid_body_listener(new_id, pos, rot)

        (marker_error,) = FloatValue.unpack(data[offset : offset + 4])
        offset += 4
        trace_mf(f"\tMean Marker Error: {marker_error:3.2f}")
        rigid_body.error = marker_error

        (param,) = struct.unpack("h", data[offset : offset + 2])
        tracking_valid = (param & 0x01) != 0
        offset += 2
        is_valid_str = "False"
        if tracking_valid:
            is_valid_str = "True"
        trace_mf(f"\tTracking Valid: {is_valid_str}")
        if tracking_valid:
            rigid_body.tracking_valid = True
        else:
            rigid_body.tracking_valid = False

        return offset, rigid_body

    def __unpack_rigid_body_2_6_to_3(
        self, data: bytes | memoryview, rb_num: int
    ) -> tuple[int, MoCapData.RigidBody]:
        """Calculates offset starting at NatNet 2.6 and going
        to (but not inclusive of 3)"""
        offset = 0

        # ID (4 bytes)
        new_id = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4

        trace_mf(f"RB: {rb_num:3d} ID: {new_id:3d}")

        # Position and orientation
        pos = Vector3.unpack(data[offset : offset + 12])
        offset += 12
        trace_mf(f"\tPosition   : [{pos[0]:3.2f}, {pos[1]:3.2f}, {pos[2]:3.2f}]")

        rot = Quaternion.unpack(data[offset : offset + 16])
        offset += 16
        trace_mf(f"\tOrientation: [{rot[0]:3.2f}, {rot[1]:3.2f}, {rot[2]:3.2f}, {rot[3]:3.2f}]")

        rigid_body = MoCapData.RigidBody(new_id, pos, rot)

        # Send information to any listener.
        if self.rigid_body_listener is not None:
            self.rigid_body_listener(new_id, pos, rot)

        marker_count = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4
        marker_count_range = range(0, marker_count)
        trace_mf(f"\tMarker Count: {marker_count}")

        rb_marker_list = []
        for _ in marker_count_range:
            rb_marker_list.append(MoCapData.RigidBodyMarker())

        # Marker positions
        for i in marker_count_range:
            pos = Vector3.unpack(data[offset : offset + 12])
            offset += 12
            trace_mf(f"\tMarker {i}: {pos[0]}, {pos[1]}, {pos[2]}")
            rb_marker_list[i].pos = pos

        for i in marker_count_range:
            new_id = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
            offset += 4
            trace_mf(f"\tMarker ID {i}: {new_id}")
            rb_marker_list[i].id = new_id

        # Marker sizes
        for i in marker_count_range:
            size = FloatValue.unpack(data[offset : offset + 4])
            offset += 4
            trace_mf(f"\tMarker Size {i}: {size[0]}")
            rb_marker_list[i].size = size

        for i in marker_count_range:
            rigid_body.add_rigid_body_marker(rb_marker_list[i])

        (marker_error,) = FloatValue.unpack(data[offset : offset + 4])
        offset += 4
        trace_mf(f"\tMean Marker Error: {marker_error:3.2f}")
        rigid_body.error = marker_error

        (param,) = struct.unpack("h", data[offset : offset + 2])
        tracking_valid = (param & 0x01) != 0
        offset += 2
        is_valid_str = "False"
        if tracking_valid:
            is_valid_str = "True"
        trace_mf(f"\tTracking Valid: {is_valid_str}")
        if tracking_valid:
            rigid_body.tracking_valid = True
        else:
            rigid_body.tracking_valid = False
        return offset, rigid_body

    def __unpack_rigid_body_pre_2_6(
        self, data: bytes | memoryview, major: int, rb_num: int
    ) -> tuple[int, MoCapData.RigidBody]:
        """Calculates offset for anything below NatNet 2.6"""
        offset = 0

        # ID (4 bytes)
        new_id = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4

        trace_mf(f"RB: {rb_num:3d} ID: {new_id:3d}")

        # Position and orientation
        pos = Vector3.unpack(data[offset : offset + 12])
        offset += 12
        trace_mf(f"\tPosition   : [{pos[0]:3.2f}, {pos[1]:3.2f}, {pos[2]:3.2f}]")

        rot = Quaternion.unpack(data[offset : offset + 16])
        offset += 16
        trace_mf(f"\tOrientation: [{rot[0]:3.2f}, {rot[1]:3.2f}, {rot[2]:3.2f}, {rot[3]:3.2f}]")

        rigid_body = MoCapData.RigidBody(new_id, pos, rot)

        # Send information to any listener.
        if self.rigid_body_listener is not None:
            self.rigid_body_listener(new_id, pos, rot)

        marker_count = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4
        marker_count_range = range(0, marker_count)
        trace_mf(f"\tMarker Count: {marker_count}")

        rb_marker_list = []
        for _ in marker_count_range:
            rb_marker_list.append(MoCapData.RigidBodyMarker())

        # Marker positions
        for i in marker_count_range:
            pos = Vector3.unpack(data[offset : offset + 12])
            offset += 12
            trace_mf(f"\tMarker {i}: {pos[0]}, {pos[1]}, {pos[2]}")
            rb_marker_list[i].pos = pos

        if major >= 2:
            # Marker ID's
            for i in marker_count_range:
                new_id = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
                offset += 4
                trace_mf(f"\tMarker ID {i}: {new_id}")
                rb_marker_list[i].id = new_id

            # Marker sizes
            for i in marker_count_range:
                size = FloatValue.unpack(data[offset : offset + 4])
                offset += 4
                trace_mf(f"\tMarker Size {i}: {size[0]}")
                rb_marker_list[i].size = size

            for i in marker_count_range:
                rigid_body.add_rigid_body_marker(rb_marker_list[i])

            if major >= 2:
                (marker_error,) = FloatValue.unpack(data[offset : offset + 4])
                offset += 4
                trace_mf(f"\tMean Marker Error: {marker_error:3.2f}")
                rigid_body.error = marker_error
        return offset, rigid_body

    def __unpack_rigid_body_0_case(
        self, data: bytes | memoryview, rb_num: int
    ) -> tuple[int, MoCapData.RigidBody]:
        """Calculates offset for case where major version is 0"""
        offset = 0

        # ID (4 bytes)
        new_id = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4

        trace_mf(f"RB: {rb_num:3d} ID: {new_id:3d}")

        # Position and orientation
        pos = Vector3.unpack(data[offset : offset + 12])
        offset += 12
        trace_mf(f"\tPosition   : [{pos[0]:3.2f}, {pos[1]:3.2f}, {pos[2]:3.2f}]")

        rot = Quaternion.unpack(data[offset : offset + 16])
        offset += 16
        trace_mf(f"\tOrientation: [{rot[0]:3.2f}, {rot[1]:3.2f}, {rot[2]:3.2f}, {rot[3]:3.2f}]")

        rigid_body = MoCapData.RigidBody(new_id, pos, rot)

        # Send information to any listener.
        if self.rigid_body_listener is not None:
            self.rigid_body_listener(new_id, pos, rot)
        return offset, rigid_body

    def __unpack_rigid_body(
        self, data: bytes | memoryview, major: int, minor: int, rb_num: int
    ) -> tuple[int, MoCapData.RigidBody]:
        if major >= 3:
            offset, rigid_body = self.__unpack_rigid_body_3_and_above(data, rb_num)
        elif major == 2 and minor >= 6:
            offset, rigid_body = self.__unpack_rigid_body_2_6_to_3(data, rb_num)
        elif major < 2 or (major == 2 and minor < 6):
            offset, rigid_body = self.__unpack_rigid_body_pre_2_6(data, major, rb_num)
        elif major == 0:
            offset, rigid_body = self.__unpack_rigid_body_0_case(data, rb_num)
        else:
            raise ValueError(f"Invalid Version {major:1d}.{minor:1d}")
        return offset, rigid_body

    # Unpack a skeleton object from a data packet
    def __unpack_skeleton(
        self, data: bytes | memoryview, major: int, minor: int, skeleton_num: int = 0
    ) -> tuple[int, MoCapData.Skeleton]:
        offset = 0
        new_id = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4
        trace_mf(f"Skeleton {skeleton_num:3d} ID: {new_id:3d}")
        skeleton = MoCapData.Skeleton(new_id)

        rigid_body_count = int.from_bytes(
            data[offset : offset + 4], byteorder="little", signed=True
        )
        offset += 4
        trace_mf(f"Rigid Body Count: {rigid_body_count:3d}")
        if rigid_body_count > 0:
            for rb_num in range(0, rigid_body_count):
                offset_tmp, rigid_body = self.__unpack_rigid_body(
                    data[offset:], major, minor, rb_num
                )
                skeleton.add_rigid_body(rigid_body)
                offset += offset_tmp

        return offset, skeleton

    def __unpack_asset(
        self, data: bytes | memoryview, major: int, minor: int, asset_num: int = 0
    ) -> tuple[int, MoCapData.Asset]:
        offset = 0
        trace_dd(f"\tAsset       : {asset_num}")
        # Asset ID 4 bytes
        new_id = int.from_bytes(data[offset : offset + 4], "little", signed=True)
        offset += 4
        asset = MoCapData.Asset()
        trace_dd(f"\tAsset ID    : {new_id}")
        asset.set_id(new_id)
        # # of RigidBodies
        numRBs = int.from_bytes(data[offset : offset + 4], "little", signed=True)
        offset += 4
        trace_dd(f"\tRigid Bodies: {numRBs}")
        offset1 = 0
        for rb_num in range(numRBs):
            # # of RigidBodies
            offset1, rigid_body = self.__unpack_asset_rigid_body_data(data[offset:], major, minor)
            offset += offset1
            rigid_body.rb_num = rb_num
            asset.add_rigid_body(rigid_body)

        # # of Markers
        numMarkers = int.from_bytes(data[offset : offset + 4], "little", signed=True)
        offset += 4
        trace_dd(f"\tMarkers     : {numMarkers}")

        for marker_num in range(numMarkers):
            # # of Markers
            offset1, marker = self.__unpack_asset_marker_data(data[offset:], major, minor)
            offset += offset1
            marker.marker_num = marker_num
            asset.add_marker(marker)

        return offset, asset

    # Unpack Mocap Data Functions

    def __unpack_frame_prefix_data(
        self, data: bytes | memoryview
    ) -> tuple[int, MoCapData.FramePrefixData]:
        offset = 0
        # Frame number (4 bytes)
        frame_number = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4
        trace_mf(f"Frame #: {frame_number:3d}")
        frame_prefix_data = MoCapData.FramePrefixData(frame_number)
        return offset, frame_prefix_data

    def __unpack_data_size(
        self, data: bytes | memoryview, major: int, minor: int
    ) -> tuple[int, int]:
        sizeInBytes = 0
        offset = 0

        if ((major == 4) and (minor > 0)) or (major > 4):
            sizeInBytes = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
            offset += 4
            trace_mf(f"Byte Count: {sizeInBytes:3d}")

        return offset, sizeInBytes

    def __unpack_legacy_other_markers(
        self, data: bytes | memoryview, packet_size: int, major: int, minor: int
    ) -> tuple[int, MoCapData.LegacyMarkerData]:
        offset = 0

        # Markerset count (4 bytes)
        other_marker_count = int.from_bytes(
            data[offset : offset + 4], byteorder="little", signed=True
        )
        offset += 4
        trace_mf(f"Other Marker Count: {other_marker_count}")

        # get data size (4 bytes)
        offset_tmp, unpackedDataSize = self.__unpack_data_size(data[offset:], major, minor)
        offset += offset_tmp

        other_marker_data = MoCapData.LegacyMarkerData()
        if other_marker_count > 0:
            # get legacy_marker positions
            # legacy_marker_data
            for j in range(0, other_marker_count):
                pos = Vector3.unpack(data[offset : offset + 12])
                offset += 12
                trace_mf(f"\tMarker {j:3d}: [x={pos[0]:3.2f},y={pos[1]:3.2f},z={pos[2]:3.2f}]")
                other_marker_data.add_pos(pos)
        return offset, other_marker_data

    def __unpack_marker_set_data(
        self, data: bytes | memoryview, packet_size: int, major: int, minor: int
    ) -> tuple[int, MoCapData.MarkerSetData]:
        marker_set_data = MoCapData.MarkerSetData()
        offset = 0
        # Markerset count (4 bytes)
        marker_set_count = int.from_bytes(
            data[offset : offset + 4], byteorder="little", signed=True
        )
        offset += 4
        trace_mf(f"Markerset Count: {marker_set_count}")

        # get data size (4 bytes)
        offset_tmp, unpackedDataSize = self.__unpack_data_size(data[offset:], major, minor)
        offset += offset_tmp

        for _ in range(0, marker_set_count):
            marker_data = MoCapData.MarkerData()
            # Model name
            model_name, separator, remainder = bytes(data[offset:]).partition(b"\0")
            offset += len(model_name) + 1
            trace_mf(f"Model Name     : {model_name.decode('utf-8')}")
            marker_data.set_model_name(model_name)
            # Marker count (4 bytes)
            marker_count = int.from_bytes(
                data[offset : offset + 4], byteorder="little", signed=True
            )
            offset += 4
            if marker_count < 0:
                print("WARNING: Early return.  Invalid marker count")
                offset = len(data)
                return offset, marker_set_data
            elif marker_count > 10000:
                print("WARNING: Early return.  Marker count too high")
                offset = len(data)
                return offset, marker_set_data

            trace_mf(f"Marker Count   : {marker_count}")
            for j in range(0, marker_count):
                if len(data) < (offset + 12):
                    print(f"WARNING: Early return.  Out of data at marker {j} of {marker_count}")
                    offset = len(data)
                    return offset, marker_set_data
                    break
                pos = Vector3.unpack(data[offset : offset + 12])
                offset += 12
                trace_mf(f"\tMarker {j:3d}: [x={pos[0]:3.2f},y={pos[1]:3.2f},z={pos[2]:3.2f}]")
                marker_data.add_pos(pos)
            marker_set_data.add_marker_data(marker_data)

        # Unlabeled markers count (4 bytes)
        # unlabeled_markers_count = int.from_bytes(data[offset:offset+4], byteorder='little',  signed=True) #type: ignore
        # offset += 4
        # trace_mf("Unlabeled Marker Count:", unlabeled_markers_count)

        # for i in range(0, unlabeled_markers_count):
        #    pos = Vector3.unpack(data[offset:offset+12])
        #    offset += 12
        #    trace_mf("\tMarker %3d: [%3.2f,%3.2f,%3.2f]" % (i, pos[0], pos[1], pos[2])) #type: ignore
        #    marker_set_data.add_unlabeled_marker(pos)
        return offset, marker_set_data

    def __unpack_rigid_body_data(
        self, data: bytes | memoryview, packet_size: int, major: int, minor: int
    ) -> tuple[int, MoCapData.RigidBodyData]:
        rigid_body_data = MoCapData.RigidBodyData()
        offset = 0
        # Rigid body count (4 bytes)
        rigid_body_count = int.from_bytes(
            data[offset : offset + 4], byteorder="little", signed=True
        )
        offset += 4
        trace_mf(f"Rigid Body Count: {rigid_body_count}")

        # get data size (4 bytes)
        offset_tmp, unpackedDataSize = self.__unpack_data_size(data[offset:], major, minor)
        offset += offset_tmp

        for i in range(0, rigid_body_count):
            offset_tmp, rigid_body = self.__unpack_rigid_body(data[offset:], major, minor, i)
            offset += offset_tmp
            rigid_body_data.add_rigid_body(rigid_body)

        return offset, rigid_body_data

    def __unpack_skeleton_data(
        self, data: bytes | memoryview, packet_size: int, major: int, minor: int
    ) -> tuple[int, MoCapData.SkeletonData]:
        skeleton_data = MoCapData.SkeletonData()

        offset = 0
        # Version 2.1 and later
        skeleton_count = 0
        if (major == 2 and minor > 0) or major > 2:
            skeleton_count = int.from_bytes(
                data[offset : offset + 4], byteorder="little", signed=True
            )
            offset += 4
            trace_mf(f"Skeleton Count: {skeleton_count}")
            # Get data size (4 bytes)
            offset_tmp, unpackedDataSize = self.__unpack_data_size(data[offset:], major, minor)
            offset += offset_tmp
            if skeleton_count > 0:
                for skeleton_num in range(0, skeleton_count):
                    rel_offset, skeleton = self.__unpack_skeleton(
                        data[offset:], major, minor, skeleton_num
                    )
                    offset += rel_offset
                    skeleton_data.add_skeleton(skeleton)

        return offset, skeleton_data

    def __decode_marker_id(self, new_id: int) -> tuple[int, int]:
        model_id = 0
        marker_id = 0
        model_id = new_id >> 16
        marker_id = new_id & 0x0000FFFF
        return model_id, marker_id

    def __unpack_labeled_marker_data(
        self, data: bytes | memoryview, packet_size: int, major: int, minor: int
    ) -> tuple[int, MoCapData.LabeledMarkerData]:
        labeled_marker_data = MoCapData.LabeledMarkerData()
        offset = 0
        # Labeled markers (Version 2.3 and later)
        labeled_marker_count = 0
        if (major == 2 and minor > 3) or major > 2:
            labeled_marker_count = int.from_bytes(
                data[offset : offset + 4], byteorder="little", signed=True
            )
            offset += 4
            trace_mf(f"Labeled Marker Count: {labeled_marker_count}")

            # get data size (4 bytes)
            offset_tmp, unpackedDataSize = self.__unpack_data_size(data[offset:], major, minor)
            offset += offset_tmp

            for lm_num in range(0, labeled_marker_count):
                model_id = 0
                marker_id = 0
                tmp_id = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
                offset += 4
                model_id, marker_id = self.__decode_marker_id(tmp_id)
                pos = Vector3.unpack(data[offset : offset + 12])
                offset += 12
                (size,) = FloatValue.unpack(data[offset : offset + 4])
                offset += 4
                trace_mf(
                    f" {lm_num:3d} ID    : [MarkerID: {marker_id:3d}] [ModelID: {model_id:3d}]"
                )
                trace_mf(f"    pos : [{pos[0]:3.2f}, {pos[1]:3.2f}, {pos[2]:3.2f}]")
                trace_mf(f"    size: [{size:3.2f}]")

                # Version 2.6 and later
                param = 0
                if (major == 2 and minor >= 6) or major > 2:
                    (param,) = struct.unpack("h", data[offset : offset + 2])
                    offset += 2
                    # occluded = (param & 0x01) != 0
                    # point_cloud_solved = (param & 0x02) != 0
                    # model_solved = (param & 0x04) != 0

                # Version 3.0 and later
                residual = 0.0
                if major >= 3:
                    (residual,) = FloatValue.unpack(data[offset : offset + 4])
                    offset += 4
                    residual = residual * 1000.0
                    trace_mf(f"    err : [{residual:3.2f}]")

                labeled_marker = MoCapData.LabeledMarker(tmp_id, pos, size, param, residual)
                labeled_marker_data.add_labeled_marker(labeled_marker)

        return offset, labeled_marker_data

    def __unpack_force_plate_data(
        self, data: bytes | memoryview, packet_size: int, major: int, minor: int
    ) -> tuple[int, MoCapData.ForcePlateData]:
        force_plate_data = MoCapData.ForcePlateData()
        n_frames_show_max = 4
        offset = 0
        # Force Plate data (version 2.9 and later)
        force_plate_count = 0
        if (major == 2 and minor >= 9) or major > 2:
            force_plate_count = int.from_bytes(
                data[offset : offset + 4], byteorder="little", signed=True
            )
            offset += 4
            trace_mf(f"Force Plate Count: {force_plate_count}")

            # get data size (4 bytes)
            offset_tmp, unpackedDataSize = self.__unpack_data_size(data[offset:], major, minor)
            offset += offset_tmp

            for i in range(0, force_plate_count):
                # ID
                force_plate_id = int.from_bytes(
                    data[offset : offset + 4], byteorder="little", signed=True
                )
                offset += 4
                force_plate = MoCapData.ForcePlate(force_plate_id)

                # Channel Count
                force_plate_channel_count = int.from_bytes(
                    data[offset : offset + 4], byteorder="little", signed=True
                )
                offset += 4

                trace_mf(
                    f"\tForce Plate {i:3d} ID: {force_plate_id:3d} Num Channels: {force_plate_channel_count:3d}"
                )

                # Channel Data
                for j in range(force_plate_channel_count):
                    fp_channel_data = MoCapData.ForcePlateChannelData()
                    force_plate_channel_frame_count = int.from_bytes(
                        data[offset : offset + 4], byteorder="little", signed=True
                    )
                    offset += 4
                    out_string = f"\tChannel {j:3d}: "
                    out_string += f"  {force_plate_channel_frame_count:3d} Frames - Frame Data: "

                    # Force plate frames
                    n_frames_show = min(force_plate_channel_frame_count, n_frames_show_max)
                    for k in range(force_plate_channel_frame_count):
                        (force_plate_channel_val,) = FloatValue.unpack(data[offset : offset + 4])
                        offset += 4
                        fp_channel_data.add_frame_entry(force_plate_channel_val)

                        if k < n_frames_show:
                            out_string += f" {force_plate_channel_val:3.2f} "
                    if n_frames_show < force_plate_channel_frame_count:
                        out_string += f" showing {n_frames_show:3d} of {force_plate_channel_frame_count:3d} frames"
                    force_plate.add_channel_data(fp_channel_data)
                force_plate_data.add_force_plate(force_plate)
        return offset, force_plate_data

    def __unpack_device_data(
        self, data: bytes | memoryview, packet_size: int, major: int, minor: int
    ) -> tuple[int, MoCapData.DeviceData]:
        device_data = MoCapData.DeviceData()
        n_frames_show_max = 4
        offset = 0
        # Device data (version 2.11 and later)
        device_count = 0
        if (major == 2 and minor >= 11) or (major > 2):
            device_count = int.from_bytes(
                data[offset : offset + 4], byteorder="little", signed=True
            )
            offset += 4
            trace_mf(f"Device Count: {device_count}")

            # get data size (4 bytes)
            offset_tmp, unpackedDataSize = self.__unpack_data_size(data[offset:], major, minor)
            offset += offset_tmp

            for i in range(0, device_count):
                # ID
                device_id = int.from_bytes(
                    data[offset : offset + 4], byteorder="little", signed=True
                )
                offset += 4
                device = MoCapData.Device(device_id)
                # Channel Count
                device_channel_count = int.from_bytes(
                    data[offset : offset + 4], byteorder="little", signed=True
                )
                offset += 4

                trace_mf(
                    f"\tDevice {i:3d}      ID: {device_id:3d} Num Channels: {device_channel_count:3d}"
                )

                # Channel Data
                for j in range(0, device_channel_count):
                    device_channel_data = MoCapData.DeviceChannelData()
                    device_channel_frame_count = int.from_bytes(
                        data[offset : offset + 4], byteorder="little", signed=True
                    )
                    offset += 4
                    out_string = f"\tChannel {j:3d} "
                    out_string += f"  {device_channel_frame_count:3d} Frames - Frame Data: "

                    # Device Frame Data
                    n_frames_show = min(device_channel_frame_count, n_frames_show_max)
                    for k in range(0, device_channel_frame_count):
                        device_channel_val = int.from_bytes(
                            data[offset : offset + 4], byteorder="little", signed=True
                        )
                        (device_channel_val,) = FloatValue.unpack(data[offset : offset + 4])
                        offset += 4
                        if k < n_frames_show:
                            out_string += f" {device_channel_val:3.2f} "

                        device_channel_data.add_frame_entry(device_channel_val)
                    if n_frames_show < device_channel_frame_count:
                        out_string += (
                            f" showing {n_frames_show:3d} of {device_channel_frame_count:3d} frames"
                        )
                    trace_mf(f" {out_string}")
                    device.add_channel_data(device_channel_data)
                device_data.add_device(device)
        return offset, device_data

    def __unpack_frame_suffix_data_4_1_to_present(
        self,
        data: bytes | memoryview,
        offset: int,
        frame_suffix_data: MoCapData.FrameSuffixData,
        param: int,
    ) -> tuple[bytes | memoryview, int, MoCapData.FrameSuffixData, int]:
        """Unpacks frame suffix data from NatNet 4.1 to present NatNet"""
        (timestamp,) = DoubleValue.unpack(data[offset : offset + 8])
        offset += 8
        trace_mf(f"Timestamp: {timestamp:3.2f}")
        frame_suffix_data.timestamp = timestamp

        stamp_camera_mid_exposure = int.from_bytes(
            data[offset : offset + 8], byteorder="little", signed=True
        )
        trace_mf(f"Mid-exposure timestamp        : {stamp_camera_mid_exposure:3d}")
        offset += 8
        frame_suffix_data.stamp_camera_mid_exposure = stamp_camera_mid_exposure

        stamp_data_received = int.from_bytes(
            data[offset : offset + 8], byteorder="little", signed=True
        )
        offset += 8
        frame_suffix_data.stamp_data_received = stamp_data_received
        trace_mf(f"Camera data received timestamp: {stamp_data_received:3d}")

        stamp_transmit = int.from_bytes(data[offset : offset + 8], byteorder="little", signed=True)
        offset += 8
        trace_mf(f"Transmit timestamp            : {stamp_transmit:3d}")
        frame_suffix_data.stamp_transmit = stamp_transmit

        prec_timestamp_secs = int.from_bytes(
            data[offset : offset + 4], byteorder="little", signed=True
        )
        # hours = int(prec_timestamp_secs/3600)
        # minutes=int(prec_timestamp_secs/60)%60
        # seconds=prec_timestamp_secs%60
        # out_string= "Precision timestamp (h:m:s) - %4d:%2d:%2d" % (hours, minutes, seconds) #type: ignore
        # trace_mf(" %s" %out_string)
        trace_mf(f"Precision timestamp (sec)     : {prec_timestamp_secs:3d}")
        offset += 4
        frame_suffix_data.prec_timestamp_secs = prec_timestamp_secs

        prec_timestamp_frac_secs = int.from_bytes(
            data[offset : offset + 4], byteorder="little", signed=True
        )
        trace_mf(f"Precision timestamp (frac sec): {prec_timestamp_frac_secs:3d}")
        offset += 4
        frame_suffix_data.prec_timestamp_frac_secs = prec_timestamp_frac_secs
        (param,) = struct.unpack("h", data[offset : offset + 2])
        offset += 2

        return data, offset, frame_suffix_data, param

    def __unpack_frame_suffix_data_3_to_4(
        self,
        data: bytes | memoryview,
        offset: int,
        frame_suffix_data: MoCapData.FrameSuffixData,
        param: int,
    ) -> tuple[bytes | memoryview, int, MoCapData.FrameSuffixData, int]:
        """Unpacks frame suffix data inclusive from NatNet 3 to NatNet 4"""
        (timestamp,) = DoubleValue.unpack(data[offset : offset + 8])
        offset += 8
        trace_mf(f"Timestamp: {timestamp:3.2f}")
        frame_suffix_data.timestamp = timestamp
        stamp_camera_mid_exposure = int.from_bytes(
            data[offset : offset + 8], byteorder="little", signed=True
        )
        trace_mf(f"Mid-exposure timestamp        : {stamp_camera_mid_exposure:3d}")
        offset += 8
        frame_suffix_data.stamp_camera_mid_exposure = stamp_camera_mid_exposure

        stamp_data_received = int.from_bytes(
            data[offset : offset + 8], byteorder="little", signed=True
        )
        offset += 8
        frame_suffix_data.stamp_data_received = stamp_data_received
        trace_mf(f"Camera data received timestamp: {stamp_data_received:3d}")

        stamp_transmit = int.from_bytes(data[offset : offset + 8], byteorder="little", signed=True)
        offset += 8
        trace_mf(f"Transmit timestamp            : {stamp_transmit:3d}")
        frame_suffix_data.stamp_transmit = stamp_transmit
        (param,) = struct.unpack("h", data[offset : offset + 2])
        offset += 2
        return data, offset, frame_suffix_data, param

    def __unpack_frame_suffix_data_2_7_to_3(
        self,
        data: bytes | memoryview,
        offset: int,
        frame_suffix_data: MoCapData.FrameSuffixData,
        param: int,
    ) -> tuple[bytes | memoryview, int, MoCapData.FrameSuffixData, int]:
        """Unpacks frame suffix data from inclusive of NatNet 2.7 to but not
        including NatNet 3"""
        (timestamp,) = DoubleValue.unpack(data[offset : offset + 8])
        offset += 8
        trace_mf(f"Timestamp: {timestamp:3.2f}")
        frame_suffix_data.timestamp = timestamp
        (param,) = struct.unpack("h", data[offset : offset + 2])
        offset += 2

        return data, offset, frame_suffix_data, param

    def __unpack_frame_suffix_data_pre_2_7(
        self,
        data: bytes | memoryview,
        offset: int,
        frame_suffix_data: MoCapData.FrameSuffixData,
        param: int,
    ) -> tuple[bytes | memoryview, int, MoCapData.FrameSuffixData, int]:
        """Unpacks frame suffix data for any NatNet version before
        NatNet 2.7"""
        (timestamp,) = FloatValue.unpack(data[offset : offset + 4])
        offset += 4
        trace_mf(f"Timestamp: {timestamp:3.2f}")
        frame_suffix_data.timestamp = timestamp
        (param,) = struct.unpack("h", data[offset : offset + 2])
        offset += 2

        return data, offset, frame_suffix_data, param

    def __unpack_frame_suffix_data_0_case(
        self,
        data: bytes | memoryview,
        offset: int,
        frame_suffix_data: MoCapData.FrameSuffixData,
        param: int,
    ) -> tuple[bytes | memoryview, int, MoCapData.FrameSuffixData, int]:
        """Unpacks frame suffix data if the major case is 0"""
        (timestamp,) = DoubleValue.unpack(data[offset : offset + 8])
        offset += 8
        trace_mf(f"Timestamp: {timestamp:3.2f}")
        frame_suffix_data.timestamp = timestamp
        (param,) = struct.unpack("h", data[offset : offset + 2])
        offset += 2
        return data, offset, frame_suffix_data, param

    def __unpack_frame_suffix_data(
        self, data: bytes | memoryview, packet_size: int, major: int, minor: int
    ) -> tuple[int, MoCapData.FrameSuffixData]:
        frame_suffix_data = MoCapData.FrameSuffixData()
        offset = 0

        # Timecode
        timecode = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4
        frame_suffix_data.timecode = timecode

        timecode_sub = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4
        frame_suffix_data.timecode_sub = timecode_sub

        param = 0
        # check to see if there is enough data
        if (packet_size - offset) <= 0:
            print("ERROR: Early End of Data Frame Suffix Data")
            print("\tNo time stamp info available")
        else:
            if major == 0:
                data, offset, frame_suffix_data, param = self.__unpack_frame_suffix_data_0_case(
                    data, offset, frame_suffix_data, param
                )
            elif major < 2 or (major <= 2 and minor < 7):
                data, offset, frame_suffix_data, param = self.__unpack_frame_suffix_data_pre_2_7(
                    data, offset, frame_suffix_data, param
                )
            elif major == 2 and minor >= 7 and major < 3:
                data, offset, frame_suffix_data, param = self.__unpack_frame_suffix_data_2_7_to_3(
                    data, offset, frame_suffix_data, param
                )
            elif major >= 3 or (major == 4 and minor == 0):
                data, offset, frame_suffix_data, param = self.__unpack_frame_suffix_data_3_to_4(
                    data, offset, frame_suffix_data, param
                )
            elif major >= 4 and minor != 0:
                data, offset, frame_suffix_data, param = (
                    self.__unpack_frame_suffix_data_4_1_to_present(
                        data, offset, frame_suffix_data, param
                    )
                )

        is_recording = (param & 0x01) != 0
        tracked_models_changed = (param & 0x02) != 0
        frame_suffix_data.param = param
        frame_suffix_data.is_recording = is_recording
        frame_suffix_data.tracked_models_changed = tracked_models_changed

        return offset, frame_suffix_data

    # Unpack data from a motion capture frame message
    def __unpack_mocap_data(
        self, data: bytes | memoryview, packet_size: int, major: int, minor: int
    ) -> tuple[int, MoCapData.MoCapData]:
        mocap_data = MoCapData.MoCapData()
        data = memoryview(data)
        offset = 0
        rel_offset = 0
        # Frame Prefix Data
        rel_offset, frame_prefix_data = self.__unpack_frame_prefix_data(data[offset:])
        offset += rel_offset
        mocap_data.set_prefix_data(frame_prefix_data)
        frame_number = frame_prefix_data.frame_number

        # Markerset Data
        rel_offset, marker_set_data = self.__unpack_marker_set_data(
            data[offset:], (packet_size - offset), major, minor
        )
        offset += rel_offset
        mocap_data.set_marker_set_data(marker_set_data)
        marker_set_count = marker_set_data.get_marker_set_count()
        unlabeled_markers_count = marker_set_data.get_unlabeled_marker_count()

        # Legacy Other Markers
        rel_offset, legacy_other_markers = self.__unpack_legacy_other_markers(
            data[offset:], (packet_size - offset), major, minor
        )
        offset += rel_offset
        mocap_data.set_legacy_other_markers(legacy_other_markers)
        marker_set_count = legacy_other_markers.get_marker_count()
        legacy_other_markers_count = marker_set_data.get_unlabeled_marker_count()  # noqa F401

        # Rigid Body Data
        rel_offset, rigid_body_data = self.__unpack_rigid_body_data(
            data[offset:], (packet_size - offset), major, minor
        )
        offset += rel_offset
        mocap_data.set_rigid_body_data(rigid_body_data)
        rigid_body_count = rigid_body_data.get_rigid_body_count()

        # Skeleton Data
        rel_offset, skeleton_data = self.__unpack_skeleton_data(
            data[offset:], (packet_size - offset), major, minor
        )
        offset += rel_offset
        mocap_data.set_skeleton_data(skeleton_data)
        skeleton_count = skeleton_data.get_skeleton_count()

        # Assets (Motive 3.1/NatNet 4.1 and greater)
        asset_count = 0
        if ((major >= 4) and (minor >= 1)) or (major > 4):
            rel_offset, asset_data = self.__unpack_asset_data(
                data[offset:], (packet_size - offset), major, minor
            )
            offset += rel_offset
            mocap_data.set_asset_data(asset_data)
            asset_count = asset_data.get_asset_count()

        # Labeled Marker Data
        rel_offset, labeled_marker_data = self.__unpack_labeled_marker_data(
            data[offset:], (packet_size - offset), major, minor
        )
        offset += rel_offset
        mocap_data.set_labeled_marker_data(labeled_marker_data)
        labeled_marker_count = labeled_marker_data.get_labeled_marker_count()

        # Force Plate Data
        rel_offset, force_plate_data = self.__unpack_force_plate_data(
            data[offset:], (packet_size - offset), major, minor
        )
        offset += rel_offset
        mocap_data.set_force_plate_data(force_plate_data)

        # Device Data
        rel_offset, device_data = self.__unpack_device_data(
            data[offset:], (packet_size - offset), major, minor
        )
        offset += rel_offset
        mocap_data.set_device_data(device_data)

        # Frame Suffix Data
        # rel_offset, timecode, timecode_sub, timestamp, is_recording, tracked_models_changed = #type: ignore
        rel_offset, frame_suffix_data = self.__unpack_frame_suffix_data(
            data[offset:], (packet_size - offset), major, minor
        )
        offset += rel_offset
        mocap_data.set_suffix_data(frame_suffix_data)

        timecode = frame_suffix_data.timecode
        timecode_sub = frame_suffix_data.timecode_sub
        timestamp = frame_suffix_data.timestamp
        is_recording = frame_suffix_data.is_recording
        tracked_models_changed = frame_suffix_data.tracked_models_changed

        # Send information to any listener.
        if self.new_frame_listener is not None:
            data_dict = {}
            data_dict["frame_number"] = frame_number
            data_dict["marker_set_count"] = marker_set_count
            data_dict["unlabeled_markers_count"] = unlabeled_markers_count
            data_dict["rigid_body_count"] = rigid_body_count
            data_dict["skeleton_count"] = skeleton_count
            data_dict["asset_count"] = asset_count
            data_dict["labeled_marker_count"] = labeled_marker_count
            data_dict["timecode"] = timecode
            data_dict["timecode_sub"] = timecode_sub
            data_dict["timestamp"] = timestamp
            data_dict["is_recording"] = is_recording
            data_dict["tracked_models_changed"] = tracked_models_changed

            self.new_frame_listener(data_dict)

        if self.new_frame_with_data_listener is not None:
            data_dict = {}
            data_dict["frame_number"] = frame_number
            data_dict["marker_set_count"] = marker_set_count
            data_dict["unlabeled_markers_count"] = unlabeled_markers_count
            data_dict["rigid_body_count"] = rigid_body_count
            data_dict["skeleton_count"] = skeleton_count
            data_dict["asset_count"] = asset_count
            data_dict["labeled_marker_count"] = labeled_marker_count
            data_dict["timecode"] = timecode
            data_dict["timecode_sub"] = timecode_sub
            data_dict["timestamp"] = timestamp
            data_dict["is_recording"] = is_recording
            data_dict["tracked_models_changed"] = tracked_models_changed
            data_dict["offset"] = offset
            data_dict["mocap_data"] = mocap_data
            self.new_frame_with_data_listener(data_dict)

        return offset, mocap_data

    def __unpack_marker_set_description(
        self, data: bytes | memoryview, major: int, minor: int
    ) -> tuple[int, DataDescriptions.MarkerSetDescription]:
        """Unpack marker description packet"""
        ms_desc = DataDescriptions.MarkerSetDescription()

        offset = 0

        name, separator, remainder = bytes(data[offset:]).partition(b"\0")
        offset += len(name) + 1
        trace_dd(f"Markerset Name: {name.decode('utf-8')}")
        ms_desc.set_name(name)

        marker_count = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4
        trace_dd(f"Marker Count: {marker_count}")
        if marker_count > 0:
            for i in range(0, marker_count):
                name, separator, remainder = bytes(data[offset:]).partition(b"\0")
                offset += len(name) + 1
                trace_dd(f"\t{i:2d} Marker Name: {name.decode('utf-8')}")
                ms_desc.add_marker_name(name)

        return offset, ms_desc

    def __unpack_rigid_body_descript_4_2_to_current(
        self, data: bytes | memoryview
    ) -> tuple[int, DataDescriptions.RigidBodyDescription]:
        """Unpack rigid body helper function for NatNet 4.2"""
        rb_desc = DataDescriptions.RigidBodyDescription()
        offset = 0

        name, separator, remainder = bytes(data[offset:]).partition(b"\0")
        offset += len(name) + 1
        rb_desc.set_name(name)
        trace_dd(f"\tRigid Body Name  : {name.decode('utf-8')}")

        # ID
        new_id = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4
        rb_desc.set_id(new_id)
        trace_dd(f"\tRigid Body ID      : {new_id}")

        # Parent ID
        parent_id = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4
        rb_desc.set_parent_id(parent_id)
        trace_dd(f"\tParent ID        : {parent_id}")

        # Position Offsets
        pos = Vector3.unpack(data[offset : offset + 12])
        offset += 12
        rb_desc.set_pos(pos[0], pos[1], pos[2])

        trace_dd(f"\tPosition         : [{pos[0]:3.2f}, {pos[1]:3.2f}, {pos[2]:3.2f}]")

        quat = Quaternion.unpack(data[offset : offset + 16])
        offset += 16
        trace_dd(
            f"\tRotation         : [{quat[0]:3.2f}, {quat[1]:3.2f}, {quat[2]:3.2f}, {quat[3]:3.2f}]"
        )

        # Marker Count
        marker_count = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4
        trace_dd(f"\tNumber of Markers: {marker_count}")
        if marker_count > 0:
            trace_dd("\tMarker Positions: ")
        marker_count_range = range(0, marker_count)
        offset1 = offset
        offset2 = offset1 + (12 * marker_count)
        offset3 = offset2 + (4 * marker_count)
        # Marker Offsets X,Y,Z
        marker_name = ""
        for marker in marker_count_range:
            # Offset
            marker_offset = Vector3.unpack(data[offset1 : offset1 + 12])
            offset1 += 12

            # Active Label
            active_label = int.from_bytes(
                data[offset2 : offset2 + 4], byteorder="little", signed=True
            )
            offset2 += 4

            marker_name, separator, remainder = bytes(data[offset3:]).partition(b"\0")
            marker_name = marker_name.decode("utf-8")
            offset3 += len(marker_name) + 1

            rb_marker = DataDescriptions.RBMarker(marker_name, active_label, marker_offset)
            rb_desc.add_rb_marker(rb_marker)
            trace_dd(
                f"\t{marker:3d} Marker Label: {active_label} Position: [{marker_offset[0]:3.2f} {marker_offset[1]:3.2f} {marker_offset[2]:3.2f}] {marker_name}"
            )
            offset = offset3
        trace_dd(f"\tunpack_rigid_body_description processed bytes: {offset}")
        return offset, rb_desc

    def __unpack_rigid_body_descript_4_n_4_1(
        self, data: bytes | memoryview
    ) -> tuple[int, DataDescriptions.RigidBodyDescription]:
        """Unpack rigid body description data for NatNet Versions 4 and
        4.1"""

        rb_desc = DataDescriptions.RigidBodyDescription()
        offset = 0

        name, separator, remainder = bytes(data[offset:]).partition(b"\0")
        offset += len(name) + 1
        rb_desc.set_name(name)
        trace_dd(f"\tRigid Body Name  : {name.decode('utf-8')}")

        # ID
        new_id = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4
        rb_desc.set_id(new_id)
        trace_dd(f"\tRigid Body ID      : {new_id}")

        # Parent ID
        parent_id = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4
        rb_desc.set_parent_id(parent_id)
        trace_dd(f"\tParent ID        : {parent_id}")

        # Position Offsets
        pos = Vector3.unpack(data[offset : offset + 12])
        offset += 12
        rb_desc.set_pos(pos[0], pos[1], pos[2])

        trace_dd(f"\tPosition         : [{pos[0]:3.2f}, {pos[1]:3.2f}, {pos[2]:3.2f}]")

        # Marker Count
        marker_count = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4
        trace_dd(f"\tNumber of Markers: {marker_count}")
        if marker_count > 0:
            trace_dd("\tMarker Positions: ")
        marker_count_range = range(0, marker_count)
        offset1 = offset
        offset2 = offset1 + (12 * marker_count)
        offset3 = offset2 + (4 * marker_count)
        # Marker Offsets X,Y,Z
        marker_name = ""
        for marker in marker_count_range:
            # Offset
            marker_offset = Vector3.unpack(data[offset1 : offset1 + 12])
            offset1 += 12

            # Active Label
            active_label = int.from_bytes(
                data[offset2 : offset2 + 4], byteorder="little", signed=True
            )
            offset2 += 4

            marker_name, separator, remainder = bytes(data[offset3:]).partition(b"\0")
            marker_name = marker_name.decode("utf-8")
            offset3 += len(marker_name) + 1

            rb_marker = DataDescriptions.RBMarker(marker_name, active_label, marker_offset)
            rb_desc.add_rb_marker(rb_marker)
            trace_dd(
                f"\t{marker:3d} Marker Label: {active_label} Position: [{marker_offset[0]:3.2f} {marker_offset[1]:3.2f} {marker_offset[2]:3.2f}] {marker_name}"
            )
            offset = offset3

        trace_dd(f"\tunpack_rigid_body_description processed bytes: {offset}")
        return offset, rb_desc

    def __unpack_rigid_body_descript_3_to_4_0(
        self, data: bytes | memoryview
    ) -> tuple[int, DataDescriptions.RigidBodyDescription]:
        """Helper function for NatNets versions 3 to 4.0
        not inclusive of 4.0"""
        rb_desc = DataDescriptions.RigidBodyDescription()
        offset = 0

        name, separator, remainder = bytes(data[offset:]).partition(b"\0")
        offset += len(name) + 1
        rb_desc.set_name(name)
        trace_dd(f"\tRigid Body Name  : {name.decode('utf-8')}")

        # ID
        new_id = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4
        rb_desc.set_id(new_id)
        trace_dd(f"\tRigid Body ID      : {new_id}")

        # Parent ID
        parent_id = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4
        rb_desc.set_parent_id(parent_id)
        trace_dd(f"\tParent ID        : {parent_id}")

        # Position Offsets
        pos = Vector3.unpack(data[offset : offset + 12])
        offset += 12
        rb_desc.set_pos(pos[0], pos[1], pos[2])

        trace_dd(f"\tPosition         : [{pos[0]:3.2f}, {pos[1]:3.2f}, {pos[2]:3.2f}]")

        # Marker Count
        marker_count = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4
        trace_dd(f"\tNumber of Markers: {marker_count}")
        if marker_count > 0:
            trace_dd("\tMarker Positions: ")
        marker_count_range = range(0, marker_count)
        offset1 = offset
        offset2 = offset1 + (12 * marker_count)
        offset3 = offset2 + (4 * marker_count)
        # Marker Offsets X,Y,Z
        marker_name = ""
        for marker in marker_count_range:
            # Offset
            marker_offset = Vector3.unpack(data[offset1 : offset1 + 12])
            offset1 += 12

            # Active Label
            active_label = int.from_bytes(
                data[offset2 : offset2 + 4], byteorder="little", signed=True
            )
            offset2 += 4

            rb_marker = DataDescriptions.RBMarker(marker_name, active_label, marker_offset)
            rb_desc.add_rb_marker(rb_marker)
            trace_dd(
                f"\t{marker:3d} Marker Label: {active_label} Position: [{marker_offset[0]:3.2f} {marker_offset[1]:3.2f} {marker_offset[2]:3.2f}] {marker_name}"
            )
            offset = offset3

        trace_dd(f"\tunpack_rigid_body_description processed bytes: {offset}")
        return offset, rb_desc

    def __unpack_rigid_body_descript_2_to_3(
        self, data: bytes | memoryview
    ) -> tuple[int, DataDescriptions.RigidBodyDescription]:
        """Helper function for NatNet version inclusive 2,
        to not inclusive 3"""
        rb_desc = DataDescriptions.RigidBodyDescription()
        offset = 0

        name, separator, remainder = bytes(data[offset:]).partition(b"\0")
        offset += len(name) + 1
        rb_desc.set_name(name)
        trace_dd(f"\tRigid Body Name  : {name.decode('utf-8')}")

        # ID
        new_id = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4
        rb_desc.set_id(new_id)
        trace_dd(f"\tRigid Body ID      : {new_id}")

        # Parent ID
        parent_id = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4
        rb_desc.set_parent_id(parent_id)
        trace_dd(f"\tParent ID        : {parent_id}")

        # Position Offsets
        pos = Vector3.unpack(data[offset : offset + 12])
        offset += 12
        rb_desc.set_pos(pos[0], pos[1], pos[2])

        trace_dd(f"\tPosition         : [{pos[0]:3.2f}, {pos[1]:3.2f}, {pos[2]:3.2f}]")

        trace_dd(f"\tunpack_rigid_body_description processed bytes: {offset}")
        return offset, rb_desc

    def __unpack_rigid_body_descript_under_2(
        self, data: bytes | memoryview
    ) -> tuple[int, DataDescriptions.RigidBodyDescription]:
        """Helper function for NatNet versions under 2"""
        rb_desc = DataDescriptions.RigidBodyDescription()
        offset = 0

        # ID
        new_id = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4
        rb_desc.set_id(new_id)
        trace_dd(f"\tRigid Body ID      : {new_id}")

        # Parent ID
        parent_id = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4
        rb_desc.set_parent_id(parent_id)
        trace_dd(f"\tParent ID        : {parent_id}")

        # Position Offsets
        pos = Vector3.unpack(data[offset : offset + 12])
        offset += 12
        rb_desc.set_pos(pos[0], pos[1], pos[2])

        trace_dd(f"\tPosition         : [{pos[0]:3.2f}, {pos[1]:3.2f}, {pos[2]:3.2f}]")

        trace_dd(f"\tunpack_rigid_body_description processed bytes: {offset}")
        return offset, rb_desc

    def __unpack_rigid_body_descript_0_case(
        self, data: bytes | memoryview
    ) -> tuple[int, DataDescriptions.RigidBodyDescription]:
        """Helper function for NatNet 0 case"""
        rb_desc = DataDescriptions.RigidBodyDescription()
        offset = 0

        name, separator, remainder = bytes(data[offset:]).partition(b"\0")
        offset += len(name) + 1
        rb_desc.set_name(name)
        trace_dd(f"\tRigid Body Name  : {name.decode('utf-8')}")

        # ID
        new_id = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4
        rb_desc.set_id(new_id)
        trace_dd(f"\tRigid Body ID      : {new_id}")

        # Parent ID
        parent_id = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4
        rb_desc.set_parent_id(parent_id)
        trace_dd(f"\tParent ID        : {parent_id}")

        # Position Offsets
        pos = Vector3.unpack(data[offset : offset + 12])
        offset += 12
        rb_desc.set_pos(pos[0], pos[1], pos[2])

        trace_dd(f"\tPosition         : [{pos[0]:3.2f}, {pos[1]:3.2f}, {pos[2]:3.2f}]")

        quat = Quaternion.unpack(data[offset : offset + 16])
        offset += 16
        trace_dd(
            f"\tRotation         : [{quat[0]:3.2f}, {quat[1]:3.2f}, {quat[2]:3.2f}, {quat[3]:3.2f}]"
        )

        # Marker Count
        marker_count = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4
        trace_dd(f"\tNumber of Markers: {marker_count}")
        if marker_count > 0:
            trace_dd("\tMarker Positions: ")
        marker_count_range = range(0, marker_count)
        offset1 = offset
        offset2 = offset1 + (12 * marker_count)
        offset3 = offset2 + (4 * marker_count)
        # Marker Offsets X,Y,Z
        marker_name = ""
        for marker in marker_count_range:
            # Offset
            marker_offset = Vector3.unpack(data[offset1 : offset1 + 12])
            offset1 += 12

            # Active Label
            active_label = int.from_bytes(
                data[offset2 : offset2 + 4], byteorder="little", signed=True
            )
            offset2 += 4

            marker_name, separator, remainder = bytes(data[offset3:]).partition(b"\0")
            marker_name = marker_name.decode("utf-8")
            offset3 += len(marker_name) + 1

            rb_marker = DataDescriptions.RBMarker(marker_name, active_label, marker_offset)
            rb_desc.add_rb_marker(rb_marker)
            trace_dd(
                f"\t{marker:3d} Marker Label: {active_label} Position: [{marker_offset[0]:3.2f} {marker_offset[1]:3.2f} {marker_offset[2]:3.2f}] {marker_name}"
            )
            offset = offset3
        trace_dd(f"\tunpack_rigid_body_description processed bytes: {offset}")
        return offset, rb_desc

    def __unpack_rigid_body_description(
        self, data: bytes | memoryview, major: int, minor: int
    ) -> tuple[int, DataDescriptions.RigidBodyDescription]:
        if major == 0:
            offset, rb_desc = self.__unpack_rigid_body_descript_0_case(data)
        elif major == 4 and minor >= 2:
            offset, rb_desc = self.__unpack_rigid_body_descript_4_2_to_current(data)
        elif major == 4:
            offset, rb_desc = self.__unpack_rigid_body_descript_4_n_4_1(data)
        elif major == 3:
            offset, rb_desc = self.__unpack_rigid_body_descript_3_to_4_0(data)
        elif major == 2:
            offset, rb_desc = self.__unpack_rigid_body_descript_2_to_3(data)
        elif major < 2:
            offset, rb_desc = self.__unpack_rigid_body_descript_under_2(data)
        else:
            raise ValueError(f"Invalid Version {major:1d}.{minor:1d}")

        return offset, rb_desc

    # Unpack a skeleton description packet
    def __unpack_skeleton_description(
        self, data: bytes | memoryview, major: int, minor: int
    ) -> tuple[int, DataDescriptions.SkeletonDescription]:
        skeleton_desc = DataDescriptions.SkeletonDescription()
        offset = 0

        # Name
        name, separator, remainder = bytes(data[offset:]).partition(b"\0")
        offset += len(name) + 1
        skeleton_desc.set_name(name)
        trace_dd(f"Name: {name.decode('utf-8')}")

        # ID
        new_id = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4
        skeleton_desc.set_id(new_id)
        trace_dd(f"ID: {new_id:3d}")

        # # of RigidBodies
        rigid_body_count = int.from_bytes(
            data[offset : offset + 4], byteorder="little", signed=True
        )
        offset += 4
        trace_dd(f"Rigid Body (Bone) Count: {rigid_body_count:3d}")

        # Loop over all Rigid Bodies
        for i in range(0, rigid_body_count):
            trace_dd(f"Rigid Body (Bone) {i}:")
            offset_tmp, rb_desc_tmp = self.__unpack_rigid_body_description(
                data[offset:], major, minor
            )
            offset += offset_tmp
            skeleton_desc.add_rigid_body_description(rb_desc_tmp)
        return offset, skeleton_desc

    def __unpack_force_plate_description(
        self, data: bytes | memoryview, major: int, minor: int
    ) -> tuple[int, DataDescriptions.ForcePlateDescription]:
        fp_desc = None
        offset = 0
        if major >= 3:
            fp_desc = DataDescriptions.ForcePlateDescription()
            # ID
            new_id = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
            offset += 4
            fp_desc.set_id(new_id)
            trace_dd(f"\tID: {new_id}")

            # Serial Number
            serial_number, separator, remainder = bytes(data[offset:]).partition(b"\0")
            offset += len(serial_number) + 1
            fp_desc.set_serial_number(serial_number)
            trace_dd(f"\tSerial Number: {serial_number.decode('utf-8')}")

            # Dimensions
            f_width = FloatValue.unpack(data[offset : offset + 4])
            offset += 4
            trace_dd(f"\tWidth : {f_width[0]:3.2f}")
            f_length = FloatValue.unpack(data[offset : offset + 4])
            offset += 4
            fp_desc.set_dimensions(f_width[0], f_length[0])
            trace_dd(f"\tLength: {f_length[0]:3.2f}")

            # Origin
            origin = Vector3.unpack(data[offset : offset + 12])
            offset += 12
            fp_desc.set_origin(origin[0], origin[1], origin[2])
            trace_dd(f"\tOrigin: [{origin[0]:3.2f}, {origin[1]:3.2f}, {origin[2]:3.2f}]")

            # Calibration Matrix 12x12 floats
            trace_dd("Cal Matrix:")
            cal_matrix_tmp = [[0.0 for col in range(12)] for row in range(12)]

            for i in range(0, 12):
                cal_matrix_row = FPCalMatrixRow.unpack(data[offset : offset + (12 * 4)])
                trace_dd(
                    f"\t{i:3d} {cal_matrix_row[0]:3.3e} {cal_matrix_row[1]:3.3e} {cal_matrix_row[2]:3.3e} {cal_matrix_row[3]:3.3e} {cal_matrix_row[4]:3.3e} {cal_matrix_row[5]:3.3e} {cal_matrix_row[6]:3.3e} {cal_matrix_row[7]:3.3e} {cal_matrix_row[8]:3.3e} {cal_matrix_row[9]:3.3e} {cal_matrix_row[10]:3.3e} {cal_matrix_row[11]:3.3e}"
                )
                cal_matrix_tmp[i] = list(copy.deepcopy(cal_matrix_row))
                offset += 12 * 4
            fp_desc.set_cal_matrix(cal_matrix_tmp)
            # Corners 4x3 floats
            corners = FPCorners.unpack(data[offset : offset + (12 * 4)])
            offset += 12 * 4
            o_2 = 0
            trace_dd("Corners:")
            corners_tmp = [[0.0 for col in range(3)] for row in range(4)]
            for i in range(0, 4):
                trace_dd(
                    f"\t{i:3d} {corners[o_2]:3.3e} {corners[o_2 + 1]:3.3e} {corners[o_2 + 2]:3.3e}"
                )
                corners_tmp[i][0] = corners[o_2]
                corners_tmp[i][1] = corners[o_2 + 1]
                corners_tmp[i][2] = corners[o_2 + 2]
                o_2 += 3
            fp_desc.set_corners(corners_tmp)

            # Plate Type int
            plate_type = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
            offset += 4
            fp_desc.set_plate_type(plate_type)
            trace_dd(f"Plate Type: {plate_type}")

            # Channel Data Type int
            channel_data_type = int.from_bytes(
                data[offset : offset + 4], byteorder="little", signed=True
            )
            offset += 4
            fp_desc.set_channel_data_type(channel_data_type)
            trace_dd(f"Channel Data Type: {channel_data_type}")

            # Number of Channels int
            num_channels = int.from_bytes(
                data[offset : offset + 4], byteorder="little", signed=True
            )
            offset += 4
            trace_dd(f"Number of Channels: {num_channels}")

            # Channel Names list of NoC strings
            for i in range(0, num_channels):
                channel_name, separator, remainder = bytes(data[offset:]).partition(b"\0")
                offset += len(channel_name) + 1
                trace_dd(f"\tChannel Name {i:3d}: {channel_name.decode('utf-8')}")
                fp_desc.add_channel_name(channel_name)
        else:
            raise ValueError(f"Invalid Version {major:1d}.{minor:1d}")

        trace_dd(f"unpackForcePlate processed {offset} bytes")
        return offset, fp_desc

    def __unpack_device_description(
        self, data: bytes | memoryview, major: int, minor: int
    ) -> tuple[int, DataDescriptions.DeviceDescription]:
        device_desc = None
        offset = 0
        if major >= 3:
            # new_id
            new_id = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
            offset += 4
            trace_dd(f"\tID: {new_id}")

            # Name
            name, separator, remainder = bytes(data[offset:]).partition(b"\0")
            offset += len(name) + 1
            trace_dd(f"\tName: {name.decode('utf-8')}")

            # Serial Number
            serial_number, separator, remainder = bytes(data[offset:]).partition(b"\0")
            offset += len(serial_number) + 1
            trace_dd(f"\tSerial Number: {serial_number.decode('utf-8')}")

            # Device Type int
            device_type = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
            offset += 4
            trace_dd(f"Device Type: {device_type}")

            # Channel Data Type int
            channel_data_type = int.from_bytes(
                data[offset : offset + 4], byteorder="little", signed=True
            )
            offset += 4
            trace_dd(f"Channel Data Type: {channel_data_type}")

            device_desc = DataDescriptions.DeviceDescription(
                new_id, name, serial_number, device_type, channel_data_type
            )

            # Number of Channels int
            num_channels = int.from_bytes(
                data[offset : offset + 4], byteorder="little", signed=True
            )
            offset += 4
            trace_dd(f"Number of Channels: {num_channels}")

            # Channel Names list of NoC strings
            for i in range(0, num_channels):
                channel_name, separator, remainder = bytes(data[offset:]).partition(b"\0")
                offset += len(channel_name) + 1
                device_desc.add_channel_name(channel_name)
                trace_dd(f"\tChannel {i} Name: {channel_name.decode('utf-8')}")
        else:
            raise ValueError(f"Invalid Version {major:1d}.{minor:1d}")

        trace_dd(f"unpack_device_description processed {offset} bytes")
        return offset, device_desc

    def __unpack_camera_description(
        self, data: bytes | memoryview, major: int, minor: int
    ) -> tuple[int, DataDescriptions.CameraDescription]:
        offset = 0
        # Name
        name, separator, remainder = bytes(data[offset:]).partition(b"\0")
        offset += len(name) + 1
        trace_dd(f"\tName      : {name.decode('utf-8')}")
        # Position
        position = Vector3.unpack(data[offset : offset + 12])
        offset += 12
        trace_dd(f"\tPosition  : [{position[0]:3.2f}, {position[1]:3.2f}, {position[2]:3.2f}]")

        # Orientation
        orientation = Quaternion.unpack(data[offset : offset + 16])
        offset += 16
        trace_dd(
            f"\tOrientation: [{orientation[0]:3.2f}, {orientation[1]:3.2f}, {orientation[2]:3.2f}, {orientation[3]:3.2f}]"
        )
        trace_dd(f"unpack_camera_description processed {offset:3d} bytes")

        camera_desc = DataDescriptions.CameraDescription(name, position, orientation)
        return offset, camera_desc

    def __unpack_marker_description(
        self, data: bytes | memoryview, major: int, minor: int
    ) -> tuple[int, DataDescriptions.MarkerDescription]:
        offset = 0

        # Name
        name, separator, remainder = bytes(data[offset:]).partition(b"\0")
        offset += len(name) + 1
        trace_dd(f"\tName      : {name.decode('utf-8')}")

        # ID
        marker_id = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4
        trace_dd(f"\tID        : {marker_id}")

        # Initial Position
        initialPosition = Vector3.unpack(data[offset : offset + 12])
        offset += 12
        trace_dd(
            f"\tPosition  : [{initialPosition[0]:3.2f}, {initialPosition[1]:3.2f}, {initialPosition[2]:3.2f}]"
        )

        # Size
        (marker_size,) = FloatValue.unpack(data[offset : offset + 4])
        offset += 4
        trace_mf(f"\tMarker Size: {marker_size}")

        # Params
        (marker_params,) = struct.unpack("h", data[offset : offset + 2])
        offset += 2
        trace_mf(f"\tParams    : {marker_params}")

        trace_dd(f"\tunpack_marker_description processed {offset:3d} bytes")

        # Package for return object
        marker_desc = DataDescriptions.MarkerDescription(
            name, marker_id, initialPosition, marker_size, marker_params
        )
        return offset, marker_desc

    def __unpack_asset_rigid_body_data(
        self, data: bytes | memoryview, major: int, minor: int
    ) -> tuple[int, MoCapData.AssetRigidBodyData]:
        offset = 0
        # ID
        rbID = int.from_bytes(data[offset : offset + 4], "little", signed=True)
        offset += 4
        trace_dd(f"\tID        : {rbID}")

        # Position: x,y,z
        pos = Vector3.unpack(data[offset : offset + 12])
        offset += 12
        trace_mf(f"\tPosition   : [{pos[0]:3.2f}, {pos[1]:3.2f}, {pos[2]:3.2f}]")

        # Orientation: qx, qy, qz, qw
        rot = Quaternion.unpack(data[offset : offset + 16])
        offset += 16
        trace_mf(f"\tOrientation: [{rot[0]:3.2f}, {rot[1]:3.2f}, {rot[2]:3.2f}, {rot[3]:3.2f}]")

        # Mean error
        (mean_error,) = FloatValue.unpack(data[offset : offset + 4])
        offset += 4
        trace_mf(f"\tMean Error : {mean_error:3.2f}")

        # Params
        (marker_params,) = struct.unpack("h", data[offset : offset + 2])
        offset += 2
        trace_mf(f"\tParams     : {marker_params}")

        trace_dd(f"unpack_marker_description processed {offset:3d} bytes")
        # Package for return object
        rigid_body_data = MoCapData.AssetRigidBodyData(rbID, pos, rot, mean_error, marker_params)

        return offset, rigid_body_data

    def __unpack_asset_marker_data(
        self, data: bytes | memoryview, major: int, minor: int
    ) -> tuple[int, MoCapData.AssetMarkerData]:
        offset = 0
        # ID
        marker_id = int.from_bytes(data[offset : offset + 4], "little", signed=True)
        offset += 4
        trace_dd(f"\tID         : {marker_id}")

        # Position: x,y,z
        pos = Vector3.unpack(data[offset : offset + 12])
        offset += 12
        trace_mf(f"\tPosition   : [{pos[0]:3.2f}, {pos[1]:3.2f}, {pos[2]:3.2f}]")

        # Size
        (marker_size,) = FloatValue.unpack(data[offset : offset + 4])
        offset += 4
        trace_mf(f"\tMarker Size: {marker_size:3.2f}")

        # Params
        (marker_params,) = struct.unpack("h", data[offset : offset + 2])
        offset += 2
        trace_mf(f"\tParams     : {marker_params}")

        # Residual
        (residual,) = FloatValue.unpack(data[offset : offset + 4])
        offset += 4
        trace_mf(f"\tResidual   : {residual:3.2f}")

        marker_data = MoCapData.AssetMarkerData(
            marker_id, pos, marker_size, marker_params, residual
        )
        return offset, marker_data

    def __unpack_asset_data(
        self, data: bytes | memoryview, packet_size: int, major: int, minor: int
    ) -> tuple[int, MoCapData.AssetData]:
        asset_data = MoCapData.AssetData()

        offset = 0

        # Asset Count
        asset_count = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4
        trace_mf(f"Asset Count: {asset_count}")

        # Get data size (4 bytes)
        offset_tmp, unpackedDataSize = self.__unpack_data_size(data[offset:], major, minor)
        offset += offset_tmp

        # Unpack assets
        for asset_num in range(0, asset_count):
            rel_offset, asset = self.__unpack_asset(data[offset:], major, minor, asset_num)
            offset += rel_offset
            asset_data.add_asset(asset)

        return offset, asset_data

    def __unpack_asset_description(
        self, data: bytes | memoryview, major: int, minor: int
    ) -> tuple[int, DataDescriptions.AssetDescription]:
        offset = 0

        # Name
        name, separator, remainder = bytes(data[offset:]).partition(b"\0")
        offset += len(name) + 1
        trace_dd(f"\tName      : {name.decode('utf-8')}")

        # Asset Type 4 bytes
        assetType = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4
        trace_dd(f"\tType      : {assetType}")

        # ID 4 bytes
        assetID = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4
        trace_dd(f"\tID        : {assetID}")

        # # of RigidBodies
        numRBs = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4
        trace_dd(f"\tRigid Body (Bone) Count: {numRBs}")
        rigidbodyArray = []
        offset1 = 0
        for rbNum in range(numRBs):
            # # of RigidBodies
            trace_dd(f"\tRigid Body (Bone) {rbNum}:")
            offset1, rigidbody = self.__unpack_rigid_body_description(data[offset:], major, minor)
            offset += offset1
            rigidbodyArray.append(rigidbody)
        # # of Markers
        numMarkers = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4
        trace_dd(f"\tMarker Count: {numMarkers}")
        markerArray = []
        for markerNum in range(numMarkers):
            # # of Markers
            trace_dd(f"\tMarker {markerNum}:")
            offset1, marker = self.__unpack_marker_description(data[offset:], major, minor)
            offset += offset1
            markerArray.append(marker)

        trace_dd(f"\tunpack_asset_description processed {offset:3d} bytes")

        # package for output
        asset_desc = DataDescriptions.AssetDescription(
            name, assetType, assetID, rigidbodyArray, markerArray
        )
        return offset, asset_desc

    # Unpack a data description packet
    def __unpack_data_descriptions(
        self, data: bytes | memoryview, packet_size: int, major: int, minor: int
    ) -> tuple[int, DataDescriptions.DataDescriptions]:
        data_descs = DataDescriptions.DataDescriptions()
        offset = 0
        # # of data sets to process
        dataset_count = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
        offset += 4
        trace_dd(f"Dataset Count: {dataset_count}")
        for i in range(0, dataset_count):
            trace_dd(f"Dataset {i}")
            data_type = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
            offset += 4
            if ((major == 4) and (minor >= 1)) or (major > 4):
                _ = int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)
                offset += 4
            data_tmp = None
            if data_type == 0:
                trace_dd("Type: 0 Markerset")
                offset_tmp, data_tmp = self.__unpack_marker_set_description(
                    data[offset:], major, minor
                )
            elif data_type == 1:
                trace_dd("Type: 1 Rigid Body")
                offset_tmp, data_tmp = self.__unpack_rigid_body_description(
                    data[offset:], major, minor
                )
            elif data_type == 2:
                trace_dd("Type: 2 Skeleton")
                offset_tmp, data_tmp = self.__unpack_skeleton_description(
                    data[offset:], major, minor
                )
            elif data_type == 3:
                trace_dd("Type: 3 Force Plate")
                offset_tmp, data_tmp = self.__unpack_force_plate_description(
                    data[offset:], major, minor
                )
            elif data_type == 4:
                trace_dd("Type: 4 Device")
                offset_tmp, data_tmp = self.__unpack_device_description(data[offset:], major, minor)
            elif data_type == 5:
                trace_dd("Type: 5 Camera")
                offset_tmp, data_tmp = self.__unpack_camera_description(data[offset:], major, minor)
            elif data_type == 6:
                trace_dd("Type: 6 Asset")
                offset_tmp, data_tmp = self.__unpack_asset_description(data[offset:], major, minor)
            else:
                print("Type: Unknown " + str(data_type))
                print("ERROR: Type decode failure")
                print(f"\t{(i + 1)} datasets processed of {dataset_count}")
                print(f"\t{offset} bytes processed of {packet_size}")
                print("\tPACKET DECODE STOPPED")
                return offset, data_descs
            offset += offset_tmp
            data_descs.add_data(data_tmp)
            trace_dd(f"\t{(i + 1)} datasets processed of {dataset_count}")
            trace_dd(f"\t{offset} bytes processed of {packet_size}")

        return offset, data_descs

    # __unpack_server_info is for local use of the client
    # and will update the values for the versions/ NatNet capabilities
    # of the server.
    def __unpack_server_info(
        self, data: bytes | memoryview, packet_size: int, major: int, minor: int
    ) -> int:
        offset = 0
        # Server name
        # szName = data[offset: offset+256]
        self.__application_name, separator, remainder = bytes(
            data[offset : offset + 256]
        ).partition(b"\0")
        self.__application_name = str(self.__application_name, "utf-8")
        offset += 256
        # Server Version info
        server_version = struct.unpack("BBBB", data[offset : offset + 4])
        offset += 4
        self.__server_version[0] = server_version[0]
        self.__server_version[1] = server_version[1]
        self.__server_version[2] = server_version[2]
        self.__server_version[3] = server_version[3]

        # NatNet Version info
        nnsvs = struct.unpack("BBBB", data[offset : offset + 4])
        offset += 4
        self.__nat_net_stream_version_server[0] = nnsvs[0]
        self.__nat_net_stream_version_server[1] = nnsvs[1]
        self.__nat_net_stream_version_server[2] = nnsvs[2]
        self.__nat_net_stream_version_server[3] = nnsvs[3]
        if (self.__nat_net_requested_version[0] == 0) and (
            self.__nat_net_requested_version[1] == 0
        ):
            print(
                f"resetting requested version to {self.__nat_net_stream_version_server[0]} {self.__nat_net_stream_version_server[1]} {self.__nat_net_stream_version_server[2]} {self.__nat_net_stream_version_server[3]} from {self.__nat_net_requested_version[0]} {self.__nat_net_requested_version[1]} {self.__nat_net_requested_version[2]} {self.__nat_net_requested_version[3]}"
            )

            self.__nat_net_requested_version[0] = self.__nat_net_stream_version_server[0]
            self.__nat_net_requested_version[1] = self.__nat_net_stream_version_server[1]
            self.__nat_net_requested_version[2] = self.__nat_net_stream_version_server[2]
            self.__nat_net_requested_version[3] = self.__nat_net_stream_version_server[3]
            # Determine if the bitstream version can be changed
            if (self.__nat_net_stream_version_server[0] >= 4) and (self.use_multicast is False):
                self.__can_change_bitstream_version = True

        trace_mf(f"Sending Application Name: {self.__application_name}")
        trace_mf(
            f"NatNetVersion {self.__nat_net_stream_version_server[0]} {self.__nat_net_stream_version_server[1]} {self.__nat_net_stream_version_server[2]} {self.__nat_net_stream_version_server[3]}",
        )

        trace_mf(
            f"ServerVersion {self.__server_version[0]} {self.__server_version[1]} {self.__server_version[2]} {self.__server_version[3]}",
        )
        return offset

    # __unpack_bitstream_info is for local use of the client
    # and will update the values for the current bitstream
    # of the server.

    def __unpack_bitstream_info(
        self, data: bytes, packet_size: int, major: int, minor: int
    ) -> list[str]:
        nn_version = []
        inString = data.decode("utf-8")
        messageList = inString.split(",")
        if len(messageList) > 1:
            if messageList[0] == "Bitstream":
                nn_version = messageList[1].split(".")
        return nn_version

    def __command_thread_function(
        self,
        in_socket: socket.socket,
        stop: Callable[[], bool],
        gprint_level: Callable[[], int],
        thread_option: str,
    ) -> int:
        message_id_dict = {}
        if not self.use_multicast:
            in_socket.settimeout(2.0)
        # 64k buffer size
        recv_buffer_size = 128 * 1024
        buffer_list_size = 4
        buffer_list = [b"" for _ in range(buffer_list_size)]
        buffer_list_recv_index = 0
        buffer_list_in_use_index = 0
        while not stop():
            # Block for input
            try:
                buffer_list[buffer_list_recv_index], addr = in_socket.recvfrom(recv_buffer_size)
                buffer_list_in_use_index = buffer_list_recv_index
                buffer_list_recv_index = (buffer_list_recv_index + 1) % buffer_list_size
            except socket.herror:
                print("ERROR: command socket access herror occurred")
                return 2
            except socket.gaierror:
                print("ERROR: command socket access gaierror occurred")
                return 3
            except TimeoutError:
                if self.use_multicast:
                    print("ERROR: command socket access timeout occurred. Server not responding")
                    # return 4
            except OSError:
                if stop():
                    # print("ERROR: command socket access error occurred:\n  %s" %msg)
                    # return 1
                    # print("shutting down")
                    pass

            if len(buffer_list[buffer_list_in_use_index]) > 0:
                # peek ahead at message_id
                message_id = get_message_id(buffer_list[buffer_list_in_use_index])
                tmp_str = f"mi_{message_id:1d}"
                if tmp_str not in message_id_dict:
                    message_id_dict[tmp_str] = 0
                message_id_dict[tmp_str] += 1
                print_level = gprint_level()
                if message_id == self.NAT_FRAMEOFDATA:
                    if print_level > 0:
                        if (message_id_dict[tmp_str] % print_level) == 0:
                            print_level = 1
                        else:
                            print_level = 0
                message_id = self.__process_message(
                    buffer_list[buffer_list_in_use_index], print_level
                )
                buffer_list[buffer_list_in_use_index] = b""
            if not self.use_multicast:
                if not stop():
                    # provides option for users to use prompting
                    if thread_option == "c":
                        time.sleep(1)
                    self.send_keep_alive(in_socket, self.server_ip_address, self.command_port)
        return 0

    def __data_thread_function(
        self, in_socket: socket.socket, stop: Callable[[], bool], gprint_level: Callable[[], int]
    ) -> int:
        message_id_dict = {}
        data = b""
        # 64k buffer size
        recv_buffer_size = 128 * 1024
        while not stop():
            # Block for input
            try:
                data, addr = in_socket.recvfrom(recv_buffer_size)
            except socket.herror:
                print("ERROR: data socket access herror occurred")
                # return 2
            except socket.gaierror:
                print("ERROR: data socket access gaierror occurred")
                # return 3
            except TimeoutError:
                # if self.use_multicast:
                # print("ERROR: data socket access timeout occurred. Server not responding")
                # return 4
                # periodic wake up to check for shutting down
                continue
            except OSError as msg:
                if not stop():
                    print(f"ERROR: data socket access error occurred:\n  {msg}")
                    return 1
            if len(data) > 0:
                # peek ahead at message_id
                message_id = get_message_id(data)
                tmp_str = f"mi_{message_id:1d}"
                if tmp_str not in message_id_dict:
                    message_id_dict[tmp_str] = 0
                message_id_dict[tmp_str] += 1
                print_level = gprint_level()
                if message_id == self.NAT_FRAMEOFDATA:
                    if print_level > 0:
                        if (message_id_dict[tmp_str] % print_level) == 0:
                            print_level = 1
                        else:
                            print_level = 0
                message_id = self.__process_message(data, print_level)
                data = b""

        return 0

    def __process_message(self, data: bytes, print_level: int = 0) -> int:
        # return message ID
        major = self.get_major()
        minor = self.get_minor()

        trace("Begin Packet\n-----------------")
        show_nat_net_version = False
        if show_nat_net_version:
            trace(
                f"NatNetVersion {self.__nat_net_requested_version[0]} {self.__nat_net_requested_version[1]} {self.__nat_net_requested_version[2]} {self.__nat_net_requested_version[3]}",
            )

        message_id = get_message_id(data)

        packet_size = int.from_bytes(data[2:4], byteorder="little", signed=True)

        # skip the 4 bytes for message ID and packet_size
        offset = 4
        if message_id == self.NAT_FRAMEOFDATA:
            trace(f"Message ID : {message_id:3d} NAT_FRAMEOFDATA")
            trace(f"Packet Size: {packet_size}")

            offset_tmp, mocap_data = self.__unpack_mocap_data(
                data[offset:], packet_size, major, minor
            )
            offset += offset_tmp
            try:
                with self._latest_mocap_lock:
                    self._latest_mocap_data = mocap_data
                self._latest_mocap_event.set()
            except Exception:
                pass
            # get a string version of the data for output
            if print_level >= 1:
                mocap_data_str = mocap_data.get_as_string()
                print(f" {mocap_data_str}\n")

        elif message_id == self.NAT_MODELDEF:
            trace(f"Message ID : {message_id:3d} NAT_MODELDEF")
            trace(f"Packet Size: {packet_size}")
            offset_tmp, data_descs = self.__unpack_data_descriptions(
                data[offset:], packet_size, major, minor
            )
            offset += offset_tmp
            print("Data Descriptions:\n")
            # get a string version of the data for output
            data_descs_str = data_descs.get_as_string()
            if print_level > 0:
                print(f"{data_descs_str}\n")
            # Call the data description callback if set
            if self.data_description_listener is not None:
                self.data_description_listener(data_descs)

        elif message_id == self.NAT_SERVERINFO:
            trace(f"Message ID : {message_id:3d} NAT_SERVERINFO")
            trace(f"Packet Size: {packet_size}")
            offset += self.__unpack_server_info(data[offset:], packet_size, major, minor)

        elif message_id == self.NAT_RESPONSE:
            trace(f"Message ID : {message_id:3d} NAT_RESPONSE")
            trace(f"Packet Size: {packet_size}")
            if packet_size == 4:
                command_response = int.from_bytes(
                    data[offset : offset + 4], byteorder="little", signed=True
                )
                trace(
                    f"Command response: {command_response} - {data[offset]} {data[offset + 1]} {data[offset + 2]} {data[offset + 3]}"
                )
                offset += 4
            else:
                show_remainder = False
                message, separator, remainder = bytes(data[offset:]).partition(b"\0")
                if len(message) < 30:
                    tmpString = message.decode("utf-8")
                    # Decode bitstream version
                    if tmpString.startswith("Bitstream"):
                        nn_version = self.__unpack_bitstream_info(
                            data[offset:], packet_size, major, minor
                        )
                        # This is the current server version
                        if len(nn_version) > 1:
                            for i in range(len(nn_version)):
                                self.__nat_net_stream_version_server[i] = int(nn_version[i])
                            for i in range(len(nn_version), 4):
                                self.__nat_net_stream_version_server[i] = 0

                offset += len(message) + 1

                if show_remainder:
                    trace(
                        f"Command response: {message.decode('utf-8')} separator: {separator} remainder: {remainder}",
                    )
                else:
                    trace(f"Command response: {message.decode('utf-8')}")
        elif message_id == self.NAT_UNRECOGNIZED_REQUEST:
            trace(f"Message ID : {message_id:3d} NAT_UNRECOGNIZED_REQUEST: ")
            trace(f"Packet Size: {packet_size}")
            trace("Received 'Unrecognized request' from server")
        elif message_id == self.NAT_MESSAGESTRING:
            trace(f"Message ID : {message_id:3d} NAT_MESSAGESTRING")
            trace(f"Packet Size: {packet_size}")
            message, separator, remainder = bytes(data[offset:]).partition(b"\0")
            offset += len(message) + 1
            trace(f"Received message from server: {message.decode('utf-8')}")
        else:
            trace(f"Message ID : {message_id:3d} UNKNOWN")
            trace("Packet Size: ", packet_size)
            trace("ERROR: Unrecognized packet type")

        trace("End Packet\n-----------------")
        return message_id

    def send_request(
        self,
        in_socket: socket.socket,
        command: int,
        command_str: str | list[int],
        address: str | tuple[str, int],
    ) -> int:
        # Compose the message in our known message format
        packet_size = 0
        if command == self.NAT_REQUEST_MODELDEF or command == self.NAT_REQUEST_FRAMEOFDATA:
            packet_size = 0
            command_str = ""
        elif command == self.NAT_REQUEST:
            packet_size = len(command_str) + 1
        elif command == self.NAT_CONNECT:
            tmp_version = [4, 2, 0, 0]
            print(
                f"NAT_CONNECT to Motive with {tmp_version[0]} {tmp_version[1]} {tmp_version[2]} {tmp_version[3]}\n"
            )

            # allocate a byte array for 270 bytes
            # to connect with a specific version
            # The first 4 bytes spell out "Ping"

            command_str = []
            command_str = [0 for i in range(270)]
            command_str[0] = 80
            command_str[1] = 105
            command_str[2] = 110
            command_str[3] = 103
            command_str[264] = 0
            command_str[265] = tmp_version[0]
            command_str[266] = tmp_version[1]
            command_str[267] = tmp_version[2]
            command_str[268] = tmp_version[3]
            packet_size = len(command_str) + 1
        elif command == self.NAT_KEEPALIVE:
            packet_size = 0
            command_str = ""

        data = command.to_bytes(2, byteorder="little", signed=True)
        data += packet_size.to_bytes(2, byteorder="little", signed=True)

        if type(command_str) is list:
            data += bytearray(command_str)
        elif type(command_str) is str:
            data += command_str.encode("utf-8")
        data += b"\0"

        return in_socket.sendto(data, address)

    def send_command(self, command_str: str | list[int]) -> int:
        # print("Send command %s" %command_str)
        nTries = 3
        ret_val = -1
        if self.command_socket is None:
            return -1
        else:
            while nTries:
                nTries -= 1
                ret_val = self.send_request(
                    self.command_socket,
                    self.NAT_REQUEST,
                    command_str,
                    (self.server_ip_address, self.command_port),
                )
                if ret_val != -1:
                    break
        return ret_val

        # return self.send_request(self.data_socket, self.NAT_REQUEST, command_str, (self.server_ip_address, self.command_port)) #type: ignore

    def send_commands(
        self, tmpCommands: list[str] | list[list[int]], print_results: bool = True
    ) -> None:
        for sz_command in tmpCommands:
            return_code = self.send_command(sz_command)
            if print_results:
                print(f"Command: {sz_command} - return_code: {return_code}")

    def send_keep_alive(
        self, in_socket: socket.socket, server_ip_address: str, server_port: int
    ) -> int:
        return self.send_request(
            in_socket, self.NAT_KEEPALIVE, "", (server_ip_address, server_port)
        )

    def get_command_port(self) -> int:
        return self.command_port

    def refresh_configuration(self) -> None:
        # query for application configuration
        # print("Request current configuration")
        sz_command = "Bitstream"
        return_code = self.send_command(sz_command)  # noqa F841
        time.sleep(0.5)

    def get_application_name(self) -> str | bytes:
        return self.__application_name

    def get_nat_net_requested_version(self) -> list[int]:
        return self.__nat_net_requested_version

    def get_nat_net_version_server(self) -> list[int]:
        return self.__nat_net_stream_version_server

    def get_server_version(self) -> list[int]:
        return self.__server_version

    def run(self, thread_option: str = "d") -> bool:
        # Create the data socket
        self.data_socket = self.__create_data_socket()
        if self.data_socket is None:
            print("Could not open data channel")
            return False
        self.data_socket.settimeout(1.0)

        # Create the command socket
        self.command_socket = self.__create_command_socket()
        if self.command_socket is None:
            print("Could not open command channel")
            return False
        self.__is_locked = True

        self.stop_threads = False

        # Create a separate thread for receiving data packets
        self.data_thread = Thread(
            target=self.__data_thread_function,
            args=(
                self.data_socket,
                lambda: self.stop_threads,
                lambda: self.print_level,
            ),
        )
        self.command_thread = Thread(
            target=self.__command_thread_function,
            args=(
                self.command_socket,
                lambda: self.stop_threads,
                lambda: self.print_level,
                thread_option,
            ),
        )
        if thread_option == "d":
            print("starting data thread")
            self.command_thread.start()
            if self.command_thread.is_alive():
                self.data_thread.start()

        # Create a separate thread for receiving command packets
        if thread_option == "c":
            self.command_thread.start()

        # Required for setup
        # Get NatNet and server versions
        self.send_request(
            self.command_socket, self.NAT_CONNECT, "", (self.server_ip_address, self.command_port)
        )

        # Example Commands
        # Get NatNet and server versions
        # self.send_request(self.command_socket, self.NAT_CONNECT, "", (self.server_ip_address, self.command_port)) #type: ignore
        # Request the model definitions
        # self.send_request(self.command_socket, self.NAT_REQUEST_MODELDEF, "", (self.server_ip_address, self.command_port)) #type: ignore
        return True

    def shutdown(self) -> None:
        print("optitrack socket/threads shutdown called")
        self.stop_threads = True
        # closing sockets causes blocking recvfrom to throw
        # an exception and break the loop
        if self.command_socket is not None:
            self.command_socket.close()
        if self.data_socket is not None:
            self.data_socket.close()
        # attempt to join the threads back.
        if self.command_thread is not None and self.command_thread.is_alive():
            self.command_thread.join()
        if self.data_thread is not None and self.data_thread.is_alive():
            self.data_thread.join()

    def get_frame(self) -> dict[int, list[list[float]]]:
        """
        Format: id_num:[[px,py,pz],[rx,ry,rz,rw]],
        """
        # get frame from latest mocap data
        self._latest_mocap_event.wait()
        with self._latest_mocap_lock:
            mocap_data = self._latest_mocap_data
        mocap_data = copy.deepcopy(mocap_data)

        if mocap_data.prefix_data is not None:
            self.latest_frame_number = mocap_data.prefix_data.frame_number
        frame: dict[int, list[list[float]]] = {}

        if mocap_data.skeleton_data is not None:
            for skeleton in mocap_data.skeleton_data.skeleton_list:
                for rb in skeleton.rigid_body_list:
                    assert isinstance(rb, MoCapData.RigidBody)
                    frame[rb.id_num] = [list(rb.pos), list(rb.rot)]

        if mocap_data.rigid_body_data is not None:
            for rb in mocap_data.rigid_body_data.rigid_body_list:
                frame[rb.id_num] = [list(rb.pos), list(rb.rot)]

        return frame

    def get_frame_number(self) -> int:
        return self.latest_frame_number


def setup_optitrack(server_address: str, client_address: str, use_multicast: bool) -> NatNetClient:
    client = NatNetClient()
    client.set_client_address(client_address)
    client.set_server_address(server_address)
    client.set_use_multicast(use_multicast)
    # # set print level to 0
    client.set_print_level(0)

    return client
