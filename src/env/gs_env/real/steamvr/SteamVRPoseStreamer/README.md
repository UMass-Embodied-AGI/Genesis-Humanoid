# SteamVRPoseStreamer

A lightweight C# console application that streams SteamVR/OpenVR tracked device poses (HMD, controllers, trackers) over UDP at a fixed rate.

## Installation
### Install .NET

Check if .NET is installed:

```
dotnet --version
```

If not installed: https://dotnet.microsoft.com/download

### Configuration

**Edit network constants in Program.cs**:

```cs
static readonly string RemoteIp = "255.255.255.255";
static readonly int RemotePort = 5005;
static readonly int TargetHz = 120;
```

- For UDP broadcast, keep `RemoteIp = "255.255.255.255"`.
- For unicast, set your receiver IP and disable broadcast in Main: `udp.EnableBroadcast = false;`.

**Update serial numbers**:

```cs
static readonly Dictionary<string, string> trackerSerialToRole = new()
{
    { "YOUR_SERIAL_1", "waist" },
    { "YOUR_SERIAL_2", "left_foot" },
    { "YOUR_SERIAL_3", "right_foot" },
};
```

### Compile

```
dotnet build -c Release
```

**Add OpenVR Binding**:

Copy `openvr_api.dll` from your SteamVR installation into the build output directory:

`steam/steamapps/common/SteamVR/bin/win64/` (or similar folder) -> `bin/Release/netX.X/`

### Run

```
./bin/Release/netX.X/SteamVRPoseStreamer
```

## Output Format (UDP)

**Example Packet**:

```
FRAME,123,
HMD,x,y,z,qx,qy,qz,qw,
LEFTHAND,x,y,z,qx,qy,qz,qw,
LEFTBUTTON,mask,
RIGHTHAND,...,
RIGHTBUTTON,mask,
WAIST,...,
LEFTFOOT,...,
RIGHTFOOT,...
```

**Units & Conventions**:

- Meters (position)
- Quaternion (x,y,z,w)
- Right-handed coordinate system

**Button Mask Bits**:

- 0:	Primary button
- 1:	Secondary button
- 2:	Trigger click
- 3:	Grip click
- 4:	Joystick click
