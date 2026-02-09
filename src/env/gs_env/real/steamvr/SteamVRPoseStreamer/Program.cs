using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Valve.VR; // OpenVR

namespace SteamVRPoseStreamer
{
    class Program
    {
        // UDP settings
        static readonly string RemoteIp = "255.255.255.255"; // broadcast
        static readonly int RemotePort = 5005;

        // Target send rate
        static readonly int TargetHz = 120;

        static ulong frameId = 0;
        
        static readonly Dictionary<string, string> trackerSerialToRole = new()
        {
            { "58-A33S00451", "waist" },
            { "58-A33S01984", "left_foot" },
            { "58-A33S04570", "right_foot" },
        };

        static string? GetDeviceSerial(CVRSystem vr, uint deviceIndex)
        {
            if (vr == null) return null;

            var err = ETrackedPropertyError.TrackedProp_Success;

            uint needed = vr.GetStringTrackedDeviceProperty(
                deviceIndex,
                ETrackedDeviceProperty.Prop_SerialNumber_String,
                null,
                0,
                ref err
            );

            if (needed <= 1)
                return null;

            if (err != ETrackedPropertyError.TrackedProp_Success &&
                err != ETrackedPropertyError.TrackedProp_BufferTooSmall)
                return null;

            var buf = new StringBuilder((int)needed);
            vr.GetStringTrackedDeviceProperty(
                deviceIndex,
                ETrackedDeviceProperty.Prop_SerialNumber_String,
                buf,
                needed,
                ref err
            );

            if (err != ETrackedPropertyError.TrackedProp_Success)
                return null;

            return buf.ToString();
        }

        static uint HMDIndex = OpenVR.k_unTrackedDeviceIndexInvalid;
        static uint leftHandIndex = OpenVR.k_unTrackedDeviceIndexInvalid;
        static uint rightHandIndex = OpenVR.k_unTrackedDeviceIndexInvalid;
        static uint waistIndex = OpenVR.k_unTrackedDeviceIndexInvalid;
        static uint leftFootIndex = OpenVR.k_unTrackedDeviceIndexInvalid;
        static uint rightFootIndex = OpenVR.k_unTrackedDeviceIndexInvalid;

        static void GetIndices(CVRSystem vr)
        {
            HMDIndex = OpenVR.k_unTrackedDeviceIndex_Hmd;
            leftHandIndex = vr.GetTrackedDeviceIndexForControllerRole(ETrackedControllerRole.LeftHand);
            rightHandIndex = vr.GetTrackedDeviceIndexForControllerRole(ETrackedControllerRole.RightHand);
            waistIndex = OpenVR.k_unTrackedDeviceIndexInvalid;
            leftFootIndex = OpenVR.k_unTrackedDeviceIndexInvalid;
            rightFootIndex = OpenVR.k_unTrackedDeviceIndexInvalid;

            for (uint i = 0; i < OpenVR.k_unMaxTrackedDeviceCount; i++)
            {
                if (vr.GetTrackedDeviceClass(i) != ETrackedDeviceClass.GenericTracker)
                    continue;

                var serial = GetDeviceSerial(vr, i);
                if (serial == null) continue;

                if (!trackerSerialToRole.TryGetValue(serial, out var role))
                {
                    Console.WriteLine($"[SteamVRPoseStreamer] Unmatched serial number: {serial}");
                    continue;
                }

                switch (role)
                {
                    case "waist":  waistIndex = i; break;
                    case "left_foot": leftFootIndex = i; break;
                    case "right_foot": rightFootIndex = i; break;
                }
            }
        }

        static void Main(string[] args)
        {
            // ---- UDP ----
            using var udp = new UdpClient();
            udp.EnableBroadcast = true;
            var remote = new IPEndPoint(IPAddress.Parse(RemoteIp), RemotePort);

            Console.WriteLine($"[SteamVRPoseStreamer] UDP broadcast to {RemoteIp}:{RemotePort} at ~{TargetHz} Hz");

            // ---- OpenVR Init ----
            EVRInitError initErr = EVRInitError.None;
            var vr = OpenVR.Init(ref initErr, EVRApplicationType.VRApplication_Other);
            if (initErr != EVRInitError.None || vr == null)
            {
                Console.WriteLine($"OpenVR init failed: {initErr}");
                Console.WriteLine("Make sure SteamVR is running and a VR system is available.");
                return;
            }

            Console.WriteLine("[SteamVRPoseStreamer] OpenVR initialized.");

            // ---- Main loop ----
            var poses = new TrackedDevicePose_t[OpenVR.k_unMaxTrackedDeviceCount];
            int sleepMs = Math.Max(1, (int)Math.Round(1000.0 / TargetHz));

            while (true)
            {
                frameId++;

                if (frameId == 1)
                {
                    GetIndices(vr);
                }

                var system = OpenVR.System;
                if (system == null)
                {
                    Console.WriteLine("OpenVR.System is null.");
                    break;
                }
                system.GetDeviceToAbsoluteTrackingPose(
                    ETrackingUniverseOrigin.TrackingUniverseStanding,
                    0f,
                    poses
                );

                bool okHMD = TryGetPose(poses, HMDIndex, out var HMDPose, out var HMDQuat);
                bool okLeftHand = TryGetPose(poses, leftHandIndex, out var leftHandPose, out var leftHandQuat);
                bool okRightHand = TryGetPose(poses, rightHandIndex, out var rightHandPose, out var rightHandQuat);
                bool okWaist = TryGetPose(poses, waistIndex, out var waistPose, out var waistQuat);
                bool okLeftFoot = TryGetPose(poses, leftFootIndex, out var leftFootPose, out var leftFootQuat);
                bool okRightFoot = TryGetPose(poses, rightFootIndex, out var rightFootPose, out var rightFootQuat);

                // bit0 primary, bit1 secondary, bit2 triggerButton, bit3 gripButton, bit4 primary2DAxisClick
                int leftButton = TryGetButtonMask(vr, leftHandIndex);
                int rightButton = TryGetButtonMask(vr, rightHandIndex);

                if (!okHMD) { HMDPose = (0, 0, 0); HMDQuat = (0, 0, 0, 1); }
                if (!okLeftHand) { leftHandPose = (0, 0, 0); leftHandQuat = (0, 0, 0, 1); }
                if (!okRightHand) { rightHandPose = (0, 0, 0); rightHandQuat = (0, 0, 0, 1); }
                if (!okWaist) { waistPose = (0, 0, 0); waistQuat = (0, 0, 0, 1); }
                if (!okLeftFoot) { leftFootPose = (0, 0, 0); leftFootQuat = (0, 0, 0, 1); }
                if (!okRightFoot) { rightFootPose = (0, 0, 0); rightFootQuat = (0, 0, 0, 1); }

                // SteamVR/OpenVR uses a right-handed coordinate system.

                string msg = string.Format(
                    System.Globalization.CultureInfo.InvariantCulture,
                    "FRAME,{0},HMD,{1},{2},{3},{4},{5},{6},{7},LEFTHAND,{8},{9},{10},{11},{12},{13},{14},LEFTBUTTON,{15},RIGHTHAND,{16},{17},{18},{19},{20},{21},{22},RIGHTBUTTON,{23},WAIST,{24},{25},{26},{27},{28},{29},{30},LEFTFOOT,{31},{32},{33},{34},{35},{36},{37},RIGHTFOOT,{38},{39},{40},{41},{42},{43},{44}",
                    frameId,
                    HMDPose.x, HMDPose.y, HMDPose.z, HMDQuat.x, HMDQuat.y, HMDQuat.z, HMDQuat.w,
                    leftHandPose.x, leftHandPose.y, leftHandPose.z, leftHandQuat.x, leftHandQuat.y, leftHandQuat.z, leftHandQuat.w,
                    leftButton,
                    rightHandPose.x, rightHandPose.y, rightHandPose.z, rightHandQuat.x, rightHandQuat.y, rightHandQuat.z, rightHandQuat.w,
                    rightButton,
                    waistPose.x, waistPose.y, waistPose.z, waistQuat.x, waistQuat.y, waistQuat.z, waistQuat.w,
                    leftFootPose.x, leftFootPose.y, leftFootPose.z, leftFootQuat.x, leftFootQuat.y, leftFootQuat.z, leftFootQuat.w,
                    rightFootPose.x, rightFootPose.y, rightFootPose.z, rightFootQuat.x, rightFootQuat.y, rightFootQuat.z, rightFootQuat.w
                );

                if (frameId == 1) {
                    var HMDConnect = (HMDIndex != OpenVR.k_unTrackedDeviceIndexInvalid);
                    var leftHandConnect = (leftHandIndex != OpenVR.k_unTrackedDeviceIndexInvalid);
                    var rightHandConnect = (rightHandIndex != OpenVR.k_unTrackedDeviceIndexInvalid);
                    var waistConnect = (waistIndex != OpenVR.k_unTrackedDeviceIndexInvalid);
                    var leftFootConnect = (leftFootIndex != OpenVR.k_unTrackedDeviceIndexInvalid);
                    var rightFootConnect = (rightFootIndex != OpenVR.k_unTrackedDeviceIndexInvalid);
                    Console.WriteLine( "[SteamVRPoseStreamer] Status:");
                    Console.WriteLine( "                Connection\tPose");
                    Console.WriteLine($"    HMD:        {HMDConnect}\t\t{okHMD}");
                    Console.WriteLine($"    LeftHand:   {leftHandConnect}\t\t{okLeftHand}");
                    Console.WriteLine($"    RightHand:  {rightHandConnect}\t\t{okRightHand}");
                    Console.WriteLine($"    Waist:      {waistConnect}\t\t{okWaist}");
                    Console.WriteLine($"    LeftFoot:   {leftFootConnect}\t\t{okLeftFoot}");
                    Console.WriteLine($"    RightFoot:  {rightFootConnect}\t\t{okRightFoot}");
                }

                byte[] bytes = Encoding.UTF8.GetBytes(msg);
                udp.Send(bytes, bytes.Length, remote);

                Thread.Sleep(sleepMs);
            }

            OpenVR.Shutdown();
        }

        // ---------------- Pose ----------------

        static bool TryGetPose(TrackedDevicePose_t[] poses, uint deviceIndex,
            out (double x, double y, double z) pos,
            out (double x, double y, double z, double w) quat)
        {
            pos = (0, 0, 0);
            quat = (0, 0, 0, 1);

            if (deviceIndex == OpenVR.k_unTrackedDeviceIndexInvalid) return false;
            if (deviceIndex >= poses.Length) return false;

            var p = poses[deviceIndex];
            if (!p.bPoseIsValid) return false;

            var m = p.mDeviceToAbsoluteTracking; // 3x4 matrix

            // position
            double px = m.m3;
            double py = m.m7;
            double pz = m.m11;

            // rotation (3x3)
            double r00 = m.m0, r01 = m.m1, r02 = m.m2;
            double r10 = m.m4, r11 = m.m5, r12 = m.m6;
            double r20 = m.m8, r21 = m.m9, r22 = m.m10;

            var q = RotMatToQuat(r00, r01, r02, r10, r11, r12, r20, r21, r22);

            pos = (px, py, pz);
            quat = q;
            return true;
        }

        // Numerically stable 3x3 rotation matrix -> quaternion (xyzw)
        static (double x, double y, double z, double w) RotMatToQuat(
            double r00, double r01, double r02,
            double r10, double r11, double r12,
            double r20, double r21, double r22)
        {
            double trace = r00 + r11 + r22;
            double qw, qx, qy, qz;

            if (trace > 0)
            {
                double s = Math.Sqrt(trace + 1.0) * 2.0; // s=4*qw
                qw = 0.25 * s;
                qx = (r21 - r12) / s;
                qy = (r02 - r20) / s;
                qz = (r10 - r01) / s;
            }
            else if ((r00 > r11) && (r00 > r22))
            {
                double s = Math.Sqrt(1.0 + r00 - r11 - r22) * 2.0; // s=4*qx
                qw = (r21 - r12) / s;
                qx = 0.25 * s;
                qy = (r01 + r10) / s;
                qz = (r02 + r20) / s;
            }
            else if (r11 > r22)
            {
                double s = Math.Sqrt(1.0 + r11 - r00 - r22) * 2.0; // s=4*qy
                qw = (r02 - r20) / s;
                qx = (r01 + r10) / s;
                qy = 0.25 * s;
                qz = (r12 + r21) / s;
            }
            else
            {
                double s = Math.Sqrt(1.0 + r22 - r00 - r11) * 2.0; // s=4*qz
                qw = (r10 - r01) / s;
                qx = (r02 + r20) / s;
                qy = (r12 + r21) / s;
                qz = 0.25 * s;
            }

            // normalize
            double norm = Math.Sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
            if (norm > 1e-12)
            {
                qx /= norm; qy /= norm; qz /= norm; qw /= norm;
            }
            return (qx, qy, qz, qw);
        }

        // ---------------- Buttons ----------------

        static int TryGetButtonMask(CVRSystem vr, uint deviceIndex)
        {
            if (vr == null) return 0;
            if (deviceIndex == OpenVR.k_unTrackedDeviceIndexInvalid) return 0;

            VRControllerState_t state = new VRControllerState_t();
            uint stateSize = (uint)System.Runtime.InteropServices.Marshal.SizeOf(typeof(VRControllerState_t));
            bool ok = vr.GetControllerState(deviceIndex, ref state, stateSize);
            if (!ok) return 0;

            int mask = 0;

            // bit0: primaryButton    -> Button.A (right) / Button.X (left)
            // bit1: secondaryButton  -> Button.B (right) / Button.Y (left)
            // bit2: triggerButton    -> interpret as trigger "click" (if present)
            // bit3: gripButton       -> grip "click" (if present)
            // bit4: primary2DAxisClick -> joystick click

            bool primary = IsPressed(state, EVRButtonId.k_EButton_A); // often A/X depending on binding
            bool secondary = IsPressed(state, EVRButtonId.k_EButton_ApplicationMenu); // often B/Y or menu
            bool triggerClick = IsPressed(state, EVRButtonId.k_EButton_SteamVR_Trigger);
            bool gripClick = IsPressed(state, EVRButtonId.k_EButton_Grip);
            bool joystickClick = IsPressed(state, EVRButtonId.k_EButton_Axis0); // Axis0 click in many bindings

            if (primary) mask |= (1 << 0);
            if (secondary) mask |= (1 << 1);
            if (triggerClick) mask |= (1 << 2);
            if (gripClick) mask |= (1 << 3);
            if (joystickClick) mask |= (1 << 4);

            return mask;
        }

        static bool IsPressed(VRControllerState_t state, EVRButtonId button)
        {
            ulong mask = 1UL << (int)button;
            return (state.ulButtonPressed & mask) != 0;
        }
    }
}
