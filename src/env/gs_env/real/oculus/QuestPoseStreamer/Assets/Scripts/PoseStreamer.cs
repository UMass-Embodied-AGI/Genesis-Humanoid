using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;
using System.Net;
using System.Net.Sockets;
using System.Text;

public class PoseStreamer : MonoBehaviour
{
    private InputDevice _headset;
    private InputDevice _leftController;
    private InputDevice _rightController;

    private UdpClient _udpClient;
    private IPEndPoint _remoteEndPoint;

    private ulong _frameId = 0;

    void Start()
    {
        int remotePort = 5005;
        _udpClient = new UdpClient();
        _udpClient.EnableBroadcast = true;
        _remoteEndPoint = new IPEndPoint(IPAddress.Broadcast, remotePort);

        Debug.Log($"[PoseStreamer] Broadcast UDP at port {remotePort}");

        // Find headset
        var headsets = new List<InputDevice>();
        InputDevices.GetDevicesWithCharacteristics(InputDeviceCharacteristics.HeadMounted, headsets);
        if (headsets.Count > 0) _headset = headsets[0];

        // Find controllers
        var controllers = new List<InputDevice>();
        InputDevices.GetDevicesWithCharacteristics(InputDeviceCharacteristics.Controller, controllers);
        if (controllers.Count > 0) _leftController = controllers[0];
        if (controllers.Count > 1) _rightController = controllers[1];
    }

    void Update()
    {
        _frameId++;

        bool okH = TryGetPose(_headset, out Vector3 hp, out Quaternion hq);
        bool okL = TryGetPose(_leftController, out Vector3 lp, out Quaternion lq);
        bool okR = TryGetPose(_rightController, out Vector3 rp, out Quaternion rq);
        int lb = TryGetButtons(_leftController);
        int rb = TryGetButtons(_rightController);

        string msg = string.Format(
            "FRAME,{0},HPOSE,{1},{2},{3},{4},{5},{6},{7},LPOSE,{8},{9},{10},{11},{12},{13},{14},LB,{15},RPOSE,{16},{17},{18},{19},{20},{21},{22},RB,{23}",
            _frameId,
            hp.x, hp.y, hp.z, hq.x, hq.y, hq.z, hq.w,
            lp.x, lp.y, lp.z, lq.x, lq.y, lq.z, lq.w, lb,
            rp.x, rp.y, rp.z, rq.x, rq.y, rq.z, rq.w, rb
        );

        byte[] bytes = Encoding.UTF8.GetBytes(msg);
        _udpClient.Send(bytes, bytes.Length, _remoteEndPoint);
    }

    private bool TryGetPose(InputDevice device, out Vector3 pos, out Quaternion rot)
    {
        pos = Vector3.zero;
        rot = Quaternion.identity;
        if (!device.isValid) return false;

        bool okPos = device.TryGetFeatureValue(CommonUsages.devicePosition, out pos);
        bool okRot = device.TryGetFeatureValue(CommonUsages.deviceRotation, out rot);
        if (!okPos) pos = Vector3.zero;
        if (!okRot) rot = Quaternion.identity;
        return okPos && okRot;
    }

    private int TryGetButtons(InputDevice device)
    {
        if (!device.isValid) return 0;

        int mask = 0;

        bool b;
        if (device.TryGetFeatureValue(CommonUsages.primaryButton, out b) && b) mask |= (1 << 0);
        if (device.TryGetFeatureValue(CommonUsages.secondaryButton, out b) && b) mask |= (1 << 1);
        if (device.TryGetFeatureValue(CommonUsages.triggerButton, out b) && b) mask |= (1 << 2);
        if (device.TryGetFeatureValue(CommonUsages.gripButton, out b) && b) mask |= (1 << 3);
        if (device.TryGetFeatureValue(CommonUsages.primary2DAxisClick, out b) && b) mask |= (1 << 4);

        return mask;
    }

    private void OnDestroy()
    {
        if (_udpClient != null) _udpClient.Close();
        _udpClient = null;
    }
}
