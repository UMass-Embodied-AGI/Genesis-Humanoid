from typing import Any, Literal

import numpy as np
import torch
from genesis.utils import geom as gu


@torch.jit.script
def quat_mul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Quaternion multiplication for batched tensors [w, x, y, z]."""
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)

    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


@torch.jit.script
def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    Converts quaternion(s) to rotation matrix (3x3).
    Input:
        q: Tensor of shape [..., 4] where the quaternion is in (w, x, y, z) format.
    Output:
        Rotation matrix of shape [..., 3, 3]
    """
    assert q.shape[-1] == 4, "Quaternion must be of shape [..., 4]"

    w, x, y, z = q.unbind(-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    rot = torch.stack(
        [
            torch.stack([ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
            torch.stack([2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)], dim=-1),
            torch.stack([2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz], dim=-1),
        ],
        dim=-2,
    )  # shape: [..., 3, 3]

    return rot


def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """R: (...,3,3) -> q: (...,4) wxyz. Assumes R is a proper rotation."""
    # Robust conversion (branch on diagonal dominance)
    t = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

    q = torch.empty(R.shape[:-2] + (4,), dtype=R.dtype, device=R.device)

    cond = t > 0.0
    # t > 0
    if cond.any():
        tr = t[cond]
        s = torch.sqrt(tr + 1.0) * 2.0
        q[cond, 0] = 0.25 * s
        q[cond, 1] = (R[cond, 2, 1] - R[cond, 1, 2]) / s
        q[cond, 2] = (R[cond, 0, 2] - R[cond, 2, 0]) / s
        q[cond, 3] = (R[cond, 1, 0] - R[cond, 0, 1]) / s

    # t <= 0: find major diagonal
    cond0 = (~cond) & (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2])
    cond1 = (~cond) & (~cond0) & (R[..., 1, 1] > R[..., 2, 2])
    cond2 = (~cond) & (~cond0) & (~cond1)

    if cond0.any():
        s = torch.sqrt(1.0 + R[cond0, 0, 0] - R[cond0, 1, 1] - R[cond0, 2, 2]) * 2.0
        q[cond0, 0] = (R[cond0, 2, 1] - R[cond0, 1, 2]) / s
        q[cond0, 1] = 0.25 * s
        q[cond0, 2] = (R[cond0, 0, 1] + R[cond0, 1, 0]) / s
        q[cond0, 3] = (R[cond0, 0, 2] + R[cond0, 2, 0]) / s

    if cond1.any():
        s = torch.sqrt(1.0 + R[cond1, 1, 1] - R[cond1, 0, 0] - R[cond1, 2, 2]) * 2.0
        q[cond1, 0] = (R[cond1, 0, 2] - R[cond1, 2, 0]) / s
        q[cond1, 1] = (R[cond1, 0, 1] + R[cond1, 1, 0]) / s
        q[cond1, 2] = 0.25 * s
        q[cond1, 3] = (R[cond1, 1, 2] + R[cond1, 2, 1]) / s

    if cond2.any():
        s = torch.sqrt(1.0 + R[cond2, 2, 2] - R[cond2, 0, 0] - R[cond2, 1, 1]) * 2.0
        q[cond2, 0] = (R[cond2, 1, 0] - R[cond2, 0, 1]) / s
        q[cond2, 1] = (R[cond2, 0, 2] + R[cond2, 2, 0]) / s
        q[cond2, 2] = (R[cond2, 1, 2] + R[cond2, 2, 1]) / s
        q[cond2, 3] = 0.25 * s

    # normalize
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    return q


@torch.jit.script
def rotmat_to_rotation_6D(rot: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrix/matrices to the 6D rotation representation.
    Input:
        rot: Tensor of shape [..., 3, 3] rotation matrix.
    Output:
        Rotation 6D tensor of shape [..., 6], formed by concatenating the first two columns.
    """
    return torch.cat([rot[..., :, 0], rot[..., :, 1]], dim=-1)


@torch.jit.script
def quat_to_rotation_6D(q: torch.Tensor) -> torch.Tensor:
    """
    Converts quaternions to rotation 6D representation.
    Input:
        q: Tensor of shape [..., 4] where the quaternion is in (w, x, y, z) format.
    Output:
        Rotation 6D representation of shape [..., 6].
    """
    return rotmat_to_rotation_6D(quat_to_rotmat(q))


@torch.jit.script
def quat_to_euler(q: torch.Tensor) -> torch.Tensor:  # xyz
    """Convert quaternions to Euler angles (roll, pitch, yaw).
    Args:
        quat: Tensor of shape (N, 4), where each row is a quaternion in (w, x, y, z) format.
    Returns:
        Tensor of shape (N, 3), where each row is (roll, pitch, yaw) in radians.
    """
    assert q.shape[-1] == 4, "Quaternion must be of shape [..., 4]"

    qw, qx, qy, qz = q.unbind(-1)

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * torch.tensor(torch.pi / 2),
        torch.asin(sinp),
    )

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)


@torch.jit.script
def quat_to_angle_axis(quat: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Convert rotations given as quaternions to axis/angle.

    Args:
        quat: The quaternion orientation in (w, x, y, z). Shape is (..., 4).
        eps: The tolerance for Taylor approximation. Defaults to 1.0e-6.

    Returns:
        Rotations given as a vector in axis angle form. Shape is (..., 3).
        The vector's magnitude is the angle turned anti-clockwise in radians around the vector's direction.

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L526-L554
    """
    # Modified to take in quat as [q_w, q_x, q_y, q_z]
    # Quaternion is [q_w, q_x, q_y, q_z] = [cos(theta/2), n_x * sin(theta/2), n_y * sin(theta/2), n_z * sin(theta/2)]
    # Axis-angle is [a_x, a_y, a_z] = [theta * n_x, theta * n_y, theta * n_z]
    # Thus, axis-angle is [q_x, q_y, q_z] / (sin(theta/2) / theta)
    # When theta = 0, (sin(theta/2) / theta) is undefined
    # However, as theta --> 0, we can use the Taylor approximation 1/2 - theta^2 / 48
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
    mag = torch.linalg.norm(quat[..., 1:], dim=-1)
    half_angle = torch.atan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    # check whether to apply Taylor approximation
    sin_half_angles_over_angles = torch.where(
        angle.abs() > eps, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48
    )
    return quat[..., 1:4] / sin_half_angles_over_angles.unsqueeze(-1)


@torch.jit.script
def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Normalizes a given input tensor to unit length.

    Args:
        x: Input tensor of shape (N, dims).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Returns:
        Normalized tensor of shape (N, dims).
    """
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as angle-axis to quaternions.

    Args:
        angle: The angle turned anti-clockwise in radians around the vector's direction. Shape is (N,).
        axis: The axis of rotation. Shape is (N, 3).

    Returns:
        The quaternion in (w, x, y, z). Shape is (N, 4).
    """
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return normalize(torch.cat([w, xyz], dim=-1))


@torch.jit.script
def quat_from_euler(euler: torch.Tensor) -> torch.Tensor:
    """Convert Euler angles to quaternions in (w, x, y, z) format.

    Args:
        euler: Tensor of shape (N, 3), where each row is (roll, pitch, yaw) in radians.

    Returns:
        Quaternion tensor of shape (N, 4) in (w, x, y, z) format.
    """
    half_euler = euler * 0.5
    c = half_euler.cos()
    s = half_euler.sin()

    cr, cp, cy = c.unbind(-1)
    sr, sp, sy = s.unbind(-1)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    quat = torch.stack([w, x, y, z], dim=-1)
    return normalize(quat)


@torch.jit.script
def apply_delta_pose(
    source_pos: torch.Tensor,
    source_rot: torch.Tensor,
    delta_pose: torch.Tensor,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies delta pose transformation on source pose.

    The first three elements of `delta_pose` are interpreted as cartesian position displacement.
    The remaining three elements of `delta_pose` are interpreted as orientation displacement
    in the angle-axis format.

    Args:
        source_pos: Position of source frame. Shape is (N, 3).
        source_rot: Quaternion orientation of source frame in (w, x, y, z). Shape is (N, 4)..
        delta_pose: Position and orientation displacements. Shape is (N, 6).
        eps: The tolerance to consider orientation displacement as zero. Defaults to 1.0e-6.

    Returns:
        A tuple containing the displaced position and orientation frames.
        Shape of the tensors are (N, 3) and (N, 4) respectively.
    """
    # number of poses given
    num_poses = source_pos.shape[0]
    device = source_pos.device

    # interpret delta_pose[:, 0:3] as target position displacements
    target_pos = source_pos + delta_pose[:, 0:3]
    # interpret delta_pose[:, 3:6] as target rotation displacements
    rot_actions = delta_pose[:, 3:6]
    angle = torch.linalg.vector_norm(rot_actions, dim=1)
    axis = rot_actions / angle.unsqueeze(-1)
    # change from axis-angle to quat convention
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(num_poses, 1)
    rot_delta_quat = torch.where(
        angle.unsqueeze(-1).repeat(1, 4) > eps, quat_from_angle_axis(angle, axis), identity_quat
    )
    # TODO: Check if this is the correct order for this multiplication.
    target_rot = quat_mul(rot_delta_quat, source_rot)

    return target_pos, target_rot


@torch.jit.script
def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply a quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    # store shape
    shape = vec.shape
    # reshape to (N, 3) for multiplication
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    # extract components from quaternions
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Computes the conjugate of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        The conjugate quaternion in (w, x, y, z). Shape is (..., 4).
    """
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((q[:, 0:1], -q[:, 1:]), dim=-1).view(shape)


@torch.jit.script
def quat_inv(q: torch.Tensor) -> torch.Tensor:
    """Compute the inverse of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (N, 4).

    Returns:
        The inverse quaternion in (w, x, y, z). Shape is (N, 4).
    """
    return normalize(quat_conjugate(q))


@torch.jit.script
def quat_diff(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Compute the difference between two quaternions.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (..., 4).
        q2: The second quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        The difference between the two quaternions. Shape is (..., 4).
    """
    dq = quat_mul(q2, quat_conjugate(q1))
    return dq


@torch.jit.script
def quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Computes the rotation difference between two quaternions.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (..., 4).
        q2: The second quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        Angular error between input quaternions in radians.
    """
    quat_diff = quat_mul(q1, quat_conjugate(q2))
    return torch.norm(quat_to_angle_axis(quat_diff), dim=-1)


@torch.jit.script
def slerp(q1: torch.Tensor, q2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Spherical linear interpolation between two quaternions.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (..., 4).
        q2: The second quaternion in (w, x, y, z). Shape is (..., 4).
        t: The interpolation factor. Shape is (...,).

    Returns:
        The interpolated quaternion in (w, x, y, z). Shape is (..., 4).
    """
    assert len(t.shape) == len(q1.shape) - 1, "Shape of t must be (...,)"
    cos_half_theta = torch.sum(q1 * q2, dim=-1)

    neg_mask = cos_half_theta < 0
    q2 = torch.where(neg_mask.unsqueeze(-1), -q2, q2)

    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

    t = t.unsqueeze(-1)
    ratioA = torch.sin((1 - t) * half_theta) / sin_half_theta
    ratioB = torch.sin(t * half_theta) / sin_half_theta

    new_q = ratioA * q1 + ratioB * q2

    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q1 + 0.5 * q2, new_q)
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q1, new_q)

    return new_q


@torch.jit.script
def skew_symmetric_matrix(vec: torch.Tensor) -> torch.Tensor:
    """Computes the skew-symmetric matrix of a vector.

    Args:
        vec: The input vector. Shape is (3,) or (N, 3).

    Returns:
        The skew-symmetric matrix. Shape is (1, 3, 3) or (N, 3, 3).

    Raises:
        ValueError: If input tensor is not of shape (..., 3).
    """
    # check input is correct
    if vec.shape[-1] != 3:
        raise ValueError(f"Expected input vector shape mismatch: {vec.shape} != (..., 3).")
    # unsqueeze the last dimension
    if vec.ndim == 1:
        vec = vec.unsqueeze(0)
    # create a skew-symmetric matrix
    skew_sym_mat = torch.zeros(vec.shape[0], 3, 3, device=vec.device, dtype=vec.dtype)
    skew_sym_mat[:, 0, 1] = -vec[:, 2]
    skew_sym_mat[:, 0, 2] = vec[:, 1]
    skew_sym_mat[:, 1, 2] = -vec[:, 0]
    skew_sym_mat[:, 1, 0] = vec[:, 2]
    skew_sym_mat[:, 2, 0] = -vec[:, 1]
    skew_sym_mat[:, 2, 1] = vec[:, 0]

    return skew_sym_mat


"""
Transformations
"""


def is_identity_pose(pos: torch.Tensor, rot: torch.Tensor) -> bool:
    """Checks if input poses are identity transforms.

    The function checks if the input position and orientation are close to zero and
    identity respectively using L2-norm. It does NOT check the error in the orientation.

    Args:
        pos: The cartesian position. Shape is (N, 3).
        rot: The quaternion in (w, x, y, z). Shape is (N, 4).

    Returns:
        True if all the input poses result in identity transform. Otherwise, False.
    """
    # create identity transformations
    pos_identity = torch.zeros_like(pos)
    rot_identity = torch.zeros_like(rot)
    rot_identity[..., 0] = 1
    # compare input to identity
    return torch.allclose(pos, pos_identity) and torch.allclose(rot, rot_identity)


@torch.jit.script
def combine_frame_transforms(
    t01: torch.Tensor,
    q01: torch.Tensor,
    t12: torch.Tensor | None = None,
    q12: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Combine transformations between two reference frames into a stationary frame.

    It performs the following transformation operation: :math:`T_{02} = T_{01} \times T_{12}`,
    where :math:`T_{AB}` is the homogeneous transformation matrix from frame A to B.

    Args:
        t01: Position of frame 1 w.r.t. frame 0. Shape is (N, 3).
        q01: Quaternion orientation of frame 1 w.r.t. frame 0 in (w, x, y, z). Shape is (N, 4).
        t12: Position of frame 2 w.r.t. frame 1. Shape is (N, 3).
            Defaults to None, in which case the position is assumed to be zero.
        q12: Quaternion orientation of frame 2 w.r.t. frame 1 in (w, x, y, z). Shape is (N, 4).
            Defaults to None, in which case the orientation is assumed to be identity.

    Returns:
        A tuple containing the position and orientation of frame 2 w.r.t. frame 0.
        Shape of the tensors are (N, 3) and (N, 4) respectively.
    """
    # compute orientation
    if q12 is not None:
        q02 = quat_mul(q01, q12)
    else:
        q02 = q01
    # compute translation
    if t12 is not None:
        t02 = t01 + quat_apply(q01, t12)
    else:
        t02 = t01

    return t02, q02


# @torch.jit.script
def subtract_frame_transforms(
    t01: torch.Tensor,
    q01: torch.Tensor,
    t02: torch.Tensor | None = None,
    q02: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Subtract transformations between two reference frames into a stationary frame.

    It performs the following transformation operation: :math:`T_{12} = T_{01}^{-1} \times T_{02}`,
    where :math:`T_{AB}` is the homogeneous transformation matrix from frame A to B.

    Args:
        t01: Position of frame 1 w.r.t. frame 0. Shape is (N, 3).
        q01: Quaternion orientation of frame 1 w.r.t. frame 0 in (w, x, y, z). Shape is (N, 4).
        t02: Position of frame 2 w.r.t. frame 0. Shape is (N, 3).
            Defaults to None, in which case the position is assumed to be zero.
        q02: Quaternion orientation of frame 2 w.r.t. frame 0 in (w, x, y, z). Shape is (N, 4).
            Defaults to None, in which case the orientation is assumed to be identity.

    Returns:
        A tuple containing the position and orientation of frame 2 w.r.t. frame 1.
        Shape of the tensors are (N, 3) and (N, 4) respectively.
    """
    # compute orientation
    q10 = quat_inv(q01)
    if q02 is not None:
        q12 = quat_mul(q10, q02)
    else:
        q12 = q10
    # compute translation
    if t02 is not None:
        t12 = quat_apply(q10, t02 - t01)
    else:
        t12 = quat_apply(q10, -t01)
    return t12, q12


def compute_pose_error(
    t01: torch.Tensor,
    q01: torch.Tensor,
    t02: torch.Tensor,
    q02: torch.Tensor,
    rot_error_type: Literal["quat", "axis_angle"] = "axis_angle",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the position and orientation error between source and target frames.

    Args:
        t01: Position of source frame. Shape is (N, 3).
        q01: Quaternion orientation of source frame in (w, x, y, z). Shape is (N, 4).
        t02: Position of target frame. Shape is (N, 3).
        q02: Quaternion orientation of target frame in (w, x, y, z). Shape is (N, 4).
        rot_error_type: The rotation error type to return: "quat", "axis_angle".
            Defaults to "axis_angle".

    Returns:
        A tuple containing position and orientation error. Shape of position error is (N, 3).
        Shape of orientation error depends on the value of :attr:`rot_error_type`:

        - If :attr:`rot_error_type` is "quat", the orientation error is returned
          as a quaternion. Shape is (N, 4).
        - If :attr:`rot_error_type` is "axis_angle", the orientation error is
          returned as an axis-angle vector. Shape is (N, 3).

    Raises:
        ValueError: Invalid rotation error type.
    """
    # Compute quaternion error (i.e., difference quaternion)
    # Reference: https://personal.utdallas.edu/~sxb027100/dock/quaternion.html
    # q_current_norm = q_current * q_current_conj
    source_quat_norm = quat_mul(q01, quat_conjugate(q01))[:, 0]
    # q_current_inv = q_current_conj / q_current_norm
    source_quat_inv = quat_conjugate(q01) / source_quat_norm.unsqueeze(-1)
    # q_error = q_target * q_current_inv
    quat_error = quat_mul(q02, source_quat_inv)

    # Compute position error
    pos_error = t02 - t01

    # return error based on specified type
    if rot_error_type == "quat":
        return pos_error, quat_error
    elif rot_error_type == "axis_angle":
        # Convert to axis-angle error
        axis_angle_error = quat_to_angle_axis(quat_error)
        return pos_error, axis_angle_error
    else:
        raise ValueError(
            f"Unsupported orientation error type: {rot_error_type}. Valid: 'quat', 'axis_angle'."
        )


def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to rotation matrices.
    """
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def pose_mul(
    T1: torch.Tensor, R1: torch.Tensor, T2: torch.Tensor, R2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the local transform (T2, R2) to the pose (T1, R1);
    / Apply the global transform (T1, R1) to the pose (T2, R2).
    R = R1 * R2
    T = R1 * T2 + T1
    """
    R_out = R1 @ R2
    T_out = (R1 @ T2.unsqueeze(-1)).squeeze(-1) + T1
    return T_out, R_out


def pose_diff(
    T1: torch.Tensor, R1: torch.Tensor, T2: torch.Tensor, R2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get local transform from (T1, R1) to (T2, R2)
    R = R1^t * R2
    T = R1^t * (T2 - T1)
    """
    R_out = R1.transpose(-1, -2) @ R2
    T_out = (R1.transpose(-1, -2) @ (T2 - T1).unsqueeze(-1)).squeeze(-1)
    return T_out, R_out


def pose_mul_quat(
    T1: torch.Tensor, R1: torch.Tensor, T2: torch.Tensor, R2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the local transform (T2, R2) to the pose (T1, R1);
    / Apply the global transform (T1, R1) to the pose (T2, R2).
    R = R1 * R2
    T = R1 * T2 + T1
    """
    R_out = quat_mul(R1, R2)
    T_out = quat_apply(R1, T2) + T1
    return T_out, R_out


def pose_diff_quat(
    T1: torch.Tensor, R1: torch.Tensor, T2: torch.Tensor, R2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get local transform from (T1, R1) to (T2, R2)
    R = R1^t * R2
    T = R1^t * (T2 - T1)
    """
    R_out = quat_mul(quat_inv(R1), R2)
    T_out = quat_apply(quat_inv(R1), T2 - T1)
    return T_out, R_out


def pose_inv(T: torch.Tensor, R: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Inverse local transform (T, R)
    R_inv = R^t
    T_inv = -R^t * T
    """
    R_out = R.transpose(-1, -2)
    T_out = -(R_out @ T.unsqueeze(-1)).squeeze(-1)
    return T_out, R_out


def np_pose_mul(
    T1: np.typing.NDArray[np.float32],
    R1: np.typing.NDArray[np.float32],
    T2: np.typing.NDArray[np.float32],
    R2: np.typing.NDArray[np.float32],
) -> tuple[np.typing.NDArray[np.float32], np.typing.NDArray[np.float32]]:
    """
    Apply the local transform (T2, R2) to the pose (T1, R1);
    / Apply the global transform (T1, R1) to the pose (T2, R2).
    R = R1 * R2
    T = R1 * T2 + T1
    """
    R_out = np.array(gu.transform_quat_by_quat(R2, R1))
    T_out = np.array(gu.transform_by_quat(T2, R1) + T1)
    return T_out, R_out


def np_pose_diff(
    T1: np.typing.NDArray[np.float32],
    R1: np.typing.NDArray[np.float32],
    T2: np.typing.NDArray[np.float32],
    R2: np.typing.NDArray[np.float32],
) -> tuple[np.typing.NDArray[np.float32], np.typing.NDArray[np.float32]]:
    """
    Get the local transform from (T1, R1) to (T2, R2).
    R = R1^t * R2
    T = R1^t * (T2 - T1)
    """
    R_out = np.array(gu.transform_quat_by_quat(R2, gu.inv_quat(R1)))
    T_out = np.array(gu.transform_by_quat(T2 - T1, gu.inv_quat(R1)))
    return T_out, R_out


def cross_correlation(
    a: np.typing.NDArray[Any], b: np.typing.NDArray[Any], allow_flip: bool = False
) -> float:
    """
    Compute the sub-sample cross-correlation lag between two 1D signals a and b.
    Handles both normal and flipped correlation (max or min peak).
    Returns lag in samples (positive means b lags behind a).
    """
    assert a.ndim == 1 and b.ndim == 1 and len(a) == len(b), (
        "Inputs must be 1D arrays of the same length"
    )
    a = a - np.mean(a)
    b = b - np.mean(b)
    corr = np.correlate(a, b, mode="full")
    lags = np.arange(-len(a) + 1, len(a))
    overlap = len(a) - np.abs(lags)
    corr_unbiased = corr / overlap
    k_max = np.argmax(corr)
    k_min = np.argmin(corr)
    k_peak = k_min if abs(corr[k_min]) > abs(corr[k_max]) and allow_flip else k_max
    lag_int = lags[k_peak]

    # Parabolic interpolation: use corr[k-1], corr[k], corr[k+1]
    if 0 < k_peak < len(corr) - 1:
        c_minus = corr_unbiased[k_peak - 1]
        c_0 = corr_unbiased[k_peak]
        c_plus = corr_unbiased[k_peak + 1]

        denom = -2 * c_0 + c_minus + c_plus
        if denom != 0:
            delta = 0.5 * (c_minus - c_plus) / denom
        else:
            delta = 0.0
    else:
        delta = 0.0

    lag_subsample = lag_int + delta
    return -lag_subsample
