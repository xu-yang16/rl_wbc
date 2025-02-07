import torch
import numpy as np


# numpy implementation
def getMatrixFromQuaternion(x, y, z, w):
    # Precompute repeated values
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    # Compute the matrix elements
    matrix = np.array(
        [
            [1 - 2 * (y2 + z2), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (x2 + z2), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (x2 + y2)],
        ]
    )

    return matrix


@torch.jit.script
def quat_to_rot_mat(q):
    n = q.shape[0]

    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    Nq = w * w + x * x + y * y + z * z
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ = y * Y, y * Z
    zZ = z * Z

    rotation_matrix = torch.stack(
        [
            torch.stack([1.0 - (yY + zZ), xY - wZ, xZ + wY], dim=-1),
            torch.stack([xY + wZ, 1.0 - (xX + zZ), yZ - wX], dim=-1),
            torch.stack([xZ - wY, yZ + wX, 1.0 - (xX + yY)], dim=-1),
        ],
        dim=-2,
    )

    return rotation_matrix


@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)


@torch.jit.script
def get_euler_xyz_from_quaternion(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = (
        q[:, qw] * q[:, qw]
        - q[:, qx] * q[:, qx]
        - q[:, qy] * q[:, qy]
        + q[:, qz] * q[:, qz]
    )
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(
        torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp)
    )

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = (
        q[:, qw] * q[:, qw]
        + q[:, qx] * q[:, qx]
        - q[:, qy] * q[:, qy]
        - q[:, qz] * q[:, qz]
    )
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack(
        (roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)), dim=1
    )


@torch.jit.script
def angle_normalize(x):
    return torch.remainder(x + torch.pi, 2 * torch.pi) - torch.pi
