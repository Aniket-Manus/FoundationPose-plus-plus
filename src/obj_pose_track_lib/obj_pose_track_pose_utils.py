"""Pose math helpers for 6DoF tracking.

Includes conversions between matrix and 6D pose representations, projection
utilities, and translation adjustment from image-space points.
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def adjust_pose_to_image_point(
    ob_in_cam: torch.Tensor,
    K: torch.Tensor,
    x: float = -1.0,
    y: float = -1.0,
) -> torch.Tensor:
    """Adjust translation so the projected object center matches image point (x, y)."""
    device = ob_in_cam.device
    dtype = ob_in_cam.dtype

    is_batched = ob_in_cam.ndim == 3
    if not is_batched:
        ob_in_cam = ob_in_cam.unsqueeze(0)

    batch = ob_in_cam.shape[0]
    ob_in_cam_new = torch.eye(4, device=device, dtype=dtype).repeat(batch, 1, 1)

    for i in range(batch):
        rot = ob_in_cam[i, :3, :3]
        trans = ob_in_cam[i, :3, 3]

        tx, ty = get_pose_xy_from_image_point(ob_in_cam[i], K, x, y)
        trans_new = torch.tensor([tx, ty, trans[2]], device=device, dtype=dtype)

        ob_in_cam_new[i, :3, :3] = rot
        ob_in_cam_new[i, :3, 3] = trans_new

    return ob_in_cam_new if is_batched else ob_in_cam_new[0]


def get_pose_xy_from_image_point(
    ob_in_cam: torch.Tensor,
    K: torch.Tensor,
    x: float = -1.0,
    y: float = -1.0,
) -> tuple:
    """Compute camera-space tx/ty from desired image-space pixel coordinates."""
    is_batched = ob_in_cam.ndim == 3
    if is_batched:
        ob_in_cam_new = ob_in_cam[0].cpu()
    else:
        ob_in_cam_new = ob_in_cam.cpu()

    if x == -1.0 or y == -1.0:
        return x, y

    trans = ob_in_cam_new[:3, 3]

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    tz = trans[2]

    tx = (x - cx) * tz / fx
    ty = (y - cy) * tz / fy

    return tx, ty


def project_3d_to_2d(point_3d_homogeneous, K, ob_in_cam):
    """Project a homogeneous 3D point into image pixel coordinates."""
    point_cam = ob_in_cam @ point_3d_homogeneous

    x = point_cam[0] / point_cam[2]
    y = point_cam[1] / point_cam[2]

    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]

    return int(u), int(v)


def get_mat_from_6d_pose_arr(pose_arr):
    """Convert a 6D pose vector [xyz, euler_xyz] into a 4x4 transform matrix."""
    xyz = pose_arr[:3]
    euler_angles = pose_arr[3:]

    rotation = Rotation.from_euler('xyz', euler_angles, degrees=False)
    rotation_matrix = rotation.as_matrix()

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = xyz

    return transformation_matrix


def get_6d_pose_arr_from_mat(pose):
    """Convert a 4x4 transform matrix (or tensor) into [xyz, euler_xyz]."""
    if torch.is_tensor(pose):
        is_batched = pose.ndim == 3
        if is_batched:
            pose_np = pose[0].cpu().numpy()
        else:
            pose_np = pose.cpu().numpy()
    else:
        pose_np = pose

    xyz = pose_np[:3, 3]
    rotation_matrix = pose_np[:3, :3]
    euler_angles = Rotation.from_matrix(rotation_matrix).as_euler('xyz', degrees=False)
    return np.r_[xyz, euler_angles]
