"""Path and dataset resolution helpers for object pose tracking.

This module centralizes sequence discovery and default path selection for RGB,
depth, mesh, masks, and camera intrinsics.
"""

import os
import re
from typing import List

import numpy as np


def get_sorted_frame_list(dir_path: str) -> List[str]:
    """Return numerically sorted image filenames from a directory."""
    files = os.listdir(dir_path)
    if not files:
        return []
    files = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]
    if not files:
        return []
    if files[0].count('.') == 1:
        files.sort(key=lambda x: int(x.split('.')[0]))
    elif files[0].count('.') == 2:
        files.sort(key=lambda x: int(x.split('.')[0] + x.split('.')[1]))
    return files


def find_latest_sequence_dir(sequence_root: str) -> str:
    """Find latest tes/test_sequence folder under sequence_root."""
    if not os.path.isdir(sequence_root):
        raise FileNotFoundError(f"Sequence root not found: {sequence_root}")

    pattern = re.compile(r"^(?:tes|test)_sequence(?:_(\d{8})(?:_(\d{6}))?)?$")
    candidates = []

    for name in os.listdir(sequence_root):
        full = os.path.join(sequence_root, name)
        if not os.path.isdir(full):
            continue
        m = pattern.match(name)
        if m:
            date = m.group(1) or "00000000"
            time = m.group(2) or "000000"
            candidates.append((f"{date}_{time}", full))

    if not candidates:
        return sequence_root

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def choose_mesh_path(mesh_dir: str, preferred_mesh_name: str = None) -> str:
    """Choose a mesh file from a directory, preferring a named mesh when present."""
    if not os.path.isdir(mesh_dir):
        raise FileNotFoundError(f"Mesh directory not found: {mesh_dir}")

    if preferred_mesh_name:
        preferred = os.path.join(mesh_dir, preferred_mesh_name)
        if os.path.isfile(preferred):
            return preferred

    mesh_candidates = []
    for name in sorted(os.listdir(mesh_dir)):
        if name.lower().endswith((".obj", ".stl", ".ply")):
            mesh_candidates.append(os.path.join(mesh_dir, name))

    if not mesh_candidates:
        raise FileNotFoundError(f"No mesh file found in {mesh_dir}")

    return mesh_candidates[0]


def choose_first_existing_path(paths):
    """Return the first existing path from a priority-ordered candidate list."""
    for p in paths:
        if p is not None and os.path.exists(p):
            return p
    return None


def choose_init_mask_path(sequence_dir: str) -> str:
    """Find an initial mask in a sequence directory using common filename patterns."""
    candidate_files = [
        "0_mask.png",
        "mask.png",
        "init_mask.png",
    ]
    for file_name in candidate_files:
        direct = os.path.join(sequence_dir, file_name)
        if os.path.isfile(direct):
            return direct

    masks_dir = os.path.join(sequence_dir, "masks")
    if os.path.isdir(masks_dir):
        mask_files = [f for f in os.listdir(masks_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if mask_files:
            mask_files.sort()
            return os.path.join(masks_dir, mask_files[0])

    raise FileNotFoundError(f"No initial mask found in {sequence_dir}")


def resolve_cam_k(args, sequence_dir: str, sequence_root: str) -> np.ndarray:
    """Resolve camera intrinsics from CLI input or cam_K.txt files."""
    if args.cam_K is not None:
        return np.array(args.cam_K)

    cam_k_path = choose_first_existing_path([
        os.path.join(sequence_dir, "cam_K.txt"),
        os.path.join(sequence_root, "cam_K.txt"),
    ])

    if cam_k_path is None:
        raise FileNotFoundError(
            "cam_K not provided and cam_K.txt not found in selected sequence directory or sequence root."
        )

    cam_k = np.loadtxt(cam_k_path).reshape(3, 3)
    print(f"[INFO] Loaded cam_K from: {cam_k_path}")
    return cam_k


def resolve_runtime_paths(args):
    """Resolve effective runtime input/output paths from CLI arguments and defaults."""
    sequence_root = os.path.expanduser(args.sequence_root)
    sequence_dir = find_latest_sequence_dir(sequence_root)
    default_data_dir = sequence_dir

    rgb_seq_path = args.rgb_seq_path or choose_first_existing_path([
        os.path.join(default_data_dir, "color"),
        os.path.join(default_data_dir, "rgb"),
    ])
    depth_seq_path = args.depth_seq_path or choose_first_existing_path([
        os.path.join(default_data_dir, "depth"),
    ])
    init_mask_path = args.init_mask_path or choose_first_existing_path([
        os.path.join(default_data_dir, "0_mask.png"),
        os.path.join(default_data_dir, "mask.png"),
        os.path.join(default_data_dir, "init_mask.png"),
    ])
    if init_mask_path is None:
        init_mask_path = choose_init_mask_path(default_data_dir)
    pose_output_path = args.pose_output_path or os.path.join(default_data_dir, "pose.npy")

    if rgb_seq_path is None:
        raise FileNotFoundError(f"Could not find RGB folder in {sequence_dir}")
    if depth_seq_path is None:
        raise FileNotFoundError(f"Could not find depth folder in {sequence_dir}")

    mask_visualization_path = args.mask_visualization_path
    bbox_visualization_path = args.bbox_visualization_path
    pose_visualization_path = args.pose_visualization_path
    if args.save_visualizations:
        if mask_visualization_path is None:
            mask_visualization_path = os.path.join(sequence_dir, "mask_visualization")
        if bbox_visualization_path is None:
            bbox_visualization_path = os.path.join(sequence_dir, "bbox_visualization")
        if pose_visualization_path is None:
            pose_visualization_path = os.path.join(sequence_dir, "pose_visualization")

    mesh_path = args.mesh_path
    if mesh_path is None:
        mesh_path = choose_first_existing_path([
            os.path.join(sequence_root, "mesh", args.preferred_mesh_name),
            os.path.join(sequence_root, args.preferred_mesh_name),
            os.path.join(default_data_dir, "mesh", args.preferred_mesh_name),
            os.path.join(default_data_dir, args.preferred_mesh_name),
        ])
        if mesh_path is None and os.path.isdir(os.path.join(sequence_root, "mesh")):
            mesh_path = choose_mesh_path(os.path.join(sequence_root, "mesh"), preferred_mesh_name=args.preferred_mesh_name)
        if mesh_path is None and os.path.isdir(os.path.join(default_data_dir, "mesh")):
            mesh_path = choose_mesh_path(os.path.join(default_data_dir, "mesh"), preferred_mesh_name=args.preferred_mesh_name)
        if mesh_path is None:
            mesh_path = choose_mesh_path(sequence_root, preferred_mesh_name=args.preferred_mesh_name)

    return {
        "sequence_dir": sequence_dir,
        "default_data_dir": default_data_dir,
        "rgb_seq_path": rgb_seq_path,
        "depth_seq_path": depth_seq_path,
        "mesh_path": mesh_path,
        "init_mask_path": init_mask_path,
        "pose_output_path": pose_output_path,
        "mask_visualization_path": mask_visualization_path,
        "bbox_visualization_path": bbox_visualization_path,
        "pose_visualization_path": pose_visualization_path,
    }
