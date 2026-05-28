"""CLI entrypoint for FoundationPose++ object pose tracking.

This script parses command-line arguments, resolves runtime paths/intrinsics,
and delegates execution to the tracking runner in `obj_pose_track_lib`.
"""

import argparse
import json
import os
import sys

import torch

from obj_pose_track_lib.obj_pose_track_paths import resolve_cam_k, resolve_runtime_paths
from obj_pose_track_lib.obj_pose_track_runner import pose_track


src_path = os.path.join(os.path.dirname(__file__), "..")
foundationpose_path = os.path.join(src_path, "FoundationPose")
if src_path not in sys.path:
    sys.path.append(src_path)
if foundationpose_path not in sys.path:
    sys.path.append(foundationpose_path)


def build_parser():
    """Build and return the argument parser for tracking configuration."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--sequence_root", type=str, default="~/FoundationPose/data/my_custom_object", help="Root containing sequence folders like tes_sequence_yyyymmdd")
    parser.add_argument("--preferred_mesh_name", type=str, default="Mpro_CasingLeft_V007_blender.obj", help="Preferred mesh file name for auto mesh discovery")
    parser.add_argument("--rgb_seq_path", type=str, default=None)
    parser.add_argument("--depth_seq_path", type=str, default=None)
    parser.add_argument("--mesh_path", type=str, default=None)
    parser.add_argument("--init_mask_path", type=str, default=None)
    parser.add_argument("--pose_output_path", type=str, default=None)
    parser.add_argument("--mask_visualization_path", type=str, default=None)
    parser.add_argument("--bbox_visualization_path", type=str, default=None)
    parser.add_argument("--pose_visualization_path", type=str, default=None)
    parser.add_argument("--cam_K", type=json.loads, default=None, help="Camera intrinsic parameters; if omitted, load from cam_K.txt")
    parser.add_argument("--est_refine_iter", type=int, default=10, help="FoundationPose initial refine iterations, see https://github.com/NVlabs/FoundationPose")
    parser.add_argument("--track_refine_iter", type=int, default=5, help="FoundationPose tracking refine iterations, see https://github.com/NVlabs/FoundationPose")
    parser.add_argument("--activate_2d_tracker", action='store_true', help="activate 2d tracker")
    parser.add_argument("--activate_kalman_filter", action='store_true', help="activate kalman_filter")
    parser.add_argument("--kf_measurement_noise_scale", type=float, default=0.05, help="The scale of measurement noise relative to prediction in kalman filter, greater value means more filtering. Only effective if activate_kalman_filter")
    parser.add_argument("--apply_scale", type=float, default=1.0, help="Mesh scale factor in meters (1.0 means no scaling), commonly use 0.01")
    parser.add_argument("--force_apply_color", action='store_true', help="force a color for colorless mesh")
    parser.add_argument("--apply_color", type=json.loads, default="[0, 159, 237]", help="RGB color to apply, in format 'r,g,b'. Only effective if force_apply_color")
    parser.add_argument("--show_window", dest="show_window", action='store_true', help="show realtime tracking visualization window")
    parser.add_argument("--no_show_window", dest="show_window", action='store_false', help="disable realtime tracking visualization window")
    parser.add_argument("--window_name", type=str, default="FoundationPose Tracking", help="window title for live preview")
    parser.add_argument("--window_wait_ms", type=int, default=0, help="delay in milliseconds between displayed frames")
    parser.add_argument("--save_visualizations", action='store_true', help="save tracking visualization images to disk")
    parser.set_defaults(show_window=True)

    return parser


def main():
    """Resolve runtime inputs and execute the end-to-end tracking pipeline."""
    parser = build_parser()
    args = parser.parse_args()

    resolved = resolve_runtime_paths(args)
    cam_k = resolve_cam_k(args, resolved["sequence_dir"], os.path.expanduser(args.sequence_root))

    print(f"[INFO] Using sequence directory: {resolved['sequence_dir']}")
    print(f"[INFO] Using default data directory: {resolved['default_data_dir']}")
    print(f"[INFO] Using mesh path: {resolved['mesh_path']}")

    pose_track(
        resolved["rgb_seq_path"],
        resolved["depth_seq_path"],
        resolved["mesh_path"],
        resolved["init_mask_path"],
        cam_k,
        resolved["pose_output_path"],
        resolved["mask_visualization_path"],
        resolved["bbox_visualization_path"],
        resolved["pose_visualization_path"],
        args.est_refine_iter,
        args.track_refine_iter,
        args.apply_scale,
        args.force_apply_color,
        args.apply_color,
        args.kf_measurement_noise_scale,
        args.activate_2d_tracker,
        args.activate_kalman_filter,
        args.show_window,
        args.window_name,
        args.window_wait_ms,
        args.save_visualizations,
    )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
