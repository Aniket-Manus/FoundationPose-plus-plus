"""Runtime tracking pipeline for FoundationPose++.

This module wires together dataset IO, mesh setup, FoundationPose inference,
optional 2D tracking, optional Kalman smoothing, visualization, and output
serialization.
"""

import os

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import trimesh

from VOT import Cutie, Tracker_2D
from utils.kalman_filter_6d import KalmanFilter6D

from obj_pose_track_lib.obj_pose_track_paths import get_sorted_frame_list
from obj_pose_track_lib.obj_pose_track_pose_utils import (
    adjust_pose_to_image_point,
    get_6d_pose_arr_from_mat,
    get_mat_from_6d_pose_arr,
    get_pose_xy_from_image_point,
)


def pose_track(
    rgb_seq_path: str,
    depth_seq_path: str,
    mesh_path: str,
    init_mask_path: str,
    cam_K: np.ndarray,
    pose_output_path: str,
    mask_visualization_path: str,
    bbox_visualization_path: str,
    pose_visualization_path: str,
    est_refine_iter: int,
    track_refine_iter: int,
    apply_scale: float,
    force_apply_color: bool,
    apply_color,
    kf_measurement_noise_scale: float,
    activate_2d_tracker: bool = False,
    activate_kalman_filter: bool = False,
    show_window: bool = False,
    window_name: str = "FoundationPose Tracking",
    window_wait_ms: int = 1,
    save_visualizations: bool = False,
):
    """Run full-frame 6DoF pose tracking over an RGB-D sequence.

    Args:
        rgb_seq_path: Directory containing RGB frames.
        depth_seq_path: Directory containing aligned depth frames.
        mesh_path: Object mesh path used by FoundationPose.
        init_mask_path: Initial object mask for first-frame registration.
        cam_K: 3x3 camera intrinsic matrix.
        pose_output_path: Output .npy file path for pose sequence.
        mask_visualization_path: Optional output directory for mask visuals.
        bbox_visualization_path: Optional output directory for bbox visuals.
        pose_visualization_path: Optional output directory for pose overlays.
        est_refine_iter: Registration refinement iterations for first frame.
        track_refine_iter: Tracking refinement iterations for subsequent frames.
        apply_scale: Mesh scale factor.
        force_apply_color: Whether to force a color on textureless mesh.
        apply_color: RGB color used when force_apply_color is enabled.
        kf_measurement_noise_scale: Kalman measurement noise multiplier.
        activate_2d_tracker: Enable auxiliary 2D tracker updates.
        activate_kalman_filter: Enable Kalman filter smoothing.
        show_window: Enable live preview window.
        window_name: Live preview window title.
        window_wait_ms: Delay for cv2.waitKey between frames.
        save_visualizations: Whether to save per-frame visualization images.
    """
    init_mask = cv2.imread(init_mask_path, cv2.IMREAD_GRAYSCALE)
    if init_mask is None:
        print(f"Failed to read mask file {init_mask_path}.")
        return
    init_mask = init_mask.astype(bool)

    frame_color_list = get_sorted_frame_list(rgb_seq_path)
    frame_depth_list = get_sorted_frame_list(depth_seq_path)
    if not frame_color_list or not frame_depth_list:
        print("No RGB frames found.")
        return

    init_frame_filename = frame_color_list[0]
    init_frame_path = os.path.join(rgb_seq_path, init_frame_filename)
    init_frame = cv2.imread(init_frame_path)
    if init_frame is None:
        print("Failed to read initial frame.")
        return

    from FoundationPose.estimater import trimesh_add_pure_colored_texture

    mesh_file = os.path.join(mesh_path)
    if not os.path.exists(mesh_file):
        print("Mesh file not found.")
        return

    mesh = trimesh.load(mesh_file)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    mesh.apply_scale(apply_scale)
    if force_apply_color:
        mesh = trimesh_add_pure_colored_texture(mesh, color=np.array(apply_color), resolution=10)

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    from FoundationPose.estimater import (
        FoundationPose,
        PoseRefinePredictor,
        ScorePredictor,
        dr,
        draw_posed_3d_box,
        draw_xyz_axis,
        logging,
    )

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    pts_verified = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    normals_verified = np.ascontiguousarray(mesh.vertex_normals, dtype=np.float32)

    print("\n[BLACKWELL MESH LOG] Stabilizing 3D Tracking Topology Assets...")
    print(f"[BLACKWELL MESH LOG] Extracted Points Array Shape: {pts_verified.shape}")
    print(f"[BLACKWELL MESH LOG] Extracted Normals Array Shape: {normals_verified.shape}")

    if len(pts_verified) <= 1 or pts_verified.ndim != 2:
        raise ValueError("[FATAL] Mesh vertices matrix is empty or malformed. Check your model scale or layout.")

    est = FoundationPose(
        model_pts=pts_verified,
        model_normals=normals_verified,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        glctx=glctx,
    )
    logging.info("Estimator initialization done")

    tracker_2d = Cutie() if activate_2d_tracker else Tracker_2D()

    if activate_kalman_filter:
        kf = KalmanFilter6D(kf_measurement_noise_scale)

    can_show_window = show_window
    if can_show_window:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except cv2.error as e:
            print(f"[WARN] Unable to create display window ({e}). Continue without live preview.")
            can_show_window = False

    total_frames = len(frame_color_list)
    pose_seq = [None] * total_frames
    kf_mean, kf_covariance = None, None

    for i in range(total_frames):
        frame_color_filename = frame_color_list[i]
        frame_depth_filename = frame_depth_list[i]
        color = imageio.imread(os.path.join(rgb_seq_path, frame_color_filename))[..., :3]
        color = cv2.resize(color, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)

        depth = cv2.imread(os.path.join(depth_seq_path, frame_depth_filename), -1) / 1e3
        depth = cv2.resize(depth, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
        depth[(depth < 0.001) | (depth >= np.inf)] = 0

        if color is None or depth is None:
            print(f"Failed to read color frame {frame_color_filename} or depth map {frame_depth_filename}")
            continue

        if i == 0:
            mask = init_mask.astype(np.uint8) * 255
            pose = est.register(K=cam_K, rgb=color, depth=depth, ob_mask=mask, iteration=est_refine_iter)
            if activate_kalman_filter:
                kf_mean, kf_covariance = kf.initiate(get_6d_pose_arr_from_mat(pose))

            mask_vis_file = None
            bbox_vis_file = None
            if save_visualizations:
                if mask_visualization_path is not None:
                    os.makedirs(mask_visualization_path, exist_ok=True)
                    mask_vis_file = os.path.join(mask_visualization_path, frame_color_filename)
                if bbox_visualization_path is not None:
                    os.makedirs(bbox_visualization_path, exist_ok=True)
                    bbox_vis_file = os.path.join(bbox_visualization_path, frame_color_filename)

            if activate_2d_tracker:
                tracker_2d.initialize(
                    color,
                    init_info={"mask": init_mask},
                    mask_visualization_path=mask_vis_file,
                    bbox_visualization_path=bbox_vis_file,
                )
        else:
            mask_vis_file = None
            bbox_vis_file = None
            if save_visualizations:
                if mask_visualization_path is not None:
                    os.makedirs(mask_visualization_path, exist_ok=True)
                    mask_vis_file = os.path.join(mask_visualization_path, frame_color_filename)
                if bbox_visualization_path is not None:
                    os.makedirs(bbox_visualization_path, exist_ok=True)
                    bbox_vis_file = os.path.join(bbox_visualization_path, frame_color_filename)

            if activate_2d_tracker:
                bbox_2d = tracker_2d.track(
                    color,
                    mask_visualization_path=mask_vis_file,
                    bbox_visualization_path=bbox_vis_file,
                )

            if activate_2d_tracker:
                if not activate_kalman_filter:
                    est.pose_last = adjust_pose_to_image_point(
                        ob_in_cam=est.pose_last,
                        K=cam_K,
                        x=bbox_2d[0] + bbox_2d[2] / 2,
                        y=bbox_2d[1] + bbox_2d[3] / 2,
                    )
                else:
                    kf_mean, kf_covariance = kf.update(
                        kf_mean,
                        kf_covariance,
                        get_6d_pose_arr_from_mat(est.pose_last),
                    )
                    measurement_xy = np.array(
                        get_pose_xy_from_image_point(
                            ob_in_cam=est.pose_last,
                            K=cam_K,
                            x=bbox_2d[0] + bbox_2d[2] / 2,
                            y=bbox_2d[1] + bbox_2d[3] / 2,
                        )
                    )
                    kf_mean, kf_covariance = kf.update_from_xy(kf_mean, kf_covariance, measurement_xy)
                    est.pose_last = torch.from_numpy(get_mat_from_6d_pose_arr(kf_mean[:6])).unsqueeze(0).to(est.pose_last.device)

            pose = est.track_one(rgb=color, depth=depth, K=cam_K, iteration=track_refine_iter)
            if activate_2d_tracker and activate_kalman_filter:
                kf_mean, kf_covariance = kf.predict(kf_mean, kf_covariance)

        pose_seq[i] = pose.reshape(4, 4)

        vis_color = None
        if (save_visualizations and pose_visualization_path is not None) or can_show_window:
            center_pose = pose @ np.linalg.inv(to_origin)
            vis_color = draw_posed_3d_box(cam_K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis_color = draw_xyz_axis(
                vis_color,
                ob_in_cam=center_pose,
                scale=0.1,
                K=cam_K,
                thickness=3,
                transparency=0,
                is_input_rgb=True,
            )

            if save_visualizations and pose_visualization_path is not None:
                os.makedirs(pose_visualization_path, exist_ok=True)
                pose_vis_file = os.path.join(pose_visualization_path, frame_color_filename)
                imageio.imwrite(pose_vis_file, vis_color)

        if can_show_window and vis_color is not None:
            try:
                cv2.imshow(window_name, cv2.cvtColor(vis_color, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(max(1, window_wait_ms)) & 0xFF
                if key == ord('q'):
                    print("[INFO] Live preview interrupted by user ('q').")
                    can_show_window = False
                    cv2.destroyWindow(window_name)
            except cv2.error as e:
                print(f"[WARN] Live preview failed ({e}). Continue without live preview.")
                can_show_window = False

    pose_seq_array = np.array(pose_seq)
    np.save(pose_output_path, pose_seq_array)

    if can_show_window:
        cv2.destroyWindow(window_name)

    torch.cuda.empty_cache()
