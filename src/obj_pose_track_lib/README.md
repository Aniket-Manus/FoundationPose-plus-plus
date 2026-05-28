# obj_pose_track_lib

This folder contains the refactored building blocks for the object pose tracking pipeline.

## Module layout

- `obj_pose_track_paths.py`
  - Path and data resolution helpers.
  - Finds the active sequence folder.
  - Resolves RGB/depth/mask/mesh/camera intrinsics paths.

- `obj_pose_track_pose_utils.py`
  - Pose math utilities.
  - Converts between 4x4 matrices and 6D pose vectors.
  - Applies image-point-based translation adjustment.
  - Projects 3D points to 2D image coordinates.

- `obj_pose_track_runner.py`
  - Runtime tracking pipeline.
  - Runs FoundationPose registration/tracking.
  - Integrates optional 2D tracker and Kalman filter.
  - Handles visualization and output pose saving.

## Entry point

The CLI entry script remains:

- `src/obj_pose_track.py`

That file parses command-line arguments, resolves runtime paths, and calls `pose_track(...)` from `obj_pose_track_runner.py`.

## High-level flow

1. Resolve paths and camera intrinsics.
2. Load initial mask and frame sequence.
3. Initialize mesh and FoundationPose estimator.
4. Run per-frame tracking loop.
5. Optionally apply 2D tracker + Kalman correction.
6. Save pose sequence and optional visualizations.
