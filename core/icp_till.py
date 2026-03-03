import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import List, Literal, Tuple

import laspy
import numpy as np
from datetime import datetime

def _get_aoi_name(config) -> str:
    return str(getattr(config, "icp_current_aoi_name", "unknown_aoi"))


def _ensure_dirs(config) -> Tuple[str, str, str]:
    aoi_name = _get_aoi_name(config)
    run_name = str(getattr(config, "run_name", "default_run"))
    aligned_dir = os.path.join(config.preprocessed_dir, run_name, "aligned_strips", aoi_name)
    inter_dir = os.path.join(config.preprocessed_dir, run_name, "icp_intermediate", aoi_name)
    log_dir = os.path.join(config.results_dir, config.run_name, "icp_logs", aoi_name)
    os.makedirs(aligned_dir, exist_ok=True)
    os.makedirs(inter_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return aligned_dir, inter_dir, log_dir

def _as_point_cloud(points: np.ndarray):
    import open3d as o3d

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud

def _read_las_points(las_path: str) -> np.ndarray:
    with laspy.open(las_path) as fh:
        las = fh.read()
    return np.column_stack((np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)))


def extract_overlap_area(strip_a_points: np.ndarray, strip_b_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract overlap in XY bounding boxes for two point sets."""
    if strip_a_points.size == 0 or strip_b_points.size == 0:
        return np.empty((0, 3)), np.empty((0, 3))

    a_min = strip_a_points[:, :2].min(axis=0)
    a_max = strip_a_points[:, :2].max(axis=0)
    b_min = strip_b_points[:, :2].min(axis=0)
    b_max = strip_b_points[:, :2].max(axis=0)

    ov_min = np.maximum(a_min, b_min)
    ov_max = np.minimum(a_max, b_max)

    if np.any(ov_max <= ov_min):
        return np.empty((0, 3)), np.empty((0, 3))

    a_mask = (
        (strip_a_points[:, 0] >= ov_min[0])
        & (strip_a_points[:, 0] <= ov_max[0])
        & (strip_a_points[:, 1] >= ov_min[1])
        & (strip_a_points[:, 1] <= ov_max[1])
    )
    b_mask = (
        (strip_b_points[:, 0] >= ov_min[0])
        & (strip_b_points[:, 0] <= ov_max[0])
        & (strip_b_points[:, 1] >= ov_min[1])
        & (strip_b_points[:, 1] <= ov_max[1])
    )

    return strip_a_points[a_mask], strip_b_points[b_mask]

from typing import Dict, Tuple, Optional

def run_icp(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    config,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, float, float]:
    """Run rigid ICP once (Open3D handles convergence internally)."""
    import open3d as o3d

    max_iter = int(getattr(config, "icp_max_iterations", 50))
    voxel_size = float(getattr(config, "icp_voxel_size", 1.0))
    max_corr = float(getattr(config, "icp_max_correspondence_distance", 2.0))
    mode = str(getattr(config, "icp_estimation", "point_to_point")).lower()

    source = _as_point_cloud(source_xyz)
    target = _as_point_cloud(target_xyz)

    if voxel_size > 0:
        source = source.voxel_down_sample(voxel_size)
        target = target.voxel_down_sample(voxel_size)

    if mode == "point_to_plane":
        normal_radius = float(getattr(config, "icp_normal_radius", 2.0))
        normal_max_nn = int(getattr(config, "icp_normal_max_nn", 30))
        search = o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn)

        source.estimate_normals(search)
        target.estimate_normals(search)
        source.normalize_normals()
        target.normalize_normals()

        estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_corr,
        np.eye(4),
        estimator,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
    )

    transform = np.asarray(result.transformation, dtype=np.float64)
    fitness = float(result.fitness)
    rmse = float(result.inlier_rmse)

    if logger is not None:
        logger.info("ICP done | fitness=%.6f rmse=%.6f", fitness, rmse)

    return transform, fitness, rmse

def apply_rigid_transform_to_las(
    input_path: str,
    transform_4x4: np.ndarray,
    output_path: str,
) -> None:
    """
    Apply a 4x4 rigid transform to LAS/LAZ XYZ while preserving all other attributes.
    """
    transform_4x4 = np.asarray(transform_4x4, dtype=np.float64)
    if transform_4x4.shape != (4, 4):
        raise ValueError("transform_4x4 must be a (4, 4) matrix")

    las = laspy.read(input_path)

    xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
    ones = np.ones((xyz.shape[0], 1), dtype=np.float64)
    xyz_h = np.hstack((xyz, ones))

    xyz_t = (transform_4x4 @ xyz_h.T).T[:, :3]

    las.x = xyz_t[:, 0]
    las.y = xyz_t[:, 1]
    las.z = xyz_t[:, 2]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    las.write(output_path)

def save_icp_log_for_pair(
    config,
    pair_id: str,
    transform: np.ndarray,
    fitness: float,
    rmse: float,
    metadata: dict | None = None,
    save_matrix_npy: bool = True,
) -> str:
    """
    Save ICP results for a strip-pair using repo folder conventions.

    Returns:
        Path to the JSON log file.
    """
    _, _, log_dir = _ensure_dirs(config)

    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("Transform must be a 4x4 matrix.")

    os.makedirs(log_dir, exist_ok=True)

    json_path = os.path.join(log_dir, f"{pair_id}.json")
    npy_path = os.path.join(log_dir, f"{pair_id}.npy")

    log_data = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "aoi": _get_aoi_name(config),
        "run_name": str(getattr(config, "run_name", "default_run")),
        "pair_id": pair_id,
        "fitness": float(fitness),
        "rmse": float(rmse),
        "transformation_matrix": transform.tolist(),
    }

    if metadata:
        log_data["metadata"] = metadata

    with open(json_path, "w") as f:
        json.dump(log_data, f, indent=2)

    if save_matrix_npy:
        np.save(npy_path, transform)

    return json_path

from typing import List, Tuple, Optional
import os
import shutil
import numpy as np


def align_strips_incremental(
    strip_paths: List[str],
    config,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """
    Very simple incremental ICP strip alignment.

    - First strip copied unchanged.
    - Each next strip aligned to previous aligned strip.
    - Optional XY overlap crop.
    - Single ICP run.
    - Accept/reject based on fitness + translation magnitude.
    - Saves JSON log per pair (no .npy).
    """

    if not strip_paths:
        return []

    aligned_dir, _, _ = _ensure_dirs(config)

    use_overlap_crop = bool(getattr(config, "icp_use_overlap_crop", True))
    min_overlap_points = int(getattr(config, "icp_min_overlap_points", 5000))
    min_fitness = float(getattr(config, "icp_min_fitness", 0.2))
    max_translation_m = float(getattr(config, "icp_max_translation_m", 50.0))

    aligned_paths = []

    # 1) First strip = reference
    first_strip = strip_paths[0]
    first_out = os.path.join(aligned_dir, os.path.basename(first_strip))
    shutil.copy2(first_strip, first_out)
    aligned_paths.append(first_out)

    # 2) Align remaining strips
    for i in range(1, len(strip_paths)):
        src_path = strip_paths[i]
        tgt_path = aligned_paths[i - 1]

        src_num = i + 1
        tgt_num = i
        pair_id = f"strip{src_num:02d}_to_strip{tgt_num:02d}"

        out_path = os.path.join(aligned_dir, os.path.basename(src_path))

        try:
            src_xyz = _read_las_points(src_path)
            tgt_xyz = _read_las_points(tgt_path)

            if src_xyz.size == 0 or tgt_xyz.size == 0:
                shutil.copy2(src_path, out_path)
                aligned_paths.append(out_path)
                continue

            # Optional overlap crop
            if use_overlap_crop:
                src_est, tgt_est = extract_overlap_area(src_xyz, tgt_xyz)
                if min(len(src_est), len(tgt_est)) < min_overlap_points:
                    shutil.copy2(src_path, out_path)
                    aligned_paths.append(out_path)
                    continue
            else:
                src_est, tgt_est = src_xyz, tgt_xyz

            # Run ICP
            transform, fitness, rmse = run_icp(src_est, tgt_est, config)

            translation_m = float(np.linalg.norm(transform[:3, 3]))

            accepted = (
                fitness >= min_fitness
                and translation_m <= max_translation_m
            )

            # Save JSON log
            save_icp_log_for_pair(
                config=config,
                pair_id=pair_id,
                transform=transform,
                fitness=fitness,
                rmse=rmse,
                metadata={
                    "source": os.path.abspath(src_path),
                    "target": os.path.abspath(tgt_path),
                    "accepted": accepted,
                    "translation_m": translation_m,
                    "used_overlap_crop": use_overlap_crop,
                    "n_source_icp": int(len(src_est)),
                    "n_target_icp": int(len(tgt_est)),
                },
                save_matrix_npy=False,  # <- disabled
            )

            if not accepted:
                shutil.copy2(src_path, out_path)
                aligned_paths.append(out_path)
                continue

            # Apply transform to full strip
            apply_rigid_transform_to_las(src_path, transform, out_path)
            aligned_paths.append(out_path)

        except Exception as e:
            if logger:
                logger.error("ICP failed for %s: %s", pair_id, e)
            shutil.copy2(src_path, out_path)
            aligned_paths.append(out_path)

    return aligned_paths