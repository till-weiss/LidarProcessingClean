import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from typing import List, Optional, Tuple

import laspy
import numpy as np
import pdal




def _make_icp_ready_strip(
    in_processed_strip: str,
    out_icp_ready_strip: str,
    config,
    logger: Optional[logging.Logger] = None,
) -> str:
    os.makedirs(os.path.dirname(out_icp_ready_strip), exist_ok=True)

    smrf_slope = float(getattr(config, "icp_smrf_slope", 0.2))
    smrf_window = float(getattr(config, "icp_smrf_window", 16.0))
    smrf_threshold = float(getattr(config, "icp_smrf_threshold", 0.45))
    smrf_scalar = float(getattr(config, "icp_smrf_scalar", 1.2))
    min_ground_points = int(getattr(config, "icp_min_ground_points", 50000))

    use_ground_outlier = bool(getattr(config, "icp_ground_use_outlier", False))
    outlier_mean_k = int(getattr(config, "icp_ground_outlier_mean_k", 8))
    outlier_multiplier = float(getattr(config, "icp_ground_outlier_multiplier", 2.0))

    cluster_tolerance = float(getattr(config, "icp_cluster_tolerance_m", 2.0))
    cluster_min_points = int(getattr(config, "icp_cluster_min_points", 1000))

    points_in = int(laspy.open(in_processed_strip).header.point_count)
    ground_points = 0
    clusters_found = 0
    kept_points = 0

    tmp_ground = None
    tmp_clustered = None
    try:
        with tempfile.NamedTemporaryFile(suffix="_ground.laz", delete=False) as t1:
            tmp_ground = t1.name
        with tempfile.NamedTemporaryFile(suffix="_clustered.laz", delete=False) as t2:
            tmp_clustered = t2.name

        pipe = [
            {"type": "readers.las", "filename": in_processed_strip},
            {
                "type": "filters.smrf",
                "slope": smrf_slope,
                "window": smrf_window,
                "threshold": smrf_threshold,
                "scalar": smrf_scalar,
            },
            {"type": "filters.range", "limits": "Classification[2:2]"},
        ]
        if use_ground_outlier:
            pipe.append(
                {
                    "type": "filters.outlier",
                    "method": "statistical",
                    "mean_k": outlier_mean_k,
                    "multiplier": outlier_multiplier,
                }
            )
        pipe.append({"type": "writers.las", "filename": tmp_ground, "compression": "laszip"})
        pdal.Pipeline(json.dumps(pipe)).execute()

        ground_points = int(laspy.open(tmp_ground).header.point_count)
        if ground_points < min_ground_points:
            shutil.copy2(in_processed_strip, out_icp_ready_strip)
            kept_points = int(laspy.open(out_icp_ready_strip).header.point_count)
            clusters_found = 0
            return out_icp_ready_strip

        cluster_pipe = [
            {"type": "readers.las", "filename": tmp_ground},
            {
                "type": "filters.cluster",
                "min_points": cluster_min_points,
                "tolerance": cluster_tolerance,
            },
            {"type": "writers.las", "filename": tmp_clustered, "compression": "laszip"},
        ]
        pdal.Pipeline(json.dumps(cluster_pipe)).execute()

        clustered = laspy.read(tmp_clustered)
        dim_names = list(clustered.point_format.dimension_names)
        cluster_dim = "ClusterID" if "ClusterID" in dim_names else ("ClusterId" if "ClusterId" in dim_names else None)

        if cluster_dim is None:
            shutil.copy2(tmp_ground, out_icp_ready_strip)
            kept_points = int(laspy.open(out_icp_ready_strip).header.point_count)
            clusters_found = 0
            return out_icp_ready_strip

        cluster_ids = np.asarray(clustered[cluster_dim])
        valid = cluster_ids >= 0
        if not np.any(valid):
            shutil.copy2(tmp_ground, out_icp_ready_strip)
            kept_points = int(laspy.open(out_icp_ready_strip).header.point_count)
            clusters_found = 0
            return out_icp_ready_strip

        valid_ids = cluster_ids[valid]
        unique_ids, counts = np.unique(valid_ids, return_counts=True)
        clusters_found = int(len(unique_ids))
        keep_id = int(unique_ids[np.argmax(counts)])
        keep_mask = cluster_ids == keep_id

        out_las = laspy.LasData(clustered.header)
        out_las.points = clustered.points[keep_mask]
        out_las.write(out_icp_ready_strip)
        kept_points = int(np.count_nonzero(keep_mask))
        return out_icp_ready_strip

    except Exception:
        shutil.copy2(in_processed_strip, out_icp_ready_strip)
        kept_points = int(laspy.open(out_icp_ready_strip).header.point_count)
        clusters_found = 0
        return out_icp_ready_strip
    finally:
        if logger:
            frac = (kept_points / points_in) if points_in else 0.0
            logger.info(
                "ICP-ready strip | in_points=%d ground_points=%d clusters_found=%d kept_points=%d kept_fraction=%.4f output=%s",
                points_in,
                ground_points,
                clusters_found,
                kept_points,
                frac,
                out_icp_ready_strip,
            )
        else:
            frac = (kept_points / points_in) if points_in else 0.0
            print(
                f"ICP-ready strip | in_points={points_in} ground_points={ground_points} clusters_found={clusters_found} "
                f"kept_points={kept_points} kept_fraction={frac:.4f} output={out_icp_ready_strip}"
            )
        for path in (tmp_ground, tmp_clustered):
            if path and os.path.exists(path):
                os.remove(path)

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


def _aligned_output_path(aligned_dir: str, strip_path: str) -> str:
    base, ext = os.path.splitext(os.path.basename(strip_path))
    return os.path.join(aligned_dir, f"{base}_aligned{ext}")


def _read_las_points(las_path: str) -> np.ndarray:
    with laspy.open(las_path) as fh:
        las = fh.read()
    return np.column_stack((np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)))


def extract_overlap_area(strip_a_points: np.ndarray, strip_b_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def run_icp(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    config,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, float, float]:
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


def apply_rigid_transform_to_las(input_path: str, transform_4x4: np.ndarray, output_path: str) -> None:
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
) -> str:
    _, _, _ = _ensure_dirs(config)

    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("Transform must be a 4x4 matrix.")

    run_name = str(getattr(config, "run_name", "default_run"))
    run_log_dir = os.path.join(config.results_dir, "icp_logs", run_name)
    os.makedirs(run_log_dir, exist_ok=True)

    jsonl_path = os.path.join(run_log_dir, f"{_get_aoi_name(config)}_icp_summary.jsonl")

    log_data = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "aoi": _get_aoi_name(config),
        "run_name": run_name,
        "pair_id": pair_id,
        "fitness": float(fitness),
        "rmse": float(rmse),
        "transformation_matrix": transform.tolist(),
    }
    if metadata:
        log_data["metadata"] = metadata

    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_data) + "\n")

    return jsonl_path


def align_strips_incremental(
    strip_paths: List[str],
    config,
    logger: Optional[logging.Logger] = None,
    icp_ready_paths: Optional[List[str]] = None,
) -> List[str]:
    if not strip_paths:
        return []

    aligned_dir, _, _ = _ensure_dirs(config)

    use_overlap_crop = bool(getattr(config, "icp_use_overlap_crop", True))
    min_overlap_points = int(getattr(config, "icp_min_overlap_points", 5000))
    min_fitness = float(getattr(config, "icp_min_fitness", 0.2))
    max_translation_m = float(getattr(config, "icp_max_translation_m", 50.0))
    max_median_z_diff_m = float(getattr(config, "icp_max_median_z_diff_m", 10.0))

    icp_input_by_strip = {}
    if icp_ready_paths and len(icp_ready_paths) == len(strip_paths):
        icp_input_by_strip = dict(zip(strip_paths, icp_ready_paths))

    aligned_paths = []
    first_strip = strip_paths[0]
    first_out = _aligned_output_path(aligned_dir, first_strip)
    shutil.copy2(first_strip, first_out)
    aligned_paths.append(first_out)

    for i in range(1, len(strip_paths)):
        src_path = strip_paths[i]
        tgt_path = aligned_paths[i - 1]
        pair_id = f"strip{i + 1:02d}_to_strip{i:02d}"
        out_path = _aligned_output_path(aligned_dir, src_path)

        try:
            src_icp_path = icp_input_by_strip.get(src_path, src_path)
            tgt_icp_path = icp_input_by_strip.get(strip_paths[i - 1], tgt_path)

            msg = (
                f"ICP correspondence uses icp-ready strips | source={src_icp_path} "
                f"target={tgt_icp_path}; transform applied to full strip={src_path}"
            )
            if logger:
                logger.info(msg)
            else:
                print(msg)

            src_xyz = _read_las_points(src_icp_path)
            tgt_xyz = _read_las_points(tgt_icp_path)
            if src_xyz.size == 0 or tgt_xyz.size == 0:
                shutil.copy2(src_path, out_path)
                aligned_paths.append(out_path)
                continue

            if use_overlap_crop:
                src_est, tgt_est = extract_overlap_area(src_xyz, tgt_xyz)
                if min(len(src_est), len(tgt_est)) < min_overlap_points:
                    shutil.copy2(src_path, out_path)
                    aligned_paths.append(out_path)
                    continue
            else:
                src_est, tgt_est = src_xyz, tgt_xyz

            transform, fitness, rmse = run_icp(src_est, tgt_est, config, logger=logger)
            translation_m = float(np.linalg.norm(transform[:3, 3]))
            median_z_diff = float(abs(np.median(src_est[:, 2]) - np.median(tgt_est[:, 2])))

            accepted = (
                fitness >= min_fitness
                and translation_m <= max_translation_m
                and median_z_diff <= max_median_z_diff_m
            )

            summary_path = save_icp_log_for_pair(
                config=config,
                pair_id=pair_id,
                transform=transform,
                fitness=fitness,
                rmse=rmse,
                metadata={
                    "source": os.path.abspath(src_path),
                    "target": os.path.abspath(tgt_path),
                    "source_icp": os.path.abspath(src_icp_path),
                    "target_icp": os.path.abspath(tgt_icp_path),
                    "accepted": accepted,
                    "translation_m": translation_m,
                    "used_overlap_crop": use_overlap_crop,
                    "n_source_icp": int(len(src_est)),
                    "n_target_icp": int(len(tgt_est)),
                    "median_z_diff_m": median_z_diff,
                    "used_icp_ready": bool(icp_input_by_strip),
                },
            )
            if logger:
                logger.info("ICP summary appended: %s", summary_path)
            else:
                print(f"ICP summary appended: {summary_path}")

            if not accepted:
                if median_z_diff > max_median_z_diff_m:
                    warn_msg = (
                        f"ICP auto-reject {pair_id}: possible vertical datum mismatch / bad ICP input "
                        f"(median_z_diff_m={median_z_diff:.3f} > {max_median_z_diff_m:.3f})"
                    )
                    if logger:
                        logger.warning(warn_msg)
                    else:
                        print(warn_msg)
                shutil.copy2(src_path, out_path)
                aligned_paths.append(out_path)
                continue

            apply_rigid_transform_to_las(src_path, transform, out_path)
            aligned_paths.append(out_path)

        except Exception as e:
            if logger:
                logger.error("ICP failed for %s: %s", pair_id, e)
            else:
                print(f"ICP failed for {pair_id}: {e}")
            shutil.copy2(src_path, out_path)
            aligned_paths.append(out_path)

    return aligned_paths
