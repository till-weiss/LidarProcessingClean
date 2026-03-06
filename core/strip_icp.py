import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

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

        pipeline = [
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
            pipeline.append(
                {
                    "type": "filters.outlier",
                    "method": "statistical",
                    "mean_k": outlier_mean_k,
                    "multiplier": outlier_multiplier,
                }
            )
        pipeline.append({"type": "writers.las", "filename": tmp_ground, "compression": "laszip"})
        pdal.Pipeline(json.dumps(pipeline)).execute()

        ground_points = int(laspy.open(tmp_ground).header.point_count)
        if ground_points < min_ground_points:
            shutil.copy2(in_processed_strip, out_icp_ready_strip)
            kept_points = int(laspy.open(out_icp_ready_strip).header.point_count)
            return out_icp_ready_strip

        cluster_pipeline = [
            {"type": "readers.las", "filename": tmp_ground},
            {
                "type": "filters.cluster",
                "min_points": cluster_min_points,
                "tolerance": cluster_tolerance,
            },
            {"type": "writers.las", "filename": tmp_clustered, "compression": "laszip"},
        ]
        pdal.Pipeline(json.dumps(cluster_pipeline)).execute()

        clustered = laspy.read(tmp_clustered)
        dim_names = list(clustered.point_format.dimension_names)
        cluster_dim = "ClusterID" if "ClusterID" in dim_names else ("ClusterId" if "ClusterId" in dim_names else None)

        if cluster_dim is None:
            shutil.copy2(tmp_ground, out_icp_ready_strip)
            kept_points = int(laspy.open(out_icp_ready_strip).header.point_count)
            return out_icp_ready_strip

        cluster_ids = np.asarray(clustered[cluster_dim])
        valid = cluster_ids >= 0
        if not np.any(valid):
            shutil.copy2(tmp_ground, out_icp_ready_strip)
            kept_points = int(laspy.open(out_icp_ready_strip).header.point_count)
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
        return out_icp_ready_strip
    finally:
        kept_fraction = (kept_points / points_in) if points_in else 0.0
        msg = (
            "ICP-ready strip | in_points=%d ground_points=%d clusters_found=%d "
            "kept_points=%d kept_fraction=%.4f output=%s"
        )
        vals = (points_in, ground_points, clusters_found, kept_points, kept_fraction, out_icp_ready_strip)
        if logger:
            logger.info(msg, *vals)
        else:
            print(msg % vals)

        for path in (tmp_ground, tmp_clustered):
            if path and os.path.exists(path):
                os.remove(path)


def _get_aoi_name(config) -> str:
    return str(getattr(config, "icp_current_aoi_name", "unknown_aoi"))


def _aoi_base_dir(config) -> str:
    run_name = str(getattr(config, "run_name", "default_run"))
    return os.path.join(config.preprocessed_dir, run_name, _get_aoi_name(config))


def _ensure_dirs(config) -> Tuple[str, str, str, str]:
    aoi_base = _aoi_base_dir(config)
    aligned_dir = os.path.join(aoi_base, "aligned_strips")
    icp_results_dir = os.path.join(aoi_base, "results", "icp")
    os.makedirs(aligned_dir, exist_ok=True)
    os.makedirs(icp_results_dir, exist_ok=True)
    os.makedirs(os.path.join(icp_results_dir, "qc"), exist_ok=True)
    os.makedirs(os.path.join(aoi_base, "aligned_strips", "strip_adjustment"), exist_ok=True)

    log_jsonl_path = os.path.join(icp_results_dir, "pairwise_icp_log.jsonl")
    summary_path = os.path.join(icp_results_dir, "pairwise_icp_summary.json")
    return aligned_dir, icp_results_dir, log_jsonl_path, summary_path


def _as_point_cloud(points: np.ndarray):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


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


def _compute_qc_metrics(source_xyz: np.ndarray, target_xyz: np.ndarray) -> Dict[str, Optional[float]]:
    metrics = {
        "n_points": None,
        "mean_distance": None,
        "median_distance": None,
        "std_distance": None,
        "rmse_nn": None,
        "median_z_diff_m": None,
    }

    if source_xyz.size == 0 or target_xyz.size == 0:
        return metrics

    try:
        source = _as_point_cloud(source_xyz)
        target = _as_point_cloud(target_xyz)
        distances = np.asarray(source.compute_point_cloud_distance(target), dtype=np.float64)
        if distances.size == 0:
            return metrics

        metrics["n_points"] = int(distances.size)
        metrics["mean_distance"] = float(np.mean(distances))
        metrics["median_distance"] = float(np.median(distances))
        metrics["std_distance"] = float(np.std(distances))
        metrics["rmse_nn"] = float(np.sqrt(np.mean(distances ** 2)))
        metrics["median_z_diff_m"] = float(abs(np.median(source_xyz[:, 2]) - np.median(target_xyz[:, 2])))
        return metrics
    except Exception:
        return metrics


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
    inlier_rmse = float(result.inlier_rmse)
    if logger is not None:
        logger.info("ICP done | fitness=%.6f inlier_rmse=%.6f", fitness, inlier_rmse)
    return transform, fitness, inlier_rmse


def apply_rigid_transform_to_las(input_path: str, transform_4x4: np.ndarray, output_path: str) -> None:
    transform_4x4 = np.asarray(transform_4x4, dtype=np.float64)
    if transform_4x4.shape != (4, 4):
        raise ValueError("transform_4x4 must be a (4, 4) matrix")

    las = laspy.read(input_path)
    xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
    xyz_h = np.hstack((xyz, np.ones((xyz.shape[0], 1), dtype=np.float64)))
    xyz_t = (transform_4x4 @ xyz_h.T).T[:, :3]

    las.x = xyz_t[:, 0]
    las.y = xyz_t[:, 1]
    las.z = xyz_t[:, 2]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    las.write(output_path)


def _append_pairwise_log(config, entry: dict) -> str:
    _, _, jsonl_path, summary_path = _ensure_dirs(config)

    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    # lightweight summary refresh
    total = 0
    qc_passed = 0
    transform_applied = 0
    would_be_rejected = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            qc_passed += int(bool(rec.get("qc_passed")))
            transform_applied += int(bool(rec.get("transform_applied")))
            would_be_rejected += int(bool(rec.get("would_be_rejected")))

    summary = {
        "aoi": _get_aoi_name(config),
        "run_name": str(getattr(config, "run_name", "default_run")),
        "log_file": os.path.basename(jsonl_path),
        "pair_count": total,
        "qc_passed_count": qc_passed,
        "transform_applied_count": transform_applied,
        "would_be_rejected_count": would_be_rejected,
        "updated_utc": datetime.utcnow().isoformat(),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return jsonl_path


def _relative_to_aoi(path: str, config) -> str:
    if not path:
        return ""
    try:
        return os.path.relpath(path, _aoi_base_dir(config))
    except Exception:
        return os.path.basename(path)


def align_strips_incremental(
    strip_paths: List[str],
    config,
    logger: Optional[logging.Logger] = None,
    icp_ready_paths: Optional[List[str]] = None,
) -> List[str]:
    if not strip_paths:
        return []

    aligned_dir, _, _, _ = _ensure_dirs(config)

    use_overlap_crop = bool(getattr(config, "icp_use_overlap_crop", True))
    min_overlap_points = int(getattr(config, "icp_min_overlap_points", 5000))
    min_fitness = float(getattr(config, "icp_min_fitness", 0.2))
    max_translation_m = float(getattr(config, "icp_max_translation_m", 50.0))
    max_median_z_diff_m = float(getattr(config, "icp_max_median_z_diff_m", 10.0))
    qc_gating_enabled = bool(getattr(config, "icp_enforce_qc_thresholds", True))

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

        reject_reason = None
        transform_applied = False

        try:
            src_icp_path = icp_input_by_strip.get(src_path, src_path)
            tgt_icp_path = icp_input_by_strip.get(strip_paths[i - 1], tgt_path)

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

            pre_qc = _compute_qc_metrics(src_est, tgt_est)

            transform, fitness, inlier_rmse = run_icp(src_est, tgt_est, config, logger=logger)
            src_est_h = np.hstack((src_est, np.ones((src_est.shape[0], 1), dtype=np.float64)))
            src_est_post = (transform @ src_est_h.T).T[:, :3]
            post_qc = _compute_qc_metrics(src_est_post, tgt_est)

            translation_xyz = [float(x) for x in transform[:3, 3]]
            translation_norm = float(np.linalg.norm(transform[:3, 3]))
            effective_delta = np.mean(src_est_post, axis=0) - np.mean(src_est, axis=0)
            effective_shift_xy_m = float(np.linalg.norm(effective_delta[:2]))
            effective_shift_3d_m = float(np.linalg.norm(effective_delta))

            median_z_diff_for_gate = pre_qc.get("median_z_diff_m")
            qc_passed = (
                fitness >= min_fitness
                and translation_norm <= max_translation_m
                and (median_z_diff_for_gate is not None and median_z_diff_for_gate <= max_median_z_diff_m)
            )

            would_be_rejected = not qc_passed
            if not qc_passed:
                if fitness < min_fitness:
                    reject_reason = "fitness_below_threshold"
                elif translation_norm > max_translation_m:
                    reject_reason = "translation_norm_above_threshold"
                elif median_z_diff_for_gate is None:
                    reject_reason = "median_z_diff_unavailable"
                elif median_z_diff_for_gate > max_median_z_diff_m:
                    reject_reason = "median_z_diff_above_threshold"
                else:
                    reject_reason = "qc_failed"

            transform_applied = qc_passed or not qc_gating_enabled
            if transform_applied:
                apply_rigid_transform_to_las(src_path, transform, out_path)
            else:
                shutil.copy2(src_path, out_path)
            aligned_paths.append(out_path)

            icp_entry = {
                "timestamp_utc": datetime.utcnow().isoformat(),
                "aoi": _get_aoi_name(config),
                "run_name": str(getattr(config, "run_name", "default_run")),
                "pair_id": pair_id,
                "source_strip": os.path.basename(src_path),
                "target_strip": os.path.basename(strip_paths[i - 1]),
                "qc_gating_enabled": qc_gating_enabled,
                "qc_passed": qc_passed,
                "would_be_rejected": would_be_rejected,
                "transform_applied": transform_applied,
                "reject_reason": reject_reason,
                "pre_icp_qc": pre_qc,
                "icp_result": {
                    "fitness": fitness,
                    "inlier_rmse": inlier_rmse,
                    "transformation_matrix": transform.tolist(),
                    "translation_xyz": translation_xyz,
                    "translation_norm": translation_norm,
                    "effective_shift_xy_m": effective_shift_xy_m,
                    "effective_shift_3d_m": effective_shift_3d_m,
                },
                "post_icp_qc": post_qc,
                "files": {
                    "source_strip": _relative_to_aoi(src_path, config),
                    "target_strip": _relative_to_aoi(tgt_path, config),
                    "source_icp": _relative_to_aoi(src_icp_path, config),
                    "target_icp": _relative_to_aoi(tgt_icp_path, config),
                },
            }

            log_path = _append_pairwise_log(config, icp_entry)
            if logger:
                logger.info("ICP pair logged: %s", log_path)

        except Exception as e:
            if logger:
                logger.error("ICP failed for %s: %s", pair_id, e)
            else:
                print(f"ICP failed for {pair_id}: {e}")
            shutil.copy2(src_path, out_path)
            aligned_paths.append(out_path)

    return aligned_paths
