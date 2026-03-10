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
    os.makedirs(os.path.join(aligned_dir, "strip_adjustment"), exist_ok=True)

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


def _build_strip_metadata(strip_path: str, strip_id: str) -> Dict[str, object]:
    with laspy.open(strip_path) as fh:
        header = fh.header
        mins = header.mins
        maxs = header.maxs
        n_points = int(header.point_count)
    pts = _read_las_points(strip_path)
    centroid = np.mean(pts, axis=0) if pts.size else np.array([None, None, None], dtype=object)
    return {
        "path": strip_path,
        "strip_id": strip_id,
        "minx": float(mins[0]),
        "maxx": float(maxs[0]),
        "miny": float(mins[1]),
        "maxy": float(maxs[1]),
        "minz": float(mins[2]),
        "maxz": float(maxs[2]),
        "centroid_x": None if centroid[0] is None else float(centroid[0]),
        "centroid_y": None if centroid[1] is None else float(centroid[1]),
        "centroid_z": None if centroid[2] is None else float(centroid[2]),
        "n_points": n_points,
    }


def _bbox_plausible(meta_a: Dict[str, object], meta_b: Dict[str, object], buffer_m: float) -> bool:
    return not (
        float(meta_a["maxx"]) < float(meta_b["minx"]) - buffer_m
        or float(meta_a["minx"]) > float(meta_b["maxx"]) + buffer_m
        or float(meta_a["maxy"]) < float(meta_b["miny"]) - buffer_m
        or float(meta_a["miny"]) > float(meta_b["maxy"]) + buffer_m
    )


def _transform_points(points_xyz: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    if points_xyz.size == 0:
        return points_xyz
    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float64)
    points_h = np.hstack((points_xyz.astype(np.float64), ones))
    return (transform_4x4 @ points_h.T).T[:, :3]


def _sample_points(points_xyz: np.ndarray, sample_n: int) -> np.ndarray:
    if points_xyz.size == 0:
        return points_xyz
    n = points_xyz.shape[0]
    if n <= sample_n:
        return points_xyz
    idx = np.linspace(0, n - 1, sample_n, dtype=int)
    return points_xyz[idx]


def _compute_qc_metrics(source_xyz: np.ndarray, target_xyz: np.ndarray) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {
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
        metrics["rmse_nn"] = float(np.sqrt(np.mean(distances**2)))
        metrics["median_z_diff_m"] = float(abs(np.median(source_xyz[:, 2]) - np.median(target_xyz[:, 2])))
        return metrics
    except Exception:
        return metrics


def _relative_to_aoi(path: str, config) -> str:
    if not path:
        return ""
    try:
        return os.path.relpath(path, _aoi_base_dir(config))
    except Exception:
        return os.path.basename(path)


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
    xyz_t = _transform_points(xyz, transform_4x4)

    las.x = xyz_t[:, 0]
    las.y = xyz_t[:, 1]
    las.z = xyz_t[:, 2]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    las.write(output_path)


def _append_pairwise_log(config, entry: dict) -> str:
    _, _, jsonl_path, _ = _ensure_dirs(config)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return jsonl_path


def _write_pairwise_summary(config, summary: dict) -> str:
    _, _, _, summary_path = _ensure_dirs(config)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary_path


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

    bbox_buffer_m = float(getattr(config, "icp_pair_bbox_buffer_m", 20.0))
    pair_sample_n = int(getattr(config, "icp_pair_sample_n", 4000))
    pair_max_median_dist_m = float(getattr(config, "icp_pair_max_median_dist_m", 8.0))
    pair_max_mean_dist_m = float(getattr(config, "icp_pair_max_mean_dist_m", 0.0))

    strip_id_by_path = {p: os.path.splitext(os.path.basename(p))[0] for p in strip_paths}
    icp_input_by_strip = {}
    if icp_ready_paths and len(icp_ready_paths) == len(strip_paths):
        icp_input_by_strip = dict(zip(strip_paths, icp_ready_paths))

    icp_points_by_strip = {p: _read_las_points(icp_input_by_strip.get(p, p)) for p in strip_paths}
    meta_by_strip = {
        p: _build_strip_metadata(icp_input_by_strip.get(p, p), strip_id_by_path[p])
        for p in strip_paths
    }

    # reference strip: largest point count
    reference_strip = max(strip_paths, key=lambda p: int(meta_by_strip[p].get("n_points", 0)))
    if logger:
        logger.info("ICP reference strip selected: %s", os.path.basename(reference_strip))

    identity = np.eye(4, dtype=np.float64)
    global_transform_by_strip: Dict[str, np.ndarray] = {reference_strip: identity}
    depth_by_strip: Dict[str, int] = {reference_strip: 0}
    parent_by_strip: Dict[str, Optional[str]] = {reference_strip: None}

    aligned_set = {reference_strip}
    pending_set = set(strip_paths) - aligned_set

    aligned_output_by_strip: Dict[str, str] = {}
    ref_out = _aligned_output_path(aligned_dir, reference_strip)
    shutil.copy2(reference_strip, ref_out)
    aligned_output_by_strip[reference_strip] = ref_out

    accepted_edges = 0
    rejected_edges = 0

    while pending_set:
        best = None
        best_score = None

        for src_path in list(pending_set):
            src_meta = meta_by_strip[src_path]
            src_points_global = _transform_points(icp_points_by_strip[src_path], global_transform_by_strip.get(src_path, identity))
            src_sample = _sample_points(src_points_global, pair_sample_n)

            for tgt_path in aligned_set:
                tgt_meta = meta_by_strip[tgt_path]
                bbox_ok = _bbox_plausible(src_meta, tgt_meta, bbox_buffer_m)

                pre_pair_qc = {
                    "n_points_used": None,
                    "mean_distance": None,
                    "median_distance": None,
                    "std_distance": None,
                    "rmse_nn": None,
                }

                candidate_passed_preselection = False
                if bbox_ok:
                    tgt_points_global = _transform_points(icp_points_by_strip[tgt_path], global_transform_by_strip[tgt_path])
                    tgt_sample = _sample_points(tgt_points_global, pair_sample_n)
                    qcm = _compute_qc_metrics(src_sample, tgt_sample)
                    pre_pair_qc = {
                        "n_points_used": qcm["n_points"],
                        "mean_distance": qcm["mean_distance"],
                        "median_distance": qcm["median_distance"],
                        "std_distance": qcm["std_distance"],
                        "rmse_nn": qcm["rmse_nn"],
                    }
                    med = qcm["median_distance"]
                    mean = qcm["mean_distance"]
                    candidate_passed_preselection = (
                        med is not None
                        and med <= pair_max_median_dist_m
                        and (pair_max_mean_dist_m <= 0 or (mean is not None and mean <= pair_max_mean_dist_m))
                    )

                pre_entry = {
                    "timestamp_utc": datetime.utcnow().isoformat(),
                    "aoi": _get_aoi_name(config),
                    "run_name": str(getattr(config, "run_name", "default_run")),
                    "pair_id": f"{strip_id_by_path[src_path]}_to_{strip_id_by_path[tgt_path]}",
                    "source_strip": os.path.basename(src_path),
                    "target_strip": os.path.basename(tgt_path),
                    "target_in_aligned_set": True,
                    "parent_strip": os.path.basename(tgt_path),
                    "chain_depth": int(depth_by_strip[tgt_path] + 1),
                    "candidate_passed_preselection": candidate_passed_preselection,
                    "pre_pair_qc": pre_pair_qc,
                }
                _append_pairwise_log(config, pre_entry)

                if not candidate_passed_preselection:
                    continue

                score = (pre_pair_qc["median_distance"], pre_pair_qc["mean_distance"] or 0.0)
                if best_score is None or score < best_score:
                    best_score = score
                    best = (src_path, tgt_path, pre_pair_qc)

        if best is None:
            break

        src_path, tgt_path, pre_pair_qc = best
        pair_id = f"{strip_id_by_path[src_path]}_to_{strip_id_by_path[tgt_path]}"
        out_path = _aligned_output_path(aligned_dir, src_path)

        reject_reason = None
        transform_applied = False

        try:
            src_icp_local = icp_points_by_strip[src_path]
            tgt_icp_local = icp_points_by_strip[tgt_path]

            if use_overlap_crop:
                src_est, tgt_est = extract_overlap_area(src_icp_local, tgt_icp_local)
                if min(len(src_est), len(tgt_est)) < min_overlap_points:
                    reject_reason = "insufficient_overlap_points"
                    rejected_edges += 1
                    shutil.copy2(src_path, out_path)
                    aligned_output_by_strip[src_path] = out_path
                    pending_set.remove(src_path)
                    aligned_set.add(src_path)
                    global_transform_by_strip[src_path] = global_transform_by_strip[tgt_path]
                    depth_by_strip[src_path] = depth_by_strip[tgt_path] + 1
                    parent_by_strip[src_path] = tgt_path
                    continue
            else:
                src_est, tgt_est = src_icp_local, tgt_icp_local

            pre_qc = _compute_qc_metrics(src_est, tgt_est)
            transform_child_to_parent, fitness, inlier_rmse = run_icp(src_est, tgt_est, config, logger=logger)

            global_transform = global_transform_by_strip[tgt_path] @ transform_child_to_parent
            src_est_post_global = _transform_points(src_est, global_transform)
            tgt_est_global = _transform_points(tgt_est, global_transform_by_strip[tgt_path])
            post_qc = _compute_qc_metrics(src_est_post_global, tgt_est_global)

            translation_xyz = [float(x) for x in transform_child_to_parent[:3, 3]]
            translation_norm = float(np.linalg.norm(transform_child_to_parent[:3, 3]))
            effective_delta = np.mean(src_est_post_global, axis=0) - np.mean(_transform_points(src_est, global_transform_by_strip.get(src_path, identity)), axis=0)
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
                apply_rigid_transform_to_las(src_path, global_transform, out_path)
                accepted_edges += 1
                global_transform_by_strip[src_path] = global_transform
            else:
                rejected_edges += 1
                shutil.copy2(src_path, out_path)
                global_transform_by_strip[src_path] = global_transform_by_strip[tgt_path]

            aligned_output_by_strip[src_path] = out_path
            aligned_set.add(src_path)
            pending_set.remove(src_path)
            depth_by_strip[src_path] = depth_by_strip[tgt_path] + 1
            parent_by_strip[src_path] = tgt_path

            icp_entry = {
                "timestamp_utc": datetime.utcnow().isoformat(),
                "aoi": _get_aoi_name(config),
                "run_name": str(getattr(config, "run_name", "default_run")),
                "pair_id": pair_id,
                "source_strip": os.path.basename(src_path),
                "target_strip": os.path.basename(tgt_path),
                "target_in_aligned_set": True,
                "parent_strip": os.path.basename(tgt_path),
                "chain_depth": int(depth_by_strip[src_path]),
                "pre_pair_qc": pre_pair_qc,
                "qc_gating_enabled": qc_gating_enabled,
                "qc_passed": qc_passed,
                "would_be_rejected": would_be_rejected,
                "transform_applied": transform_applied,
                "reject_reason": reject_reason,
                "pre_icp_qc": pre_qc,
                "icp_result": {
                    "fitness": fitness,
                    "inlier_rmse": inlier_rmse,
                    "transformation_matrix": transform_child_to_parent.tolist(),
                    "translation_xyz": translation_xyz,
                    "translation_norm": translation_norm,
                    "effective_shift_xy_m": effective_shift_xy_m,
                    "effective_shift_3d_m": effective_shift_3d_m,
                },
                "post_icp_qc": post_qc,
                "files": {
                    "source_strip": _relative_to_aoi(src_path, config),
                    "target_strip": _relative_to_aoi(aligned_output_by_strip[tgt_path], config),
                    "source_icp": _relative_to_aoi(icp_input_by_strip.get(src_path, src_path), config),
                    "target_icp": _relative_to_aoi(icp_input_by_strip.get(tgt_path, tgt_path), config),
                },
            }
            _append_pairwise_log(config, icp_entry)

        except Exception as e:
            if logger:
                logger.error("ICP failed for %s: %s", pair_id, e)
            else:
                print(f"ICP failed for {pair_id}: {e}")
            rejected_edges += 1
            shutil.copy2(src_path, out_path)
            aligned_output_by_strip[src_path] = out_path
            aligned_set.add(src_path)
            pending_set.remove(src_path)
            global_transform_by_strip[src_path] = global_transform_by_strip.get(tgt_path, identity)
            depth_by_strip[src_path] = depth_by_strip.get(tgt_path, 0) + 1
            parent_by_strip[src_path] = tgt_path

    # keep unconnected strips as unaligned copies
    unconnected = sorted(pending_set)
    for strip in unconnected:
        out_path = _aligned_output_path(aligned_dir, strip)
        shutil.copy2(strip, out_path)
        aligned_output_by_strip[strip] = out_path

    summary = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "aoi": _get_aoi_name(config),
        "run_name": str(getattr(config, "run_name", "default_run")),
        "selected_reference_strip": os.path.basename(reference_strip),
        "total_strips": len(strip_paths),
        "aligned_strips_count": len(strip_paths) - len(unconnected),
        "skipped_unconnected_strips_count": len(unconnected),
        "accepted_icp_edges": accepted_edges,
        "rejected_icp_edges": rejected_edges,
        "unconnected_strips": [os.path.basename(p) for p in unconnected],
    }
    _write_pairwise_summary(config, summary)

    return [aligned_output_by_strip[p] for p in strip_paths]
