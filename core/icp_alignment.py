import logging
import os
import shutil
from typing import List, Literal, Tuple

import laspy
import numpy as np


def save_icp_log(logger, message: str, level: str = "info") -> None:
    """Write an ICP log message with the requested level."""
    log_fn = getattr(logger, level.lower(), logger.info)
    log_fn(message)


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




def _sanitize_aligned_basename(input_path: str) -> str:
    base = os.path.basename(input_path)
    stem, ext = os.path.splitext(base)
    if stem.endswith("_preicp_utm"):
        stem = stem[: -len("_preicp_utm")]
    return f"{stem}_aligned{ext}"


def _validate_run_scoped_output_dir(aligned_dir: str, run_name: str) -> None:
    marker = os.path.join(aligned_dir, ".icp_aligned_run_name")
    if os.path.exists(marker):
        with open(marker, "r", encoding="utf-8") as f:
            recorded = f.read().strip()
        if recorded and recorded != run_name:
            raise RuntimeError(f"Aligned output directory collision: {aligned_dir} belongs to run_name={recorded}, current run_name={run_name}")
    else:
        with open(marker, "w", encoding="utf-8") as f:
            f.write(run_name)

def _build_logger(config):
    _, _, log_dir = _ensure_dirs(config)
    logger = logging.getLogger(f"icp.{_get_aoi_name(config)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    if getattr(config, "icp_save_logs", True):
        file_handler = logging.FileHandler(os.path.join(log_dir, "icp_debug.log"), mode="w")
        file_handler.setFormatter(logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s"))
        logger.addHandler(file_handler)

    return logger


def _close_logger(logger):
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def _as_point_cloud(points: np.ndarray):
    import open3d as o3d

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud


def detect_xy_units(points_xy: np.ndarray) -> Literal["degrees", "metres", "unknown"]:
    """Detect likely XY units from numeric ranges."""
    if points_xy.size == 0:
        return "unknown"

    x = points_xy[:, 0]
    y = points_xy[:, 1]
    x_in_range = np.mean((x >= -180.0) & (x <= 180.0))
    y_in_range = np.mean((y >= -90.0) & (y <= 90.0))

    if x_in_range > 0.95 and y_in_range > 0.95:
        return "degrees"
    return "metres"


def _assert_metric_points(points: np.ndarray, label: str) -> None:
    units = detect_xy_units(points[:, :2]) if points.size else "unknown"
    if units == "degrees":
        raise ValueError(
            f"{label} appears geographic (degrees). ICP is metric-only; "
            "ensure pre-ICP reprojection to projected UTM succeeded."
        )


def _voxel_downsample_indices(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if points.size == 0:
        return np.array([], dtype=np.int64)
    if voxel_size <= 0:
        return np.arange(points.shape[0], dtype=np.int64)

    mins = points.min(axis=0)
    grid = np.floor((points - mins) / voxel_size).astype(np.int64)
    key_to_idx = {}
    for i, key in enumerate(map(tuple, grid)):
        if key not in key_to_idx:
            key_to_idx[key] = i
    return np.fromiter(key_to_idx.values(), dtype=np.int64)


def _read_las_points(las_path: str) -> np.ndarray:
    with laspy.open(las_path) as fh:
        las = fh.read()
    return np.column_stack((np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)))


def _read_selected_points_with_indices(las_path: str, config) -> Tuple[np.ndarray, np.ndarray, dict]:
    points = _read_las_points(las_path)
    _assert_metric_points(points, f"Input strip {os.path.basename(las_path)}")

    if len(points) == 0:
        return np.empty((0, 3)), np.array([], dtype=np.int64), {
            "units_mode": "metres",
            "effective_grid_x": float(getattr(config, "icp_ground_grid_size", 1.0)),
            "effective_grid_y": float(getattr(config, "icp_ground_grid_size", 1.0)),
            "fallback_used": False,
        }

    grid_size = max(float(getattr(config, "icp_ground_grid_size", 1.0)), 1e-9)

    if not getattr(config, "icp_use_ground_only", True):
        return points, np.arange(points.shape[0], dtype=np.int64), {
            "units_mode": "metres",
            "effective_grid_x": grid_size,
            "effective_grid_y": grid_size,
            "fallback_used": False,
        }

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    ix = np.floor((x - x.min()) / grid_size).astype(np.int64)
    iy = np.floor((y - y.min()) / grid_size).astype(np.int64)

    key_to_best = {}
    for idx, key in enumerate(zip(ix, iy)):
        best_idx = key_to_best.get(key)
        if best_idx is None or z[idx] < z[best_idx]:
            key_to_best[key] = idx

    ground_idx = np.asarray(list(key_to_best.values()), dtype=np.int64)

    tol = float(getattr(config, "threshold", 0.5))
    z_map = {k: z[v] for k, v in key_to_best.items()}
    keep = []
    for idx in ground_idx:
        cx, cy = ix[idx], iy[idx]
        local_min = z[idx]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbor_z = z_map.get((cx + dx, cy + dy))
                if neighbor_z is not None and neighbor_z < local_min:
                    local_min = neighbor_z
        if z[idx] <= local_min + tol:
            keep.append(idx)

    selected_idx = np.asarray(keep if keep else ground_idx.tolist(), dtype=np.int64)

    min_selected = int(getattr(config, "icp_min_selected_points", 50_000))
    fallback_used = False
    if selected_idx.size < min_selected:
        fallback_used = True
        selected_idx = np.arange(points.shape[0], dtype=np.int64)
        voxel = max(float(getattr(config, "icp_voxel_size", 1.0)), 1e-9)
        keep_local = _voxel_downsample_indices(points[selected_idx], voxel)
        selected_idx = selected_idx[keep_local]

    cap = int(getattr(config, "icp_ground_max_points", 5_000_000))
    if selected_idx.size > cap:
        rng = np.random.default_rng(42)
        selected_idx = rng.choice(selected_idx, size=cap, replace=False)

    metadata = {
        "units_mode": "metres",
        "effective_grid_x": grid_size,
        "effective_grid_y": grid_size,
        "fallback_used": fallback_used,
    }
    return points[selected_idx], selected_idx, metadata


def select_icp_points(las_path: str, config) -> np.ndarray:
    """Return XYZ points for ICP selection (ground-candidate or full cloud)."""
    selected_points, _, _ = _read_selected_points_with_indices(las_path, config)
    return selected_points


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


def _overlap_metrics(strip_a_points: np.ndarray, strip_b_points: np.ndarray) -> dict:
    """Compute XY bbox overlap metrics and corresponding overlap subsets."""
    if strip_a_points.size == 0 or strip_b_points.size == 0:
        return {
            "area": 0.0,
            "source_overlap": np.empty((0, 3)),
            "target_overlap": np.empty((0, 3)),
            "source_mask": np.zeros((strip_a_points.shape[0],), dtype=bool),
            "target_mask": np.zeros((strip_b_points.shape[0],), dtype=bool),
        }

    a_min = strip_a_points[:, :2].min(axis=0)
    a_max = strip_a_points[:, :2].max(axis=0)
    b_min = strip_b_points[:, :2].min(axis=0)
    b_max = strip_b_points[:, :2].max(axis=0)

    ov_min = np.maximum(a_min, b_min)
    ov_max = np.minimum(a_max, b_max)
    ov_dims = ov_max - ov_min
    if np.any(ov_dims <= 0):
        return {
            "area": 0.0,
            "source_overlap": np.empty((0, 3)),
            "target_overlap": np.empty((0, 3)),
            "source_mask": np.zeros((strip_a_points.shape[0],), dtype=bool),
            "target_mask": np.zeros((strip_b_points.shape[0],), dtype=bool),
        }

    source_mask = (
        (strip_a_points[:, 0] >= ov_min[0])
        & (strip_a_points[:, 0] <= ov_max[0])
        & (strip_a_points[:, 1] >= ov_min[1])
        & (strip_a_points[:, 1] <= ov_max[1])
    )
    target_mask = (
        (strip_b_points[:, 0] >= ov_min[0])
        & (strip_b_points[:, 0] <= ov_max[0])
        & (strip_b_points[:, 1] >= ov_min[1])
        & (strip_b_points[:, 1] <= ov_max[1])
    )
    return {
        "area": float(ov_dims[0] * ov_dims[1]),
        "source_overlap": strip_a_points[source_mask],
        "target_overlap": strip_b_points[target_mask],
        "source_mask": source_mask,
        "target_mask": target_mask,
    }


def run_icp(source_points: np.ndarray, target_points: np.ndarray, config) -> tuple[np.ndarray, float, float, list[dict]]:
    """Run rigid ICP and capture per-iteration metrics in metric space."""
    _assert_metric_points(source_points, "ICP source points")
    _assert_metric_points(target_points, "ICP target points")

    max_iter = int(getattr(config, "icp_max_iterations", 50))
    voxel_size = float(getattr(config, "icp_voxel_size", 1.0))
    max_corr = float(getattr(config, "icp_max_correspondence_distance", 2.0))

    import open3d as o3d

    source = _as_point_cloud(source_points)
    target = _as_point_cloud(target_points)

    if voxel_size > 0:
        source = source.voxel_down_sample(voxel_size)
        target = target.voxel_down_sample(voxel_size)

    transform = np.eye(4)
    details = []
    last_fitness = 0.0
    last_rmse = float("inf")

    for i in range(1, max_iter + 1):
        prev_transform = transform.copy()
        reg = o3d.pipelines.registration.registration_icp(
            source,
            target,
            max_corr,
            transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1),
        )
        transform = reg.transformation
        last_fitness = float(reg.fitness)
        last_rmse = float(reg.inlier_rmse)

        delta = np.linalg.norm(transform - prev_transform)
        reason = "converged" if delta < 1e-8 else "running"

        details.append(
            {
                "iteration": i,
                "fitness": last_fitness,
                "rmse": last_rmse,
                "converged_or_reason": reason,
            }
        )

        if reason == "converged":
            break

    if details and details[-1]["converged_or_reason"] == "running":
        details[-1]["converged_or_reason"] = "max_iterations"

    return transform, last_fitness, last_rmse, details


def apply_transformation_to_las(input_path: str, transform_matrix: np.ndarray, output_path: str) -> None:
    """Apply rigid transform to XYZ while preserving all LAS attributes and CRS/header."""
    las = laspy.read(input_path)
    xyz = np.column_stack((np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)))
    xyz_h = np.hstack((xyz, np.ones((xyz.shape[0], 1), dtype=xyz.dtype)))
    transformed = (transform_matrix @ xyz_h.T).T[:, :3]

    las.x = transformed[:, 0]
    las.y = transformed[:, 1]
    las.z = transformed[:, 2]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    las.write(output_path)


def save_icp_iteration_details(output_path: str, iteration_details: list[dict]) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("iteration\tfitness\trmse\tconverged_or_reason\n")
        for row in iteration_details:
            f.write(
                f"{row['iteration']}\t{row['fitness']:.8f}\t{row['rmse']:.8f}\t{row['converged_or_reason']}\n"
            )


def save_intermediate_las_subset(input_path: str, indices_or_mask, output_path: str) -> None:
    """Save LAS subset preserving point attributes and CRS/header."""
    las = laspy.read(input_path)
    subset = las.points[indices_or_mask]
    out_las = laspy.LasData(las.header)
    out_las.points = subset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_las.write(output_path)


def _rotation_angle_deg(transform: np.ndarray) -> float:
    rot = transform[:3, :3]
    trace_val = np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace_val)))


def align_strips_incremental(strip_paths: List[str], config) -> List[str]:
    """Incrementally align strips using metric-only ICP; fallback to passthrough on errors/low overlap."""
    if not strip_paths:
        return []

    aligned_dir, inter_dir, log_dir = _ensure_dirs(config)
    logger = _build_logger(config)
    aligned_paths: List[str] = []
    aligned_cache: List[dict] = []

    try:
        save_icp_log(logger, f"AOI={_get_aoi_name(config)} | strips={len(strip_paths)}")
        save_icp_log(logger, "ICP mode=metric-only (degrees are not supported here).")

        _validate_run_scoped_output_dir(aligned_dir, str(getattr(config, "run_name", "default_run")))

        first_out = os.path.abspath(os.path.join(aligned_dir, _sanitize_aligned_basename(strip_paths[0])))
        shutil.copy2(strip_paths[0], first_out)
        aligned_paths.append(first_out)

        first_points, first_idx, first_meta = _read_selected_points_with_indices(first_out, config)
        aligned_cache.append(
            {
                "path": first_out,
                "strip_num": 1,
                "selected_points": first_points,
                "selected_idx": first_idx,
                "meta": first_meta,
            }
        )

        for idx in range(1, len(strip_paths)):
            src_original = strip_paths[idx]
            tgt_aligned_prev = aligned_paths[idx - 1]
            src_num = idx + 1
            tgt_num = idx
            src_out = os.path.abspath(os.path.join(aligned_dir, _sanitize_aligned_basename(src_original)))

            try:
                src_points, src_idx, src_meta = _read_selected_points_with_indices(src_original, config)
                save_icp_log(logger, "Units mode=metres")
                save_icp_log(
                    logger,
                    f"Effective voxel_size={float(getattr(config, 'icp_voxel_size', 1.0)):.6f} max_corr_distance={float(getattr(config, 'icp_max_correspondence_distance', 2.0)):.6f}",
                )
                save_icp_log(logger, f"Selected points source={len(src_points)}")

                if src_meta.get("fallback_used"):
                    save_icp_log(logger, f"Source fallback triggered: selected<{int(getattr(config, 'icp_min_selected_points', 50000))}, switched to full-cloud + voxel downsample", "warning")

                if len(src_points) == 0:
                    save_icp_log(logger, f"Pair strip{src_num:02d}->N/A status=skipped_low_overlap reason=empty source selection; passthrough.", "warning")
                    shutil.copy2(src_original, src_out)
                    aligned_paths.append(src_out)
                    aligned_cache.append(
                        {
                            "path": src_out,
                            "strip_num": src_num,
                            "selected_points": src_points,
                            "selected_idx": src_idx,
                            "meta": src_meta,
                        }
                    )
                    continue

                if len(src_points) < 1000:
                    save_icp_log(logger, f"Pair strip{src_num:02d}->N/A status=skipped_low_overlap reason=insufficient source selected points ({len(src_points)}); passthrough.", "warning")
                    shutil.copy2(src_original, src_out)
                    aligned_paths.append(src_out)
                    aligned_cache.append(
                        {
                            "path": src_out,
                            "strip_num": src_num,
                            "selected_points": src_points,
                            "selected_idx": src_idx,
                            "meta": src_meta,
                        }
                    )
                    continue

                min_overlap_area = float(getattr(config, "icp_min_overlap_area", 10_000.0))
                min_overlap_points = int(getattr(config, "icp_min_overlap_points", 300_000))

                best_candidate = None
                for candidate in aligned_cache:
                    tgt_points = candidate["selected_points"]
                    if len(tgt_points) == 0:
                        continue
                    metrics = _overlap_metrics(src_points, tgt_points)
                    overlap_count = min(len(metrics["source_overlap"]), len(metrics["target_overlap"]))
                    src_overlap_ratio = float(len(metrics["source_overlap"]) / max(len(src_points), 1))
                    tgt_overlap_ratio = float(len(metrics["target_overlap"]) / max(len(tgt_points), 1))
                    if not (metrics["area"] > min_overlap_area or overlap_count > min_overlap_points):
                        continue
                    score = (metrics["area"], overlap_count, src_overlap_ratio + tgt_overlap_ratio)
                    if best_candidate is None or score > best_candidate["score"]:
                        best_candidate = {
                            "candidate": candidate,
                            "metrics": metrics,
                            "score": score,
                        }

                if best_candidate is None:
                    save_icp_log(logger, f"Pair strip{src_num:02d}->N/A status=skipped_low_overlap reason=no aligned target met overlap gate (area>{min_overlap_area:.3f} or points>{min_overlap_points}); passthrough.", "warning")
                    shutil.copy2(src_original, src_out)
                    aligned_paths.append(src_out)
                    aligned_cache.append(
                        {
                            "path": src_out,
                            "strip_num": src_num,
                            "selected_points": src_points,
                            "selected_idx": src_idx,
                            "meta": src_meta,
                        }
                    )
                    continue

                tgt_entry = best_candidate["candidate"]
                tgt_points = tgt_entry["selected_points"]
                tgt_idx = tgt_entry["selected_idx"]
                tgt_meta = tgt_entry["meta"]
                tgt_aligned_prev = tgt_entry["path"]
                tgt_num = tgt_entry["strip_num"]

                src_overlap = best_candidate["metrics"]["source_overlap"]
                tgt_overlap = best_candidate["metrics"]["target_overlap"]
                src_ov_mask = best_candidate["metrics"]["source_mask"]
                tgt_ov_mask = best_candidate["metrics"]["target_mask"]
                overlap_area = float(best_candidate["metrics"]["area"])
                overlap_count = min(len(src_overlap), len(tgt_overlap))
                src_overlap_ratio = float(len(src_overlap) / max(len(src_points), 1))
                tgt_overlap_ratio = float(len(tgt_overlap) / max(len(tgt_points), 1))

                save_icp_log(
                    logger,
                    f"Pair strip{src_num:02d}->{tgt_num:02d} | source={os.path.abspath(src_original)} | target={os.path.abspath(tgt_aligned_prev)} | selected_target_strip={os.path.abspath(tgt_aligned_prev)}",
                )
                save_icp_log(
                    logger,
                    f"Effective grid source=({src_meta['effective_grid_x']:.6f}, {src_meta['effective_grid_y']:.6f}) target=({tgt_meta['effective_grid_x']:.6f}, {tgt_meta['effective_grid_y']:.6f})",
                )
                save_icp_log(logger, f"Selected points source={len(src_points)}, target={len(tgt_points)}")

                if tgt_meta.get("fallback_used"):
                    save_icp_log(logger, f"Target fallback triggered: selected<{int(getattr(config, 'icp_min_selected_points', 50000))}, switched to full-cloud + voxel downsample", "warning")

                if len(tgt_points) < 1000:
                    save_icp_log(logger, f"Pair strip{src_num:02d}->{tgt_num:02d} status=skipped_low_overlap reason=insufficient target selected points ({len(tgt_points)}); passthrough.", "warning")
                    shutil.copy2(src_original, src_out)
                    aligned_paths.append(src_out)
                    aligned_cache.append(
                        {
                            "path": src_out,
                            "strip_num": src_num,
                            "selected_points": src_points,
                            "selected_idx": src_idx,
                            "meta": src_meta,
                        }
                    )
                    continue

                smin, smax = src_points[:, :2].min(axis=0), src_points[:, :2].max(axis=0)
                tmin, tmax = tgt_points[:, :2].min(axis=0), tgt_points[:, :2].max(axis=0)
                save_icp_log(logger, f"Source XY bbox min={smin.tolist()} max={smax.tolist()}")
                save_icp_log(logger, f"Target XY bbox min={tmin.tolist()} max={tmax.tolist()}")
                save_icp_log(
                    logger,
                    f"Overlap area_m2={overlap_area:.3f} points_source={len(src_overlap)} points_target={len(tgt_overlap)} overlap_ratio_source={src_overlap_ratio:.6f} overlap_ratio_target={tgt_overlap_ratio:.6f}",
                )

                transform, fitness, rmse, details = run_icp(src_overlap, tgt_overlap, config)

                translation_m = float(np.linalg.norm(transform[:3, 3]))
                rotation_deg = _rotation_angle_deg(transform)
                min_fitness = float(getattr(config, "icp_min_fitness", 0.2))
                max_translation_m = float(getattr(config, "icp_max_translation_m", 50.0))
                max_rotation_deg = float(getattr(config, "icp_max_rotation_deg", 2.0))

                save_icp_log(logger, f"ICP fitness={fitness:.6f} rmse={rmse:.6f}")
                save_icp_log(logger, f"Transform:\n{transform}")
                save_icp_log(logger, f"Transform stats translation_m={translation_m:.4f}, rotation_deg={rotation_deg:.4f}")

                reject_reasons = []
                if translation_m > max_translation_m:
                    reject_reasons.append(f"translation_m {translation_m:.4f} > {max_translation_m:.4f}")
                if rotation_deg > max_rotation_deg:
                    reject_reasons.append(f"rotation_deg {rotation_deg:.4f} > {max_rotation_deg:.4f}")
                if fitness < min_fitness:
                    reject_reasons.append(f"fitness {fitness:.6f} < {min_fitness:.6f}")

                if reject_reasons:
                    save_icp_log(logger, f"Pair strip{src_num:02d}->{tgt_num:02d} status=rejected_transform reason={' ; '.join(reject_reasons)}. Passthrough unchanged.", "warning")
                    shutil.copy2(src_original, src_out)
                    aligned_paths.append(src_out)
                    aligned_cache.append(
                        {
                            "path": src_out,
                            "strip_num": src_num,
                            "selected_points": src_points,
                            "selected_idx": src_idx,
                            "meta": src_meta,
                        }
                    )
                    continue

                save_icp_log(logger, f"Pair strip{src_num:02d}->{tgt_num:02d} status=accepted")
                apply_transformation_to_las(src_original, transform, src_out)
                aligned_paths.append(src_out)

                out_points, out_idx, out_meta = _read_selected_points_with_indices(src_out, config)
                aligned_cache.append(
                    {
                        "path": src_out,
                        "strip_num": src_num,
                        "selected_points": out_points,
                        "selected_idx": out_idx,
                        "meta": out_meta,
                    }
                )

                if getattr(config, "icp_save_logs", True):
                    iter_path = os.path.join(log_dir, f"icp_iterations_strip{src_num:02d}_to_strip{tgt_num:02d}.txt")
                    save_icp_iteration_details(iter_path, details)

                if getattr(config, "icp_save_intermediate", True):
                    save_intermediate_las_subset(
                        src_original,
                        src_idx,
                        os.path.join(inter_dir, f"selected_points_source_strip_{src_num:02d}.laz"),
                    )
                    save_intermediate_las_subset(
                        tgt_aligned_prev,
                        tgt_idx,
                        os.path.join(inter_dir, f"selected_points_target_strip_{tgt_num:02d}.laz"),
                    )

                    save_intermediate_las_subset(
                        src_original,
                        src_idx[src_ov_mask],
                        os.path.join(inter_dir, f"overlap_source_strip_{src_num:02d}.laz"),
                    )
                    save_intermediate_las_subset(
                        tgt_aligned_prev,
                        tgt_idx[tgt_ov_mask],
                        os.path.join(inter_dir, f"overlap_target_strip_{tgt_num:02d}.laz"),
                    )

            except Exception as pair_err:
                save_icp_log(
                    logger,
                    f"Pair strip{src_num:02d}->{tgt_num:02d} failed: {pair_err}. Passthrough unchanged.",
                    "error",
                )
                shutil.copy2(src_original, src_out)
                aligned_paths.append(src_out)
                try:
                    out_points, out_idx, out_meta = _read_selected_points_with_indices(src_out, config)
                except Exception:
                    out_points, out_idx, out_meta = np.empty((0, 3)), np.array([], dtype=np.int64), {}
                aligned_cache.append(
                    {
                        "path": src_out,
                        "strip_num": src_num,
                        "selected_points": out_points,
                        "selected_idx": out_idx,
                        "meta": out_meta,
                    }
                )

    finally:
        _close_logger(logger)

    return aligned_paths
