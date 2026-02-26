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
    aligned_dir = os.path.join(config.preprocessed_dir, "aligned_strips", aoi_name)
    inter_dir = os.path.join(config.preprocessed_dir, "icp_intermediate", aoi_name)
    log_dir = os.path.join(config.results_dir, "icp_logs", aoi_name)
    os.makedirs(aligned_dir, exist_ok=True)
    os.makedirs(inter_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return aligned_dir, inter_dir, log_dir


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
    """Detect whether XY values are likely geodetic degrees or projected metres."""
    if points_xy.size == 0:
        return "unknown"

    x = points_xy[:, 0]
    y = points_xy[:, 1]

    x_in_range = np.mean((x >= -180.0) & (x <= 180.0))
    y_in_range = np.mean((y >= -90.0) & (y <= 90.0))

    if x_in_range > 0.95 and y_in_range > 0.95:
        return "degrees"
    return "metres"


def _resolve_units_mode(points_xy: np.ndarray, config) -> Literal["degrees", "metres", "unknown"]:
    forced = str(getattr(config, "icp_param_units", "auto")).lower()
    if forced in {"degrees", "metres"}:
        return forced
    return detect_xy_units(points_xy)


def _metres_to_degrees(lat_mean: float) -> Tuple[float, float]:
    deg_per_m_lat = 1.0 / 111320.0
    cos_lat = np.cos(np.radians(lat_mean))
    cos_lat = np.clip(np.abs(cos_lat), 1e-6, None)
    deg_per_m_lon = 1.0 / (111320.0 * cos_lat)
    return deg_per_m_lon, deg_per_m_lat


def _effective_xy_scales(grid_size_m: float, units_mode: str, points_xy: np.ndarray) -> Tuple[float, float]:
    if units_mode == "degrees":
        lat_mean = float(np.mean(points_xy[:, 1])) if points_xy.size else 0.0
        deg_per_m_lon, deg_per_m_lat = _metres_to_degrees(lat_mean)
        gx = max(grid_size_m * deg_per_m_lon, 1e-12)
        gy = max(grid_size_m * deg_per_m_lat, 1e-12)
        return gx, gy
    return max(grid_size_m, 1e-9), max(grid_size_m, 1e-9)


def _effective_distance(value_m: float, units_mode: str, points_xy: np.ndarray) -> float:
    if units_mode == "degrees":
        lat_mean = float(np.mean(points_xy[:, 1])) if points_xy.size else 0.0
        deg_per_m_lon, deg_per_m_lat = _metres_to_degrees(lat_mean)
        return max(value_m * max(deg_per_m_lon, deg_per_m_lat), 1e-12)
    return max(value_m, 1e-9)




def _lonlat_to_local_metric(points_xyz: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    lat0_rad = np.radians(lat0)
    cos_lat0 = np.clip(np.cos(lat0_rad), 1e-6, None)
    x_m = (points_xyz[:, 0] - lon0) * cos_lat0 * 111320.0
    y_m = (points_xyz[:, 1] - lat0) * 111320.0
    z_m = points_xyz[:, 2]
    return np.column_stack((x_m, y_m, z_m))


def _local_metric_to_lonlat(points_xyz_m: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    lat0_rad = np.radians(lat0)
    cos_lat0 = np.clip(np.cos(lat0_rad), 1e-6, None)
    lon = (points_xyz_m[:, 0] / (cos_lat0 * 111320.0)) + lon0
    lat = (points_xyz_m[:, 1] / 111320.0) + lat0
    z = points_xyz_m[:, 2]
    return np.column_stack((lon, lat, z))


def _rotation_angle_deg(transform: np.ndarray) -> float:
    rot = transform[:3, :3]
    trace_val = np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace_val)))

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
    if len(points) == 0:
        return np.empty((0, 3)), np.array([], dtype=np.int64), {
            "units_mode": "unknown",
            "effective_grid_x": float(getattr(config, "icp_ground_grid_size", 1.0)),
            "effective_grid_y": float(getattr(config, "icp_ground_grid_size", 1.0)),
            "fallback_used": False,
        }

    units_mode = _resolve_units_mode(points[:, :2], config)
    grid_size_m = float(getattr(config, "icp_ground_grid_size", 1.0))
    grid_x, grid_y = _effective_xy_scales(grid_size_m, units_mode, points[:, :2])

    if not getattr(config, "icp_use_ground_only", True):
        return points, np.arange(points.shape[0], dtype=np.int64), {
            "units_mode": units_mode,
            "effective_grid_x": grid_x,
            "effective_grid_y": grid_y,
            "fallback_used": False,
        }

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    ix = np.floor((x - x.min()) / grid_x).astype(np.int64)
    iy = np.floor((y - y.min()) / grid_y).astype(np.int64)

    key_to_best = {}
    for idx, key in enumerate(zip(ix, iy)):
        best_idx = key_to_best.get(key)
        if best_idx is None or z[idx] < z[best_idx]:
            key_to_best[key] = idx

    ground_idx = np.asarray(list(key_to_best.values()), dtype=np.int64)
    if ground_idx.size == 0:
        ground_idx = np.array([], dtype=np.int64)

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

        fallback_voxel = _effective_distance(float(getattr(config, "icp_voxel_size", 1.0)), units_mode, points[:, :2])
        keep_local = _voxel_downsample_indices(points[selected_idx], fallback_voxel)
        selected_idx = selected_idx[keep_local]

    cap = int(getattr(config, "icp_ground_max_points", 5_000_000))
    if selected_idx.size > cap:
        rng = np.random.default_rng(42)
        selected_idx = rng.choice(selected_idx, size=cap, replace=False)

    metadata = {
        "units_mode": units_mode,
        "effective_grid_x": grid_x,
        "effective_grid_y": grid_y,
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


def run_icp(source_points: np.ndarray, target_points: np.ndarray, config) -> tuple[np.ndarray, float, float, list[dict]]:
    """Run rigid ICP and capture per-iteration metrics."""
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


def apply_transformation_to_las(input_path: str, transform_matrix: np.ndarray, output_path: str, local_metric_reference: dict | None = None) -> None:
    """Apply rigid transform to XYZ while preserving all LAS attributes and CRS/header."""
    las = laspy.read(input_path)
    xyz = np.column_stack((np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)))

    if local_metric_reference and local_metric_reference.get("mode") == "degrees":
        lon0 = float(local_metric_reference["lon0"])
        lat0 = float(local_metric_reference["lat0"])
        xyz_local = _lonlat_to_local_metric(xyz, lon0=lon0, lat0=lat0)
        xyz_h = np.hstack((xyz_local, np.ones((xyz_local.shape[0], 1), dtype=xyz_local.dtype)))
        transformed_local = (transform_matrix @ xyz_h.T).T[:, :3]
        transformed = _local_metric_to_lonlat(transformed_local, lon0=lon0, lat0=lat0)
    else:
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


def align_strips_incremental(strip_paths: List[str], config) -> List[str]:
    """Incrementally align strips using ICP; fallback to passthrough on errors/low overlap."""
    if not strip_paths:
        return []

    aligned_dir, inter_dir, log_dir = _ensure_dirs(config)
    logger = _build_logger(config)
    aligned_paths: List[str] = []

    try:
        save_icp_log(logger, f"AOI={_get_aoi_name(config)} | strips={len(strip_paths)}")

        first_out = os.path.join(aligned_dir, "strip_01_aligned.laz")
        shutil.copy2(strip_paths[0], first_out)
        aligned_paths.append(first_out)

        for idx in range(1, len(strip_paths)):
            src_original = strip_paths[idx]
            tgt_aligned_prev = aligned_paths[idx - 1]
            src_num = idx + 1
            tgt_num = idx
            src_out = os.path.join(aligned_dir, f"strip_{src_num:02d}_aligned.laz")

            try:
                src_points, src_idx, src_meta = _read_selected_points_with_indices(src_original, config)
                tgt_points, tgt_idx, tgt_meta = _read_selected_points_with_indices(tgt_aligned_prev, config)

                pair_units = src_meta["units_mode"] if src_meta["units_mode"] != "unknown" else tgt_meta["units_mode"]
                combined_xy = np.vstack((src_points[:, :2], tgt_points[:, :2])) if len(src_points) and len(tgt_points) else np.empty((0, 2))
                if pair_units == "degrees":
                    eff_voxel = float(getattr(config, "icp_voxel_size", 1.0))
                    eff_corr = float(getattr(config, "icp_max_correspondence_distance", 2.0))
                else:
                    eff_voxel = _effective_distance(float(getattr(config, "icp_voxel_size", 1.0)), pair_units, combined_xy)
                    eff_corr = _effective_distance(float(getattr(config, "icp_max_correspondence_distance", 2.0)), pair_units, combined_xy)

                save_icp_log(
                    logger,
                    f"Pair strip{src_num:02d}->{tgt_num:02d} | source={os.path.basename(src_original)} | target={os.path.basename(tgt_aligned_prev)}",
                )
                save_icp_log(logger, f"Units mode={pair_units}")
                save_icp_log(
                    logger,
                    f"Effective grid source=({src_meta['effective_grid_x']:.12f}, {src_meta['effective_grid_y']:.12f}) target=({tgt_meta['effective_grid_x']:.12f}, {tgt_meta['effective_grid_y']:.12f})",
                )
                save_icp_log(logger, f"Effective voxel_size={eff_voxel:.12f} max_corr_distance={eff_corr:.12f}")
                save_icp_log(logger, f"Selected points source={len(src_points)}, target={len(tgt_points)}")

                if src_meta.get("fallback_used"):
                    save_icp_log(logger, f"Source fallback triggered: selected<{int(getattr(config, 'icp_min_selected_points', 50000))}, switched to full-cloud + voxel downsample", "warning")
                if tgt_meta.get("fallback_used"):
                    save_icp_log(logger, f"Target fallback triggered: selected<{int(getattr(config, 'icp_min_selected_points', 50000))}, switched to full-cloud + voxel downsample", "warning")

                if len(src_points) == 0 or len(tgt_points) == 0:
                    save_icp_log(logger, "Selection produced empty points; passthrough.", "warning")
                    shutil.copy2(src_original, src_out)
                    aligned_paths.append(src_out)
                    continue

                if len(src_points) < 1000 or len(tgt_points) < 1000:
                    save_icp_log(logger, f"Insufficient selected points for ICP (source={len(src_points)}, target={len(tgt_points)}); passthrough.", "warning")
                    shutil.copy2(src_original, src_out)
                    aligned_paths.append(src_out)
                    continue

                smin, smax = src_points[:, :2].min(axis=0), src_points[:, :2].max(axis=0)
                tmin, tmax = tgt_points[:, :2].min(axis=0), tgt_points[:, :2].max(axis=0)
                save_icp_log(logger, f"Source XY bbox min={smin.tolist()} max={smax.tolist()}")
                save_icp_log(logger, f"Target XY bbox min={tmin.tolist()} max={tmax.tolist()}")

                src_overlap, tgt_overlap = extract_overlap_area(src_points, tgt_points)
                save_icp_log(logger, f"Overlap counts source={len(src_overlap)} target={len(tgt_overlap)}")

                min_overlap = int(getattr(config, "icp_min_overlap_points", 10000))
                if min(len(src_overlap), len(tgt_overlap)) < min_overlap:
                    save_icp_log(
                        logger,
                        f"Not enough overlap points (<{min_overlap}); passthrough for strip {src_num:02d}.",
                        "warning",
                    )
                    shutil.copy2(src_original, src_out)
                    aligned_paths.append(src_out)
                    continue

                local_metric_reference = None
                if pair_units == "degrees":
                    lon0 = float(np.mean(tgt_overlap[:, 0]))
                    lat0 = float(np.mean(tgt_overlap[:, 1]))
                    src_for_icp = _lonlat_to_local_metric(src_overlap, lon0=lon0, lat0=lat0)
                    tgt_for_icp = _lonlat_to_local_metric(tgt_overlap, lon0=lon0, lat0=lat0)
                    local_metric_reference = {"mode": "degrees", "lon0": lon0, "lat0": lat0}
                    save_icp_log(logger, f"Local metric frame origin lon0={lon0:.12f}, lat0={lat0:.12f}")
                else:
                    src_for_icp = src_overlap
                    tgt_for_icp = tgt_overlap

                original_voxel = getattr(config, "icp_voxel_size", 1.0)
                original_corr = getattr(config, "icp_max_correspondence_distance", 2.0)
                config.icp_voxel_size = eff_voxel
                config.icp_max_correspondence_distance = eff_corr
                try:
                    transform, fitness, rmse, details = run_icp(src_for_icp, tgt_for_icp, config)
                finally:
                    config.icp_voxel_size = original_voxel
                    config.icp_max_correspondence_distance = original_corr

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

                if pair_units == "degrees":
                    lat0_for_log = local_metric_reference["lat0"] if local_metric_reference else float(np.mean(tgt_overlap[:, 1]))
                    lat0_rad = np.radians(lat0_for_log)
                    cos_lat0 = np.clip(np.cos(lat0_rad), 1e-6, None)
                    dlon = transform[0, 3] / (cos_lat0 * 111320.0)
                    dlat = transform[1, 3] / 111320.0
                    save_icp_log(logger, f"Translation approx degrees dlon={dlon:.12f}, dlat={dlat:.12f}, dz={transform[2, 3]:.6f}")

                if reject_reasons:
                    save_icp_log(logger, f"Transform rejected: {'; '.join(reject_reasons)}. Passthrough unchanged.", "warning")
                    shutil.copy2(src_original, src_out)
                    aligned_paths.append(src_out)
                    continue

                save_icp_log(logger, "Transform accepted and applied.")
                apply_transformation_to_las(src_original, transform, src_out, local_metric_reference=local_metric_reference)
                aligned_paths.append(src_out)

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

                    src_ov_mask = (
                        (src_points[:, 0] >= max(smin[0], tmin[0]))
                        & (src_points[:, 0] <= min(smax[0], tmax[0]))
                        & (src_points[:, 1] >= max(smin[1], tmin[1]))
                        & (src_points[:, 1] <= min(smax[1], tmax[1]))
                    )
                    tgt_ov_mask = (
                        (tgt_points[:, 0] >= max(smin[0], tmin[0]))
                        & (tgt_points[:, 0] <= min(smax[0], tmax[0]))
                        & (tgt_points[:, 1] >= max(smin[1], tmin[1]))
                        & (tgt_points[:, 1] <= min(smax[1], tmax[1]))
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

    finally:
        _close_logger(logger)

    return aligned_paths
