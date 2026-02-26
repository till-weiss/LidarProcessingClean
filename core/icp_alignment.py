import logging
import os
import shutil
from typing import List, Tuple

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

    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()

    if getattr(config, "icp_save_logs", True):
        fh = logging.FileHandler(os.path.join(log_dir, "icp_debug.log"), mode="w")
        fh.setFormatter(logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s"))
        logger.addHandler(fh)

    return logger


def _close_logger(logger):
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()


def _as_point_cloud(points: np.ndarray):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def _read_selected_points_with_indices(las_path: str, config) -> Tuple[np.ndarray, np.ndarray]:
    with laspy.open(las_path) as fh:
        las = fh.read()

    x = np.asarray(las.x)
    y = np.asarray(las.y)
    z = np.asarray(las.z)
    points = np.column_stack((x, y, z))

    if len(points) == 0:
        return np.empty((0, 3)), np.array([], dtype=np.int64)

    if not getattr(config, "icp_use_ground_only", True):
        return points, np.arange(points.shape[0], dtype=np.int64)

    cell_size = float(getattr(config, "icp_ground_grid_size", 1.0))
    ix = np.floor((x - x.min()) / cell_size).astype(np.int64)
    iy = np.floor((y - y.min()) / cell_size).astype(np.int64)

    key_to_best = {}
    for idx, key in enumerate(zip(ix, iy)):
        best_idx = key_to_best.get(key)
        if best_idx is None or z[idx] < z[best_idx]:
            key_to_best[key] = idx

    ground_idx = np.fromiter(key_to_best.values(), dtype=np.int64)
    if ground_idx.size == 0:
        return np.empty((0, 3)), np.array([], dtype=np.int64)

    # Lightweight SMRF-like gate: discard points too high above 3x3 neighborhood minimums.
    tol = float(getattr(config, "threshold", 0.5))
    z_map = {k: z[v] for k, v in key_to_best.items()}
    keep = []
    for idx in ground_idx:
        cx, cy = ix[idx], iy[idx]
        local_min = z[idx]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nz = z_map.get((cx + dx, cy + dy))
                if nz is not None and nz < local_min:
                    local_min = nz
        if z[idx] <= local_min + tol:
            keep.append(idx)

    if not keep:
        keep = ground_idx.tolist()

    selected_idx = np.asarray(keep, dtype=np.int64)
    cap = int(getattr(config, "icp_ground_max_points", 5_000_000))
    if selected_idx.size > cap:
        rng = np.random.default_rng(42)
        selected_idx = rng.choice(selected_idx, size=cap, replace=False)

    return points[selected_idx], selected_idx


def select_icp_points(las_path: str, config) -> np.ndarray:
    """Return XYZ points for ICP selection (ground-candidate or full cloud)."""
    selected_points, _ = _read_selected_points_with_indices(las_path, config)
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
    voxel_size = float(getattr(config, "icp_voxel_size", 1.0))
    max_corr = float(getattr(config, "icp_max_correspondence_distance", 2.0))
    max_iter = int(getattr(config, "icp_max_iterations", 50))

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
                src_points, src_idx = _read_selected_points_with_indices(src_original, config)
                tgt_points, tgt_idx = _read_selected_points_with_indices(tgt_aligned_prev, config)

                save_icp_log(
                    logger,
                    f"Pair strip{src_num:02d}->{tgt_num:02d} | source={os.path.basename(src_original)} | target={os.path.basename(tgt_aligned_prev)}",
                )
                save_icp_log(logger, f"Selected points source={len(src_points)}, target={len(tgt_points)}")

                if len(src_points) == 0 or len(tgt_points) == 0:
                    save_icp_log(logger, "Selection produced empty points; passthrough.", "warning")
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

                transform, fitness, rmse, details = run_icp(src_overlap, tgt_overlap, config)
                save_icp_log(logger, f"ICP fitness={fitness:.6f} rmse={rmse:.6f}")
                save_icp_log(logger, f"Transform:\n{transform}")

                apply_transformation_to_las(src_original, transform, src_out)
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
