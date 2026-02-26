import logging
import os
import traceback
from typing import List, Tuple

import laspy
import numpy as np


LOGGER_NAME = "lidar.icp_alignment"


def _get_aoi_name(config) -> str:
    return getattr(config, "icp_current_aoi", "unknown_aoi")


def _get_output_dirs(config) -> Tuple[str, str, str]:
    aoi_name = _get_aoi_name(config)
    aligned_dir = os.path.join(config.preprocessed_dir, "aligned_strips", aoi_name)
    intermediate_dir = os.path.join(config.preprocessed_dir, "icp_intermediate", aoi_name)
    log_dir = os.path.join(config.results_dir, "icp_logs", aoi_name)
    os.makedirs(aligned_dir, exist_ok=True)
    os.makedirs(intermediate_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return aligned_dir, intermediate_dir, log_dir


def _setup_logger(config) -> logging.Logger:
    _, _, log_dir = _get_output_dirs(config)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_path = os.path.join(log_dir, "icp_debug.log")
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == log_path for h in logger.handlers):
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)

    return logger


def save_icp_log(config, message: str, level: str = "info") -> None:
    if not getattr(config, "icp_save_logs", True):
        return
    logger = _setup_logger(config)
    getattr(logger, level.lower(), logger.info)(message)


def save_icp_iteration_details(config, pair_label: str, iteration_rows: List[str]) -> None:
    if not getattr(config, "icp_save_logs", True):
        return
    _, _, log_dir = _get_output_dirs(config)
    out_path = os.path.join(log_dir, f"icp_iterations_{pair_label}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(iteration_rows))


def save_intermediate_pointcloud(points: np.ndarray, reference_las_path: str, output_path: str) -> None:
    if points.size == 0:
        return

    src = laspy.read(reference_las_path)
    xyz = np.vstack((src.x, src.y, src.z)).T
    rounded_ref = np.round(xyz, 4)
    rounded_pts = np.round(points, 4)
    keep = np.isin(rounded_ref.view([("x", "f8"), ("y", "f8"), ("z", "f8")]), rounded_pts.view([("x", "f8"), ("y", "f8"), ("z", "f8")]))
    keep = keep.reshape(-1)

    subset = src.points[keep]
    out_las = laspy.LasData(src.header)
    out_las.points = subset
    out_las.write(output_path)


def filter_ground_points(las_path):
    las = laspy.read(las_path)
    xyz = np.vstack((las.x, las.y, las.z)).T

    if hasattr(las, "classification"):
        mask = las.classification == 2
        if np.any(mask):
            return xyz[mask]

    return xyz


def extract_overlap_area(strip_a, strip_b):
    las_a = laspy.read(strip_a)
    las_b = laspy.read(strip_b)

    xyz_a = np.vstack((las_a.x, las_a.y, las_a.z)).T
    xyz_b = np.vstack((las_b.x, las_b.y, las_b.z)).T

    min_ax, min_ay, max_ax, max_ay = las_a.header.min[0], las_a.header.min[1], las_a.header.max[0], las_a.header.max[1]
    min_bx, min_by, max_bx, max_by = las_b.header.min[0], las_b.header.min[1], las_b.header.max[0], las_b.header.max[1]

    int_min_x = max(min_ax, min_bx)
    int_min_y = max(min_ay, min_by)
    int_max_x = min(max_ax, max_bx)
    int_max_y = min(max_ay, max_by)

    if int_min_x >= int_max_x or int_min_y >= int_max_y:
        return np.empty((0, 3)), np.empty((0, 3))

    mask_a = (
        (xyz_a[:, 0] >= int_min_x) & (xyz_a[:, 0] <= int_max_x) &
        (xyz_a[:, 1] >= int_min_y) & (xyz_a[:, 1] <= int_max_y)
    )
    mask_b = (
        (xyz_b[:, 0] >= int_min_x) & (xyz_b[:, 0] <= int_max_x) &
        (xyz_b[:, 1] >= int_min_y) & (xyz_b[:, 1] <= int_max_y)
    )

    return xyz_a[mask_a], xyz_b[mask_b]


def run_icp(source_points, target_points, config):
    import open3d as o3d

    src = o3d.geometry.PointCloud()
    tgt = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(source_points)
    tgt.points = o3d.utility.Vector3dVector(target_points)

    voxel = getattr(config, "icp_voxel_size", 1.0)
    if voxel and voxel > 0:
        src = src.voxel_down_sample(voxel)
        tgt = tgt.voxel_down_sample(voxel)

    current_transform = np.eye(4)
    rows = ["iteration\tfitness\trmse\tconverged"]
    previous_rmse = None
    converged = False

    for idx in range(int(getattr(config, "icp_max_iterations", 50))):
        result = o3d.pipelines.registration.registration_icp(
            src,
            tgt,
            getattr(config, "icp_max_correspondence_distance", 2.0),
            current_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1),
        )
        current_transform = result.transformation

        if previous_rmse is not None and abs(previous_rmse - result.inlier_rmse) < 1e-9:
            converged = True
        previous_rmse = result.inlier_rmse
        rows.append(f"{idx + 1}\t{result.fitness:.6f}\t{result.inlier_rmse:.6f}\t{converged}")

    return current_transform, result.fitness, result.inlier_rmse, rows


def apply_transformation_to_las(input_path, transform_matrix, output_path):
    las = laspy.read(input_path)
    xyz = np.vstack((las.x, las.y, las.z)).T

    xyz_h = np.hstack((xyz, np.ones((xyz.shape[0], 1))))
    transformed = (transform_matrix @ xyz_h.T).T[:, :3]

    las.x = transformed[:, 0]
    las.y = transformed[:, 1]
    las.z = transformed[:, 2]

    las.write(output_path)


def align_strips_incremental(strip_paths: list[str], config) -> list[str]:
    if not strip_paths:
        return strip_paths

    aligned_dir, intermediate_dir, _ = _get_output_dirs(config)
    save_icp_log(config, f"Starting incremental ICP alignment for {_get_aoi_name(config)} ({len(strip_paths)} strips).")

    try:
        import open3d  # noqa: F401
    except Exception as exc:
        save_icp_log(config, f"Open3D unavailable. Skipping ICP and using original strips. Error: {exc}", level="warning")
        return strip_paths

    aligned_paths = []
    first_name = os.path.splitext(os.path.basename(strip_paths[0]))[0]
    first_ext = os.path.splitext(strip_paths[0])[1]
    first_out = os.path.join(aligned_dir, f"{first_name}_aligned{first_ext}")
    if not os.path.exists(first_out):
        laspy.read(strip_paths[0]).write(first_out)
    aligned_paths.append(first_out)

    for idx in range(len(strip_paths) - 1):
        ref_path = aligned_paths[-1]
        src_path = strip_paths[idx + 1]

        pair_label = f"strip{idx + 1:02d}_to_strip{idx + 2:02d}"
        save_icp_log(config, f"Running ICP for pair {pair_label}: reference={ref_path}, source={src_path}")

        try:
            source_overlap, target_overlap = extract_overlap_area(src_path, ref_path)

            if getattr(config, "icp_use_ground_only", True):
                src_ground = filter_ground_points(src_path)
                tgt_ground = filter_ground_points(ref_path)

                if getattr(config, "icp_save_intermediate", True):
                    save_intermediate_pointcloud(
                        src_ground,
                        src_path,
                        os.path.join(intermediate_dir, f"ground_filtered_strip_{idx + 2:02d}{os.path.splitext(src_path)[1]}"),
                    )

                source_overlap = source_overlap[np.isin(np.round(source_overlap, 4).view([("x", "f8"), ("y", "f8"), ("z", "f8")]), np.round(src_ground, 4).view([("x", "f8"), ("y", "f8"), ("z", "f8")])).reshape(-1)]
                target_overlap = target_overlap[np.isin(np.round(target_overlap, 4).view([("x", "f8"), ("y", "f8"), ("z", "f8")]), np.round(tgt_ground, 4).view([("x", "f8"), ("y", "f8"), ("z", "f8")])).reshape(-1)]

            overlap_points = min(len(source_overlap), len(target_overlap))
            save_icp_log(config, f"{pair_label}: overlap points source={len(source_overlap)}, target={len(target_overlap)}")

            if getattr(config, "icp_save_intermediate", True):
                ext = os.path.splitext(src_path)[1]
                save_intermediate_pointcloud(source_overlap, src_path, os.path.join(intermediate_dir, f"overlap_source_strip_{idx + 2:02d}{ext}"))
                save_intermediate_pointcloud(target_overlap, ref_path, os.path.join(intermediate_dir, f"overlap_target_strip_{idx + 1:02d}{ext}"))

            if overlap_points < getattr(config, "icp_min_overlap_points", 10000):
                save_icp_log(config, f"{pair_label}: skipped ICP due to insufficient overlap ({overlap_points}).", level="warning")
                passthrough_out = os.path.join(aligned_dir, f"{os.path.splitext(os.path.basename(src_path))[0]}_aligned{os.path.splitext(src_path)[1]}")
                if not os.path.exists(passthrough_out):
                    laspy.read(src_path).write(passthrough_out)
                aligned_paths.append(passthrough_out)
                continue

            transform, fitness, rmse, iteration_rows = run_icp(source_overlap, target_overlap, config)
            save_icp_log(config, f"{pair_label}: fitness={fitness:.6f}, rmse={rmse:.6f}")
            save_icp_log(config, f"{pair_label}: transformation=\n{transform}")
            save_icp_iteration_details(config, pair_label, iteration_rows)

            out_path = os.path.join(aligned_dir, f"{os.path.splitext(os.path.basename(src_path))[0]}_aligned{os.path.splitext(src_path)[1]}")
            apply_transformation_to_las(src_path, transform, out_path)
            aligned_paths.append(out_path)

            if getattr(config, "icp_save_intermediate", True):
                ext = os.path.splitext(src_path)[1]
                aligned_copy = os.path.join(intermediate_dir, f"strip_{idx + 2:02d}_aligned{ext}")
                if not os.path.exists(aligned_copy):
                    laspy.read(out_path).write(aligned_copy)

        except Exception as exc:
            save_icp_log(config, f"{pair_label}: ICP failed with error: {exc}\n{traceback.format_exc()}", level="warning")
            passthrough_out = os.path.join(aligned_dir, f"{os.path.splitext(os.path.basename(src_path))[0]}_aligned{os.path.splitext(src_path)[1]}")
            if not os.path.exists(passthrough_out):
                laspy.read(src_path).write(passthrough_out)
            aligned_paths.append(passthrough_out)

    save_icp_log(config, f"Completed ICP alignment for {_get_aoi_name(config)}. Generated {len(aligned_paths)} aligned strips.")
    return aligned_paths
