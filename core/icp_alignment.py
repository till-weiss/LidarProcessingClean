import logging
import os
import shutil
from typing import Any

import laspy
import numpy as np


def save_icp_log(logger: logging.Logger, message: str, level: str = "info") -> None:
    log_fn = getattr(logger, level, logger.info)
    log_fn(message)


def save_icp_iteration_details(output_path: str, iteration_details: list[dict[str, Any]]) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("iteration\tfitness\trmse\tconverged\n")
        for item in iteration_details:
            f.write(
                f"{item.get('iteration', '')}\t{item.get('fitness', '')}\t"
                f"{item.get('rmse', '')}\t{item.get('converged', '')}\n"
            )


def save_intermediate_pointcloud(points: np.ndarray, output_path: str, template_header: laspy.LasHeader) -> None:
    if points.size == 0:
        return
    header = laspy.LasHeader(point_format=template_header.point_format, version=template_header.version)
    header.scales = template_header.scales
    header.offsets = template_header.offsets
    if template_header.parse_crs() is not None:
        header.add_crs(template_header.parse_crs())
    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.write(output_path)


def filter_ground_points(las_path: str) -> np.ndarray:
    las = laspy.read(las_path)
    if "classification" not in las.point_format.dimension_names:
        return np.column_stack((las.x, las.y, las.z))
    mask = np.array(las.classification) == 2
    return np.column_stack((las.x[mask], las.y[mask], las.z[mask]))


def extract_overlap_area(strip_a: np.ndarray, strip_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if strip_a.size == 0 or strip_b.size == 0:
        return np.empty((0, 3)), np.empty((0, 3))

    min_a, max_a = strip_a[:, :2].min(axis=0), strip_a[:, :2].max(axis=0)
    min_b, max_b = strip_b[:, :2].min(axis=0), strip_b[:, :2].max(axis=0)

    overlap_min = np.maximum(min_a, min_b)
    overlap_max = np.minimum(max_a, max_b)

    if np.any(overlap_min >= overlap_max):
        return np.empty((0, 3)), np.empty((0, 3))

    mask_a = (
        (strip_a[:, 0] >= overlap_min[0])
        & (strip_a[:, 0] <= overlap_max[0])
        & (strip_a[:, 1] >= overlap_min[1])
        & (strip_a[:, 1] <= overlap_max[1])
    )
    mask_b = (
        (strip_b[:, 0] >= overlap_min[0])
        & (strip_b[:, 0] <= overlap_max[0])
        & (strip_b[:, 1] >= overlap_min[1])
        & (strip_b[:, 1] <= overlap_max[1])
    )
    return strip_a[mask_a], strip_b[mask_b]


def run_icp(source_points: np.ndarray, target_points: np.ndarray, config: Any) -> tuple[np.ndarray, float, float, list[dict[str, Any]]]:
    import open3d as o3d

    voxel_size = max(float(getattr(config, "icp_voxel_size", 1.0)), 0.001)
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_points))
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_points))
    source = source.voxel_down_sample(voxel_size)
    target = target.voxel_down_sample(voxel_size)

    icp_result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        float(getattr(config, "icp_max_correspondence_distance", 2.0)),
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=int(getattr(config, "icp_max_iterations", 50))
        ),
    )

    details = [
        {
            "iteration": int(getattr(config, "icp_max_iterations", 50)),
            "fitness": icp_result.fitness,
            "rmse": icp_result.inlier_rmse,
            "converged": icp_result.fitness > 0,
        }
    ]

    return icp_result.transformation, icp_result.fitness, icp_result.inlier_rmse, details


def apply_transformation_to_las(input_path: str, transform_matrix: np.ndarray, output_path: str) -> None:
    las = laspy.read(input_path)
    xyz = np.column_stack((las.x, las.y, las.z))
    xyz_h = np.column_stack((xyz, np.ones(xyz.shape[0])))
    transformed = (transform_matrix @ xyz_h.T).T[:, :3]
    las.x = transformed[:, 0]
    las.y = transformed[:, 1]
    las.z = transformed[:, 2]
    las.write(output_path)


def align_strips_incremental(strip_paths: list[str], config: Any) -> list[str]:
    if not strip_paths:
        return strip_paths

    aoi_name = getattr(config, "current_aoi_name", "unknown_aoi")
    aligned_dir = os.path.join(config.preprocessed_dir, "aligned_strips", aoi_name)
    intermediate_dir = os.path.join(config.preprocessed_dir, "icp_intermediate", aoi_name)
    logs_dir = os.path.join(config.results_dir, "icp_logs", aoi_name)
    os.makedirs(aligned_dir, exist_ok=True)
    os.makedirs(intermediate_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    logger = logging.getLogger(f"icp_{aoi_name}")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    if getattr(config, "icp_save_logs", True):
        file_handler = logging.FileHandler(os.path.join(logs_dir, "icp_debug.log"), mode="w")
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    aligned_paths = []
    first_output = os.path.join(aligned_dir, "strip_01_aligned.laz")
    shutil.copy2(strip_paths[0], first_output)
    aligned_paths.append(first_output)
    save_icp_log(logger, f"Using first strip as reference: {strip_paths[0]}")

    for idx in range(1, len(strip_paths)):
        target_path = aligned_paths[-1]
        source_path = strip_paths[idx]
        output_path = os.path.join(aligned_dir, f"strip_{idx + 1:02d}_aligned.laz")
        iteration_file = os.path.join(logs_dir, f"icp_iterations_strip{idx:02d}_to_strip{idx + 1:02d}.txt")

        try:
            if getattr(config, "icp_use_ground_only", True):
                target_points = filter_ground_points(target_path)
                source_points = filter_ground_points(source_path)
            else:
                target_las = laspy.read(target_path)
                source_las = laspy.read(source_path)
                target_points = np.column_stack((target_las.x, target_las.y, target_las.z))
                source_points = np.column_stack((source_las.x, source_las.y, source_las.z))

            overlap_source, overlap_target = extract_overlap_area(source_points, target_points)
            save_icp_log(logger, f"Strip pair {idx}->{idx + 1} overlap points: src={len(overlap_source)}, tgt={len(overlap_target)}")

            if getattr(config, "icp_save_intermediate", True):
                target_header = laspy.read(target_path).header
                source_header = laspy.read(source_path).header
                save_intermediate_pointcloud(source_points, os.path.join(intermediate_dir, f"ground_filtered_strip_{idx + 1:02d}.laz"), source_header)
                save_intermediate_pointcloud(overlap_source, os.path.join(intermediate_dir, f"overlap_source_strip_{idx:02d}.laz"), source_header)
                save_intermediate_pointcloud(overlap_target, os.path.join(intermediate_dir, f"overlap_target_strip_{idx + 1:02d}.laz"), target_header)

            min_overlap = int(getattr(config, "icp_min_overlap_points", 10000))
            if len(overlap_source) < min_overlap or len(overlap_target) < min_overlap:
                save_icp_log(logger, f"Skipping ICP for strip {idx + 1}: not enough overlap points.", "warning")
                shutil.copy2(source_path, output_path)
                aligned_paths.append(output_path)
                continue

            transform, fitness, rmse, details = run_icp(overlap_source, overlap_target, config)
            apply_transformation_to_las(source_path, transform, output_path)
            aligned_paths.append(output_path)

            save_icp_log(logger, f"Strip {idx + 1} aligned to strip {idx}: fitness={fitness:.6f}, rmse={rmse:.6f}")
            save_icp_log(logger, f"Transformation matrix:\n{transform}")
            save_icp_iteration_details(iteration_file, details)

        except Exception as exc:
            save_icp_log(logger, f"ICP failed for strip {idx + 1}: {exc}", "error")
            shutil.copy2(source_path, output_path)
            aligned_paths.append(output_path)

    for handler in logger.handlers:
        handler.close()
    logger.handlers = []
    return aligned_paths

