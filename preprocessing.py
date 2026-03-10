import os
import time
import json
import shutil
import re
from datetime import timedelta
from datetime import datetime
from multiprocessing import Pool
import numpy as np

import pdal
import laspy
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
import geopandas as gpd
from tqdm import tqdm
from shapely.geometry import shape
from shapely.geometry import box
from shapely.wkt import loads as wkt_loads, dumps as wkt_dumps

from core.reprojection import get_utm_epsg, reproject_las, is_utm_crs
from core.preprocess_windowed import (
    create_chunks_from_wkt,
    process_chunk,
    process_chunk_wrapper,
    merge_and_crop_chunks,
    merge_chunks_to_strip,
    preprocess_window,
    has_points,
    get_las_bounds_wkt,
    get_las_header,
    process_chunk_with_provenance,
    group_chunks_by_strip,
    build_strip_chunk_tasks,
)
from core.extract_footprints import extract_footprint_batch
from core.utils import split_gpkg


def _safe_json(obj):
    """Convert numpy-heavy dictionaries to JSON-serialisable python values."""
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_safe_json(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _las_bounds_polygon(las_path):
    with laspy.open(las_path) as f:
        hdr = f.header
        return box(hdr.mins[0], hdr.mins[1], hdr.maxs[0], hdr.maxs[1]), int(hdr.point_count)


def _extract_timestamp(strip_path):
    """Best-effort timestamp extraction for chronological strip sorting."""
    file_name = os.path.basename(strip_path)
    match = re.search(r"(\d{8}T\d{6})", file_name)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")
        except ValueError:
            pass

    with laspy.open(strip_path) as f:
        date = f.header.creation_date
    if date:
        return datetime.combine(date, datetime.min.time())

    return datetime.utcfromtimestamp(os.path.getmtime(strip_path))


def _read_las_points(las_path):
    las = laspy.read(las_path)
    xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
    cls = np.asarray(las.classification) if hasattr(las, "classification") else None
    return las, xyz, cls


def _classify_ground_smrf(temp_input, temp_output, cfg):
    pipeline = {
        "pipeline": [
            {"type": "readers.las", "filename": temp_input},
            {
                "type": "filters.smrf",
                "window": float(getattr(cfg, "smrf_window_size", 20.0)),
                "slope": float(getattr(cfg, "smrf_slope", 0.2)),
                "scalar": float(getattr(cfg, "smrf_scalar", 2.0)),
                "threshold": float(getattr(cfg, "threshold", 0.5)),
            },
            {"type": "writers.las", "filename": temp_output, "minor_version": 4, "dataformat_id": 6},
        ]
    }
    pipe = pdal.Pipeline(json.dumps(pipeline))
    pipe.execute()


def _prepare_icp_points(xyz, cls, cfg, temp_dir, prefix):
    """
    Build ICP-ready points from overlap cloud.
    Prioritises ground-only when possible, with graceful fallback.
    """
    warnings = []
    used_ground_only = False
    used_fallback = False

    if len(xyz) == 0:
        return xyz, {"used_ground_only": False, "used_fallback": True, "warnings": ["empty input"]}

    points = xyz
    if getattr(cfg, "icp_use_ground_only", True):
        ground_mask = None

        if cls is not None and len(cls) == len(xyz):
            ground_mask = cls == 2
        else:
            try:
                temp_in = os.path.join(temp_dir, f"{prefix}_overlap_in.laz")
                temp_out = os.path.join(temp_dir, f"{prefix}_overlap_smrf.laz")
                las = laspy.create(file_version="1.4", point_format=6)
                las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
                las.classification = np.zeros(len(xyz), dtype=np.uint8)
                las.write(temp_in)
                _classify_ground_smrf(temp_in, temp_out, cfg)
                smrf_las = laspy.read(temp_out)
                ground_mask = np.asarray(smrf_las.classification) == 2
            except Exception as exc:
                warnings.append(f"SMRF classification failed: {exc}")

        if ground_mask is not None:
            ground_points = xyz[ground_mask]
            min_ground = int(getattr(cfg, "icp_min_ground_points", 800))
            if len(ground_points) >= min_ground:
                points = ground_points
                used_ground_only = True
            else:
                used_fallback = True
                warnings.append(f"Too few ground points ({len(ground_points)}<{min_ground}), fallback to non-ground")
        else:
            used_fallback = True

    voxel = float(getattr(cfg, "icp_voxel_size", 1.0))
    if voxel > 0 and len(points) > 0:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcd = pcd.voxel_down_sample(voxel)
        points = np.asarray(pcd.points)

    return points, {
        "used_ground_only": used_ground_only,
        "used_fallback": used_fallback,
        "warnings": warnings,
    }


def _run_pair_icp(source_pts, target_pts, cfg):
    local_origin = np.mean(np.vstack([source_pts, target_pts]), axis=0)
    src_local = source_pts - local_origin
    tgt_local = target_pts - local_origin

    src_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_local))
    tgt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt_local))

    normal_radius = max(float(getattr(cfg, "icp_voxel_size", 1.0)) * 2.5, 0.25)
    src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
    tgt_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))

    reg = o3d.pipelines.registration.registration_icp(
        src_pcd,
        tgt_pcd,
        float(getattr(cfg, "icp_max_correspondence_distance", 2.0)),
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(getattr(cfg, "icp_max_iterations", 80))),
    )

    t_local = reg.transformation
    t_global = np.eye(4)
    t_global[:3, 3] = local_origin
    t_global = t_global @ t_local
    t_back = np.eye(4)
    t_back[:3, 3] = -local_origin
    t_global = t_global @ t_back

    return reg, t_global, local_origin


def _rotation_angle_deg(transform):
    rot = transform[:3, :3]
    trace = np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def _apply_transform_to_strip(input_strip, output_strip, transform):
    m = transform.flatten()
    matrix = " ".join([f"{v:.12g}" for v in m])
    pipeline = {
        "pipeline": [
            {"type": "readers.las", "filename": input_strip},
            {"type": "filters.transformation", "matrix": matrix},
            {"type": "writers.las", "filename": output_strip, "compression": "laszip"},
        ]
    }
    pdal.Pipeline(json.dumps(pipeline)).execute()


def align_strips_incremental_icp(processed_strip_files, target_fp, config):
    """Sequential strip alignment: first strip fixed, each next strip aligned to already aligned strips."""
    if len(processed_strip_files) < 2 or not getattr(config, "enable_icp", True):
        return processed_strip_files

    ordered = sorted(processed_strip_files, key=_extract_timestamp)
    aoi_name = os.path.splitext(str(target_fp))[0]
    results_root = os.path.join(config.results_dir, aoi_name, config.run_name)
    logs_dir = os.path.join(results_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    run_log_path = os.path.join(logs_dir, "icp_alignment_log.jsonl")
    run_summary_path = os.path.join(logs_dir, "icp_run_summary.json")

    aligned_dir = os.path.join(os.path.dirname(ordered[0]), "aligned_strips")
    temp_icp_dir = os.path.join(os.path.dirname(ordered[0]), "icp_temp")
    os.makedirs(aligned_dir, exist_ok=True)
    os.makedirs(temp_icp_dir, exist_ok=True)

    aligned_outputs = []
    fixed_first = os.path.join(aligned_dir, os.path.basename(ordered[0]))
    shutil.copy2(ordered[0], fixed_first)
    aligned_outputs.append(fixed_first)

    summary = {"total_pairs": 0, "attempted": 0, "accepted": 0, "rejected": 0, "skipped": 0}

    with open(run_log_path, "a", encoding="utf-8") as run_log:
        for idx, source in enumerate(ordered[1:], start=1):
            start_pair = time.time()
            target = aligned_outputs[-1] if len(aligned_outputs) == 1 else "MERGED_ALIGNED_SET"
            pair_id = f"{idx:03d}_{os.path.basename(source)}"

            src_las, src_xyz, src_cls = _read_las_points(source)
            target_xyz_parts = []
            for aligned_target in aligned_outputs:
                _, t_xyz, _ = _read_las_points(aligned_target)
                target_xyz_parts.append(t_xyz)
            tgt_xyz = np.vstack(target_xyz_parts) if target_xyz_parts else np.empty((0, 3))

            src_poly, src_count = _las_bounds_polygon(source)
            tgt_poly = box(np.min(tgt_xyz[:, 0]), np.min(tgt_xyz[:, 1]), np.max(tgt_xyz[:, 0]), np.max(tgt_xyz[:, 1]))
            overlap = src_poly.intersection(tgt_poly).buffer(float(getattr(config, "icp_overlap_buffer", 0.0)))

            pair_log = {
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "aoi": aoi_name,
                "run_name": config.run_name,
                "pair_id": pair_id,
                "source_strip": source,
                "target_strip": target,
                "source_full_point_count": int(src_count),
                "target_full_point_count": int(len(tgt_xyz)),
                "overlap_method": "bbox_intersection",
                "icp_parameters": {
                    "method": getattr(config, "icp_method", "point_to_plane"),
                    "voxel_size": float(getattr(config, "icp_voxel_size", 1.0)),
                    "max_correspondence_distance": float(getattr(config, "icp_max_correspondence_distance", 2.0)),
                    "max_iterations": int(getattr(config, "icp_max_iterations", 80)),
                },
                "qc_gating_enabled": bool(getattr(config, "icp_qc_enabled", True)),
                "icp_attempted": False,
                "transform_estimated": False,
                "qc_passed": False,
                "would_be_rejected": False,
                "transform_applied": False,
                "reject_reason": None,
                "warnings_or_errors": [],
            }
            summary["total_pairs"] += 1

            if overlap.is_empty:
                pair_log["reject_reason"] = "overlap too small"
                pair_log["would_be_rejected"] = True
                summary["skipped"] += 1
            else:
                minx, miny, maxx, maxy = overlap.bounds
                src_mask = (src_xyz[:, 0] >= minx) & (src_xyz[:, 0] <= maxx) & (src_xyz[:, 1] >= miny) & (src_xyz[:, 1] <= maxy)
                tgt_mask = (tgt_xyz[:, 0] >= minx) & (tgt_xyz[:, 0] <= maxx) & (tgt_xyz[:, 1] >= miny) & (tgt_xyz[:, 1] <= maxy)
                src_ov = src_xyz[src_mask]
                tgt_ov = tgt_xyz[tgt_mask]

                pair_log["overlap_point_count_source"] = int(len(src_ov))
                pair_log["overlap_point_count_target"] = int(len(tgt_ov))
                pair_log["overlap_fraction_source"] = float(len(src_ov) / max(len(src_xyz), 1))
                pair_log["overlap_fraction_target"] = float(len(tgt_ov) / max(len(tgt_xyz), 1))

                min_overlap_points = int(getattr(config, "icp_min_overlap_points", 2500))
                if len(src_ov) < min_overlap_points or len(tgt_ov) < min_overlap_points:
                    pair_log["reject_reason"] = "too few overlap points"
                    pair_log["would_be_rejected"] = True
                    summary["skipped"] += 1
                else:
                    src_icp, src_meta = _prepare_icp_points(src_ov, src_cls[src_mask] if src_cls is not None else None, config, temp_icp_dir, f"{pair_id}_src")
                    tgt_icp, tgt_meta = _prepare_icp_points(tgt_ov, None, config, temp_icp_dir, f"{pair_id}_tgt")
                    pair_log["ground_only_mode_used"] = bool(src_meta["used_ground_only"] and tgt_meta["used_ground_only"])
                    pair_log["fallback_mode_used"] = bool(src_meta["used_fallback"] or tgt_meta["used_fallback"])
                    pair_log["warnings_or_errors"].extend(src_meta["warnings"] + tgt_meta["warnings"])
                    pair_log["pre_icp_qc_statistics"] = {
                        "source_icp_points": int(len(src_icp)),
                        "target_icp_points": int(len(tgt_icp)),
                    }

                    if len(src_icp) < min_overlap_points or len(tgt_icp) < min_overlap_points:
                        pair_log["reject_reason"] = "too few ground points"
                        pair_log["would_be_rejected"] = True
                        summary["skipped"] += 1
                    else:
                        pair_log["icp_attempted"] = True
                        summary["attempted"] += 1
                        try:
                            reg, transform, origin = _run_pair_icp(src_icp, tgt_icp, config)
                            pair_log["transform_estimated"] = True
                            pair_log["local_origin_used_for_icp"] = origin.tolist()
                            pair_log["fitness"] = float(reg.fitness)
                            pair_log["rmse"] = float(reg.inlier_rmse)
                            pair_log["transformation_matrix"] = transform.tolist()

                            tv = transform[:3, 3]
                            rot_deg = _rotation_angle_deg(transform)
                            horizontal = float(np.linalg.norm(tv[:2]))
                            vertical = float(abs(tv[2]))
                            pair_log["translation_vector"] = tv.tolist()
                            pair_log["translation_norm"] = float(np.linalg.norm(tv))
                            pair_log["rotation_angle_deg"] = rot_deg
                            pair_log["effective_horizontal_shift"] = horizontal
                            pair_log["effective_vertical_shift"] = vertical

                            if bool(getattr(config, "icp_qc_enabled", True)):
                                qc_ok = (
                                    pair_log["translation_norm"] <= float(getattr(config, "icp_max_translation_norm", 5.0))
                                    and horizontal <= float(getattr(config, "icp_max_horizontal_shift", 4.0))
                                    and vertical <= float(getattr(config, "icp_max_vertical_shift", 1.5))
                                    and rot_deg <= float(getattr(config, "icp_max_rotation_deg", 3.0))
                                    and pair_log["fitness"] >= float(getattr(config, "icp_min_fitness", 0.25))
                                    and pair_log["rmse"] <= float(getattr(config, "icp_max_rmse", 1.25))
                                )
                            else:
                                qc_ok = True

                            pair_log["qc_passed"] = bool(qc_ok)
                            pair_log["would_be_rejected"] = not qc_ok
                            pair_log["post_icp_qc_statistics"] = {
                                "fitness": pair_log["fitness"],
                                "rmse": pair_log["rmse"],
                                "translation_norm": pair_log["translation_norm"],
                                "rotation_angle_deg": pair_log["rotation_angle_deg"],
                            }

                            out_name = os.path.basename(source).replace(".laz", "_aligned.laz").replace(".las", "_aligned.las")
                            aligned_source = os.path.join(aligned_dir, out_name)
                            if qc_ok:
                                _apply_transform_to_strip(source, aligned_source, transform)
                                pair_log["transform_applied"] = True
                                summary["accepted"] += 1
                            else:
                                pair_log["reject_reason"] = "low fitness" if pair_log["fitness"] < float(getattr(config, "icp_min_fitness", 0.25)) else "high RMSE"
                                shutil.copy2(source, aligned_source)
                                summary["rejected"] += 1

                            aligned_outputs.append(aligned_source)
                        except Exception as exc:
                            pair_log["reject_reason"] = f"ICP failure: {exc}"
                            pair_log["warnings_or_errors"].append(str(exc))
                            summary["rejected"] += 1
                            passthrough = os.path.join(aligned_dir, os.path.basename(source))
                            shutil.copy2(source, passthrough)
                            aligned_outputs.append(passthrough)

            pair_log["runtime_seconds"] = float(time.time() - start_pair)
            run_log.write(json.dumps(_safe_json(pair_log)) + "\n")

    with open(run_summary_path, "w", encoding="utf-8") as f:
        json.dump(_safe_json(summary), f, indent=2)

    shutil.rmtree(temp_icp_dir, ignore_errors=True)
    return aligned_outputs


def get_las_header(las_file):
    with laspy.open(las_file) as las:
        header = las.header
        scale = header.scales
        offset = header.offsets
        crs = header.parse_crs()
        crs_epsg = crs.to_epsg() if crs else 4979
    return scale, offset, crs_epsg


def process_chunk_wrapper(args):
    return process_chunk(*args)

def plot_target_and_footprints(target_gdf, matched_las_paths, las_footprint_dir, output_path):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot target area in red
    target_gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=2, label='Target Area')

    # Overlay LAS footprints in blue
    for las_path in matched_las_paths:
        las_name = os.path.splitext(os.path.basename(las_path))[0]
        las_fp_path = os.path.join(las_footprint_dir, las_name + ".gpkg")
        if os.path.exists(las_fp_path):
            las_gdf = gpd.read_file(las_fp_path)
            if las_gdf.crs != target_gdf.crs:
                las_gdf = las_gdf.to_crs(target_gdf.crs)
            las_gdf.plot(ax=ax, facecolor='blue', edgecolor='blue', alpha=0.3, label='Matched LAS Footprint')

    plt.title('Target Area and Matched LAS Footprints')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def match_footprints(target_footprint_dir, las_footprint_dir, las_file_dir, out_dir, threshold=0.5, filter_date=True, start_date=None, end_date=None):
    os.makedirs(las_footprint_dir, exist_ok=True)

    print("\nMatching Lidar footprints...")
    start = time.time()

    if not os.listdir(las_footprint_dir):
        print("No footprint files found. Generating footprints first.")
        extract_footprint_batch(las_file_dir, las_footprint_dir)

    target_footprints = [
        os.path.join(target_footprint_dir, f)
        for f in os.listdir(target_footprint_dir) if f.endswith(".gpkg")
    ]
    las_footprints = [
        os.path.join(las_footprint_dir, f)
        for f in os.listdir(las_footprint_dir) if f.endswith(".gpkg")
    ]

    target_dict = {}

    for target_fp in tqdm(target_footprints, desc="Finding target areas", unit="areas"):
        target_gdf = gpd.read_file(target_fp)
        target_name = os.path.splitext(os.path.basename(target_fp))[0]
        las_paths = []

        for las_fp in tqdm(las_footprints, desc="Checking LAS footprints", unit="footprints"):
            las_gdf = gpd.read_file(las_fp)
            if target_gdf.crs != las_gdf.crs:
                las_gdf = las_gdf.to_crs(target_gdf.crs)

            joined = gpd.sjoin(las_gdf, target_gdf, predicate="intersects")
            if not joined.empty:
                intersection = gpd.overlay(las_gdf, target_gdf, how="intersection")
                intersection_area = intersection.area.sum()
                target_area = target_gdf.geometry.area.sum()

                if intersection_area / target_area > threshold:
                    las_name = os.path.splitext(os.path.basename(las_fp))[0]
                    # Check for both .las and .laz files
                    las_path = os.path.join(las_file_dir, las_name + ".las")
                    laz_path = os.path.join(las_file_dir, las_name + ".laz")
                    
                    if os.path.exists(las_path):
                        las_path = las_path
                    elif os.path.exists(laz_path):
                        las_path = laz_path
                    else:
                        las_path = None

                    if las_path:
                        if filter_date and (start_date or end_date):

                            if isinstance(start_date, str):
                                start_date = datetime.strptime(start_date, "%Y-%m-%d").date()

                            if isinstance(end_date, str):
                                end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

                            try:
                                with laspy.open(las_path) as las_file:
                                    las_date = las_file.header.creation_date

                                print(f"{las_file} Creation date: {las_date}")
                                if las_date:
                                    if start_date and las_date < start_date:
                                        continue
                                    if end_date and las_date > end_date:
                                        continue
                                else:
                                    continue  # Skip if no creation date

                            except Exception as e:
                                print(f"Failed to read LAS header from {las_path}: {e}")
                                continue

                        las_paths.append(las_path)

        target_dict[target_name] = las_paths

        if las_paths:
            output_plot_path = os.path.join(out_dir, f"{target_name}_footprints.png")
            plot_target_and_footprints(target_gdf, las_paths, las_footprint_dir, output_plot_path)


        print(f"Target area: {target_name}, LAS files found: {len(las_paths)}")


    print(f"Footprint matching completed in {timedelta(seconds=int(time.time() - start))}. Found {len(target_dict)} target areas.")
    return target_dict


def merge_and_clean_las(
    las_dict,
    preprocessed_dir,
    run_name,
    target_footprint_dir,
    config,
    num_workers,
    chunk_size=1000,
):
    """
    Main preprocessing entry point.

    Modes:
    - config.preprocess_by_strip = False:
        original chunk mode:
        process all chunks and merge directly to one final AOI output

    - config.preprocess_by_strip = True:
        strip-preserving chunked mode:
        process chunks per strip -> merge to processed strip files -> merge strips to final AOI output
    """
    run_merged_dir = os.path.join(preprocessed_dir, run_name)
    os.makedirs(run_merged_dir, exist_ok=True)

    print("\nProcessing LAS files...")
    start = time.time()
    processed_strips_by_target = {}

    for target_fp, las_files in tqdm(las_dict.items(), desc="Processing target areas", unit="area"):
        if not las_files:
            print(f"No valid LAS files for {target_fp}. Skipping.")
            continue

        clean_target_fp = os.path.splitext(target_fp)[0]
        final_output_file = os.path.join(run_merged_dir, f"{clean_target_fp}.laz")

        if os.path.exists(final_output_file):
            if has_points(final_output_file):
                print(f"Skipping {target_fp}: Already processed.")
                continue
            os.remove(final_output_file)
            print(f"[WARN] Existing output was empty and will be reprocessed: {final_output_file}")

        footprint_path = os.path.join(
            target_footprint_dir,
            target_fp if target_fp.endswith(".gpkg") else f"{target_fp}.gpkg",
        )
        if not os.path.exists(footprint_path):
            print(f"Footprint file {footprint_path} not found. Skipping.")
            continue

        gdf = gpd.read_file(footprint_path)

        temp_dir = os.path.join(run_merged_dir, target_fp, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        target_geom_wkt = wkt_dumps(shape(gdf.geometry.iloc[0]))

        # -------------------------------------------------------------
        # strip-preserving chunked mode
        # -------------------------------------------------------------
        if config.preprocess_by_strip:

            processed_strip_files = preprocess_window(
                strip_files=las_files,
                config=config,
                target_fp=target_fp,
                run_merged_dir=run_merged_dir,
                temp_dir=temp_dir,
                target_gdf=gdf,
                num_workers=num_workers,
                chunk_size=chunk_size,
            )

            if getattr(config, "enable_icp", True) and len(processed_strip_files) > 1:
                print(f"Running sequential ICP strip alignment for {target_fp}...")
                processed_strip_files = align_strips_incremental_icp(
                    processed_strip_files=processed_strip_files,
                    target_fp=target_fp,
                    config=config,
                )

            processed_strips_by_target[target_fp] = processed_strip_files

            if processed_strip_files:
                merge_chunks_to_strip(processed_strip_files, final_output_file)
                if has_points(final_output_file):
                    print(f"Final processed LAS file saved: {final_output_file}")
                else:
                    os.remove(final_output_file)
                    print(f"[WARN] Final processed LAS is empty and was removed: {final_output_file}")
            else:
                print(f"No processed strips available for {target_fp}.")

            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

            target_fp_dir = os.path.join(run_merged_dir, target_fp)
            if os.path.isdir(target_fp_dir) and not os.listdir(target_fp_dir):
                os.rmdir(target_fp_dir)

            continue

        # -------------------------------------------------------------
        # original chunk mode
        # -------------------------------------------------------------
        processed_chunks = []
        process_args = []

        print(f"Input strips for {target_fp}: {len(las_files)}")

        for input_file in las_files:
            strip_input = input_file

            if not is_utm_crs(strip_input):
                base_name = os.path.basename(strip_input)
                base_name = base_name.replace(".las", "_utm.las").replace(".laz", "_utm.laz")
                utm_output_file = os.path.join(temp_dir, base_name)
                strip_input = reproject_las(strip_input, utm_output_file)

            ref_scale, ref_offset, ref_crs = get_las_header(strip_input)

            gdf_proj = gdf.to_crs(epsg=ref_crs) if gdf.crs.to_epsg() != ref_crs else gdf
            target_geom_local = shape(gdf_proj.geometry.iloc[0])
            target_geom_wkt_local = wkt_dumps(target_geom_local)

            strip_geom_wkt = get_las_bounds_wkt(strip_input)
            strip_geom = wkt_loads(strip_geom_wkt)
            process_geom = target_geom_local.intersection(strip_geom)

            if process_geom.is_empty:
                print(f"[WARN] Strip does not intersect AOI and is skipped: {strip_input}")
                continue

            chunks = create_chunks_from_wkt(wkt_dumps(process_geom), chunk_size)
            print(f"Chunks created for strip {os.path.basename(strip_input)}: {len(chunks)}")

            all_z = laspy.read(strip_input).z
            if getattr(config, "max_elevation_threshold", None):
                max_z = np.quantile(all_z, config.max_elevation_threshold)
                min_z = np.quantile(all_z, 1 - config.max_elevation_threshold)
            else:
                max_z = np.max(all_z)
                min_z = np.min(all_z)

            for chunk in chunks:
                process_args.append(
                    (
                        strip_input,
                        chunk,
                        temp_dir,
                        max_z,
                        min_z,
                        config.knn,
                        config.multiplier,
                        ref_scale,
                        ref_offset,
                        ref_crs,
                    )
                )

        with tqdm(total=len(process_args), desc=f"Processing {target_fp}", unit="chunk") as pbar:
            with Pool(processes=num_workers) as pool:
                for processed_chunk in pool.imap_unordered(process_chunk_wrapper, process_args):
                    if processed_chunk and has_points(processed_chunk):
                        processed_chunks.append(processed_chunk)
                    pbar.update(1)

        processed_strips_by_target[target_fp] = []

        if processed_chunks:
            merge_and_crop_chunks(processed_chunks, target_geom_wkt, final_output_file)
            if has_points(final_output_file):
                print(f"Final processed LAS file saved: {final_output_file}")
            else:
                os.remove(final_output_file)
                print(f"[WARN] Final processed LAS is empty and was removed: {final_output_file}")
        else:
            print(f"No processed chunks available for {target_fp}.")

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        target_fp_dir = os.path.join(run_merged_dir, target_fp)
        if os.path.isdir(target_fp_dir) and not os.listdir(target_fp_dir):
            os.rmdir(target_fp_dir)

    print(f"\nProcessing completed in {str(timedelta(seconds=time.time() - start)).split('.')[0]}.")
    return processed_strips_by_target

def preprocess_all(conf):
    global config
    config = conf

    print("\n========== Starting Preprocessing ==========")
    start = time.time()

    run_name = config.run_name

    os.makedirs(os.path.join(config.preprocessed_dir, run_name), exist_ok=True)
    os.makedirs(os.path.join(config.results_dir, run_name), exist_ok=True)

    gdfs = os.listdir(config.target_area_dir)
    for gdf in gdfs:
        gdf_path = os.path.join(config.target_area_dir, gdf)
        gdf_loaded = gpd.read_file(gdf_path)
        if len(gdf_loaded) > 1:
            print("\n--- Target areas are multi-geometry. Splitting into separate files ---")
            for gdf_name in os.listdir(config.target_area_dir):
                split_gpkg(os.path.join(config.target_area_dir, gdf_name), config.target_area_dir, field_name=config.target_name_field)
            break

    
    out_dir = os.path.join(config.preprocessed_dir, config.run_name)

    print("\n--- Matching footprints to LAS files ---")
    target_dict = match_footprints(
        target_footprint_dir=config.target_area_dir,
        las_footprint_dir=config.las_footprints_dir,
        las_file_dir=config.las_files_dir,
        out_dir=out_dir,
        threshold=config.overlap,
        filter_date=config.filter_date,
        start_date=config.start_date,
        end_date=config.end_date
    )

    print("\n--- Merging and Cleaning LAS files ---")
    merge_and_clean_las(
        target_footprint_dir=config.target_area_dir,
        las_dict=target_dict,
        preprocessed_dir=config.preprocessed_dir,
        num_workers=config.num_workers,
        run_name=run_name,
        chunk_size=config.chunk_size,
        config=config,
    )

    print(f"\nPreprocessing completed in {str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n")
