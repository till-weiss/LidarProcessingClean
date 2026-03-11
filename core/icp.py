import json
import os
import re
import shutil
import time
from datetime import datetime
import warnings

import laspy
import numpy as np
import open3d as o3d
import pdal
from shapely.geometry import box


def extract_start_timestamp(strip_path):
    """Extract acquisition start timestamp from filename, e.g. 20230707T165034."""
    file_name = os.path.basename(strip_path)
    match = re.search(r"(\d{8}T\d{6})", file_name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")
    except ValueError:
        return None

def las_bounds_polygon(las_path):
    with laspy.open(las_path) as f:
        hdr = f.header
        return box(hdr.mins[0], hdr.mins[1], hdr.maxs[0], hdr.maxs[1]), int(hdr.point_count)

def read_las_points(las_path):
    las = laspy.read(las_path)
    xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
    cls = np.asarray(las.classification) if hasattr(las, "classification") else None
    return las, xyz, cls


def classify_ground_smrf(temp_input, temp_output, cfg):
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
            {
                "type": "writers.las",
                "filename": temp_output,
                "minor_version": 4,
                "dataformat_id": 6,
            },
        ]
    }
    pipe = pdal.Pipeline(json.dumps(pipeline))
    pipe.execute()


def write_xyz_cloud(xyz, output_path):
    if xyz is None or len(xyz) == 0:
        return None
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    las = laspy.create(file_version="1.4", point_format=6)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.write(output_path)
    return output_path


def filter_and_voxel(xyz, cls, config, temp_dir, prefix):
    """
    Build ICP-ready points from overlap cloud.
    Prioritises ground-only when possible, with graceful fallback.
    """
    points = xyz

    if getattr(config, "icp_use_ground_only", True):
        min_ground = int(getattr(config, "icp_min_ground_points", 800))

        temp_in = os.path.join(temp_dir, f"{prefix}_overlap_in.laz")
        temp_out = os.path.join(temp_dir, f"{prefix}_overlap_smrf.laz")

        las = laspy.create(file_version="1.4", point_format=6)
        las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        las.classification = np.zeros(len(xyz), dtype=np.uint8)
        las.write(temp_in)

        classify_ground_smrf(temp_in, temp_out, config)

        smrf_las = laspy.read(temp_out)
        smrf_cls = np.asarray(smrf_las.classification)
        ground_points = xyz[smrf_cls == 2]
        ground_point_count = int(len(ground_points))

        if ground_point_count >= min_ground:
            points = ground_points
        else:
            points = xyz

    voxel = float(getattr(config, "icp_voxel_size", 1.0))
    if voxel > 0 and len(points) > 0:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcd = pcd.voxel_down_sample(voxel)
        points = np.asarray(pcd.points)

    return points


def build_icp_ready_strip(input_strip, output_strip, config, temp_dir, prefix):
    _, xyz, cls = read_las_points(input_strip)

    input_point_count = int(len(xyz)) if xyz is not None else 0

    icp_xyz = filter_and_voxel(xyz, cls, config, temp_dir, prefix)

    output_point_count = int(len(icp_xyz)) if icp_xyz is not None else 0

    out_path = write_xyz_cloud(icp_xyz, output_strip)

    meta = {
        "input_strip": input_strip,
        "output_strip": out_path,
        "prefix": prefix,
        "input_point_count": input_point_count,
        "output_point_count": output_point_count,
    }

    return out_path, meta


def _safe_json(obj):
    """Convert numpy-heavy objects to JSON-serialisable Python values."""
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def local_icp_to_global_transform(t_local, local_origin):
    T_to_local = np.eye(4)
    T_to_local[:3, 3] = -local_origin

    T_to_global = np.eye(4)
    T_to_global[:3, 3] = local_origin

    return T_to_global @ t_local @ T_to_local


def append_icp_log(log_file, log_entry):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(_safe_json(log_entry)) + "\n")


def run_pair_icp(
    source_pts,
    target_pts,
    cfg,
    source_file,
    target_file,
    log_file,
    source_icp_ready_file=None,
    target_icp_ready_file=None,
):
    local_origin = np.mean(np.vstack([source_pts, target_pts]), axis=0)
    src_local = source_pts - local_origin
    tgt_local = target_pts - local_origin

    src_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_local))
    tgt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt_local))

    normal_radius = max(float(getattr(cfg, "icp_voxel_size", 1.0)) * 2.5, 0.25)

    src_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=30,
        )
    )
    tgt_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=30,
        )
    )

    reg = o3d.pipelines.registration.registration_icp(
        src_pcd,
        tgt_pcd,
        float(getattr(cfg, "icp_max_correspondence_distance", 2.0)),
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=int(getattr(cfg, "icp_max_iterations", 80))
        ),
    )

    t_local = reg.transformation
    t_global = local_icp_to_global_transform(t_local, local_origin)

    relative_shift_xyz = t_local[:3, 3]
    relative_shift_norm = float(np.linalg.norm(relative_shift_xyz))

    log_entry = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "source_file": source_file,
        "target_file": target_file,
        "source_file_name": os.path.basename(source_file),
        "target_file_name": os.path.basename(target_file),
        "source_icp_ready_file": source_icp_ready_file,
        "target_icp_ready_file": target_icp_ready_file,
        "source_point_count": int(len(source_pts)),
        "target_point_count": int(len(target_pts)),
        "local_origin": local_origin,
        "fitness": float(reg.fitness),
        "inlier_rmse": float(reg.inlier_rmse),
        "transform_local": t_local,
        "transform_global": t_global,
        "relative_shift_xyz": relative_shift_xyz,
        "relative_shift_norm": relative_shift_norm,
        "rotation_matrix_local": t_local[:3, :3],
        "translation_local": t_local[:3, 3],
        "translation_global": t_global[:3, 3],
        "icp_settings": {
            "max_correspondence_distance": float(
                getattr(cfg, "icp_max_correspondence_distance", 2.0)
            ),
            "max_iterations": int(getattr(cfg, "icp_max_iterations", 80)),
            "normal_radius": float(normal_radius),
            "voxel_size": float(getattr(cfg, "icp_voxel_size", 1.0)),
            "method": "point_to_plane",
        },
    }

    append_icp_log(log_file, log_entry)

    return {
        "registration": reg,
        "transform_local": t_local,
        "transform_global": t_global,
        "relative_shift_xyz": relative_shift_xyz,
        "relative_shift_norm": relative_shift_norm,
        "log_entry": log_entry,
    }


def apply_transform_to_strip(input_strip, output_strip, transform):
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
    """Sequential strip alignment: first strip fixed, each next strip aligned to previous aligned strip."""

    if len(processed_strip_files) < 2 or not getattr(config, "enable_icp", True):
        return processed_strip_files

    ordered = sorted(
        [
            f for f in processed_strip_files
            if "_aligned" not in os.path.splitext(os.path.basename(f))[0]
        ],
        key=extract_start_timestamp
    )

    if len(ordered) < 2:
        return processed_strip_files

    print("[ICP] Final strip order before incremental alignment:")
    for idx, strip in enumerate(ordered, start=1):
        print(f"  {idx:02d}. {os.path.basename(strip)}")

    aoi_name = os.path.splitext(str(target_fp))[0]

    preprocessed_aoi_root = os.path.join(config.preprocessed_dir, config.run_name, aoi_name)
    logs_dir = os.path.join(preprocessed_aoi_root, "logs")
    aligned_dir = os.path.join(preprocessed_aoi_root, "aligned_strips")
    icp_ready_dir = os.path.join(preprocessed_aoi_root, "icp_ready_strips")
    temp_icp_dir = os.path.join(preprocessed_aoi_root, "icp_temp")

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(aligned_dir, exist_ok=True)
    os.makedirs(icp_ready_dir, exist_ok=True)
    os.makedirs(temp_icp_dir, exist_ok=True)

    log_file = os.path.join(logs_dir, "icp_log.jsonl")

    icp_ready_original_by_strip = {}

    for strip in ordered:
        strip_base = os.path.splitext(os.path.basename(strip))[0]
        out_icp_ready = os.path.join(icp_ready_dir, f"{strip_base}_icp_ready.laz")

        icp_path, icp_meta = build_icp_ready_strip(
            strip,
            out_icp_ready,
            config,
            temp_icp_dir,
            f"{strip_base}_full",
        )

        if not icp_path or not os.path.exists(icp_path):
            print(f"[ICP] Warning: ICP-ready strip was not saved for {strip}")
            icp_path = None

        icp_ready_original_by_strip[strip] = icp_path

    aligned_outputs = []
    aligned_icp_ready_by_full_strip = {}

    fixed_first = os.path.join(aligned_dir, os.path.basename(ordered[0]))
    if os.path.abspath(ordered[0]) != os.path.abspath(fixed_first):
        shutil.copy2(ordered[0], fixed_first)

    if not os.path.exists(fixed_first):
        print(f"[ICP] Failed to save fixed first strip: {fixed_first}")
        shutil.rmtree(temp_icp_dir, ignore_errors=True)
        return processed_strip_files

    aligned_outputs.append(fixed_first)
    aligned_icp_ready_by_full_strip[fixed_first] = icp_ready_original_by_strip.get(ordered[0])

    for idx, source in enumerate(ordered[1:], start=1):
        start_pair = time.time()

        target_full = aligned_outputs[-1]
        source_name = os.path.splitext(os.path.basename(source))[0]
        target_name = os.path.splitext(os.path.basename(target_full))[0]
        pair_id = f"{source_name}_to_{target_name}"

        _, src_xyz_full, src_cls_full = read_las_points(source)
        _, tgt_xyz_full, _ = read_las_points(target_full)

        source_icp_ready_path = icp_ready_original_by_strip.get(source)
        target_icp_ready_path = aligned_icp_ready_by_full_strip.get(target_full)

        if source_icp_ready_path and os.path.exists(source_icp_ready_path):
            _, src_icp_full, src_icp_cls = read_las_points(source_icp_ready_path)
        else:
            src_icp_full, src_icp_cls = src_xyz_full, src_cls_full

        if target_icp_ready_path and os.path.exists(target_icp_ready_path):
            _, tgt_icp_full, _ = read_las_points(target_icp_ready_path)
        else:
            tgt_icp_full = tgt_xyz_full

        src_poly, src_count = las_bounds_polygon(source)
        tgt_poly = box(
            np.min(tgt_icp_full[:, 0]),
            np.min(tgt_icp_full[:, 1]),
            np.max(tgt_icp_full[:, 0]),
            np.max(tgt_icp_full[:, 1]),
        )

        overlap = src_poly.intersection(tgt_poly).buffer(
            float(getattr(config, "icp_overlap_buffer", 0.0))
        )

        if overlap.is_empty:
            print(f"[ICP] Skipping {pair_id}: overlap too small")

            fallback_out = os.path.join(aligned_dir, os.path.basename(source))
            if os.path.abspath(source) != os.path.abspath(fallback_out):
                shutil.copy2(source, fallback_out)

            if os.path.exists(fallback_out):
                aligned_outputs.append(fallback_out)
                aligned_icp_ready_by_full_strip[fallback_out] = source_icp_ready_path
            else:
                print(f"[ICP] Failed to save fallback strip: {fallback_out}")

            continue

        minx, miny, maxx, maxy = overlap.bounds

        src_mask = (
            (src_icp_full[:, 0] >= minx)
            & (src_icp_full[:, 0] <= maxx)
            & (src_icp_full[:, 1] >= miny)
            & (src_icp_full[:, 1] <= maxy)
        )
        tgt_mask = (
            (tgt_icp_full[:, 0] >= minx)
            & (tgt_icp_full[:, 0] <= maxx)
            & (tgt_icp_full[:, 1] >= miny)
            & (tgt_icp_full[:, 1] <= maxy)
        )

        src_ov = src_icp_full[src_mask]
        tgt_ov = tgt_icp_full[tgt_mask]

        min_overlap_points = int(getattr(config, "icp_min_overlap_points", 2500))
        if len(src_ov) < min_overlap_points or len(tgt_ov) < min_overlap_points:
            print(
                f"[ICP] Skipping {pair_id}: too few overlap points "
                f"(src={len(src_ov)}, tgt={len(tgt_ov)})"
            )

            fallback_out = os.path.join(aligned_dir, os.path.basename(source))
            if os.path.abspath(source) != os.path.abspath(fallback_out):
                shutil.copy2(source, fallback_out)

            if os.path.exists(fallback_out):
                aligned_outputs.append(fallback_out)
                aligned_icp_ready_by_full_strip[fallback_out] = source_icp_ready_path
            else:
                print(f"[ICP] Failed to save fallback strip: {fallback_out}")

            continue

        src_icp = src_ov
        tgt_icp = tgt_ov

        icp_result = run_pair_icp(
            src_icp,
            tgt_icp,
            config,
            source,
            target_full,
            log_file=log_file,
            source_icp_ready_file=source_icp_ready_path,
            target_icp_ready_file=target_icp_ready_path,
        )

        transform = icp_result["transform_global"]

        source_base, source_ext = os.path.splitext(os.path.basename(source))
        aligned_source = os.path.join(aligned_dir, f"{source_base}_aligned{source_ext}")

        apply_transform_to_strip(source, aligned_source, transform)

        if not os.path.exists(aligned_source):
            print(f"[ICP] Failed to save aligned strip: {aligned_source}")

            fallback_out = os.path.join(aligned_dir, os.path.basename(source))
            if os.path.abspath(source) != os.path.abspath(fallback_out):
                shutil.copy2(source, fallback_out)

            if os.path.exists(fallback_out):
                aligned_outputs.append(fallback_out)
                aligned_icp_ready_by_full_strip[fallback_out] = source_icp_ready_path
            else:
                print(f"[ICP] Failed to save fallback strip: {fallback_out}")

            continue

        aligned_outputs.append(aligned_source)

        aligned_icp_ready_path = os.path.join(
            icp_ready_dir,
            f"{os.path.splitext(os.path.basename(aligned_source))[0]}_icp_ready.laz",
        )

        out_ready, out_meta = build_icp_ready_strip(
            aligned_source,
            aligned_icp_ready_path,
            config,
            temp_icp_dir,
            f"{os.path.splitext(os.path.basename(aligned_source))[0]}_full",
        )

        if not out_ready or not os.path.exists(out_ready):
            print(f"[ICP] Warning: aligned ICP-ready strip was not saved for {aligned_source}")
            out_ready = None

        aligned_icp_ready_by_full_strip[aligned_source] = out_ready

        elapsed = time.time() - start_pair
        print(
            f"[ICP] Finished {pair_id} in {elapsed:.2f}s | "
            f"fitness={icp_result['registration'].fitness:.4f} | "
            f"rmse={icp_result['registration'].inlier_rmse:.4f} | "
            f"shift={icp_result['relative_shift_norm']:.4f} m"
        )

    merged_aligned_file = os.path.join(
        preprocessed_aoi_root,
        f"{aoi_name}_aligned_merged.laz"
    )

    merge_aligned_strips(aligned_outputs, merged_aligned_file)

    print(f"[ICP] Saved merged aligned point cloud: {merged_aligned_file}")

    shutil.rmtree(temp_icp_dir, ignore_errors=True)
    return aligned_outputs

def merge_aligned_strips(strip_files, output_file):
    if not strip_files:
        return None

    pipeline = {
        "pipeline": []
    }

    for f in strip_files:
        pipeline["pipeline"].append({
            "type": "readers.las",
            "filename": f
        })

    pipeline["pipeline"].append({
        "type": "filters.merge"
    })

    pipeline["pipeline"].append({
        "type": "writers.las",
        "filename": output_file,
        "compression": "laszip"
    })

    pdal.Pipeline(json.dumps(pipeline)).execute()

    return output_file