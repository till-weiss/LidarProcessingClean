import json
import os
import re
import shutil
import time
from datetime import datetime

import laspy
import numpy as np
import open3d as o3d
import pdal
from shapely.geometry import box

from collections import defaultdict

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

def is_crossing_strip(las_file, ratio_threshold=3):
    with laspy.open(las_file) as f:
        h = f.header

    width = h.maxs[0] - h.mins[0]
    height = h.maxs[1] - h.mins[1]

    long_side = max(width, height)
    short_side = min(width, height)

    if short_side == 0:
        return True

    return (long_side / short_side) < ratio_threshold

def las_bounds_polygon(las_path):
    with laspy.open(las_path) as f:
        hdr = f.header
        return box(hdr.mins[0], hdr.mins[1], hdr.maxs[0], hdr.maxs[1]), int(hdr.point_count)

def overlap_ratio_from_polygons(poly_a, poly_b):
    if poly_a is None or poly_b is None:
        return 0.0
    if poly_a.is_empty or poly_b.is_empty:
        return 0.0

    inter_area = poly_a.intersection(poly_b).area
    if inter_area <= 0:
        return 0.0

    smaller_area = min(poly_a.area, poly_b.area)
    if smaller_area <= 0:
        return 0.0

    return inter_area / smaller_area

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

def merge_aligned_strips(strip_files, output_file):
    """
    Merge a list of LAS/LAZ strips into a single LAS/LAZ output using PDAL.
    """
    if not strip_files:
        return None

    for f in strip_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Input strip does not exist: {f}")

    pipeline = {"pipeline": []}

    for f in strip_files:
        pipeline["pipeline"].append({
            "type": "readers.las",
            "filename": f
        })

    pipeline["pipeline"].append({"type": "filters.merge"})
    pipeline["pipeline"].append({
        "type": "writers.las",
        "filename": output_file,
        "compression": "laszip"
    })

    pdal.Pipeline(json.dumps(pipeline)).execute()
    return output_file

def get_aoi_name(target_fp):
    """Extract AOI name from footprint path or filename."""
    return os.path.splitext(os.path.basename(str(target_fp)))[0]


def extract_year_from_strip_filename(strip_path):
    """
    Extract year from filenames like:
    FULL_ALS_L1B_20230707T152300_153124_mta2_utm_aoi.laz
    -> 2023
    """
    name = os.path.basename(strip_path)
    match = re.search(r"(\d{4})\d{4}T\d{6}", name)
    if match:
        return match.group(1)
    return "unknownyear"


def build_cluster_output_filename(target_fp, strip_path, cluster_id):
    """
    Build merged cluster output filename including AOI name and year.
    Example:
    WC_PeelSlumps_20230707_15cm_01_2023_cluster_1.laz
    """
    aoi_name = get_aoi_name(target_fp)
    year = extract_year_from_strip_filename(strip_path)
    return f"{aoi_name}_{year}_cluster_{cluster_id}.laz"


def save_fixed_seed_strip(strip_path, aligned_dir):
    """
    Save a fixed copy of the initial seed strip into aligned_dir.
    Returns path to the fixed file.
    """
    base, ext = os.path.splitext(os.path.basename(strip_path))
    fixed_path = os.path.join(aligned_dir, f"{base}_fixed{ext}")

    if os.path.abspath(strip_path) != os.path.abspath(fixed_path):
        shutil.copy2(strip_path, fixed_path)

    if not os.path.exists(fixed_path):
        raise RuntimeError(f"Failed to save fixed seed strip: {fixed_path}")

    return fixed_path


def save_cluster_laz(cluster_members, cluster_dir, target_fp, cluster_id, source_strip_for_year):
    """
    Merge one cluster and save it with AOI name + year + cluster id.
    Returns output file path.
    """
    output_name = build_cluster_output_filename(
        target_fp=target_fp,
        strip_path=source_strip_for_year,
        cluster_id=cluster_id,
    )
    output_path = os.path.join(cluster_dir, output_name)

    merge_aligned_strips(cluster_members, output_path)

    if not os.path.exists(output_path):
        raise RuntimeError(f"Cluster output was not created: {output_path}")

    return output_path

def align_strips_incremental_icp(processed_strip_files, target_fp, config):
    """
    Overlap-based incremental ICP alignment.

    Main logic:
    - Build ICP-ready versions for all strips
    - Choose an initial seed strip
    - Repeatedly align remaining strips against the BEST accepted reference
      based on overlap, not simply the previous strip
    - Only successfully accepted strips become references
    - If no suitable reference exists, start a new cluster with a new fixed strip
    - Save each cluster individually as its own merged output

    Returns a dict with:
    - accepted_outputs
    - fallback_outputs
    - discarded_outputs
    - clusters
    - cluster_output_files
    """

    if len(processed_strip_files) < 2 or not getattr(config, "enable_icp", True):
        return {
            "accepted_outputs": processed_strip_files,
            "fallback_outputs": [],
            "discarded_outputs": [],
            "clusters": {1: processed_strip_files} if processed_strip_files else {},
            "cluster_output_files": {},
        }

    ordered = sorted(
        [
            f for f in processed_strip_files
            if "_aligned" not in os.path.splitext(os.path.basename(f))[0]
        ],
        key=extract_start_timestamp
    )

    if len(ordered) < 2:
        return {
            "accepted_outputs": processed_strip_files,
            "fallback_outputs": [],
            "discarded_outputs": [],
            "clusters": {1: processed_strip_files} if processed_strip_files else {},
            "cluster_output_files": {},
        }

    print("[ICP] Final strip order before overlap-based alignment:")
    for idx, strip in enumerate(ordered, start=1):
        print(f"  {idx:02d}. {os.path.basename(strip)}")

    aoi_name = get_aoi_name(target_fp)

    preprocessed_aoi_root = os.path.join(config.preprocessed_dir, config.run_name, aoi_name)

    logs_dir = os.path.join(preprocessed_aoi_root, "logs")
    aligned_dir = os.path.join(preprocessed_aoi_root, "aligned_strips")
    icp_ready_dir = os.path.join(preprocessed_aoi_root, "icp_ready_strips")
    temp_icp_dir = os.path.join(preprocessed_aoi_root, "icp_temp")
    cluster_dir = os.path.join(preprocessed_aoi_root, "aligned_clusters")

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(aligned_dir, exist_ok=True)
    os.makedirs(icp_ready_dir, exist_ok=True)
    os.makedirs(temp_icp_dir, exist_ok=True)
    os.makedirs(cluster_dir, exist_ok=True)

    log_file = os.path.join(logs_dir, "icp_log.jsonl")

    min_bbox_overlap = float(getattr(config, "icp_min_bbox_overlap_ratio", 0.05))
    min_overlap_points = int(getattr(config, "icp_min_overlap_points", 2500))
    min_fitness = float(getattr(config, "icp_min_fitness", 0.60))
    max_rmse = float(getattr(config, "icp_max_rmse", 0.80))
    max_shift = float(getattr(config, "icp_max_shift_m", 3.0))
    max_shift_xy = float(getattr(config, "icp_max_shift_xy_m", max_shift))
    max_shift_z = float(getattr(config, "icp_max_shift_z_m", 1.5))
    overlap_buffer = float(getattr(config, "icp_overlap_buffer", 0.0))
    max_passes = int(getattr(config, "icp_max_passes", 5))

    # ------------------------------------------------------------------
    # Build ICP-ready versions and cache point clouds / simple footprints
    # ------------------------------------------------------------------
    strip_info = {}

    for strip in ordered:
        strip_base = os.path.splitext(os.path.basename(strip))[0]
        out_icp_ready = os.path.join(icp_ready_dir, f"{strip_base}_icp_ready.laz")

        icp_path, _ = build_icp_ready_strip(
            strip,
            out_icp_ready,
            config,
            temp_icp_dir,
            f"{strip_base}_full",
        )

        if not icp_path or not os.path.exists(icp_path):
            print(f"[ICP] Warning: ICP-ready strip was not saved for {strip}")
            icp_path = None

        try:
            _, src_xyz_full, src_cls_full = read_las_points(strip)

            if icp_path and os.path.exists(icp_path):
                _, icp_xyz, icp_cls = read_las_points(icp_path)
            else:
                icp_xyz, icp_cls = src_xyz_full, src_cls_full

            if icp_xyz is None or len(icp_xyz) == 0:
                print(f"[ICP] Warning: empty ICP cloud for {os.path.basename(strip)}")
                strip_info[strip] = None
                continue

            footprint = box(
                np.min(icp_xyz[:, 0]),
                np.min(icp_xyz[:, 1]),
                np.max(icp_xyz[:, 0]),
                np.max(icp_xyz[:, 1]),
            )

            strip_info[strip] = {
                "original_path": strip,
                "icp_ready_path": icp_path,
                "full_xyz": src_xyz_full,
                "full_cls": src_cls_full,
                "icp_xyz": icp_xyz,
                "icp_cls": icp_cls,
                "footprint": footprint,
                "area": footprint.area,
            }

        except Exception as e:
            print(f"[ICP] Warning: failed to prepare strip {os.path.basename(strip)}: {e}")
            strip_info[strip] = None

    valid_ordered = [s for s in ordered if strip_info.get(s) is not None]
    discarded_outputs = [s for s in ordered if strip_info.get(s) is None]

    if len(valid_ordered) == 0:
        print("[ICP] No valid strips available after ICP-ready preparation.")
        shutil.rmtree(temp_icp_dir, ignore_errors=True)
        return {
            "accepted_outputs": [],
            "fallback_outputs": [],
            "discarded_outputs": discarded_outputs,
            "clusters": {},
            "cluster_output_files": {},
        }

    # ------------------------------------------------------------------
    # Helper to score overlap between source and accepted reference
    # ------------------------------------------------------------------
    def compute_pair_overlap(source_path, ref_output_path, ref_icp_ready_path):
        src = strip_info[source_path]
        src_poly = src["footprint"]

        try:
            if ref_output_path in accepted_cache and accepted_cache[ref_output_path]["footprint"] is not None:
                ref_poly = accepted_cache[ref_output_path]["footprint"]
                ref_icp_xyz = accepted_cache[ref_output_path]["icp_xyz"]
            else:
                if ref_icp_ready_path and os.path.exists(ref_icp_ready_path):
                    _, ref_icp_xyz, _ = read_las_points(ref_icp_ready_path)
                else:
                    _, ref_icp_xyz, _ = read_las_points(ref_output_path)

                if ref_icp_xyz is None or len(ref_icp_xyz) == 0:
                    return None

                ref_poly = box(
                    np.min(ref_icp_xyz[:, 0]),
                    np.min(ref_icp_xyz[:, 1]),
                    np.max(ref_icp_xyz[:, 0]),
                    np.max(ref_icp_xyz[:, 1]),
                )

                accepted_cache[ref_output_path] = {
                    "icp_xyz": ref_icp_xyz,
                    "footprint": ref_poly,
                }

            overlap_ratio = overlap_ratio_from_polygons(src_poly, ref_poly)
            overlap_geom = src_poly.intersection(ref_poly)

            if overlap_buffer != 0.0 and not overlap_geom.is_empty:
                overlap_geom = overlap_geom.buffer(overlap_buffer)

            overlap_area = 0.0 if overlap_geom.is_empty else overlap_geom.area

            return {
                "overlap_ratio": overlap_ratio,
                "overlap_area": overlap_area,
                "overlap_geom": overlap_geom,
                "ref_icp_xyz": ref_icp_xyz,
                "ref_poly": ref_poly,
            }

        except Exception as e:
            print(
                f"[ICP] Overlap evaluation failed for "
                f"{os.path.basename(source_path)} -> {os.path.basename(ref_output_path)}: {e}"
            )
            return None

    # ------------------------------------------------------------------
    # Seed selection: choose strip with largest footprint area
    # ------------------------------------------------------------------
    initial_seed = max(valid_ordered, key=lambda s: strip_info[s]["area"])
    print(f"[ICP] Initial seed selected by largest footprint: {os.path.basename(initial_seed)}")

    accepted_outputs = []
    fallback_outputs = []
    cluster_members = defaultdict(list)
    accepted_icp_ready = {}
    accepted_source_origin = {}
    accepted_cache = {}
    unresolved = [s for s in valid_ordered if s != initial_seed]

    try:
        fixed_seed = save_fixed_seed_strip(initial_seed, aligned_dir)
        fixed_seed_icp = strip_info[initial_seed]["icp_ready_path"]
    except Exception as e:
        print(f"[ICP] Failed to initialize seed: {e}")
        shutil.rmtree(temp_icp_dir, ignore_errors=True)
        return {
            "accepted_outputs": valid_ordered,
            "fallback_outputs": [],
            "discarded_outputs": discarded_outputs,
            "clusters": {1: valid_ordered},
            "cluster_output_files": {},
        }

    accepted_outputs.append(fixed_seed)
    accepted_icp_ready[fixed_seed] = fixed_seed_icp
    accepted_source_origin[fixed_seed] = initial_seed
    cluster_id = 1
    cluster_members[cluster_id].append(fixed_seed)

    print(f"[ICP] Cluster {cluster_id} seed: {os.path.basename(fixed_seed)}")

    # ------------------------------------------------------------------
    # Multi-pass growth of clusters
    # ------------------------------------------------------------------
    pass_idx = 0
    while unresolved and pass_idx < max_passes:
        pass_idx += 1
        print(f"\n[ICP] Starting pass {pass_idx} with {len(unresolved)} unresolved strips")

        progress_made = False
        next_unresolved = []

        for source in unresolved:
            source_name = os.path.splitext(os.path.basename(source))[0]

            best_ref = None
            best_ref_icp_ready = None
            best_overlap = None

            for ref_output in accepted_outputs:
                ref_icp_ready = accepted_icp_ready.get(ref_output)
                overlap_info = compute_pair_overlap(source, ref_output, ref_icp_ready)

                if overlap_info is None:
                    continue

                if overlap_info["overlap_ratio"] < min_bbox_overlap:
                    continue

                if best_overlap is None:
                    best_ref = ref_output
                    best_ref_icp_ready = ref_icp_ready
                    best_overlap = overlap_info
                else:
                    if (
                        overlap_info["overlap_area"] > best_overlap["overlap_area"]
                        or (
                            np.isclose(overlap_info["overlap_area"], best_overlap["overlap_area"])
                            and overlap_info["overlap_ratio"] > best_overlap["overlap_ratio"]
                        )
                    ):
                        best_ref = ref_output
                        best_ref_icp_ready = ref_icp_ready
                        best_overlap = overlap_info

            if best_ref is None or best_overlap is None:
                print(
                    f"[ICP] No suitable accepted reference found for "
                    f"{os.path.basename(source)} in pass {pass_idx}; postponing"
                )
                next_unresolved.append(source)
                continue

            target_name = os.path.splitext(os.path.basename(best_ref))[0]
            pair_id = f"{source_name}_to_{target_name}"

            print(
                f"[ICP] Best reference for {os.path.basename(source)} is "
                f"{os.path.basename(best_ref)} | "
                f"overlap_ratio={best_overlap['overlap_ratio']:.3f} | "
                f"overlap_area={best_overlap['overlap_area']:.2f}"
            )

            src_icp_full = strip_info[source]["icp_xyz"]
            tgt_icp_full = best_overlap["ref_icp_xyz"]
            overlap_geom = best_overlap["overlap_geom"]

            if overlap_geom.is_empty:
                print(f"[ICP] Empty overlap geometry for {pair_id}; postponing")
                next_unresolved.append(source)
                continue

            minx, miny, maxx, maxy = overlap_geom.bounds

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

            print(f"[ICP] Overlap crop for {pair_id}: src={len(src_ov)}, tgt={len(tgt_ov)}")

            if len(src_ov) < min_overlap_points or len(tgt_ov) < min_overlap_points:
                print(
                    f"[ICP] Too few overlap points for {pair_id}; postponing "
                    f"(src={len(src_ov)}, tgt={len(tgt_ov)})"
                )
                next_unresolved.append(source)
                continue

            start_pair = time.time()
            source_icp_ready_path = strip_info[source]["icp_ready_path"]

            try:
                icp_result = run_pair_icp(
                    src_ov,
                    tgt_ov,
                    config,
                    source,
                    best_ref,
                    log_file=log_file,
                    source_icp_ready_file=source_icp_ready_path,
                    target_icp_ready_file=best_ref_icp_ready,
                )
            except Exception as e:
                print(f"[ICP] ICP execution failed for {pair_id}: {e}")
                next_unresolved.append(source)
                continue

            fitness = float(icp_result["registration"].fitness)
            rmse = float(icp_result["registration"].inlier_rmse)
            shift = float(icp_result["relative_shift_norm"])
            shift_xyz = icp_result.get("relative_shift_xyz", [0.0, 0.0, 0.0])
            shift_xy = float(np.hypot(shift_xyz[0], shift_xyz[1]))
            shift_z = float(abs(shift_xyz[2]))

            is_identity = np.allclose(icp_result["transform_local"], np.eye(4), atol=1e-10)

            if fitness == 0.0 or (is_identity and rmse == 0.0):
                print(f"[ICP] ICP returned no valid alignment for {pair_id}; postponing")
                next_unresolved.append(source)
                continue

            if (
                fitness < min_fitness
                or rmse > max_rmse
                or shift > max_shift
                or shift_xy > max_shift_xy
                or shift_z > max_shift_z
            ):
                print(
                    f"[ICP] Quality check failed for {pair_id} | "
                    f"fitness={fitness:.4f}, rmse={rmse:.4f}, "
                    f"shift={shift:.4f}, shift_xy={shift_xy:.4f}, shift_z={shift_z:.4f}"
                )
                next_unresolved.append(source)
                continue

            transform = icp_result["transform_global"]

            source_base, source_ext = os.path.splitext(os.path.basename(source))
            aligned_source = os.path.join(aligned_dir, f"{source_base}_aligned{source_ext}")

            try:
                apply_transform_to_strip(source, aligned_source, transform)
            except Exception as e:
                print(f"[ICP] Failed to apply transform for {pair_id}: {e}")
                next_unresolved.append(source)
                continue

            if not os.path.exists(aligned_source):
                print(f"[ICP] Failed to save aligned strip: {aligned_source}")
                next_unresolved.append(source)
                continue

            aligned_icp_ready_path = os.path.join(
                icp_ready_dir,
                f"{os.path.splitext(os.path.basename(aligned_source))[0]}_icp_ready.laz",
            )

            out_ready, _ = build_icp_ready_strip(
                aligned_source,
                aligned_icp_ready_path,
                config,
                temp_icp_dir,
                f"{os.path.splitext(os.path.basename(aligned_source))[0]}_full",
            )

            if not out_ready or not os.path.exists(out_ready):
                print(f"[ICP] Warning: aligned ICP-ready strip was not saved for {aligned_source}")
                out_ready = None

            accepted_outputs.append(aligned_source)
            accepted_icp_ready[aligned_source] = out_ready
            accepted_source_origin[aligned_source] = source

            ref_cluster_id = None
            for cid, members in cluster_members.items():
                if best_ref in members:
                    ref_cluster_id = cid
                    break

            if ref_cluster_id is None:
                ref_cluster_id = cluster_id

            cluster_members[ref_cluster_id].append(aligned_source)

            elapsed = time.time() - start_pair
            print(
                f"[ICP] Accepted {pair_id} in {elapsed:.2f}s | "
                f"fitness={fitness:.4f} | rmse={rmse:.4f} | "
                f"shift={shift:.4f} m | ref={os.path.basename(best_ref)}"
            )

            progress_made = True

        unresolved = next_unresolved

        if unresolved and not progress_made:
            new_seed = max(unresolved, key=lambda s: strip_info[s]["area"])
            unresolved.remove(new_seed)

            try:
                fixed_seed = save_fixed_seed_strip(new_seed, aligned_dir)
                fixed_seed_icp = strip_info[new_seed]["icp_ready_path"]
            except Exception as e:
                print(f"[ICP] Failed to start new cluster seed for {os.path.basename(new_seed)}: {e}")
                fallback_outputs.append(new_seed)
                continue

            cluster_id += 1
            accepted_outputs.append(fixed_seed)
            accepted_icp_ready[fixed_seed] = fixed_seed_icp
            accepted_source_origin[fixed_seed] = new_seed
            cluster_members[cluster_id].append(fixed_seed)

            print(
                f"[ICP] No progress in pass {pass_idx}; "
                f"starting new cluster {cluster_id} with seed {os.path.basename(fixed_seed)}"
            )

    fallback_outputs.extend(unresolved)

    # ------------------------------------------------------------------
    # Save each cluster individually
    # ------------------------------------------------------------------
    cluster_output_files = {}

    for cid, members in cluster_members.items():
        if not members:
            continue

        first_member = members[0]
        source_strip_for_year = accepted_source_origin.get(first_member, first_member)

        try:
            cluster_output_file = save_cluster_laz(
                cluster_members=members,
                cluster_dir=cluster_dir,
                target_fp=target_fp,
                cluster_id=cid,
                source_strip_for_year=source_strip_for_year,
            )
            cluster_output_files[cid] = cluster_output_file
            print(f"[ICP] Saved cluster {cid}: {cluster_output_file}")
        except Exception as e:
            print(f"[ICP] Failed to save cluster {cid}: {e}")

    if fallback_outputs:
        print("[ICP] Fallback/unresolved strips:")
        for strip in fallback_outputs:
            print(f"  - {os.path.basename(strip)}")

    if discarded_outputs:
        print("[ICP] Discarded strips:")
        for strip in discarded_outputs:
            print(f"  - {os.path.basename(strip)}")

    print("[ICP] Cluster membership:")
    for cid, members in cluster_members.items():
        print(f"  Cluster {cid}:")
        for m in members:
            print(f"    - {os.path.basename(m)}")

    summary = {
        "stage": "overlap_based_icp_summary",
        "accepted_outputs": [os.path.basename(x) for x in accepted_outputs],
        "fallback_outputs": [os.path.basename(x) for x in fallback_outputs],
        "discarded_outputs": [os.path.basename(x) for x in discarded_outputs],
        "clusters": {
            str(cid): [os.path.basename(x) for x in members]
            for cid, members in cluster_members.items()
        },
        "cluster_output_files": {
            str(cid): os.path.basename(path)
            for cid, path in cluster_output_files.items()
        },
    }

    with open(log_file, "a") as f:
        json.dump(summary, f)
        f.write("\n")

    shutil.rmtree(temp_icp_dir, ignore_errors=True)

    return {
        "accepted_outputs": accepted_outputs,
        "fallback_outputs": fallback_outputs,
        "clusters": dict(cluster_members),
        "cluster_output_files": cluster_output_files,
    }