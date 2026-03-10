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

    return datetime.max


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




def _write_xyz_cloud(xyz, output_path):
    if xyz is None or len(xyz) == 0:
        return None
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    las = laspy.create(file_version="1.4", point_format=6)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.write(output_path)
    return output_path


def _build_icp_ready_strip(input_strip, output_strip, cfg, temp_dir, prefix):
    _, xyz, cls = _read_las_points(input_strip)
    icp_xyz, meta = _prepare_icp_points(xyz, cls, cfg, temp_dir, prefix)
    out_path = _write_xyz_cloud(icp_xyz, output_strip)
    return out_path, meta


def _prepare_icp_points(xyz, cls, cfg, temp_dir, prefix):
    """
    Build ICP-ready points from overlap cloud.
    Prioritises ground-only when possible, with graceful fallback.
    """
    warnings = []
    used_ground_only = False
    used_fallback = False
    ground_point_count = 0

    if len(xyz) == 0:
        return xyz, {
            "used_ground_only": False,
            "used_fallback": True,
            "warnings": ["empty input"],
            "ground_point_count": 0,
        }

    points = xyz
    if getattr(cfg, "icp_use_ground_only", True):
        min_ground = int(getattr(cfg, "icp_min_ground_points", 800))

        if cls is not None and len(cls) == len(xyz):
            ground_points = xyz[cls == 2]
            ground_point_count = int(len(ground_points))
            if len(ground_points) >= min_ground:
                points = ground_points
                used_ground_only = True
            else:
                used_fallback = True
                warnings.append(f"Too few classified ground points ({len(ground_points)}<{min_ground}), fallback to non-ground")
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
                ground_points = xyz[np.asarray(smrf_las.classification) == 2]
                ground_point_count = int(len(ground_points))
                if len(ground_points) >= min_ground:
                    points = ground_points
                    used_ground_only = True
                else:
                    used_fallback = True
                    warnings.append(f"Too few SMRF ground points ({len(ground_points)}<{min_ground}), fallback to non-ground")
            except Exception as exc:
                used_fallback = True
                warnings.append(f"SMRF classification failed: {exc}")

    voxel = float(getattr(cfg, "icp_voxel_size", 1.0))
    if voxel > 0 and len(points) > 0:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcd = pcd.voxel_down_sample(voxel)
        points = np.asarray(pcd.points)

    return points, {
        "used_ground_only": used_ground_only,
        "used_fallback": used_fallback,
        "warnings": warnings,
        "ground_point_count": int(ground_point_count),
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
    """Sequential strip alignment: first strip fixed, each next strip aligned to previous aligned strip."""
    if len(processed_strip_files) < 2 or not getattr(config, "enable_icp", True):
        return processed_strip_files

    ordered = sorted(processed_strip_files, key=_extract_timestamp)
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

    run_log_path = os.path.join(logs_dir, "icp_alignment_log.jsonl")
    run_summary_path = os.path.join(logs_dir, "icp_run_summary.json")

    # Build ICP-ready version for every original strip before alignment.
    icp_ready_original_by_strip = {}
    icp_ready_meta_by_strip = {}
    for strip in ordered:
        strip_base = os.path.splitext(os.path.basename(strip))[0]
        out_icp_ready = os.path.join(icp_ready_dir, f"{strip_base}_icp_ready.laz")
        icp_path, icp_meta = _build_icp_ready_strip(strip, out_icp_ready, config, temp_icp_dir, f"{strip_base}_full")
        icp_ready_original_by_strip[strip] = icp_path
        icp_ready_meta_by_strip[strip] = icp_meta

    aligned_outputs = []
    aligned_icp_ready_by_full_strip = {}

    fixed_first = os.path.join(aligned_dir, os.path.basename(ordered[0]))
    shutil.copy2(ordered[0], fixed_first)
    aligned_outputs.append(fixed_first)
    aligned_icp_ready_by_full_strip[fixed_first] = icp_ready_original_by_strip.get(ordered[0])

    summary = {"total_pairs": 0, "attempted": 0, "accepted": 0, "rejected": 0, "skipped": 0}

    with open(run_log_path, "a", encoding="utf-8") as run_log:
        for idx, source in enumerate(ordered[1:], start=1):
            start_pair = time.time()
            target_full = aligned_outputs[-1]
            source_name = os.path.splitext(os.path.basename(source))[0]
            target_name = os.path.splitext(os.path.basename(target_full))[0]
            pair_id = f"{source_name}_to_{target_name}"

            _, src_xyz_full, src_cls_full = _read_las_points(source)
            _, tgt_xyz_full, _ = _read_las_points(target_full)

            source_icp_ready_path = icp_ready_original_by_strip.get(source)
            target_icp_ready_path = aligned_icp_ready_by_full_strip.get(target_full)

            if source_icp_ready_path and os.path.exists(source_icp_ready_path):
                _, src_icp_full, src_icp_cls = _read_las_points(source_icp_ready_path)
            else:
                src_icp_full, src_icp_cls = src_xyz_full, src_cls_full

            if target_icp_ready_path and os.path.exists(target_icp_ready_path):
                _, tgt_icp_full, _ = _read_las_points(target_icp_ready_path)
            else:
                tgt_icp_full = tgt_xyz_full

            src_poly, src_count = _las_bounds_polygon(source)
            tgt_poly = box(np.min(tgt_icp_full[:, 0]), np.min(tgt_icp_full[:, 1]), np.max(tgt_icp_full[:, 0]), np.max(tgt_icp_full[:, 1]))
            overlap = src_poly.intersection(tgt_poly).buffer(float(getattr(config, "icp_overlap_buffer", 0.0)))

            pair_log = {
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "aoi": aoi_name,
                "run_name": config.run_name,
                "pair_id": pair_id,
                "source_strip": source,
                "target_strip": target_full,
                "source_full_point_count": int(src_count),
                "target_full_point_count": int(len(tgt_xyz_full)),
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
                "icp_ready_source_path": source_icp_ready_path,
                "icp_ready_target_path": target_icp_ready_path,
                "ground_source_point_count": int(icp_ready_meta_by_strip.get(source, {}).get("ground_point_count", 0)),
                "ground_target_point_count": 0,
            }
            summary["total_pairs"] += 1

            if overlap.is_empty:
                pair_log["reject_reason"] = "overlap too small"
                pair_log["would_be_rejected"] = True
                summary["skipped"] += 1
            else:
                minx, miny, maxx, maxy = overlap.bounds
                src_mask = (src_icp_full[:, 0] >= minx) & (src_icp_full[:, 0] <= maxx) & (src_icp_full[:, 1] >= miny) & (src_icp_full[:, 1] <= maxy)
                tgt_mask = (tgt_icp_full[:, 0] >= minx) & (tgt_icp_full[:, 0] <= maxx) & (tgt_icp_full[:, 1] >= miny) & (tgt_icp_full[:, 1] <= maxy)
                src_ov = src_icp_full[src_mask]
                tgt_ov = tgt_icp_full[tgt_mask]

                pair_log["overlap_point_count_source"] = int(len(src_ov))
                pair_log["overlap_point_count_target"] = int(len(tgt_ov))
                pair_log["overlap_fraction_source"] = float(len(src_ov) / max(len(src_icp_full), 1))
                pair_log["overlap_fraction_target"] = float(len(tgt_ov) / max(len(tgt_icp_full), 1))

                min_overlap_points = int(getattr(config, "icp_min_overlap_points", 2500))
                if len(src_ov) < min_overlap_points or len(tgt_ov) < min_overlap_points:
                    pair_log["reject_reason"] = "too few overlap points"
                    pair_log["would_be_rejected"] = True
                    summary["skipped"] += 1
                else:
                    # Already ICP-ready per-strip, so keep overlap subsets as-is.
                    src_icp = src_ov
                    tgt_icp = tgt_ov
                    pair_log["ground_only_mode_used"] = bool(icp_ready_meta_by_strip.get(source, {}).get("used_ground_only", False))
                    pair_log["fallback_mode_used"] = bool(icp_ready_meta_by_strip.get(source, {}).get("used_fallback", False))
                    pair_log["warnings_or_errors"].extend(icp_ready_meta_by_strip.get(source, {}).get("warnings", []))

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

                            aligned_icp_ready_path = os.path.join(
                                icp_ready_dir,
                                f"{os.path.splitext(os.path.basename(aligned_source))[0]}_icp_ready.laz",
                            )
                            out_ready, out_meta = _build_icp_ready_strip(
                                aligned_source,
                                aligned_icp_ready_path,
                                config,
                                temp_icp_dir,
                                f"{os.path.splitext(os.path.basename(aligned_source))[0]}_full",
                            )
                            aligned_icp_ready_by_full_strip[aligned_source] = out_ready
                            pair_log["ground_target_point_count"] = int(out_meta.get("ground_point_count", 0))
                        except Exception as exc:
                            pair_log["reject_reason"] = f"ICP failure: {exc}"
                            pair_log["warnings_or_errors"].append(str(exc))
                            summary["rejected"] += 1
                            passthrough = os.path.join(aligned_dir, os.path.basename(source))
                            shutil.copy2(source, passthrough)
                            aligned_outputs.append(passthrough)
                            aligned_icp_ready_by_full_strip[passthrough] = icp_ready_original_by_strip.get(source)

            pair_log["runtime_seconds"] = float(time.time() - start_pair)
            run_log.write(json.dumps(_safe_json(pair_log)) + "\n")

    with open(run_summary_path, "w", encoding="utf-8") as f:
        json.dump(_safe_json(summary), f, indent=2)

    shutil.rmtree(temp_icp_dir, ignore_errors=True)
    return aligned_outputs


