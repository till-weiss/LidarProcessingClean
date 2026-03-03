import json
import os
import shutil
import tempfile
from typing import Optional

import laspy
import numpy as np
import pdal


def _log(logger: Optional[object], msg: str) -> None:
    if logger:
        logger.info(msg)
    else:
        print(msg)


def _warn(logger: Optional[object], msg: str) -> None:
    if logger:
        logger.warning(msg)
    else:
        print(f"[WARN] {msg}")


def make_icp_ready_strip(
    in_processed_strip: str,
    out_icp_ready_strip: str,
    config,
    logger=None,
) -> str:
    """
    Build ICP-ready strip (ground-only + de-islanded largest cluster).

    Fallback behavior:
    - if SMRF/ground extraction fails or yields too few points: copy original processed strip
    - if clustering fails: keep ground-only strip
    """
    os.makedirs(os.path.dirname(out_icp_ready_strip), exist_ok=True)

    smrf_slope = float(getattr(config, "icp_smrf_slope", 0.2))
    smrf_window = float(getattr(config, "icp_smrf_window", 16.0))
    smrf_threshold = float(getattr(config, "icp_smrf_threshold", 0.45))
    smrf_scalar = float(getattr(config, "icp_smrf_scalar", 1.2))

    cluster_tolerance = float(
        getattr(config, "icp_cluster_tolerance_m", getattr(config, "icp_cluster_tolerance", 2.0))
    )
    cluster_min_points = int(
        getattr(config, "icp_cluster_min_points", 1000)
    )
    min_ground_points = int(getattr(config, "icp_min_ground_points", 50000))

    use_ground_outlier = bool(getattr(config, "icp_ground_use_outlier", False))
    outlier_mean_k = int(getattr(config, "icp_ground_outlier_mean_k", 8))
    outlier_multiplier = float(getattr(config, "icp_ground_outlier_multiplier", 2.0))

    points_in = int(laspy.open(in_processed_strip).header.point_count)

    tmp_ground = None
    tmp_clustered = None
    try:
        with tempfile.NamedTemporaryFile(suffix="_ground.laz", delete=False) as t1:
            tmp_ground = t1.name
        with tempfile.NamedTemporaryFile(suffix="_clustered.laz", delete=False) as t2:
            tmp_clustered = t2.name

        # Pass 1a: SMRF classification + ground extraction
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

        ground_n = int(laspy.open(tmp_ground).header.point_count)
        if ground_n < min_ground_points:
            _warn(
                logger,
                f"ICP-ready fallback for {os.path.basename(in_processed_strip)}: ground points {ground_n} < icp_min_ground_points {min_ground_points}; using original processed strip.",
            )
            shutil.copy2(in_processed_strip, out_icp_ready_strip)
            return out_icp_ready_strip

        # Pass 1b: clustering on ground points
        cluster_pipeline = [
            {"type": "readers.las", "filename": tmp_ground},
            {
                "type": "filters.cluster",
                "tolerance": cluster_tolerance,
                "min_points": cluster_min_points,
            },
            {"type": "writers.las", "filename": tmp_clustered, "compression": "laszip"},
        ]
        pdal.Pipeline(json.dumps(cluster_pipeline)).execute()

        clustered = laspy.read(tmp_clustered)
        cluster_dim = None
        for candidate in ("ClusterID", "ClusterId"):
            if candidate in clustered.point_format.dimension_names:
                cluster_dim = candidate
                break

        if cluster_dim is None:
            _warn(
                logger,
                f"ICP-ready clustering dimension missing for {os.path.basename(in_processed_strip)}; keeping ground-only output.",
            )
            shutil.copy2(tmp_ground, out_icp_ready_strip)
            return out_icp_ready_strip

        cluster_ids = np.asarray(clustered[cluster_dim])
        valid = cluster_ids >= 0
        if not np.any(valid):
            _warn(
                logger,
                f"ICP-ready clustering produced no valid clusters for {os.path.basename(in_processed_strip)}; keeping ground-only output.",
            )
            shutil.copy2(tmp_ground, out_icp_ready_strip)
            return out_icp_ready_strip

        valid_ids = cluster_ids[valid]
        unique_ids, counts = np.unique(valid_ids, return_counts=True)
        keep_id = int(unique_ids[np.argmax(counts)])
        keep_mask = cluster_ids == keep_id

        out_las = laspy.LasData(clustered.header)
        out_las.points = clustered.points[keep_mask]
        out_las.write(out_icp_ready_strip)

        kept_n = int(np.count_nonzero(keep_mask))
        kept_frac = (kept_n / max(ground_n, 1)) * 100.0
        _log(
            logger,
            (
                f"ICP-ready strip built: in={points_in} ground={ground_n} "
                f"clusters={len(unique_ids)} kept={kept_n} ({kept_frac:.2f}%) "
                f"out={out_icp_ready_strip}"
            ),
        )
        return out_icp_ready_strip
    except Exception as e:
        _warn(
            logger,
            f"ICP-ready generation failed for {os.path.basename(in_processed_strip)} ({e}); using original processed strip.",
        )
        shutil.copy2(in_processed_strip, out_icp_ready_strip)
        return out_icp_ready_strip
    finally:
        for path in (tmp_ground, tmp_clustered):
            if path and os.path.exists(path):
                os.remove(path)
