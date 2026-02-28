import os
import time
from datetime import timedelta
from pathlib import Path

import laspy

import config.config as configuration
import preprocessing as pre
import processing as pro


def count_points(las_path: str) -> int:
    with laspy.open(las_path) as f:
        return int(f.header.point_count)


def main() -> None:
    config = configuration.Configuration()

    # --- EDIT THESE TO YOUR PATHS ---
    config.run_name = "ICP_trialrun"
    config.target_area_dir = "/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/data/area"
    config.las_files_dir = "/isipd/projects/p_planetdw/data/lidar/02_pointclouds/2023"
    config.las_footprints_dir = "/isipd/projects/p_planetdw/data/lidar/03_las_footprints/2023"
    config.preprocessed_dir = "/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/preprocessed"
    config.results_dir = "/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/results"

    # --- optional: first test run settings ---
    config.num_workers = 16
    config.chunk_size = 500
    config.overlap = 0.2

    # ICP toggle + metric-mode tuning
    config.use_strip_icp = True
    config.icp_use_ground_only = True

    # Overlap-driven pairing gates
    config.icp_min_overlap_area = 10_000
    config.icp_min_overlap_points = 300_000
    config.icp_min_selected_points = 5_000

    # Optional ground selection mode for ICP (default-safe: heuristic)
    # Set to "classification" to use existing Classification or PDAL SMRF/CSF classification with fallback
    config.icp_ground_method = "classification"
    config.icp_ground_classifier = "smrf"  # "smrf" | "csf"
    config.icp_ground_class = 2
    config.icp_use_existing_classification = True
    config.icp_classification_cache = True
    config.icp_classification_voxel_size = 1.0
    config.icp_min_classified_points = 50_000
    config.icp_pdal_smrf_params = {
        "window": 20.0,
        "slope": 0.2,
        "threshold": 0.5,
        "scalar": 2.0,
    }
    # Example CSF params if you switch classifier to csf:
    # config.icp_pdal_csf_params = {"resolution": 1.0, "rigidness": 3, "step": 1.0}

    # Optional ICP estimator mode
    config.icp_estimation = "point_to_plane"  # "point_to_point" | "point_to_plane"
    config.icp_normal_radius = 2.0
    config.icp_normal_max_nn = 30

    # Keep current transform acceptance thresholds
    config.icp_min_fitness = 0.05
    config.icp_max_translation_m = 200.0
    config.icp_max_rotation_deg = 5.0

    config.create_DSM = True
    config.create_DEM = True
    config.create_CHM = False

    print("\n========== STEP A: Footprint matching (for logging) ==========")
    t0 = time.time()

    run_out = os.path.join(config.preprocessed_dir, config.run_name)
    os.makedirs(run_out, exist_ok=True)

    las_dict = pre.match_footprints(
        target_footprint_dir=config.target_area_dir,
        las_footprint_dir=config.las_footprints_dir,
        las_file_dir=config.las_files_dir,
        out_dir=run_out,
        threshold=config.overlap,
        filter_date=config.filter_date,
        start_date=config.start_date,
        end_date=config.end_date,
    )

    print(f"Footprint matching finished in {time.time() - t0:.1f}s")

    print("\nTiles per target + raw point counts:")
    raw_points_by_target = {}
    for target, tiles in las_dict.items():
        raw_points = sum(count_points(tile) for tile in tiles)
        raw_points_by_target[target] = raw_points
        print(f"  {target}: tiles={len(tiles)} | raw_points={raw_points:,}")

    print("\n========== STEP B: preprocess_all(config) ==========")
    t1 = time.time()
    pre.preprocess_all(config)
    print(f"Preprocessing completed in {timedelta(seconds=int(time.time() - t1))}")

    print("\nCleaned point counts (raw → cleaned):")
    run_pre_dir = Path(config.preprocessed_dir) / config.run_name

    for target, raw_points in raw_points_by_target.items():
        cleaned_las = run_pre_dir / f"{Path(target).stem}.las"
        if not cleaned_las.exists():
            print(f"  {Path(target).stem}: NOT FOUND -> {cleaned_las}")
            continue

        cleaned_points = count_points(str(cleaned_las))
        removed_pct = 100 * (1 - cleaned_points / raw_points) if raw_points else 0.0
        print(f"  {Path(target).stem}: {raw_points:,} → {cleaned_points:,} ({removed_pct:.1f}% removed)")

    print("\n========== STEP C: process_all(config) ==========")
    t2 = time.time()
    pro.process_all(config)
    print(f"Processing completed in {timedelta(seconds=int(time.time() - t2))}")

    print("\nDone.")
    print(f"Cleaned LAS: {Path(config.preprocessed_dir) / config.run_name}")
    print(f"Results:     {Path(config.results_dir) / config.run_name}")
    print(f"ICP logs:    {Path(config.results_dir) / config.run_name / 'icp_logs'}")


if __name__ == "__main__":
    main()
