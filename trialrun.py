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
    config.run_name = "Inuvik_test_area"
    config.target_area_dir = "/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/data/area"
    config.las_files_dir = "/isipd/projects/p_planetdw/data/lidar/02_pointclouds/2025"
    config.las_footprints_dir = "/isipd/projects/p_planetdw/data/lidar/03_las_footprints/2025"
    config.preprocessed_dir = "/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/preprocessed"
    config.results_dir = "/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/results"

    config.smrf_filter = False  # use SMRF filter 
    config.csf_filter = True  # use cloth simulation method
    config.threshold = 0.5  # vertical tolerance (m) for extra clipping. Typical 0.5–2.

        # CSF (Cloth Simulation)
    config.csf_rigidness = 1  # cloth stiffness. 1–2 for rugged/steep; 3–4 for very flat urban.
    config.csf_iterations = 500  # steps. 200–1000. More = better fit, slower.
    config.csf_time_step = 1  # integration step. 0.5–1.0 common. Smaller = stable/accurate, slower.
    config.csf_cloth_resolution = 1  # grid spacing (m). 0.5–2 typical. Smaller = finer ground detail, heavier.

        # SMRF ground classification
    config.smrf_window_size = 12.0
    config.smrf_slope = 0.2
    config.smrf_scalar = 1.5
    config.threshold = 0.5

    config.enable_xdem_coreg = True  # run strip-wise DEM rasterisation + xDEM NuthKaab coregistration
    config.xdem_input_dir = '/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/preprocessed/Inuvik_test_area/inuvik_test/strips'
    config.xdem_filename_token = '_utm_aoi'

    #config.overlap = 0.2

    config.fill_gaps = False
    config.chunk_size = 10000

    config.create_DSM = True
    config.create_DEM = False
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
        cleaned_las = run_pre_dir / f"{Path(target).stem}.laz"
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
