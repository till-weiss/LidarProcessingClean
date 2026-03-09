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
    config.run_name = "Inuvik_no_ICP"
    config.target_area_dir = "/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/data/area"
    config.las_files_dir = "/isipd/projects/p_planetdw/data/lidar/02_pointclouds/2023"
    config.las_footprints_dir = "/isipd/projects/p_planetdw/data/lidar/03_las_footprints/2023"
    config.preprocessed_dir = "/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/preprocessed"
    config.results_dir = "/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/results"

    # --- preprocessing controls (includes new strip/chunk toggle) ---
    config.num_workers = 16
    config.chunk_size = 500
    config.chunk_overlap = 0.1
    config.overlap = 0.2
    config.preprocess_use_chunks = False  # True=chunk mode, False=strip mode

    # --- optional matching/date filters ---
    config.filter_date = False
    config.start_date = "2023-07-10"
    config.end_date = "2023-07-10"

    # --- cleaning controls ---
    config.max_elevation_threshold = 0.99
    config.knn = 100
    config.multiplier = 1

    # --- downstream processing outputs ---
    config.create_DSM = True
    config.create_DEM = True
    config.create_CHM = False
    config.fill_gaps = True
    config.resolution = 1

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

        processed_strips_dir = run_pre_dir / Path(target).stem / "processed_strips"
        if processed_strips_dir.exists():
            strip_files = sorted(processed_strips_dir.glob("*_processed.laz"))
            print(f"    processed strip files: {len(strip_files)} in {processed_strips_dir}")

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
