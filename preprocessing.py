import os
import time
import json
import shutil
from datetime import timedelta
from datetime import datetime
from multiprocessing import Pool

import pdal
import laspy
import numpy as np
from matplotlib import pyplot as plt
import geopandas as gpd
from tqdm import tqdm
from shapely.geometry import shape, box
from shapely.wkt import loads as wkt_loads, dumps as wkt_dumps

from core.reprojection import get_utm_epsg, reproject_las, is_utm_crs
from core.preprocess_windowed import create_chunks_from_wkt, process_chunk, merge_and_crop_chunks, merge_chunks_to_strip
from core.extract_footprints import extract_footprint_batch
from core.utils import split_gpkg


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


def get_las_bounds_wkt(las_file):
    with laspy.open(las_file) as las:
        min_x, min_y = las.header.mins[0], las.header.mins[1]
        max_x, max_y = las.header.maxs[0], las.header.maxs[1]
    return wkt_dumps(box(min_x, min_y, max_x, max_y))

def build_strip_chunk_tasks(strip_path, chunks, temp_dir, max_z, min_z, sor_knn, sor_multiplier, ref_scale, ref_offset, ref_crs):
    return [
        (strip_path, (strip_path, chunk, temp_dir, max_z, min_z, sor_knn, sor_multiplier, ref_scale, ref_offset, ref_crs))
        for chunk in chunks
    ]


def process_chunk_with_provenance(task):
    strip_path, chunk_args = task
    processed_chunk = process_chunk(*chunk_args)
    return strip_path, processed_chunk


def group_chunks_by_strip(processed_results):
    grouped = {}
    for strip_path, chunk_file in processed_results:
        if chunk_file:
            grouped.setdefault(strip_path, []).append(chunk_file)
    return grouped


def merge_chunks_for_strip(strip_output_name, strip_chunks, strips_dir):
    strip_output_file = os.path.join(strips_dir, f"processed_strip_{strip_output_name}.laz")
    return merge_chunks_to_strip(strip_chunks, strip_output_file)


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




def preprocess_window(strip_files, config, target_fp, run_merged_dir, temp_dir, target_gdf):
    """Process each strip clipped to AOI and preserve strip identity."""
    processed_strips_dir = os.path.join(run_merged_dir, target_fp, "processed_strips")
    os.makedirs(processed_strips_dir, exist_ok=True)

    processed_strip_files = []
    print(f"Input strips for {target_fp}: {len(strip_files)}")

    for input_file in strip_files:
        strip_input = input_file
        if not is_utm_crs(strip_input):
            base_name = os.path.basename(strip_input)
            base_name = base_name.replace('.las', '_utm.las').replace('.laz', '_utm.las')
            utm_output_file = os.path.join(temp_dir, base_name)
            strip_input = reproject_las(strip_input, utm_output_file)

        ref_scale, ref_offset, ref_crs = get_las_header(strip_input)
        target_proj_gdf = target_gdf.to_crs(epsg=ref_crs) if target_gdf.crs.to_epsg() != ref_crs else target_gdf
        target_geom = shape(target_proj_gdf.geometry.iloc[0])

        strip_geom_wkt = get_las_bounds_wkt(strip_input)
        strip_geom = wkt_loads(strip_geom_wkt)
        process_geom = target_geom.intersection(strip_geom)

        if process_geom.is_empty:
            print(f"[WARN] Strip does not intersect AOI and is skipped: {strip_input}")
            continue

        if config.max_elevation_threshold:
            all_z = laspy.read(strip_input).z
            max_z = np.quantile(all_z, config.max_elevation_threshold)
            min_z = np.quantile(all_z, 1 - config.max_elevation_threshold)
        else:
            all_z = laspy.read(strip_input).z
            max_z = np.max(all_z)
            min_z = np.min(all_z)

        points_before = laspy.open(strip_input).header.point_count
        processed_strip_chunk = process_chunk(
            strip_input,
            process_geom,
            temp_dir,
            max_z,
            min_z,
            config.knn,
            config.multiplier,
            ref_scale,
            ref_offset,
            ref_crs,
        )

        strip_name = os.path.splitext(os.path.basename(strip_input))[0]
        output_strip = os.path.join(processed_strips_dir, f"{strip_name}_processed.laz")

        if not processed_strip_chunk or not os.path.exists(processed_strip_chunk):
            print(f"[WARN] Strip produced no valid points and is skipped: {strip_input}")
            continue

        points_after = laspy.open(processed_strip_chunk).header.point_count
        strip_output_file = merge_chunks_to_strip([processed_strip_chunk], output_strip)
        if not strip_output_file:
            print(f"[WARN] Failed to write processed strip output: {output_strip}")
            continue
        if os.path.exists(processed_strip_chunk):
            os.remove(processed_strip_chunk)
        processed_strip_files.append(strip_output_file)
        print(f"Processed strip (AOI-clipped) | input: {strip_input} | points before: {points_before} | points after: {points_after} | output: {strip_output_file}")

    if len(processed_strip_files) != len(strip_files):
        print(f"Sanity check: input strips={len(strip_files)}, processed strips={len(processed_strip_files)}")
    else:
        print(f"Sanity check passed: input strips={len(strip_files)}, processed strips={len(processed_strip_files)}")

    return processed_strip_files


def merge_and_clean_las(las_dict, preprocessed_dir, run_name, target_footprint_dir, max_elev, sor_knn, sor_multiplier, num_workers, chunk_size=1000, preprocess_use_chunks=True):

    run_merged_dir = os.path.join(preprocessed_dir, run_name)
    os.makedirs(run_merged_dir, exist_ok=True)

    print("\nProcessing LAS files in chunks...")
    start = time.time()
    processed_strips_by_target = {}

    for target_fp, las_files in tqdm(las_dict.items(), desc="Processing target areas", unit="area"):
        if not las_files:
            print(f"No valid LAS files for {target_fp}. Skipping.")
            continue

        clean_target_fp = os.path.splitext(target_fp)[0]
        final_output_file = os.path.join(run_merged_dir, f"{clean_target_fp}.las")

        if os.path.exists(final_output_file):
            print(f"Skipping {target_fp}: Already processed.")
            continue

        footprint_path = os.path.join(target_footprint_dir, target_fp if target_fp.endswith('.gpkg') else f"{target_fp}.gpkg")
        if not os.path.exists(footprint_path):
            print(f"Footprint file {footprint_path} not found. Skipping.")
            continue

        gdf = gpd.read_file(footprint_path)
        temp_dir = os.path.join(run_merged_dir, target_fp, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        target_geom_wkt = wkt_dumps(shape(gdf.geometry.iloc[0]))

        if not preprocess_use_chunks:
            processed_strip_files = preprocess_window(
                strip_files=las_files,
                config=config,
                target_fp=target_fp,
                run_merged_dir=run_merged_dir,
                temp_dir=temp_dir,
                target_gdf=gdf,
            )
            processed_strips_by_target[target_fp] = processed_strip_files

            if processed_strip_files:
                merge_and_crop_chunks(processed_strip_files, target_geom_wkt, final_output_file)
                print(f"Final processed LAS file saved: {final_output_file}")
            else:
                print(f"No processed strips available for {target_fp}.")

            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

            target_fp_dir = os.path.join(run_merged_dir, target_fp)
            if os.path.isdir(target_fp_dir) and not os.listdir(target_fp_dir):
                os.rmdir(target_fp_dir)
            continue

        processed_chunks = []
        strips_dir = os.path.join(run_merged_dir, target_fp, "strips")
        os.makedirs(strips_dir, exist_ok=True)

        strip_chunk_counts = {}
        all_strip_tasks = []

        print(f"Input strips for {target_fp}: {len(las_files)}")

        for input_file in las_files:
            if not is_utm_crs(input_file):
                # Handle both .las and .laz extensions
                base_name = os.path.basename(input_file)
                base_name = base_name.replace('.las', '_utm.las').replace('.laz', '_utm.las')
                utm_output_file = os.path.join(temp_dir, base_name)
                input_file = reproject_las(input_file, utm_output_file)

            ref_scale, ref_offset, ref_crs = get_las_header(input_file)

            if gdf.crs.to_epsg() != ref_crs:
                gdf = gdf.to_crs(epsg=ref_crs)

            strip_geom_wkt = get_las_bounds_wkt(input_file)
            strip_geom = wkt_loads(strip_geom_wkt)
            target_geom = wkt_loads(target_geom_wkt)
            process_geom = target_geom.intersection(strip_geom)
            if process_geom.is_empty:
                print(f"[WARN] Strip does not intersect AOI and is skipped: {input_file}")
                continue
            chunks = create_chunks_from_wkt(wkt_dumps(process_geom), chunk_size)

            if max_elev:
                all_z = laspy.read(input_file).z
                max_z = np.quantile(all_z, max_elev)
                min_z = np.quantile(all_z, 1 - max_elev)
            else:
                all_z = laspy.read(input_file).z
                max_z = np.max(all_z)
                min_z = np.min(all_z)

            strip_tasks = build_strip_chunk_tasks(
                strip_path=input_file,
                chunks=chunks,
                temp_dir=temp_dir,
                max_z=max_z,
                min_z=min_z,
                sor_knn=sor_knn,
                sor_multiplier=sor_multiplier,
                ref_scale=ref_scale,
                ref_offset=ref_offset,
                ref_crs=ref_crs,
            )
            strip_path = input_file
            strip_chunk_counts[strip_path] = len(strip_tasks)
            all_strip_tasks.extend(strip_tasks)

        strip_output_name_by_path = {
            strip_path: f"{idx:03d}_{os.path.splitext(os.path.basename(strip_path))[0]}"
            for idx, strip_path in enumerate(strip_chunk_counts.keys(), start=1)
        }

        for strip_path, count in strip_chunk_counts.items():
            print(f"Chunks created for strip {os.path.basename(strip_path)}: {count}")

        processed_results = []
        with tqdm(total=len(all_strip_tasks), desc=f"Processing {target_fp}", unit="chunk") as pbar:
            with Pool(processes=num_workers) as pool:
                for result in pool.imap_unordered(process_chunk_with_provenance, all_strip_tasks):
                    processed_results.append(result)
                    pbar.update(1)

        chunks_by_strip = group_chunks_by_strip(processed_results)

        processed_strip_files = []
        for strip_path, strip_chunks in chunks_by_strip.items():
            print(f"Processed chunks for strip {os.path.basename(strip_path)}: {len(strip_chunks)}")
            processed_chunks.extend(strip_chunks)
            strip_output_file = merge_chunks_for_strip(strip_output_name_by_path[strip_path], strip_chunks, strips_dir)
            if strip_output_file:
                processed_strip_files.append(strip_output_file)
                print(f"Saved strip-level LAS: {strip_output_file}")

        if len(processed_strip_files) != len(las_files):
            missing = set(las_files) - set(chunks_by_strip.keys())
            if missing:
                missing_names = sorted(os.path.basename(path) for path in missing)
                print(f"Strips with zero valid points (no merged output): {missing_names}")
            print(f"Sanity check: input strips={len(las_files)}, merged strip outputs={len(processed_strip_files)}")
        else:
            print(f"Sanity check passed: input strips={len(las_files)}, merged strip outputs={len(processed_strip_files)}")

        processed_strips_by_target[target_fp] = processed_strip_files

        if processed_chunks:
            merge_and_crop_chunks(processed_chunks, target_geom_wkt, final_output_file)
            print(f"Final processed LAS file saved: {final_output_file}")
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
        max_elev=config.max_elevation_threshold,
        sor_knn=config.knn,
        sor_multiplier=config.multiplier,
        num_workers=config.num_workers,
        run_name=run_name,
        chunk_size=config.chunk_size,
        preprocess_use_chunks=config.preprocess_use_chunks,
    )

    print(f"\nPreprocessing completed in {str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n")
