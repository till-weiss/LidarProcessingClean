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
from shapely.geometry import shape
from shapely.wkt import loads as wkt_loads, dumps as wkt_dumps

from core.reprojection import get_utm_epsg, reproject_las, is_utm_crs
from core.preprocess_windowed import create_chunks_from_wkt, process_chunk, merge_and_crop_chunks, process_strip, merge_and_crop_strips
from core.extract_footprints import extract_footprint_batch
from core.utils import split_gpkg
import config.config as config
from core.icp import align_strips_incremental_icp


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


def merge_and_clean_las(las_dict, preprocessed_dir, run_name, target_footprint_dir, max_elev, sor_knn, sor_multiplier,
                        sor_passes, elm_filter, elm_cell, elm_threshold,
                        radius_filter, radius_filter_radius, radius_filter_min_count,
                        num_workers, chunk_size=500, chunk_overlap=0.1,
                        reproject_vertical=True, target_vertical_epsg=3855):

    run_merged_dir = os.path.join(preprocessed_dir, run_name)
    os.makedirs(run_merged_dir, exist_ok=True)

    print("\nProcessing LAS files in chunks...")
    start = time.time()

    for target_fp, las_files in tqdm(las_dict.items(), desc="Processing target areas", unit="area"):
        if not las_files:
            print(f"No valid LAS files for {target_fp}. Skipping.")
            continue

        clean_target_fp = os.path.splitext(target_fp)[0]
        final_output_file = os.path.join(run_merged_dir, f"{clean_target_fp}.laz")

        if os.path.exists(final_output_file):
            print(f"Skipping {target_fp}: Already processed.")
            continue

        footprint_path = os.path.join(
            target_footprint_dir,
            target_fp if target_fp.endswith(".gpkg") else f"{target_fp}.gpkg"
        )
        if not os.path.exists(footprint_path):
            print(f"Footprint file {footprint_path} not found. Skipping.")
            continue

        gdf = gpd.read_file(footprint_path)

        temp_dir = os.path.join(run_merged_dir, target_fp, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        target_geom_wkt = wkt_dumps(shape(gdf.geometry.iloc[0]))

        # -------------------------------------------------------------
        # strip-preserving mode
        # -------------------------------------------------------------
        if config.preprocess_by_strip:
            processed_strip_files = []

            out_dir = os.path.join(run_merged_dir, target_fp, "strips")
            os.makedirs(out_dir, exist_ok=True)

            print(f"Input strips for {target_fp}: {len(las_files)}")

            for input_file in las_files:
                try:
                    strip_input = input_file

                    if not is_utm_crs(strip_input):
                        base_name = os.path.basename(strip_input)
                        base_name = base_name.replace(".las", "_utm.las").replace(".laz", "_utm.laz")
                        utm_output_file = os.path.join(temp_dir, base_name)
                        strip_input = reproject_las(strip_input, utm_output_file)

                    ref_scale, ref_offset, ref_crs = get_las_header(strip_input)

                    if gdf.crs.to_epsg() != ref_crs:
                        gdf_local = gdf.to_crs(epsg=ref_crs)
                    else:
                        gdf_local = gdf

                    target_geom_wkt = wkt_dumps(shape(gdf_local.geometry.iloc[0]))

                    all_z = laspy.read(strip_input).z
                    if max_elev:
                        max_z = np.quantile(all_z, max_elev)
                        min_z = np.quantile(all_z, 1 - max_elev)
                    else:
                        max_z = np.max(all_z)
                        min_z = np.min(all_z)

                    processed_file = process_strip(
                        input_file=strip_input,
                        target_gdf=target_geom_wkt,
                        out_dir=out_dir,
                        max_z=max_z,
                        min_z=min_z,
                        sor_knn=config.sor_knn,
                        sor_multiplier=config.sor_multiplier,
                        ref_scale=ref_scale,
                        ref_offset=ref_offset,
                        ref_crs=ref_crs,
                    )

                    if processed_file and os.path.exists(processed_file):
                        processed_strip_files.append(processed_file)

                except Exception as e:
                    print(f"Error preparing strip {input_file}: {e}")

            if getattr(config, "enable_icp", True) and len(processed_strip_files) > 1:
                print(f"Running sequential ICP strip alignment for {target_fp}...")
                icp_result = align_strips_incremental_icp(
                    processed_strip_files=processed_strip_files,
                    target_fp=target_fp,
                    config=config,
                )

                accepted_outputs = icp_result.get("accepted_outputs", [])
                fallback_outputs = icp_result.get("fallback_outputs", [])
                discarded_outputs = icp_result.get("discarded_outputs", [])

                if accepted_outputs:
                    merged_aligned_file = os.path.join(
                        run_merged_dir, f"{target_fp}_aligned_merged.laz"
                    )
                    merge_and_crop_strips(
                        accepted_outputs,
                        target_geom_wkt,
                        merged_aligned_file
                    )
                else:
                    print(f"No accepted ICP strips available for {target_fp}.")

                all_outputs = accepted_outputs + fallback_outputs
                if all_outputs:
                    final_output_file = os.path.join(run_merged_dir, f"{clean_target_fp}.laz")
                    merge_and_crop_strips(
                        all_outputs,
                        target_geom_wkt,
                        final_output_file
                    )
                    print(f"Final processed LAS file saved: {final_output_file}")
                else:
                    print(f"No processed strips available for {target_fp}.")

        # -------------------------------------------------------------
        # original chunk mode
        # -------------------------------------------------------------
        processed_chunks = []
        process_args = []

        for input_file in tqdm(las_files, desc=f"Preparing {target_fp}", unit="las"):
            strip_input = input_file

            if not is_utm_crs(strip_input):
                base_name = os.path.basename(strip_input)
                base_name = base_name.replace(".las", "_utm.las").replace(".laz", "_utm.laz")
                utm_output_file = os.path.join(temp_dir, base_name)
                strip_input = reproject_las(strip_input, utm_output_file)

            ref_scale, ref_offset, ref_crs = get_las_header(strip_input)

            if gdf.crs.to_epsg() != ref_crs:
                gdf_local = gdf.to_crs(epsg=ref_crs)
            else:
                gdf_local = gdf

            target_geom_wkt = wkt_dumps(shape(gdf_local.geometry.iloc[0]))
            chunks = create_chunks_from_wkt(target_geom_wkt, chunk_size)

            all_z = laspy.read(strip_input).z
            if max_elev:
                max_z = np.quantile(all_z, max_elev)
                min_z = np.quantile(all_z, 1 - max_elev)
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
                        sor_knn,
                        sor_multiplier,
                        ref_scale,
                        ref_offset,
                        ref_crs,
                    )
                )

        with tqdm(total=len(process_args), desc=f"Processing {target_fp}", unit="chunk") as pbar:
            with Pool(processes=num_workers) as pool:
                for processed_chunk in pool.imap_unordered(process_chunk_wrapper, process_args):
                    if processed_chunk:
                        processed_chunks.append(processed_chunk)
                    pbar.update(1)

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


def preprocess_all(conf):
    global config
    config = conf

    print("\n========== Starting Preprocessing ==========")
    start = time.time()

    run_name = config.run_name

    os.makedirs(os.path.join(config.preprocessed_dir, run_name), exist_ok=True)
    os.makedirs(os.path.join(config.results_dir, run_name), exist_ok=True)

    gdfs = sorted(
        f for f in os.listdir(config.target_area_dir)
        if f.lower().endswith(".gpkg") and not f.startswith("."))    
    
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
        sor_knn=config.sor_knn,
        sor_multiplier=config.sor_multiplier,
        sor_passes=config.sor_passes,
        elm_filter=config.elm_filter,
        elm_cell=config.elm_cell,
        elm_threshold=config.elm_threshold,
        radius_filter=config.radius_filter,
        radius_filter_radius=config.radius_filter_radius,
        radius_filter_min_count=config.radius_filter_min_count,
        reproject_vertical=config.preprocess_reproject_vertical,
        target_vertical_epsg=config.preprocess_vertical_target_epsg,
        num_workers=config.num_workers,
        run_name=run_name,
        chunk_size=config.preprocess_chunk_size,
        chunk_overlap=config.preprocess_chunk_overlap,
        config=config
    )

    print(f"\nPreprocessing completed in {str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n")
