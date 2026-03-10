import pdal
import json
import laspy
from pyproj import CRS
from multiprocessing import Pool
from shapely.geometry import box, shape
import os
import numpy as np 

from shapely.wkt import loads as wkt_loads, dumps as wkt_dumps
from tqdm import tqdm

from core.reprojection import get_utm_epsg, reproject_las, is_utm_crs

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def create_chunks_from_wkt(target_geom_wkt, chunk_size=100):
    """Create grid chunks based on the bounding box of the target geometry."""
    target_geom = wkt_loads(target_geom_wkt)
    min_x, min_y, max_x, max_y = target_geom.bounds
    
    chunks = []
    for x in range(int(min_x), int(max_x), chunk_size):
        for y in range(int(min_y), int(max_y), chunk_size):
            chunk_bbox = box(x, y, x + chunk_size, y + chunk_size)
            if target_geom.intersects(chunk_bbox):
                chunks.append(chunk_bbox)
    
    return chunks

def has_points(las_file):
    try:
        return laspy.open(las_file).header.point_count > 0
    except Exception:
        return False


def get_las_bounds_wkt(las_file):
    with laspy.open(las_file) as las:
        min_x, min_y = las.header.mins[0], las.header.mins[1]
        max_x, max_y = las.header.maxs[0], las.header.maxs[1]
    return wkt_dumps(box(min_x, min_y, max_x, max_y))

def build_strip_chunk_tasks(
    strip_path,
    chunks,
    temp_dir,
    max_z,
    min_z,
    sor_knn,
    sor_multiplier,
    ref_scale,
    ref_offset,
    ref_crs,
):
    """
    Build tasks while preserving strip provenance.
    Returns:
        [(strip_path, chunk_args_tuple), ...]
    """
    return [
        (
            strip_path,
            (
                strip_path,
                chunk,
                temp_dir,
                max_z,
                min_z,
                sor_knn,
                sor_multiplier,
                ref_scale,
                ref_offset,
                ref_crs,
            ),
        )
        for chunk in chunks
    ]

def process_chunk_with_provenance(task):
    """
    task = (strip_path, chunk_args)
    returns (strip_path, processed_chunk_path)
    """
    strip_path, chunk_args = task
    processed_chunk = process_chunk(*chunk_args)
    return strip_path, processed_chunk


def group_chunks_by_strip(processed_results):
    """
    Input:
        [(strip_path, chunk_file), ...]

    Output:
        {
            strip_path: [chunk1, chunk2, ...],
            ...
        }
    """
    grouped = {}
    for strip_path, chunk_file in processed_results:
        if chunk_file and has_points(chunk_file):
            grouped.setdefault(strip_path, []).append(chunk_file)
    return grouped



def merge_and_crop_chunks(chunk_files, target_geom_wkt, output_file):
    """
    Original chunk mode:
    merge all processed chunks and crop to AOI.
    """
    target_geom = wkt_loads(target_geom_wkt)

    pipeline = [{"type": "readers.las", "filename": f} for f in chunk_files]
    pipeline.append({"type": "filters.merge"})
    pipeline.append({"type": "filters.crop", "polygon": wkt_dumps(target_geom)})

    writer = {"type": "writers.las", "filename": output_file}
    if output_file.lower().endswith(".laz"):
        writer["compression"] = "laszip"
    pipeline.append(writer)

    try:
        pdal.Pipeline(json.dumps(pipeline)).execute()
        return output_file
    except Exception as e:
        print(f"Error merging and cropping: {e}")
        return None

def merge_chunks_to_strip(chunk_files, output_file, crop_geom_wkt=wkt_dumps(process_geom)):
    """
    Merge chunk files into one strip-level LAS/LAZ file.
    """
    pipeline = [{"type": "readers.las", "filename": f} for f in chunk_files]
    pipeline.append({"type": "filters.merge"})

    if crop_geom_wkt:
        pipeline.append({"type": "filters.crop", "polygon": crop_geom_wkt})

    writer = {"type": "writers.las", "filename": output_file}
    if output_file.lower().endswith(".laz"):
        writer["compression"] = "laszip"
    pipeline.append(writer)

    try:
        pdal.Pipeline(json.dumps(pipeline)).execute()
        return output_file
    except Exception as e:
        print(f"Error merging chunks to strip {output_file}: {e}")
        return None


def merge_chunks_for_strip(strip_output_name, strip_chunks, strips_dir, crop_geom_wkt=None):
    strip_output_file = os.path.join(strips_dir, f"{strip_output_name}_processed.laz")
    return merge_chunks_to_strip(strip_chunks, strip_output_file, crop_geom_wkt=crop_geom_wkt)


def process_las_files(las_dict, preprocessed_dir, num_workers=4, chunk_size=100, sor_knn=8, sor_multiplier=2.0):
    """Process multiple LAS files for different target areas."""
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    for target_area, las_files in las_dict.items():
        temp_dir = os.path.join(preprocessed_dir, target_area, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        gdf, input_files = las_files['gdf'], las_files['files']
        target_geom_wkt = wkt_dumps(shape(gdf.geometry.iloc[0]))
        chunks = create_chunks_from_wkt(target_geom_wkt, chunk_size)
        
        process_args = []
        for input_file in input_files:
            ref_scale, ref_offset, ref_crs = get_las_header(input_file)
            for chunk in chunks:
                process_args.append((input_file, chunk, temp_dir, sor_knn, sor_multiplier, ref_scale, ref_offset, ref_crs))
        
        processed_chunks = []
        with tqdm(total=len(process_args), desc=f"Processing {target_area}", unit="chunk") as pbar:
            with Pool(num_workers) as pool:
                for processed_chunk in pool.imap_unordered(lambda args: process_chunk(*args), process_args):
                    if processed_chunk:
                        processed_chunks.append(processed_chunk)
                    pbar.update(1)
        
        if processed_chunks:
            final_output_file = os.path.join(preprocessed_dir, f"{target_area}_processed.las")
            merge_and_crop_chunks(processed_chunks, target_geom_wkt, final_output_file)
        else:
            print(f"No processed chunks available for {target_area}.")

def get_las_header(las_file):
    """Extracts scale, offset, and CRS from an input LAS file."""
    with laspy.open(las_file) as las:
        header = las.header
        scale = header.scales
        offset = header.offsets
        crs = header.parse_crs()
        crs_epsg = crs.to_epsg() if crs else 4979
    return scale, offset, crs_epsg


# ---------------------------------------------------------------------
# Core chunk processing
# ---------------------------------------------------------------------

def process_chunk(
    input_file,
    chunk_bbox,
    temp_dir,
    max_z,
    min_z,
    sor_knn=8,
    sor_multiplier=2.0,
    ref_scale=None,
    ref_offset=None,
    ref_crs=None,
):
    """
    Process one chunk:
    - read LAS/LAZ
    - crop early to chunk geometry
    - convert ellipsoidal heights to EGM2008 orthometric heights
    - remove statistical outliers
    - remove class 7 noise
    - write chunk output
    """
    base = os.path.splitext(os.path.basename(input_file))[0]
    chunk_file = os.path.join(
        temp_dir,
        f"{base}_chunk_{int(chunk_bbox.bounds[0])}_{int(chunk_bbox.bounds[1])}.laz"
    )

    # derive vertical ellipsoid EPSG from horizontal CRS datum
    hcrs = CRS.from_epsg(ref_crs)
    datum = hcrs.datum.name.lower()

    if "wgs 84" in datum:
        vert_ellipsoid = 4979
    elif "etrs89" in datum:
        vert_ellipsoid = 4936
    else:
        vert_ellipsoid = 4979  # fallback

    in_srs = f"EPSG:{ref_crs}+{vert_ellipsoid}"
    out_srs = f"EPSG:{ref_crs}+3855"  # same horizontal CRS, EGM2008 vertical

    pipeline = [
        {"type": "readers.las", "filename": input_file},

        # crop first so later steps run only on needed points
        {"type": "filters.crop", "polygon": wkt_dumps(chunk_bbox)},

        # vertical transformation
        {
            "type": "filters.reprojection",
            "in_srs": in_srs,
            "out_srs": out_srs,
        },
        # second crop in output CRS to remove leftovers after reprojection
        {"type": "filters.crop", "polygon": wkt_dumps(chunk_bbox)},

        # statistical outlier removal
        {
            "type": "filters.outlier",
            "method": "statistical",
            "mean_k": sor_knn,
            "multiplier": sor_multiplier,
        },

        # optional z clamp if you want it
        {"type": "filters.range", "limits": f"Z[{min_z}:{max_z}]"},

        # remove class 7 noise
        {"type": "filters.range", "limits": "Classification![7:7]"},

        {
            "type": "writers.las",
            "filename": chunk_file,
            "scale_x": str(ref_scale[0]),
            "scale_y": str(ref_scale[1]),
            "scale_z": str(ref_scale[2]),
            "offset_x": str(ref_offset[0]),
            "offset_y": str(ref_offset[1]),
            "offset_z": str(ref_offset[2]),
            "a_srs": out_srs,
        },
    ]

    try:
        pdal.Pipeline(json.dumps(pipeline)).execute()
        return chunk_file
    except Exception as e:
        print(f"Error processing chunk {chunk_file}: {e}")
        return None


def process_chunk_wrapper(args):
    return process_chunk(*args)


# ---------------------------------------------------------------------
# Strip-preserving chunked preprocessing
# ---------------------------------------------------------------------

def preprocess_window(
    strip_files,
    config,
    target_fp,
    run_merged_dir,
    temp_dir,
    target_gdf,
    num_workers,
    chunk_size=1000,
):
    """
    Process each strip in chunks, then merge chunk outputs back into one processed strip file.

    Returns:
        list[str] of processed strip files
    """
    processed_strips_dir = os.path.join(run_merged_dir, target_fp, "processed_strips")
    os.makedirs(processed_strips_dir, exist_ok=True)

    processed_strip_files = []
    process_tasks = []

    print(f"Input strips for {target_fp}: {len(strip_files)}")

    for input_file in strip_files:
        strip_input = input_file

        # ensure horizontal CRS is suitable
        if not is_utm_crs(strip_input):
            base_name = os.path.basename(strip_input)
            base_name = base_name.replace(".las", "_utm.laz").replace(".laz", "_utm.las")
            utm_output_file = os.path.join(temp_dir, base_name)
            strip_input = reproject_las(strip_input, utm_output_file)

        ref_scale, ref_offset, ref_crs = get_las_header(strip_input)

        # AOI -> strip CRS
        target_proj_gdf = (
            target_gdf.to_crs(epsg=ref_crs)
            if target_gdf.crs.to_epsg() != ref_crs
            else target_gdf
        )
        target_geom = shape(target_proj_gdf.geometry.iloc[0])

        # strip extent
        strip_geom_wkt = get_las_bounds_wkt(strip_input)
        strip_geom = wkt_loads(strip_geom_wkt)

        # overlap
        process_geom = target_geom.intersection(strip_geom)
        if process_geom.is_empty:
            print(f"[WARN] Strip does not intersect AOI and is skipped: {strip_input}")
            continue

        chunks = create_chunks_from_wkt(wkt_dumps(process_geom), chunk_size)
        if not chunks:
            print(f"[WARN] No chunks created for strip: {strip_input}")
            continue

        print(f"Chunks created for strip {os.path.basename(strip_input)}: {len(chunks)}")

        # compute per-strip z limits once
        all_z = laspy.read(strip_input).z
        if getattr(config, "max_elevation_threshold", None):
            max_z = np.quantile(all_z, config.max_elevation_threshold)
            min_z = np.quantile(all_z, 1 - config.max_elevation_threshold)
        else:
            max_z = np.max(all_z)
            min_z = np.min(all_z)

        strip_tasks = build_strip_chunk_tasks(
            strip_path=strip_input,
            chunks=chunks,
            temp_dir=temp_dir,
            max_z=max_z,
            min_z=min_z,
            sor_knn=config.knn,
            sor_multiplier=config.multiplier,
            ref_scale=ref_scale,
            ref_offset=ref_offset,
            ref_crs=ref_crs,
        )
        process_tasks.extend(strip_tasks)

    if not process_tasks:
        print(f"No valid processing tasks created for {target_fp}.")
        return []

    processed_results = []
    with tqdm(total=len(process_tasks), desc=f"Processing {target_fp}", unit="chunk") as pbar:
        with Pool(processes=num_workers) as pool:
            for result in pool.imap_unordered(process_chunk_with_provenance, process_tasks):
                processed_results.append(result)
                pbar.update(1)

    grouped_chunks = group_chunks_by_strip(processed_results)

    for strip_path, strip_chunks in grouped_chunks.items():
        if not strip_chunks:
            continue

        strip_name = os.path.splitext(os.path.basename(strip_path))[0]
        strip_output_file = os.path.join(processed_strips_dir, f"{strip_name}_processed.laz")

        merged_strip = merge_chunks_to_strip(
            strip_chunks,
            strip_output_file,
            crop_geom_wkt=None,  # chunks already come from process_geom
        )

        if merged_strip and has_points(merged_strip):
            processed_strip_files.append(merged_strip)
            print(
                f"Processed strip | input: {strip_path} | "
                f"chunks merged: {len(strip_chunks)} | output: {merged_strip}"
            )
        else:
            print(f"[WARN] Failed to create processed strip output for: {strip_path}")
            if merged_strip and os.path.exists(merged_strip):
                os.remove(merged_strip)

        # cleanup temporary chunk files
        for chunk_file in strip_chunks:
            if os.path.exists(chunk_file):
                os.remove(chunk_file)

    print(
        f"Sanity check: input strips={len(strip_files)}, "
        f"processed strips={len(processed_strip_files)}"
    )

    return processed_strip_files