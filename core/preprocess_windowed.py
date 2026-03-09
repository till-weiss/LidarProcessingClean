import pdal
import json
import laspy
from pyproj import CRS
from multiprocessing import Pool
from shapely.geometry import box, shape
import os

from shapely.wkt import loads as wkt_loads, dumps as wkt_dumps
from tqdm import tqdm

from core.reprojection import get_utm_epsg, reproject_las, is_utm_crs


def _is_non_empty_las(las_path):
    if not las_path or not os.path.exists(las_path):
        return False
    try:
        with laspy.open(las_path) as las:
            return las.header.point_count > 0
    except Exception:
        return False


def _cleanup_empty_las(las_path):
    if las_path and os.path.exists(las_path) and not _is_non_empty_las(las_path):
        os.remove(las_path)
        return True
    return False


def _get_combined_geom(gdf):
    if gdf is None or gdf.empty:
        return None
    geom = gdf.geometry.unary_union
    if hasattr(geom, "is_empty") and geom.is_empty:
        return None
    return geom

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
    ref_crs=None,            # horizontal EPSG, e.g. 32632 or 4326
):
    """Process a chunk: convert ellipsoid→EGM2008 orthometric (same horizontal), crop, filter, write."""
    # build the chunk filename
    base = os.path.splitext(os.path.basename(input_file))[0]
    chunk_file = os.path.join(
        temp_dir,
        f"{base}_chunk_{int(chunk_bbox.bounds[0])}_{int(chunk_bbox.bounds[1])}.las"
    )

    #find which vertical-ellipsoid code goes with your horizontal CRS
    hcrs = CRS.from_epsg(ref_crs)
    datum = hcrs.datum.name.lower()
    if "wgs 84" in datum:
        vert_ellipsoid = 4979
    elif "etrs89" in datum:
        vert_ellipsoid = 4936
    else:
        # fallback to WGS84 ellipsoid
        vert_ellipsoid = 4979

    in_srs  = f"EPSG:{ref_crs}+{vert_ellipsoid}"
    out_srs = f"EPSG:{ref_crs}+3855"   # same horizontal, EGM2008 vertical

 
    pipeline = [
        # read raw LAS (ellipsoidal heights)
        {"type": "readers.las", "filename": input_file},

        # reproject vertical only: ellipsoid→EGM2008
        {"type": "filters.reprojection",
         "in_srs":  in_srs,
         "out_srs": out_srs
        },

        # crop to this chunk
        {"type": "filters.crop", "polygon": wkt_dumps(chunk_bbox)},

        # remove statistical outliers
        {"type": "filters.outlier",
         "method": "statistical",
         "mean_k": sor_knn,
         #"multiplier": sor_multiplier
        },

        # clamp to your Z-range (now orthometric)
        #{"type": "filters.range", "limits": f"Z[{min_z}:{max_z}]"},

        # drop noise class 7
        {"type": "filters.range", "limits": "Classification![7:7]"},

        # write the chunk, tagging the compound CRS
        {"type": "writers.las",
         "filename": chunk_file,
         "scale_x":  str(ref_scale[0]),
         "scale_y":  str(ref_scale[1]),
         "scale_z":  str(ref_scale[2]),
         "offset_x": str(ref_offset[0]),
         "offset_y": str(ref_offset[1]),
         "offset_z": str(ref_offset[2]),
         "a_srs":    out_srs
        },
    ]

    # 3) execute
    try:
        pdal.Pipeline(json.dumps(pipeline)).execute()
        return chunk_file
    except Exception as e:
        print(f"Error processing chunk {chunk_file}: {e}")
        return None


def merge_and_crop_chunks(chunk_files, target_geom_wkt, output_file):
    """Merge processed chunks and crop them to the target geometry."""
    target_geom = wkt_loads(target_geom_wkt)
    
    valid_chunk_files = []
    for f in chunk_files:
        if _is_non_empty_las(f):
            valid_chunk_files.append(f)
        elif os.path.exists(f):
            print(f"Warning: Removing empty intermediate file: {f}")
            os.remove(f)

    if not valid_chunk_files:
        print("Warning: No non-empty chunk files available for merging.")
        return None

    pipeline = [{"type": "readers.las", "filename": f} for f in valid_chunk_files]
    pipeline.append({"type": "filters.merge"})
    pipeline.append({"type": "filters.crop", "polygon": wkt_dumps(target_geom)})
    pipeline.append({"type": "writers.las", "filename": output_file})
    
    try:
        pdal.pipeline.Pipeline(json.dumps(pipeline)).execute()
        if _cleanup_empty_las(output_file):
            print(f"Warning: Removed empty merged output file: {output_file}")
            return None
        return output_file
    except Exception as e:
        print(f"Error merging and cropping: {e}")
        return None


def preprocess_window(strip_files, config, target_fp, run_merged_dir, gdf):
    processed_dir = os.path.join(run_merged_dir, target_fp, "processed_strips")
    temp_dir = os.path.join(run_merged_dir, target_fp, "temp")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    input_count = len(strip_files)
    processed = []
    skipped_no_intersection = 0
    skipped_zero_points = 0

    aoi_source = gdf.copy()

    for strip_file in strip_files:
        strip_input_file = strip_file
        strip_base_name = os.path.splitext(os.path.basename(strip_file))[0]

        try:
            if not is_utm_crs(strip_input_file):
                utm_strip = os.path.join(temp_dir, f"{strip_base_name}_utm.las")
                strip_input_file = reproject_las(strip_input_file, utm_strip)

            with laspy.open(strip_input_file) as las:
                header = las.header
                minx, miny, _ = header.mins
                maxx, maxy, _ = header.maxs
                strip_crs = header.parse_crs()
                ref_scale = header.scales
                ref_offset = header.offsets
                ref_crs = strip_crs.to_epsg() if strip_crs else 4979
                points_before = int(header.point_count)

            strip_extent = box(minx, miny, maxx, maxy)

            aoi_gdf = aoi_source.copy()
            if aoi_gdf.crs and strip_crs and aoi_gdf.crs != strip_crs:
                aoi_gdf = aoi_gdf.to_crs(strip_crs)

            aoi_geom = _get_combined_geom(aoi_gdf)
            if aoi_geom is None:
                skipped_no_intersection += 1
                print(f"Warning: AOI geometry is empty for {target_fp}. Skipping strip {strip_file}.")
                continue

            process_geom = aoi_geom.intersection(strip_extent)

            if process_geom.is_empty:
                skipped_no_intersection += 1
                print(f"Warning: No AOI intersection for strip {strip_file}. Skipping.")
                continue

            output_path = os.path.join(processed_dir, f"{strip_base_name}_processed.laz")

            hcrs = CRS.from_epsg(ref_crs)
            datum = hcrs.datum.name.lower()
            if "wgs 84" in datum:
                vert_ellipsoid = 4979
            elif "etrs89" in datum:
                vert_ellipsoid = 4936
            else:
                vert_ellipsoid = 4979

            in_srs = f"EPSG:{ref_crs}+{vert_ellipsoid}"
            out_srs = f"EPSG:{ref_crs}+3855"

            pipeline = [
                {"type": "readers.las", "filename": strip_input_file},
                {"type": "filters.reprojection", "in_srs": in_srs, "out_srs": out_srs},
                {"type": "filters.crop", "polygon": wkt_dumps(process_geom)},
                {"type": "filters.outlier", "method": "statistical", "mean_k": config.knn},
                {"type": "filters.range", "limits": "Classification![7:7]"},
                {
                    "type": "writers.las",
                    "filename": output_path,
                    "scale_x": str(ref_scale[0]),
                    "scale_y": str(ref_scale[1]),
                    "scale_z": str(ref_scale[2]),
                    "offset_x": str(ref_offset[0]),
                    "offset_y": str(ref_offset[1]),
                    "offset_z": str(ref_offset[2]),
                    "a_srs": out_srs,
                },
            ]

            pdal.Pipeline(json.dumps(pipeline)).execute()

            if _cleanup_empty_las(output_path):
                skipped_zero_points += 1
                print(f"Warning: Strip produced empty output and was removed: {strip_file}")
                continue

            with laspy.open(output_path) as out_las:
                points_after = int(out_las.header.point_count)

            if points_after == 0:
                os.remove(output_path)
                skipped_zero_points += 1
                print(f"Warning: Strip has zero valid points after processing: {strip_file}")
                continue

            print(
                f"Processed strip | input: {strip_file} | points before: {points_before} | "
                f"points after: {points_after} | output: {output_path}"
            )
            processed.append(output_path)

        except Exception as e:
            print(f"Warning: Failed to preprocess strip {strip_file}: {e}")

    print(
        f"Strip preprocessing summary for {target_fp}: "
        f"input={input_count}, processed={len(processed)}, "
        f"skipped_no_intersection={skipped_no_intersection}, "
        f"skipped_zero_points={skipped_zero_points}"
    )

    return processed

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
