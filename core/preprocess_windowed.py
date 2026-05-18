import pdal
import json
import laspy
import numpy as np
from pyproj import CRS
from multiprocessing import Pool
from shapely.geometry import box, shape
import os

from shapely.wkt import loads as wkt_loads, dumps as wkt_dumps
from tqdm import tqdm
import config.config as config
from core.reprojection import get_utm_epsg, reproject_las, is_utm_crs


def create_chunks_from_wkt(target_geom_wkt, chunk_size=100):
    """Create grid chunks based on the bounding box of the target geometry."""
    target_geom = wkt_loads(target_geom_wkt)
    min_x, min_y, max_x, max_y = target_geom.bounds

    enlarged_size = chunk_size * (1 + overlap)
    half_extra = (enlarged_size - chunk_size) / 2

    large_chunks = []
    orig_chunks = []

    for x in np.arange(min_x, max_x, chunk_size):
        for y in np.arange(min_y, max_y, chunk_size):
            orig_chunk = box(x, y, x + chunk_size, y + chunk_size)
            large_chunk = box(
                x - half_extra, y - half_extra,
                x + enlarged_size - half_extra, y + enlarged_size - half_extra
            )
            if target_geom.intersects(orig_chunk):
                large_chunks.append(large_chunk)
                orig_chunks.append(orig_chunk)

    return large_chunks, orig_chunks


def process_chunk(
    input_file,
    large_chunk_bbox,
    small_chunk_bbox,
    temp_dir,
    max_z,
    min_z,
    sor_knn=100,
    sor_multiplier=1.0,
    sor_passes=3,
    elm_filter=True,
    elm_cell=10.0,
    elm_threshold=1.0,
    radius_filter=False,
    radius_filter_radius=1.0,
    radius_filter_min_count=4,
    reproject_vertical=True,
    target_vertical_epsg=3855,
    ref_scale=None,
    ref_offset=None,
    ref_crs=None,            # horizontal EPSG, e.g. 32632 or 4326
):
    """
    Process a preprocessing chunk:
      1. Reproject ellipsoid heights → EGM2008 orthometric
      2. Crop to large bbox (with overlap) so filters have full neighbourhood context
      3. Optionally apply ELM (Extended Local Minimum) to mark low-noise points
      4. Apply SOR (Statistical Outlier Removal) N times
      5. Optionally apply radius-based outlier removal
      6. Drop noise-classified points (class 7)
      7. Crop to original tile bbox and write
    """
    base = os.path.splitext(os.path.basename(input_file))[0]
    chunk_file = os.path.join(
        temp_dir,
        f"{base}_chunk_{int(small_chunk_bbox.bounds[0])}_{int(small_chunk_bbox.bounds[1])}.las"
    )

    pipeline = [
        # 1. Read raw LAS (ellipsoidal heights)
        {"type": "readers.las", "filename": input_file},

        # 2. Crop to the enlarged bbox so filters have full neighbourhood context
        {"type": "filters.crop", "polygon": wkt_dumps(large_chunk_bbox)},
    ]

    writer_srs = f"EPSG:{ref_crs}"
    if reproject_vertical:
        # Determine the vertical ellipsoid that matches the horizontal datum.
        hcrs = CRS.from_epsg(ref_crs)
        datum = hcrs.datum.name.lower()
        if "wgs 84" in datum:
            vert_ellipsoid = 4979
        elif "etrs89" in datum:
            vert_ellipsoid = 4936
        else:
            vert_ellipsoid = 4979  # fallback

        in_srs = f"EPSG:{ref_crs}+{vert_ellipsoid}"
        out_srs = f"EPSG:{ref_crs}+{int(target_vertical_epsg)}"
        pipeline.insert(1, {"type": "filters.reprojection", "in_srs": in_srs, "out_srs": out_srs})
        writer_srs = out_srs

    # 4. ELM: marks isolated low points (below-ground noise) as class 7
    if elm_filter:
        pipeline.append({
            "type": "filters.elm",
            "cell": float(elm_cell),
            "threshold": float(elm_threshold),
        })

    # 5. SOR applied sor_passes times for progressive outlier removal
    for _ in range(max(1, int(sor_passes))):
        pipeline.append({
            "type": "filters.outlier",
            "method": "statistical",
            "mean_k": int(sor_knn),
            "multiplier": float(sor_multiplier),
        })

    # 6. Radius-based outlier removal (optional additional filter)
    if radius_filter:
        pipeline.append({
            "type": "filters.outlier",
            "method": "radius",
            "radius": float(radius_filter_radius),
            "min_k": int(radius_filter_min_count),
        })

    pipeline += [
        # 7. Drop all noise-classified points (class 7, set by ELM and SOR above)
        {"type": "filters.range", "limits": "Classification![7:7]"},

        # 8. Crop back to the true tile bbox (removes the overlap border)
        {"type": "filters.crop", "polygon": wkt_dumps(small_chunk_bbox)},

        # 9. Write with original scale/offset and the configured CRS
        {"type": "writers.las",
         "filename": chunk_file,
         "scale_x":  str(ref_scale[0]),
         "scale_y":  str(ref_scale[1]),
         "scale_z":  str(ref_scale[2]),
         "offset_x": str(ref_offset[0]),
         "offset_y": str(ref_offset[1]),
         "offset_z": str(ref_offset[2]),
         "a_srs":    writer_srs},
    ]

    try:
        pdal.Pipeline(json.dumps(pipeline)).execute()
        return chunk_file
    except Exception as e:
        print(f"Error processing chunk {chunk_file}: {e}")
        return None

def merge_and_crop_chunks(chunk_files, target_geom_wkt, output_file):
    """Merge processed chunks and crop them to the target geometry."""
    target_geom = wkt_loads(target_geom_wkt)
    
    pipeline = [{"type": "readers.las", "filename": f} for f in chunk_files]
    pipeline.append({"type": "filters.merge"})
    pipeline.append({"type": "filters.crop", "polygon": wkt_dumps(target_geom)})
    pipeline.append({"type": "writers.las", "filename": output_file})
    
    try:
        pdal.pipeline.Pipeline(json.dumps(pipeline)).execute()
        return output_file
    except Exception as e:
        print(f"Error merging and cropping: {e}")
        return None

def process_strip(
    input_file,
    target_gdf,
    out_dir,
    max_z,
    min_z,
    sor_knn=8,
    sor_multiplier=2.0,
    ref_scale=None,
    ref_offset=None,
    ref_crs=None,            # horizontal EPSG, e.g. 32632 or 4326
):
    """Process one strip: vertical transform, crop, filter, write."""
    # build the chunk filename
    base = os.path.splitext(os.path.basename(input_file))[0]
    strip_file = os.path.join(
        out_dir,
        f"{base}_aoi.laz"
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
        
        {"type": "filters.crop", "polygon": target_gdf},

        # reproject vertical only: ellipsoid→EGM2008
        {"type": "filters.reprojection",
         "in_srs":  in_srs,
         "out_srs": out_srs
        },

        # crop to this chunk
        {"type": "filters.crop", "polygon": target_gdf},

        # remove statistical outliers
        {"type": "filters.outlier",
         "method": "statistical",
         "mean_k": sor_knn,
         "multiplier": sor_multiplier
        },

        # clamp to your Z-range (now orthometric)
        {"type": "filters.range", "limits": f"Z[{min_z}:{max_z}]"},

        # drop noise class 7
        {"type": "filters.range", "limits": "Classification![7:7]"},

        # write the chunk, tagging the compound CRS
        {"type": "writers.las",
         "filename": strip_file,
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
        return strip_file
    except Exception as e:
        print(f"Error processing chunk {strip_file}: {e}")
        return None

def merge_and_crop_strips(strip_files, target_geom_wkt, output_file):
    """Merge processed chunks and crop them to the target geometry."""
    target_geom = wkt_loads(target_geom_wkt)
    
    pipeline = [{"type": "readers.las", "filename": f} for f in strip_files]
    pipeline.append({"type": "filters.merge"})
    pipeline.append({"type": "filters.crop", "polygon": wkt_dumps(target_geom)})
    pipeline.append({"type": "writers.las", "filename": output_file})
    
    try:
        pdal.pipeline.Pipeline(json.dumps(pipeline)).execute()
        return output_file
    except Exception as e:
        print(f"Error merging and cropping: {e}")
        return None

def process_las_files(las_dict, preprocessed_dir, num_workers=4, chunk_size=100, sor_knn=8, sor_multiplier=2.0):
def process_las_files(
    las_dict, preprocessed_dir, num_workers=4,
    chunk_size=500, chunk_overlap=0.1,
    sor_knn=100, sor_multiplier=1.0, sor_passes=3,
    elm_filter=True, elm_cell=10.0, elm_threshold=1.0,
    radius_filter=False, radius_filter_radius=1.0, radius_filter_min_count=4,
):
    """Process multiple LAS files for different target areas."""
    os.makedirs(preprocessed_dir, exist_ok=True)

    for target_area, las_files in las_dict.items():
        temp_dir = os.path.join(preprocessed_dir, target_area, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        gdf, input_files = las_files['gdf'], las_files['files']
        target_geom_wkt = wkt_dumps(shape(gdf.geometry.iloc[0]))
        large_chunks, orig_chunks = create_chunks_from_wkt(target_geom_wkt, chunk_size, chunk_overlap)

        process_args = []
        for input_file in input_files:
            ref_scale, ref_offset, ref_crs = get_las_header(input_file)
            for large_chunk, orig_chunk in zip(large_chunks, orig_chunks):
                process_args.append((
                    input_file, large_chunk, orig_chunk, temp_dir, None, None,
                    sor_knn, sor_multiplier, sor_passes,
                    elm_filter, elm_cell, elm_threshold,
                    radius_filter, radius_filter_radius, radius_filter_min_count,
                    ref_scale, ref_offset, ref_crs,
                ))

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