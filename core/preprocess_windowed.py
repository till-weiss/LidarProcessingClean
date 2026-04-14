import pdal
import json
import laspy
from pyproj import CRS
from multiprocessing import Pool
from shapely.geometry import box, shape
import os

from shapely.wkt import loads as wkt_loads, dumps as wkt_dumps
from tqdm import tqdm
import config.config as config
from core.reprojection import get_utm_epsg, reproject_las, is_utm_crs


def create_chunks_from_wkt(target_geom_wkt, chunk_size=100, buffer_size=0.0):
    """Create core and buffered chunks based on target geometry bounds."""
    target_geom = wkt_loads(target_geom_wkt)
    min_x, min_y, max_x, max_y = target_geom.bounds
    
    core_chunks = []
    buffered_chunks = []
    buffer_size = max(0.0, float(buffer_size))
    for x in range(int(min_x), int(max_x), chunk_size):
        for y in range(int(min_y), int(max_y), chunk_size):
            core_bbox = box(x, y, x + chunk_size, y + chunk_size)
            if target_geom.intersects(core_bbox):
                buffered_bbox = box(
                    x - buffer_size,
                    y - buffer_size,
                    x + chunk_size + buffer_size,
                    y + chunk_size + buffer_size,
                )
                core_chunks.append(core_bbox)
                buffered_chunks.append(buffered_bbox)
    
    return core_chunks, buffered_chunks


def process_chunk(
    input_file,
    core_chunk_bbox,
    buffered_chunk_bbox,
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
        f"{base}_chunk_{int(core_chunk_bbox.bounds[0])}_{int(core_chunk_bbox.bounds[1])}.las"
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

        # crop to buffered chunk (processing context)
        {"type": "filters.crop", "polygon": wkt_dumps(buffered_chunk_bbox)},

        # remove statistical outliers
        {"type": "filters.outlier",
         "method": "statistical",
         "mean_k": sor_knn,
         "multiplier": sor_multiplier
        },

        # clamp to your Z-range (now orthometric)
        #{"type": "filters.range", "limits": f"Z[{min_z}:{max_z}]"},

        # drop noise class 7
        {"type": "filters.range", "limits": "Classification![7:7]"},

        # crop back to core chunk (no overlap in final tiles)
        {"type": "filters.crop", "polygon": wkt_dumps(core_chunk_bbox)},

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

def process_las_files(las_dict, preprocessed_dir, num_workers=4, chunk_size=100, buffer_size=0.0, sor_knn=8, sor_multiplier=2.0):
    """Process multiple LAS files for different target areas."""
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    for target_area, las_files in las_dict.items():
        temp_dir = os.path.join(preprocessed_dir, target_area, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        gdf, input_files = las_files['gdf'], las_files['files']
        target_geom_wkt = wkt_dumps(shape(gdf.geometry.iloc[0]))
        core_chunks, buffered_chunks = create_chunks_from_wkt(target_geom_wkt, chunk_size, buffer_size=buffer_size)
        
        process_args = []
        for input_file in input_files:
            ref_scale, ref_offset, ref_crs = get_las_header(input_file)
            for core_chunk, buffered_chunk in zip(core_chunks, buffered_chunks):
                process_args.append((input_file, core_chunk, buffered_chunk, temp_dir, sor_knn, sor_multiplier, ref_scale, ref_offset, ref_crs))
        
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
