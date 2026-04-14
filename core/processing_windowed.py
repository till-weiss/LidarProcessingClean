import laspy
import pdal
import numpy as np
import pandas as pd
import os
import rasterio
from rasterio.merge import merge
from osgeo import gdal
import shutil
import json
import subprocess
from shapely.geometry import box, shape
from shapely.wkt import loads as wkt_loads, dumps as wkt_dumps

def create_chunks_from_wkt(input_wkt, chunk_size=1000, overlap=0.2, buffer_size=None):
    """
    Create processing chunks from AOI WKT.
    Returns buffered chunk bounds (for processing context) and core chunk bounds
    (for final clipping / merge without overlap).

    Parameters
    ----------
    input_wkt : str
        AOI geometry as WKT.
    chunk_size : float
        Core chunk edge length in map units.
    overlap : float
        Backward-compatible fractional overlap. Used only when buffer_size is None.
    buffer_size : float | None
        Buffer distance (map units) added to each chunk on all sides.
    """

    geom= wkt_loads(input_wkt)
    min_x, min_y, max_x, max_y = geom.bounds
    buffered_chunks = []
    core_chunks = []

    if buffer_size is None:
        # Keep compatibility with existing configs based on fractional overlap.
        buffer_size = (chunk_size * float(overlap)) / 2.0
    buffer_size = max(0.0, float(buffer_size))

    for x in np.arange(min_x, max_x, chunk_size):
        for y in np.arange(min_y, max_y, chunk_size):
            core_chunk_bbox = box(x, y, x + chunk_size, y + chunk_size)
            if not geom.intersects(core_chunk_bbox):
                continue

            buffered_chunk_bbox = box(
                x - buffer_size,
                y - buffer_size,
                x + chunk_size + buffer_size,
                y + chunk_size + buffer_size,
            )
            buffered_chunks.append(buffered_chunk_bbox)
            core_chunks.append(core_chunk_bbox)

    return buffered_chunks, core_chunks

def process_chunk_to_dsm(input_file, large_chunk_bbox, small_chunk_bbox, temp_dir, resolution):

    chunk_file = os.path.join(
        temp_dir,
        f"{os.path.basename(input_file).replace('.las', '')}_chunk_{int(small_chunk_bbox.bounds[0])}_{int(small_chunk_bbox.bounds[1])}.tif"
    )

    pipeline = [
        {"type": "readers.las", "filename": input_file},
        {"type": "filters.crop", "polygon": wkt_dumps(large_chunk_bbox)},
        {"type": "filters.ferry", "dimensions": "Z=>Elevation"},
        {
            "type": "filters.range",
            "limits": "Classification![7:7]"  # Use all points for initial DSM, except noise
        },
        {"type": "filters.crop", "polygon": wkt_dumps(small_chunk_bbox)},
        {
            "type": "writers.gdal",
            "filename": chunk_file,
            "resolution": resolution,
            "output_type": "max",
            "nodata": -9999,
            "gdalopts": "COMPRESS=LZW"
        }
    ]

    try:
        #print("[INFO] Running PDAL pipeline...")
        pdal.pipeline.Pipeline(json.dumps(pipeline)).execute()
        #print("[INFO] PDAL execution completed.")
    except RuntimeError as e:
        f"[ERROR] PDAL execution failed: {e}. Empty chunk."
        return None

    try:
        subprocess.run([
            "gdal_fillnodata.py",
            "-md", "10",
            "-si", "2",
            chunk_file,
            chunk_file
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        #print("[INFO] Nodata gaps filled with GDAL.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] GDAL fillnodata failed: {e}")


def process_chunk_to_dem(input_file, large_chunk_bbox, small_chunk_bbox, temp_dir, rigidness, iterations, resolution, time_step, cloth_resolution=1, fill_gaps=True, filter_smrf=False, scalar=None, slope=None, window=None, threshold=None, filter_csf=False):

    chunk_file = os.path.join(
        temp_dir,
        f"{os.path.basename(input_file).replace('.las', '')}_chunk_{int(small_chunk_bbox.bounds[0])}_{int(small_chunk_bbox.bounds[1])}.tif"
    )

    pipeline = [
        {"type": "readers.las", "filename": input_file},
        {"type": "filters.crop", "polygon": wkt_dumps(large_chunk_bbox)},

    ]
    if filter_smrf:

        pipeline.append({"type": "filters.smrf",
         "scalar": float(scalar),
         "slope": float(slope),
         "window": float(window)})
        
    if filter_csf:
        pipeline.append(
        {"type": "filters.csf",
         "resolution": float(cloth_resolution),
         "rigidness": int(rigidness),
         "iterations": int(iterations),
         "step": float(time_step)})
        
    pipeline += [
        {"type": "filters.ferry", "dimensions": "Z=>Elevation"},
        {"type": "filters.range", "limits": "Classification[2:2]"},
        {"type": "filters.crop", "polygon": wkt_dumps(small_chunk_bbox)},
        {"type": "writers.gdal",
         "filename": chunk_file,
         "resolution": float(cloth_resolution),
         "output_type": "idw",
         "nodata": -9999,
         "gdalopts": "COMPRESS=LZW"}
    ]

    try:
        pdal.pipeline.Pipeline(json.dumps(pipeline)).execute()
    except RuntimeError as e:
        f"[INFO] PDAL execution failed: {e}. No Points in chunk after filterering."
        return None

    resampled_file = chunk_file.replace('.tif', '_resampled.tif')
    minx, miny, maxx, maxy = small_chunk_bbox.bounds

    try:
        subprocess.run([
            "gdalwarp",
            "-tr", str(resolution), str(resolution),
            "-r", "bilinear",
            "-tap",
            "-te", str(minx), str(miny), str(maxx), str(maxy),
            "-overwrite",
            chunk_file,
            resampled_file
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        shutil.move(resampled_file, chunk_file)

    except subprocess.CalledProcessError as e:
        print("[ERROR] gdalwarp failed:")
        print(e.stderr.decode('utf-8'))
        return None

    if fill_gaps:
        try:
            subprocess.run([
                "gdal_fillnodata.py",
                "-md", "10",
                "-si", "2",
                chunk_file,
                chunk_file
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] GDAL fillnodata failed: {e}")
            return None

    return chunk_file

def merge_chunks(input_files, output_file):
    """
    Merges multiple raster files into a single raster and saves the output.
    
    Parameters:
        input_files (list): List of file paths to raster files.
        output_file (str): Path to the output merged raster file.
    """
    
    # Open all raster files
    src_files = [rasterio.open(f) for f in input_files]
    
    # Merge rasters
    mosaic, out_transform = merge(src_files)
    
    # Copy metadata from one of the source files
    out_meta = src_files[0].meta.copy()
    
    # Update metadata for the merged raster
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform
    })
    
    # Write the merged raster to disk
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(mosaic)

    nodata_val = out_meta.get("nodata", None)
    
    # Close all source files
    for src in src_files:
        src.close()

    
    #Sprint(f"Merged raster saved at: {output_file}")
