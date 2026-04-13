import json
from pathlib import Path
from typing import Dict, List, Tuple

import laspy
import numpy as np
import pandas as pd
import pdal
from rasterio.transform import from_origin
import xdem


GROUND_CLASS = 2


def find_icp_ready_files(input_dir: str) -> List[Path]:
    """Find LAS/LAZ files containing 'icp_ready' in the filename (sorted for reproducibility)."""
    base = Path(input_dir)
    files = [
        p
        for p in base.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".las", ".laz"}
        and "icp_ready" in p.name.lower()
    ]
    return sorted(files)


def _read_bounds_and_crs(las_path: Path) -> Tuple[Tuple[float, float, float, float], object]:
    with laspy.open(las_path) as las:
        h = las.header
        bounds = (float(h.mins[0]), float(h.mins[1]), float(h.maxs[0]), float(h.maxs[1]))
        crs = h.parse_crs()
    return bounds, crs


def build_template_grid(all_las_files: List[Path], resolution: float = 1.0) -> Dict[str, object]:
    """Build one common grid (extent/shape/transform) for all strip DEMs."""
    if not all_las_files:
        raise ValueError("No LAS/LAZ files provided to build_template_grid().")

    minxs, minys, maxxs, maxys = [], [], [], []
    first_crs = None

    for las_path in all_las_files:
        (minx, miny, maxx, maxy), crs = _read_bounds_and_crs(las_path)
        minxs.append(minx)
        minys.append(miny)
        maxxs.append(maxx)
        maxys.append(maxy)

        if first_crs is None:
            first_crs = crs
        elif crs is not None and first_crs is not None and crs != first_crs:
            raise ValueError(f"CRS mismatch: {las_path} has {crs}, expected {first_crs}.")

    minx = np.floor(min(minxs) / resolution) * resolution
    miny = np.floor(min(minys) / resolution) * resolution
    maxx = np.ceil(max(maxxs) / resolution) * resolution
    maxy = np.ceil(max(maxys) / resolution) * resolution

    width = int(round((maxx - minx) / resolution))
    height = int(round((maxy - miny) / resolution))
    transform = from_origin(minx, maxy, resolution, resolution)

    return {
        "bounds": (minx, miny, maxx, maxy),
        "resolution": resolution,
        "width": width,
        "height": height,
        "transform": transform,
        "crs": first_crs,
    }


def _has_classification(las_path: Path) -> bool:
    with laspy.open(las_path) as las:
        dims = {d.lower() for d in las.header.point_format.dimension_names}
    return "classification" in dims


def rasterise_strip_to_dem(
    las_path: Path,
    template_grid: Dict[str, object],
    output_path: Path,
    nodata: float = -9999.0,
) -> Path:
    """Rasterise one strip to DEM using PDAL writer.gdal with a common template grid."""
    minx, miny, maxx, maxy = template_grid["bounds"]
    resolution = template_grid["resolution"]

    pipeline_steps = [{"type": "readers.las", "filename": str(las_path)}]

    if _has_classification(las_path):
        pipeline_steps.append({"type": "filters.range", "limits": f"Classification[{GROUND_CLASS}:{GROUND_CLASS}]"})

    pipeline_steps.append(
        {
            "type": "writers.gdal",
            "filename": str(output_path),
            "resolution": resolution,
            "bounds": f"([{minx},{maxx}],[{miny},{maxy}])",
            "output_type": "mean",
            "data_type": "float32",
            "gdalopts": "COMPRESS=LZW",
            "nodata": nodata,
            "dimension": "Z",
        }
    )

    pipeline = pdal.Pipeline(json.dumps({"pipeline": pipeline_steps}))
    pipeline.execute()
    return output_path


def compute_dem_stats(reference: xdem.DEM, target: xdem.DEM) -> Dict[str, float]:
    """Compute basic dDEM diagnostics on overlapping valid pixels only."""
    diff = (target - reference).data
    values = np.asarray(diff.compressed())
    if values.size == 0:
        return {"mean": np.nan, "median": np.nan, "nmad": np.nan, "count": 0}

    median = float(np.nanmedian(values))
    nmad = float(1.4826 * np.nanmedian(np.abs(values - median)))
    return {
        "mean": float(np.nanmean(values)),
        "median": median,
        "nmad": nmad,
        "count": int(values.size),
    }


def coregister_dems(
    dem_paths: List[Path],
    output_dir: Path,
    save_diagnostics: bool = True,
) -> Tuple[List[Path], Path]:
    """Coregister DEMs with xDEM Nuth & Kääb using first DEM as reference."""
    if len(dem_paths) < 2:
        raise ValueError("Need at least two DEMs for coregistration.")

    output_dir.mkdir(parents=True, exist_ok=True)
    aligned_paths: List[Path] = [dem_paths[0]]
    reference = xdem.DEM(str(dem_paths[0]))
    stats_rows = []

    for target_path in dem_paths[1:]:
        target = xdem.DEM(str(target_path))

        before = compute_dem_stats(reference, target)

        coreg = xdem.coreg.NuthKaab()
        coreg.fit(reference, target)
        aligned = coreg.apply(target)

        aligned_name = f"{target_path.stem}_aligned.tif"
        aligned_path = output_dir / aligned_name
        aligned.save(str(aligned_path))

        params_path = output_dir / f"{target_path.stem}_coreg_params.json"
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(coreg.to_matrix().tolist(), f, indent=2)

        after = compute_dem_stats(reference, aligned)

        stats_rows.append(
            {
                "reference_dem": dem_paths[0].name,
                "target_dem": target_path.name,
                "aligned_dem": aligned_name,
                "pre_mean": before["mean"],
                "pre_median": before["median"],
                "pre_nmad": before["nmad"],
                "post_mean": after["mean"],
                "post_median": after["median"],
                "post_nmad": after["nmad"],
                "valid_overlap_count": after["count"],
            }
        )

        aligned_paths.append(aligned_path)

    diagnostics_path = output_dir / "coregistration_diagnostics.csv"
    if save_diagnostics:
        pd.DataFrame(stats_rows).to_csv(diagnostics_path, index=False)

    return aligned_paths, diagnostics_path


def main(
    input_dir: str,
    output_dir: str,
    resolution: float = 1.0,
    nodata: float = -9999.0,
    save_diagnostics: bool = True,
) -> None:
    """Workflow: find icp_ready strips -> rasterize to common-grid DEMs -> xDEM coregistration."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    dem_dir = output_path / "strip_dems"
    aligned_dir = output_path / "coregistered_dems"
    dem_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir.mkdir(parents=True, exist_ok=True)

    las_files = find_icp_ready_files(str(input_path))
    if not las_files:
        raise FileNotFoundError(f"No *icp_ready*.las/.laz files found in {input_path}.")

    template_grid = build_template_grid(las_files, resolution=resolution)

    dem_paths: List[Path] = []
    for i, las_path in enumerate(las_files, start=1):
        dem_path = dem_dir / f"dem_strip_{i:03d}.tif"
        rasterise_strip_to_dem(las_path, template_grid=template_grid, output_path=dem_path, nodata=nodata)
        dem_paths.append(dem_path)

    coregister_dems(dem_paths, output_dir=aligned_dir, save_diagnostics=save_diagnostics)


if __name__ == "__main__":
    # Example usage; adjust paths to your environment.
    main(input_dir="icp_ready", output_dir="outputs/dem_coreg", resolution=1.0)
