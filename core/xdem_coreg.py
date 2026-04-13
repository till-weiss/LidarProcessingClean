import csv
import glob
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import laspy
import numpy as np
import pdal
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from core.utils import output_exists


@dataclass
class TemplateGrid:
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    resolution: float
    width: int
    height: int
    crs: CRS
    transform: object


# Central step switches for checkpointed strip workflow.
CONFIG = {
    "merge": False,
    "filter": False,
    "crop": False,
    "rasterise": True,
    "coreg_xdem": True,
    "overwrite": False,
}


@dataclass
class PipelineStep:
    name: str
    func: Callable
    outputs: list[Path]
    output_dir: Path
    enabled: bool = True
    dependencies: list[str] = field(default_factory=list)


def step_done_flag(step: PipelineStep) -> Path:
    return step.output_dir / f"{step.name}.done"


def outputs_exist(outputs: list[Path]) -> bool:
    return all(path.exists() for path in outputs)


def is_step_complete(step: PipelineStep) -> bool:
    return step_done_flag(step).exists() and outputs_exist(step.outputs)


def run_step(step: PipelineStep, completed_steps: set[str], overwrite: bool = False) -> None:
    if not step.enabled:
        print(f"[SKIP] {step.name} (disabled)")
        return

    for dependency in step.dependencies:
        if dependency not in completed_steps:
            raise RuntimeError(f"{step.name} requires {dependency}")

    if is_step_complete(step) and not overwrite:
        print(f"[SKIP] {step.name} (already done)")
        completed_steps.add(step.name)
        return

    print(f"[RUN] {step.name}")
    step.output_dir.mkdir(parents=True, exist_ok=True)
    step.func()
    step_done_flag(step).touch()
    completed_steps.add(step.name)


def find_icp_ready_files(input_dir: str, filename_token: str = "icp_ready") -> list[str]:
    """Find and sort LAS/LAZ files containing the expected ICP-ready token."""
    pattern_las = os.path.join(input_dir, "**", "*.las")
    pattern_laz = os.path.join(input_dir, "**", "*.laz")

    candidates = glob.glob(pattern_las, recursive=True) + glob.glob(pattern_laz, recursive=True)
    token = filename_token.lower()
    selected = [p for p in candidates if token in os.path.basename(p).lower()]
    return sorted(selected)


def build_template_grid(all_las_files: list[str], resolution: float = 1.0) -> TemplateGrid:
    """Build a single aligned grid (extent/shape/transform) shared by all DEMs."""
    if not all_las_files:
        raise ValueError("No ICP-ready LAS/LAZ files found.")

    min_x, min_y = np.inf, np.inf
    max_x, max_y = -np.inf, -np.inf
    crs_ref = None

    for las_path in all_las_files:
        with laspy.open(las_path) as src:
            hdr = src.header
            min_x = min(min_x, hdr.mins[0])
            min_y = min(min_y, hdr.mins[1])
            max_x = max(max_x, hdr.maxs[0])
            max_y = max(max_y, hdr.maxs[1])
            las_crs = hdr.parse_crs()

        if las_crs is None:
            raise ValueError(f"Missing CRS in file: {las_path}")

        if crs_ref is None:
            crs_ref = CRS.from_user_input(las_crs)
        elif CRS.from_user_input(las_crs) != crs_ref:
            raise ValueError(f"CRS mismatch: {las_path} differs from the first strip CRS")

    aligned_min_x = np.floor(min_x / resolution) * resolution
    aligned_min_y = np.floor(min_y / resolution) * resolution
    aligned_max_x = np.ceil(max_x / resolution) * resolution
    aligned_max_y = np.ceil(max_y / resolution) * resolution

    width = int(round((aligned_max_x - aligned_min_x) / resolution))
    height = int(round((aligned_max_y - aligned_min_y) / resolution))
    transform = from_origin(aligned_min_x, aligned_max_y, resolution, resolution)

    return TemplateGrid(
        min_x=aligned_min_x,
        min_y=aligned_min_y,
        max_x=aligned_max_x,
        max_y=aligned_max_y,
        resolution=resolution,
        width=width,
        height=height,
        crs=crs_ref,
        transform=transform,
    )


def _enforce_template_grid(raster_path: str, template_grid: TemplateGrid, nodata: float) -> None:
    """Force exact shared transform/shape/extent for every generated DEM."""
    with rasterio.open(raster_path) as src:
        source = src.read(1)
        source_profile = src.profile.copy()
        source_nodata = src.nodata if src.nodata is not None else nodata

        aligned = np.full((template_grid.height, template_grid.width), nodata, dtype=np.float32)
        reproject(
            source=source,
            destination=aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=source_nodata,
            dst_transform=template_grid.transform,
            dst_crs=template_grid.crs,
            dst_nodata=nodata,
            resampling=Resampling.bilinear,
        )

    source_profile.update(
        driver="GTiff",
        height=template_grid.height,
        width=template_grid.width,
        transform=template_grid.transform,
        crs=template_grid.crs,
        dtype="float32",
        nodata=nodata,
        compress="lzw",
    )

    with rasterio.open(raster_path, "w", **source_profile) as dst:
        dst.write(aligned.astype(np.float32), 1)


def rasterise_strip_to_dem(
    las_path: str,
    template_grid: TemplateGrid,
    output_path: str,
    nodata: float = -9999.0,
    use_ground_only: bool = True,
) -> str:
    """Rasterise one ICP-ready strip to DEM using one common template grid."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    bounds = f"([{template_grid.min_x},{template_grid.max_x}],[{template_grid.min_y},{template_grid.max_y}])"
    pipeline = [{"type": "readers.las", "filename": las_path}]

    if use_ground_only:
        pipeline.append({"type": "filters.range", "limits": "Classification[2:2]"})

    pipeline.append(
        {
            "type": "writers.gdal",
            "filename": output_path,
            "resolution": float(template_grid.resolution),
            "bounds": bounds,
            "output_type": "mean",
            "nodata": float(nodata),
            "data_type": "float32",
            "gdalopts": "COMPRESS=LZW",
        }
    )

    pdal.Pipeline(json.dumps(pipeline)).execute()
    _enforce_template_grid(output_path, template_grid, nodata)
    return output_path


def compute_dem_stats(reference, target) -> dict[str, float]:
    """Compute dDEM metrics on overlapping valid pixels only."""
    ref_data = np.asarray(reference.data)
    tgt_data = np.asarray(target.data)
    ref_mask = np.ma.getmaskarray(reference.data)
    tgt_mask = np.ma.getmaskarray(target.data)

    valid = (~ref_mask) & (~tgt_mask) & np.isfinite(ref_data) & np.isfinite(tgt_data)
    if not np.any(valid):
        return {"mean": np.nan, "median": np.nan, "nmad": np.nan, "count": 0}

    diff = tgt_data[valid] - ref_data[valid]
    med = float(np.median(diff))
    nmad = float(1.4826 * np.median(np.abs(diff - med)))

    return {
        "mean": float(np.mean(diff)),
        "median": med,
        "nmad": nmad,
        "count": int(diff.size),
    }


def coregister_dems(dem_paths: list[str], output_dir: str, write_diagnostics: bool = True) -> dict[str, str]:
    """Coregister DEMs strip-wise against the first DEM using xDEM Nuth & Kääb."""
    if len(dem_paths) < 2:
        raise ValueError("At least two DEMs are required for xDEM coregistration.")

    try:
        import xdem
    except ImportError as exc:
        raise ImportError("xDEM is required for DEM coregistration. Install `xdem`.") from exc

    os.makedirs(output_dir, exist_ok=True)
    aligned_dir = os.path.join(output_dir, "aligned")
    os.makedirs(aligned_dir, exist_ok=True)

    reference_dem = xdem.DEM(dem_paths[0])
    reference_name = os.path.splitext(os.path.basename(dem_paths[0]))[0]

    stats_rows = []
    param_records = {}

    for target_path in dem_paths[1:]:
        target_dem = xdem.DEM(target_path)
        target_name = os.path.splitext(os.path.basename(target_path))[0]

        before_stats = compute_dem_stats(reference_dem, target_dem)

        ref_data = np.asarray(reference_dem.data)
        tgt_data = np.asarray(target_dem.data)
        ref_mask = np.ma.getmaskarray(reference_dem.data)
        tgt_mask = np.ma.getmaskarray(target_dem.data)
        overlap_mask = (~ref_mask) & (~tgt_mask) & np.isfinite(ref_data) & np.isfinite(tgt_data)

        coreg = xdem.coreg.NuthKaab()
        coreg.fit(reference_dem, target_dem, inlier_mask=overlap_mask)
        aligned_dem = coreg.apply(target_dem)

        aligned_path = os.path.join(aligned_dir, f"{target_name}_aligned.tif")
        aligned_dem.save(aligned_path)

        after_stats = compute_dem_stats(reference_dem, aligned_dem)
        stats_rows.extend(
            [
                {
                    "reference": reference_name,
                    "target": target_name,
                    "stage": "before",
                    **before_stats,
                },
                {
                    "reference": reference_name,
                    "target": target_name,
                    "stage": "after",
                    **after_stats,
                },
            ]
        )

        param_records[target_name] = {
            "coreg_model": "NuthKaab",
            "matrix": coreg.to_matrix().tolist() if hasattr(coreg, "to_matrix") else None,
            "metadata": coreg.meta if hasattr(coreg, "meta") else None,
            "repr": repr(coreg),
        }

    params_json = os.path.join(output_dir, "coregistration_parameters.json")
    with open(params_json, "w", encoding="utf-8") as f:
        json.dump(param_records, f, indent=2)

    stats_csv = ""
    if write_diagnostics and stats_rows:
        stats_csv = os.path.join(output_dir, "coregistration_stats.csv")
        fieldnames = ["reference", "target", "stage", "mean", "median", "nmad", "count"]
        with open(stats_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(stats_rows)

    return {"parameters_json": params_json, "stats_csv": stats_csv, "aligned_dir": aligned_dir}


def run_xdem_coreg_workflow(config) -> dict[str, str]:
    """Checkpointed strip workflow: rasterise DEMs, then xDEM coregister."""
    input_dir = Path(getattr(config, "xdem_input_dir", "") or os.path.join(config.preprocessed_dir, config.run_name, "icp_ready"))
    output_dir = Path(getattr(config, "xdem_output_dir", "") or os.path.join(config.results_dir, config.run_name, "xdem_coreg"))
    dem_dir = output_dir / "dems"
    aligned_dir = output_dir / "aligned"
    dem_dir.mkdir(parents=True, exist_ok=True)

    las_files = find_icp_ready_files(str(input_dir), filename_token=getattr(config, "xdem_filename_token", "icp_ready"))
    if not las_files:
        raise ValueError(f"No ICP-ready LAS/LAZ files found in: {input_dir}")

    template_grid = build_template_grid(las_files, resolution=float(getattr(config, "xdem_resolution", 1.0)))

    dem_paths: list[Path] = []
    for idx, las_path in enumerate(las_files):
        strip_id = Path(las_path).stem
        dem_paths.append(dem_dir / f"dem_strip_{idx:03d}_{strip_id}.tif")

    def rasterise_all_strips() -> None:
        for las_path, dem_path in zip(las_files, dem_paths):
            if output_exists(dem_path, min_size_mb=1.0) and not bool(getattr(config, "overwrite_outputs", False)):
                print(f"[SKIP] rasterise output exists -> {dem_path}")
                continue
            rasterise_strip_to_dem(
                las_path=las_path,
                template_grid=template_grid,
                output_path=str(dem_path),
                nodata=float(getattr(config, "xdem_nodata", -9999.0)),
                use_ground_only=bool(getattr(config, "xdem_ground_only", True)),
            )

    aligned_outputs = [aligned_dir / f"{dem_path.stem}_aligned.tif" for dem_path in dem_paths[1:]]

    def run_xdem_coregistration() -> None:
        coregister_dems(
            dem_paths=[str(path) for path in dem_paths],
            output_dir=str(output_dir),
            write_diagnostics=bool(getattr(config, "xdem_write_diagnostics", True)),
        )

    steps = [
        PipelineStep(
            name="rasterise",
            func=rasterise_all_strips,
            outputs=dem_paths,
            output_dir=dem_dir,
            enabled=bool(getattr(config, "xdem_enable_rasterise", CONFIG["rasterise"])),
            dependencies=[],
        ),
        PipelineStep(
            name="coreg_xdem",
            func=run_xdem_coregistration,
            outputs=aligned_outputs,
            output_dir=output_dir,
            enabled=bool(getattr(config, "enable_xdem_coreg", CONFIG["coreg_xdem"])),
            dependencies=["rasterise"],
        ),
    ]

    overwrite = bool(getattr(config, "overwrite_outputs", CONFIG["overwrite"]))
    completed_steps: set[str] = set()
    for step in steps:
        run_step(step, completed_steps=completed_steps, overwrite=overwrite)

    params_json = output_dir / "coregistration_parameters.json"
    stats_csv = output_dir / "coregistration_stats.csv"
    return {
        "parameters_json": str(params_json),
        "stats_csv": str(stats_csv) if stats_csv.exists() else "",
        "aligned_dir": str(aligned_dir),
    }


def main() -> None:
    from config.config import Configuration

    cfg = Configuration().validate()
    results = run_xdem_coreg_workflow(cfg)
    print("xDEM coregistration outputs:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
