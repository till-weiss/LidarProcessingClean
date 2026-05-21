"""
change_detection
----------------
Airborne LiDAR DEM change-detection pipeline.

Dependencies (not in pixi.toml — install separately):
    conda install -c conda-forge xdem geoutils

Typical usage:
    from change_detection.config import Config
    from change_detection.run import run
    cfg = Config(
        aoi_name="Peel1", dem_type="DSM",
        ref_year=2023, target_year=2025,
        dem_reference_path="...", dem_target_path="...",
        output_dir="results/change_detection",
    )
    run(cfg)
"""

from .config import Config, make_config, validate_config_paths
from .coregister import run_coregistration, CoregResult, sanitize_dem_nodata
from .change import compute_change, add_volume_budget, ChangeResult
from .report import save_report
from .run import run

__all__ = [
    "Config",
    "make_config",
    "validate_config_paths",
    "run_coregistration",
    "CoregResult",
    "sanitize_dem_nodata",
    "compute_change",
    "add_volume_budget",
    "ChangeResult",
    "save_report",
    "run",
]
