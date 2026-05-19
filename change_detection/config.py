from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Path-driven configuration for one DEM change-detection run."""

    dem_reference_path: Path
    dem_target_path: Path
    output_dir: Path
    stable_ground_path: Path | None = None

    terrain_mode: str = "flat"  # flat | sloped
    apply_terrain_bias: bool = False
    apply_deramp: bool = False

    outlier_clip_m: float = 10.0
    min_stable_pixels: int = 1000
    median_warn_threshold_m: float = 0.2
    change_threshold_m: float | None = None

    output_prefix: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "dem_reference_path", Path(self.dem_reference_path))
        object.__setattr__(self, "dem_target_path", Path(self.dem_target_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.stable_ground_path is not None:
            object.__setattr__(self, "stable_ground_path", Path(self.stable_ground_path))

        if self.terrain_mode not in {"flat", "sloped"}:
            raise ValueError("terrain_mode must be 'flat' or 'sloped'.")

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def output_path(self, filename: str) -> Path:
        name = f"{self.output_prefix}{filename}" if self.output_prefix else filename
        return self.output_dir / name

    def describe_pipeline(self) -> str:
        steps = ["VerticalShift"]
        if self.terrain_mode == "flat":
            steps += ["LeastZDifference", "DhMinimize"]
        else:
            steps += ["NuthKaab"]
            if self.apply_terrain_bias:
                steps += ["TerrainBias"]
        if self.apply_deramp:
            steps += ["Deramp"]
        return " → ".join(steps)


# ---------------------------------------------------------------------------
# User-editable settings
# ---------------------------------------------------------------------------

BASE_DIR = Path(".")
DEM_REFERENCE_PATH = BASE_DIR / "data/reference_dem.tif"
DEM_TARGET_PATH = BASE_DIR / "data/target_dem.tif"
STABLE_GROUND_PATH: Path | None = BASE_DIR / "data/stable_mask.gpkg"
OUTPUT_DIR = BASE_DIR / "results/change_detection"

TERRAIN_MODE = "flat"  # flat | sloped
APPLY_TERRAIN_BIAS = False
APPLY_DERAMP = False

OUTLIER_CLIP_M = 10.0
MIN_STABLE_PIXELS = 1000
MEDIAN_WARN_THRESHOLD_M = 0.2
CHANGE_THRESHOLD_M: float | None = None

OUTPUT_PREFIX = ""


def make_config() -> Config:
    """Create Config from user-editable constants above."""
    return Config(
        dem_reference_path=DEM_REFERENCE_PATH,
        dem_target_path=DEM_TARGET_PATH,
        output_dir=OUTPUT_DIR,
        stable_ground_path=STABLE_GROUND_PATH,
        terrain_mode=TERRAIN_MODE,
        apply_terrain_bias=APPLY_TERRAIN_BIAS,
        apply_deramp=APPLY_DERAMP,
        outlier_clip_m=OUTLIER_CLIP_M,
        min_stable_pixels=MIN_STABLE_PIXELS,
        median_warn_threshold_m=MEDIAN_WARN_THRESHOLD_M,
        change_threshold_m=CHANGE_THRESHOLD_M,
        output_prefix=OUTPUT_PREFIX,
    )
