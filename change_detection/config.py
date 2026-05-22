from dataclasses import dataclass
from pathlib import Path

# -------------------------------------------------
# Central settings for one change-detection run
# -------------------------------------------------

AOI_NAME = "Inuvik_Airport_rerun"
DEM_TYPE = "DSM"   # "DTM" or "DSM"
REF_YEAR = 2023
TARGET_YEAR = 2025
COREG_METHOD = "vertical_shift" # or "nuth_kaab"

DEM_REFERENCE_PATH = "/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/results/Inuvik_Airport_2023_rerun/DSM/Inuvik_Airport_DSM_2m.tif"
DEM_TARGET_PATH = "/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/results/Inuvik_Airport_2025_rerun/DSM/Inuvik_Airport_DSM_2m.tif"
STABLE_GROUND_PATH = "/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/AOI/StableReferences/Inuvik_Airport_Stable_Ground.gpkg"
OUTPUT_DIR = "/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/results/ChangeDetection"


@dataclass
class ChangeDetectionConfig:
    aoi_name: str = AOI_NAME
    dem_type: str = DEM_TYPE
    ref_year: int = REF_YEAR
    target_year: int = TARGET_YEAR
    coreg_method: str = COREG_METHOD
    dem_reference_path: str = DEM_REFERENCE_PATH
    dem_target_path: str = DEM_TARGET_PATH
    stable_ground_path: str | None = STABLE_GROUND_PATH
    output_root: str = OUTPUT_DIR

    outlier_clip_m: float = 100.0
    change_threshold_m: float | None = None

    def __post_init__(self):
        self.dem_type = self.dem_type.upper()
        if self.dem_type not in {"DTM", "DSM"}:
            raise ValueError("DEM_TYPE must be 'DTM' or 'DSM'")

        self.output_dir = Path(self.output_root) / self.aoi_name / self.dem_type / f"{self.ref_year}_{self.target_year}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        stem = f"{self.aoi_name}_{self.dem_type}_{self.ref_year}_{self.target_year}"
        self.file_stem = stem

        self.corrected_target_tif = self.output_dir / f"{stem}_target_coreg.tif"
        self.ddem_tif = self.output_dir / f"{stem}_ddem.tif"
        self.summary_csv = self.output_dir / f"{stem}_summary.csv"
        self.agreement_png = self.output_dir / f"{stem}_agreement.png"
        self.distribution_png = self.output_dir / f"{stem}_distribution.png"


def get_config() -> ChangeDetectionConfig:
    return ChangeDetectionConfig()
