from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import xdem
import geoutils as gu

from config import Config
from coregister import run_coregistration
from change import compute_change, add_volume_budget
from report import save_report


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(cfg: Config) -> None:
    """
    Full pipeline:
        1. Load DEMs and stable mask
        2. Co-register DEMs
        3. Compute dDEM and statistics
        4. Compute volume budget
        5. Save outputs
    """

    t0 = time.time()

    print("=" * 60)
    print("  Change detection pipeline")
    print(f"  Reference : {cfg.dem_reference_path}")
    print(f"  Target    : {cfg.dem_target_path}")
    print(f"  Output    : {cfg.output_dir}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1 — Load inputs
    # ------------------------------------------------------------------

    print("\n[1/4] Loading inputs")

    ref_dem = xdem.DEM(cfg.dem_reference_path)

    tba_dem = xdem.DEM(cfg.dem_target_path)
    tba_dem = tba_dem.reproject(ref_dem)

    stable_mask = None

    if cfg.stable_ground_path is not None:

        stable_vec = gu.Vector(cfg.stable_ground_path)

        stable_mask = np.array(
            stable_vec.create_mask(ref_dem).data
        ).astype(bool)

        print(f"  Stable mask: {stable_mask.sum():,} pixels")

    else:
        print("  No stable mask — using all valid pixels")

    # ------------------------------------------------------------------
    # Step 2 — Co-registration
    # ------------------------------------------------------------------

    print("\n[2/4] Co-registration")

    coreg_results = run_coregistration(
        cfg,
        ref_dem=ref_dem,
        tba_dem=tba_dem,
        stable_mask=stable_mask,
    )

    # ------------------------------------------------------------------
    # Step 3 — Change detection
    # ------------------------------------------------------------------

    print("\n[3/4] Change detection")

    change = compute_change(
        coreg_results,
        cfg,
        ref_dem=ref_dem,
        stable_mask=stable_mask,
    )

    # Volume calculation
    pixel_size = ref_dem.res[0]

    add_volume_budget(
        change,
        pixel_size_m=pixel_size,
    )

    # ------------------------------------------------------------------
    # Step 4 — Save outputs
    # ------------------------------------------------------------------

    print("\n[4/4] Saving outputs")

    save_report(coreg_results, change, cfg)

    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.1f} s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Run DEM change detection pipeline"
    )

    parser.add_argument(
        "--reference",
        required=True,
        help="Reference DEM path",
    )

    parser.add_argument(
        "--target",
        required=True,
        help="Target DEM path",
    )

    parser.add_argument(
        "--stable",
        required=False,
        default=None,
        help="Stable terrain vector path",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    args = _parse_args()

    cfg = Config(
        dem_reference_path=args.reference,
        dem_target_path=args.target,
        stable_ground_path=args.stable,
        output_dir=Path(args.output)
        terrain_mode=args.terrain_mode
    )

    run(cfg)