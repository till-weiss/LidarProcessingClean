from __future__ import annotations

import time

import numpy as np
import xdem
import geoutils as gu

from config import Config, make_config, validate_config_paths
from coregister import run_coregistration
from change import compute_change, add_volume_budget
from report import save_report


def run(cfg: Config) -> None:
    t0 = time.time()
    print("=" * 60)
    print("  Change detection pipeline")
    print(f"  Reference : {cfg.dem_reference_path}")
    print(f"  Target    : {cfg.dem_target_path}")
    print(f"  Output    : {cfg.output_dir}")
    print("=" * 60)

    validate_config_paths(cfg)

    print("\n[1/4] Loading inputs")
    ref_dem = xdem.DEM(cfg.dem_reference_path)
    tba_dem = xdem.DEM(cfg.dem_target_path).reproject(ref_dem)

    stable_mask = None
    if cfg.stable_ground_path is not None:
        stable_vec = gu.Vector(cfg.stable_ground_path)
        stable_mask = np.array(stable_vec.create_mask(ref_dem).data).astype(bool)
        print(f"  Stable mask: {stable_mask.sum():,} pixels")
    else:
        print("  No stable mask — using all valid pixels")

    print("\n[2/4] Co-registration")
    coreg_result = run_coregistration(cfg, ref_dem=ref_dem, tba_dem=tba_dem, stable_mask=stable_mask)

    print("\n[3/4] Change detection")
    change = compute_change(coreg_result, cfg, ref_dem=ref_dem, stable_mask=stable_mask)
    add_volume_budget(change, pixel_size_m=ref_dem.res[0])

    print("\n[4/4] Saving outputs")
    save_report(coreg_result, change, cfg)

    print(f"\nDone in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    run(make_config())
