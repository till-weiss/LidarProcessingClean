"""
change.py
---------
Computes the change DEM (dDEM = target - reference) from the best-aligned
DEM pair and derives summary statistics.

Designed to be extended: terrain-unit segregation, volumetric budgets,
and rugosity change can all be added here as new functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import xdem
import geoutils as gu

from config import Config
from coregister import CoregResult, best_result, sanitize_dem_nodata


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ChangeResult:
    """
    Holds the dDEM array and all change statistics.

    All statistics are computed over the full AOI (excluding nodata).
    Stable-ground stats are also reported separately so you can see
    the residual uncertainty alongside the real signal.
    """

    # The differenced DEM object (can be saved directly as a GeoTIFF)
    ddem: xdem.DEM

    # Change threshold (m) used to define 'detectable' change
    threshold_m: float

    # ------------------------------------------------------------------
    # Full-AOI statistics (all valid pixels)
    # ------------------------------------------------------------------
    aoi_mean: float = np.nan
    aoi_median: float = np.nan
    aoi_nmad: float = np.nan
    aoi_std: float = np.nan
    aoi_q683: float = np.nan    # ~1-sigma equivalent
    aoi_q95: float = np.nan
    aoi_n_pixels: int = 0

    # Fraction of AOI pixels with |dDEM| > threshold (detectable change)
    aoi_change_fraction: float = np.nan

    # ------------------------------------------------------------------
    # Stable-ground residuals (diagnostic — should be near zero)
    # ------------------------------------------------------------------
    stable_median: float = np.nan
    stable_nmad: float = np.nan
    stable_n_pixels: int = 0

    # ------------------------------------------------------------------
    # Signed change areas (pixels above threshold only)
    # ------------------------------------------------------------------
    subsidence_mean_m: float = np.nan      # mean of pixels < -threshold
    subsidence_n_pixels: int = 0
    heave_mean_m: float = np.nan           # mean of pixels > +threshold
    heave_n_pixels: int = 0

    # Placeholder for volumetric budget — populated by add_volume_budget()
    volume_loss_m3: Optional[float] = None
    volume_gain_m3: Optional[float] = None
    pixel_area_m2: Optional[float] = None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_change(
    coreg_results: CoregResult,
    cfg: Config,
    ref_dem: Optional[xdem.DEM] = None,
    stable_mask: Optional[np.ndarray] = None,
) -> ChangeResult:
    """
    Subtract the reference DEM from the best-aligned target DEM.

    Parameters
    ----------
    coreg_results : dict
        Output of coregister.run_coregistration().
    cfg : Config
        Configuration object — used for threshold and output resolution.
    ref_dem : xdem.DEM, optional
        Pre-loaded reference DEM. If None it is reloaded from cfg.
    stable_mask : np.ndarray, optional
        Boolean mask of stable ground. Used to report residual uncertainty.

    Returns
    -------
    ChangeResult
    """

    best: CoregResult = best_result(coreg_results)

    if ref_dem is None:
        print("Loading reference DEM for change computation...")
        ref_dem = sanitize_dem_nodata(xdem.DEM(cfg.dem_reference_path))

    # Compute dDEM: positive = surface raised, negative = surface lowered
    print(f"Computing dDEM using pipeline: {best.pipeline_description}")
    aligned = sanitize_dem_nodata(best.aligned_dem)
    ref_dem = sanitize_dem_nodata(ref_dem)
    ddem = aligned - ref_dem

    # ------------------------------------------------------------------
    # Determine change threshold
    # ------------------------------------------------------------------

    if cfg.change_threshold_m is not None:
        threshold = cfg.change_threshold_m
    else:
        # Default: 2 * NMAD of the best co-registration residuals.
        # This propagates the alignment uncertainty into the threshold
        # (analogous to Höhle & Höhle 2009 approach).
        threshold = 2.0 * best.nmad if np.isfinite(best.nmad) else np.nan
        print(f"  Auto-threshold: 2 × NMAD = 2 × {best.nmad:.3f} = {threshold:.3f} m")

    # ------------------------------------------------------------------
    # Extract valid pixels
    # ------------------------------------------------------------------

    ddem_arr = np.array(ddem.data).astype(np.float32)
    valid = np.isfinite(ddem_arr) & np.isfinite(np.array(ref_dem.data)) & np.isfinite(np.array(aligned.data))
    ddem_arr[~valid] = np.nan
    vals = ddem_arr[valid]

    if len(vals) == 0:
        raise RuntimeError("dDEM contains no valid pixels.")

    # ------------------------------------------------------------------
    # Full-AOI statistics
    # ------------------------------------------------------------------

    aoi_mean   = float(np.nanmean(vals))
    aoi_median = float(np.nanmedian(vals))
    aoi_nmad   = float(gu.stats.nmad(vals))
    aoi_std    = float(np.nanstd(vals))
    aoi_q683   = float(np.nanpercentile(np.abs(vals), 68.3))
    aoi_q95    = float(np.nanpercentile(np.abs(vals), 95.0))

    change_fraction = float(np.mean(np.abs(vals) > threshold)) if np.isfinite(threshold) else np.nan

    print(
        f"  AOI  median={aoi_median:+.3f} m  "
        f"NMAD={aoi_nmad:.3f} m  "
        f"change_fraction={change_fraction:.1%}"
    )

    # ------------------------------------------------------------------
    # Stable-ground residuals (sanity check)
    # ------------------------------------------------------------------

    stable_median = np.nan
    stable_nmad   = np.nan
    stable_n      = 0

    if stable_mask is not None:
        stable_vals = ddem_arr[stable_mask & valid]
        stable_vals = stable_vals[np.abs(stable_vals) < cfg.outlier_clip_m]
        if len(stable_vals) > 0:
            stable_median = float(np.nanmedian(stable_vals))
            stable_nmad   = float(gu.stats.nmad(stable_vals))
            stable_n      = int(len(stable_vals))
            print(
                f"  Stable ground residual  "
                f"median={stable_median:+.3f} m  "
                f"NMAD={stable_nmad:.3f} m  "
                f"n={stable_n}"
            )

    # ------------------------------------------------------------------
    # Signed change areas
    # ------------------------------------------------------------------

    if np.isfinite(threshold):
        subsidence_mask = valid & (ddem_arr < -threshold)
        heave_mask      = valid & (ddem_arr >  threshold)
    else:
        subsidence_mask = np.zeros_like(valid, dtype=bool)
        heave_mask = np.zeros_like(valid, dtype=bool)

    subsidence_mean   = float(np.nanmean(ddem_arr[subsidence_mask])) if subsidence_mask.any() else np.nan
    heave_mean        = float(np.nanmean(ddem_arr[heave_mask]))      if heave_mask.any()      else np.nan
    subsidence_pixels = int(subsidence_mask.sum())
    heave_pixels      = int(heave_mask.sum())

    print(
        f"  Subsidence pixels: {subsidence_pixels}  "
        f"mean={subsidence_mean:+.3f} m"
        if not np.isnan(subsidence_mean) else
        f"  Subsidence pixels: {subsidence_pixels}"
    )
    print(
        f"  Heave pixels:      {heave_pixels}  "
        f"mean={heave_mean:+.3f} m"
        if not np.isnan(heave_mean) else
        f"  Heave pixels:      {heave_pixels}"
    )

    return ChangeResult(
        ddem=ddem,
        threshold_m=threshold,
        aoi_mean=aoi_mean,
        aoi_median=aoi_median,
        aoi_nmad=aoi_nmad,
        aoi_std=aoi_std,
        aoi_q683=aoi_q683,
        aoi_q95=aoi_q95,
        aoi_n_pixels=int(len(vals)),
        aoi_change_fraction=change_fraction,
        stable_median=stable_median,
        stable_nmad=stable_nmad,
        stable_n_pixels=stable_n,
        subsidence_mean_m=subsidence_mean,
        subsidence_n_pixels=subsidence_pixels,
        heave_mean_m=heave_mean,
        heave_n_pixels=heave_pixels,
    )


# ---------------------------------------------------------------------------
# Optional extension: volumetric budget
# ---------------------------------------------------------------------------

def add_volume_budget(change: ChangeResult, pixel_size_m: float) -> ChangeResult:
    """
    Populate volume_loss_m3 and volume_gain_m3 on an existing ChangeResult.

    Call after compute_change() when you need volumetric estimates.
    pixel_size_m is the DEM grid spacing in metres (e.g. 1.0 for 1 m resolution).

    Volume is computed only over pixels that exceed the change threshold,
    consistent with the detectable-change definition used in the rest of the module.
    """

    pixel_area = pixel_size_m ** 2
    change.pixel_area_m2 = pixel_area

    ddem_arr = np.array(change.ddem.data).astype(np.float32)
    valid = np.isfinite(ddem_arr)

    if not np.isfinite(change.threshold_m):
        change.volume_loss_m3 = np.nan
        change.volume_gain_m3 = np.nan
        print("  Volume loss: nan m³  Volume gain: nan m³ (invalid threshold)")
        return change

    sub_vals  = ddem_arr[valid & (ddem_arr < -change.threshold_m)]
    heave_vals = ddem_arr[valid & (ddem_arr >  change.threshold_m)]

    change.volume_loss_m3 = float(np.sum(np.abs(sub_vals)) * pixel_area)
    change.volume_gain_m3 = float(np.sum(heave_vals) * pixel_area)

    print(
        f"  Volume loss:  {change.volume_loss_m3:,.1f} m³  "
        f"Volume gain: {change.volume_gain_m3:,.1f} m³"
    )

    return change