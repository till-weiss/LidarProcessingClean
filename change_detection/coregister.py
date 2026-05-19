"""
coregister.py
-------------
Builds and runs a fixed sequential co-registration pipeline whose steps
depend on terrain_mode in the Config object.

Flat mode:   VerticalShift → LeastZDifference → DhMinimize [→ Deramp]
Sloped mode: VerticalShift → NuthKaab [→ TerrainBias] [→ Deramp]

The pipeline is not a benchmark — the same sequence runs on every AOI of
the same terrain type. Post-correction evaluation (NMAD, median, aspect
diagnostic) is reported but does not change which corrections were applied.

Returns a CoregResult dataclass with the aligned DEM, all metrics, and
the aspect-vs-dDEM arrays needed for diagnostic plotting in report.py.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import xdem
import geoutils as gu

from config import Config


def _make_step(*names):
    """Return first available xdem.coreg step class instantiated."""
    for name in names:
        cls = getattr(xdem.coreg, name, None)
        if cls is not None:
            return cls()
    raise AttributeError(f"None of the coreg steps are available: {names}")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class StepDiagnostic:
    step_name: str
    pipeline_description: str
    corrected_dem: xdem.DEM
    residuals: np.ndarray
    median: float
    nmad: float
    std: float
    mae: float
    rmse: float
    n_stable: int
    aspect_r2: float
    aspect_bin_centres: np.ndarray = field(default_factory=lambda: np.array([]))
    aspect_bin_means: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class CoregResult:
    """Aligned DEM plus all evaluation outputs for one AOI."""

    # The pipeline description string, e.g. "VerticalShift → NuthKaab"
    pipeline_description: str

    # The fully corrected DEM
    aligned_dem: xdem.DEM

    # ------------------------------------------------------------------
    # Stable-ground residual statistics (post-correction)
    # ------------------------------------------------------------------
    median: float = np.nan
    nmad: float = np.nan
    std: float = np.nan
    mae: float = np.nan
    rmse: float = np.nan
    n_stable: int = 0

    # Raw clipped residuals — kept for histogram plotting
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))

    # ------------------------------------------------------------------
    # Aspect-vs-dDEM diagnostic arrays
    # aspect_bin_centres : degrees (0–360, 18 bins of 20°)
    # aspect_bin_means   : mean dDEM per bin (m)
    # aspect_bin_stds    : std per bin (m)
    # sinusoid_r2        : R² of fitted sinusoid (NaN if < 5 valid bins)
    # ------------------------------------------------------------------
    aspect_bin_centres: np.ndarray = field(default_factory=lambda: np.array([]))
    aspect_bin_means: np.ndarray = field(default_factory=lambda: np.array([]))
    aspect_bin_stds: np.ndarray = field(default_factory=lambda: np.array([]))
    sinusoid_r2: float = np.nan

    # Quality flag — True if the result should be treated with caution
    flagged: bool = False
    flag_reason: str = ""

    # True if the pipeline raised an exception
    failed: bool = False
    failure_reason: str = ""

    step_diagnostics: list[StepDiagnostic] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def _build_pipeline(cfg: Config):
    """
    Assemble an xdem coregistration pipeline from the Config settings.

    Returns an xdem coreg object (single step or pipeline via +).
    """

    steps = [_make_step("VerticalShift")]

    if cfg.terrain_mode == "flat":
        # Prefer LeastZDifference when available; avoid ICP fallback because
        # some xdem installs require optional pytransform3d for ICP.
        try:
            steps.append(_make_step("LeastZDifference"))
        except AttributeError:
            steps.append(_make_step("NuthKaab"))
        steps.append(_make_step("DhMinimize"))

    elif cfg.terrain_mode == "sloped":
        steps.append(_make_step("NuthKaab"))
        if cfg.apply_terrain_bias:
            steps.append(_make_step("TerrainBias"))

    else:
        raise ValueError(
            f"Unknown terrain_mode '{cfg.terrain_mode}'. "
            "Choose 'flat' or 'sloped'."
        )

    if cfg.apply_deramp:
        # First-order plane correction for scene-wide tilt.
        # Valid in both modes; only enable after inspecting residual map.
        steps.append(_make_step("Deramp"))

    # Chain steps into a single pipeline object
    pipeline = steps[0]
    for s in steps[1:]:
        pipeline = pipeline + s

    return steps, pipeline


# ---------------------------------------------------------------------------
# Aspect-vs-dDEM diagnostic
# ---------------------------------------------------------------------------

def _aspect_ddem_diagnostic(
    ddem_arr: np.ndarray,
    ref_dem: xdem.DEM,
    valid_mask: np.ndarray,
    n_bins: int = 18,
) -> dict:
    """
    Bin stable-ground dDEM values by terrain aspect and fit a sinusoid.

    The sinusoidal fit follows Nuth & Kääb (2011):
        dh / tan(slope) = A * cos(aspect - phi) + C
    but here we report the raw dh per aspect bin rather than the
    normalised form, because tan(slope) ≈ 0 in flat terrain makes
    normalisation numerically unstable.

    Returns a dict with bin_centres, bin_means, bin_stds, sinusoid_r2.
    The R² value is the primary interpretive metric:
      - R² > 0.7 and amplitude > 0.1 m  → likely residual horizontal offset
      - R² < 0.3 or amplitude < 0.05 m  → no meaningful pattern detected
    In flat terrain, low R² is expected and is itself a valid result.
    """

    from scipy.optimize import curve_fit
    from scipy.stats import pearsonr

    # Compute aspect from the reference DEM
    # xdem.terrain.aspect returns degrees 0–360
    try:
        aspect = xdem.terrain.aspect(ref_dem)
        aspect_arr = np.array(aspect.data).astype(np.float32)
    except Exception:
        return {
            "bin_centres": np.array([]),
            "bin_means": np.array([]),
            "bin_stds": np.array([]),
            "sinusoid_r2": np.nan,
        }

    aspect_valid = (
        valid_mask
        & np.isfinite(ddem_arr)
        & np.isfinite(aspect_arr)
    )

    if aspect_valid.sum() < 50:
        return {
            "bin_centres": np.array([]),
            "bin_means": np.array([]),
            "bin_stds": np.array([]),
            "sinusoid_r2": np.nan,
        }

    dh   = ddem_arr[aspect_valid]
    asp  = aspect_arr[aspect_valid]

    # Bin
    bin_edges   = np.linspace(0, 360, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_means   = np.full(n_bins, np.nan)
    bin_stds    = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = (asp >= bin_edges[i]) & (asp < bin_edges[i + 1])
        vals = dh[mask]
        if len(vals) >= 5:
            bin_means[i] = np.nanmedian(vals)
            bin_stds[i]  = np.nanstd(vals)

    valid_bins = np.isfinite(bin_means)

    # Fit sinusoid if enough bins are populated
    sinusoid_r2 = np.nan
    if valid_bins.sum() >= 5:
        x = np.deg2rad(bin_centres[valid_bins])
        y = bin_means[valid_bins]

        def sinusoid(x, A, phi, C):
            return A * np.cos(x - phi) + C

        try:
            popt, _ = curve_fit(
                sinusoid, x, y,
                p0=[np.nanstd(y), 0.0, np.nanmean(y)],
                maxfev=2000,
            )
            y_pred = sinusoid(x, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            sinusoid_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        except Exception:
            sinusoid_r2 = np.nan

    return {
        "bin_centres": bin_centres,
        "bin_means":   bin_means,
        "bin_stds":    bin_stds,
        "sinusoid_r2": sinusoid_r2,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(residuals_raw: np.ndarray, clip_m: float) -> dict:
    """
    Robust and classical accuracy metrics from a 1-D residual array.
    NMAD is the primary quality indicator (Höhle & Höhle, 2009).
    """

    res = residuals_raw[np.isfinite(residuals_raw)]
    res = res[np.abs(res) < clip_m]

    if len(res) == 0:
        return None

    return {
        "median": float(np.nanmedian(res)),
        "nmad":   float(gu.stats.nmad(res)),
        "std":    float(np.nanstd(res)),
        "mae":    float(np.nanmean(np.abs(res))),
        "rmse":   float(np.sqrt(np.nanmean(res ** 2))),
        "n":      int(len(res)),
        "residuals": res,
    }




def _describe_pipeline_from_steps(steps: list) -> str:
    return " → ".join(type(s).__name__ for s in steps)


def sanitize_dem_nodata(dem: xdem.DEM) -> xdem.DEM:
    """Convert DEM nodata and common sentinel values to NaN in-place."""
    arr = np.array(dem.data, dtype=np.float32)
    nodata = getattr(dem, "nodata", None)
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[np.isin(arr, (-9999.0, -99999.0, -32768.0))] = np.nan
    dem.data = arr
    try:
        dem.set_nodata(np.nan)
    except Exception:
        pass
    return dem
# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_coregistration(
    cfg: Config,
    ref_dem: Optional[xdem.DEM] = None,
    tba_dem: Optional[xdem.DEM] = None,
    stable_mask: Optional[np.ndarray] = None,
) -> CoregResult:
    """
    Build and run the co-registration pipeline defined by cfg.

    Parameters
    ----------
    cfg : Config
    ref_dem, tba_dem : optional pre-loaded DEMs (avoids reloading in run.py)
    stable_mask : optional boolean array; if None and no path given, all
                  valid pixels are used (with a warning)

    Returns
    -------
    CoregResult
    """

    pipeline_desc = cfg.describe_pipeline()
    print(f"  Pipeline (requested): {pipeline_desc}")

    # ------------------------------------------------------------------
    # Load if not supplied
    # ------------------------------------------------------------------

    if ref_dem is None:
        ref_dem = sanitize_dem_nodata(xdem.DEM(cfg.dem_reference_path))
    if tba_dem is None:
        tba_dem = sanitize_dem_nodata(xdem.DEM(cfg.dem_target_path).reproject(ref_dem))

    ref_dem = sanitize_dem_nodata(ref_dem)
    tba_dem = sanitize_dem_nodata(tba_dem)

    valid_overlap = np.isfinite(np.array(ref_dem.data)) & np.isfinite(np.array(tba_dem.data))

    if stable_mask is None:
        if cfg.stable_ground_path is not None:
            stable_vec  = gu.Vector(cfg.stable_ground_path)
            stable_mask = np.array(
                stable_vec.create_mask(ref_dem).data
            ).astype(bool) & valid_overlap
        else:
            warnings.warn(
                "No stable ground mask — using all valid pixels. "
                "Results may reflect real terrain change.",
                UserWarning,
            )
            stable_mask = valid_overlap.copy()

    # ------------------------------------------------------------------
    # Build and run pipeline stepwise
    # ------------------------------------------------------------------

    actual_pipeline_desc = pipeline_desc
    step_diagnostics: list[StepDiagnostic] = []
    aligned = tba_dem

    def _evaluate_step(step_name: str, pipeline_name: str, dem_now: xdem.DEM):
        residual_dem = dem_now - ref_dem
        residuals_raw = np.array(residual_dem.data).astype(np.float32)
        residuals_raw[~np.isfinite(np.array(ref_dem.data)) | ~np.isfinite(np.array(dem_now.data))] = np.nan
        valid = stable_mask & np.isfinite(residuals_raw)
        m = _compute_metrics(residuals_raw[valid], cfg.outlier_clip_m) if int(valid.sum()) > 0 else None
        if m is None:
            median = nmad = std = mae = rmse = np.nan
            residuals = np.array([])
            n_stable_eval = 0
        else:
            median, nmad, std, mae, rmse = m['median'], m['nmad'], m['std'], m['mae'], m['rmse']
            residuals = m['residuals']
            n_stable_eval = m['n']
        aspect_diag = _aspect_ddem_diagnostic(residuals_raw, ref_dem, valid)
        step_diagnostics.append(StepDiagnostic(step_name=step_name, pipeline_description=pipeline_name, corrected_dem=dem_now, residuals=residuals, median=median, nmad=nmad, std=std, mae=mae, rmse=rmse, n_stable=n_stable_eval, aspect_r2=aspect_diag['sinusoid_r2'], aspect_bin_centres=aspect_diag['bin_centres'], aspect_bin_means=aspect_diag['bin_means']))
        return residuals_raw, valid, aspect_diag, median, nmad, std, mae, rmse, n_stable_eval, residuals

    try:
        steps, _ = _build_pipeline(cfg)
        actual_pipeline_desc = _describe_pipeline_from_steps(steps)
        print(f"  Pipeline (actual): {actual_pipeline_desc}")

        residuals_raw, valid, aspect_diag, median, nmad, std, mae, rmse, n_stable, residuals = _evaluate_step("Raw", "Raw DEM", aligned)
        names=[]
        for step in steps:
            step.fit(reference_elev=ref_dem, to_be_aligned_elev=aligned, inlier_mask=stable_mask)
            aligned = step.apply(aligned)
            names.append(type(step).__name__)
            pipeline_name = " → ".join(names)
            residuals_raw, valid, aspect_diag, median, nmad, std, mae, rmse, n_stable, residuals = _evaluate_step(
                f"{len(names)}",
                pipeline_name,
                aligned,
            )

    except Exception as exc:
        print(f"  FAILED: {exc}")
        return CoregResult(pipeline_description=actual_pipeline_desc, aligned_dem=tba_dem, failed=True, failure_reason=str(exc), step_diagnostics=step_diagnostics)

    n_stable = int(valid.sum())

    flagged      = False
    flag_reasons = []

    if n_stable < cfg.min_stable_pixels:
        flagged = True
        flag_reasons.append(
            f"Only {n_stable} stable pixels (min {cfg.min_stable_pixels})"
        )

    m = _compute_metrics(residuals_raw[valid], cfg.outlier_clip_m) if n_stable > 0 else None

    if m is None:
        flagged = True
        flag_reasons.append("No valid residuals after clipping")
        median = nmad = std = mae = rmse = np.nan
        residuals = np.array([])
        n_stable  = 0
    else:
        median, nmad, std, mae, rmse = (
            m["median"], m["nmad"], m["std"], m["mae"], m["rmse"]
        )
        residuals = m["residuals"]
        n_stable  = m["n"]

        if abs(median) > cfg.median_warn_threshold_m:
            flagged = True
            flag_reasons.append(
                f"Residual median {median:+.3f} m exceeds "
                f"warning threshold ±{cfg.median_warn_threshold_m} m"
            )

    print(
        f"  median={median:+.4f} m  NMAD={nmad:.4f} m  "
        f"STD={std:.4f} m  n={n_stable}"
        if not np.isnan(median) else
        f"  Evaluation: no valid residuals"
    )

    if flagged:
        print(f"  FLAG: {'; '.join(flag_reasons)}")

    # ------------------------------------------------------------------
    # Aspect-vs-dDEM diagnostic
    # ------------------------------------------------------------------

    print("  Computing aspect-dDEM diagnostic...")
    r2 = aspect_diag["sinusoid_r2"]
    if not np.isnan(r2):
        print(f"  Aspect sinusoid R² = {r2:.3f}", end="")
        if cfg.terrain_mode == "flat":
            print("  (low R² expected in flat terrain — diagnostic only)")
        elif r2 > 0.7:
            print("  → strong pattern; consider NuthKaab if not already applied")
        else:
            print()

    return CoregResult(
        pipeline_description=actual_pipeline_desc,
        aligned_dem=aligned,
        median=median,
        nmad=nmad,
        std=std,
        mae=mae,
        rmse=rmse,
        n_stable=n_stable,
        residuals=residuals,
        aspect_bin_centres=aspect_diag["bin_centres"],
        aspect_bin_means=aspect_diag["bin_means"],
        aspect_bin_stds=aspect_diag["bin_stds"],
        sinusoid_r2=aspect_diag["sinusoid_r2"],
        flagged=flagged,
        flag_reason="; ".join(flag_reasons),
        step_diagnostics=step_diagnostics,
    )


def best_result(coreg_results):
    """Backwards-compatible selector for best result."""
    if isinstance(coreg_results, CoregResult):
        return coreg_results
    if isinstance(coreg_results, dict):
        if "_best" in coreg_results:
            return coreg_results[coreg_results["_best"]]
        for value in coreg_results.values():
            if isinstance(value, CoregResult):
                return value
    raise TypeError("Unsupported coreg_results container.")
