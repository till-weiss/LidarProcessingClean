"""
report.py
---------
Produces all output files: plots, summary CSV, and aligned/dDEM GeoTIFFs.

This module only writes — it does not compute metrics.
All inputs come from the structured results produced by coregister.py
and change.py, so you can re-run plots without re-running ICP.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

from config import Config
from coregister import CoregResult
from change import ChangeResult


def _as_result_map(coreg_results):
    if isinstance(coreg_results, CoregResult):
        return {"coreg": coreg_results, "_best": "coreg"}
    return coreg_results


# ---------------------------------------------------------------------------
# Helper: shared figure style
# ---------------------------------------------------------------------------

def _apply_style(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------------
# 1. Co-registration comparison histogram
# ---------------------------------------------------------------------------

def plot_coreg_histograms(
    coreg_results: dict[str, CoregResult],
    cfg: Config,
    filename: str = "coreg_comparison.png",
) -> Path:
    """
    Overlapping residual histograms for all successful methods,
    annotated with NMAD. Best method drawn on top with full opacity.
    """

    coreg_results = _as_result_map(coreg_results)
    successful = {
        k: v for k, v in coreg_results.items()
        if k != "_best" and not v.failed and len(v.residuals) > 0
    }

    if not successful:
        raise RuntimeError("No successful co-registration results to plot.")

    best_name = coreg_results["_best"]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Draw non-best methods first (lower alpha)
    for name, result in successful.items():
        if name == best_name:
            continue
        ax.hist(
            result.residuals,
            bins=200,
            alpha=0.25,
            density=True,
            label=f"{name}  NMAD={result.nmad:.3f} m",
        )

    # Draw best method on top
    best = successful[best_name]
    ax.hist(
        best.residuals,
        bins=200,
        alpha=0.75,
        density=True,
        color="steelblue",
        label=f"{best_name}  NMAD={best.nmad:.3f} m  ★ best",
        zorder=5,
    )

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlim(-1.5, 1.5)

    _apply_style(
        ax,
        title="Co-registration residuals on stable ground",
        xlabel="Residual (aligned − reference)  [m]",
        ylabel="Density",
    )

    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()

    out = cfg.output_path(filename)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# 2. Residual map of best method
# ---------------------------------------------------------------------------

def plot_residual_map(
    coreg_results: dict[str, CoregResult],
    cfg: Config,
    filename: str = "residual_map_best.png",
    vrange: float = 0.5,
) -> Path:
    """
    Spatial map of stable-ground residuals for the best co-registration method.
    """

    import xdem

    best_name = coreg_results["_best"]
    best: CoregResult = coreg_results[best_name]

    ref_dem = xdem.DEM(cfg.dem_reference_path)
    residual_dem = best.aligned_dem - ref_dem
    residual_arr = np.array(residual_dem.data).astype(np.float32)

    fig, ax = plt.subplots(figsize=(10, 7))

    im = ax.imshow(
        residual_arr,
        cmap="RdBu_r",
        vmin=-vrange,
        vmax=vrange,
        interpolation="nearest",
    )

    plt.colorbar(im, ax=ax, label="Residual [m]", shrink=0.7)

    _apply_style(
        ax,
        title=f"Residual map — {best_name}  (NMAD = {best.nmad:.3f} m)",
    )

    fig.tight_layout()
    out = cfg.output_path(filename)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# 3. dDEM map
# ---------------------------------------------------------------------------

def plot_ddem_map(
    change: ChangeResult,
    cfg: Config,
    filename: str = "ddem_map.png",
    vrange: Optional[float] = None,
) -> Path:
    """
    Spatial map of the dDEM. Diverging colormap centred on zero.
    Pixels below the change threshold are shown in a muted neutral tone
    to visually separate noise from real change.
    """

    ddem_arr = np.array(change.ddem.data).astype(np.float32)

    # Auto-range: 95th percentile of absolute values
    if vrange is None:
        vrange = float(np.nanpercentile(np.abs(ddem_arr[np.isfinite(ddem_arr)]), 95))
        vrange = max(vrange, 0.05)    # floor to avoid degenerate colorbars

    # Build a masked version that greys out sub-threshold pixels
    display = ddem_arr.copy()
    below_threshold = np.abs(display) < change.threshold_m
    display[below_threshold] = np.nan   # will render as "bad" color in colormap

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#d0d0d0")       # grey for sub-threshold

    fig, ax = plt.subplots(figsize=(10, 7))

    im = ax.imshow(
        display,
        cmap=cmap,
        vmin=-vrange,
        vmax=vrange,
        interpolation="nearest",
    )

    cbar = plt.colorbar(im, ax=ax, label="Elevation change [m]", shrink=0.7)

    _apply_style(
        ax,
        title=(
            f"dDEM  (threshold ±{change.threshold_m:.3f} m  "
            f"| subsidence={change.subsidence_n_pixels} px  "
            f"heave={change.heave_n_pixels} px)"
        ),
    )

    # Annotate stats
    stats_text = (
        f"median  {change.aoi_median:+.3f} m\n"
        f"NMAD    {change.aoi_nmad:.3f} m\n"
        f"changed {change.aoi_change_fraction:.1%}"
    )
    ax.text(
        0.02, 0.97, stats_text,
        transform=ax.transAxes,
        fontsize=8,
        va="top", ha="left",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    fig.tight_layout()
    out = cfg.output_path(filename)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# 4. dDEM histogram
# ---------------------------------------------------------------------------

def plot_ddem_histogram(
    change: ChangeResult,
    cfg: Config,
    filename: str = "ddem_histogram.png",
) -> Path:
    """
    Histogram of dDEM values with vertical lines marking the change threshold
    and the median. Subsidence and heave tails are shaded.
    """

    ddem_arr = np.array(change.ddem.data).astype(np.float32)
    vals = ddem_arr[np.isfinite(ddem_arr)]

    # Clip for display (5th–95th percentile range)
    lo = float(np.nanpercentile(vals, 1))
    hi = float(np.nanpercentile(vals, 99))

    fig, ax = plt.subplots(figsize=(9, 4))

    n, bins, patches = ax.hist(
        vals,
        bins=300,
        range=(lo, hi),
        density=True,
        color="#888888",
        alpha=0.6,
        linewidth=0,
    )

    # Shade tails beyond threshold
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < -change.threshold_m:
            patch.set_facecolor("#4878CF")
            patch.set_alpha(0.8)
        elif left_edge > change.threshold_m:
            patch.set_facecolor("#CF4848")
            patch.set_alpha(0.8)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", label="zero")
    ax.axvline(
        change.aoi_median, color="navy", linewidth=1.2, linestyle="-",
        label=f"median  {change.aoi_median:+.3f} m"
    )
    ax.axvline(
        -change.threshold_m, color="steelblue", linewidth=1.0, linestyle=":",
        label=f"±threshold  {change.threshold_m:.3f} m"
    )
    ax.axvline(change.threshold_m, color="steelblue", linewidth=1.0, linestyle=":")

    _apply_style(
        ax,
        title="Distribution of elevation change",
        xlabel="dDEM  [m]",
        ylabel="Density",
    )

    ax.legend(fontsize=8)
    fig.tight_layout()

    out = cfg.output_path(filename)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# 5. Summary CSV
# ---------------------------------------------------------------------------

def save_summary_csv(
    coreg_results: dict[str, CoregResult],
    change: ChangeResult,
    cfg: Config,
    filename: str = "summary.csv",
) -> Path:
    """
    Write a flat CSV with one row per co-registration method plus a
    separate row block for change statistics.
    """

    coreg_results = _as_result_map(coreg_results)
    # --- Co-registration table ---
    coreg_rows = []
    best_name = coreg_results["_best"]

    for name, r in coreg_results.items():
        if name == "_best":
            continue
        coreg_rows.append({
            "method":   name,
            "failed":   r.failed,
            "failure":  r.failure_reason,
            "median_m": round(r.median, 4) if not np.isnan(r.median) else "",
            "nmad_m":   round(r.nmad,   4) if not np.isnan(r.nmad)   else "",
            "std_m":    round(r.std,    4) if not np.isnan(r.std)    else "",
            "mae_m":    round(r.mae,    4) if not np.isnan(r.mae)    else "",
            "rmse_m":   round(r.rmse,   4) if not np.isnan(r.rmse)   else "",
            "n_stable": r.n_stable,
            "best":     name == best_name,
        })

    coreg_df = pd.DataFrame(coreg_rows)

    # --- Change statistics table ---
    change_rows = [{
        "metric":  k,
        "value":   v,
    } for k, v in {
        "threshold_m":          round(change.threshold_m, 4),
        "aoi_mean_m":           round(change.aoi_mean,    4),
        "aoi_median_m":         round(change.aoi_median,  4),
        "aoi_nmad_m":           round(change.aoi_nmad,    4),
        "aoi_std_m":            round(change.aoi_std,     4),
        "aoi_q683_m":           round(change.aoi_q683,    4),
        "aoi_q95_m":            round(change.aoi_q95,     4),
        "aoi_n_pixels":         change.aoi_n_pixels,
        "aoi_change_fraction":  round(change.aoi_change_fraction, 4),
        "stable_median_m":      round(change.stable_median, 4) if not np.isnan(change.stable_median) else "",
        "stable_nmad_m":        round(change.stable_nmad,   4) if not np.isnan(change.stable_nmad)   else "",
        "stable_n_pixels":      change.stable_n_pixels,
        "subsidence_mean_m":    round(change.subsidence_mean_m, 4) if not np.isnan(change.subsidence_mean_m) else "",
        "subsidence_n_pixels":  change.subsidence_n_pixels,
        "heave_mean_m":         round(change.heave_mean_m, 4) if not np.isnan(change.heave_mean_m) else "",
        "heave_n_pixels":       change.heave_n_pixels,
        "volume_loss_m3":       round(change.volume_loss_m3, 1) if change.volume_loss_m3 is not None else "",
        "volume_gain_m3":       round(change.volume_gain_m3, 1) if change.volume_gain_m3 is not None else "",
    }.items()]

    change_df = pd.DataFrame(change_rows)

    out = cfg.output_path(filename)

    with open(out, "w") as f:
        f.write("# Co-registration results\n")
        coreg_df.to_csv(f, index=False)
        f.write("\n# Change statistics\n")
        change_df.to_csv(f, index=False)

    print(f"  Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# 6. Save aligned DEM and dDEM as GeoTIFFs
# ---------------------------------------------------------------------------

def save_rasters(
    coreg_results: dict[str, CoregResult],
    change: ChangeResult,
    cfg: Config,
) -> tuple[Path, Path]:
    """Save the best-aligned DEM and the dDEM as GeoTIFFs."""

    coreg_results = _as_result_map(coreg_results)
    from coregister import best_result

    best = best_result(coreg_results)

    aligned_path = cfg.output_path("aligned_best.tif")
    best.aligned_dem.save(str(aligned_path))
    print(f"  Saved: {aligned_path}")

    ddem_path = cfg.output_path("ddem.tif")
    change.ddem.save(str(ddem_path))
    print(f"  Saved: {ddem_path}")

    return aligned_path, ddem_path


# ---------------------------------------------------------------------------
# Convenience: run all outputs at once
# ---------------------------------------------------------------------------

def save_report(
    coreg_results: dict[str, CoregResult],
    change: ChangeResult,
    cfg: Config,
) -> None:
    """
    Generate all standard outputs in one call.
    Individual functions can still be called separately for custom workflows.
    """

    print("\n--- Saving report ---")
    plot_coreg_histograms(coreg_results, cfg)
    plot_residual_map(coreg_results, cfg)
    plot_ddem_map(change, cfg)
    plot_ddem_histogram(change, cfg)
    save_summary_csv(coreg_results, change, cfg)
    save_rasters(coreg_results, change, cfg)
    print("--- Report complete ---\n")