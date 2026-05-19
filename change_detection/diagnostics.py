from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import Config
from coregister import CoregResult, StepDiagnostic


def stepwise_table(coreg: CoregResult) -> pd.DataFrame:
    rows = []
    for d in coreg.step_diagnostics:
        rows.append({
            "step": d.step_name,
            "pipeline": d.pipeline_description,
            "median_m": round(d.median, 4) if np.isfinite(d.median) else "",
            "nmad_m": round(d.nmad, 4) if np.isfinite(d.nmad) else "",
            "std_m": round(d.std, 4) if np.isfinite(d.std) else "",
            "mae_m": round(d.mae, 4) if np.isfinite(d.mae) else "",
            "rmse_m": round(d.rmse, 4) if np.isfinite(d.rmse) else "",
            "n_stable": d.n_stable,
            "aspect_r2": round(d.aspect_r2, 4) if np.isfinite(d.aspect_r2) else "",
        })
    return pd.DataFrame(rows)


def save_stepwise_evolution_csv(coreg: CoregResult, cfg: Config, filename: str = "coreg_stepwise_evolution.csv") -> Path:
    out = cfg.output_path(filename)
    stepwise_table(coreg).to_csv(out, index=False)
    print(f"  Saved: {out}")
    return out


def save_final_coreg_diagnostic_plots(coreg: CoregResult, ref_dem, cfg: Config) -> list[Path]:
    if not coreg.step_diagnostics:
        return []

    final_step: StepDiagnostic = coreg.step_diagnostics[-1]
    outputs: list[Path] = []

    ref_arr = np.array(ref_dem.data, dtype=np.float32)
    corr_arr = np.array(final_step.corrected_dem.data, dtype=np.float32)
    valid = np.isfinite(ref_arr) & np.isfinite(corr_arr)
    x = ref_arr[valid]
    y = corr_arr[valid]

    if len(x) == 0:
        return outputs

    # 1. Scatter corrected vs reference
    fig, ax = plt.subplots(figsize=(5, 5))
    n = min(len(x), 15000)
    idx = np.random.default_rng(0).choice(len(x), n, replace=False)
    ax.scatter(x[idx], y[idx], s=2, alpha=0.3)
    lo = float(np.nanpercentile(np.concatenate([x, y]), 1))
    hi = float(np.nanpercentile(np.concatenate([x, y]), 99))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    mae = float(np.nanmean(np.abs(y - x)))
    rmse = float(np.sqrt(np.nanmean((y - x) ** 2)))
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2) if len(x) > 2 else np.nan
    ax.text(0.02, 0.98, f"MAE={mae:.3f}\nRMSE={rmse:.3f}\nR²={r2:.3f}", transform=ax.transAxes, va="top", fontsize=8)
    ax.set_title(f"Final coreg scatter ({final_step.pipeline_description})", fontsize=10)
    ax.set_xlabel("Reference elevation [m]")
    ax.set_ylabel("Corrected elevation [m]")
    ax.spines[["top", "right"]].set_visible(False)
    out = cfg.output_path("diag_final_scatter.png")
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig); outputs.append(out)

    # 2. Residual histogram
    fig, ax = plt.subplots(figsize=(6, 3.5))
    res = final_step.residuals[np.isfinite(final_step.residuals)]
    if len(res) > 0:
        ax.hist(res, bins=120, color="0.5", alpha=0.8, density=True)
        ax.axvline(final_step.median, color="r", ls="--", lw=1)
    ax.set_title("Final stable-ground residuals", fontsize=10)
    ax.set_xlabel("Residual [m]")
    ax.set_ylabel("Density")
    ax.spines[["top", "right"]].set_visible(False)
    out = cfg.output_path("diag_final_residual_hist.png")
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig); outputs.append(out)

    # 3. Elevation distribution comparison
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(x, bins=120, alpha=0.5, density=True, label="Reference")
    ax.hist(y, bins=120, alpha=0.5, density=True, label="Corrected")
    ax.legend(fontsize=8)
    ax.set_title("Final elevation distributions", fontsize=10)
    ax.set_xlabel("Elevation [m]")
    ax.set_ylabel("Density")
    ax.spines[["top", "right"]].set_visible(False)
    out = cfg.output_path("diag_final_elevation_hist.png")
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig); outputs.append(out)

    # 4. Aspect diagnostic
    if len(final_step.aspect_bin_centres) > 0:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(final_step.aspect_bin_centres, final_step.aspect_bin_means, "o-")
        ax.text(0.02, 0.95, f"R²={final_step.aspect_r2:.3f}", transform=ax.transAxes, va="top", fontsize=8)
        ax.set_title("Final aspect-vs-residual diagnostic", fontsize=10)
        ax.set_xlabel("Aspect [deg]")
        ax.set_ylabel("Median residual [m]")
        ax.spines[["top", "right"]].set_visible(False)
        out = cfg.output_path("diag_final_aspect.png")
        fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig); outputs.append(out)

    return outputs
