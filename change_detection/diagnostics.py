from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from config import Config
from coregister import CoregResult, StepDiagnostic


def _apply_style(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)


def _step_tag(name: str) -> str:
    return name.replace(" + ", "_").replace(" ", "_")


def save_stepwise_diagnostics(coreg: CoregResult, ref_dem, cfg: Config) -> list[Path]:
    outputs: list[Path] = []
    ref_arr = np.array(ref_dem.data, dtype=np.float32)

    for step in coreg.step_diagnostics:
        tag = _step_tag(step.step_name)
        corr_arr = np.array(step.corrected_dem.data, dtype=np.float32)
        valid = np.isfinite(ref_arr) & np.isfinite(corr_arr)
        x = ref_arr[valid]
        y = corr_arr[valid]

        if len(x) == 0:
            continue

        # 1 scatter
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
        _apply_style(ax, f"Scatter: {step.step_name}", "Reference", "Corrected")
        out = cfg.output_path(f"diag_scatter_{tag}.png")
        fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig); outputs.append(out)

        # 2 residual histogram
        fig, ax = plt.subplots(figsize=(6, 3.5))
        res = step.residuals[np.isfinite(step.residuals)]
        if len(res) > 0:
            ax.hist(res, bins=120, color="0.5", alpha=0.8, density=True)
            ax.axvline(step.median, color="r", ls="--", lw=1)
        _apply_style(ax, f"Residuals: {step.step_name}", "Residual [m]", "Density")
        out = cfg.output_path(f"diag_residual_hist_{tag}.png")
        fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig); outputs.append(out)

        # 3 elevation distributions
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(x, bins=120, alpha=0.5, density=True, label="Reference")
        ax.hist(y, bins=120, alpha=0.5, density=True, label="Corrected")
        ax.legend(fontsize=8)
        _apply_style(ax, f"Elevation distribution: {step.step_name}", "Elevation [m]", "Density")
        out = cfg.output_path(f"diag_elev_hist_{tag}.png")
        fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig); outputs.append(out)

        # 4 aspect plot
        if len(step.aspect_bin_centres) > 0:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(step.aspect_bin_centres, step.aspect_bin_means, "o-")
            ax.text(0.02, 0.95, f"R²={step.aspect_r2:.3f}", transform=ax.transAxes, va="top", fontsize=8)
            _apply_style(ax, f"Aspect vs dDEM: {step.step_name}", "Aspect [deg]", "Median residual [m]")
            out = cfg.output_path(f"diag_aspect_{tag}.png")
            fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig); outputs.append(out)

        # 5 residual map optional
        fig, ax = plt.subplots(figsize=(6, 5))
        residual_map = corr_arr - ref_arr
        im = ax.imshow(residual_map, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        plt.colorbar(im, ax=ax, shrink=0.7)
        _apply_style(ax, f"Residual map: {step.step_name}")
        out = cfg.output_path(f"diag_residual_map_{tag}.png")
        fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig); outputs.append(out)

    return outputs
