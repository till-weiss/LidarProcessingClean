from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import Config
from coregister import CoregResult, StepDiagnostic


def _clean(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _nmad(values: np.ndarray) -> float:
    if len(values) == 0:
        return np.nan
    med = np.nanmedian(values)
    return float(1.4826 * np.nanmedian(np.abs(values - med)))


def compute_distribution_stats(values: np.ndarray) -> dict:
    vals = _clean(values)
    if len(vals) == 0:
        return {k: np.nan for k in ["n", "mean", "median", "std", "nmad", "mae", "rmse", "skewness", "kurtosis"]}

    mean = float(np.mean(vals))
    median = float(np.median(vals))
    std = float(np.std(vals))
    centered = vals - mean
    m2 = float(np.mean(centered ** 2))
    m3 = float(np.mean(centered ** 3))
    m4 = float(np.mean(centered ** 4))
    skew = float(m3 / (m2 ** 1.5)) if m2 > 0 else np.nan
    kurt = float(m4 / (m2 ** 2) - 3.0) if m2 > 0 else np.nan

    return {
        "n": int(len(vals)),
        "mean": mean,
        "median": median,
        "std": std,
        "nmad": _nmad(vals),
        "mae": float(np.mean(np.abs(vals))),
        "rmse": float(np.sqrt(np.mean(vals ** 2))),
        "skewness": skew,
        "kurtosis": kurt,
    }


def _percentile_clip(values: np.ndarray, qlo: float = 0.5, qhi: float = 99.5) -> tuple[np.ndarray, tuple[float, float]]:
    vals = _clean(values)
    if len(vals) == 0:
        return vals, (np.nan, np.nan)
    lo, hi = np.percentile(vals, [qlo, qhi])
    return vals[(vals >= lo) & (vals <= hi)], (float(lo), float(hi))


def _kde_curve(values: np.ndarray, xgrid: np.ndarray) -> np.ndarray:
    vals = _clean(values)
    if len(vals) < 2:
        return np.zeros_like(xgrid)
    std = np.std(vals)
    if std <= 0:
        return np.zeros_like(xgrid)
    bw = 1.06 * std * (len(vals) ** (-1 / 5))
    bw = max(bw, 1e-6)
    z = (xgrid[:, None] - vals[None, :]) / bw
    dens = np.exp(-0.5 * z ** 2).sum(axis=1) / (len(vals) * bw * np.sqrt(2 * np.pi))
    return dens


def plot_histogram_panel(ax, values: np.ndarray, xlim: tuple[float, float], title: str, show_sigma: bool = True) -> None:
    vals = _clean(values)
    stats = compute_distribution_stats(vals)
    if len(vals) == 0:
        ax.set_title(title)
        return

    ax.hist(vals, bins=80, range=xlim, density=True, color="#4C78A8", alpha=0.5, edgecolor="none")
    xg = np.linspace(xlim[0], xlim[1], 400)
    ax.plot(xg, _kde_curve(vals, xg), color="#1f4f75", lw=1.5, label="KDE")

    mean, median, nmad, std = stats["mean"], stats["median"], stats["nmad"], stats["std"]
    ax.axvline(mean, color="red", ls="--", lw=1.1, label="mean")
    ax.axvline(median, color="black", ls="-", lw=1.2, label="median")
    if np.isfinite(nmad):
        ax.axvline(median - nmad, color="purple", ls=":", lw=1.0)
        ax.axvline(median + nmad, color="purple", ls=":", lw=1.0, label="±NMAD")
    if show_sigma and np.isfinite(std):
        ax.axvline(mean - 2 * std, color="gray", ls="--", lw=0.8)
        ax.axvline(mean + 2 * std, color="gray", ls="--", lw=0.8, label="±2σ")

    box = (
        f"n={stats['n']:,}\nmean={mean:.3f}\nmedian={median:.3f}\nstd={std:.3f}\n"
        f"NMAD={nmad:.3f}\nMAE={stats['mae']:.3f}\nRMSE={stats['rmse']:.3f}\n"
        f"skew={stats['skewness']:.3f}\nkurt={stats['kurtosis']:.3f}"
    )
    ax.text(0.02, 0.98, box, transform=ax.transAxes, va="top", ha="left", fontsize=7,
            bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9))
    ax.set_xlim(xlim)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Elevation difference [m]")
    ax.set_ylabel("Density")
    ax.spines[["top", "right"]].set_visible(False)


def plot_qq_panel(ax, values: np.ndarray, title: str, max_points: int = 20000) -> None:
    vals = _clean(values)
    if len(vals) == 0:
        ax.set_title(title)
        return
    rng = np.random.default_rng(0)
    if len(vals) > max_points:
        vals = rng.choice(vals, size=max_points, replace=False)
    vals = np.sort(vals)
    n = len(vals)
    q = (np.arange(n) + 0.5) / n
    normal_sample = np.sort(rng.normal(size=max(200000, n)))
    theo = np.quantile(normal_sample, q)

    slope, intercept = np.polyfit(theo, vals, 1)
    pred = slope * theo + intercept
    ss_res = np.sum((vals - pred) ** 2)
    ss_tot = np.sum((vals - np.mean(vals)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    ax.scatter(theo, vals, s=4, alpha=0.25, color="#2C7FB8")
    lo = min(theo.min(), vals.min())
    hi = max(theo.max(), vals.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="1:1")
    ax.plot([lo, hi], [slope * lo + intercept, slope * hi + intercept], color="red", lw=1.2, label="fit")
    ax.text(0.02, 0.98, f"slope={slope:.3f}\nintercept={intercept:.3f}\nR²={r2:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Theoretical normal quantiles")
    ax.set_ylabel("Observed quantiles [m]")
    ax.spines[["top", "right"]].set_visible(False)


def create_distribution_figure(values_all: np.ndarray, values_stable: np.ndarray, title: str, out_path: Path, dpi: int = 400) -> Path:
    all_clipped, (lo_all, hi_all) = _percentile_clip(values_all)
    st_clipped, (lo_st, hi_st) = _percentile_clip(values_stable)

    finite_lims = [v for v in [lo_all, hi_all, lo_st, hi_st] if np.isfinite(v)]
    if len(finite_lims) < 2:
        raise RuntimeError(f"No finite values for distribution figure: {title}")
    xlim = (min(finite_lims), max(finite_lims))

    fig, axs = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    fig.suptitle(title, fontsize=14)

    plot_qq_panel(axs[0, 0], all_clipped, "QQ plot (all pixels)")
    plot_qq_panel(axs[0, 1], st_clipped, "QQ plot (stable terrain)")
    plot_histogram_panel(axs[1, 0], all_clipped, xlim, "Histogram + KDE (all pixels)")
    plot_histogram_panel(axs[1, 1], st_clipped, xlim, "Histogram + KDE (stable terrain)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


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
