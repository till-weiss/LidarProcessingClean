import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ─────────────────────────────────────────────
# Global style
# ─────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Colour palette used across figures
METHOD_COLORS = {
    "none":           "#5B9BD5",   # steel blue
    "vertical_shift": "#E07B54",   # coral
    "nuth_kaab":      "#6BAD6B",   # sage green
}

# Human-readable x-axis labels for the violin plot
METHOD_LABELS = {
    "none":           "No correction",
    "vertical_shift": "Vertical shift",
    "nuth_kaab":      "Nuth & Kääb",
}

# Noise-floor band drawn on the violin plot (±m)
NOISE_FLOOR_M = 0.05
percentile = 99

def _add_grid(ax, axis="y"):
    """Subtle background grid; always drawn below data."""
    ax.set_axisbelow(True)
    if axis in ("y", "both"):
        ax.yaxis.grid(
            True,
            linestyle="--",
            linewidth=0.4,
            alpha=0.5,
            color="grey",
        )
    if axis in ("x", "both"):
        ax.xaxis.grid(
            True,
            linestyle="--",
            linewidth=0.4,
            alpha=0.5,
            color="grey",
        )


def _hex_scatter(ax, x, y, title):
    hb = ax.hexbin(
        x,
        y,
        gridsize=110,
        bins="log",
        mincnt=1,
        cmap="viridis",
    )

    line_min = np.nanpercentile(np.r_[x, y], 1)
    line_max = np.nanpercentile(np.r_[x, y], percentile)

    # White 1:1 line – visible against both dense/sparse hexbin regions
    ax.plot(
        [line_min, line_max],
        [line_min, line_max],
        color="white",
        lw=0.8,
        ls="--",
    )

    ax.set_title(title)
    ax.set_xlabel("Reference elevation [m]")
    ax.set_ylabel("Target elevation [m]")
    _add_grid(ax, axis="both")

    return hb


def save_outputs(cfg, coreg_data, change_data):
    print("\n[3/3] Save rasters, CSV, and diagnostics")

    ref_arr = np.array(
        coreg_data["reference_dem"].data
    ).astype(np.float32)

    tgt_arr = np.array(
        coreg_data["target_coreg"].data
    ).astype(np.float32)

    stable = coreg_data["stable_mask"]
    valid  = coreg_data["valid_mask"]

    ddem     = change_data["ddem"]
    ddem_arr = change_data["ddem_arr"].astype(np.float32)
    stats    = change_data["change_stats"]

    # ── Percentile clipping (plotting only) ──────────────────
    ddem_abs   = np.abs(ddem_arr[np.isfinite(ddem_arr)])
    plot_limit = np.nanpercentile(ddem_abs, percentile)

    ddem_plot = np.where(
        np.abs(ddem_arr) > plot_limit,
        np.nan,
        ddem_arr,
    )

    # ── Plot masks ────────────────────────────────────────────
    residual = tgt_arr - ref_arr

    plot_valid = (
        valid
        & np.isfinite(ref_arr)
        & np.isfinite(tgt_arr)
    )

    plot_valid_stable = plot_valid & stable

    stable_ddem = ddem_plot[
        plot_valid_stable & np.isfinite(ddem_plot)
    ]

    # ── All-pixel dDEM statistics ─────────────────────────────
    all_ddem   = ddem_plot[np.isfinite(ddem_plot)]
    all_mean   = np.nanmean(all_ddem)
    all_median = np.nanmedian(all_ddem)
    all_nmad   = 1.4826 * np.nanmedian(
        np.abs(all_ddem - all_median)
    )
    all_std  = np.nanstd(all_ddem)
    all_mae  = np.nanmean(np.abs(all_ddem))
    all_rmse = np.sqrt(np.nanmean(all_ddem ** 2))

    # ── Stable-ground statistics ──────────────────────────────
    stable_median = np.nanmedian(stable_ddem)
    stable_nmad   = stats["stable_nmad"]
    stable_std    = np.nanstd(stable_ddem)
    stable_rmse   = stats["stable_rmse"]

    # ── Save rasters ──────────────────────────────────────────
    coreg_data["target_coreg"].save(str(cfg.corrected_target_tif))
    ddem.save(str(cfg.ddem_tif))

    # ── Save summary CSV ──────────────────────────────────────
    rows = [
        {"metric": "aoi_name",            "value": cfg.aoi_name},
        {"metric": "dem_type",            "value": cfg.dem_type},
        {"metric": "coreg_method",        "value": cfg.coreg_method},
        {"metric": "years",               "value": f"{cfg.ref_year}-{cfg.target_year}"},
        {"metric": "plot_clip_percentile","value": percentile},
        {"metric": "plot_clip_limit_m",   "value": float(plot_limit)},
    ]

    rows += [{"metric": k, "value": v} for k, v in stats.items()]

    rows += [
        {"metric": "all_mean",        "value": all_mean},
        {"metric": "all_median",      "value": all_median},
        {"metric": "all_nmad",        "value": all_nmad},
        {"metric": "all_std",         "value": all_std},
        {"metric": "all_mae",         "value": all_mae},
        {"metric": "all_rmse",        "value": all_rmse},
        {"metric": "all_pixel_count", "value": int(all_ddem.size)},
    ]

    comparison = coreg_data.get("coreg_comparison", {})

    for method_name, result in comparison.items():
        ms = result["stats"]
        rows += [
            {"metric": f"{method_name}_median", "value": ms["median"]},
            {"metric": f"{method_name}_nmad",   "value": ms["nmad"]},
            {"metric": f"{method_name}_rmse",   "value": ms["rmse"]},
            {"metric": f"{method_name}_std",    "value": ms["std"]},
        ]

    pd.DataFrame(rows).to_csv(cfg.summary_csv, index=False)

    # =========================================================
    # Figure 1: DEM agreement + distributions
    # =========================================================
    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 9),
        constrained_layout=True,
    )

    # Scatter – all pixels
    hb1 = _hex_scatter(
        axes[0, 0],
        ref_arr[plot_valid],
        tgt_arr[plot_valid],
        "All pixels",
    )
    fig.colorbar(hb1, ax=axes[0, 0], label="log₁₀(N)")

    # Scatter – stable ground
    hb2 = _hex_scatter(
        axes[0, 1],
        ref_arr[plot_valid_stable],
        tgt_arr[plot_valid_stable],
        "Stable-ground pixels",
    )
    fig.colorbar(hb2, ax=axes[0, 1], label="log₁₀(N)")

    # Elevation distribution – all pixels
    axes[1, 0].hist(
        ref_arr[plot_valid],
        bins=120,
        density=True,
        histtype="step",
        linewidth=1.5,
        label=str(cfg.ref_year),
    )
    axes[1, 0].hist(
        tgt_arr[plot_valid],
        bins=120,
        density=True,
        histtype="step",
        linewidth=1.5,
        label=str(cfg.target_year),
    )
    axes[1, 0].set_title("Elevation distribution (all pixels)")
    axes[1, 0].set_xlabel("Elevation [m]")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].legend()
    _add_grid(axes[1, 0])

    axes[1, 0].text(
        0.98, 0.97,
        (
            f"N: {all_ddem.size:,}\n"
            f"Mean: {all_mean:+.3f} m\n"
            f"Median: {all_median:+.3f} m\n"
            f"NMAD: {all_nmad:.3f} m\n"
            f"STD: {all_std:.3f} m\n"
            f"RMSE: {all_rmse:.3f} m"
        ),
        transform=axes[1, 0].transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(facecolor="white", alpha=0.80, edgecolor="0.7", boxstyle="round,pad=0.3"),
    )

    # Elevation distribution – stable ground
    axes[1, 1].hist(
        ref_arr[plot_valid_stable],
        bins=100,
        density=True,
        histtype="step",
        linewidth=1.5,
        label=str(cfg.ref_year),
    )
    axes[1, 1].hist(
        tgt_arr[plot_valid_stable],
        bins=100,
        density=True,
        histtype="step",
        linewidth=1.5,
        label=str(cfg.target_year),
    )
    axes[1, 1].set_title("Elevation distribution (stable ground)")
    axes[1, 1].set_xlabel("Elevation [m]")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].legend()
    _add_grid(axes[1, 1])

    axes[1, 1].text(
        0.98, 0.97,
        (
            f"N: {stable_ddem.size:,}\n"
            f"Median: {stable_median:+.3f} m\n"
            f"NMAD: {stable_nmad:.3f} m\n"
            f"STD: {stable_std:.3f} m\n"
            f"RMSE: {stable_rmse:.3f} m"
        ),
        transform=axes[1, 1].transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(facecolor="white", alpha=0.80, edgecolor="0.7", boxstyle="round,pad=0.3"),
    )

    fig.suptitle(
        f"{cfg.aoi_name}  |  {cfg.dem_type}: DEM comparison "
        f"{cfg.ref_year} vs {cfg.target_year}",
        fontsize=13, fontweight="bold",
    )

    fig.savefig(cfg.agreement_png, dpi=300)
    plt.close(fig)

    # =========================================================
    # Figure 2: dDEM distributions
    # =========================================================
    fig2, ax2 = plt.subplots(
        1, 2,
        figsize=(11, 4.5),
        constrained_layout=True,
    )

    # All-pixel dDEM
    ax2[0].hist(
        all_ddem,
        bins=120,
        density=True,
        color="0.5",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.2,
    )
    ax2[0].axvline(0,          color="k",   lw=1,   ls="--", label="Zero")
    ax2[0].axvline(all_median, color="C3",  lw=1.5,          label=f"Median {all_median:+.3f} m")
    ax2[0].axvspan(
        all_median - all_nmad,
        all_median + all_nmad,
        alpha=0.20, color="C3",
        label=f"±NMAD ({all_nmad:.3f} m)",
    )
    ax2[0].set_title("All pixels")
    ax2[0].set_xlabel(f"d{cfg.dem_type} [m]")
    ax2[0].set_ylabel("Density")
    ax2[0].legend(fontsize=9)
    _add_grid(ax2[0])

    ax2[0].text(
        0.02, 0.97,
        (
            f"N: {all_ddem.size:,}\n"
            f"Mean: {all_mean:+.3f} m\n"
            f"Median: {all_median:+.3f} m\n"
            f"NMAD: {all_nmad:.3f} m\n"
            f"STD: {all_std:.3f} m\n"
            f"RMSE: {all_rmse:.3f} m\n"
            f"P{percentile} clip: ±{plot_limit:.2f} m"
        ),
        transform=ax2[0].transAxes,
        ha="left", va="top", fontsize=9,
        bbox=dict(facecolor="white", alpha=0.80, edgecolor="0.7", boxstyle="round,pad=0.3"),
    )

    # Stable-ground dDEM  – tight x-axis
    ax2[1].hist(
        stable_ddem,
        bins=100,
        density=True,
        color="teal",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.2,
    )
    ax2[1].axvline(0,             color="k",    lw=1,   ls="--")
    ax2[1].axvline(stable_median, color="C3",   lw=1.5, label=f"Median {stable_median:+.3f} m")
    ax2[1].axvspan(
        stable_median - stable_nmad,
        stable_median + stable_nmad,
        alpha=0.25, color="C3",
        label=f"±NMAD ({stable_nmad:.3f} m)",
    )
    ax2[1].set_title("Stable ground")
    ax2[1].set_xlabel(f"d{cfg.dem_type} [m]")
    ax2[1].set_ylabel("Density")
    ax2[1].legend(fontsize=9)
    _add_grid(ax2[1])

    # Clip to ±0.3 m so the tight distribution is readable
    ax2[1].set_xlim(-0.30, 0.30)

    ax2[1].text(
        0.98, 0.97,
        (
            f"N: {stable_ddem.size:,}\n"
            f"Median: {stable_median:+.3f} m\n"
            f"NMAD: {stable_nmad:.3f} m\n"
            f"STD: {stable_std:.3f} m\n"
            f"RMSE: {stable_rmse:.3f} m"
        ),
        transform=ax2[1].transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(facecolor="white", alpha=0.80, edgecolor="0.7", boxstyle="round,pad=0.3"),
    )

    fig2.suptitle(
        f"{cfg.aoi_name}  |  {cfg.dem_type}  {cfg.ref_year}–{cfg.target_year}  dDEM distributions",
        fontsize=13, fontweight="bold",
    )

    fig2.savefig(cfg.distribution_png, dpi=220)
    plt.close(fig2)

    # =========================================================
    # Figure 3: Co-registration method comparison (violin)
    # =========================================================
    if len(comparison) > 1:

        methods        = list(comparison.keys())
        residual_lists = []

        for method_name, result in comparison.items():
            res = result["residuals"]
            residual_lists.append(res[np.isfinite(res)])

        display_labels = [
            METHOD_LABELS.get(m, m) for m in methods
        ]
        colors = [
            METHOD_COLORS.get(m, "#AAAAAA") for m in methods
        ]

        fig3, ax3 = plt.subplots(figsize=(9, 5.5), constrained_layout=True)

        _add_grid(ax3)

        # ── Noise-floor band ──────────────────────────────────
       # ax3.axhspan(
       #     -NOISE_FLOOR_M, NOISE_FLOOR_M,
       #     color="gold", alpha=0.18,
       #     label=f"±{NOISE_FLOOR_M} m noise floor",
       #     zorder=0,
       # )

        # ── Zero line ─────────────────────────────────────────
        ax3.axhline(0, color="k", lw=1, ls="--", zorder=1)

        # ── Violin plots (one colour per method) ─────────────
        positions = np.arange(1, len(methods) + 1)

        violin = ax3.violinplot(
            residual_lists,
            positions=positions,
            showmeans=False,
            showmedians=False,   # we draw our own via boxplot
            showextrema=False,
        )

        for body, col in zip(violin["bodies"], colors):
            body.set_facecolor(col)
            body.set_edgecolor("0.3")
            body.set_linewidth(0.8)
            body.set_alpha(0.55)

        # ── Boxplot overlay ───────────────────────────────────
        bp = ax3.boxplot(
            residual_lists,
            positions=positions,
            widths=0.16,
            showfliers=False,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.8),
            boxprops=dict(facecolor="none", edgecolor="0.3", linewidth=1),
            whiskerprops=dict(color="0.3", linewidth=1),
            capprops=dict(color="0.3", linewidth=1),
        )

        # ── NMAD annotations above each violin ───────────────
        ymax_annot = np.nanpercentile(
            np.abs(np.concatenate(residual_lists)), 98
        )

        for i, res in enumerate(residual_lists):
            nmad_val = 1.4826 * np.nanmedian(
                np.abs(res - np.nanmedian(res))
            )
            ax3.text(
                positions[i],
                ymax_annot * 1.08,
                f"NMAD\n{nmad_val:.3f} m",
                ha="center", va="bottom",
                fontsize=9, color="0.2",
            )

        # ── Axes ──────────────────────────────────────────────
        ax3.set_xticks(positions)
        ax3.set_xticklabels(display_labels, fontsize=11)
        ax3.set_ylabel("Stable-ground residuals [m]")
        ax3.set_title(
            f"{cfg.aoi_name}  |  Co-registration method comparison",
            fontweight="bold",
        )

        ymax = np.nanpercentile(
            np.abs(np.concatenate(residual_lists)), percentile
        )
        ax3.set_ylim(-ymax * 1.25, ymax * 1.35)   # extra headroom for annotations

        ax3.legend(loc="lower right", fontsize=9)

        fig3.savefig(
            cfg.output_dir / "coreg_comparison_violin.png",
            dpi=220,
        )
        plt.close(fig3)

    # =========================================================
    # Figure 5: Residuals vs terrain slope
    # =========================================================
    slope_valid    = change_data["slope"][plot_valid_stable]
    residual_valid = residual[plot_valid_stable]

    # Identify outlier count before clipping for annotation
    clip_y = 0.5
    n_outliers = int(np.sum(np.abs(residual_valid) > clip_y))

    fig5, ax5 = plt.subplots(figsize=(7, 5), constrained_layout=True)

    _add_grid(ax5, axis="both")

    hb = ax5.hexbin(
        slope_valid,
        residual_valid,
        gridsize=60,
        bins="log",
        mincnt=1,
        cmap="viridis",
    )

    cb = fig5.colorbar(hb, ax=ax5)
    cb.set_label("log₁₀(N)")

    ax5.axhline(0, color="k", ls="--", lw=1, label="Zero")

    # ±NMAD band on the stable-ground residuals
    res_nmad = 1.4826 * np.nanmedian(
        np.abs(residual_valid - np.nanmedian(residual_valid))
    )
    ax5.axhspan(
        -res_nmad, res_nmad,
        color="C3", alpha=0.12,
        label=f"±NMAD ({res_nmad:.3f} m)",
    )

    ax5.set_ylim(-clip_y, clip_y)
    ax5.set_xlabel("Slope [°]")
    ax5.set_ylabel("Residual [m]")
    ax5.set_title(
        f"{cfg.aoi_name}  |  Stable-ground residuals vs terrain slope",
        fontweight="bold",
    )

    ax5.legend(fontsize=9, loc="upper right")

    # Annotate number of off-screen outliers
    if n_outliers > 0:
        ax5.text(
            0.01, percentile/100,
            f"{n_outliers:,} points outside ±{clip_y} m not shown",
            transform=ax5.transAxes,
            ha="left", va="top", fontsize=8.5,
            color="0.4", style="italic",
        )

    fig5.savefig(
        cfg.output_dir / "residual_vs_slope.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig5)