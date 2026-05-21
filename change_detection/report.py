import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    line_max = np.nanpercentile(np.r_[x, y], 99)

    ax.plot(
        [line_min, line_max],
        [line_min, line_max],
        "r--",
        lw=1.2,
    )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Reference elevation [m]")
    ax.set_ylabel("Target elevation [m]")

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
    valid = coreg_data["valid_mask"]

    ddem = change_data["ddem"]

    ddem_arr = change_data["ddem_arr"].astype(
        np.float32
    )

    stats = change_data["change_stats"]

    # ---------------------------------------------------------
    # Percentile clipping (plotting only)
    # ---------------------------------------------------------
    ddem_abs = np.abs(
        ddem_arr[np.isfinite(ddem_arr)]
    )

    plot_limit = np.nanpercentile(
        ddem_abs,
        99,
    )

    ddem_plot = np.where(
        np.abs(ddem_arr) > plot_limit,
        np.nan,
        ddem_arr,
    )

    # ---------------------------------------------------------
    # Plot masks
    # ---------------------------------------------------------
    residual = tgt_arr - ref_arr

    plot_valid = (
        valid
        & np.isfinite(ref_arr)
        & np.isfinite(tgt_arr)
    )

    plot_valid_stable = (
        plot_valid
        & stable
    )

    stable_ddem = ddem_plot[
        plot_valid_stable
        & np.isfinite(ddem_plot)
    ]

    # ---------------------------------------------------------
    # All-pixel dDEM statistics
    # ---------------------------------------------------------
    all_ddem = ddem_plot[
        np.isfinite(ddem_plot)
    ]

    all_mean = np.nanmean(all_ddem)

    all_median = np.nanmedian(all_ddem)

    all_nmad = 1.4826 * np.nanmedian(
        np.abs(all_ddem - all_median)
    )

    all_std = np.nanstd(all_ddem)

    all_mae = np.nanmean(
        np.abs(all_ddem)
    )

    all_rmse = np.sqrt(
        np.nanmean(all_ddem ** 2)
    )

    # ---------------------------------------------------------
    # Stable-ground statistics
    # ---------------------------------------------------------
    stable_median = np.nanmedian(
        stable_ddem
    )

    stable_nmad = stats["stable_nmad"]

    stable_std = np.nanstd(
        stable_ddem
    )

    stable_rmse = stats["stable_rmse"]

    # ---------------------------------------------------------
    # Save rasters
    # ---------------------------------------------------------
    coreg_data["target_coreg"].save(
        str(cfg.corrected_target_tif)
    )

    ddem.save(str(cfg.ddem_tif))

    # ---------------------------------------------------------
    # Save summary CSV
    # ---------------------------------------------------------
    rows = [
        {
            "metric": "aoi_name",
            "value": cfg.aoi_name,
        },
        {
            "metric": "dem_type",
            "value": cfg.dem_type,
        },
        {
            "metric": "coreg_method",
            "value": cfg.coreg_method,
        },
        {
            "metric": "years",
            "value": (
                f"{cfg.ref_year}-"
                f"{cfg.target_year}"
            ),
        },
        {
            "metric": "plot_clip_percentile",
            "value": 99,
        },
        {
            "metric": "plot_clip_limit_m",
            "value": float(plot_limit),
        },
    ]

    # ---------------------------------------------------------
    # Change statistics
    # ---------------------------------------------------------
    rows += [
        {"metric": k, "value": v}
        for k, v in stats.items()
    ]

    # ---------------------------------------------------------
    # All-pixel statistics
    # ---------------------------------------------------------
    rows += [
        {
            "metric": "all_mean",
            "value": all_mean,
        },
        {
            "metric": "all_median",
            "value": all_median,
        },
        {
            "metric": "all_nmad",
            "value": all_nmad,
        },
        {
            "metric": "all_std",
            "value": all_std,
        },
        {
            "metric": "all_mae",
            "value": all_mae,
        },
        {
            "metric": "all_rmse",
            "value": all_rmse,
        },
        {
            "metric": "all_pixel_count",
            "value": int(all_ddem.size),
        },
    ]

    # ---------------------------------------------------------
    # Stable-ground coreg stats
    # ---------------------------------------------------------
    rows += [
        {
            "metric": f"coreg_{k}",
            "value": v,
        }
        for k, v in coreg_data[
            "stable_stats"
        ].items()
    ]

    pd.DataFrame(rows).to_csv(
        cfg.summary_csv,
        index=False,
    )

    # =========================================================
    # Figure 1: DEM agreement + distributions
    # =========================================================
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
        constrained_layout=True,
    )

    # ---------------------------------------------------------
    # Scatter: all pixels
    # ---------------------------------------------------------
    hb1 = _hex_scatter(
        axes[0, 0],
        ref_arr[plot_valid],
        tgt_arr[plot_valid],
        "All pixels",
    )

    fig.colorbar(
        hb1,
        ax=axes[0, 0],
        label="log10(N)",
    )

    # ---------------------------------------------------------
    # Scatter: stable ground
    # ---------------------------------------------------------
    hb2 = _hex_scatter(
        axes[0, 1],
        ref_arr[plot_valid_stable],
        tgt_arr[plot_valid_stable],
        "Stable-ground pixels",
    )

    fig.colorbar(
        hb2,
        ax=axes[0, 1],
        label="log10(N)",
    )

    # ---------------------------------------------------------
    # Elevation distribution (all pixels)
    # ---------------------------------------------------------
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

    axes[1, 0].set_title(
        "Elevation distribution (all pixels)"
    )

    axes[1, 0].set_xlabel(
        "Elevation [m]"
    )

    axes[1, 0].set_ylabel(
        "Density"
    )

    axes[1, 0].legend()

    # ---------------------------------------------------------
    # All-pixel textbox
    # ---------------------------------------------------------
    all_stats_txt = (
        f"N: {all_ddem.size:,}\n"
        f"Mean: {all_mean:+.3f} m\n"
        f"Median: {all_median:+.3f} m\n"
        f"NMAD: {all_nmad:.3f} m\n"
        f"STD: {all_std:.3f} m\n"
        f"RMSE: {all_rmse:.3f} m"
    )

    axes[1, 0].text(
        0.98,
        0.97,
        all_stats_txt,
        transform=axes[1, 0].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(
            facecolor="white",
            alpha=0.75,
            edgecolor="0.7",
        ),
    )

    # ---------------------------------------------------------
    # Elevation distribution (stable terrain)
    # ---------------------------------------------------------
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

    axes[1, 1].set_title(
        "Elevation distribution (stable ground)"
    )

    axes[1, 1].set_xlabel(
        "Elevation [m]"
    )

    axes[1, 1].set_ylabel(
        "Density"
    )

    axes[1, 1].legend()

    # ---------------------------------------------------------
    # Stable-ground textbox
    # ---------------------------------------------------------
    stable_txt = (
        f"N: {stable_ddem.size:,}\n"
        f"Median: {stable_median:+.3f} m\n"
        f"NMAD: {stable_nmad:.3f} m\n"
        f"STD: {stable_std:.3f} m\n"
        f"RMSE: {stable_rmse:.3f} m"
    )

    axes[1, 1].text(
        0.98,
        0.97,
        stable_txt,
        transform=axes[1, 1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(
            facecolor="white",
            alpha=0.75,
            edgecolor="0.7",
        ),
    )

    fig.suptitle(
        f"{cfg.aoi_name} "
        f"{cfg.dem_type}: "
        f"DEM comparison "
        f"{cfg.ref_year} vs "
        f"{cfg.target_year}",
        fontsize=13,
    )

    fig.savefig(
        cfg.agreement_png,
        dpi=300,
    )

    plt.close(fig)

    # =========================================================
    # Figure 2: dDEM distributions
    # =========================================================
    fig2, ax2 = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
        constrained_layout=True,
    )

    # ---------------------------------------------------------
    # All-pixel dDEM distribution
    # ---------------------------------------------------------
    ax2[0].hist(
        all_ddem,
        bins=120,
        density=True,
        color="0.5",
        alpha=0.85,
    )

    ax2[0].axvline(
        0,
        color="k",
        lw=1,
        ls="--",
    )

    ax2[0].axvline(
        all_median,
        color="red",
        lw=1.5,
    )

    ax2[0].axvspan(
        all_median - all_nmad,
        all_median + all_nmad,
        alpha=0.15,
    )

    ax2[0].set_title(
        "All pixels"
    )

    ax2[0].set_xlabel(
        f"d{cfg.dem_type} [m]"
    )

    ax2[0].set_ylabel(
        "Density"
    )

    all_txt = (
        f"N: {all_ddem.size:,}\n"
        f"Mean: {all_mean:+.3f} m\n"
        f"Median: {all_median:+.3f} m\n"
        f"NMAD: {all_nmad:.3f} m\n"
        f"STD: {all_std:.3f} m\n"
        f"RMSE: {all_rmse:.3f} m\n"
        f"P99 clip: ±{plot_limit:.2f} m"
    )

    ax2[0].text(
        0.98,
        0.97,
        all_txt,
        transform=ax2[0].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(
            facecolor="white",
            alpha=0.75,
            edgecolor="0.7",
        ),
    )

    # ---------------------------------------------------------
    # Stable-ground dDEM distribution
    # ---------------------------------------------------------
    ax2[1].hist(
        stable_ddem,
        bins=100,
        density=True,
        color="teal",
        alpha=0.85,
    )

    ax2[1].axvline(
        0,
        color="k",
        lw=1,
        ls="--",
    )

    ax2[1].axvline(
        stable_median,
        color="red",
        lw=1.5,
    )

    ax2[1].axvspan(
        stable_median - stable_nmad,
        stable_median + stable_nmad,
        alpha=0.15,
    )

    ax2[1].set_title(
        "Stable ground"
    )

    ax2[1].set_xlabel(
        f"d{cfg.dem_type} [m]"
    )

    ax2[1].set_ylabel(
        "Density"
    )

    stable_txt = (
        f"N: {stable_ddem.size:,}\n"
        f"Median: {stable_median:+.3f} m\n"
        f"NMAD: {stable_nmad:.3f} m\n"
        f"STD: {stable_std:.3f} m\n"
        f"RMSE: {stable_rmse:.3f} m"
    )

    ax2[1].text(
        0.98,
        0.97,
        stable_txt,
        transform=ax2[1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(
            facecolor="white",
            alpha=0.75,
            edgecolor="0.7",
        ),
    )

    # ---------------------------------------------------------
    # Consistent x-axis scaling
    # ---------------------------------------------------------
    #for a in ax2:
    #    a.set_xlim(
    #        -plot_limit,
    #        plot_limit,
    #    )

    fig2.suptitle(
        f"{cfg.aoi_name} "
        f"{cfg.dem_type} "
        f"{cfg.ref_year}-"
        f"{cfg.target_year} "
        f"dDEM distributions",
        fontsize=13,
    )

    fig2.savefig(
        cfg.distribution_png,
        dpi=220,
    )

    plt.close(fig2)