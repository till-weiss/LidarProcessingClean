import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _hex_scatter(ax, x, y, title):
    hb = ax.hexbin(x, y, gridsize=110, bins="log", mincnt=1, cmap="viridis")
    line_min = np.nanpercentile(np.r_[x, y], 1)
    line_max = np.nanpercentile(np.r_[x, y], 99)
    ax.plot([line_min, line_max], [line_min, line_max], "r--", lw=1.2)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Reference elevation [m]")
    ax.set_ylabel("Target elevation [m]")
    return hb


def save_outputs(cfg, coreg_data, change_data):
    print("\n[3/3] Save rasters, CSV, and diagnostics")
    ref_arr = np.array(coreg_data["reference_dem"].data).astype(np.float32)
    tgt_arr = np.array(coreg_data["target_coreg"].data).astype(np.float32)
    stable = coreg_data["stable_mask"]
    ddem = change_data["ddem"]
    ddem_arr = change_data["ddem_arr"]
    stats = change_data["change_stats"]

    coreg_data["target_coreg"].save(str(cfg.corrected_target_tif))
    ddem.save(str(cfg.ddem_tif))

    rows = [
        {"metric": "aoi_name", "value": cfg.aoi_name},
        {"metric": "dem_type", "value": cfg.dem_type},
        {"metric": "years", "value": f"{cfg.ref_year}-{cfg.target_year}"},
    ]
    rows += [{"metric": k, "value": v} for k, v in stats.items()]
    rows += [{"metric": f"coreg_{k}", "value": v} for k, v in coreg_data["stable_stats"].items()]
    pd.DataFrame(rows).to_csv(cfg.summary_csv, index=False)

    valid = np.isfinite(ref_arr) & np.isfinite(tgt_arr)
    valid_stable = valid & stable

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    hb1 = _hex_scatter(axes[0, 0], ref_arr[valid], tgt_arr[valid], "All pixels")
    fig.colorbar(hb1, ax=axes[0, 0], label="log10(N)")
    hb2 = _hex_scatter(axes[0, 1], ref_arr[valid_stable], tgt_arr[valid_stable], "Stable-ground pixels")
    fig.colorbar(hb2, ax=axes[0, 1], label="log10(N)")

    axes[1, 0].hist(ddem_arr[np.isfinite(ddem_arr)], bins=220, color="grey", alpha=0.8)
    axes[1, 0].axvline(0, color="k", ls="--", lw=1)
    axes[1, 0].set_title(f"d{cfg.dem_type} distribution (all pixels)")
    axes[1, 0].set_xlabel("Elevation change [m]")

    stable_ddem = ddem_arr[valid_stable]
    axes[1, 1].hist(stable_ddem, bins=180, color="steelblue", alpha=0.85)
    axes[1, 1].axvline(0, color="k", ls="--", lw=1)
    axes[1, 1].set_title("dDEM on stable ground")
    axes[1, 1].set_xlabel("Elevation change [m]")

    stats_txt = (
        f"Stable median: {stats['stable_median']:+.3f} m\n"
        f"Stable NMAD: {stats['stable_nmad']:.3f} m\n"
        f"Stable STD: {stats['stable_std']:.3f} m\n"
        f"Stable RMSE: {stats['stable_rmse']:.3f} m"
    )
    axes[1, 1].text(0.98, 0.97, stats_txt, transform=axes[1, 1].transAxes, ha="right", va="top", fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="0.7"))

    fig.suptitle(f"{cfg.aoi_name} {cfg.dem_type}: DEM agreement {cfg.ref_year} vs {cfg.target_year}", fontsize=13)
    fig.savefig(cfg.agreement_png, dpi=220)
    plt.close(fig)

    fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    ax2[0].hist(ddem_arr[np.isfinite(ddem_arr)], bins=260, density=True, color="0.4")
    ax2[0].set_title("All pixels")
    ax2[0].set_xlabel(f"d{cfg.dem_type} [m]")
    ax2[1].hist(stable_ddem, bins=220, density=True, color="teal")
    ax2[1].set_title("Stable ground")
    ax2[1].set_xlabel(f"d{cfg.dem_type} [m]")
    for a in ax2:
        a.axvline(0, color="k", lw=1, ls="--")
    fig2.suptitle(f"{cfg.aoi_name} {cfg.dem_type} {cfg.ref_year}-{cfg.target_year} distribution")
    fig2.savefig(cfg.distribution_png, dpi=220)
    plt.close(fig2)
