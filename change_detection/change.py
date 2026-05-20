import numpy as np
import geoutils as gu


def compute_change_products(cfg, coreg_data):
    print("\n[2/3] Compute dDEM / dDSM")
    ref_dem = coreg_data["reference_dem"]
    target_coreg = coreg_data["target_coreg"]
    stable_mask = coreg_data["stable_mask"]

    ddem = target_coreg - ref_dem
    ddem_arr = np.array(ddem.data).astype(np.float32)
    valid = np.isfinite(ddem_arr)

    all_vals = ddem_arr[valid]
    stable_vals = ddem_arr[stable_mask & valid]
    stable_vals = stable_vals[np.abs(stable_vals) < cfg.outlier_clip_m]

    if cfg.change_threshold_m is None:
        threshold_m = 2.0 * float(gu.stats.nmad(stable_vals))
    else:
        threshold_m = cfg.change_threshold_m

    change_stats = {
        "aoi_median": float(np.nanmedian(all_vals)),
        "aoi_nmad": float(gu.stats.nmad(all_vals)),
        "aoi_std": float(np.nanstd(all_vals)),
        "aoi_rmse": float(np.sqrt(np.nanmean(all_vals ** 2))),
        "n_all": int(all_vals.size),
        "stable_median": float(np.nanmedian(stable_vals)),
        "stable_nmad": float(gu.stats.nmad(stable_vals)),
        "stable_std": float(np.nanstd(stable_vals)),
        "stable_rmse": float(np.sqrt(np.nanmean(stable_vals ** 2))),
        "n_stable": int(stable_vals.size),
        "threshold_m": float(threshold_m),
    }

    px = abs(ref_dem.res[0] * ref_dem.res[1])
    subsidence = all_vals[all_vals < -threshold_m]
    heave = all_vals[all_vals > threshold_m]
    change_stats["volume_loss_m3"] = float(np.sum(np.abs(subsidence)) * px)
    change_stats["volume_gain_m3"] = float(np.sum(heave) * px)

    print(f"  d{cfg.dem_type} threshold: ±{threshold_m:.3f} m")
    return {"ddem": ddem, "ddem_arr": ddem_arr, "change_stats": change_stats}
