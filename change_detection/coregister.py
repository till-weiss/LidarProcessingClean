import numpy as np
import xdem
import geoutils as gu

from scipy.ndimage import binary_dilation


# =====================================================
# Stable terrain mask
# =====================================================

def load_stable_mask(
    stable_ground_path,
    ref_dem,
):

    if stable_ground_path is None:

        return np.isfinite(
            np.array(ref_dem.data)
        )

    stable_vector = gu.Vector(
        stable_ground_path
    )

    stable_mask = np.array(
        stable_vector.create_mask(
            ref_dem
        ).data
    ).astype(bool)

    return stable_mask


# =====================================================
# Water mask
# =====================================================

def load_water_mask(
    water_mask_path,
    ref_dem,
    buffer_pixels=3,
):

    if water_mask_path is None:

        return np.zeros(
            ref_dem.data.shape,
            dtype=bool,
        )

    water_mask = xdem.DEM(
        water_mask_path
    ).reproject(
        ref_dem,
        resampling="nearest",
        nodata=0,
    )

    water_arr = np.array(
        water_mask.data
    )

    nodata = water_mask.nodata

    if nodata is not None:
        water_arr = np.where(
            water_arr == nodata,
            0,
            water_arr
        )

    water_mask = water_arr.astype(bool)

    if buffer_pixels > 0:

        water_mask = binary_dilation(
            water_mask,
            iterations=buffer_pixels,
        )

    return water_mask


# =====================================================
# Valid data mask
# =====================================================

def build_valid_mask(
    ref_arr,
    target_arr,
):

    invalid_values = [
        -9999,
        -99999,
        -32768,
    ]

    valid_mask = (
        np.isfinite(ref_arr)
        & np.isfinite(target_arr)
    )

    for invalid in invalid_values:

        valid_mask &= (
            ref_arr != invalid
        )

        valid_mask &= (
            target_arr != invalid
        )

    return valid_mask


# =====================================================
# Helper: compute stable-ground stats from a
# co-registered DEM against the reference
# =====================================================

def _stable_residual_stats(tgt, ref_dem, valid_stable, outlier_clip_m):
    """
    Returns (residual_array, stats_dict) for pixels in valid_stable,
    clipped to ±outlier_clip_m.
    """
    res = np.array(
        (tgt - ref_dem).data
    ).astype(np.float32)

    vals = res[valid_stable]
    vals = vals[np.abs(vals) < outlier_clip_m]

    stats = {
        "median": float(np.nanmedian(vals)),
        "nmad":   float(gu.stats.nmad(vals)),
        "std":    float(np.nanstd(vals)),
        "rmse":   float(np.sqrt(np.nanmean(vals ** 2))),
        "n":      int(vals.size),
    }

    return vals, stats


# =====================================================
# Main co-registration workflow
# =====================================================

def co_register_dem_pair(cfg):

    print("\n[1/3] Co-register DEMs")

    # =================================================
    # Load DEMs
    # =================================================

    ref_dem = xdem.DEM(
        cfg.dem_reference_path
    )

    target_dem = xdem.DEM(
        cfg.dem_target_path
    ).reproject(
        ref_dem,
        resampling="max",
    )

    ref_arr = np.array(
        ref_dem.data
    ).astype(np.float32)

    target_arr = np.array(
        target_dem.data
    ).astype(np.float32)

    # =================================================
    # Stable terrain mask
    # =================================================

    stable_mask = load_stable_mask(
        cfg.stable_ground_path,
        ref_dem,
    )

    # =================================================
    # Water masks
    # =================================================

    ref_water_mask = load_water_mask(
        cfg.reference_water_mask_path,
        ref_dem,
        buffer_pixels=cfg.water_mask_buffer_pixels,
    )

    target_water_mask = load_water_mask(
        cfg.target_water_mask_path,
        ref_dem,
        buffer_pixels=cfg.water_mask_buffer_pixels,
    )

    combined_water_mask = (
        ref_water_mask
        | target_water_mask
    )

    non_water_mask = ~combined_water_mask

    # =================================================
    # Valid-data mask
    # =================================================

    valid_mask = build_valid_mask(
        ref_arr,
        target_arr,
    )

    # =================================================
    # Final inlier mask
    # =================================================

    valid_stable = (
        valid_mask
        & stable_mask
        & non_water_mask
    )

    print(
        f"  valid stable pixels: "
        f"{np.count_nonzero(valid_stable):,}"
    )

    if np.count_nonzero(valid_stable) == 0:

        raise RuntimeError(
            "No valid stable-ground pixels "
            "available for co-registration."
        )

    print(
        "  Stable pixels:",
        np.count_nonzero(stable_mask)
    )

    print(
        "  Stable pixels after water masking:",
        np.count_nonzero(
            stable_mask & non_water_mask
        )
    )

    # =================================================
    # Pre-coregistration residuals (no correction)
    # =================================================

    raw_vals, pre_coreg_stats = _stable_residual_stats(
        target_dem,
        ref_dem,
        valid_stable,
        cfg.outlier_clip_m,
    )

    print(
        f"\n  Pre-coreg  — "
        f"median={pre_coreg_stats['median']:+.3f} m, "
        f"NMAD={pre_coreg_stats['nmad']:.3f} m, "
        f"STD={pre_coreg_stats['std']:.3f} m"
    )

    # =================================================
    # Run both co-registration methods unconditionally
    # so the violin comparison is always available
    # =================================================

    vs = xdem.coreg.VerticalShift()
    nk = xdem.coreg.NuthKaab()

    vs.fit(
        reference_elev=ref_dem,
        to_be_aligned_elev=target_dem,
        inlier_mask=valid_stable,
    )

    nk.fit(
        reference_elev=ref_dem,
        to_be_aligned_elev=target_dem,
        inlier_mask=valid_stable,
    )

    tgt_vs = vs.apply(target_dem)
    tgt_nk = nk.apply(target_dem)

    res_vs, stats_vs = _stable_residual_stats(
        tgt_vs, ref_dem, valid_stable, cfg.outlier_clip_m
    )

    res_nk, stats_nk = _stable_residual_stats(
        tgt_nk, ref_dem, valid_stable, cfg.outlier_clip_m
    )

    # =================================================
    # Select final output based on config
    # =================================================

    if cfg.coreg_method == "vertical_shift":
        target_coreg = tgt_vs
        stats        = stats_vs
        coreg_name   = "VerticalShift"

    elif cfg.coreg_method == "nuth_kaab":
        target_coreg = tgt_nk
        stats        = stats_nk
        coreg_name   = "NuthKaab"

    else:
        raise ValueError(
            f"Unsupported co-registration method: "
            f"{cfg.coreg_method}"
        )

    print(
        f"\n  Final method: {coreg_name}"
    )

    print(
        f"  Post-coreg — "
        f"median={stats['median']:+.3f} m, "
        f"NMAD={stats['nmad']:.3f} m, "
        f"STD={stats['std']:.3f} m  "
        f"(n={stats['n']:,})"
    )

    # =================================================
    # Comparison dict (feeds violin plot in save_outputs)
    # =================================================

    coreg_comparison = {
        "none": {
            "residuals": raw_vals,
            "stats":     pre_coreg_stats,
        },
        "vertical_shift": {
            "residuals": res_vs,
            "stats":     stats_vs,
        },
        "nuth_kaab": {
            "residuals": res_nk,
            "stats":     stats_nk,
        },
    }

    # =================================================
    # Merge pre- and post-coreg stats for CSV export
    # =================================================

    full_stats = {
        "coreg_method":    coreg_name,
        "pre_coreg_median": pre_coreg_stats["median"],
        "pre_coreg_nmad":   pre_coreg_stats["nmad"],
        "pre_coreg_std":    pre_coreg_stats["std"],
        "pre_coreg_n":      pre_coreg_stats["n"],
        **{f"post_{k}": v for k, v in stats.items()},
    }

    return {
        "reference_dem":    ref_dem,
        "target_coreg":     target_coreg,
        "stable_mask":      stable_mask,
        "valid_mask":       valid_mask,
        "water_mask":       combined_water_mask,
        "stable_residuals": res_vs if coreg_name == "VerticalShift" else res_nk,
        "stable_stats":     full_stats,
        "coreg_method":     coreg_name,
        "coreg_comparison": coreg_comparison,
    }