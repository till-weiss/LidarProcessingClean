import numpy as np
import xdem
import geoutils as gu


def load_stable_mask(stable_ground_path, ref_dem):
    if stable_ground_path is None:
        return np.isfinite(np.array(ref_dem.data))

    stable_vector = gu.Vector(stable_ground_path)
    return np.array(stable_vector.create_mask(ref_dem).data).astype(bool)


def build_valid_mask(ref_arr, target_arr):
    invalid_values = [-9999, -99999, -32768]

    valid_mask = (
        np.isfinite(ref_arr)
        & np.isfinite(target_arr)
    )

    for invalid in invalid_values:
        valid_mask &= (ref_arr != invalid)
        valid_mask &= (target_arr != invalid)

    return valid_mask


def co_register_dem_pair(cfg):
    print("\n[1/3] Co-register DEMs")

    ref_dem = xdem.DEM(cfg.dem_reference_path)

    target_dem = xdem.DEM(
        cfg.dem_target_path
    ).reproject(
        ref_dem,
        resampling="max"
    )

    ref_arr = np.array(ref_dem.data).astype(np.float32)
    target_arr = np.array(target_dem.data).astype(np.float32)

    stable_mask = load_stable_mask(
        cfg.stable_ground_path,
        ref_dem
    )

    valid_mask = build_valid_mask(
        ref_arr,
        target_arr
    )

    valid_stable = valid_mask & stable_mask

    if np.count_nonzero(valid_stable) == 0:
        raise RuntimeError(
            "No valid stable-ground pixels available for co-registration."
        )

    # ------------------------------------------------------------------
    # Co-registration method selection
    # ------------------------------------------------------------------

    coreg_method = getattr(cfg, "coreg_method", "vertical_shift")

    if coreg_method == "vertical_shift":
        coreg = xdem.coreg.VerticalShift()
        coreg_name = "VerticalShift"

    elif coreg_method == "nuth_kaab":
        coreg = xdem.coreg.NuthKaab()
        coreg_name = "NuthKaab"

    else:
        raise ValueError(
            f"Unsupported co-registration method: {coreg_method}"
        )

    print(f"  method: {coreg_name}")

    # ------------------------------------------------------------------
    # Fit co-registration
    # ------------------------------------------------------------------

    coreg.fit(
        reference_elev=ref_dem,
        to_be_aligned_elev=target_dem,
        inlier_mask=valid_stable,
    )

    target_coreg = coreg.apply(target_dem)

    # ------------------------------------------------------------------
    # Residual analysis
    # ------------------------------------------------------------------

    residual = np.array(
        (target_coreg - ref_dem).data
    ).astype(np.float32)

    stable_vals = residual[valid_stable]

    stable_vals = stable_vals[
        np.abs(stable_vals) < cfg.outlier_clip_m
    ]

    if stable_vals.size == 0:
        raise RuntimeError(
            "No stable-ground residual pixels after clipping."
        )

    stats = {
        "coreg_method": coreg_name,
        "median": float(np.nanmedian(stable_vals)),
        "nmad": float(gu.stats.nmad(stable_vals)),
        "std": float(np.nanstd(stable_vals)),
        "rmse": float(
            np.sqrt(np.nanmean(stable_vals ** 2))
        ),
        "n": int(stable_vals.size),
    }

    print(f"  stable pixels: {stats['n']:,}")

    print(
        f"  median={stats['median']:+.3f} m, "
        f"NMAD={stats['nmad']:.3f} m, "
        f"STD={stats['std']:.3f} m"
    )

    return {
        "reference_dem": ref_dem,
        "target_coreg": target_coreg,
        "stable_mask": stable_mask,
        "valid_mask": valid_mask,
        "stable_residuals": stable_vals,
        "stable_stats": stats,
        "coreg_method": coreg_name,
        "coreg_model": coreg,
    }