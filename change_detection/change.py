import numpy as np
import geoutils as gu


def compute_change_products(
    cfg,
    coreg_data,
):

    print("\n[2/3] Compute dDEM / dDSM")

    # =================================================
    # Inputs
    # =================================================

    ref_dem = coreg_data[
        "reference_dem"
    ]

    target_coreg = coreg_data[
        "target_coreg"
    ]

    stable_mask = coreg_data[
        "stable_mask"
    ]

    water_mask = coreg_data[
        "water_mask"
    ]

    non_water_mask = (
        ~water_mask
    )

    # =================================================
    # Create dDEM
    # =================================================

    ddem = (
        target_coreg
        - ref_dem
    )

    ddem_arr = np.array(
        ddem.data
    ).astype(np.float32)

    # =================================================
    # Remove water from dDEM
    # =================================================

    ddem_arr[
        water_mask
    ] = np.nan

    # =================================================
    # Write masked array back
    # =================================================

    ddem.data = ddem_arr

    # =================================================
    # Valid pixels
    # =================================================

    valid = np.isfinite(
        ddem_arr
    )

    # =================================================
    # Analysis masks
    # =================================================

    analysis_mask = (
        valid
        & non_water_mask
    )

    stable_analysis_mask = (
        stable_mask
        & valid
        & non_water_mask
    )

    # =================================================
    # Extract values
    # =================================================

    all_vals = ddem_arr[
        analysis_mask
    ]

    stable_vals = ddem_arr[
        stable_analysis_mask
    ]

    # =================================================
    # Outlier clipping
    # =================================================

    stable_vals = stable_vals[
        np.abs(stable_vals)
        < cfg.outlier_clip_m
    ]

    # =================================================
    # Threshold estimation
    # =================================================

    if cfg.change_threshold_m is None:

        threshold_m = (
            2.0
            * float(
                gu.stats.nmad(
                    stable_vals
                )
            )
        )

    else:

        threshold_m = (
            cfg.change_threshold_m
        )

    # =================================================
    # Statistics
    # =================================================

    change_stats = {

        "aoi_median":
            float(
                np.nanmedian(
                    all_vals
                )
            ),

        "aoi_nmad":
            float(
                gu.stats.nmad(
                    all_vals
                )
            ),

        "aoi_std":
            float(
                np.nanstd(
                    all_vals
                )
            ),

        "aoi_rmse":
            float(
                np.sqrt(
                    np.nanmean(
                        all_vals ** 2
                    )
                )
            ),

        "n_all":
            int(
                all_vals.size
            ),

        "stable_median":
            float(
                np.nanmedian(
                    stable_vals
                )
            ),

        "stable_nmad":
            float(
                gu.stats.nmad(
                    stable_vals
                )
            ),

        "stable_std":
            float(
                np.nanstd(
                    stable_vals
                )
            ),

        "stable_rmse":
            float(
                np.sqrt(
                    np.nanmean(
                        stable_vals ** 2
                    )
                )
            ),

        "n_stable":
            int(
                stable_vals.size
            ),

        "threshold_m":
            float(
                threshold_m
            ),
    }

    # =================================================
    # Pixel area
    # =================================================

    px = abs(
        ref_dem.res[0]
        * ref_dem.res[1]
    )

    # =================================================
    # Significant change
    # =================================================

    subsidence = all_vals[
        all_vals < -threshold_m
    ]

    heave = all_vals[
        all_vals > threshold_m
    ]

    # =================================================
    # Volume calculations
    # =================================================

    change_stats[
        "volume_loss_m3"
    ] = float(
        np.sum(
            np.abs(
                subsidence
            )
        ) * px
    )

    change_stats[
        "volume_gain_m3"
    ] = float(
        np.sum(
            heave
        )
        * px
    )

    # =================================================
    # Terrain slope
    # =================================================

    slope = ref_dem.slope()

    slope_arr = np.array(
        slope.data
    ).astype(np.float32)

    print(
        f"  d{cfg.dem_type} "
        f"threshold: "
        f"±{threshold_m:.3f} m"
    )

    print(
        f"  analysed pixels "
        f"(non-water): "
        f"{all_vals.size:,}"
    )

    # =================================================
    # Return outputs
    # =================================================

    return {

        "ddem":
            ddem,

        "ddem_arr":
            ddem_arr,

        "change_stats":
            change_stats,

        "slope":
            slope_arr,
    }