import numpy as np
import xdem
import geoutils as gu


def load_stable_mask(stable_ground_path, ref_dem):
    if stable_ground_path is None:
        return np.isfinite(np.array(ref_dem.data))
    stable_vector = gu.Vector(stable_ground_path)
    return np.array(stable_vector.create_mask(ref_dem).data).astype(bool)


def co_register_dem_pair(cfg):
    print("\n[1/3] Co-register DEMs")
    ref_dem = xdem.DEM(cfg.dem_reference_path)
    target_dem = xdem.DEM(cfg.dem_target_path).reproject(ref_dem)

    stable_mask = load_stable_mask(cfg.stable_ground_path, ref_dem)

    coreg = xdem.coreg.VerticalShift()
    coreg.fit(reference_elev=ref_dem, to_be_aligned_elev=target_dem, inlier_mask=stable_mask)
    target_coreg = coreg.apply(target_dem)

    residual = np.array((target_coreg - ref_dem).data).astype(np.float32)
    stable_vals = residual[stable_mask & np.isfinite(residual)]
    stable_vals = stable_vals[np.abs(stable_vals) < cfg.outlier_clip_m]

    if stable_vals.size == 0:
        raise RuntimeError("No stable-ground residual pixels after clipping.")

    stats = {
        "median": float(np.nanmedian(stable_vals)),
        "nmad": float(gu.stats.nmad(stable_vals)),
        "std": float(np.nanstd(stable_vals)),
        "rmse": float(np.sqrt(np.nanmean(stable_vals ** 2))),
        "n": int(stable_vals.size),
    }

    print(f"  stable pixels: {stats['n']:,}")
    print(f"  median={stats['median']:+.3f} m, NMAD={stats['nmad']:.3f} m, STD={stats['std']:.3f} m")

    return {
        "reference_dem": ref_dem,
        "target_coreg": target_coreg,
        "stable_mask": stable_mask,
        "stable_residuals": stable_vals,
        "stable_stats": stats,
    }
