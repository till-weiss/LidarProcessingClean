import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import xdem
import matplotlib.pyplot as plt
import rasterio
from rasterio.enums import Resampling

AOI_name = "Aklavik_noICP"
DEM = "DTM"
output_dir = f"/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/results/DoD/{AOI_name}"

dem_2023 = xdem.DEM("/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/results/Aklavik_2023/DTM/Aklavik_DTM.tif")
dem_2025 = xdem.DEM("/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/results/Aklavik_2025/DTM/Aklavik_DTM.tif")


def calc_ddem_stats(ddem):
    arr = ddem.data
    nodata = getattr(ddem, "nodata", None)

    if np.ma.isMaskedArray(arr):
        vals = arr.compressed().astype(float)
    else:
        vals = np.asarray(arr, dtype=float).ravel()

    vals = vals[np.isfinite(vals)]

    if nodata is not None and np.isfinite(nodata):
        vals = vals[vals != float(nodata)]

    if vals.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "nmad": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "p95_abs": np.nan,
        }

    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "nmad": float(xdem.spatialstats.nmad(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "p95_abs": float(np.percentile(np.abs(vals), 95)),
    }


def calc_stats_from_mask(ddem, mask):
    arr = ddem.data
    nodata = getattr(ddem, "nodata", None)

    if np.ma.isMaskedArray(arr):
        data = arr.filled(np.nan)
    else:
        data = np.asarray(arr, dtype=float)

    vals = data[mask]
    vals = vals[np.isfinite(vals)]

    if nodata is not None and np.isfinite(nodata):
        vals = vals[~np.isclose(vals, float(nodata))]

    if vals.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "nmad": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "p95_abs": np.nan,
        }

    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "nmad": float(xdem.spatialstats.nmad(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "p95_abs": float(np.percentile(np.abs(vals), 95)),
    }


def get_valid_raster_values(raster):
    arr = np.asarray(raster.data, dtype=float).ravel()
    nodata = getattr(raster, "nodata", None)

    vals = arr[np.isfinite(arr)]

    if nodata is not None and np.isfinite(nodata):
        vals = vals[~np.isclose(vals, float(nodata))]

    return vals


def save_mask_as_dem(mask, ref_dem, out_fp):
    mask_dem = ref_dem.copy()
    mask_dem.data = mask.astype(np.uint8)
    mask_dem.to_file(out_fp)


def build_stable_terrain_mask(ref, target_on_ref, ddem,
                              slope_min=3, slope_max=20,
                              max_abs_ddem=0.5):
    slope = xdem.terrain.slope(ref)

    ref_arr = np.asarray(ref.data, dtype=float)
    tgt_arr = np.asarray(target_on_ref.data, dtype=float)
    slope_arr = np.asarray(slope.data, dtype=float)

    if np.ma.isMaskedArray(ddem.data):
        ddem_arr = ddem.data.filled(np.nan)
    else:
        ddem_arr = np.asarray(ddem.data, dtype=float)

    ref_valid = np.isfinite(ref_arr)
    tgt_valid = np.isfinite(tgt_arr)
    slope_valid = np.isfinite(slope_arr)
    ddem_valid = np.isfinite(ddem_arr)

    if np.ma.isMaskedArray(ref.data):
        ref_valid &= ~ref.data.mask
    if np.ma.isMaskedArray(target_on_ref.data):
        tgt_valid &= ~target_on_ref.data.mask
    if np.ma.isMaskedArray(slope.data):
        slope_valid &= ~slope.data.mask
    if np.ma.isMaskedArray(ddem.data):
        ddem_valid &= ~ddem.data.mask

    stable_mask = (
        ref_valid &
        tgt_valid &
        slope_valid &
        ddem_valid &
        (slope_arr >= slope_min) &
        (slope_arr <= slope_max) &
        (np.abs(ddem_arr) <= max_abs_ddem)
    )

    return stable_mask


def save_ddem_png(ddem, out_fp, title="dDEM", vmin=-2, vmax=2, cmap="RdBu"):
    arr = ddem.data

    if np.ma.isMaskedArray(arr):
        plot_arr = arr.filled(np.nan)
    else:
        plot_arr = np.asarray(arr, dtype=float)

    plt.figure(figsize=(8, 6), dpi=300)
    im = plt.imshow(plot_arr, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="Elevation change (m)")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_fp, dpi=200, bbox_inches="tight")
    plt.close()


def save_downscaled_dem(dem, out_fp, scale_factor=4, resampling=Resampling.average):
    arr = dem.data

    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)
    else:
        arr = np.asarray(arr, dtype=float)

    if arr.ndim == 3:
        arr = arr[0]

    height, width = arr.shape
    new_height = max(1, height // scale_factor)
    new_width = max(1, width // scale_factor)

    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            crs=dem.crs,
            transform=dem.transform,
            nodata=np.nan
        ) as ds:
            ds.write(arr.astype("float32"), 1)

            downscaled = ds.read(
                1,
                out_shape=(new_height, new_width),
                resampling=resampling
            )

    old_transform = dem.transform
    new_transform = rasterio.Affine(
        old_transform.a * scale_factor,
        old_transform.b,
        old_transform.c,
        old_transform.d,
        old_transform.e * scale_factor,
        old_transform.f
    )

    os.makedirs(os.path.dirname(out_fp), exist_ok=True)

    with rasterio.open(
        out_fp,
        "w",
        driver="GTiff",
        height=new_height,
        width=new_width,
        count=1,
        dtype="float32",
        crs=dem.crs,
        transform=new_transform,
        nodata=np.nan,
        compress="LZW"
    ) as dst:
        dst.write(downscaled.astype("float32"), 1)

    print(f"Saved downscaled DEM: {out_fp}")


def save_downscaled_hillshade(dem, out_fp, scale_factor=4, azimuth=315, altitude=45):
    arr = dem.data

    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)
    else:
        arr = np.asarray(arr, dtype=float)

    if arr.ndim == 3:
        arr = arr[0]

    height, width = arr.shape
    new_height = max(1, height // scale_factor)
    new_width = max(1, width // scale_factor)

    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            crs=dem.crs,
            transform=dem.transform,
            nodata=np.nan
        ) as ds:
            ds.write(arr.astype("float32"), 1)

            downscaled = ds.read(
                1,
                out_shape=(new_height, new_width),
                resampling=Resampling.average
            )

    old_transform = dem.transform
    new_transform = rasterio.Affine(
        old_transform.a * scale_factor,
        old_transform.b,
        old_transform.c,
        old_transform.d,
        old_transform.e * scale_factor,
        old_transform.f
    )

    xres = abs(new_transform.a)
    yres = abs(new_transform.e)

    grad_y, grad_x = np.gradient(downscaled, yres, xres)

    slope = np.pi / 2.0 - np.arctan(np.sqrt(grad_x**2 + grad_y**2))
    aspect = np.arctan2(-grad_x, grad_y)

    az_rad = np.deg2rad(azimuth)
    alt_rad = np.deg2rad(altitude)

    hillshade = (
        np.sin(alt_rad) * np.sin(slope) +
        np.cos(alt_rad) * np.cos(slope) * np.cos(az_rad - aspect)
    )

    hillshade = np.clip(hillshade, 0, 1)
    hillshade = (hillshade * 255).astype("uint8")

    os.makedirs(os.path.dirname(out_fp), exist_ok=True)

    with rasterio.open(
        out_fp,
        "w",
        driver="GTiff",
        height=new_height,
        width=new_width,
        count=1,
        dtype="uint8",
        crs=dem.crs,
        transform=new_transform,
        compress="LZW"
    ) as dst:
        dst.write(hillshade, 1)

    print(f"Saved downscaled hillshade: {out_fp}")


def save_summary_log(summary_df, aoi_name, dem_type, output_dir,
                     slope_min, slope_max, nmad_factor, min_inliers):

    os.makedirs(output_dir, exist_ok=True)
    log_fp = os.path.join(output_dir, f"{aoi_name}_d{dem_type}_coreg_log.json")

    summary_records = summary_df.replace({np.nan: None}).to_dict(orient="records")

    run_entry = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "AOI": aoi_name,
        "DEM_type": dem_type,
        "parameters": {
            "slope_min": slope_min,
            "slope_max": slope_max,
            "nmad_factor": nmad_factor,
            "min_inliers": min_inliers
        },
        "results": summary_records
    }

    if os.path.exists(log_fp):
        with open(log_fp, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(run_entry)

    with open(log_fp, "w") as f:
        json.dump(data, f, indent=2)

    return log_fp


def simple_dod(ref, target, aoi_name, dem_type, output_dir):
    target_on_ref = target.reproject(ref)
    ddem = target_on_ref - ref

    os.makedirs(output_dir, exist_ok=True)
    ddem_fp = os.path.join(output_dir, f"{aoi_name}_d{dem_type}_simple.tif")
    ddem_png_fp = os.path.join(output_dir, f"{aoi_name}_d{dem_type}_simple.png")

    ddem.to_file(ddem_fp)
    save_ddem_png(ddem, ddem_png_fp, title="Simple dDEM")

    return {
        "target_coreg": target_on_ref,
        "ddem": ddem,
        "shift_value": None,
        "inlier_mask": None,
        "files": {
            "ddem": ddem_fp,
            "ddem_png": ddem_png_fp,
        },
        "status": "success",
    }


def correct_median(ref, target, aoi_name, dem_type, output_dir):
    target_on_ref = target.reproject(ref)
    diff_before = target_on_ref - ref

    vals = get_valid_raster_values(diff_before)

    os.makedirs(output_dir, exist_ok=True)

    if vals.size == 0:
        ddem = diff_before
        ddem_fp = os.path.join(output_dir, f"{aoi_name}_d{dem_type}_median_failed.tif")
        ddem_png_fp = os.path.join(output_dir, f"{aoi_name}_d{dem_type}_median_failed.png")

        ddem.to_file(ddem_fp)
        save_ddem_png(ddem, ddem_png_fp, title="Median dDEM failed")

        return {
            "target_coreg": target_on_ref,
            "ddem": ddem,
            "shift_value": None,
            "files": {
                "ddem": ddem_fp,
                "ddem_png": ddem_png_fp,
            },
            "status": "failed_no_valid_pixels",
        }

    median_diff = float(np.median(vals))

    target_corr = target_on_ref - median_diff
    ddem = target_corr - ref

    target_fp = os.path.join(output_dir, f"{aoi_name}_{dem_type}_coreg_median.tif")
    ddem_fp = os.path.join(output_dir, f"{aoi_name}_d{dem_type}_median.tif")
    ddem_png_fp = os.path.join(output_dir, f"{aoi_name}_d{dem_type}_median.png")

    target_corr.to_file(target_fp)
    ddem.to_file(ddem_fp)
    save_ddem_png(ddem, ddem_png_fp, title="Median-corrected dDEM")

    return {
        "target_coreg": target_corr,
        "ddem": ddem,
        "shift_value": median_diff,
        "files": {
            "target_coreg": target_fp,
            "ddem": ddem_fp,
            "ddem_png": ddem_png_fp,
        },
        "status": "success",
    }


def coregister_kaab(ref, target, aoi_name, dem_type, output_dir,
                    slope_min=3, slope_max=40, nmad_factor=3, min_inliers=1000):
    target_on_ref = target.reproject(ref)
    diff_before = target_on_ref - ref

    slope = xdem.terrain.slope(ref)

    slope_arr = np.asarray(slope.data, dtype=float)
    ref_arr = np.asarray(ref.data, dtype=float)
    target_arr = np.asarray(target_on_ref.data, dtype=float)
    diff_arr = np.asarray(diff_before.data, dtype=float)

    ref_valid = np.isfinite(ref_arr)
    target_valid = np.isfinite(target_arr)
    slope_valid = np.isfinite(slope_arr)
    diff_valid = np.isfinite(diff_arr)

    if np.ma.isMaskedArray(ref.data):
        ref_valid &= ~ref.data.mask
    if np.ma.isMaskedArray(target_on_ref.data):
        target_valid &= ~target_on_ref.data.mask
    if np.ma.isMaskedArray(slope.data):
        slope_valid &= ~slope.data.mask
    if np.ma.isMaskedArray(diff_before.data):
        diff_valid &= ~diff_before.data.mask

    terrain_mask = (
        (slope_arr > slope_min) &
        (slope_arr < slope_max) &
        slope_valid &
        ref_valid &
        target_valid &
        diff_valid
    )

    os.makedirs(output_dir, exist_ok=True)

    terrain_mask_fp = os.path.join(output_dir, f"{aoi_name}_{dem_type}_terrain_mask.tif")
    save_mask_as_dem(terrain_mask, ref, terrain_mask_fp)

    valid_vals = diff_arr[terrain_mask]
    valid_vals = valid_vals[np.isfinite(valid_vals)]

    if valid_vals.size == 0:
        ddem = diff_before
        ddem_fp = os.path.join(output_dir, f"{aoi_name}_d{dem_type}_kaab_failed.tif")
        ddem_png_fp = os.path.join(output_dir, f"{aoi_name}_d{dem_type}_kaab_failed.png")

        ddem.to_file(ddem_fp)
        save_ddem_png(ddem, ddem_png_fp, title="Kaab dDEM failed")

        return {
            "target_coreg": target_on_ref,
            "ddem": ddem,
            "shift_value": None,
            "inlier_mask": terrain_mask,
            "files": {
                "ddem": ddem_fp,
                "ddem_png": ddem_png_fp,
                "terrain_mask": terrain_mask_fp,
            },
            "status": "failed_no_valid_terrain_pixels",
            "n_inliers": 0,
        }

    med = np.nanmedian(valid_vals)
    nmad = xdem.spatialstats.nmad(valid_vals)

    residual_mask = np.abs(diff_arr - med) < (nmad_factor * nmad)
    residual_mask &= diff_valid

    inlier_mask = terrain_mask & residual_mask

    inlier_mask_fp = os.path.join(output_dir, f"{aoi_name}_{dem_type}_inlier_mask_kaab.tif")
    save_mask_as_dem(inlier_mask, ref, inlier_mask_fp)

    n_inliers = int(np.count_nonzero(inlier_mask))
    n_valid = int(np.count_nonzero(ref_valid & target_valid))
    inlier_fraction = n_inliers / n_valid if n_valid > 0 else np.nan

    if n_inliers < min_inliers:
        ddem = diff_before
        ddem_fp = os.path.join(output_dir, f"{aoi_name}_d{dem_type}_kaab_failed.tif")
        ddem_png_fp = os.path.join(output_dir, f"{aoi_name}_d{dem_type}_kaab_failed.png")

        ddem.to_file(ddem_fp)
        save_ddem_png(ddem, ddem_png_fp, title="Kaab dDEM failed")

        return {
            "target_coreg": target_on_ref,
            "ddem": ddem,
            "shift_value": None,
            "inlier_mask": inlier_mask,
            "files": {
                "ddem": ddem_fp,
                "ddem_png": ddem_png_fp,
                "terrain_mask": terrain_mask_fp,
                "inlier_mask": inlier_mask_fp,
            },
            "status": "failed_too_few_inliers",
            "n_inliers": n_inliers,
            "inlier_fraction": float(inlier_fraction),
            "median_before": float(med),
            "nmad_before": float(nmad),
        }

    coreg = xdem.coreg.NuthKaab()
    #coreg = xdem.coreg.ICP() + xdem.coreg.NuthKaab()
    coreg.fit(ref, target_on_ref, inlier_mask=inlier_mask)

    target_coreg = coreg.apply(target_on_ref)
    ddem = target_coreg - ref

    target_fp = os.path.join(output_dir, f"{aoi_name}_{dem_type}_2025to2023_coreg_kaab.tif")
    ddem_fp = os.path.join(output_dir, f"{aoi_name}_d{dem_type}_kaab.tif")
    ddem_png_fp = os.path.join(output_dir, f"{aoi_name}_d{dem_type}_kaab.png")

    target_coreg.to_file(target_fp)
    ddem.to_file(ddem_fp)
    save_ddem_png(ddem, ddem_png_fp, title="Kaab dDEM")

    return {
        "target_coreg": target_coreg,
        "ddem": ddem,
        "shift_value": None,
        "inlier_mask": inlier_mask,
        "files": {
            "target_coreg": target_fp,
            "ddem": ddem_fp,
            "ddem_png": ddem_png_fp,
            "terrain_mask": terrain_mask_fp,
            "inlier_mask": inlier_mask_fp,
        },
        "status": "success",
        "n_inliers": n_inliers,
        "inlier_fraction": float(inlier_fraction),
        "median_before": float(med),
        "nmad_before": float(nmad),
    }


def run_dod_comparison(ref, target, aoi_name, dem_type, output_dir,
                       slope_min=3, slope_max=40, nmad_factor=3, min_inliers=1000):

    results = {}

    results["simple"] = simple_dod(ref, target, aoi_name, dem_type, output_dir)
    results["median"] = correct_median(ref, target, aoi_name, dem_type, output_dir)
    results["kaab"] = coregister_kaab(
        ref, target, aoi_name, dem_type, output_dir,
        slope_min=slope_min,
        slope_max=slope_max,
        nmad_factor=nmad_factor,
        min_inliers=min_inliers
    )

    rows = []

    for method_name, result in results.items():
        stats = calc_ddem_stats(result["ddem"])

        row = {
            "method": method_name,
            "shift_value": result.get("shift_value"),
            "n_inliers": result.get("n_inliers"),
            "inlier_fraction": result.get("inlier_fraction"),
            "status": result.get("status", "success"),
            **stats,
        }
        rows.append(row)

    summary_df = pd.DataFrame(rows)

    print(summary_df.round(4).to_string(index=False))

    os.makedirs(output_dir, exist_ok=True)

    log_fp = save_summary_log(
        summary_df,
        aoi_name,
        dem_type,
        output_dir,
        slope_min,
        slope_max,
        nmad_factor,
        min_inliers
    )

    print(f"Updated JSON log: {log_fp}")

    return results, summary_df


def save_stable_terrain_summary(results, ref, target, aoi_name, dem_type, output_dir,
                                slope_min=3, slope_max=20, max_abs_ddem=0.5):
    rows = []

    target_on_ref = target.reproject(ref)

    for method_name, result in results.items():
        ddem = result["ddem"]

        stable_mask = build_stable_terrain_mask(
            ref=ref,
            target_on_ref=target_on_ref,
            ddem=ddem,
            slope_min=slope_min,
            slope_max=slope_max,
            max_abs_ddem=max_abs_ddem
        )

        mask_fp = os.path.join(
            output_dir,
            f"{aoi_name}_{dem_type}_{method_name}_stable_terrain_mask.tif"
        )
        save_mask_as_dem(stable_mask, ref, mask_fp)

        whole_stats = calc_ddem_stats(ddem)
        stable_stats = calc_stats_from_mask(ddem, stable_mask)

        rows.append({
            "method": method_name,
            "area": "whole_aoi",
            **whole_stats
        })

        rows.append({
            "method": method_name,
            "area": "stable_terrain",
            **stable_stats
        })

    stable_df = pd.DataFrame(rows)

    csv_fp = os.path.join(output_dir, f"{aoi_name}_d{dem_type}_stable_terrain_comparison.csv")
    json_fp = os.path.join(output_dir, f"{aoi_name}_d{dem_type}_stable_terrain_comparison.json")

    stable_df.to_csv(csv_fp, index=False)

    with open(json_fp, "w") as f:
        json.dump(
            {
                "timestamp_utc": datetime.utcnow().isoformat(),
                "AOI": aoi_name,
                "DEM_type": dem_type,
                "stable_terrain_parameters": {
                    "slope_min": slope_min,
                    "slope_max": slope_max,
                    "max_abs_ddem": max_abs_ddem
                },
                "results": stable_df.replace({np.nan: None}).to_dict(orient="records")
            },
            f,
            indent=2
        )

    print("\nStable terrain comparison:")
    print(stable_df.round(4).to_string(index=False))
    print(f"\nSaved stable terrain CSV: {csv_fp}")
    print(f"Saved stable terrain JSON: {json_fp}")

    return stable_df


os.makedirs(output_dir, exist_ok=True)

scale_factor = 4

dem_2023_downscaled_fp = os.path.join(output_dir, f"{AOI_name}_2023_{DEM}_downscaled.tif")
dem_2025_downscaled_fp = os.path.join(output_dir, f"{AOI_name}_2025_{DEM}_downscaled.tif")
hs_2023_fp = os.path.join(output_dir, f"{AOI_name}_2023_{DEM}_hillshade_downscaled.tif")
hs_2025_fp = os.path.join(output_dir, f"{AOI_name}_2025_{DEM}_hillshade_downscaled.tif")

save_downscaled_dem(dem_2023, dem_2023_downscaled_fp, scale_factor=scale_factor)
save_downscaled_dem(dem_2025, dem_2025_downscaled_fp, scale_factor=scale_factor)
save_downscaled_hillshade(dem_2023, hs_2023_fp, scale_factor=scale_factor)
save_downscaled_hillshade(dem_2025, hs_2025_fp, scale_factor=scale_factor)

results, summary_df = run_dod_comparison(
    ref=dem_2023,
    target=dem_2025,
    aoi_name=AOI_name,
    dem_type=DEM,
    output_dir=output_dir,
    slope_min=3,
    slope_max=40,
    nmad_factor=3,
    min_inliers=1000
)

stable_df = save_stable_terrain_summary(
    results=results,
    ref=dem_2023,
    target=dem_2025,
    aoi_name=AOI_name,
    dem_type=DEM,
    output_dir=output_dir,
    slope_min=3,
    slope_max=20,
    max_abs_ddem=0.5
)