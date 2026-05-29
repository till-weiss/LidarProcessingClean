"""
Cross-AOI Reliability Figures
===============================
Produces three outputs for thesis results section 4.3:
  1. NMAD vs scene-wide median scatter (key figure)
  2. Multi-panel dDEM distribution violin/boxplot
  3. Cross-AOI reliability summary table (CSV + styled PNG)

Requirements: numpy, pandas, matplotlib, rasterio, scipy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
from pathlib import Path

# =============================================================================
# CONFIGURATION — fill in your AOI names and file paths
# =============================================================================

AOI_CONFIG = [
    {
        "name": "Aklavik",
        "dem_type": "DSM",
        "summary_csv": "/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/results/ChangeDetection/Aklavik_WM/DTM/2023_2025/Aklavik_WM_DTM_2023_2025_summary.csv",
        "ddem_tif": "/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/results/ChangeDetection/Aklavik_WM/DTM/2023_2025/Aklavik_WM_DTM_2023_2025_ddem.tif",
        "icp_applied": False,
    },

    {
        "name": "Fort_McPherson",
        "dem_type": "DTM",
        "summary_csv": "/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/results/ChangeDetection/FortMcPherson_WaterMask/DTM/2023_2025/FortMcPherson_WaterMask_DTM_2023_2025_summary.csv",
        "ddem_tif": "/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/results/ChangeDetection/FortMcPherson_WaterMask/DTM/2023_2025/FortMcPherson_WaterMask_DTM_2023_2025_ddem.tif",
        "icp_applied": False,
    },

    {
        "name": "Tuktoyaktuk",
        "dem_type": "DSM",
        "summary_csv": "/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/results/ChangeDetection/Tuk_WM/DTM/2023_2025/Tuk_WM_DTM_2023_2025_summary.csv",
        "ddem_tif": "/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/results/ChangeDetection/Tuk_WM/DTM/2023_2025/Tuk_WM_DTM_2023_2025_ddem.tif",
        "icp_applied": False,
    },

    #{
    #    "name": "Inuvik",
    #    "dem_type": "DSM",
    #    "summary_csv": "/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/results/ChangeDetection/Inuvik_23_25/summary_statistics.csv",
    #    "ddem_tif": "/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/results/ChangeDetection/Inuvik_23_25/ddem.tif",
    #    "icp_applied": True,
    #},

    #{
    #    "name": "Peel",
    #    "dem_type": "DSM",
    #    "summary_csv": "/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/results/ChangeDetection/Peel_23_25/summary_statistics.csv",
    #    "ddem_tif": "/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/results/ChangeDetection/Peel_23_25/ddem.tif",
    #    "icp_applied": True,
    #},
    # Add remaining AOIs here following the same pattern
]

# Confidence thresholds — adjust to match your noise floor context
SIGNAL_THRESHOLD_FACTOR = 2.0   # scene median must exceed N × stable NMAD
SAMPLE_N = 100_000               # max pixels sampled per AOI for violin plot

OUTPUT_DIR = Path("/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/outputs/plots/master_results")           # where figures are saved

# =============================================================================
# HELPERS
# =============================================================================

def load_summary(csv_path: str) -> dict:
    """Load a summary CSV into a flat dict."""
    df = pd.read_csv(csv_path, header=None, names=["metric", "value"])
    return dict(zip(df["metric"], df["value"]))


def load_ddem_sample(tif_path: str, n: int = SAMPLE_N) -> np.ndarray:
    """
    Load dDEM raster and return a flat array of valid (non-NaN) values.
    Randomly subsamples to n pixels to keep memory manageable.
    """
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(float)
        nodata = src.nodata
    if nodata is not None:
        data[data == nodata] = np.nan
    valid = data[np.isfinite(data)].ravel()
    if len(valid) > n:
        rng = np.random.default_rng(42)
        valid = rng.choice(valid, size=n, replace=False)
    return valid


def assess_confidence(scene_median: float, stable_nmad: float,
                       threshold_factor: float = SIGNAL_THRESHOLD_FACTOR) -> str:
    """Assign confidence label based on whether median exceeds N×NMAD."""
    threshold = threshold_factor * stable_nmad
    if abs(scene_median) >= threshold:
        return "Detectable"
    elif abs(scene_median) >= 0.5 * threshold:
        return "Marginal"
    else:
        return "Below threshold"


# =============================================================================
# LOAD DATA
# =============================================================================

records = []
ddem_samples = {}

for aoi in AOI_CONFIG:
    s = load_summary(aoi["summary_csv"])
    scene_median = float(s.get("aoi_median") or s.get("all_median"))
    stable_nmad  = float(s["stable_nmad"])
    threshold    = SIGNAL_THRESHOLD_FACTOR * stable_nmad
    confidence   = assess_confidence(scene_median, stable_nmad)

    records.append({
        "AOI":               aoi["name"],
        "DEM Type":          aoi["dem_type"],
        "ICP Applied":       aoi["icp_applied"],
        "NMAD pre (m)":      round(float(s.get("none_nmad", np.nan)), 4),
        "NMAD post (m)":     round(stable_nmad, 4),
        "Best Method":       str(s.get("coreg_method", "—")),
        "Scene Median (m)":  round(scene_median, 4),
        "Threshold ±2×NMAD (m)": round(threshold, 4),
        "Signal":            confidence,
    })

    samples = load_ddem_sample(aoi["ddem_tif"])
    # Clip to ±5× NMAD for display
    clip = 5 * stable_nmad
    samples = samples[np.abs(samples) < clip]
    ddem_samples[aoi["name"]] = samples

summary_df = pd.DataFrame(records)

# =============================================================================
# FIGURE 1 — NMAD vs Scene-Wide Median Scatter (key figure)
# =============================================================================

fig1, ax = plt.subplots(figsize=(7, 6))

confidence_colors = {
    "Detectable":      "#2166ac",
    "Marginal":        "#f4a582",
    "Below threshold": "#999999",
}

icp_markers = {True: "^", False: "o", "partial": "D"}

for _, row in summary_df.iterrows():
    color  = confidence_colors[row["Signal"]]
    marker = icp_markers.get(row["ICP Applied"], "o")
    ax.scatter(
        row["NMAD post (m)"],
        abs(row["Scene Median (m)"]),
        c=color, marker=marker, s=100, zorder=5,
        edgecolors="white", linewidths=0.6,
    )
    ax.annotate(
        row["AOI"],
        xy=(row["NMAD post (m)"], abs(row["Scene Median (m)"])),
        xytext=(5, 4), textcoords="offset points", fontsize=9,
    )

# Detection threshold line: y = SIGNAL_THRESHOLD_FACTOR × x
nmad_range = np.linspace(0, summary_df["NMAD post (m)"].max() * 1.3, 100)
ax.plot(
    nmad_range, SIGNAL_THRESHOLD_FACTOR * nmad_range,
    "k--", linewidth=1.2, label=f"Detection threshold ({SIGNAL_THRESHOLD_FACTOR}×NMAD)",
)
ax.fill_between(nmad_range, 0, SIGNAL_THRESHOLD_FACTOR * nmad_range,
                alpha=0.07, color="black")

# Legend — confidence
conf_patches = [
    mpatches.Patch(color=c, label=l)
    for l, c in confidence_colors.items()
]
# Legend — ICP
icp_handles = [
    plt.Line2D([0], [0], marker=m, color="gray", linestyle="None",
               markersize=8, label=f"ICP: {k}")
    for k, m in icp_markers.items()
]
ax.legend(handles=conf_patches + icp_handles, fontsize=8, loc="upper left")

ax.set_xlabel("Stable-ground NMAD — noise floor (m)", fontsize=11)
ax.set_ylabel("|Scene-wide median dDEM| (m)", fontsize=11)
ax.set_title("Cross-AOI signal detectability", fontsize=12)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.grid(True, linestyle=":", alpha=0.5)

fig1.tight_layout()
fig1.savefig(OUTPUT_DIR / "fig_nmad_vs_median_scatter.png", dpi=200)
plt.close(fig1)
print("Saved: fig_nmad_vs_median_scatter.png")

# =============================================================================
# FIGURE 2 — Multi-Panel dDEM Distribution (violin per AOI, shared axes)
# =============================================================================

aoi_names = list(ddem_samples.keys())
n_aois    = len(aoi_names)

fig2, ax2 = plt.subplots(figsize=(max(6, n_aois * 1.4), 5))

positions = np.arange(1, n_aois + 1)
vp = ax2.violinplot(
    [ddem_samples[name] for name in aoi_names],
    positions=positions,
    showmedians=True,
    showextrema=False,
    widths=0.7,
)

# Colour by confidence
for i, (body, name) in enumerate(zip(vp["bodies"], aoi_names)):
    row   = summary_df[summary_df["AOI"] == name].iloc[0]
    color = confidence_colors[row["Signal"]]
    body.set_facecolor(color)
    body.set_alpha(0.75)
    body.set_edgecolor("white")

vp["cmedians"].set_color("black")
vp["cmedians"].set_linewidth(1.5)

# Zero line and ±2×NMAD bands per AOI
ax2.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=2)
for i, name in enumerate(aoi_names):
    row    = summary_df[summary_df["AOI"] == name].iloc[0]
    thresh = row["Threshold ±2×NMAD (m)"]
    ax2.bar(
        positions[i], 2 * thresh, bottom=-thresh,
        color="black", alpha=0.08, width=0.65, zorder=1,
    )

ax2.set_xticks(positions)
ax2.set_xticklabels(aoi_names, rotation=20, ha="right", fontsize=10)
ax2.set_ylabel("dDTM elevation change (m)", fontsize=11)
ax2.set_title("dDEM distributions — DTM, all AOIs", fontsize=12)
ax2.grid(axis="y", linestyle=":", alpha=0.4)

# Legend
conf_patches2 = [
    mpatches.Patch(color=c, alpha=0.75, label=l)
    for l, c in confidence_colors.items()
]
ax2.legend(handles=conf_patches2, fontsize=8, loc="upper right")

fig2.tight_layout()
fig2.savefig(OUTPUT_DIR / "fig_ddem_distributions_violin.png", dpi=200)
plt.close(fig2)
print("Saved: fig_ddem_distributions_violin.png")

# =============================================================================
# FIGURE 3 — Summary Table as PNG
# =============================================================================

display_cols = [
    "AOI", "DEM Type", "NMAD pre (m)", "NMAD post (m)",
    "Best Method", "Scene Median (m)", "Threshold ±2×NMAD (m)", "Signal",
]
table_df = summary_df[display_cols].copy()

fig3, ax3 = plt.subplots(figsize=(13, 0.5 + 0.4 * len(table_df)))
ax3.axis("off")

tbl = ax3.table(
    cellText=table_df.values,
    colLabels=table_df.columns,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.auto_set_column_width(col=list(range(len(table_df.columns))))

# Colour rows by confidence
for i, row in enumerate(table_df.itertuples(index=False)):
    color = confidence_colors.get(row.Signal, "white")
    for j in range(len(display_cols)):
        tbl[i + 1, j].set_facecolor(color)
        tbl[i + 1, j].set_alpha(0.25)

# Header style
for j in range(len(display_cols)):
    tbl[0, j].set_facecolor("#2c2c2c")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

fig3.tight_layout()
fig3.savefig(OUTPUT_DIR / "fig_reliability_table.png", dpi=200, bbox_inches="tight")
plt.close(fig3)
print("Saved: fig_reliability_table.png")

# Also save as CSV for your records
summary_df.to_csv(OUTPUT_DIR / "cross_aoi_reliability_summary.csv", index=False)
print("Saved: cross_aoi_reliability_summary.csv")