import os
import re
import laspy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib import cm, colors

# ============================================================
# SETTINGS
# ============================================================

BASE_2023 = "/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/preprocessed/Inuvik_2023_3/WC_Inuvik_20230705_15cm_01/strips"
BASE_2025 = "/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/preprocessed/Inuvik_2025_3/WC_Inuvik_20230705_15cm_01/strips"

OUTDIR = "/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/results/icp_maps"
os.makedirs(OUTDIR, exist_ok=True)

# ============================================================
# INPUT DATA
# ============================================================

# ---------- INUVIK 2023 ----------
links_2023 = [
    {"source": "154923_155343", "target": "161948_162507", "fitness": 0.2950, "rmse": 0.5259, "shift": 0.7381, "status": "Weak fit"},
    {"source": "160053_160326", "target": "161948_162507", "fitness": 0.2826, "rmse": 0.5278, "shift": 0.7891, "status": "Weak fit"},
    {"source": "160549_161112", "target": "161948_162507", "fitness": 0.6425, "rmse": 0.4802, "shift": 0.3337, "status": "Good"},
    {"source": "161339_161649", "target": "161948_162507", "fitness": 0.6873, "rmse": 0.4638, "shift": 0.1142, "status": "Good"},
    {"source": "162757_163111", "target": "161339_161649", "fitness": 0.2370, "rmse": 0.6096, "shift": 1.0562, "status": "Poor"},
    {"source": "154923_155343", "target": "160549_161112", "fitness": 0.6767, "rmse": 0.4307, "shift": 0.7439, "status": "Good"},
    {"source": "160053_160326", "target": "161339_161649", "fitness": 0.5882, "rmse": 0.5421, "shift": 0.6589, "status": "Moderate"},
    {"source": "162757_163111", "target": "160053_160326", "fitness": 0.6726, "rmse": 0.4642, "shift": 0.3660, "status": "Good"},
]

clusters_2023 = {
    "161948_162507": 1,
    "160549_161112": 1,
    "161339_161649": 1,
    "154923_155343": 1,
    "160053_160326": 2,
    "162757_163111": 2,
}

# ---------- INUVIK 2025 ----------
links_2025 = [
    {"source": "214223_214546", "target": "212043_212341", "fitness": 0.6258, "rmse": 0.5516, "shift": 0.5241, "status": "Good"},
    {"source": "214845_215320", "target": "212043_212341", "fitness": 0.6823, "rmse": 0.5560, "shift": 0.8666, "status": "Good"},
    {"source": "220134_220606", "target": "214223_214546", "fitness": 0.6432, "rmse": 0.5124, "shift": 0.6601, "status": "Good"},
    {"source": "220905_221303", "target": "214845_215320", "fitness": 0.6623, "rmse": 0.5260, "shift": 0.7368, "status": "Good"},
    {"source": "221613_222031", "target": "220134_220606", "fitness": 0.8697, "rmse": 0.5408, "shift": 1.0904, "status": "Very good fitness, but relatively large shift"},
    {"source": "222242_222711", "target": "220905_221303", "fitness": 0.5389, "rmse": 0.5714, "shift": 0.9960, "status": "Moderate"},
    {"source": "190327_190710", "target": "220134_220606", "fitness": 0.3295, "rmse": 0.5217, "shift": 0.8673, "status": "Weak"},
]

clusters_2025 = {
    "212043_212341": 1,
    "214223_214546": 1,
    "214845_215320": 1,
    "220134_220606": 1,
    "220905_221303": 1,
    "221613_222031": 1,
    "222242_222711": 2,   # adjust if you want another cluster ID
    "190327_190710": 3,
}

# ============================================================
# HELPERS
# ============================================================

def extract_time_id(filename: str):
    """
    Extract short strip ID like 154923_155343 or 20250809T190327_190710.
    """
    name = os.path.basename(filename)

    m_full = re.search(r"(\d{8}T\d{6}_\d{6})", name)
    if m_full:
        return m_full.group(1)

    m_short = re.search(r"(\d{6}_\d{6})", name)
    if m_short:
        return m_short.group(1)

    return None


def find_strip_files(base_dir):
    """
    Return dict: short_id -> full_path
    Matches both short (161948_162507) and long (20230705T161948_162507)
    """
    out = {}

    for fn in os.listdir(base_dir):
        if not fn.lower().endswith((".laz", ".las")):
            continue

        full_path = os.path.join(base_dir, fn)

        # extract both versions
        m_full = re.search(r"(\d{8}T\d{6}_\d{6})", fn)
        m_short = re.search(r"(\d{6}_\d{6})", fn)

        if m_full:
            out[m_full.group(1)] = full_path

        if m_short:
            out[m_short.group(1)] = full_path

    return out


def get_bbox_and_center(laz_path):
    with laspy.open(laz_path) as f:
        h = f.header
        xmin, ymin, zmin = h.mins
        xmax, ymax, zmax = h.maxs
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    return {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "cx": cx,
        "cy": cy,
        "width": xmax - xmin,
        "height": ymax - ymin,
    }


def build_strip_geoms(base_dir, cluster_map):
    files = find_strip_files(base_dir)
    geoms = {}
    missing = []

    for sid in cluster_map.keys():
        if sid not in files:
            missing.append(sid)
            continue
        geoms[sid] = get_bbox_and_center(files[sid])

    if missing:
        print("Missing strip files:", missing)

    return geoms


def plot_icp_map(title, geoms, links, clusters, output_path):
    fig, ax = plt.subplots(figsize=(11, 10))

    # Cluster colours
    cluster_palette = {
        1: "#4C78A8",
        2: "#F58518",
        3: "#54A24B",
        4: "#E45756",
        5: "#72B7B2",
    }

    # Arrow colour by shift norm
    shifts = [d["shift"] for d in links]
    norm = colors.Normalize(vmin=min(shifts), vmax=max(shifts))
    cmap = cm.viridis

    # Plot footprints
    for sid, g in geoms.items():
        cid = clusters.get(sid, 0)
        facecolor = cluster_palette.get(cid, "#BBBBBB")
        rect = Rectangle(
            (g["xmin"], g["ymin"]),
            g["width"],
            g["height"],
            facecolor=facecolor,
            edgecolor="black",
            linewidth=1.2,
            alpha=0.28,
        )
        ax.add_patch(rect)

        ax.text(
            g["cx"], g["cy"], sid,
            fontsize=8, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
        )

    # Plot arrows
    for row in links:
        s = row["source"]
        t = row["target"]

        if s not in geoms or t not in geoms:
            continue

        gs = geoms[s]
        gt = geoms[t]

        arrow = FancyArrowPatch(
            (gs["cx"], gs["cy"]),
            (gt["cx"], gt["cy"]),
            arrowstyle="->",
            mutation_scale=14,
            linewidth=1.0 + 4.0 * row["fitness"],   # thickness by fitness
            color=cmap(norm(row["shift"])),         # colour by shift norm
            alpha=0.9,
            zorder=5,
        )
        ax.add_patch(arrow)

    # Axis styling
    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")

    # Cluster legend
    used_clusters = sorted(set(clusters.values()))
    cluster_handles = [
        Rectangle((0, 0), 1, 1, facecolor=cluster_palette.get(cid, "#BBBBBB"),
                  edgecolor="black", alpha=0.28, label=f"Cluster {cid}")
        for cid in used_clusters
    ]

    # Fitness linewidth legend
    fit_vals = [0.3, 0.6, 0.85]
    fit_handles = [
        Line2D([0], [0], color="black", linewidth=1.0 + 4.0 * f, label=f"Fitness {f:.2f}")
        for f in fit_vals
    ]

    leg1 = ax.legend(handles=cluster_handles, loc="upper left", title="Clusters")
    ax.add_artist(leg1)
    ax.legend(handles=fit_handles, loc="upper right", title="Arrow width = fitness")

    # Shift colour bar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Shift norm (m)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================
# RUN
# ============================================================

geoms_2023 = build_strip_geoms(BASE_2023, clusters_2023)
plot_icp_map(
    title="Inuvik 2023 – ICP strip alignment map",
    geoms=geoms_2023,
    links=links_2023,
    clusters=clusters_2023,
    output_path=os.path.join(OUTDIR, "inuvik_2023_icp_map.png"),
)

geoms_2025 = build_strip_geoms(BASE_2025, clusters_2025)
plot_icp_map(
    title="Inuvik 2025 – ICP strip alignment map",
    geoms=geoms_2025,
    links=links_2025,
    clusters=clusters_2025,
    output_path=os.path.join(OUTDIR, "inuvik_2025_icp_map.png"),
)