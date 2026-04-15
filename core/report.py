"""
report.py
─────────
Generates a self-contained HTML processing report from a Configuration object.

Usage
-----
    from report import write_report
    write_report(config)           # writes to config.results_dir / config.run_name / reports /
    write_report(config, "/path")  # writes to an explicit directory
"""

import os
import datetime


# ── helpers ───────────────────────────────────────────────────────────────────

def _bool(val):
    cls = "bool-t" if val else "bool-f"
    text = "True" if val else "False"
    return f'<span class="{cls}"><span class="bool-dot"></span>{text}</span>'


def _val(v, unit=""):
    unit_html = f' <span class="v-unit">{unit}</span>' if unit else ""
    return f'<span class="v">{v}{unit_html}</span>'


def _path(v):
    return f'<span class="path">{v}</span>'


def _row(key, value, desc=""):
    desc_html = f'<div class="kd">{desc}</div>' if desc else ""
    return f"""
        <tr class="param-row">
          <td class="k">{key}{desc_html}</td>
          <td class="v-cell">{value}</td>
        </tr>"""


def _group(label):
    return f"""
        <tr class="tbl-group"><td colspan="2">{label}</td></tr>"""


def _section(idx, title, count, rows_html):
    return f"""
    <div class="section" id="s{idx:02d}">
      <div class="sec-head">
        <span class="si">{idx:02d}</span>
        <span class="st">{title}</span>
        <span class="sc">{count}</span>
      </div>
      <table class="tbl">{rows_html}
      </table>
    </div>"""


# ── HTML skeleton ─────────────────────────────────────────────────────────────

_CSS = """
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:    #ffffff; --bg-row: #fafafa; --line: #ebebeb; --line-md: #d4d4d4;
  --text:  #0a0a0a; --mid:    #525252; --soft: #a3a3a3; --xs:      #d4d4d4;
  --blue:  #2563eb; --green:  #16a34a; --red:  #dc2626;
  --mono: 'JetBrains Mono', monospace;
  --sans: 'Inter', sans-serif;
}
html { scroll-behavior: smooth; }
body {
  font-family: var(--sans); font-size: 13px; line-height: 1.5;
  background: var(--bg); color: var(--text);
  -webkit-font-smoothing: antialiased;
  font-feature-settings: "cv01","cv02","cv03","cv04","ss01";
}
/* topbar */
.top {
  height: 44px; border-bottom: 1px solid var(--line);
  display: flex; align-items: center; padding: 0 24px; gap: 20px;
  position: sticky; top: 0; background: var(--bg); z-index: 10;
}
.top-brand { font-size: 13px; font-weight: 600; color: var(--text); letter-spacing: -0.2px; }
.top-sep   { color: var(--line-md); font-weight: 300; }
.top-run   { font-size: 13px; color: var(--mid); }
.top-right { margin-left: auto; display: flex; align-items: center; gap: 16px; }
.top-meta  { font-size: 12px; color: var(--soft); }
.top-meta strong { color: var(--mid); font-weight: 500; }
/* layout */
.wrap { display: grid; grid-template-columns: 200px 1fr; max-width: 1080px; margin: 0 auto; }
/* nav */
.nav { border-right: 1px solid var(--line); padding: 28px 0; position: sticky; top: 44px; height: calc(100vh - 44px); overflow-y: auto; }
.nav-link { display: flex; align-items: center; gap: 10px; padding: 6px 16px; font-size: 12.5px; color: var(--soft); text-decoration: none; transition: color 0.1s; position: relative; }
.nav-link:hover { color: var(--text); }
.nav-link.active { color: var(--text); font-weight: 500; }
.nav-link.active::before { content: ''; position: absolute; left: 0; width: 2px; height: 20px; background: var(--text); border-radius: 0 2px 2px 0; }
.n-idx { font-family: var(--mono); font-size: 10px; color: var(--xs); min-width: 16px; }
.nav-link.active .n-idx { color: var(--soft); }
/* content */
.content { padding: 40px 40px 80px; min-width: 0; }
.page-top { margin-bottom: 40px; }
.pt-label { font-size: 11px; font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; color: var(--soft); margin-bottom: 8px; }
.pt-title { font-size: 22px; font-weight: 600; letter-spacing: -0.4px; color: var(--text); margin-bottom: 6px; line-height: 1.2; }
.pt-desc  { font-size: 13px; color: var(--mid); line-height: 1.6; max-width: 480px; margin-bottom: 20px; }
.tags { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 16px; }
.t { font-size: 11.5px; font-weight: 500; padding: 2px 8px; border-radius: 3px; border: 1px solid var(--line-md); color: var(--mid); background: var(--bg-row); }
.t-green { color: var(--green); border-color: #bbf7d0; background: #f0fdf4; }
.gh { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 500; color: var(--soft); text-decoration: none; border-bottom: 1px solid var(--line-md); padding-bottom: 1px; transition: color 0.15s, border-color 0.15s; }
.gh:hover { color: var(--text); border-color: var(--text); }
/* notice */
.notice { font-size: 12.5px; color: var(--mid); background: var(--bg-row); border-left: 2px solid var(--line-md); padding: 10px 14px; margin-bottom: 36px; line-height: 1.6; }
/* section */
.section { margin-bottom: 40px; }
.sec-head { display: flex; align-items: baseline; gap: 8px; margin-bottom: 0; padding-bottom: 8px; border-bottom: 1px solid var(--line); }
.si { font-family: var(--mono); font-size: 10.5px; color: var(--xs); }
.st { font-size: 13px; font-weight: 600; color: var(--text); letter-spacing: -0.1px; }
.sc { margin-left: auto; font-size: 11px; color: var(--soft); font-family: var(--mono); }
/* table */
.tbl { width: 100%; border-collapse: collapse; }
.tbl-group td { font-size: 10.5px; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; color: var(--soft); padding: 12px 0 4px; }
.param-row td { padding: 9px 0; border-bottom: 1px solid var(--line); vertical-align: middle; transition: background 0.1s, padding 0.1s; }
.param-row:last-child td { border-bottom: none; }
.param-row:hover td { background: var(--bg-row); }
.param-row:hover td:first-child { padding-left: 6px; }
.param-row:hover td:last-child  { padding-right: 6px; }
.k  { font-family: var(--mono); font-size: 12px; color: var(--mid); width: 55%; }
.kd { font-size: 11px; color: var(--soft); font-family: var(--sans); margin-top: 1px; }
.v-cell { text-align: right; vertical-align: middle; }
.v  { font-family: var(--mono); font-size: 12.5px; font-weight: 500; color: var(--text); }
.v-unit { font-size: 10.5px; font-weight: 400; color: var(--soft); }
.bool { display: inline-flex; align-items: center; gap: 4px; font-family: var(--mono); font-size: 11.5px; font-weight: 500; }
.bool-t { color: var(--green); } .bool-f { color: var(--red); }
.bool-dot { width: 5px; height: 5px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
.bool-t .bool-dot { background: var(--green); } .bool-f .bool-dot { background: var(--red); }
.path { font-family: var(--mono); font-size: 10.5px; color: var(--soft); text-align: right; word-break: break-all; white-space: normal; max-width: 320px; margin-left: auto; display: block; }
/* footer */
.footer { padding-top: 20px; border-top: 1px solid var(--line); display: flex; justify-content: space-between; align-items: center; margin-top: 40px; }
.fl { font-size: 12px; color: var(--soft); }
.fr { font-family: var(--mono); font-size: 11px; color: var(--xs); text-align: right; line-height: 1.8; }
@media (max-width: 640px) { .wrap { grid-template-columns: 1fr; } .nav { display: none; } .content { padding: 24px 16px 60px; } }
</style>
"""

_GH_ICON = '<svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>'

_JS = """
<script>
  const secs = document.querySelectorAll('.section[id]');
  const links = document.querySelectorAll('.nav-link');
  const io = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        links.forEach(l => l.classList.remove('active'));
        const a = document.querySelector('.nav-link[href="#' + e.target.id + '"]');
        if (a) a.classList.add('active');
      }
    });
  }, { rootMargin: '-10% 0px -80% 0px' });
  secs.forEach(s => io.observe(s));
</script>
"""

_NAV_ITEMS = [
    (1,  "Run Metadata"),
    (2,  "Paths"),
    (3,  "Target Area"),
    (4,  "Preprocessing"),
    (5,  "Raster"),
    (6,  "Ground Filtering"),
    (7,  "Validation"),
    (8,  "Advanced"),
    (9,  "GDAL / Runtime"),
]


# ── section builders ──────────────────────────────────────────────────────────

def _s1(c):
    rows = (
        _row("run_name",         _val(c.run_name))
        + _row("year",           _val(c.year))
        + _row("create_DSM",     _bool(c.create_DSM))
        + _row("create_DEM",     _bool(c.create_DEM))
        + _row("create_CHM",     _bool(c.create_CHM))
    )
    return _section(1, "Run Metadata", 5, rows)


def _s2(c):
    rows = (
        _group("Input")
        + _row("target_area_dir",   _path(c.target_area_dir),    "Study area polygons")
        + _row("las_files_dir",     _path(c.las_files_dir),      "Raw point clouds")
        + _row("las_footprints_dir",_path(c.las_footprints_dir), "Tile footprints")
        + _group("Output")
        + _row("preprocessed_dir",  _path(c.preprocessed_dir))
        + _row("results_dir",       _path(c.results_dir))
        + _row("validation_dir",    _path(c.validation_dir))
    )
    return _section(2, "Paths", 6, rows)


def _s3(c):
    rows = (
        _row("multiple_targets",  _bool(c.multiple_targets), "Process multiple areas in one run")
        + _row("target_name_field", _val(c.target_name_field), "Attribute field used as target identifier")
    )
    return _section(3, "Target Area Handling", 2, rows)


def _s4(c):
    rows = (
        _row("max_elevation_threshold", _val(c.max_elevation_threshold), "Upper quantile cutoff for elevation outliers")
        + _group("Statistical Outlier Removal (SOR)")
        + _row("knn",        _val(c.knn),        "Number of nearest neighbours")
        + _row("multiplier", _val(c.multiplier), "Standard deviation multiplier threshold")
    )
    return _section(4, "Preprocessing", 3, rows)


def _s5(c):
    rows = (
        _row("fill_gaps",           _bool(c.fill_gaps),          "Interpolate no-data regions")
        + _row("resolution",        _val(c.resolution, "m"),     "Output cell size")
        + _row("point_density_method", _val(c.point_density_method), "density | sampling")
    )
    return _section(5, "Raster Processing", 3, rows)


def _s6(c):
    rows = (
        _row("smrf_filter",  _bool(c.smrf_filter), "Simple Morphological Filter")
        + _row("csf_filter", _bool(c.csf_filter),  "Cloth Simulation Filter")
        + _row("threshold",  _val(c.threshold, "m"))
        + _group("SMRF")
        + _row("smrf_window_size", _val(c.smrf_window_size, "m"))
        + _row("smrf_slope",       _val(c.smrf_slope))
        + _row("smrf_scalar",      _val(c.smrf_scalar))
        + _group("CSF")
        + _row("csf_rigidness",       _val(c.csf_rigidness),         "Higher = flatter terrain assumption")
        + _row("csf_iterations",      _val(c.csf_iterations),        "Simulation steps")
        + _row("csf_time_step",       _val(c.csf_time_step))
        + _row("csf_cloth_resolution",_val(c.csf_cloth_resolution, "m"))
    )
    return _section(6, "Ground Filtering / Classification", 10, rows)


def _s7(c):
    rows = (
        _row("data_type",         _val(c.data_type),          "raster | vector (points)")
        + _row("validation_target", _val(c.validation_target), "DSM | DEM | CHM")
        + _group("Raster")
        + _row("val_band_raster", _val(c.val_band_raster))
        + _group("Point (only if data_type = vector)")
        + _row("val_column_point",_val(c.val_column_point))
        + _row("sample_size",     _val(c.sample_size))
    )
    return _section(7, "Validation", 5, rows)


def _s8(c):
    rows = (
        _row("overlap",         _val(c.overlap),     "Min overlap fraction between pointcloud and AOI")
        + _group("Date Filtering")
        + _row("filter_date",            _bool(c.filter_date))
        + _row("automatic_date_parser",  _bool(c.automatic_date_parser))
        + _row("start_date",             _val(c.start_date))
        + _row("end_date",               _val(c.end_date))
        + _group("Chunking / Parallelism")
        + _row("chunk_size",    _val(c.chunk_size,   "m"))
        + _row("buffer_size",   _val(c.buffer_size,  "m"))
        + _row("chunk_overlap", _val(c.chunk_overlap))
        + _row("num_workers",   _val(c.num_workers))
    )
    return _section(8, "Advanced Settings", 8, rows)


def _s9(c):
    rows = (
        _row("gdal.use_exceptions",    _bool(True),           "Raise Python exceptions on GDAL errors")
        + _row("gdal.cache_max_bytes", _val("32,000,000,000", "≈ 32 GB"), "Max GDAL block cache")
        + _row("python.warnings_filter", _val("ignore"))
    )
    return _section(9, "GDAL / Runtime Settings", 3, rows)


# ── public API ────────────────────────────────────────────────────────────────

def build_html(config) -> str:
    """Return the full HTML report as a string."""

    c = config
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    iso = datetime.datetime.now().isoformat(timespec="seconds")

    products = " ".join(
        f'<span class="t t-green">{p}</span>'
        for p, flag in [("DSM", c.create_DSM), ("DEM", c.create_DEM), ("CHM", c.create_CHM)]
        if flag
    )

    nav_html = "\n".join(
        f'<a class="nav-link{" active" if i == 1 else ""}" href="#s{i:02d}">'
        f'<span class="n-idx">{i:02d}</span>{label}</a>'
        for i, label in _NAV_ITEMS
    )

    sections_html = (
        _s1(c) + _s2(c) + _s3(c) + _s4(c) + _s5(c)
        + _s6(c) + _s7(c) + _s8(c) + _s9(c)
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Processing Report · {c.run_name}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
{_CSS}
</head>
<body>

<div class="top">
  <span class="top-brand">LidarProcessing</span>
  <span class="top-sep">/</span>
  <span class="top-run">{c.run_name}</span>
  <div class="top-right">
    <span class="top-meta"><strong>{now}</strong></span>
    <span class="top-meta"><strong>v1</strong></span>
  </div>
</div>

<div class="wrap">
  <nav class="nav">{nav_html}</nav>

  <div class="content">

    <div class="page-top">
      <div class="pt-label">Processing Report</div>
      <div class="pt-title">{c.run_name}</div>
      <p class="pt-desc">
        Parameter snapshot for reproducibility and audit.
        Path existence results are marked UNKNOWN unless written at runtime.
      </p>
      <div class="tags">
        {products}
        <span class="t">{c.resolution} m</span>
        <span class="t">{"IDW" if c.fill_gaps else "no gap fill"}</span>
        <span class="t">{"SMRF" if c.smrf_filter else ""}{" + " if c.smrf_filter and c.csf_filter else ""}{"CSF" if c.csf_filter else ""}</span>
        <span class="t">{c.num_workers} workers</span>
      </div>
      <a class="gh" href="https://github.com/awi-response/LidarProcessing" target="_blank">
        {_GH_ICON} awi-response/LidarProcessing
      </a>
    </div>

    <div class="notice">
      This file is a <strong>parameter snapshot</strong> intended for reproducibility + later audit.
      Path existence / folder creation results are marked UNKNOWN unless the pipeline writes them at runtime.
    </div>

    {sections_html}

    <div class="footer">
      <span class="fl">{c.run_name} · Configuration v1</span>
      <span class="fr">Generated {iso}</span>
    </div>

  </div>
</div>

{_JS}
</body>
</html>"""


def write_report(config, output_dir: str = None) -> str:
    """
    Write the HTML report to disk.

    Parameters
    ----------
    config     : Configuration instance (validated or not)
    output_dir : Directory to write to. Defaults to
                 {config.results_dir}/{config.run_name}/reports/

    Returns
    -------
    str : Absolute path of the written file.
    """
    if output_dir is None:
        output_dir = os.path.join(config.results_dir, config.run_name, "reports")

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"report_{config.run_name}_{timestamp}.html"
    filepath  = os.path.join(output_dir, filename)

    html = build_html(config)

    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(html)

    print(f"Processing report has been written to {filepath}")
    return filepath
