from __future__ import annotations
from typing import List, Any, Callable
from collections import deque
import io
import sys
import os
import threading
import traceback

from config import config as configuration

from textual.app import App, ComposeResult
from textual.widgets import (
    Header, Footer, Static, Collapsible,
    Input, Select, Checkbox, Button, Log, ProgressBar, Sparkline
)
from textual.containers import VerticalScroll, Horizontal, Container, Vertical


class _IOToLog(io.TextIOBase):
    """Redirect text writes into a Textual Log widget."""
    def __init__(self, get_log: Callable[[], Log]):
        self._get_log = get_log

    def write(self, s: str) -> int:
        if s:
            self._get_log().write(s)
        return len(s)

    def flush(self) -> None:  # pragma: no cover
        pass


class LidarProcessing(App):
    """Config editor (left) + controls, output, RAM bar + Run + bottom per-core CPU sparklines."""
    CSS_PATH = "hanna.css"

    # Enumerated option fields (use the config as reference for allowed values)
    OPTIONS: dict[str, list[str]] = {
        "point_density_method": ["sampling", "density"],
        "data_type": ["raster", "vector"],
        "validation_target": ["DSM", "DEM", "CHM"],
        # "resolution": ["Auto", "0.5", "1", "2", "5"],  # optional
    }

    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("pageup", "scroll_up", "Scroll up"),
        ("pagedown", "scroll_down", "Scroll down"),
        ("up", "scroll_up", "Scroll up"),
        ("down", "scroll_down", "Scroll down"),
        ("home", "scroll_top", "Top"),
        ("end", "scroll_bottom", "Bottom"),
        ("c", "collapse_or_expand(True)", "Collapse all"),
        ("e", "collapse_or_expand(False)", "Expand all"),
        ("r", "run_now", "Run"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.cfg = configuration.Configuration()
        # help texts extracted from inline comments in Configuration.__init__()
        self._help_map: dict[str, str] = {}
        if hasattr(configuration, "get_help_map"):
            try:
                self._help_map = configuration.get_help_map() or {}
            except Exception:
                self._help_map = {}

        # Which pipeline parts to run
        self.cfg.run_preprocessing = True
        self.cfg.run_processing = True
        self.cfg.run_validation = True

        # stdout/stderr redirection
        self._orig_stdout = None
        self._orig_stderr = None

        # psutil (lazy)
        self._psutil_ok: bool | None = None
        self._psutil = None

        # CPU sparkline data (filled on mount)
        self._cpu_values: list[deque[float]] = []

        # Run state
        self._runner: threading.Thread | None = None

    # ---------- Compose ----------
    def compose(self) -> ComposeResult:
        yield Header()

        # Middle: left config + right controls/output/ram+run
        yield Horizontal(
            # LEFT: Config panel
            Container(
                VerticalScroll(
                    Static("Configuration Variables", classes="header"),
                    *self._build_collapsible_groups(),
                    id="config_scroll",
                ),
                id="config_box",
            ),

            # RIGHT: Actions, Products, Output, RAM bar + Run button
            Container(
                Vertical(
                    Static("Process", classes="section-header"),
                    Container(
                        self._make_action_button("btn-preprocess", "Preprocess", self.cfg.run_preprocessing),
                        self._make_action_button("btn-process", "Process", self.cfg.run_processing),
                        self._make_action_button("btn-validate", "Validate", self.cfg.run_validation),
                        id="actions_row", classes="btn-row",
                    ),
                    Static("Products", classes="section-header", id="products_header"),
                    Container(
                        self._make_product_button("btn-prod-dsm", "DSM", self.cfg.create_DSM, enabled=self.cfg.run_processing),
                        self._make_product_button("btn-prod-dem", "DEM/DTM", self.cfg.create_DEM, enabled=self.cfg.run_processing),
                        self._make_product_button("btn-prod-chm", "CHM", self.cfg.create_CHM, enabled=self.cfg.run_processing),
                        id="products_row", classes="btn-row",
                    ),
                    Static("Output", id="output_header"),
                    Container(
                        Log(id="output_log"),
                        id="output_box",
                    ),
                    Static("RAM usage", id="ram_header"),
                    Container(
                        ProgressBar(total=100, show_eta=False, id="ram_bar"),
                        Button("Run", id="btn-run", variant="primary"),
                        id="ram_box",
                    ),
                    id="right_stack",
                ),
                id="right_box",
            ),
            id="config_row",
        )

        # Bottom full-width per-core CPU sparklines (added at very bottom)
        yield Container(id="cpu_row")

        yield Footer()

    # ---------- Lifecycle ----------
    def on_mount(self) -> None:

        self.add_class(
            "theme-terminal"
            if (os.getenv("COLORTERM") or os.getenv("TERM_PROGRAM") or os.getenv("WT_SESSION") or os.getenv("TERM"))
            else "theme-default"
        )

        # Focus the left scroll so keyboard scrolling works immediately
        self.query_one("#config_scroll").focus()
        self._update_products_enabled()

        # Redirect stdout/stderr into the on-screen Log
        log_getter = lambda: self.query_one("#output_log", Log)
        self._orig_stdout, self._orig_stderr = sys.stdout, sys.stderr
        sys.stdout = _IOToLog(log_getter)
        sys.stderr = _IOToLog(log_getter)

        # Start periodic updates
        self.set_interval(1.0, self._update_ram)
        self._init_cpu_sparklines()
        self.set_interval(1.0, self._update_cpu)

        # Make the log auto-scroll as lines arrive
        self.query_one("#output_log", Log).auto_scroll = True

        print("Processing Shell started. Output will appear here.\n")

    def on_unmount(self) -> None:
        # Restore stdio
        if self._orig_stdout is not None:
            sys.stdout = self._orig_stdout
        if self._orig_stderr is not None:
            sys.stderr = self._orig_stderr

    # ---------- Builders ----------
    def _build_collapsible_groups(self) -> List[Collapsible]:
        groups: dict[str, list[str]] = {
            "General Settings": ["run_name"],
            "Paths": [
                "target_area_dir", "las_files_dir", "las_footprints_dir",
                "preprocessed_dir", "results_dir", "validation_dir",
            ],
            "Preprocessing": [
                "multiple_targets", "target_name_field",
                "max_elevation_threshold", "knn", "multiplier",
            ],
            "Processing": [
                "fill_gaps", "resolution", "point_density_method",
            ],
            "Ground Filtering": [
                "smrf_filter", "csf_filter", "threshold",
                "smrf_window_size", "smrf_slope", "smrf_scalar",
                "csf_rigidness", "csf_iterations", "csf_time_step",
                "csf_cloth_resolution",
            ],
            "Validation": [
                "data_type", "validation_target", "val_column_point",
                "val_band_raster", "sample_size",
            ],
            "Advanced Settings": [
                "overlap", "filter_date", "start_date", "end_date",
                "preprocess_use_chunks", "chunk_size", "chunk_overlap", "num_workers",
            ],
        }

        collapsibles: list[Collapsible] = []
        for group_name, var_names in groups.items():
            rows: list[Horizontal] = []

            for var_name in var_names:
                value = getattr(self.cfg, var_name)

                # Label
                name = Static(f"{var_name}:", classes="var-name")
                name.styles.width = 24
                name.styles.min_width = 12
                # Tooltip from inline comments (hover over the label)
                help_text = self._help_map.get(var_name)
                if help_text:
                    name.tooltip = help_text

                # Editor
                editor = self._make_editor(var_name, value)
                editor.styles.width = "1fr"
                editor.styles.min_width = 0

                rows.append(Horizontal(name, editor, classes="row"))

            # Wrap rows in a scrollable body so expanded groups don't fill the whole screen
            body = VerticalScroll(*rows, classes="group-body")

            # Collapsible group (collapsed by default)
            collapsibles.append(
                Collapsible(
                    body,
                    title=group_name,
                    collapsed=True,
                    classes="config-group",
                )
            )

        return collapsibles

    def _make_editor(self, var_name: str, value: Any):
        editor_id = f"edit-{var_name}"

        # Enumerated options -> Select
        if var_name in self.OPTIONS:
            options = [(opt, opt) for opt in self.OPTIONS[var_name]]
            v = str(value)
            if v not in [o[1] for o in options]:
                options = [(v, v)] + options  # keep current unknown value selectable
            return Select(options=options, value=v, id=editor_id, classes="var-editor")

        # Booleans -> Checkbox
        if isinstance(value, bool):
            return Checkbox(value=value, id=editor_id, classes="var-editor")

        # Everything else -> Input (string/numeric/dates/paths)
        return Input(value=str(value), id=editor_id, classes="var-editor")

    # ---------- Events: push edits into self.cfg ----------
    def on_input_submitted(self, event: Input.Submitted) -> None:
        var = (event.input.id or "").removeprefix("edit-")
        self._set_config_value(var, event.value)

    def on_input_blurred(self, event: Input.Blurred) -> None:
        var = (event.input.id or "").removeprefix("edit-")
        self._set_config_value(var, event.input.value)

    def on_select_changed(self, event: Select.Changed) -> None:
        var = (event.select.id or "").removeprefix("edit-")
        self._set_config_value(var, event.value)

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        var = (event.checkbox.id or "").removeprefix("edit-")
        self._set_config_value(var, event.value)

    # ---------- Action & product buttons ----------
    def _make_action_button(self, btn_id: str, label: str, active: bool) -> Button:
        # Active -> primary; inactive -> default + dim style (still clickable)
        btn = Button(label, id=btn_id, classes="action-btn",
                     variant="primary" if active else "default")
        if not active:
            btn.add_class("as-disabled")
        return btn

    def _make_product_button(self, btn_id: str, label: str, active: bool, enabled: bool) -> Button:
        return Button(
            label, id=btn_id, classes="product-btn",
            variant="primary" if active else "default",
            disabled=not enabled,
        )

    def _update_products_enabled(self) -> None:
        processing_on = bool(getattr(self.cfg, "run_processing", True))
        for btn_id, cfg_attr in [
            ("#btn-prod-dsm", "create_DSM"),
            ("#btn-prod-dem", "create_DEM"),
            ("#btn-prod-chm", "create_CHM"),
        ]:
            btn = self.query_one(btn_id, Button)
            btn.disabled = not processing_on
            # keep variant in sync with config value
            btn.variant = "primary" if getattr(self.cfg, cfg_attr) else "default"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn = event.button
        # Action buttons
        if btn.id == "btn-preprocess":
            self._toggle_action(btn, "run_preprocessing", "Preprocess")
        elif btn.id == "btn-process":
            self._toggle_action(btn, "run_processing", "Process")
            self._update_products_enabled()
        elif btn.id == "btn-validate":
            self._toggle_action(btn, "run_validation", "Validate")
        # Product buttons (won't fire if disabled)
        elif btn.id == "btn-prod-dsm":
            self._toggle_product(btn, "create_DSM")
        elif btn.id == "btn-prod-dem":
            self._toggle_product(btn, "create_DEM")
        elif btn.id == "btn-prod-chm":
            self._toggle_product(btn, "create_CHM")
        # Run button
        elif btn.id == "btn-run":
            self._start_run()

    # quick keybinding
    def action_run_now(self) -> None:
        self._start_run()

    def _toggle_action(self, btn: Button, cfg_attr: str, label_text: str) -> None:
        if btn.variant == "primary":
            btn.variant = "default"
            btn.add_class("as-disabled")
            btn.label = label_text
            setattr(self.cfg, cfg_attr, False)
            print(f"{label_text}: OFF\n")
        else:
            btn.variant = "primary"
            btn.remove_class("as-disabled")
            btn.label = label_text
            setattr(self.cfg, cfg_attr, True)
            print(f"{label_text}: ON\n")

    def _toggle_product(self, btn: Button, cfg_attr: str) -> None:
        new_val = not bool(getattr(self.cfg, cfg_attr))
        setattr(self.cfg, cfg_attr, new_val)
        btn.variant = "primary" if new_val else "default"
        print(f"Product {cfg_attr} -> {new_val}\n")

    # ---------- Run pipeline ----------
    def _start_run(self) -> None:
        # Already running?
        if self._runner and self._runner.is_alive():
            print("A run is already in progress.\n")
            return

        # Update button UI
        run_btn = self.query_one("#btn-run", Button)
        run_btn.disabled = True
        run_btn.label = "Running…"

        print("Starting run with current configuration...\n")

        # Launch background thread
        self._runner = threading.Thread(target=self._run_pipeline, daemon=True)
        self._runner.start()

    def _run_pipeline(self) -> None:
        try:
            # Validate config
            try:
                self.cfg.validate()
            except Exception as e:
                print(f"[CONFIG ERROR] {e}\n")
                return

            # Import pipeline modules lazily
            try:
                import preprocessing  # type: ignore
            except Exception:
                preprocessing = None  # type: ignore
            try:
                import processing  # type: ignore
            except Exception:
                processing = None  # type: ignore
            try:
                import validation  # type: ignore
            except Exception:
                validation = None  # type: ignore

            # Run selected stages
            if getattr(self.cfg, "run_preprocessing", False):
                if preprocessing and hasattr(preprocessing, "preprocess_all"):
                    print("==> Preprocessing...\n")
                    preprocessing.preprocess_all(self.cfg)  # type: ignore[attr-defined]
                    print("Preprocessing done.\n")
                else:
                    print("Preprocessing module/function not available.\n")

            if getattr(self.cfg, "run_processing", False):
                if processing and hasattr(processing, "process_all"):
                    print("==> Processing...\n")
                    processing.process_all(self.cfg)  # type: ignore[attr-defined]
                    print("Processing done.\n")
                else:
                    print("Processing module/function not available.\n")

            if getattr(self.cfg, "run_validation", False):
                if validation and hasattr(validation, "validate_all"):
                    print("==> Validation...\n")
                    validation.validate_all(self.cfg)  # type: ignore[attr-defined]
                    print("Validation done.\n")
                else:
                    print("Validation module/function not available.\n")

            print("Run complete.\n")

        except Exception:
            tb = traceback.format_exc()
            print(f"[RUN ERROR]\n{tb}\n")

        finally:
            # Re-enable Run button on the UI thread
            def _finish():
                btn = self.query_one("#btn-run", Button)
                btn.disabled = False
                btn.label = "Run"
            self.call_from_thread(_finish)

    # ---------- Scroll actions ----------
    def action_scroll_up(self) -> None:
        self.query_one("#config_scroll", VerticalScroll).scroll_relative(y=-5)

    def action_scroll_down(self) -> None:
        self.query_one("#config_scroll", VerticalScroll).scroll_relative(y=5)

    def action_scroll_top(self) -> None:
        self.query_one("#config_scroll", VerticalScroll).scroll_home()

    def action_scroll_bottom(self) -> None:
        self.query_one("#config_scroll", VerticalScroll).scroll_end()

    def action_collapse_or_expand(self, collapse: bool) -> None:
        for coll in self.walk_children(Collapsible):
            coll.collapsed = collapse

    # ---------- Casting & assignment ----------
    def _set_config_value(self, var_name: str, raw_value: Any) -> None:
        if not var_name:
            return

        if var_name in self.OPTIONS:
            setattr(self.cfg, var_name, str(raw_value))
            return

        current = getattr(self.cfg, var_name)

        if var_name == "resolution":
            if isinstance(raw_value, str) and raw_value.strip().lower() == "auto":
                setattr(self.cfg, var_name, "Auto")
                return
            try:
                v = raw_value.strip() if isinstance(raw_value, str) else raw_value
                setattr(self.cfg, var_name, int(v))
            except Exception:
                try:
                    setattr(self.cfg, var_name, float(v))  # type: ignore[name-defined]
                except Exception:
                    setattr(self.cfg, var_name, str(raw_value))
            return

        if isinstance(current, bool):
            setattr(self.cfg, var_name, bool(raw_value))
            return

        if isinstance(current, int) and not isinstance(current, bool):
            try:
                setattr(self.cfg, var_name, int(str(raw_value).strip()))
            except Exception:
                setattr(self.cfg, var_name, str(raw_value))
            return

        if isinstance(current, float):
            try:
                setattr(self.cfg, var_name, float(str(raw_value).strip()))
            except Exception:
                setattr(self.cfg, var_name, str(raw_value))
            return

        setattr(self.cfg, var_name, str(raw_value))

    # ---------- RAM polling ----------
    def _update_ram(self) -> None:
        """Poll system RAM % and push into the progress bar once per second."""
        if self._psutil_ok is None:
            if not self._ensure_psutil():
                print("psutil not available; RAM bar disabled.\n")
                return
        if not self._psutil_ok:
            return

        try:
            mem = self._psutil.virtual_memory()
            pct = float(mem.percent)  # 0..100
        except Exception:
            return

        bar = self.query_one("#ram_bar", ProgressBar)
        bar.progress = max(0, min(100, int(round(pct))))

    # ---------- CPU sparklines ----------
    def _ensure_psutil(self) -> bool:
        """Try to import psutil once."""
        try:
            import psutil  # type: ignore
            self._psutil_ok = True
            self._psutil = psutil
        except Exception:
            self._psutil_ok = False
        return bool(self._psutil_ok)

    def _init_cpu_sparklines(self) -> None:
        """Create one sparkline per logical CPU, full-width at bottom."""
        if self._psutil_ok is None:
            if not self._ensure_psutil():
                print("psutil not available; CPU sparklines disabled.\n")
                return
        if not self._psutil_ok:
            return

        # Determine core count, warm up cpu_percent, and build deques
        ncores = self._psutil.cpu_count(logical=True) or 1
        self._psutil.cpu_percent(percpu=True)  # prime measurement
        self._cpu_values = [deque(maxlen=120) for _ in range(ncores)]  # 2 minutes @ 1Hz

        row = self.query_one("#cpu_row", Container)
        widgets: list[Sparkline] = []
        for i in range(ncores):
            sp = Sparkline(id=f"cpu_sp_{i}")
            sp.add_class("cpu-spark")
            sp.max_value = 100.0
            widgets.append(sp)

        row.mount(*widgets)

    def _update_cpu(self) -> None:
        """Update per-core sparkline data every second."""
        if not self._psutil_ok:
            return
        try:
            per = self._psutil.cpu_percent(percpu=True)
        except Exception:
            return
        if not per:
            return

        # Append data and refresh widgets
        for i, pct in enumerate(per):
            if i >= len(self._cpu_values):
                break
            self._cpu_values[i].append(float(pct))
            try:
                sp = self.query_one(f"#cpu_sp_{i}", Sparkline)
                sp.data = list(self._cpu_values[i])
            except Exception:
                pass


if __name__ == "__main__":
    LidarProcessing().run()
