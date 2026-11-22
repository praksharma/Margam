"""Dedicated Solution Explorer window for interactive 1D, 2D, 3D, and parametric post-processing."""
from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
import numpy as np
from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .icons import icon
from .plots import CurvePlot
from .theme import colors, configure_theme, theme_name
from .viewport_2d import ColorbarWidget, Viewport2D
from .viewport_3d import Viewport3D
from ..visualization.colormaps import COLORMAPS, COLORMAP_LABELS
from ..visualization.dataset import DISPLAY_MODES, SolutionDataset
from ..visualization.loaders import from_file, from_output_result, from_run_archive


class SolutionExplorerWindow(QMainWindow):
    """Standalone modeless post-processor supporting 1D, 2D, 3D, transient, and parametric PINNs."""

    def __init__(self, owner=None, dataset: SolutionDataset | None = None):
        super().__init__(owner)
        self.owner = owner
        self.dataset: SolutionDataset | None = None
        self.datasets_by_id: dict[str, SolutionDataset] = {}
        self.study = None
        self._updating_controls = False
        self._locked_limits: tuple[float, float] | None = None
        self._last_line_cut: dict[str, Any] | None = None

        self.setWindowTitle("Margam · Solution Explorer")
        self.resize(1360, 880)

        self._build_toolbar()
        self._build_central_area()
        self._build_settings_dock()
        self._build_probe_dock()

        self.play_timer = QTimer(self)
        self.play_timer.setInterval(180)
        self.play_timer.timeout.connect(self._advance_time_frame)

        self.statusBar().showMessage("Ready · open a trained output, saved run, study folder, or 2D/3D data file")
        if dataset is not None:
            self.set_dataset(dataset)
        elif owner is not None:
            self.sync_from_owner()

    def _build_toolbar(self):
        tb = self.addToolBar("Solution Explorer")
        tb.setIconSize(QSize(18, 18))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.explorer_toolbar = tb

        self.sync_owner_action = tb.addAction("Sync active run")
        self.sync_owner_action.setIcon(icon("play"))
        self.sync_owner_action.setEnabled(self.owner is not None)
        self.sync_owner_action.triggered.connect(self.sync_from_owner)

        self.open_file_action = tb.addAction("Open file / run…")
        self.open_file_action.setIcon(icon("open"))
        self.open_file_action.triggered.connect(lambda: self.open_source_file())

        self.open_study_action = tb.addAction("Open study…")
        self.open_study_action.setIcon(icon("comparison"))
        self.open_study_action.triggered.connect(lambda: self.open_study_folder())

        tb.addSeparator()
        tb.addWidget(QLabel(" Output: "))
        self.output_combo = QComboBox()
        self.output_combo.setObjectName("explorer_output_combo")
        self.output_combo.setMinimumWidth(180)
        self.output_combo.currentIndexChanged.connect(self._on_output_selected)
        tb.addWidget(self.output_combo)

        tb.addWidget(QLabel(" Mode: "))
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.setObjectName("explorer_view_mode")
        for key, label in (
            ("auto", "Auto"),
            ("1d", "1D Profile"),
            ("2d", "2D Field"),
            ("3d", "3D Viewport"),
            ("compare", "3-Panel Comparison"),
        ):
            self.view_mode_combo.addItem(label, key)
        self.view_mode_combo.currentIndexChanged.connect(self.refresh_view)
        tb.addWidget(self.view_mode_combo)

        tb.addSeparator()
        self.reset_view_action = tb.addAction("Reset view")
        self.reset_view_action.setIcon(icon("fit"))
        self.reset_view_action.triggered.connect(self.reset_camera)

        self.snapshot_action = tb.addAction("Snapshot PNG…")
        self.snapshot_action.setIcon(icon("export"))
        self.snapshot_action.triggered.connect(lambda: self.export_snapshot())

        self.export_frames_action = tb.addAction("Export frames…")
        self.export_frames_action.setIcon(icon("export"))
        self.export_frames_action.triggered.connect(lambda: self.export_frame_sequence())

        self.export_probe_action = tb.addAction("Export probe CSV…")
        self.export_probe_action.setIcon(icon("save"))
        self.export_probe_action.triggered.connect(lambda: self.export_probe_csv())

    def _build_central_area(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        self.view_stack = QStackedWidget()
        layout.addWidget(self.view_stack, 1)

        # Page 0: 1D Profile
        self.view_1d = CurvePlot(logarithmic=False)
        self.view_1d.setObjectName("explorer_view_1d")
        self.view_stack.addWidget(self.view_1d)

        # Page 1: 2D Field + Colorbar
        page_2d = QWidget()
        row_2d = QHBoxLayout(page_2d)
        row_2d.setContentsMargins(0, 0, 0, 0)
        self.viewport_2d = Viewport2D()
        self.viewport_2d.setObjectName("explorer_viewport_2d")
        self.colorbar_2d = ColorbarWidget()
        row_2d.addWidget(self.viewport_2d, 1)
        row_2d.addWidget(self.colorbar_2d)
        self.view_stack.addWidget(page_2d)

        self.viewport_2d.probeMoved.connect(self._on_probe_moved)
        self.viewport_2d.lineCutChanged.connect(self._on_line_cut_changed)

        # Page 2: 3D Viewport + Colorbar
        self.page_3d = QWidget()
        self.layout_3d = QHBoxLayout(self.page_3d)
        self.layout_3d.setContentsMargins(0, 0, 0, 0)
        self.viewport_3d = Viewport3D()
        self.viewport_3d.setObjectName("explorer_viewport_3d")
        self.colorbar_3d = ColorbarWidget()
        self.layout_3d.addWidget(self.viewport_3d, 1)
        self.layout_3d.addWidget(self.colorbar_3d)
        self.view_stack.addWidget(self.page_3d)

        # Bottom Time Scrubber Bar (hidden for steady datasets)
        self.time_bar = QWidget()
        t_row = QHBoxLayout(self.time_bar)
        t_row.setContentsMargins(4, 2, 4, 2)
        self.play_button = QPushButton("Play")
        self.play_button.setObjectName("explorer_time_play")
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self._toggle_play)
        self.step_back_button = QPushButton("−")
        self.step_back_button.setFixedWidth(28)
        self.step_back_button.clicked.connect(lambda: self.set_time_index(self.time_slider.value() - 1))
        self.step_fwd_button = QPushButton("+")
        self.step_fwd_button.setFixedWidth(28)
        self.step_fwd_button.clicked.connect(lambda: self.set_time_index(self.time_slider.value() + 1))
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setObjectName("explorer_time_slider")
        self.time_slider.valueChanged.connect(self._on_time_changed)
        self.time_label = QLabel("t = 0")
        self.time_label.setObjectName("explorer_time_label")
        self.fps_combo = QComboBox()
        self.fps_combo.setObjectName("explorer_fps_combo")
        for fps in (2, 5, 10, 20):
            self.fps_combo.addItem(f"{fps} FPS", fps)
        self.fps_combo.setCurrentIndex(1)
        self.fps_combo.currentIndexChanged.connect(self._on_fps_changed)
        self.loop_check = QCheckBox("Loop")
        self.loop_check.setChecked(True)
        self.time_trace_button = QPushButton("Time trace")
        self.time_trace_button.setObjectName("explorer_time_trace")
        self.time_trace_button.clicked.connect(lambda: self.plot_time_trace())
        for w in (
            QLabel("Time:"),
            self.play_button,
            self.step_back_button,
            self.time_slider,
            self.step_fwd_button,
            self.time_label,
            self.fps_combo,
            self.loop_check,
            self.time_trace_button,
        ):
            t_row.addWidget(w)
        layout.addWidget(self.time_bar)
        self.time_bar.hide()

        self.setCentralWidget(central)

    def _build_settings_dock(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        panel = QWidget()
        vbox = QVBoxLayout(panel)

        # 1. Field & Display
        grp_field = QGroupBox("Field & quantity")
        form_field = QFormLayout(grp_field)
        self.field_combo = QComboBox()
        self.field_combo.setObjectName("explorer_field_combo")
        self.field_combo.currentIndexChanged.connect(self.refresh_view)
        form_field.addRow("Field", self.field_combo)

        self.display_mode_combo = QComboBox()
        self.display_mode_combo.setObjectName("explorer_display_mode")
        for key, label in DISPLAY_MODES:
            self.display_mode_combo.addItem(label, key)
        self.display_mode_combo.currentIndexChanged.connect(self.refresh_view)
        form_field.addRow("Quantity", self.display_mode_combo)
        vbox.addWidget(grp_field)

        # 2. Colormap & Scale
        grp_cmap = QGroupBox("Colormap & scale")
        form_cmap = QFormLayout(grp_cmap)
        self.cmap_combo = QComboBox()
        self.cmap_combo.setObjectName("explorer_cmap_combo")
        for key in COLORMAPS:
            self.cmap_combo.addItem(COLORMAP_LABELS[key], key)
        self.cmap_combo.currentIndexChanged.connect(self.refresh_view)
        form_cmap.addRow("Colormap", self.cmap_combo)

        self.auto_range_check = QCheckBox("Auto range")
        self.auto_range_check.setObjectName("explorer_auto_range")
        self.auto_range_check.setChecked(True)
        self.auto_range_check.toggled.connect(self._on_range_toggle)
        form_cmap.addRow(self.auto_range_check)

        self.symmetric_check = QCheckBox("Symmetric ±max around 0")
        self.symmetric_check.setObjectName("explorer_symmetric")
        self.symmetric_check.toggled.connect(self.refresh_view)
        form_cmap.addRow(self.symmetric_check)

        self.lock_range_check = QCheckBox("Lock range across time/params")
        self.lock_range_check.setObjectName("explorer_lock_range")
        self.lock_range_check.toggled.connect(self._on_lock_range_toggled)
        form_cmap.addRow(self.lock_range_check)

        self.log_scale_check = QCheckBox("Log10 |value| scale")
        self.log_scale_check.setObjectName("explorer_log_scale")
        self.log_scale_check.toggled.connect(self.refresh_view)
        form_cmap.addRow(self.log_scale_check)

        self.vmin_spin = QDoubleSpinBox()
        self.vmin_spin.setRange(-1e9, 1e9)
        self.vmin_spin.setDecimals(5)
        self.vmin_spin.setKeyboardTracking(False)
        self.vmin_spin.setEnabled(False)
        self.vmin_spin.valueChanged.connect(self.refresh_view)
        form_cmap.addRow("Min", self.vmin_spin)

        self.vmax_spin = QDoubleSpinBox()
        self.vmax_spin.setRange(-1e9, 1e9)
        self.vmax_spin.setDecimals(5)
        self.vmax_spin.setKeyboardTracking(False)
        self.vmax_spin.setEnabled(False)
        self.vmax_spin.valueChanged.connect(self.refresh_view)
        form_cmap.addRow("Max", self.vmax_spin)
        vbox.addWidget(grp_cmap)

        # 3. 2D Display Settings
        self.group_2d = QGroupBox("2D display & probes")
        form_2d = QFormLayout(self.group_2d)
        self.repr_2d_combo = QComboBox()
        self.repr_2d_combo.setObjectName("explorer_repr_2d")
        for key, label in (
            ("heatmap_contours", "Heatmap + contours"),
            ("heatmap", "Smooth heatmap"),
            ("filled_contours", "Filled contours"),
            ("contours", "Contour lines only"),
            ("points", "Sample points"),
        ):
            self.repr_2d_combo.addItem(label, key)
        self.repr_2d_combo.currentIndexChanged.connect(self.refresh_view)
        form_2d.addRow("Style", self.repr_2d_combo)

        self.contour_levels_spin = QSpinBox()
        self.contour_levels_spin.setRange(2, 40)
        self.contour_levels_spin.setValue(12)
        self.contour_levels_spin.setKeyboardTracking(False)
        self.contour_levels_spin.valueChanged.connect(self.refresh_view)
        form_2d.addRow("Contour levels", self.contour_levels_spin)

        self.quiver_check = QCheckBox("Vector quiver arrows")
        self.quiver_check.setObjectName("explorer_quiver")
        self.quiver_check.toggled.connect(self.refresh_view)
        form_2d.addRow(self.quiver_check)

        self.quiver_stride_spin = QSpinBox()
        self.quiver_stride_spin.setRange(1, 20)
        self.quiver_stride_spin.setValue(4)
        self.quiver_stride_spin.setKeyboardTracking(False)
        self.quiver_stride_spin.valueChanged.connect(self.refresh_view)
        form_2d.addRow("Quiver stride", self.quiver_stride_spin)

        self.streamlines_check = QCheckBox("Streamlines (RK4)")
        self.streamlines_check.setObjectName("explorer_streamlines")
        self.streamlines_check.toggled.connect(self.refresh_view)
        form_2d.addRow(self.streamlines_check)

        self.streamline_density_spin = QSpinBox()
        self.streamline_density_spin.setRange(3, 18)
        self.streamline_density_spin.setValue(8)
        self.streamline_density_spin.setKeyboardTracking(False)
        self.streamline_density_spin.valueChanged.connect(self.refresh_view)
        form_2d.addRow("Streamline density", self.streamline_density_spin)

        self.points_overlay_check = QCheckBox("Overlay sample points")
        self.points_overlay_check.toggled.connect(self.refresh_view)
        form_2d.addRow(self.points_overlay_check)

        self.line_cut_check = QCheckBox("Drag line-cut probe")
        self.line_cut_check.setObjectName("explorer_line_cut_mode")
        self.line_cut_check.toggled.connect(self._on_line_cut_mode)
        form_2d.addRow(self.line_cut_check)

        cut_row = QHBoxLayout()
        btn_h = QPushButton("Mid H-cut")
        btn_h.clicked.connect(lambda: self.apply_preset_cut("horizontal"))
        btn_v = QPushButton("Mid V-cut")
        btn_v.clicked.connect(lambda: self.apply_preset_cut("vertical"))
        btn_clr = QPushButton("Clear")
        btn_clr.clicked.connect(self._clear_line_cut)
        for b in (btn_h, btn_v, btn_clr):
            cut_row.addWidget(b)
        form_2d.addRow(cut_row)
        vbox.addWidget(self.group_2d)

        # 4. 3D / Slice Settings
        self.group_3d = QGroupBox("3D viewport & slice settings")
        self.form_3d = QFormLayout(self.group_3d)

        cam_row = QHBoxLayout()
        for key, label in (("iso", "Iso"), ("xy", "XY"), ("xz", "XZ"), ("yz", "YZ")):
            btn = QPushButton(label)
            btn.setObjectName(f"explorer_cam_{key}")
            btn.clicked.connect(lambda checked=False, k=key: self.viewport_3d.set_preset(k))
            cam_row.addWidget(btn)
        self.form_3d.addRow("Camera", cam_row)

        self.ortho_check = QCheckBox("Orthographic projection")
        self.ortho_check.setObjectName("explorer_ortho")
        self.ortho_check.toggled.connect(lambda checked: self.viewport_3d.set_orthographic(checked))
        self.form_3d.addRow(self.ortho_check)

        self.mode_3d_combo = QComboBox()
        self.mode_3d_combo.setObjectName("explorer_mode_3d")
        for key, label in (
            ("auto", "Auto (heightfield / slices / cloud)"),
            ("heightfield", "3D heightfield surface z=f(x,y)"),
            ("slices", "Orthogonal 3D slices (X/Y/Z)"),
            ("cloud", "3D point cloud (thresholded)"),
            ("isosurface", "3D isosurface (marching tets)"),
            ("slices_iso", "Slices + Isosurface"),
        ):
            self.mode_3d_combo.addItem(label, key)
        self.mode_3d_combo.currentIndexChanged.connect(self.refresh_view)
        self.form_3d.addRow("3D style", self.mode_3d_combo)

        self.slice_axis_combo = QComboBox()
        self.slice_axis_combo.setObjectName("explorer_slice_axis")
        for axis in ("z", "y", "x"):
            self.slice_axis_combo.addItem(f"{axis.upper()} normal plane (2D view)", axis)
        self.slice_axis_combo.currentIndexChanged.connect(self._on_slice_axis_changed)
        self.form_3d.addRow("2D slice plane", self.slice_axis_combo)

        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setObjectName("explorer_slice_slider")
        self.slice_slider.setRange(0, 100)
        self.slice_slider.setValue(50)
        self.slice_slider.valueChanged.connect(self.refresh_view)
        self.slice_value_label = QLabel("0.5")
        s_row = QHBoxLayout()
        s_row.addWidget(self.slice_slider, 1)
        s_row.addWidget(self.slice_value_label)
        self.form_3d.addRow("2D slice pos", s_row)

        # Orthogonal 3D X/Y/Z slice toggles + sliders
        self.slice_3d_checks: dict[str, QCheckBox] = {}
        self.slice_3d_sliders: dict[str, QSlider] = {}
        for axis, default_on in (("x", False), ("y", False), ("z", True)):
            chk = QCheckBox(f"{axis.upper()} slice")
            chk.setObjectName(f"explorer_slice3d_{axis}_check")
            chk.setChecked(default_on)
            chk.toggled.connect(self.refresh_view)
            sld = QSlider(Qt.Orientation.Horizontal)
            sld.setObjectName(f"explorer_slice3d_{axis}_slider")
            sld.setRange(0, 100)
            sld.setValue(50)
            sld.valueChanged.connect(self.refresh_view)
            row_s = QHBoxLayout()
            row_s.addWidget(chk)
            row_s.addWidget(sld, 1)
            self.form_3d.addRow(row_s)
            self.slice_3d_checks[axis] = chk
            self.slice_3d_sliders[axis] = sld

        self.iso_slider = QSlider(Qt.Orientation.Horizontal)
        self.iso_slider.setObjectName("explorer_iso_slider")
        self.iso_slider.setRange(1, 99)
        self.iso_slider.setValue(50)
        self.iso_slider.valueChanged.connect(self.refresh_view)
        self.form_3d.addRow("Isosurface level", self.iso_slider)

        self.iso_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.iso_opacity_slider.setRange(40, 255)
        self.iso_opacity_slider.setValue(200)
        self.iso_opacity_slider.valueChanged.connect(self.refresh_view)
        self.form_3d.addRow("Isosurface alpha", self.iso_opacity_slider)

        self.cloud_point_spin = QDoubleSpinBox()
        self.cloud_point_spin.setRange(1.5, 12.0)
        self.cloud_point_spin.setSingleStep(0.5)
        self.cloud_point_spin.setValue(4.0)
        self.cloud_point_spin.valueChanged.connect(self.refresh_view)
        self.form_3d.addRow("Point size", self.cloud_point_spin)

        self.thresh_min_slider = QSlider(Qt.Orientation.Horizontal)
        self.thresh_min_slider.setObjectName("explorer_thresh_min")
        self.thresh_min_slider.setRange(0, 100)
        self.thresh_min_slider.setValue(0)
        self.thresh_min_slider.valueChanged.connect(self.refresh_view)
        self.form_3d.addRow("Cloud min %", self.thresh_min_slider)

        self.thresh_max_slider = QSlider(Qt.Orientation.Horizontal)
        self.thresh_max_slider.setObjectName("explorer_thresh_max")
        self.thresh_max_slider.setRange(0, 100)
        self.thresh_max_slider.setValue(100)
        self.thresh_max_slider.valueChanged.connect(self.refresh_view)
        self.form_3d.addRow("Cloud max %", self.thresh_max_slider)

        self.clip_axis_combo = QComboBox()
        for key, label in (("none", "None"), ("x", "Clip X ≤"), ("y", "Clip Y ≤"), ("z", "Clip Z ≤")):
            self.clip_axis_combo.addItem(label, key)
        self.clip_axis_combo.currentIndexChanged.connect(self.refresh_view)
        self.clip_slider = QSlider(Qt.Orientation.Horizontal)
        self.clip_slider.setRange(0, 100)
        self.clip_slider.setValue(100)
        self.clip_slider.valueChanged.connect(self.refresh_view)
        clip_row = QHBoxLayout()
        clip_row.addWidget(self.clip_axis_combo)
        clip_row.addWidget(self.clip_slider, 1)
        self.form_3d.addRow("Cutaway", clip_row)

        self.warp_scale_spin = QDoubleSpinBox()
        self.warp_scale_spin.setRange(0.05, 1.5)
        self.warp_scale_spin.setSingleStep(0.1)
        self.warp_scale_spin.setValue(0.55)
        self.warp_scale_spin.valueChanged.connect(self.refresh_view)
        self.form_3d.addRow("Height warp", self.warp_scale_spin)

        self.wireframe_check = QCheckBox("Wireframe overlay")
        self.wireframe_check.toggled.connect(self.refresh_view)
        self.form_3d.addRow(self.wireframe_check)

        self.show_box_check = QCheckBox("Show bounding box & axes")
        self.show_box_check.setChecked(True)
        self.show_box_check.toggled.connect(self.refresh_view)
        self.form_3d.addRow(self.show_box_check)

        self.show_obstacle_check = QCheckBox("Show obstacle mesh")
        self.show_obstacle_check.setChecked(True)
        self.show_obstacle_check.toggled.connect(self.refresh_view)
        self.form_3d.addRow(self.show_obstacle_check)

        vbox.addWidget(self.group_3d)

        # 5. Parametric Settings (populated dynamically for studies & parametric PINNs)
        self.group_parametric = QGroupBox("Parametric controls")
        self.form_parametric = QFormLayout(self.group_parametric)
        self.parametric_placeholder = QLabel("Load a parameter study or continuous parametric PINN to scrub parameter axes.")
        self.parametric_placeholder.setWordWrap(True)
        self.form_parametric.addRow(self.parametric_placeholder)
        vbox.addWidget(self.group_parametric)

        vbox.addStretch()
        scroll.setWidget(panel)

        self.settings_dock = QDockWidget("Display settings", self)
        self.settings_dock.setObjectName("explorer_settings_dock")
        self.settings_dock.setWidget(scroll)
        self.settings_dock.setMinimumWidth(300)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.settings_dock)

    def _build_probe_dock(self):
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(6, 4, 6, 4)

        top_row = QHBoxLayout()
        self.stats_label = QLabel("No solution loaded.")
        self.stats_label.setObjectName("explorer_stats_label")
        self.stats_label.setWordWrap(True)
        self.locate_max_error_button = QPushButton("Locate max |error|")
        self.locate_max_error_button.setObjectName("explorer_locate_max_error")
        self.locate_max_error_button.setEnabled(False)
        self.locate_max_error_button.clicked.connect(self.locate_max_error)
        top_row.addWidget(self.stats_label, 1)
        top_row.addWidget(self.locate_max_error_button)
        vbox.addLayout(top_row)

        self.probe_readout = QLabel("Hover over the 2D field to inspect coordinates and values, or use Drag line-cut probe.")
        self.probe_readout.setObjectName("explorer_probe_readout")
        vbox.addWidget(self.probe_readout)

        self.probe_plot = CurvePlot(logarithmic=False)
        self.probe_plot.setObjectName("explorer_probe_plot")
        self.probe_plot.setMinimumHeight(150)
        vbox.addWidget(self.probe_plot, 1)

        self.probe_dock = QDockWidget("Probe & diagnostics", self)
        self.probe_dock.setObjectName("explorer_probe_dock")
        self.probe_dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.probe_dock)

    def apply_theme(self):
        self.viewport_2d.update()
        self.colorbar_2d.update()
        self.colorbar_3d.update()
        self.view_1d.update()
        self.probe_plot.update()
        if self.viewport_3d is not None:
            self.viewport_3d.update()

    def sync_from_owner(self, preferred_id: str | None = None) -> bool:
        """Pull all available evaluated outputs from the owning Margam desktop window."""
        if self.owner is None:
            return False
        outputs = (self.owner.result or {}).get("outputs", {})
        if not outputs:
            self.statusBar().showMessage("No evaluated outputs in active run yet. Train or Evaluate outputs first.")
            return False
        live_model = (
            self.owner.last_run.model
            if self.owner.last_run is not None and self.owner.last_run.model is not None
            else None
        )
        titles = {n.identifier: n.title for n in self.owner.scene.nodes()}
        loaded: dict[str, SolutionDataset] = {}
        for oid, payload in outputs.items():
            loaded[oid] = from_output_result(payload, title=titles.get(oid, oid), live_model=live_model)
        self.set_datasets(loaded, preferred_id=preferred_id)
        return True

    def open_source_file(self, path: str | Path | None = None) -> bool:
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Open solution data or saved run",
                "",
                "All supported (*.margam-run.pt *.npz *.csv *.nc);;Margam run (*.margam-run.pt);;NumPy archive (*.npz);;CSV (*.csv);;NetCDF (*.nc)",
            )
            if not path:
                return False
        target = Path(path)
        try:
            if target.suffix.lower() == ".pt":
                loaded = from_run_archive(target)
                if not loaded:
                    raise ValueError("Selected run archive has no saved prediction outputs.")
                self.set_datasets(loaded)
            else:
                ds = from_file(target)
                self.set_dataset(ds)
        except Exception as exc:
            QMessageBox.warning(self, "Could not open solution source", str(exc))
            return False
        return True

    def open_study_folder(self, path: str | Path | None = None) -> bool:
        if path is None:
            path = QFileDialog.getExistingDirectory(self, "Open parameter study results folder")
            if not path:
                return False
        try:
            from ..visualization.loaders import StudyCollection
            collection = StudyCollection(path)
            self.set_study(collection)
        except Exception as exc:
            QMessageBox.warning(self, "Could not open study folder", str(exc))
            return False
        return True

    def _clear_parametric_panel(self):
        while self.form_parametric.rowCount():
            self.form_parametric.removeRow(0)
        self._study_axis_combos: list[QComboBox] = []
        self._study_diff_check = None
        self._study_baseline_combo = None
        self._study_resp_axis_combo = None
        self._study_resp_metric_combo = None
        self._study_status_label = None

    def set_study(self, collection):
        """Attach a StudyCollection and populate dynamic parametric scrubbers and response tools."""
        self.study = collection
        self._clear_parametric_panel()
        self._study_output_id = None
        if not collection.completed_indices:
            self.form_parametric.addRow(QLabel("No completed trials found in this study folder."))
            self.statusBar().showMessage(f"Loaded study {collection.root.name} · 0 completed trials")
            return

        # Populate available output IDs from the first completed trial
        first_idx = collection.completed_indices[0]
        output_ids = collection.output_ids_for_trial(first_idx)
        self._study_output_id = output_ids[0] if output_ids else None
        if output_ids:
            self._updating_controls = True
            self.output_combo.blockSignals(True)
            self.output_combo.clear()
            for oid in output_ids:
                self.output_combo.addItem(f"Output {oid[:8]}", oid)
            self.output_combo.blockSignals(False)
            self._updating_controls = False

        # One combo + step slider per swept parameter axis
        first_choices = collection.trial_choices(first_idx)
        for ax in collection.axes_meta:
            combo = QComboBox()
            combo.setObjectName(f"explorer_study_axis_{ax['index']}")
            for val in ax["values"]:
                combo.addItem(str(val), val)
            init_val = first_choices[ax["index"]]["value"]
            for i, val in enumerate(ax["values"]):
                if val == init_val:
                    combo.setCurrentIndex(i)
                    break
            combo.currentIndexChanged.connect(self._on_study_param_changed)
            if len(ax["values"]) > 1:
                sld = QSlider(Qt.Orientation.Horizontal)
                sld.setRange(0, len(ax["values"]) - 1)
                sld.setValue(combo.currentIndex())
                sld.valueChanged.connect(combo.setCurrentIndex)
                combo.currentIndexChanged.connect(sld.setValue)
                row_w = QWidget()
                rl = QHBoxLayout(row_w)
                rl.setContentsMargins(0, 0, 0, 0)
                rl.addWidget(sld, 1)
                rl.addWidget(combo)
                self.form_parametric.addRow(ax["label"], row_w)
            else:
                self.form_parametric.addRow(ax["label"], combo)
            self._study_axis_combos.append(combo)

        self._study_diff_check = QCheckBox("Compare difference vs baseline DP")
        self._study_diff_check.setObjectName("explorer_study_diff")
        self._study_diff_check.toggled.connect(self._on_study_param_changed)
        self.form_parametric.addRow(self._study_diff_check)

        self._study_baseline_combo = QComboBox()
        self._study_baseline_combo.setObjectName("explorer_study_baseline")
        for idx in collection.completed_indices:
            summary = ", ".join(f"{c['parameter']}={c['value']}" for c in collection.trial_choices(idx))
            self._study_baseline_combo.addItem(f"DP{idx + 1:04d} ({summary})", idx)
        self._study_baseline_combo.currentIndexChanged.connect(self._on_study_param_changed)
        self.form_parametric.addRow("Baseline DP", self._study_baseline_combo)

        self._study_resp_axis_combo = QComboBox()
        self._study_resp_axis_combo.setObjectName("explorer_study_resp_axis")
        for ax in collection.axes_meta:
            self._study_resp_axis_combo.addItem(ax["label"], ax["index"])
        self.form_parametric.addRow("Response axis", self._study_resp_axis_combo)

        self._study_resp_metric_combo = QComboBox()
        self._study_resp_metric_combo.setObjectName("explorer_study_resp_metric")
        for key, label in (
            ("max", "Spatial max"),
            ("min", "Spatial min"),
            ("mean", "Spatial mean"),
            ("l2", "Field L2"),
            ("rmse", "Reference RMSE"),
            ("rel_l2", "Reference rel L2"),
            ("max_error", "Reference max |error|"),
        ):
            self._study_resp_metric_combo.addItem(label, key)
        self.form_parametric.addRow("Response metric", self._study_resp_metric_combo)

        btn_resp = QPushButton("Plot parametric response")
        btn_resp.setObjectName("explorer_study_plot_response")
        btn_resp.clicked.connect(self.plot_study_response)
        self.form_parametric.addRow(btn_resp)

        self._study_status_label = QLabel("")
        self._study_status_label.setWordWrap(True)
        self.form_parametric.addRow(self._study_status_label)

        self._on_study_param_changed()
        self.statusBar().showMessage(
            f"Loaded study {collection.root.name} · {len(collection.completed_indices)} completed trial(s)"
        )

    def _selected_study_values(self) -> list[Any]:
        return [combo.currentData() for combo in getattr(self, "_study_axis_combos", [])]

    def _on_study_param_changed(self):
        if self.study is None or self._updating_controls:
            return
        values = self._selected_study_values()
        trial_idx = self.study.trial_for_values(values)
        if trial_idx is None:
            if self._study_status_label is not None:
                self._study_status_label.setText("No trial matches this parameter combination.")
            return
        status = self.study.trial_status(trial_idx)
        if status != "completed":
            if self._study_status_label is not None:
                self._study_status_label.setText(f"DP{trial_idx + 1:04d} status is '{status}' (not completed).")
            return
        oid = self.output_combo.currentData() or getattr(self, "_study_output_id", None)
        ds = self.study.load_trial_dataset(trial_idx, output_id=oid)
        if ds is None:
            return
        if self._study_diff_check is not None and self._study_diff_check.isChecked() and self._study_baseline_combo is not None:
            base_idx = self._study_baseline_combo.currentData()
            base_ds = self.study.load_trial_dataset(base_idx, output_id=oid) if base_idx is not None else None
            if base_ds is not None:
                try:
                    ds = ds.with_difference(base_ds, label=f"DP{trial_idx + 1:04d} − DP{base_idx + 1:04d}")
                except ValueError as exc:
                    if self._study_status_label is not None:
                        self._study_status_label.setText(str(exc))
                    return
        if self._study_status_label is not None:
            self._study_status_label.setText(f"Active: DP{trial_idx + 1:04d}")
        self.set_dataset(ds, preserve_lock=True)

    def plot_study_response(self):
        if self.study is None or self.dataset is None or self._study_resp_axis_combo is None:
            return [], []
        axis_index = self._study_resp_axis_combo.currentData()
        metric = self._study_resp_metric_combo.currentData() or "max"
        field_name = self.field_combo.currentText()
        oid = self.output_combo.currentData() or getattr(self, "_study_output_id", None)
        t_idx = self.time_slider.value() if self.dataset.is_time_dependent else 0
        xs_raw, ys = self.study.parametric_response(
            axis_index,
            self._selected_study_values(),
            field_name=field_name,
            metric=metric,
            output_id=oid,
            time_index=t_idx,
        )
        if not ys:
            self.probe_plot.set_series([], "No numeric response values for this axis/metric.")
            return [], []
        # Map numeric values directly or categorical values to integer indices
        try:
            xs_num = [float(v) for v in xs_raw]
        except (TypeError, ValueError):
            xs_num = list(range(len(xs_raw)))
        axis_label = self.study.axes_meta[axis_index]["label"]
        self.probe_plot.set_series(
            [(f"{field_name} ({metric})", list(zip(xs_num, ys)), "#2b9d91")],
            f"Parametric response · {field_name} {metric} vs {axis_label}",
        )
        return xs_raw, ys

    def set_datasets(self, datasets: dict[str, SolutionDataset], *, preferred_id: str | None = None):
        self.study = None
        self.datasets_by_id = dict(datasets)
        self._updating_controls = True
        self.output_combo.blockSignals(True)
        self.output_combo.clear()
        for oid, ds in self.datasets_by_id.items():
            self.output_combo.addItem(ds.title, oid)
        idx = self.output_combo.findData(preferred_id) if preferred_id else 0
        self.output_combo.setCurrentIndex(max(0, idx))
        self.output_combo.blockSignals(False)
        self._updating_controls = False
        chosen = self.output_combo.currentData()
        if chosen in self.datasets_by_id:
            self.set_dataset(self.datasets_by_id[chosen])

    def set_dataset(self, dataset: SolutionDataset, *, preserve_lock: bool = False):
        self.play_timer.stop()
        self.play_button.setChecked(False)
        self.dataset = dataset
        if not preserve_lock:
            self._locked_limits = None
            self.lock_range_check.setChecked(False)
        self._last_line_cut = None
        self.viewport_2d.clear_line_cut()
        self.viewport_2d.set_highlight_point(None)

        self._updating_controls = True
        prev_field = self.field_combo.currentText()
        self.field_combo.blockSignals(True)
        self.field_combo.clear()
        for fname in dataset.available_fields():
            self.field_combo.addItem(fname)
        if prev_field in dataset.available_fields():
            self.field_combo.setCurrentText(prev_field)
        self.field_combo.blockSignals(False)

        # Configure time bar
        if dataset.is_time_dependent and dataset.num_frames > 1:
            self.time_slider.blockSignals(True)
            self.time_slider.setRange(0, dataset.num_frames - 1)
            self.time_slider.setValue(0)
            self.time_slider.blockSignals(False)
            self.time_bar.show()
        else:
            self.time_bar.hide()

        # Enable vector overlays only when vector fields exist
        has_vec = dataset.has_vector_2d or dataset.has_vector_3d
        self.quiver_check.setEnabled(has_vec)
        self.quiver_stride_spin.setEnabled(has_vec)
        self.streamlines_check.setEnabled(has_vec)
        self.streamline_density_spin.setEnabled(has_vec)

        self._updating_controls = False
        self.setWindowTitle(f"Margam · Solution Explorer — {dataset.title}")
        self.refresh_view()

    def _on_output_selected(self):
        if self._updating_controls:
            return
        if self.study is not None:
            self._on_study_param_changed()
            return
        oid = self.output_combo.currentData()
        if oid in self.datasets_by_id:
            self.set_dataset(self.datasets_by_id[oid])

    def _on_range_toggle(self, auto_checked: bool):
        self.vmin_spin.setEnabled(not auto_checked)
        self.vmax_spin.setEnabled(not auto_checked)
        self.refresh_view()

    def _on_lock_range_toggled(self, locked: bool):
        if locked:
            if self.view_stack.currentIndex() == 2 and self.viewport_3d is not None:
                self._locked_limits = (self.viewport_3d.effective_vmin, self.viewport_3d.effective_vmax)
            else:
                self._locked_limits = (self.viewport_2d.effective_vmin, self.viewport_2d.effective_vmax)
        else:
            self._locked_limits = None
        self.refresh_view()

    def _on_line_cut_mode(self, enabled: bool):
        self.viewport_2d.line_cut_mode = bool(enabled)

    def _clear_line_cut(self):
        self._last_line_cut = None
        self.viewport_2d.clear_line_cut()
        self.probe_plot.set_series([], "Drag a line cut on the 2D field or choose Mid H-cut / Mid V-cut.")

    def _on_slice_axis_changed(self):
        self.refresh_view()

    def _slice_value_for_axis(self, axis: str) -> float | None:
        if self.dataset is None or self.dataset.spatial_dim != 3 or not self.dataset.spatial_bounds:
            return None
        idx = ("x", "y", "z").index(axis)
        lo, hi = self.dataset.spatial_bounds[idx]
        frac = self.slice_slider.value() / 100.0
        return lo + frac * (hi - lo)

    def active_view_mode(self) -> str:
        chosen = self.view_mode_combo.currentData() or "auto"
        if chosen != "auto":
            return chosen
        if self.dataset is None:
            return "2d"
        if self.dataset.spatial_dim == 1:
            return "1d"
        if self.dataset.spatial_dim >= 3 and self.viewport_3d is not None:
            return "3d"
        return "2d"

    def refresh_view(self):
        if self._updating_controls or self.dataset is None:
            return
        field_name = self.field_combo.currentText()
        if not field_name:
            return
        mode = self.display_mode_combo.currentData() or "prediction"
        has_ref = self.dataset.has_reference_for(field_name, self.time_slider.value())
        if mode != "prediction" and not has_ref:
            mode = "prediction"

        t_idx = self.time_slider.value() if self.dataset.is_time_dependent else 0
        if self.dataset.is_time_dependent:
            t_val = self.dataset.frame_time(t_idx)
            self.time_label.setText(f"Frame {t_idx + 1}/{self.dataset.num_frames} · t = {t_val:.5g}")

        view_mode = self.active_view_mode()
        slice_axis = self.slice_axis_combo.currentData() or "z"
        slice_val = self._slice_value_for_axis(slice_axis)
        if slice_val is not None:
            self.slice_value_label.setText(f"{slice_val:.4g}")

        self.group_3d.setVisible(self.dataset.spatial_dim >= 3 or view_mode == "3d")
        self.group_2d.setVisible(view_mode in ("2d", "compare") or (view_mode == "3d" and self.viewport_3d is None))

        if view_mode == "1d" or self.dataset.spatial_dim == 1:
            self.view_stack.setCurrentIndex(0)
            pts = self.dataset.spatial_points(t_idx)
            xs = pts[:, 0]
            order = np.argsort(xs)
            vals = self.dataset.field_values(field_name, mode=mode, time_index=t_idx)
            series = [(f"{field_name} ({mode})", list(zip(xs[order].tolist(), vals[order].tolist())), "#2b9d91")]
            if mode == "prediction" and has_ref:
                refs = self.dataset.field_values(field_name, mode="reference", time_index=t_idx)
                series.append(("Reference", list(zip(xs[order].tolist(), refs[order].tolist())), "#d58b35"))
            self.view_1d.set_logarithmic(self.log_scale_check.isChecked())
            self.view_1d.set_series(series, f"{self.dataset.title} · {field_name}")
        elif view_mode == "3d" and self.viewport_3d is not None:
            self.view_stack.setCurrentIndex(2)
            self._refresh_3d_viewport(field_name, mode, t_idx)
        else:
            self.view_stack.setCurrentIndex(1)
            auto_range = self.auto_range_check.isChecked() and self._locked_limits is None
            vmin = self._locked_limits[0] if self._locked_limits is not None else self.vmin_spin.value()
            vmax = self._locked_limits[1] if self._locked_limits is not None else self.vmax_spin.value()
            cmap = self.cmap_combo.currentData() or "viridis"
            self.viewport_2d.configure(
                dataset=self.dataset,
                field_name=field_name,
                display_mode=mode,
                time_index=t_idx,
                slice_axis=slice_axis,
                slice_value=slice_val,
                three_panel=(view_mode == "compare"),
                representation=self.repr_2d_combo.currentData() or "heatmap_contours",
                cmap=cmap,
                auto_range=auto_range,
                symmetric=self.symmetric_check.isChecked(),
                log_scale=self.log_scale_check.isChecked(),
                manual_vmin=vmin,
                manual_vmax=vmax,
                contour_levels=self.contour_levels_spin.value(),
                show_quiver=self.quiver_check.isChecked(),
                quiver_stride=self.quiver_stride_spin.value(),
                show_streamlines=self.streamlines_check.isChecked(),
                streamline_density=self.streamline_density_spin.value(),
                show_points=self.points_overlay_check.isChecked(),
            )
            eff_lo, eff_hi = self.viewport_2d.effective_vmin, self.viewport_2d.effective_vmax
            if self.auto_range_check.isChecked():
                self.vmin_spin.blockSignals(True)
                self.vmax_spin.blockSignals(True)
                self.vmin_spin.setValue(eff_lo)
                self.vmax_spin.setValue(eff_hi)
                self.vmin_spin.blockSignals(False)
                self.vmax_spin.blockSignals(False)
            bar_cmap = "coolwarm" if (mode == "signed_error" and cmap == "viridis") else cmap
            self.colorbar_2d.configure(bar_cmap, eff_lo, eff_hi, log_scale=self.log_scale_check.isChecked(), label=field_name)

        # Update diagnostics summary
        stats = self.dataset.statistics(field_name, mode=mode, time_index=t_idx)
        parts = [
            f"Field {field_name}",
            f"n={stats['count']}",
            f"min={stats['min']:.4g}",
            f"max={stats['max']:.4g}",
            f"mean={stats['mean']:.4g}",
            f"L2={stats['l2']:.4g}",
        ]
        if stats["rmse"] is not None:
            parts.append(f"RMSE={stats['rmse']:.4g}")
            parts.append(f"max|err|={stats['max_error']:.4g}")
            if stats["rel_l2"] is not None:
                parts.append(f"rel L2={stats['rel_l2']:.4g}")
        self.stats_label.setText(" · ".join(parts))
        self.locate_max_error_button.setEnabled(stats["max_error_point"] is not None)

        if self.viewport_2d.line_cut is not None:
            self._on_line_cut_changed(*self.viewport_2d.line_cut)

    def _refresh_3d_viewport(self, field_name: str, mode: str, t_idx: int):
        auto_range = self.auto_range_check.isChecked() and self._locked_limits is None
        vmin = self._locked_limits[0] if self._locked_limits is not None else self.vmin_spin.value()
        vmax = self._locked_limits[1] if self._locked_limits is not None else self.vmax_spin.value()
        cmap = self.cmap_combo.currentData() or "viridis"
        self.viewport_3d.configure(
            dataset=self.dataset,
            field_name=field_name,
            display_mode=mode,
            time_index=t_idx,
            mode_3d=self.mode_3d_combo.currentData() or "auto",
            cmap=cmap,
            auto_range=auto_range,
            symmetric=self.symmetric_check.isChecked(),
            log_scale=self.log_scale_check.isChecked(),
            manual_vmin=vmin,
            manual_vmax=vmax,
            show_slice_x=self.slice_3d_checks["x"].isChecked(),
            show_slice_y=self.slice_3d_checks["y"].isChecked(),
            show_slice_z=self.slice_3d_checks["z"].isChecked(),
            slice_x_frac=self.slice_3d_sliders["x"].value() / 100.0,
            slice_y_frac=self.slice_3d_sliders["y"].value() / 100.0,
            slice_z_frac=self.slice_3d_sliders["z"].value() / 100.0,
            iso_frac=self.iso_slider.value() / 100.0,
            iso_alpha=self.iso_opacity_slider.value(),
            point_size=self.cloud_point_spin.value(),
            threshold_min_frac=self.thresh_min_slider.value() / 100.0,
            threshold_max_frac=self.thresh_max_slider.value() / 100.0,
            clip_axis=self.clip_axis_combo.currentData() or "none",
            clip_frac=self.clip_slider.value() / 100.0,
            warp_scale=self.warp_scale_spin.value(),
            show_wireframe=self.wireframe_check.isChecked(),
            show_box=self.show_box_check.isChecked(),
            show_axes=self.show_box_check.isChecked(),
            show_obstacle=self.show_obstacle_check.isChecked(),
        )
        eff_lo, eff_hi = self.viewport_3d.effective_vmin, self.viewport_3d.effective_vmax
        if self.auto_range_check.isChecked():
            self.vmin_spin.blockSignals(True)
            self.vmax_spin.blockSignals(True)
            self.vmin_spin.setValue(eff_lo)
            self.vmax_spin.setValue(eff_hi)
            self.vmin_spin.blockSignals(False)
            self.vmax_spin.blockSignals(False)
        bar_cmap = "coolwarm" if (mode == "signed_error" and cmap == "viridis") else cmap
        self.colorbar_3d.configure(bar_cmap, eff_lo, eff_hi, log_scale=self.log_scale_check.isChecked(), label=field_name)

    def apply_preset_cut(self, orientation: str):
        if self.dataset is None or self.dataset.spatial_dim < 2 or not self.dataset.spatial_bounds:
            return
        if self.dataset.spatial_dim == 2:
            (x0, x1), (y0, y1) = self.dataset.spatial_bounds[:2]
        else:
            axis = self.slice_axis_combo.currentData() or "z"
            other = [i for i in range(3) if i != ("x", "y", "z").index(axis)]
            x0, x1 = self.dataset.spatial_bounds[other[0]]
            y0, y1 = self.dataset.spatial_bounds[other[1]]
        if orientation == "horizontal":
            ym = 0.5 * (y0 + y1)
            self.viewport_2d.set_line_cut((x0, ym), (x1, ym))
        else:
            xm = 0.5 * (x0 + x1)
            self.viewport_2d.set_line_cut((xm, y0), (xm, y1))

    def _on_line_cut_changed(self, p0: tuple[float, float], p1: tuple[float, float]):
        if self.dataset is None:
            return
        field_name = self.field_combo.currentText()
        if not field_name:
            return
        t_idx = self.time_slider.value() if self.dataset.is_time_dependent else 0
        mode = self.display_mode_combo.currentData() or "prediction"
        if mode != "prediction" and not self.dataset.has_reference_for(field_name, t_idx):
            mode = "prediction"
        if self.dataset.spatial_dim == 3:
            axis = self.slice_axis_combo.currentData() or "z"
            sv = float(self._slice_value_for_axis(axis) or 0.0)
            if axis == "z":
                c0, c1 = (p0[0], p0[1], sv), (p1[0], p1[1], sv)
            elif axis == "y":
                c0, c1 = (p0[0], sv, p0[1]), (p1[0], sv, p1[1])
            else:
                c0, c1 = (sv, p0[0], p0[1]), (sv, p1[0], p1[1])
        else:
            c0, c1 = p0, p1
        cut = self.dataset.sample_line(field_name, c0, c1, n_samples=120, mode=mode, time_index=t_idx)
        self._last_line_cut = cut
        s = cut["s"]
        vals = cut["values"]
        valid = np.isfinite(vals)
        series = [(f"{field_name} ({mode})", list(zip(s[valid].tolist(), vals[valid].tolist())), "#2b9d91")]
        if cut["reference"] is not None:
            rv = cut["reference"]
            vr = np.isfinite(rv)
            series.append(("Reference", list(zip(s[vr].tolist(), rv[vr].tolist())), "#d58b35"))
        caption = f"Line cut ({p0[0]:.3g}, {p0[1]:.3g}) → ({p1[0]:.3g}, {p1[1]:.3g}) · {field_name}"
        self.probe_plot.set_series(series, caption)

    def _on_probe_moved(self, info: dict[str, Any]):
        coord_str = ", ".join(f"{name}={val:.4g}" for name, val in zip(self.dataset.spatial_names, info["point"]))
        text = f"Probe ({coord_str}) · value={info['value']:.5g}"
        if info["reference"] is not None:
            text += f" · ref={info['reference']:.5g} · err={info['error']:.4g}"
        self.probe_readout.setText(text)
        self.statusBar().showMessage(text)

    def locate_max_error(self):
        if self.dataset is None:
            return None
        field_name = self.field_combo.currentText()
        t_idx = self.time_slider.value() if self.dataset.is_time_dependent else 0
        stats = self.dataset.statistics(field_name, time_index=t_idx)
        pt = stats.get("max_error_point")
        if pt is None:
            return None
        if len(pt) >= 2:
            self.viewport_2d.set_highlight_point((pt[0], pt[1]))
        coord_str = ", ".join(f"{n}={v:.4g}" for n, v in zip(self.dataset.spatial_names, pt))
        msg = f"Max |error| = {stats['max_error']:.5g} at ({coord_str})"
        self.probe_readout.setText(msg)
        self.statusBar().showMessage(msg)
        return pt

    def reset_camera(self):
        self.viewport_2d.reset_view()
        if self.viewport_3d is not None and hasattr(self.viewport_3d, "reset_camera"):
            self.viewport_3d.reset_camera()

    def _on_fps_changed(self):
        fps = int(self.fps_combo.currentData() or 5)
        self.play_timer.setInterval(max(30, int(round(1000.0 / fps))))

    def plot_time_trace(self, coord: tuple[float, ...] | None = None):
        if self.dataset is None or not self.dataset.is_time_dependent:
            return None
        field_name = self.field_combo.currentText()
        if not field_name:
            return None
        if coord is None:
            if self.viewport_2d._hover_data is not None and self.dataset.spatial_dim == 2:
                coord = self.viewport_2d._hover_data
            elif self.dataset.spatial_bounds:
                coord = tuple(0.5 * (lo + hi) for lo, hi in self.dataset.spatial_bounds)
            else:
                coord = tuple(0.0 for _ in range(self.dataset.spatial_dim))
        mode = self.display_mode_combo.currentData() or "prediction"
        if mode != "prediction" and not self.dataset.has_reference_for(field_name, 0):
            mode = "prediction"
        ts, vals, refs = self.dataset.time_trace(coord, field_name, mode=mode)
        if len(ts) == 0:
            return None
        series = [(f"{field_name} ({mode})", list(zip(ts.tolist(), vals.tolist())), "#2b9d91")]
        if refs is not None:
            series.append(("Reference", list(zip(ts.tolist(), refs.tolist())), "#d58b35"))
        c_str = ", ".join(f"{n}={v:.3g}" for n, v in zip(self.dataset.spatial_names, coord))
        self.probe_plot.set_series(series, f"Time trace at ({c_str}) · {field_name}")
        return ts, vals, refs

    def export_frame_sequence(self, directory: str | Path | None = None, *, scale: int = 1) -> list[Path]:
        if self.dataset is None:
            return []
        if directory is None:
            directory = QFileDialog.getExistingDirectory(self, "Choose folder for PNG frame sequence")
            if not directory:
                return []
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        if self.dataset.is_time_dependent and self.dataset.num_frames > 1:
            saved_idx = self.time_slider.value()
            try:
                for frame in range(self.dataset.num_frames):
                    self.set_time_index(frame)
                    out_file = out_dir / f"frame_{frame:04d}.png"
                    if self.export_snapshot(out_file, scale=scale):
                        written.append(out_file)
            finally:
                self.set_time_index(saved_idx)
        elif self.study is not None and getattr(self, "_study_axis_combos", None):
            axis_i = self._study_resp_axis_combo.currentData() if self._study_resp_axis_combo is not None else 0
            combo = self._study_axis_combos[axis_i]
            saved_idx = combo.currentIndex()
            try:
                for step in range(combo.count()):
                    combo.setCurrentIndex(step)
                    out_file = out_dir / f"param_{step:04d}.png"
                    if self.export_snapshot(out_file, scale=scale):
                        written.append(out_file)
            finally:
                combo.setCurrentIndex(saved_idx)
        else:
            out_file = out_dir / "frame_0000.png"
            if self.export_snapshot(out_file, scale=scale):
                written.append(out_file)
        self.statusBar().showMessage(f"Exported {len(written)} frame(s) to {out_dir.name}")
        return written

    def set_time_index(self, index: int):
        if self.dataset is None or not self.dataset.is_time_dependent:
            return
        clamped = max(0, min(int(index), self.dataset.num_frames - 1))
        self.time_slider.setValue(clamped)

    def _on_time_changed(self, _value: int):
        if not self._updating_controls:
            self.refresh_view()

    def _toggle_play(self, playing: bool):
        if playing and self.dataset is not None and self.dataset.is_time_dependent and self.dataset.num_frames > 1:
            self.play_button.setText("Pause")
            self.play_timer.start()
        else:
            self.play_button.setText("Play")
            self.play_timer.stop()

    def _advance_time_frame(self):
        if self.dataset is None or not self.dataset.is_time_dependent:
            self.play_button.setChecked(False)
            return
        nxt = self.time_slider.value() + 1
        if nxt >= self.dataset.num_frames:
            if self.loop_check.isChecked():
                nxt = 0
            else:
                self.play_button.setChecked(False)
                return
        self.time_slider.setValue(nxt)

    def export_snapshot(self, path: str | Path | None = None, *, scale: int = 2) -> bool:
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Export viewport snapshot", "solution.png", "PNG image (*.png)")
            if not path:
                return False
        target = Path(path)
        if target.suffix.lower() != ".png":
            target = target.with_suffix(target.suffix + ".png" if target.suffix else ".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        idx = self.view_stack.currentIndex()
        if idx == 1:
            img = self.viewport_2d.render_image(scale=scale)
        elif idx == 2 and self.viewport_3d is not None and hasattr(self.viewport_3d, "render_image"):
            img = self.viewport_3d.render_image(scale=scale)
        else:
            img = self.view_stack.currentWidget().grab().toImage()
        with NamedTemporaryFile(dir=target.parent, suffix=".png", delete=False) as handle:
            temp = Path(handle.name)
        try:
            if not img.save(str(temp), "PNG"):
                raise OSError("Could not save PNG snapshot.")
            os.replace(temp, target)
        except Exception:
            temp.unlink(missing_ok=True)
            raise
        self.statusBar().showMessage(f"Exported snapshot · {target.name}")
        return True

    def export_probe_csv(self, path: str | Path | None = None) -> bool:
        if self._last_line_cut is None:
            self.statusBar().showMessage("Create a 1D line cut first before exporting probe CSV.")
            return False
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Export probe line CSV", "probe_line.csv", "CSV (*.csv)")
            if not path:
                return False
        target = Path(path)
        if target.suffix.lower() != ".csv":
            target = target.with_suffix(target.suffix + ".csv" if target.suffix else ".csv")
        target.parent.mkdir(parents=True, exist_ok=True)
        cut = self._last_line_cut
        coord_names = list(self.dataset.spatial_names) if self.dataset is not None else ["x", "y"]
        headers = ["s"] + coord_names + ["value"] + (["reference"] if cut["reference"] is not None else [])
        lines = [",".join(headers)]
        for i in range(len(cut["s"])):
            row = [f"{cut['s'][i]:.10g}"] + [f"{v:.10g}" for v in cut["coords"][i]] + [f"{cut['values'][i]:.10g}"]
            if cut["reference"] is not None:
                row.append(f"{cut['reference'][i]:.10g}")
            lines.append(",".join(row))
        with NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
            temp = Path(handle.name)
            handle.write("\n".join(lines) + "\n")
        os.replace(temp, target)
        self.statusBar().showMessage(f"Exported probe CSV · {target.name}")
        return True

    def closeEvent(self, event):
        self.play_timer.stop()
        super().closeEvent(event)


def launch_solution_explorer(source=None, *, block: bool = True, owner=None, **kwargs):
    """Open the dedicated Solution Explorer window from Python or a running desktop session."""
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication([])
        configure_theme(app, "light")

    dataset = None
    study_path = None
    if isinstance(source, SolutionDataset):
        dataset = source
    elif isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_dir():
            study_path = p
        else:
            dataset = from_file(p)
    elif isinstance(source, dict) and "points" in source and "config" in source:
        dataset = from_output_result(source)

    window = SolutionExplorerWindow(owner=owner, dataset=dataset)
    if study_path is not None:
        window.open_study_folder(study_path)
    window.show()
    window.raise_()
    if block and owns_app:
        app.exec()
    return window
