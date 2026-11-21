"""Interactive 2D scientific field viewport with contours, vectors, streamlines, and probes."""
from __future__ import annotations

import math
from typing import Any
import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from .theme import colors
from ..visualization.colormaps import colormap_lut, map_scalars_to_rgba
from ..visualization.field_ops import compute_streamlines_2d, marching_squares


def _rgba_to_qimage(rgba_ny_nx: np.ndarray) -> QImage:
    """Convert an ``(ny, nx, 4)`` uint8 RGBA array (row 0 = ymin) into a flipped QImage."""
    # Flip vertically so row 0 of QImage corresponds to ymax (top of viewport)
    flipped = np.ascontiguousarray(rgba_ny_nx[::-1, :, :])
    h, w, _ = flipped.shape
    img = QImage(flipped.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
    return img.copy()


class ColorbarWidget(QWidget):
    """Vertical scientific colorbar legend synced with active colormap and limits."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cmap = "viridis"
        self.vmin = 0.0
        self.vmax = 1.0
        self.log_scale = False
        self.label = ""
        self.setFixedWidth(88)
        self.setMinimumHeight(140)

    def configure(self, cmap: str, vmin: float, vmax: float, *, log_scale: bool = False, label: str = ""):
        self.cmap = cmap
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.log_scale = bool(log_scale)
        self.label = label
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = colors()
        p.fillRect(self.rect(), QColor(c["base"]))
        p.setPen(QColor(c["text"]))
        top = 28
        bottom = max(top + 40, self.height() - 18)
        bar = QRectF(10, top, 18, bottom - top)
        lut = colormap_lut(self.cmap)
        strip = np.empty((256, 1, 4), dtype=np.uint8)
        strip[:, 0, :3] = lut[::-1]
        strip[:, 0, 3] = 255
        img = QImage(strip.data, 1, 256, 4, QImage.Format.Format_RGBA8888).copy()
        p.drawImage(bar, img)
        p.setPen(QPen(QColor(c["border"]), 1))
        p.drawRect(bar)
        p.setPen(QColor(c["text"]))
        if self.label:
            p.drawText(QRectF(4, 4, self.width() - 8, 20), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.label)
        lo, hi = self.vmin, self.vmax
        if self.log_scale and lo > 0 and hi > 0:
            ticks = [hi, math.sqrt(lo * hi), lo]
        else:
            ticks = [hi, 0.5 * (lo + hi), lo]
        ys = [bar.top() + 6, bar.center().y() + 4, bar.bottom()]
        for val, py in zip(ticks, ys):
            p.drawText(int(bar.right() + 6), int(py), f"{val:.3g}")


class Viewport2D(QWidget):
    """Interactive 2D field renderer supporting single-field and 3-panel comparison views."""

    probeMoved = Signal(object)
    lineCutChanged = Signal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(320, 260)

        self.dataset = None
        self.field_name = ""
        self.display_mode = "prediction"
        self.time_index = 0
        self.slice_axis = "z"
        self.slice_value: float | None = None
        self.three_panel = False

        # Rendering options
        self.representation = "heatmap_contours"  # heatmap | filled_contours | contours | heatmap_contours | points
        self.cmap = "viridis"
        self.error_cmap = "coolwarm"
        self.auto_range = True
        self.symmetric = False
        self.log_scale = False
        self.manual_vmin: float | None = None
        self.manual_vmax: float | None = None
        self.contour_levels = 12
        self.show_quiver = False
        self.quiver_stride = 4
        self.show_streamlines = False
        self.streamline_density = 8
        self.show_points = False
        self.resolution = 96

        # Interactive camera & probe state
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.line_cut_mode = False
        self.line_cut: tuple[tuple[float, float], tuple[float, float]] | None = None
        self.highlight_point: tuple[float, float] | None = None
        self._dragging_pan = False
        self._dragging_cut = False
        self._last_pos = QPointF()
        self._hover_data: tuple[float, float] | None = None

        # Cached rendered data
        self._panels: list[dict[str, Any]] = []
        self._panel_rects: list[QRectF] = []
        self.effective_vmin = 0.0
        self.effective_vmax = 1.0
        self._error_message = ""

    def reset_view(self):
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()

    def set_line_cut(self, p0: tuple[float, float] | None, p1: tuple[float, float] | None):
        if p0 is None or p1 is None:
            self.line_cut = None
        else:
            self.line_cut = ((float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1])))
            self.lineCutChanged.emit(self.line_cut[0], self.line_cut[1])
        self.update()

    def clear_line_cut(self):
        self.line_cut = None
        self.update()

    def set_highlight_point(self, pt: tuple[float, float] | None):
        self.highlight_point = (float(pt[0]), float(pt[1])) if pt is not None else None
        self.update()

    def configure(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.rebuild()

    def rebuild(self):
        self._panels = []
        self._error_message = ""
        if self.dataset is None or not self.field_name:
            self.update()
            return
        try:
            if self.three_panel and self.dataset.has_reference_for(self.field_name, self.time_index):
                specs = [
                    ("Prediction", "prediction", self.cmap, False),
                    ("Reference", "reference", self.cmap, False),
                    ("Pred − Ref", "signed_error", self.error_cmap, True),
                ]
                # Shared color scale between Prediction and Reference
                xs, ys, g_pred, labels = self.dataset.to_2d_grid(
                    self.field_name, mode="prediction", time_index=self.time_index,
                    resolution=self.resolution, slice_axis=self.slice_axis, slice_value=self.slice_value,
                )
                _, _, g_ref, _ = self.dataset.to_2d_grid(
                    self.field_name, mode="reference", time_index=self.time_index,
                    resolution=self.resolution, slice_axis=self.slice_axis, slice_value=self.slice_value,
                )
                g_err = g_pred - g_ref
                finite_pr = np.concatenate([g_pred[np.isfinite(g_pred)], g_ref[np.isfinite(g_ref)]])
                shared_lo = float(np.min(finite_pr)) if len(finite_pr) else 0.0
                shared_hi = float(np.max(finite_pr)) if len(finite_pr) else 1.0
                if not self.auto_range and self.manual_vmin is not None and self.manual_vmax is not None:
                    shared_lo, shared_hi = self.manual_vmin, self.manual_vmax
                grids = [g_pred, g_ref, g_err]
                for (title, mode, cmap_name, is_err), grid in zip(specs, grids):
                    panel = self._build_panel(
                        title, xs, ys, grid, labels,
                        cmap=cmap_name,
                        vmin=None if is_err else shared_lo,
                        vmax=None if is_err else shared_hi,
                        symmetric=True if is_err else self.symmetric,
                        log_scale=False if is_err else self.log_scale,
                        include_vectors=(mode == "prediction"),
                    )
                    self._panels.append(panel)
                self.effective_vmin, self.effective_vmax = shared_lo, shared_hi
            else:
                mode = self.display_mode
                if mode != "prediction" and not self.dataset.has_reference_for(self.field_name, self.time_index):
                    mode = "prediction"
                xs, ys, grid, labels = self.dataset.to_2d_grid(
                    self.field_name, mode=mode, time_index=self.time_index,
                    resolution=self.resolution, slice_axis=self.slice_axis, slice_value=self.slice_value,
                )
                cmap_name = self.error_cmap if mode == "signed_error" and self.cmap == "viridis" else self.cmap
                vmin = None if self.auto_range else self.manual_vmin
                vmax = None if self.auto_range else self.manual_vmax
                title = f"{self.field_name}" if mode == "prediction" else f"{self.field_name} ({mode.replace('_', ' ')})"
                if self.dataset.spatial_dim == 3:
                    val = self.slice_value if self.slice_value is not None else 0.0
                    title += f" · {self.slice_axis}={val:.3g}"
                panel = self._build_panel(
                    title, xs, ys, grid, labels,
                    cmap=cmap_name, vmin=vmin, vmax=vmax,
                    symmetric=self.symmetric or (mode == "signed_error" and self.auto_range),
                    log_scale=self.log_scale,
                    include_vectors=True,
                )
                self._panels.append(panel)
                self.effective_vmin = panel["vmin"]
                self.effective_vmax = panel["vmax"]
        except Exception as exc:
            self._error_message = str(exc)
        self.update()

    def _build_panel(
        self,
        title: str,
        xs: np.ndarray,
        ys: np.ndarray,
        grid: np.ndarray,
        labels: tuple[str, str],
        *,
        cmap: str,
        vmin: float | None,
        vmax: float | None,
        symmetric: bool,
        log_scale: bool,
        include_vectors: bool,
    ) -> dict[str, Any]:
        # Quantize values for filled contours mode
        render_grid = grid
        if self.representation == "filled_contours":
            finite = grid[np.isfinite(grid)]
            if len(finite):
                lo = float(np.min(finite)) if vmin is None else float(vmin)
                hi = float(np.max(finite)) if vmax is None else float(vmax)
                if hi > lo:
                    n_lvl = max(3, int(self.contour_levels))
                    norm = np.clip((grid - lo) / (hi - lo), 0.0, 1.0)
                    render_grid = np.where(np.isfinite(grid), lo + np.floor(norm * n_lvl) / n_lvl * (hi - lo), np.nan)

        rgba, eff_lo, eff_hi = map_scalars_to_rgba(
            render_grid, cmap=cmap, vmin=vmin, vmax=vmax, symmetric=symmetric, log_scale=log_scale
        )
        image = _rgba_to_qimage(rgba)

        contours = []
        if self.representation in ("contours", "heatmap_contours", "filled_contours") and eff_hi > eff_lo:
            n_lvl = max(2, int(self.contour_levels))
            if log_scale and eff_lo > 0 and eff_hi > 0:
                levels = np.geomspace(eff_lo, eff_hi, n_lvl + 2)[1:-1]
            else:
                levels = np.linspace(eff_lo, eff_hi, n_lvl + 2)[1:-1]
            contours = marching_squares(xs, ys, grid, levels)

        quiver = None
        streamlines = []
        if include_vectors and (self.show_quiver or self.show_streamlines):
            vf = self.dataset.vector_field_2d(
                mode="prediction", time_index=self.time_index,
                resolution=min(48, self.resolution),
                slice_axis=self.slice_axis if self.dataset.spatial_dim == 3 else None,
                slice_value=self.slice_value,
            )
            if vf is not None:
                vxs, vys, vx, vy = vf
                if self.show_quiver:
                    quiver = (vxs, vys, vx, vy)
                if self.show_streamlines:
                    streamlines = compute_streamlines_2d(vxs, vys, vx, vy, density=self.streamline_density)

        scatter_pts = None
        scatter_rgba = None
        if self.show_points or self.representation == "points":
            pts = self.dataset.spatial_points(self.time_index)
            if self.dataset.spatial_dim == 2:
                scatter_pts = pts
                vals = self.dataset.field_values(self.field_name, mode="prediction", time_index=self.time_index)
                scatter_rgba, _, _ = map_scalars_to_rgba(vals, cmap=cmap, vmin=eff_lo, vmax=eff_hi, symmetric=symmetric, log_scale=log_scale)

        return {
            "title": title,
            "xs": xs,
            "ys": ys,
            "grid": grid,
            "labels": labels,
            "image": image,
            "vmin": eff_lo,
            "vmax": eff_hi,
            "contours": contours,
            "quiver": quiver,
            "streamlines": streamlines,
            "scatter_pts": scatter_pts,
            "scatter_rgba": scatter_rgba,
        }

    def _compute_panel_rects(self, total_rect: QRectF) -> list[QRectF]:
        n = len(self._panels)
        if n == 0:
            return []
        pad_l, pad_r, pad_t, pad_b = 54.0, 22.0, 34.0, 34.0
        gap = 28.0
        avail_w = max(80.0, total_rect.width() - pad_l - pad_r - gap * (n - 1))
        avail_h = max(80.0, total_rect.height() - pad_t - pad_b)
        slot_w = avail_w / n
        rects = []
        for idx, panel in enumerate(self._panels):
            xs, ys = panel["xs"], panel["ys"]
            dx = max(1e-9, float(xs[-1] - xs[0]))
            dy = max(1e-9, float(ys[-1] - ys[0]))
            aspect = dx / dy
            if slot_w / avail_h > aspect:
                h = avail_h
                w = h * aspect
            else:
                w = slot_w
                h = w / aspect
            cx = total_rect.left() + pad_l + idx * (slot_w + gap) + slot_w * 0.5
            cy = total_rect.top() + pad_t + avail_h * 0.5
            rects.append(QRectF(cx - w * 0.5, cy - h * 0.5, w, h))
        return rects

    def _data_to_pixel(self, rect: QRectF, xs: np.ndarray, ys: np.ndarray, x: float, y: float) -> QPointF:
        x0, x1 = float(xs[0]), float(xs[-1])
        y0, y1 = float(ys[0]), float(ys[-1])
        u = (x - x0) / max(1e-12, x1 - x0)
        v = (y - y0) / max(1e-12, y1 - y0)
        # Apply interactive zoom & pan around rect center
        ux = 0.5 + (u - 0.5) * self.zoom + self.pan_x
        vy = 0.5 + (v - 0.5) * self.zoom + self.pan_y
        return QPointF(rect.left() + ux * rect.width(), rect.bottom() - vy * rect.height())

    def _pixel_to_data(self, rect: QRectF, xs: np.ndarray, ys: np.ndarray, pos: QPointF) -> tuple[float, float]:
        ux = (pos.x() - rect.left()) / max(1.0, rect.width())
        vy = (rect.bottom() - pos.y()) / max(1.0, rect.height())
        u = 0.5 + (ux - 0.5 - self.pan_x) / max(1e-6, self.zoom)
        v = 0.5 + (vy - 0.5 - self.pan_y) / max(1e-6, self.zoom)
        x = float(xs[0]) + u * float(xs[-1] - xs[0])
        y = float(ys[0]) + v * float(ys[-1] - ys[0])
        return x, y

    def paint_to(self, painter: QPainter, target_rect: QRectF):
        c = colors()
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(target_rect, QColor(c["canvas"]))
        if self._error_message:
            painter.setPen(QColor(c["text"]))
            painter.drawText(target_rect.adjusted(20, 20, -20, -20), Qt.TextFlag.TextWordWrap, self._error_message)
            painter.restore()
            return
        if not self._panels:
            painter.setPen(QColor(c["muted"]))
            painter.drawText(target_rect, Qt.AlignmentFlag.AlignCenter, "Load a run, study, or 2D/3D field to inspect the solution.")
            painter.restore()
            return

        rects = self._compute_panel_rects(target_rect)
        self._panel_rects = rects
        for panel, rect in zip(self._panels, rects):
            xs, ys = panel["xs"], panel["ys"]
            x_lbl, y_lbl = panel["labels"]
            painter.setPen(QColor(c["text"]))
            painter.drawText(QRectF(rect.left(), rect.top() - 26, rect.width(), 22), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, panel["title"])

            # Clipped data region
            painter.save()
            painter.setClipRect(rect)
            painter.fillRect(rect, QColor(c["base"]))
            p_tl = self._data_to_pixel(rect, xs, ys, float(xs[0]), float(ys[-1]))
            p_br = self._data_to_pixel(rect, xs, ys, float(xs[-1]), float(ys[0]))
            img_rect = QRectF(p_tl, p_br).normalized()

            if self.representation in ("heatmap", "filled_contours", "heatmap_contours"):
                painter.drawImage(img_rect, panel["image"])

            if panel["contours"]:
                pen_color = QColor("#182026") if self.representation != "contours" else QColor(c["text"])
                pen_color.setAlpha(150 if self.representation != "contours" else 220)
                painter.setPen(QPen(pen_color, 1.0))
                for _, segs in panel["contours"]:
                    for seg in segs:
                        a = self._data_to_pixel(rect, xs, ys, float(seg[0, 0]), float(seg[0, 1]))
                        b = self._data_to_pixel(rect, xs, ys, float(seg[1, 0]), float(seg[1, 1]))
                        painter.drawLine(a, b)

            if panel["streamlines"]:
                s_pen = QPen(QColor("#ffffff" if self.representation != "contours" else c["highlight"]), 1.35)
                painter.setPen(s_pen)
                for poly in panel["streamlines"]:
                    path = QPainterPath()
                    for idx, pt in enumerate(poly):
                        qpt = self._data_to_pixel(rect, xs, ys, float(pt[0]), float(pt[1]))
                        if idx == 0:
                            path.moveTo(qpt)
                        else:
                            path.lineTo(qpt)
                    painter.drawPath(path)

            if panel["quiver"] is not None:
                vxs, vys, vx, vy = panel["quiver"]
                stride = max(1, int(self.quiver_stride))
                spd = np.hypot(vx, vy)
                finite = spd[np.isfinite(spd)]
                max_spd = float(np.max(finite)) if len(finite) else 1.0
                if max_spd > 1e-12:
                    arrow_len = 0.045 * min(rect.width(), rect.height()) * max(0.5, min(2.0, self.zoom))
                    painter.setPen(QPen(QColor("#111111" if self.representation != "contours" else c["text"]), 1.2))
                    for j in range(0, len(vys), stride):
                        for i in range(0, len(vxs), stride):
                            u_val, v_val = vx[j, i], vy[j, i]
                            if not (np.isfinite(u_val) and np.isfinite(v_val)):
                                continue
                            mag = math.hypot(u_val, v_val)
                            if mag < 1e-6 * max_spd:
                                continue
                            p0 = self._data_to_pixel(rect, xs, ys, float(vxs[i]), float(vys[j]))
                            scale = arrow_len * (mag / max_spd)
                            dx_px = (u_val / mag) * scale
                            dy_px = -(v_val / mag) * scale
                            p1 = QPointF(p0.x() + dx_px, p0.y() + dy_px)
                            painter.drawLine(p0, p1)
                            # Small arrowhead
                            ang = math.atan2(dy_px, dx_px)
                            hl = max(3.0, 0.28 * scale)
                            for side in (-2.5, 2.5):
                                hx = p1.x() + hl * math.cos(ang + side)
                                hy = p1.y() + hl * math.sin(ang + side)
                                painter.drawLine(p1, QPointF(hx, hy))

            if panel["scatter_pts"] is not None and panel["scatter_rgba"] is not None:
                pts = panel["scatter_pts"]
                srgba = panel["scatter_rgba"]
                step = max(1, len(pts) // 2500)
                radius = 3.2 if self.representation == "points" else 2.0
                painter.setPen(Qt.PenStyle.NoPen)
                for idx in range(0, len(pts), step):
                    qpt = self._data_to_pixel(rect, xs, ys, float(pts[idx, 0]), float(pts[idx, 1]))
                    if self.representation == "points":
                        r, g, b, a = (int(v) for v in srgba[idx])
                        painter.setBrush(QColor(r, g, b, a))
                    else:
                        painter.setBrush(QColor(20, 20, 20, 140))
                    painter.drawEllipse(qpt, radius, radius)

            # Draw obstacle boundary circle if applicable
            obs = self.dataset.obstacle if self.dataset is not None else None
            if obs is not None:
                kind = obs.get("kind")
                cx_o = cy_o = r_o = None
                if kind == "cylinder" and self.dataset.spatial_dim == 2:
                    cx_o, cy_o = obs["center"][:2]
                    r_o = float(obs["radius"])
                elif kind == "sphere" and self.dataset.spatial_dim == 3:
                    scx, scy, scz = obs["center"]
                    sr = float(obs["radius"])
                    sval = float(self.slice_value or 0.0)
                    if self.slice_axis == "z":
                        d = abs(sval - scz)
                        if d < sr:
                            cx_o, cy_o, r_o = scx, scy, math.sqrt(sr * sr - d * d)
                    elif self.slice_axis == "y":
                        d = abs(sval - scy)
                        if d < sr:
                            cx_o, cy_o, r_o = scx, scz, math.sqrt(sr * sr - d * d)
                    elif self.slice_axis == "x":
                        d = abs(sval - scx)
                        if d < sr:
                            cx_o, cy_o, r_o = scy, scz, math.sqrt(sr * sr - d * d)
                if r_o is not None and r_o > 0:
                    c_px = self._data_to_pixel(rect, xs, ys, cx_o, cy_o)
                    e_px = self._data_to_pixel(rect, xs, ys, cx_o + r_o, cy_o)
                    n_px = self._data_to_pixel(rect, xs, ys, cx_o, cy_o + r_o)
                    rx = abs(e_px.x() - c_px.x())
                    ry = abs(n_px.y() - c_px.y())
                    painter.setBrush(QColor(c["panel"]))
                    painter.setPen(QPen(QColor(c["text"]), 1.5))
                    painter.drawEllipse(c_px, rx, ry)

            # Draw active 1D line cut overlay
            if self.line_cut is not None:
                a = self._data_to_pixel(rect, xs, ys, self.line_cut[0][0], self.line_cut[0][1])
                b = self._data_to_pixel(rect, xs, ys, self.line_cut[1][0], self.line_cut[1][1])
                painter.setPen(QPen(QColor("#ff3b30"), 2.0, Qt.PenStyle.DashLine))
                painter.drawLine(a, b)
                painter.setBrush(QColor("#ff3b30"))
                painter.drawEllipse(a, 4, 4)
                painter.drawEllipse(b, 4, 4)

            # Draw max-|error| highlight marker if set
            if self.highlight_point is not None:
                hp = self._data_to_pixel(rect, xs, ys, self.highlight_point[0], self.highlight_point[1])
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor("#ff2d55"), 2.0))
                painter.drawEllipse(hp, 7, 7)
                painter.drawLine(QPointF(hp.x() - 10, hp.y()), QPointF(hp.x() + 10, hp.y()))
                painter.drawLine(QPointF(hp.x(), hp.y() - 10), QPointF(hp.x(), hp.y() + 10))

            painter.restore()

            # Panel border & axis ticks
            painter.setPen(QPen(QColor(c["border"]), 1.2))
            painter.drawRect(rect)
            painter.setPen(QColor(c["text"]))
            painter.drawText(int(rect.left()), int(rect.bottom() + 18), f"{float(xs[0]):.3g}")
            painter.drawText(int(rect.right() - 32), int(rect.bottom() + 18), f"{float(xs[-1]):.3g}")
            painter.drawText(int(rect.center().x() - 8), int(rect.bottom() + 18), x_lbl)
            painter.drawText(int(rect.left() - 46), int(rect.top() + 10), f"{float(ys[-1]):.3g}")
            painter.drawText(int(rect.left() - 46), int(rect.bottom()), f"{float(ys[0]):.3g}")
            painter.drawText(int(rect.left() - 34), int(rect.center().y() + 4), y_lbl)

        painter.restore()

    def paintEvent(self, event):
        p = QPainter(self)
        self.paint_to(p, QRectF(self.rect()))

    def render_image(self, scale: int = 2) -> QImage:
        w = max(1, int(self.width() * scale))
        h = max(1, int(self.height() * scale))
        img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        p = QPainter(img)
        p.scale(scale, scale)
        self.paint_to(p, QRectF(0, 0, self.width(), self.height()))
        p.end()
        return img

    def _hit_panel(self, pos: QPointF):
        for panel, rect in zip(self._panels, self._panel_rects):
            if rect.contains(pos):
                return panel, rect
        return None, None

    def wheelEvent(self, event):
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if delta:
            factor = 1.15 ** (delta / 120.0)
            self.zoom = max(0.5, min(12.0, self.zoom * factor))
            self.update()
        event.accept()

    def mousePressEvent(self, event):
        pos = event.position()
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and (event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        ):
            self._dragging_pan = True
            self._last_pos = pos
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self.line_cut_mode:
            panel, rect = self._hit_panel(pos)
            if panel is not None:
                p0 = self._pixel_to_data(rect, panel["xs"], panel["ys"], pos)
                self._dragging_cut = True
                self.line_cut = (p0, p0)
                self.update()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = event.position()
        if self._dragging_pan and self._panel_rects:
            rect = self._panel_rects[0]
            delta = pos - self._last_pos
            self.pan_x += delta.x() / max(1.0, rect.width())
            self.pan_y -= delta.y() / max(1.0, rect.height())
            self._last_pos = pos
            self.update()
            event.accept()
            return
        panel, rect = self._hit_panel(pos)
        if self._dragging_cut and panel is not None and self.line_cut is not None:
            p1 = self._pixel_to_data(rect, panel["xs"], panel["ys"], pos)
            self.line_cut = (self.line_cut[0], p1)
            self.lineCutChanged.emit(self.line_cut[0], self.line_cut[1])
            self.update()
            event.accept()
            return
        if panel is not None and self.dataset is not None:
            dx, dy = self._pixel_to_data(rect, panel["xs"], panel["ys"], pos)
            self._hover_data = (dx, dy)
            if self.dataset.spatial_dim == 2:
                coord = (dx, dy)
            elif self.dataset.spatial_dim == 3:
                sv = float(self.slice_value or 0.0)
                coord = {"z": (dx, dy, sv), "y": (dx, sv, dy), "x": (sv, dx, dy)}.get(self.slice_axis, (dx, dy, sv))
            else:
                coord = None
            if coord is not None:
                info = self.dataset.probe_point(coord, self.field_name, mode=self.display_mode if self.dataset.has_reference_for(self.field_name, self.time_index) else "prediction", time_index=self.time_index)
                if info is not None:
                    self.probeMoved.emit(info)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging_pan:
            self._dragging_pan = False
            event.accept()
            return
        if self._dragging_cut:
            self._dragging_cut = False
            if self.line_cut is not None:
                self.lineCutChanged.emit(self.line_cut[0], self.line_cut[1])
            event.accept()
            return
        super().mouseReleaseEvent(event)
