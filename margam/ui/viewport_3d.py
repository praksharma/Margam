"""Interactive 3D scientific viewport for surfaces, slices, isosurfaces, and thresholded clouds."""
from __future__ import annotations

import math
from typing import Any
import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from .theme import colors
from ..visualization.colormaps import colormap_lut, map_scalars_to_rgba
from ..visualization.geometry_3d import (
    Camera3D,
    build_box_edges,
    build_heightfield_mesh,
    build_obstacle_mesh,
    build_slice_mesh,
    marching_tetrahedra,
    normalize_bounds,
    normalize_points,
)


class Viewport3D(QWidget):
    """Interactive 3D orbit/pan/zoom viewport with deterministic depth-sorted shading."""

    cameraChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 260)
        self.camera = Camera3D()

        self.dataset = None
        self.field_name = ""
        self.display_mode = "prediction"
        self.time_index = 0

        # 3D mode & appearance settings
        self.mode_3d = "auto"  # auto | heightfield | slices | cloud | isosurface | slices_iso
        self.cmap = "viridis"
        self.auto_range = True
        self.symmetric = False
        self.log_scale = False
        self.manual_vmin: float | None = None
        self.manual_vmax: float | None = None

        # Slice planes (fractional 0..1 along x, y, z)
        self.show_slice_x = False
        self.show_slice_y = False
        self.show_slice_z = True
        self.slice_x_frac = 0.5
        self.slice_y_frac = 0.5
        self.slice_z_frac = 0.5

        # Isosurface settings
        self.iso_frac = 0.5
        self.iso_alpha = 200

        # Point cloud settings
        self.point_size = 4.0
        self.cloud_alpha = 215
        self.threshold_min_frac = 0.0
        self.threshold_max_frac = 1.0
        self.clip_axis = "none"  # none | x | y | z
        self.clip_frac = 1.0

        # Surface & guides
        self.warp_scale = 0.55
        self.show_wireframe = False
        self.show_box = True
        self.show_axes = True
        self.show_obstacle = True

        # Cached geometry
        self._meshes: list[dict[str, Any]] = []
        self._cloud_pts: np.ndarray | None = None
        self._cloud_rgba: np.ndarray | None = None
        self._box_edges: np.ndarray | None = None
        self._bounds: list[tuple[float, float]] = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
        self._center = np.zeros(3, dtype=np.float64)
        self._scale = 1.0
        self.effective_vmin = 0.0
        self.effective_vmax = 1.0
        self._error_message = ""

        # Pointer interaction state
        self._dragging_orbit = False
        self._dragging_pan = False
        self._last_pos = QPointF()

    def reset_camera(self):
        self.camera.reset()
        self.cameraChanged.emit()
        self.update()

    def set_preset(self, preset: str):
        self.camera.set_preset(preset)
        self.cameraChanged.emit()
        self.update()

    def set_orthographic(self, enabled: bool):
        self.camera.orthographic = bool(enabled)
        self.cameraChanged.emit()
        self.update()

    def configure(self, **kwargs):
        for key, value in kwargs.items():
            if key == "orthographic":
                self.camera.orthographic = bool(value)
            elif hasattr(self, key):
                setattr(self, key, value)
        self.rebuild()

    def _effective_mode(self) -> str:
        if self.dataset is None:
            return "slices"
        if self.mode_3d != "auto":
            return self.mode_3d
        if self.dataset.spatial_dim == 2:
            return "heightfield"
        return "slices" if self.dataset.is_regular_grid(self.time_index) else "cloud"

    def rebuild(self):
        self._meshes = []
        self._cloud_pts = None
        self._cloud_rgba = None
        self._box_edges = None
        self._error_message = ""
        if self.dataset is None or not self.field_name:
            self.update()
            return
        try:
            mode = self.display_mode
            if mode != "prediction" and not self.dataset.has_reference_for(self.field_name, self.time_index):
                mode = "prediction"
            vmin = None if self.auto_range else self.manual_vmin
            vmax = None if self.auto_range else self.manual_vmax
            sym = self.symmetric or (mode == "signed_error" and self.auto_range)
            cmap = "coolwarm" if (mode == "signed_error" and self.cmap == "viridis") else self.cmap

            vals_all = self.dataset.field_values(self.field_name, mode=mode, time_index=self.time_index)
            _, eff_lo, eff_hi = map_scalars_to_rgba(
                vals_all, cmap=cmap, vmin=vmin, vmax=vmax, symmetric=sym, log_scale=self.log_scale
            )
            self.effective_vmin = eff_lo
            self.effective_vmax = eff_hi

            eff_mode = self._effective_mode()

            if self.dataset.spatial_dim == 2 or eff_mode == "heightfield":
                xs, ys, grid, _ = self.dataset.to_2d_grid(
                    self.field_name,
                    mode=mode,
                    time_index=self.time_index,
                    resolution=44,
                    slice_axis="z" if self.dataset.spatial_dim == 3 else None,
                    slice_value=self._axis_value("z") if self.dataset.spatial_dim == 3 else None,
                )
                mesh = build_heightfield_mesh(
                    xs, ys, grid,
                    warp_scale=self.warp_scale,
                    cmap=cmap, vmin=eff_lo, vmax=eff_hi, symmetric=sym, log_scale=self.log_scale,
                )
                self._meshes.append(mesh)
                span = max(float(xs[-1] - xs[0]), float(ys[-1] - ys[0]), 1e-9)
                hx = float(xs[-1] - xs[0]) / span
                hy = float(ys[-1] - ys[0]) / span
                hz = float(self.warp_scale)
                self._bounds = [(-hx, hx), (-hy, hy), (-hz, hz)]
                self._center = np.zeros(3, dtype=np.float64)
                self._scale = 1.0
                self._box_edges = build_box_edges(self._bounds, self._center, self._scale)
            else:
                # True 3D dataset
                bounds = list(self.dataset.spatial_bounds or [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)])
                while len(bounds) < 3:
                    bounds.append((-0.5, 0.5))
                self._bounds = bounds[:3]
                self._center, self._scale = normalize_bounds(self._bounds)
                self._box_edges = build_box_edges(self._bounds, self._center, self._scale)

                if eff_mode in ("slices", "slices_iso"):
                    for axis, enabled, frac in (
                        ("x", self.show_slice_x, self.slice_x_frac),
                        ("y", self.show_slice_y, self.slice_y_frac),
                        ("z", self.show_slice_z, self.slice_z_frac),
                    ):
                        if not enabled:
                            continue
                        ax_i = ("x", "y", "z").index(axis)
                        lo_a, hi_a = self._bounds[ax_i]
                        sval = lo_a + float(frac) * (hi_a - lo_a)
                        sxs, sys, sgrid, _ = self.dataset.to_2d_grid(
                            self.field_name, mode=mode, time_index=self.time_index,
                            resolution=32, slice_axis=axis, slice_value=sval,
                        )
                        smesh = build_slice_mesh(
                            sxs, sys, sgrid, axis, sval, self._center, self._scale,
                            cmap=cmap, vmin=eff_lo, vmax=eff_hi, symmetric=sym, log_scale=self.log_scale,
                            alpha=225,
                        )
                        self._meshes.append(smesh)

                if eff_mode in ("isosurface", "slices_iso"):
                    vol_pack = self.dataset.to_3d_grid(self.field_name, mode=mode, time_index=self.time_index)
                    if vol_pack is not None:
                        vxs, vys, vzs, vol = vol_pack
                        iso_val = eff_lo + float(self.iso_frac) * (eff_hi - eff_lo)
                        lut = colormap_lut(cmap)
                        lut_idx = int(np.clip(round(self.iso_frac * 255.0), 0, 255))
                        rgb = lut[lut_idx]
                        iso_mesh = marching_tetrahedra(
                            vxs, vys, vzs, vol, iso_val, self._center, self._scale,
                            rgba_color=(int(rgb[0]), int(rgb[1]), int(rgb[2]), int(self.iso_alpha)),
                        )
                        self._meshes.append(iso_mesh)
                    elif eff_mode == "isosurface":
                        self._error_message = "Isosurface requires a regular 3D grid; use 3D Point cloud or Slices for unstructured 3D clouds."

                if eff_mode == "cloud":
                    pts = self.dataset.spatial_points(self.time_index)
                    t_lo = eff_lo + float(self.threshold_min_frac) * (eff_hi - eff_lo)
                    t_hi = eff_lo + float(self.threshold_max_frac) * (eff_hi - eff_lo)
                    if t_hi < t_lo:
                        t_lo, t_hi = t_hi, t_lo
                    keep = np.isfinite(vals_all) & (vals_all >= t_lo - 1e-12) & (vals_all <= t_hi + 1e-12)
                    if self.clip_axis in ("x", "y", "z"):
                        c_i = ("x", "y", "z").index(self.clip_axis)
                        clo, chi = self._bounds[c_i]
                        c_lim = clo + float(self.clip_frac) * (chi - clo)
                        keep = keep & (pts[:, c_i] <= c_lim + 1e-12)
                    pts_kept = pts[keep]
                    vals_kept = vals_all[keep]
                    if len(pts_kept) > 0:
                        stride = max(1, len(pts_kept) // 8000)
                        pts_kept = pts_kept[::stride]
                        vals_kept = vals_kept[::stride]
                        self._cloud_pts = normalize_points(pts_kept, self._center, self._scale)
                        rgba_c, _, _ = map_scalars_to_rgba(
                            vals_kept, cmap=cmap, vmin=eff_lo, vmax=eff_hi,
                            symmetric=sym, log_scale=self.log_scale, alpha=int(self.cloud_alpha),
                        )
                        self._cloud_rgba = rgba_c

                if self.show_obstacle and self.dataset.obstacle is not None:
                    obs_mesh = build_obstacle_mesh(self.dataset.obstacle, self._bounds, self._center, self._scale)
                    if obs_mesh is not None:
                        self._meshes.append(obs_mesh)
        except Exception as exc:
            self._error_message = str(exc)
        self.update()

    def _axis_value(self, axis: str) -> float:
        idx = ("x", "y", "z").index(axis)
        bounds = self.dataset.spatial_bounds if self.dataset is not None and self.dataset.spatial_bounds else [(-1, 1), (-1, 1), (-1, 1)]
        lo, hi = bounds[idx]
        frac = {"x": self.slice_x_frac, "y": self.slice_y_frac, "z": self.slice_z_frac}[axis]
        return lo + float(frac) * (hi - lo)

    def paint_to(self, painter: QPainter, target_rect: QRectF):
        c = colors()
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(target_rect, QColor(c["canvas"]))
        w, h = target_rect.width(), target_rect.height()

        if self._error_message:
            painter.setPen(QColor(c["text"]))
            painter.drawText(target_rect.adjusted(20, 20, -20, -20), Qt.TextFlag.TextWordWrap, self._error_message)
            painter.restore()
            return
        if self.dataset is None:
            painter.setPen(QColor(c["muted"]))
            painter.drawText(target_rect, Qt.AlignmentFlag.AlignCenter, "3D Viewport · load a 2D or 3D solution to orbit and inspect.")
            painter.restore()
            return

        # 1. Bounding box wireframe (behind geometry)
        if self.show_box and self._box_edges is not None:
            pts_box = self._box_edges.reshape(-1, 3)
            xy_b, _, vis_b = self.camera.project(pts_box, w, h)
            xy_b = xy_b.reshape(-1, 2, 2)
            vis_b = vis_b.reshape(-1, 2)
            box_pen = QPen(QColor(c["border"]), 1.0)
            painter.setPen(box_pen)
            for seg_xy, seg_vis in zip(xy_b, vis_b):
                if seg_vis[0] and seg_vis[1]:
                    painter.drawLine(
                        QPointF(target_rect.left() + seg_xy[0, 0], target_rect.top() + seg_xy[0, 1]),
                        QPointF(target_rect.left() + seg_xy[1, 0], target_rect.top() + seg_xy[1, 1]),
                    )

        # 2. Combined depth-sorted shaded triangle meshes (heightfield, slices, isosurface, obstacle)
        light_dir = np.array([0.45, 0.35, 0.82], dtype=np.float64)
        light_dir /= float(np.linalg.norm(light_dir))
        tri_polys: list[tuple[float, QPolygonF, QColor]] = []
        for mesh in self._meshes:
            verts = mesh["verts"]
            faces = mesh["faces"]
            if len(faces) == 0:
                continue
            f_colors = mesh["colors"]
            normals = mesh["normals"]
            xy, depth, vis = self.camera.project(verts, w, h)
            f_vis = vis[faces[:, 0]] & vis[faces[:, 1]] & vis[faces[:, 2]]
            if not np.any(f_vis):
                continue
            valid_idx = np.where(f_vis)[0]
            f_sel = faces[valid_idx]
            f_depths = (depth[f_sel[:, 0]] + depth[f_sel[:, 1]] + depth[f_sel[:, 2]]) / 3.0
            # Two-sided directional lighting for smooth visibility from any orbit angle
            shade = 0.55 + 0.45 * np.abs(normals[valid_idx] @ light_dir)
            cols = f_colors[valid_idx]
            ox, oy = target_rect.left(), target_rect.top()
            for loc_i, z_val in enumerate(f_depths):
                i0, i1, i2 = f_sel[loc_i]
                poly = QPolygonF(
                    [
                        QPointF(ox + xy[i0, 0], oy + xy[i0, 1]),
                        QPointF(ox + xy[i1, 0], oy + xy[i1, 1]),
                        QPointF(ox + xy[i2, 0], oy + xy[i2, 1]),
                    ]
                )
                sh = float(shade[loc_i])
                r = min(255, int(cols[loc_i, 0] * sh))
                g = min(255, int(cols[loc_i, 1] * sh))
                b = min(255, int(cols[loc_i, 2] * sh))
                a = int(cols[loc_i, 3])
                tri_polys.append((float(z_val), poly, QColor(r, g, b, a)))

        if tri_polys:
            # Painter's algorithm: farthest (largest depth) drawn first
            tri_polys.sort(key=lambda item: item[0], reverse=True)
            wire_pen = QPen(QColor(30, 30, 30, 95), 0.7) if self.show_wireframe else None
            for _, poly, col in tri_polys:
                painter.setBrush(col)
                painter.setPen(wire_pen if wire_pen is not None else QPen(col, 0.5))
                painter.drawPolygon(poly)

        # 3. Depth-sorted 3D point cloud
        if self._cloud_pts is not None and self._cloud_rgba is not None and len(self._cloud_pts) > 0:
            xy_c, depth_c, vis_c = self.camera.project(self._cloud_pts, w, h)
            valid_c = np.where(vis_c)[0]
            if len(valid_c) > 0:
                order_c = valid_c[np.argsort(depth_c[valid_c])[::-1]]
                rad = max(1.5, float(self.point_size) * 0.5)
                ox, oy = target_rect.left(), target_rect.top()
                painter.setPen(Qt.PenStyle.NoPen)
                for idx in order_c:
                    rgba = self._cloud_rgba[idx]
                    painter.setBrush(QColor(int(rgba[0]), int(rgba[1]), int(rgba[2]), int(rgba[3])))
                    painter.drawEllipse(QPointF(ox + xy_c[idx, 0], oy + xy_c[idx, 1]), rad, rad)

        # 4. Corner RGB axes orientation triad
        if self.show_axes:
            self._draw_axes_triad(painter, target_rect)

        # Compact camera info overlay
        painter.setPen(QColor(c["muted"]))
        proj_tag = "Ortho" if self.camera.orthographic else "Persp"
        painter.drawText(
            int(target_rect.left() + 12),
            int(target_rect.top() + 20),
            f"3D ({self._effective_mode()}) · {proj_tag} · Az {self.camera.azimuth_deg:.0f}° El {self.camera.elevation_deg:.0f}° · Left-drag orbit · Shift/Mid-drag pan · Wheel zoom",
        )
        painter.restore()

    def _draw_axes_triad(self, painter: QPainter, target_rect: QRectF):
        origin = QPointF(target_rect.left() + 44.0, target_rect.bottom() - 44.0)
        right, up, _ = self.camera.basis()
        axes = (
            ("X", np.array([1.0, 0.0, 0.0]), QColor("#e05252")),
            ("Y", np.array([0.0, 1.0, 0.0]), QColor("#3fa34d")),
            ("Z", np.array([0.0, 0.0, 1.0]), QColor("#3a86ff")),
        )
        length = 28.0
        for label, vec, col in axes:
            dx = float(np.dot(vec, right)) * length
            dy = -float(np.dot(vec, up)) * length
            end = QPointF(origin.x() + dx, origin.y() + dy)
            painter.setPen(QPen(col, 2.0))
            painter.drawLine(origin, end)
            painter.drawText(QPointF(end.x() + 3, end.y() + 4), label)

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

    def wheelEvent(self, event):
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if delta:
            self.camera.zoom(1.12 ** (delta / 120.0))
            self.cameraChanged.emit()
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
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging_orbit = True
            self._last_pos = pos
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = event.position()
        if self._dragging_orbit:
            delta = pos - self._last_pos
            self.camera.orbit(delta.x(), delta.y())
            self._last_pos = pos
            self.cameraChanged.emit()
            self.update()
            event.accept()
            return
        if self._dragging_pan:
            delta = pos - self._last_pos
            self.camera.pan(delta.x(), delta.y())
            self._last_pos = pos
            self.cameraChanged.emit()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging_orbit or self._dragging_pan:
            self._dragging_orbit = False
            self._dragging_pan = False
            event.accept()
            return
        super().mouseReleaseEvent(event)
