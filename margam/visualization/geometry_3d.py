"""3D camera math, heightfield meshes, orthogonal slices, and marching-tetrahedra isosurfaces."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
import numpy as np

from .colormaps import map_scalars_to_rgba


@dataclass
class Camera3D:
    """Spherical orbit camera supporting perspective and orthographic projection."""

    azimuth_deg: float = -42.0
    elevation_deg: float = 28.0
    distance: float = 3.2
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fov_deg: float = 38.0
    orthographic: bool = False

    def reset(self):
        self.azimuth_deg = -42.0
        self.elevation_deg = 28.0
        self.distance = 3.2
        self.target = (0.0, 0.0, 0.0)

    def set_preset(self, preset: str):
        presets = {
            "iso": (-42.0, 28.0),
            "xy": (-90.0, 89.5),
            "xz": (-90.0, 0.0),
            "yz": (0.0, 0.0),
        }
        az, el = presets.get(preset.lower(), (-42.0, 28.0))
        self.azimuth_deg = az
        self.elevation_deg = el

    def orbit(self, dx: float, dy: float):
        self.azimuth_deg = (self.azimuth_deg - dx * 0.55) % 360.0
        self.elevation_deg = max(-89.5, min(89.5, self.elevation_deg + dy * 0.55))

    def pan(self, dx: float, dy: float):
        right, up, _ = self.basis()
        scale = 0.0028 * max(0.2, self.distance)
        shift = (-dx * scale) * right + (dy * scale) * up
        self.target = (
            float(self.target[0] + shift[0]),
            float(self.target[1] + shift[1]),
            float(self.target[2] + shift[2]),
        )

    def zoom(self, factor: float):
        self.distance = max(0.5, min(20.0, self.distance / max(0.1, factor)))

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        az = math.radians(self.azimuth_deg)
        el = math.radians(self.elevation_deg)
        # Eye direction pointing from target toward eye
        ex = math.cos(el) * math.cos(az)
        ey = math.cos(el) * math.sin(az)
        ez = math.sin(el)
        eye_dir = np.array([ex, ey, ez], dtype=np.float64)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(eye_dir, world_up))) > 0.995:
            world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(world_up, eye_dir)
        right /= max(1e-12, float(np.linalg.norm(right)))
        up = np.cross(eye_dir, right)
        up /= max(1e-12, float(np.linalg.norm(up)))
        return right, up, eye_dir

    def eye_position(self) -> np.ndarray:
        _, _, eye_dir = self.basis()
        return np.asarray(self.target, dtype=np.float64) + eye_dir * self.distance

    def project(
        self,
        points: np.ndarray,
        width: float,
        height: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project normalized 3D coordinates ``(N, 3)`` into viewport pixel coordinates.

        Returns ``(xy_Nx2, depth_N, visible_mask_N)`` where smaller ``depth`` is
        closer to the camera (positive distance along camera view axis).
        """
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) == 0:
            return np.zeros((0, 2)), np.zeros((0,)), np.zeros((0,), dtype=bool)
        right, up, eye_dir = self.basis()
        eye = np.asarray(self.target, dtype=np.float64) + eye_dir * self.distance
        rel = pts - eye[None, :]
        # Camera coordinates: x=right, y=up, z_cam = distance in front of eye (-dot with eye_dir)
        cx = rel @ right
        cy = rel @ up
        cz = -(rel @ eye_dir)
        scale_px = min(width, height) * 0.5
        if self.orthographic:
            ortho_span = max(0.3, self.distance * math.tan(math.radians(self.fov_deg * 0.5)))
            sx = width * 0.5 + (cx / ortho_span) * scale_px
            sy = height * 0.5 - (cy / ortho_span) * scale_px
            visible = np.isfinite(cx) & np.isfinite(cy) & np.isfinite(cz)
        else:
            f = 1.0 / math.tan(math.radians(self.fov_deg * 0.5))
            visible = (cz > 0.05) & np.isfinite(cx) & np.isfinite(cy) & np.isfinite(cz)
            safe_z = np.where(visible, cz, 1.0)
            sx = width * 0.5 + (cx * f / safe_z) * scale_px
            sy = height * 0.5 - (cy * f / safe_z) * scale_px
        return np.column_stack([sx, sy]), cz, visible


def normalize_bounds(bounds: list[tuple[float, float]]) -> tuple[np.ndarray, float]:
    """Compute center and uniform half-extent scaling so domain fits cleanly inside [-1, 1]^3."""
    b = list(bounds)
    while len(b) < 3:
        b.append((-0.5, 0.5))
    lo = np.array([b[0][0], b[1][0], b[2][0]], dtype=np.float64)
    hi = np.array([b[0][1], b[1][1], b[2][1]], dtype=np.float64)
    center = 0.5 * (lo + hi)
    max_span = float(np.max(hi - lo))
    scale = 2.0 / max(max_span, 1e-9)
    return center, scale


def normalize_points(points: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    return (np.asarray(points, dtype=np.float64) - center[None, :]) * scale


def build_box_edges(bounds: list[tuple[float, float]], center: np.ndarray, scale: float) -> np.ndarray:
    """Return ``(12, 2, 3)`` normalized line segments for the 3D bounding box."""
    (x0, x1), (y0, y1), (z0, z1) = bounds[:3]
    corners = np.array(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ],
        dtype=np.float64,
    )
    c = normalize_points(corners, center, scale)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return np.array([[c[a], c[b]] for a, b in edges], dtype=np.float64)


def _compute_face_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.divide(normals, np.maximum(norms, 1e-12))


def build_heightfield_mesh(
    xs: np.ndarray,
    ys: np.ndarray,
    grid_ny_nx: np.ndarray,
    *,
    warp_scale: float = 0.5,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = False,
    log_scale: bool = False,
    max_dim: int = 44,
) -> dict[str, Any]:
    """Build a normalized 3D heightfield triangle mesh ``z = f(x, y)``."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    z = np.asarray(grid_ny_nx, dtype=np.float64)
    ny, nx = z.shape
    sx = max(1, nx // max_dim)
    sy = max(1, ny // max_dim)
    xs_s = xs[::sx]
    ys_s = ys[::sy]
    z_s = z[::sy, ::sx]
    ny_s, nx_s = z_s.shape

    rgba, eff_lo, eff_hi = map_scalars_to_rgba(
        z_s, cmap=cmap, vmin=vmin, vmax=vmax, symmetric=symmetric, log_scale=log_scale
    )
    gx, gy = np.meshgrid(xs_s, ys_s, indexing="xy")
    span = max(float(xs_s[-1] - xs_s[0]), float(ys_s[-1] - ys_s[0]), 1e-9)
    cx = 0.5 * (xs_s[0] + xs_s[-1])
    cy = 0.5 * (ys_s[0] + ys_s[-1])
    nx_norm = (gx - cx) * (2.0 / span)
    ny_norm = (gy - cy) * (2.0 / span)
    z_mid = 0.5 * (eff_lo + eff_hi)
    z_half = max(0.5 * (eff_hi - eff_lo), 1e-9)
    nz_norm = np.where(np.isfinite(z_s), ((z_s - z_mid) / z_half) * float(warp_scale), 0.0)

    verts = np.column_stack([nx_norm.ravel(), ny_norm.ravel(), nz_norm.ravel()])
    valid = np.isfinite(z_s)
    cell_valid = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    jj, ii = np.where(cell_valid)
    if len(jj) == 0:
        return {
            "verts": verts,
            "faces": np.zeros((0, 3), dtype=np.int64),
            "colors": np.zeros((0, 4), dtype=np.uint8),
            "normals": np.zeros((0, 3), dtype=np.float64),
            "vmin": eff_lo,
            "vmax": eff_hi,
        }
    v00 = jj * nx_s + ii
    v10 = jj * nx_s + (ii + 1)
    v01 = (jj + 1) * nx_s + ii
    v11 = (jj + 1) * nx_s + (ii + 1)
    f1 = np.column_stack([v00, v10, v11])
    f2 = np.column_stack([v00, v11, v01])
    faces = np.vstack([f1, f2])
    c_cell = rgba[:-1, :-1][jj, ii]
    face_colors = np.vstack([c_cell, c_cell])
    normals = _compute_face_normals(verts, faces)
    return {
        "verts": verts,
        "faces": faces,
        "colors": face_colors,
        "normals": normals,
        "vmin": eff_lo,
        "vmax": eff_hi,
    }


def build_slice_mesh(
    xs: np.ndarray,
    ys: np.ndarray,
    grid_ny_nx: np.ndarray,
    axis: str,
    slice_value: float,
    center: np.ndarray,
    scale: float,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = False,
    log_scale: bool = False,
    alpha: int = 230,
    max_dim: int = 36,
) -> dict[str, Any]:
    """Build a colored 3D quad mesh lying on the orthogonal plane ``axis = slice_value``."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    z = np.asarray(grid_ny_nx, dtype=np.float64)
    ny, nx = z.shape
    sx = max(1, nx // max_dim)
    sy = max(1, ny // max_dim)
    xs_s = xs[::sx]
    ys_s = ys[::sy]
    z_s = z[::sy, ::sx]
    ny_s, nx_s = z_s.shape

    rgba, eff_lo, eff_hi = map_scalars_to_rgba(
        z_s, cmap=cmap, vmin=vmin, vmax=vmax, symmetric=symmetric, log_scale=log_scale, alpha=alpha
    )
    ga, gb = np.meshgrid(xs_s, ys_s, indexing="xy")
    const = np.full_like(ga, float(slice_value))
    if axis == "z":
        world = np.column_stack([ga.ravel(), gb.ravel(), const.ravel()])
    elif axis == "y":
        world = np.column_stack([ga.ravel(), const.ravel(), gb.ravel()])
    else:
        world = np.column_stack([const.ravel(), ga.ravel(), gb.ravel()])
    verts = normalize_points(world, center, scale)

    valid = np.isfinite(z_s)
    cell_valid = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    jj, ii = np.where(cell_valid)
    if len(jj) == 0:
        return {
            "verts": verts,
            "faces": np.zeros((0, 3), dtype=np.int64),
            "colors": np.zeros((0, 4), dtype=np.uint8),
            "normals": np.zeros((0, 3), dtype=np.float64),
            "vmin": eff_lo,
            "vmax": eff_hi,
        }
    v00 = jj * nx_s + ii
    v10 = jj * nx_s + (ii + 1)
    v01 = (jj + 1) * nx_s + ii
    v11 = (jj + 1) * nx_s + (ii + 1)
    faces = np.vstack([np.column_stack([v00, v10, v11]), np.column_stack([v00, v11, v01])])
    c_cell = rgba[:-1, :-1][jj, ii]
    face_colors = np.vstack([c_cell, c_cell])
    normals = _compute_face_normals(verts, faces)
    return {
        "verts": verts,
        "faces": faces,
        "colors": face_colors,
        "normals": normals,
        "vmin": eff_lo,
        "vmax": eff_hi,
    }


# 6 Kuhn tetrahedra subdividing a unit cube with vertex indices:
# 0=(0,0,0), 1=(1,0,0), 2=(1,1,0), 3=(0,1,0), 4=(0,0,1), 5=(1,0,1), 6=(1,1,1), 7=(0,1,1)
_KUHN_TETS: tuple[tuple[int, int, int, int], ...] = (
    (0, 1, 2, 6),
    (0, 2, 3, 6),
    (0, 3, 7, 6),
    (0, 7, 4, 6),
    (0, 4, 5, 6),
    (0, 5, 1, 6),
)


def marching_tetrahedra(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    volume_nx_ny_nz: np.ndarray,
    isovalue: float,
    center: np.ndarray,
    scale: float,
    *,
    rgba_color: tuple[int, int, int, int] = (43, 157, 145, 210),
    max_cells_per_axis: int = 24,
) -> dict[str, Any]:
    """Extract a 3D isosurface triangle mesh via vectorized Kuhn marching tetrahedra."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    zs = np.asarray(zs, dtype=np.float64)
    vol = np.asarray(volume_nx_ny_nz, dtype=np.float64)
    nx, ny, nz = vol.shape
    sx = max(1, (nx - 1) // max_cells_per_axis)
    sy = max(1, (ny - 1) // max_cells_per_axis)
    sz = max(1, (nz - 1) // max_cells_per_axis)
    xs = xs[::sx]
    ys = ys[::sy]
    zs = zs[::sz]
    vol = vol[::sx, ::sy, ::sz]
    nx, ny, nz = vol.shape
    if nx < 2 or ny < 2 or nz < 2:
        return {"verts": np.zeros((0, 3)), "faces": np.zeros((0, 3), dtype=np.int64), "colors": np.zeros((0, 4), dtype=np.uint8), "normals": np.zeros((0, 3))}

    iso = float(isovalue)
    # Build 8 corner values & positions for all cells (nx-1, ny-1, nz-1)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    corners_val = np.stack([
        vol[:-1, :-1, :-1], vol[1:, :-1, :-1], vol[1:, 1:, :-1], vol[:-1, 1:, :-1],
        vol[:-1, :-1, 1:],  vol[1:, :-1, 1:],  vol[1:, 1:, 1:],  vol[:-1, 1:, 1:],
    ], axis=-1)  # (cx, cy, cz, 8)
    corners_pos = np.stack([
        np.stack([gx[:-1, :-1, :-1], gy[:-1, :-1, :-1], gz[:-1, :-1, :-1]], axis=-1),
        np.stack([gx[1:, :-1, :-1],  gy[1:, :-1, :-1],  gz[1:, :-1, :-1]], axis=-1),
        np.stack([gx[1:, 1:, :-1],   gy[1:, 1:, :-1],   gz[1:, 1:, :-1]], axis=-1),
        np.stack([gx[:-1, 1:, :-1],  gy[:-1, 1:, :-1],  gz[:-1, 1:, :-1]], axis=-1),
        np.stack([gx[:-1, :-1, 1:],  gy[:-1, :-1, 1:],  gz[:-1, :-1, 1:]], axis=-1),
        np.stack([gx[1:, :-1, 1:],   gy[1:, :-1, 1:],   gz[1:, :-1, 1:]], axis=-1),
        np.stack([gx[1:, 1:, 1:],    gy[1:, 1:, 1:],    gz[1:, 1:, 1:]], axis=-1),
        np.stack([gx[:-1, 1:, 1:],   gy[:-1, 1:, 1:],   gz[:-1, 1:, 1:]], axis=-1),
    ], axis=-2)  # (cx, cy, cz, 8, 3)

    valid_cell = np.all(np.isfinite(corners_val), axis=-1)
    c_min = np.min(np.where(np.isfinite(corners_val), corners_val, np.inf), axis=-1)
    c_max = np.max(np.where(np.isfinite(corners_val), corners_val, -np.inf), axis=-1)
    active = valid_cell & (c_min <= iso) & (c_max >= iso)
    if not np.any(active):
        return {"verts": np.zeros((0, 3)), "faces": np.zeros((0, 3), dtype=np.int64), "colors": np.zeros((0, 4), dtype=np.uint8), "normals": np.zeros((0, 3))}

    cv = corners_val[active]   # (M, 8)
    cp = corners_pos[active]   # (M, 8, 3)

    tri_chunks: list[np.ndarray] = []

    def interp(pa, pb, va, vb):
        denom = vb - va
        t = np.where(np.abs(denom) > 1e-15, (iso - va) / denom, 0.5)
        t = np.clip(t, 0.0, 1.0)[:, None]
        return pa + t * (pb - pa)

    for tet in _KUHN_TETS:
        tv = cv[:, tet]  # (M, 4)
        tp = cp[:, tet, :]  # (M, 4, 3)
        above = tv >= iso
        count = np.sum(above, axis=1)

        # Case 1 vertex above (count==1) or 3 vertices above (count==3) -> 1 triangle
        for target_cnt in (1, 3):
            sel = np.where(count == target_cnt)[0]
            if len(sel) == 0:
                continue
            ab = above[sel] if target_cnt == 1 else ~above[sel]
            # Find the single lone vertex index and the 3 opposite vertices
            lone = np.argmax(ab, axis=1)
            p_lone = tp[sel, lone]
            v_lone = tv[sel, lone]
            others = np.array([[i for i in range(4) if i != int(l)] for l in lone], dtype=np.int64)
            e0 = interp(p_lone, tp[sel, others[:, 0]], v_lone, tv[sel, others[:, 0]])
            e1 = interp(p_lone, tp[sel, others[:, 1]], v_lone, tv[sel, others[:, 1]])
            e2 = interp(p_lone, tp[sel, others[:, 2]], v_lone, tv[sel, others[:, 2]])
            tri_chunks.append(np.stack([e0, e1, e2], axis=1))

        # Case 2 vertices above (count==2) -> quad split into 2 triangles
        sel2 = np.where(count == 2)[0]
        if len(sel2) > 0:
            ab2 = above[sel2]
            in_idx = np.array([[i for i in range(4) if ab2[r, i]] for r in range(len(sel2))], dtype=np.int64)
            out_idx = np.array([[i for i in range(4) if not ab2[r, i]] for r in range(len(sel2))], dtype=np.int64)
            a0, a1 = in_idx[:, 0], in_idx[:, 1]
            b0, b1 = out_idx[:, 0], out_idx[:, 1]
            p00 = interp(tp[sel2, a0], tp[sel2, b0], tv[sel2, a0], tv[sel2, b0])
            p01 = interp(tp[sel2, a0], tp[sel2, b1], tv[sel2, a0], tv[sel2, b1])
            p10 = interp(tp[sel2, a1], tp[sel2, b0], tv[sel2, a1], tv[sel2, b0])
            p11 = interp(tp[sel2, a1], tp[sel2, b1], tv[sel2, a1], tv[sel2, b1])
            tri_chunks.append(np.stack([p00, p01, p11], axis=1))
            tri_chunks.append(np.stack([p00, p11, p10], axis=1))

    if not tri_chunks:
        return {"verts": np.zeros((0, 3)), "faces": np.zeros((0, 3), dtype=np.int64), "colors": np.zeros((0, 4), dtype=np.uint8), "normals": np.zeros((0, 3))}

    tris_world = np.concatenate(tri_chunks, axis=0)  # (F, 3, 3)
    verts = normalize_points(tris_world.reshape(-1, 3), center, scale)
    n_f = len(tris_world)
    faces = np.arange(n_f * 3, dtype=np.int64).reshape(n_f, 3)
    colors_arr = np.tile(np.asarray(rgba_color, dtype=np.uint8), (n_f, 1))
    normals = _compute_face_normals(verts, faces)
    return {"verts": verts, "faces": faces, "colors": colors_arr, "normals": normals}


def build_obstacle_mesh(
    obstacle: dict[str, Any] | None,
    bounds: list[tuple[float, float]],
    center: np.ndarray,
    scale: float,
) -> dict[str, Any] | None:
    """Build a shaded 3D mesh for a spherical or cylindrical obstacle."""
    if not obstacle:
        return None
    kind = obstacle.get("kind")
    if kind == "sphere":
        cx, cy, cz = obstacle["center"]
        r = float(obstacle["radius"])
        n_lat, n_lon = 12, 20
        phi = np.linspace(0.0, math.pi, n_lat)
        theta = np.linspace(0.0, 2.0 * math.pi, n_lon, endpoint=False)
        gp, gt = np.meshgrid(phi, theta, indexing="ij")
        wx = cx + r * np.sin(gp) * np.cos(gt)
        wy = cy + r * np.sin(gp) * np.sin(gt)
        wz = cz + r * np.cos(gp)
        verts = normalize_points(np.column_stack([wx.ravel(), wy.ravel(), wz.ravel()]), center, scale)
        faces_list = []
        for i in range(n_lat - 1):
            for j in range(n_lon):
                jn = (j + 1) % n_lon
                v00 = i * n_lon + j
                v01 = i * n_lon + jn
                v10 = (i + 1) * n_lon + j
                v11 = (i + 1) * n_lon + jn
                faces_list.append([v00, v10, v11])
                faces_list.append([v00, v11, v01])
        faces = np.asarray(faces_list, dtype=np.int64)
        col = np.tile(np.array([150, 160, 172, 235], dtype=np.uint8), (len(faces), 1))
        return {"verts": verts, "faces": faces, "colors": col, "normals": _compute_face_normals(verts, faces)}
    if kind == "cylinder":
        cx, cy = obstacle["center"][:2]
        r = float(obstacle["radius"])
        z0, z1 = bounds[2] if len(bounds) >= 3 else (-0.5, 0.5)
        n_lon = 24
        theta = np.linspace(0.0, 2.0 * math.pi, n_lon, endpoint=False)
        wx = cx + r * np.cos(theta)
        wy = cy + r * np.sin(theta)
        bot = np.column_stack([wx, wy, np.full(n_lon, z0)])
        top = np.column_stack([wx, wy, np.full(n_lon, z1)])
        verts = normalize_points(np.vstack([bot, top]), center, scale)
        faces_list = []
        for j in range(n_lon):
            jn = (j + 1) % n_lon
            faces_list.append([j, jn, n_lon + jn])
            faces_list.append([j, n_lon + jn, n_lon + j])
        faces = np.asarray(faces_list, dtype=np.int64)
        col = np.tile(np.array([150, 160, 172, 220], dtype=np.uint8), (len(faces), 1))
        return {"verts": verts, "faces": faces, "colors": col, "normals": _compute_face_normals(verts, faces)}
    return None
