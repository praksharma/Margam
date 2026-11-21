"""2D marching squares contours and RK4 streamline integration (NumPy only)."""
from __future__ import annotations

import math
import numpy as np


# Edge pairs for each of the 16 marching-squares cell bitmasks.
# Corners: bit0=BL (i,j), bit1=BR (i+1,j), bit2=TR (i+1,j+1), bit3=TL (i,j+1)
# Edges: 0=bottom (BL-BR), 1=right (BR-TR), 2=top (TL-TR), 3=left (BL-TL)
_MS_TABLE: tuple[tuple[tuple[int, int], ...], ...] = (
    (),
    ((3, 0),),
    ((0, 1),),
    ((3, 1),),
    ((1, 2),),
    ((3, 0), (1, 2)),
    ((0, 2),),
    ((3, 2),),
    ((2, 3),),
    ((0, 2),),
    ((0, 1), (2, 3)),
    ((1, 2),),
    ((3, 1),),
    ((0, 1),),
    ((3, 0),),
    (),
)


def marching_squares(
    xs: np.ndarray,
    ys: np.ndarray,
    grid_ny_nx: np.ndarray,
    levels: np.ndarray | list[float],
) -> list[tuple[float, np.ndarray]]:
    """Extract contour line segments on a regular 2D grid ``(ny, nx)``.

    Returns a list of ``(level, segments)`` where ``segments`` has shape
    ``(M, 2, 2)`` containing ``[(x0, y0), (x1, y1)]`` endpoints in data coords.
    Cells touching NaN values are skipped cleanly.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    z = np.asarray(grid_ny_nx, dtype=np.float64)
    ny, nx = z.shape
    if ny < 2 or nx < 2:
        return []

    z00 = z[:-1, :-1]  # BL (y_j, x_i)
    z10 = z[:-1, 1:]   # BR (y_j, x_{i+1})
    z11 = z[1:, 1:]    # TR (y_{j+1}, x_{i+1})
    z01 = z[1:, :-1]   # TL (y_{j+1}, x_i)
    valid_cell = np.isfinite(z00) & np.isfinite(z10) & np.isfinite(z11) & np.isfinite(z01)
    if not np.any(valid_cell):
        return []

    x0 = xs[:-1][None, :]
    x1 = xs[1:][None, :]
    y0 = ys[:-1][:, None]
    y1 = ys[1:][:, None]

    results: list[tuple[float, np.ndarray]] = []
    for raw_level in levels:
        lvl = float(raw_level)
        if not np.isfinite(lvl):
            continue
        mask = (
            (z00 >= lvl).astype(np.uint8)
            | ((z10 >= lvl).astype(np.uint8) << 1)
            | ((z11 >= lvl).astype(np.uint8) << 2)
            | ((z01 >= lvl).astype(np.uint8) << 3)
        )
        mask = np.where(valid_cell, mask, 0)
        active_cases = np.unique(mask)
        segs_for_level: list[np.ndarray] = []

        def interp_edge(edge_id: int, jj: np.ndarray, ii: np.ndarray) -> np.ndarray:
            if edge_id == 0:  # bottom: (x0..x1, y0)
                va, vb = z00[jj, ii], z10[jj, ii]
                denom = vb - va
                t = np.where(np.abs(denom) > 1e-15, (lvl - va) / denom, 0.5)
                t = np.clip(t, 0.0, 1.0)
                px = x0[0, ii] + t * (x1[0, ii] - x0[0, ii])
                py = y0[jj, 0]
            elif edge_id == 1:  # right: (x1, y0..y1)
                va, vb = z10[jj, ii], z11[jj, ii]
                denom = vb - va
                t = np.where(np.abs(denom) > 1e-15, (lvl - va) / denom, 0.5)
                t = np.clip(t, 0.0, 1.0)
                px = x1[0, ii]
                py = y0[jj, 0] + t * (y1[jj, 0] - y0[jj, 0])
            elif edge_id == 2:  # top: (x0..x1, y1)
                va, vb = z01[jj, ii], z11[jj, ii]
                denom = vb - va
                t = np.where(np.abs(denom) > 1e-15, (lvl - va) / denom, 0.5)
                t = np.clip(t, 0.0, 1.0)
                px = x0[0, ii] + t * (x1[0, ii] - x0[0, ii])
                py = y1[jj, 0]
            else:  # left: (x0, y0..y1)
                va, vb = z00[jj, ii], z01[jj, ii]
                denom = vb - va
                t = np.where(np.abs(denom) > 1e-15, (lvl - va) / denom, 0.5)
                t = np.clip(t, 0.0, 1.0)
                px = x0[0, ii]
                py = y0[jj, 0] + t * (y1[jj, 0] - y0[jj, 0])
            return np.column_stack([px, py])

        for case_id in active_cases:
            pairs = _MS_TABLE[int(case_id)]
            if not pairs:
                continue
            jj, ii = np.where(mask == case_id)
            if len(jj) == 0:
                continue
            for e0, e1 in pairs:
                p0 = interp_edge(e0, jj, ii)
                p1 = interp_edge(e1, jj, ii)
                segs_for_level.append(np.stack([p0, p1], axis=1))

        if segs_for_level:
            results.append((lvl, np.concatenate(segs_for_level, axis=0)))
        else:
            results.append((lvl, np.zeros((0, 2, 2), dtype=np.float64)))
    return results


def compute_streamlines_2d(
    xs: np.ndarray,
    ys: np.ndarray,
    vx_ny_nx: np.ndarray,
    vy_ny_nx: np.ndarray,
    *,
    density: int = 9,
    max_steps: int = 90,
) -> list[np.ndarray]:
    """Integrate 2D streamlines with RK4 over a regular velocity grid ``(ny, nx)``."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    vx = np.asarray(vx_ny_nx, dtype=np.float64)
    vy = np.asarray(vy_ny_nx, dtype=np.float64)
    ny, nx = vx.shape
    if ny < 2 or nx < 2:
        return []
    xmin, xmax = float(xs[0]), float(xs[-1])
    ymin, ymax = float(ys[0]), float(ys[-1])
    dx_span = xmax - xmin
    dy_span = ymax - ymin
    if dx_span <= 0 or dy_span <= 0:
        return []

    speed = np.hypot(vx, vy)
    valid = np.isfinite(speed)
    if not np.any(valid):
        return []
    max_speed = float(np.max(speed[valid]))
    if max_speed <= 1e-12:
        return []

    step_len = 0.018 * math.hypot(dx_span, dy_span)

    def sample_vel(px: float, py: float) -> tuple[float, float] | None:
        if not (xmin <= px <= xmax and ymin <= py <= ymax):
            return None
        fx = (px - xmin) / dx_span * (nx - 1)
        fy = (py - ymin) / dy_span * (ny - 1)
        ix = min(nx - 2, max(0, int(fx)))
        iy = min(ny - 2, max(0, int(fy)))
        tx = fx - ix
        ty = fy - iy
        u00, u10, u01, u11 = vx[iy, ix], vx[iy, ix + 1], vx[iy + 1, ix], vx[iy + 1, ix + 1]
        v00, v10, v01, v11 = vy[iy, ix], vy[iy, ix + 1], vy[iy + 1, ix], vy[iy + 1, ix + 1]
        if not (
            np.isfinite(u00) and np.isfinite(u10) and np.isfinite(u01) and np.isfinite(u11)
            and np.isfinite(v00) and np.isfinite(v10) and np.isfinite(v01) and np.isfinite(v11)
        ):
            return None
        u = (1 - tx) * (1 - ty) * u00 + tx * (1 - ty) * u10 + (1 - tx) * ty * u01 + tx * ty * u11
        v = (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 + (1 - tx) * ty * v01 + tx * ty * v11
        return float(u), float(v)

    def unit_vel(px: float, py: float, sign: float) -> tuple[float, float] | None:
        vel = sample_vel(px, py)
        if vel is None:
            return None
        mag = math.hypot(vel[0], vel[1])
        if mag < 1e-6 * max_speed:
            return None
        return sign * vel[0] / mag, sign * vel[1] / mag

    def trace(sx: float, sy: float, sign: float) -> list[tuple[float, float]]:
        pts: list[tuple[float, float]] = []
        x, y = sx, sy
        for _ in range(max_steps):
            k1 = unit_vel(x, y, sign)
            if k1 is None:
                break
            k2 = unit_vel(x + 0.5 * step_len * k1[0], y + 0.5 * step_len * k1[1], sign)
            if k2 is None:
                break
            k3 = unit_vel(x + 0.5 * step_len * k2[0], y + 0.5 * step_len * k2[1], sign)
            if k3 is None:
                break
            k4 = unit_vel(x + step_len * k3[0], y + step_len * k3[1], sign)
            if k4 is None:
                break
            nx_p = x + (step_len / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            ny_p = y + (step_len / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            if sample_vel(nx_p, ny_p) is None:
                break
            pts.append((nx_p, ny_p))
            x, y = nx_p, ny_p
        return pts

    n_seed = max(3, min(20, int(density)))
    seed_xs = np.linspace(xmin + 0.05 * dx_span, xmax - 0.05 * dx_span, n_seed)
    seed_ys = np.linspace(ymin + 0.05 * dy_span, ymax - 0.05 * dy_span, n_seed)
    lines: list[np.ndarray] = []
    # Occupancy grid to avoid dense overlapping clutter
    occ_nx, occ_ny = n_seed * 2, n_seed * 2
    occupied = np.zeros((occ_ny, occ_nx), dtype=bool)
    for sy in seed_ys:
        for sx in seed_xs:
            if sample_vel(float(sx), float(sy)) is None:
                continue
            gx = min(occ_nx - 1, max(0, int((sx - xmin) / dx_span * occ_nx)))
            gy = min(occ_ny - 1, max(0, int((sy - ymin) / dy_span * occ_ny)))
            if occupied[gy, gx]:
                continue
            bwd = trace(float(sx), float(sy), -1.0)
            fwd = trace(float(sx), float(sy), 1.0)
            poly = list(reversed(bwd)) + [(float(sx), float(sy))] + fwd
            if len(poly) >= 3:
                arr = np.asarray(poly, dtype=np.float64)
                lines.append(arr)
                ixs = np.clip(((arr[:, 0] - xmin) / dx_span * occ_nx).astype(int), 0, occ_nx - 1)
                iys = np.clip(((arr[:, 1] - ymin) / dy_span * occ_ny).astype(int), 0, occ_ny - 1)
                occupied[iys, ixs] = True
    return lines
