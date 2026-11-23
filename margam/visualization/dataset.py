"""Unified 1D/2D/3D/transient/parametric solution dataset for the Solution Explorer."""
from __future__ import annotations

from dataclasses import dataclass, field
import itertools
from typing import Any, Callable
import math
import numpy as np


def _regular_grid_interp(
    axes: tuple[np.ndarray, ...],
    grid: np.ndarray,
    query: np.ndarray,
    *,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Pure-NumPy multilinear interpolation on 1D/2D/3D regular Cartesian grids."""
    q = np.asarray(query, dtype=np.float64)
    lead_shape = q.shape[:-1]
    dim = len(axes)
    qf = q.reshape(-1, dim)
    m = len(qf)
    if m == 0:
        return np.empty(lead_shape, dtype=np.float64)

    in_bounds = np.ones(m, dtype=bool)
    i0_list: list[np.ndarray] = []
    i1_list: list[np.ndarray] = []
    t_list: list[np.ndarray] = []
    for d, ax in enumerate(axes):
        ax_arr = np.asarray(ax, dtype=np.float64)
        lo, hi = float(ax_arr[0]), float(ax_arr[-1])
        tol = max(1e-12, 1e-11 * max(1.0, hi - lo))
        qd = qf[:, d]
        in_bounds &= (qd >= lo - tol) & (qd <= hi + tol)
        qc = np.clip(qd, lo, hi)
        if len(ax_arr) == 1:
            i0 = np.zeros(m, dtype=np.int64)
            i1 = np.zeros(m, dtype=np.int64)
            td = np.zeros(m, dtype=np.float64)
        else:
            i1 = np.clip(np.searchsorted(ax_arr, qc, side="right"), 1, len(ax_arr) - 1)
            i0 = i1 - 1
            denom = ax_arr[i1] - ax_arr[i0]
            td = np.where(denom > 0.0, (qc - ax_arr[i0]) / np.where(denom > 0.0, denom, 1.0), 0.0)
        i0_list.append(i0)
        i1_list.append(i1)
        t_list.append(td)

    out = np.zeros(m, dtype=np.float64)
    for bits in itertools.product((0, 1), repeat=dim):
        weight = np.ones(m, dtype=np.float64)
        idx = []
        for d, bit in enumerate(bits):
            if bit == 0:
                weight *= 1.0 - t_list[d]
                idx.append(i0_list[d])
            else:
                weight *= t_list[d]
                idx.append(i1_list[d])
        out += weight * grid[tuple(idx)]
    out[~in_bounds] = fill_value
    return out.reshape(lead_shape)


def _convex_hull_2d(pts: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain 2D convex hull returning CCW vertices."""
    uniq = np.unique(np.asarray(pts, dtype=np.float64), axis=0)
    if len(uniq) <= 2:
        return uniq
    order = np.lexsort((uniq[:, 1], uniq[:, 0]))
    sorted_pts = uniq[order].tolist()

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[list[float]] = []
    for p in sorted_pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper: list[list[float]] = []
    for p in reversed(sorted_pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _points_in_convex_hull_2d(hull: np.ndarray, queries: np.ndarray) -> np.ndarray:
    if len(hull) < 3:
        return np.zeros(len(queries), dtype=bool)
    a = hull
    b = np.roll(hull, -1, axis=0)
    edge = b - a
    scale = max(1.0, float(np.max(np.abs(edge))))
    tol = 1e-10 * scale * scale
    # (M, H) cross products for every query against every CCW hull edge
    dx = queries[:, None, 0] - a[None, :, 0]
    dy = queries[:, None, 1] - a[None, :, 1]
    c = edge[None, :, 0] * dy - edge[None, :, 1] * dx
    return np.all(c >= -tol, axis=1)


def _scattered_interp(
    pts: np.ndarray,
    vals: np.ndarray,
    query: np.ndarray,
    *,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Interpolate scattered 2D/3D data, using SciPy Delaunay when installed or pure-NumPy IDW fallback."""
    pts_arr = np.asarray(pts, dtype=np.float64)
    vals_arr = np.asarray(vals, dtype=np.float64).reshape(-1)
    q = np.asarray(query, dtype=np.float64)
    lead_shape = q.shape[:-1]
    dim = pts_arr.shape[1]
    if len(pts_arr) == 0:
        return np.full(lead_shape, fill_value, dtype=np.float64)

    try:
        from scipy.interpolate import LinearNDInterpolator
    except ImportError:
        LinearNDInterpolator = None

    if LinearNDInterpolator is not None and len(pts_arr) >= dim + 1:
        try:
            interp = LinearNDInterpolator(pts_arr, vals_arr, fill_value=fill_value)
            return np.asarray(interp(q), dtype=np.float64)
        except Exception:
            pass

    qf = q.reshape(-1, dim)
    m = len(qf)
    out = np.full(m, fill_value, dtype=np.float64)
    if dim == 2:
        hull = _convex_hull_2d(pts_arr)
        valid = _points_in_convex_hull_2d(hull, qf)
    else:
        lo = np.min(pts_arr, axis=0)
        hi = np.max(pts_arr, axis=0)
        tol = np.maximum(1e-12, 1e-9 * (hi - lo))
        valid = np.all((qf >= lo - tol) & (qf <= hi + tol), axis=1)

    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        return out.reshape(lead_shape)

    k = min(8, len(pts_arr))
    for start in range(0, len(valid_idx), 512):
        batch_ids = valid_idx[start : start + 512]
        qb = qf[batch_ids]
        d2 = np.sum((qb[:, None, :] - pts_arr[None, :, :]) ** 2, axis=2)
        if k < len(pts_arr):
            top = np.argpartition(d2, k - 1, axis=1)[:, :k]
            row_i = np.arange(len(qb))[:, None]
            d2_k = d2[row_i, top]
            v_k = vals_arr[top]
        else:
            d2_k = d2
            v_k = np.broadcast_to(vals_arr[None, :], (len(qb), len(vals_arr)))
        exact = d2_k <= 1e-24
        has_exact = np.any(exact, axis=1)
        weights = 1.0 / np.maximum(d2_k, 1e-24)
        pred = np.sum(weights * v_k, axis=1) / np.sum(weights, axis=1)
        if np.any(has_exact):
            ex_rows = np.where(has_exact)[0]
            first_match = np.argmax(exact[ex_rows], axis=1)
            pred[ex_rows] = v_k[ex_rows, first_match]
        out[batch_ids] = pred
    return out.reshape(lead_shape)

DERIVED_SPEED = "Velocity magnitude |u|"
DERIVED_VORTICITY_2D = "Vorticity ω_z"
DERIVED_VORTICITY_3D = "Vorticity |ω|"

DISPLAY_MODES: tuple[tuple[str, str], ...] = (
    ("prediction", "Prediction"),
    ("reference", "Reference"),
    ("signed_error", "Signed error (pred − ref)"),
    ("abs_error", "Absolute error |pred − ref|"),
    ("rel_error", "Relative error |pred − ref| / |ref|"),
)


def _detect_regular_grid(spatial_points: np.ndarray, tol: float = 1e-7):
    """Detect whether ``spatial_points`` (N, d) form a complete Cartesian grid.

    Returns ``(axes, order_indices)`` where ``axes`` is a tuple of sorted 1D
    coordinate arrays of lengths ``(n0, n1, ...)`` and ``order_indices`` permutes
    rows into C-order over ``(axis_0, axis_1, ...)``, or ``(None, None)`` if the
    points are unstructured or missing obstacle cutouts.
    """
    n_pts, dim = spatial_points.shape
    if n_pts == 0 or dim not in (1, 2, 3):
        return None, None
    axes = []
    rounded_cols = []
    for d in range(dim):
        col = spatial_points[:, d]
        span = float(np.max(col) - np.min(col))
        scale = max(span, 1.0)
        decimals = max(0, min(10, int(round(-math.log10(tol * scale)))))
        rounded = np.round(col, decimals=decimals)
        uniq = np.unique(rounded)
        if len(uniq) < 2:
            return None, None
        axes.append(uniq)
        rounded_cols.append(rounded)
    expected = 1
    for ax in axes:
        expected *= len(ax)
    if expected != n_pts:
        return None, None
    # Verify uniqueness of grid index tuples and compute C-order permutation.
    keys = tuple(rounded_cols[d] for d in reversed(range(dim)))
    order = np.lexsort(keys)
    sorted_pts = np.column_stack([c[order] for c in rounded_cols])
    if len(np.unique(sorted_pts, axis=0)) != n_pts:
        return None, None
    # Exact values from sorted points along each axis
    exact_axes = []
    shape = tuple(len(ax) for ax in axes)
    reshaped = spatial_points[order].reshape(shape + (dim,))
    for d in range(dim):
        sl = [0] * dim
        sl[d] = slice(None)
        exact_axes.append(reshaped[tuple(sl) + (d,)].copy())
    return tuple(exact_axes), order


@dataclass
class SolutionDataset:
    """Unified field dataset supporting 1D, 2D, 3D, time-series, and parametric runs."""

    points: np.ndarray
    coordinate_names: tuple[str, ...]
    fields: dict[str, np.ndarray]
    references: dict[str, np.ndarray] = field(default_factory=dict)
    title: str = "Solution"
    source_description: str = ""
    steps: int | None = None
    point_ids: list[Any] | None = None
    obstacle: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Continuous parametric / arbitrary-time live inference hooks
    live_evaluator: Callable[..., dict[str, np.ndarray]] | None = None
    param_specs: list[dict[str, Any]] = field(default_factory=list)
    spatial_bounds: list[tuple[float, float]] | None = None
    time_bounds: tuple[float, float] | None = None

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=np.float64)
        if self.points.ndim == 1:
            self.points = self.points[:, None]
        self.coordinate_names = tuple(self.coordinate_names)
        if self.points.shape[1] != len(self.coordinate_names):
            raise ValueError(
                f"Point width {self.points.shape[1]} does not match coordinate names {self.coordinate_names}."
            )
        n = len(self.points)
        clean_fields: dict[str, np.ndarray] = {}
        for name, values in self.fields.items():
            arr = np.asarray(values, dtype=np.float64).reshape(-1)
            if len(arr) != n:
                raise ValueError(f"Field '{name}' length {len(arr)} does not match {n} points.")
            clean_fields[str(name)] = arr
        self.fields = clean_fields

        clean_refs: dict[str, np.ndarray] = {}
        for name, values in self.references.items():
            if values is None:
                continue
            arr = np.asarray(values, dtype=np.float64).reshape(-1)
            if len(arr) != n:
                raise ValueError(f"Reference '{name}' length {len(arr)} does not match {n} points.")
            clean_refs[str(name)] = arr
        self.references = clean_refs

        self.spatial_indices = tuple(
            i for i, c in enumerate(self.coordinate_names) if c in ("x", "y", "z")
        )
        self.spatial_names = tuple(self.coordinate_names[i] for i in self.spatial_indices)
        self.time_index_col = self.coordinate_names.index("t") if "t" in self.coordinate_names else None
        self.param_col_indices = tuple(
            i for i, c in enumerate(self.coordinate_names) if c not in ("x", "y", "z", "t")
        )

        if self.time_index_col is not None and n > 0:
            from ..components.prepared_data import cluster_1d
            t_col = self.points[:, self.time_index_col]
            tol = max(
                1e-12,
                1e-9 * float(np.ptp(t_col)),
                8.0 * float(np.finfo(np.float64).eps) * max(1.0, float(np.max(np.abs(t_col)))),
            )
            clusters = cluster_1d(t_col, tol)
            n_frames = int(clusters.max()) + 1
            self._frame_indices: list[np.ndarray] = [
                np.where(clusters == k)[0] for k in range(n_frames)
            ]
            self.times = np.array(
                [float(np.mean(t_col[idx])) for idx in self._frame_indices],
                dtype=np.float64,
            )
        else:
            self.times = np.asarray([], dtype=np.float64)
            self._frame_indices = [np.arange(n, dtype=np.int64)]

        if self.spatial_bounds is None and n > 0 and self.spatial_indices:
            self.spatial_bounds = [
                (float(np.min(self.points[:, idx])), float(np.max(self.points[:, idx])))
                for idx in self.spatial_indices
            ]
        if self.time_bounds is None and len(self.times) > 0:
            self.time_bounds = (float(self.times[0]), float(self.times[-1]))

        self._grid_cache: dict[int, tuple[tuple[np.ndarray, ...] | None, np.ndarray | None]] = {}
        self._interpolator_cache: dict[tuple[Any, ...], Any] = {}

    @property
    def spatial_dim(self) -> int:
        return len(self.spatial_names)

    @property
    def is_time_dependent(self) -> bool:
        return self.time_index_col is not None and len(self.times) > 0

    @property
    def num_frames(self) -> int:
        return max(1, len(self.times))

    @property
    def has_reference(self) -> bool:
        return bool(self.references)

    @property
    def has_vector_2d(self) -> bool:
        return "u" in self.fields and "v" in self.fields

    @property
    def has_vector_3d(self) -> bool:
        return "u" in self.fields and "v" in self.fields and "w" in self.fields and self.spatial_dim == 3

    def default_view_mode(self) -> str:
        if self.spatial_dim >= 3:
            return "3d"
        if self.spatial_dim == 2:
            return "2d"
        return "1d"

    def frame_row_indices(self, time_index: int | None = None) -> np.ndarray:
        if not self.is_time_dependent:
            return self._frame_indices[0]
        if time_index is None:
            idx = 0
        else:
            idx = max(0, min(int(time_index), len(self._frame_indices) - 1))
        return self._frame_indices[idx]

    def frame_time(self, time_index: int | None = None) -> float | None:
        if not self.is_time_dependent:
            return None
        idx = 0 if time_index is None else max(0, min(int(time_index), len(self.times) - 1))
        return float(self.times[idx])

    def spatial_points(self, time_index: int | None = None) -> np.ndarray:
        rows = self.frame_row_indices(time_index)
        if not self.spatial_indices:
            return self.points[rows]
        return self.points[rows][:, self.spatial_indices]

    def frame_grid(self, time_index: int | None = None):
        key = 0 if not self.is_time_dependent else max(0, min(int(time_index or 0), len(self._frame_indices) - 1))
        if key not in self._grid_cache:
            pts = self.spatial_points(key)
            self._grid_cache[key] = _detect_regular_grid(pts)
        return self._grid_cache[key]

    def is_regular_grid(self, time_index: int | None = None) -> bool:
        axes, _ = self.frame_grid(time_index)
        return axes is not None

    def available_fields(self) -> list[str]:
        names = list(self.fields.keys())
        if self.has_vector_2d:
            names.append(DERIVED_SPEED)
            if self.spatial_dim == 2:
                names.append(DERIVED_VORTICITY_2D)
        if self.has_vector_3d and self.is_regular_grid(0):
            names.append(DERIVED_VORTICITY_3D)
        return names

    def raw_field_names(self) -> list[str]:
        return list(self.fields.keys())

    def _compute_field_on_frame(self, name: str, source: dict[str, np.ndarray], time_index: int | None) -> np.ndarray | None:
        rows = self.frame_row_indices(time_index)
        if name in source:
            return source[name][rows]
        if name == DERIVED_SPEED and "u" in source and "v" in source:
            u = source["u"][rows]
            v = source["v"][rows]
            if self.spatial_dim == 3 and "w" in source:
                w = source["w"][rows]
                return np.sqrt(u * u + v * v + w * w)
            return np.sqrt(u * u + v * v)
        if name == DERIVED_VORTICITY_2D and self.spatial_dim == 2 and "u" in source and "v" in source:
            axes, order = self.frame_grid(time_index)
            if axes is not None and order is not None:
                nx, ny = len(axes[0]), len(axes[1])
                u_grid = source["u"][rows][order].reshape(nx, ny)
                v_grid = source["v"][rows][order].reshape(nx, ny)
                dv_dx = np.gradient(v_grid, axes[0], axis=0)
                du_dy = np.gradient(u_grid, axes[1], axis=1)
                vort_ordered = (dv_dx - du_dy).reshape(-1)
                inv = np.empty_like(order)
                inv[order] = np.arange(len(order))
                return vort_ordered[inv]
            # Unstructured 2D fallback: interpolate to coarse grid, compute curl, sample back
            return self._unstructured_vorticity_2d(rows, source["u"][rows], source["v"][rows])
        if name == DERIVED_VORTICITY_3D and self.spatial_dim == 3 and {"u", "v", "w"} <= source.keys():
            axes, order = self.frame_grid(time_index)
            if axes is not None and order is not None:
                nx, ny, nz = len(axes[0]), len(axes[1]), len(axes[2])
                u_g = source["u"][rows][order].reshape(nx, ny, nz)
                v_g = source["v"][rows][order].reshape(nx, ny, nz)
                w_g = source["w"][rows][order].reshape(nx, ny, nz)
                dw_dy = np.gradient(w_g, axes[1], axis=1)
                dv_dz = np.gradient(v_g, axes[2], axis=2)
                du_dz = np.gradient(u_g, axes[2], axis=2)
                dw_dx = np.gradient(w_g, axes[0], axis=0)
                dv_dx = np.gradient(v_g, axes[0], axis=0)
                du_dy = np.gradient(u_g, axes[1], axis=1)
                mag = np.sqrt((dw_dy - dv_dz) ** 2 + (du_dz - dw_dx) ** 2 + (dv_dx - du_dy) ** 2).reshape(-1)
                inv = np.empty_like(order)
                inv[order] = np.arange(len(order))
                return mag[inv]
        return None

    def _unstructured_vorticity_2d(self, rows: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        pts = self.points[rows][:, self.spatial_indices]
        if len(pts) < 4:
            return np.zeros(len(pts), dtype=np.float64)
        xmin, xmax = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
        ymin, ymax = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
        if xmax <= xmin or ymax <= ymin:
            return np.zeros(len(pts), dtype=np.float64)
        xs = np.linspace(xmin, xmax, 48)
        ys = np.linspace(ymin, ymax, 48)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        qgrid = np.stack([gx, gy], axis=-1)
        ug = _scattered_interp(pts, u, qgrid)
        vg = _scattered_interp(pts, v, qgrid)
        dv_dx = np.gradient(vg, xs, axis=0)
        du_dy = np.gradient(ug, ys, axis=1)
        vort = dv_dx - du_dy
        return _regular_grid_interp((xs, ys), np.nan_to_num(vort, nan=0.0), pts, fill_value=0.0)

    def field_values(
        self,
        field_name: str,
        *,
        mode: str = "prediction",
        time_index: int | None = None,
    ) -> np.ndarray:
        """Return 1D values for ``field_name`` on the selected time frame and mode."""
        pred = self._compute_field_on_frame(field_name, self.fields, time_index)
        if pred is None:
            raise KeyError(f"Field '{field_name}' is not available.")
        if mode == "prediction":
            return pred
        ref = self._compute_field_on_frame(field_name, self.references, time_index)
        if ref is None:
            raise ValueError(f"No reference data available for field '{field_name}'.")
        if mode == "reference":
            return ref
        if mode == "signed_error":
            return pred - ref
        if mode == "abs_error":
            return np.abs(pred - ref)
        if mode == "rel_error":
            return np.abs(pred - ref) / np.maximum(np.abs(ref), 1e-12)
        raise ValueError(f"Unknown display mode: {mode}")

    def has_reference_for(self, field_name: str, time_index: int | None = None) -> bool:
        return self._compute_field_on_frame(field_name, self.references, time_index) is not None

    def obstacle_mask_2d(self, xs: np.ndarray, ys: np.ndarray, *, slice_axis: str | None = None, slice_value: float | None = None) -> np.ndarray | None:
        obs = self.obstacle
        if not obs:
            return None
        kind = obs.get("kind")
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        if kind == "cylinder" and slice_axis is None:
            cx, cy = obs["center"][:2]
            r = float(obs["radius"])
            return (gx - cx) ** 2 + (gy - cy) ** 2 < (r * r)
        if kind == "sphere":
            cx, cy, cz = obs["center"]
            r = float(obs["radius"])
            if slice_axis == "z":
                dz = float(slice_value or 0.0) - cz
                return (gx - cx) ** 2 + (gy - cy) ** 2 + dz * dz < (r * r)
            if slice_axis == "y":
                dy = float(slice_value or 0.0) - cy
                return (gx - cx) ** 2 + dy * dy + (gy - cz) ** 2 < (r * r)
            if slice_axis == "x":
                dx = float(slice_value or 0.0) - cx
                return dx * dx + (gx - cy) ** 2 + (gy - cz) ** 2 < (r * r)
        return None

    def to_2d_grid(
        self,
        field_name: str,
        *,
        mode: str = "prediction",
        time_index: int | None = None,
        resolution: int = 96,
        slice_axis: str | None = None,
        slice_value: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, str]]:
        """Project a 2D field or a 3D orthogonal slice onto a regular 2D grid ``(ny, nx)``.

        Returns ``(xs, ys, grid_ny_nx, (x_label, y_label))`` where masked/outside
        locations are ``np.nan``.
        """
        if self.spatial_dim == 3:
            axis = (slice_axis or "z").lower()
            if axis not in ("x", "y", "z"):
                axis = "z"
            return self._slice_3d_to_2d(field_name, axis, slice_value, mode=mode, time_index=time_index, resolution=resolution)
        if self.spatial_dim != 2:
            raise ValueError("2D grid projection requires a 2D or 3D spatial dataset.")

        vals = self.field_values(field_name, mode=mode, time_index=time_index)
        axes, order = self.frame_grid(time_index)
        if axes is not None and order is not None:
            xs, ys = axes[0], axes[1]
            # frame_grid order is C-order over (x, y) -> shape (nx, ny); transpose to (ny, nx)
            grid = vals[order].reshape(len(xs), len(ys)).T.copy()
            obs_mask = self.obstacle_mask_2d(xs, ys)
            if obs_mask is not None:
                grid[obs_mask] = np.nan
            return xs, ys, grid, (self.spatial_names[0], self.spatial_names[1])

        pts = self.spatial_points(time_index)
        xmin, xmax = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
        ymin, ymax = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
        if xmax <= xmin:
            xmax = xmin + 1.0
        if ymax <= ymin:
            ymax = ymin + 1.0
        xs = np.linspace(xmin, xmax, max(16, int(resolution)))
        ys = np.linspace(ymin, ymax, max(16, int(resolution)))
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        grid = _scattered_interp(pts, vals, np.stack([gx, gy], axis=-1))
        obs_mask = self.obstacle_mask_2d(xs, ys)
        if obs_mask is not None:
            grid[obs_mask] = np.nan
        return xs, ys, grid, (self.spatial_names[0], self.spatial_names[1])

    def _slice_3d_to_2d(
        self,
        field_name: str,
        axis: str,
        slice_value: float | None,
        *,
        mode: str = "prediction",
        time_index: int | None = None,
        resolution: int = 64,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, str]]:
        axis_idx = ("x", "y", "z").index(axis)
        other_idx = [i for i in range(3) if i != axis_idx]
        labels = (self.spatial_names[other_idx[0]], self.spatial_names[other_idx[1]])
        vals = self.field_values(field_name, mode=mode, time_index=time_index)
        axes, order = self.frame_grid(time_index)
        if axes is not None and order is not None:
            nx, ny, nz = len(axes[0]), len(axes[1]), len(axes[2])
            vol = vals[order].reshape(nx, ny, nz)
            ax_coords = axes[axis_idx]
            target = float(0.5 * (ax_coords[0] + ax_coords[-1])) if slice_value is None else float(slice_value)
            target = max(float(ax_coords[0]), min(float(ax_coords[-1]), target))
            pos = int(np.searchsorted(ax_coords, target))
            if pos == 0:
                idx0, idx1, frac = 0, 0, 0.0
            elif pos >= len(ax_coords):
                idx0, idx1, frac = len(ax_coords) - 1, len(ax_coords) - 1, 0.0
            else:
                idx0, idx1 = pos - 1, pos
                denom = float(ax_coords[idx1] - ax_coords[idx0])
                frac = 0.0 if denom <= 0 else (target - float(ax_coords[idx0])) / denom
            sl0 = [slice(None), slice(None), slice(None)]
            sl1 = [slice(None), slice(None), slice(None)]
            sl0[axis_idx] = idx0
            sl1[axis_idx] = idx1
            plane = (1.0 - frac) * vol[tuple(sl0)] + frac * vol[tuple(sl1)]
            # plane has shape (len(axes[other_idx[0]]), len(axes[other_idx[1]])); transpose to (ny, nx)
            grid = plane.T.copy()
            xs, ys = axes[other_idx[0]], axes[other_idx[1]]
            obs_mask = self.obstacle_mask_2d(xs, ys, slice_axis=axis, slice_value=target)
            if obs_mask is not None:
                grid[obs_mask] = np.nan
            return xs, ys, grid, labels

        # Unstructured 3D point cloud: take thin slab around slice_value and triangulate in 2D
        pts = self.spatial_points(time_index)
        col = pts[:, axis_idx]
        cmin, cmax = float(np.min(col)), float(np.max(col))
        target = 0.5 * (cmin + cmax) if slice_value is None else float(slice_value)
        span = max(cmax - cmin, 1e-6)
        half_thick = 0.08 * span
        sel = np.abs(col - target) <= half_thick
        if int(np.sum(sel)) < 6:
            # Take nearest 15% of points to the slice plane
            k = max(6, min(len(pts), max(32, len(pts) // 6)))
            sel = np.argsort(np.abs(col - target))[:k]
        p2 = pts[sel][:, other_idx]
        v2 = vals[sel]
        x0, x1 = float(np.min(pts[:, other_idx[0]])), float(np.max(pts[:, other_idx[0]]))
        y0, y1 = float(np.min(pts[:, other_idx[1]])), float(np.max(pts[:, other_idx[1]]))
        xs = np.linspace(x0, x1, max(16, int(resolution)))
        ys = np.linspace(y0, y1, max(16, int(resolution)))
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        grid = _scattered_interp(p2, v2, np.stack([gx, gy], axis=-1))
        obs_mask = self.obstacle_mask_2d(xs, ys, slice_axis=axis, slice_value=target)
        if obs_mask is not None:
            grid[obs_mask] = np.nan
        return xs, ys, grid, labels

    def to_3d_grid(
        self,
        field_name: str,
        *,
        mode: str = "prediction",
        time_index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Return ``(xs, ys, zs, volume_nx_ny_nz)`` for a regular 3D dataset, else ``None``."""
        if self.spatial_dim != 3:
            return None
        axes, order = self.frame_grid(time_index)
        if axes is None or order is None:
            return None
        vals = self.field_values(field_name, mode=mode, time_index=time_index)
        vol = vals[order].reshape(len(axes[0]), len(axes[1]), len(axes[2])).copy()
        return axes[0], axes[1], axes[2], vol

    def vector_field_2d(
        self,
        *,
        mode: str = "prediction",
        time_index: int | None = None,
        resolution: int = 64,
        slice_axis: str | None = None,
        slice_value: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Return ``(xs, ys, vx_ny_nx, vy_ny_nx)`` for 2D quiver / streamline overlays."""
        if self.spatial_dim == 2:
            if not self.has_vector_2d:
                return None
            u_name, v_name = "u", "v"
        elif self.spatial_dim == 3:
            if not self.has_vector_3d:
                return None
            axis = (slice_axis or "z").lower()
            u_name, v_name = {"z": ("u", "v"), "y": ("u", "w"), "x": ("v", "w")}.get(axis, ("u", "v"))
        else:
            return None
        xs, ys, vx, _ = self.to_2d_grid(u_name, mode=mode, time_index=time_index, resolution=resolution, slice_axis=slice_axis, slice_value=slice_value)
        _, _, vy, _ = self.to_2d_grid(v_name, mode=mode, time_index=time_index, resolution=resolution, slice_axis=slice_axis, slice_value=slice_value)
        return xs, ys, vx, vy

    def probe_point(
        self,
        coord: tuple[float, ...],
        field_name: str,
        *,
        mode: str = "prediction",
        time_index: int | None = None,
    ) -> dict[str, Any] | None:
        """Return nearest-point (or interpolated) probe info at ``coord``."""
        pts = self.spatial_points(time_index)
        if len(pts) == 0:
            return None
        q = np.asarray(coord, dtype=np.float64).reshape(1, -1)
        if q.shape[1] != pts.shape[1]:
            return None
        dists = np.sum((pts - q) ** 2, axis=1)
        idx = int(np.argmin(dists))
        pred = self.field_values(field_name, mode="prediction", time_index=time_index)
        info: dict[str, Any] = {
            "point": tuple(float(v) for v in pts[idx]),
            "index": idx,
            "value": float(self.field_values(field_name, mode=mode, time_index=time_index)[idx]),
            "prediction": float(pred[idx]),
            "reference": None,
            "error": None,
        }
        if self.has_reference_for(field_name, time_index):
            ref = self.field_values(field_name, mode="reference", time_index=time_index)
            info["reference"] = float(ref[idx])
            info["error"] = float(pred[idx] - ref[idx])
        return info

    def sample_line(
        self,
        field_name: str,
        p0: tuple[float, ...],
        p1: tuple[float, ...],
        *,
        n_samples: int = 120,
        mode: str = "prediction",
        time_index: int | None = None,
    ) -> dict[str, np.ndarray | None]:
        """Sample a 1D line cut from ``p0`` to ``p1`` in 2D or 3D space."""
        a = np.asarray(p0, dtype=np.float64)
        b = np.asarray(p1, dtype=np.float64)
        t = np.linspace(0.0, 1.0, max(2, int(n_samples)))
        coords = a[None, :] * (1.0 - t[:, None]) + b[None, :] * t[:, None]
        s = t * float(np.linalg.norm(b - a))

        def eval_mode(m: str) -> np.ndarray:
            vals = self.field_values(field_name, mode=m, time_index=time_index)
            axes, order = self.frame_grid(time_index)
            if axes is not None and order is not None:
                shape = tuple(len(ax) for ax in axes)
                grid = vals[order].reshape(shape)
                return _regular_grid_interp(axes, grid, coords, fill_value=np.nan)
            pts = self.spatial_points(time_index)
            return _scattered_interp(pts, vals, coords, fill_value=np.nan)

        values = eval_mode(mode)
        ref_values = eval_mode("reference") if (mode == "prediction" and self.has_reference_for(field_name, time_index)) else None
        return {"s": s, "coords": coords, "values": values, "reference": ref_values}

    def time_trace(
        self,
        coord: tuple[float, ...],
        field_name: str,
        *,
        mode: str = "prediction",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Extract ``(times, values, ref_or_none)`` at the nearest point to ``coord`` across all frames."""
        if not self.is_time_dependent:
            return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), None
        vals = []
        refs = []
        has_ref = mode == "prediction" and self.has_reference_for(field_name, 0)
        for frame in range(self.num_frames):
            info = self.probe_point(coord, field_name, mode=mode, time_index=frame)
            vals.append(np.nan if info is None else info["value"])
            if has_ref:
                refs.append(np.nan if (info is None or info["reference"] is None) else info["reference"])
        return self.times.copy(), np.asarray(vals, dtype=np.float64), (np.asarray(refs, dtype=np.float64) if has_ref else None)

    def statistics(
        self,
        field_name: str,
        *,
        mode: str = "prediction",
        time_index: int | None = None,
    ) -> dict[str, Any]:
        vals = self.field_values(field_name, mode=mode, time_index=time_index)
        finite = vals[np.isfinite(vals)]
        stats: dict[str, Any] = {
            "count": int(len(finite)),
            "min": float(np.min(finite)) if len(finite) else float("nan"),
            "max": float(np.max(finite)) if len(finite) else float("nan"),
            "mean": float(np.mean(finite)) if len(finite) else float("nan"),
            "std": float(np.std(finite)) if len(finite) else float("nan"),
            "l2": float(np.sqrt(np.mean(finite**2))) if len(finite) else float("nan"),
            "rmse": None,
            "max_error": None,
            "rel_l2": None,
            "max_error_index": None,
            "max_error_point": None,
        }
        if self.has_reference_for(field_name, time_index):
            pred = self.field_values(field_name, mode="prediction", time_index=time_index)
            ref = self.field_values(field_name, mode="reference", time_index=time_index)
            err = pred - ref
            valid = np.isfinite(err)
            if np.any(valid):
                ev = err[valid]
                rv = ref[valid]
                abs_err = np.abs(err)
                best = int(np.nanargmax(abs_err))
                pts = self.spatial_points(time_index)
                denom = float(np.linalg.norm(rv))
                stats["rmse"] = float(np.sqrt(np.mean(ev**2)))
                stats["max_error"] = float(abs_err[best])
                stats["rel_l2"] = float(np.linalg.norm(ev) / denom) if denom > 0 else None
                stats["max_error_index"] = best
                stats["max_error_point"] = tuple(float(v) for v in pts[best])
        return stats

    def with_difference(self, baseline: "SolutionDataset", *, label: str = "Difference") -> "SolutionDataset":
        """Create a new dataset representing ``self - baseline`` on strictly matching points, IDs, and fields."""
        from ..run_comparison import validate_exact_alignment
        validate_exact_alignment(
            self.coordinate_names,
            baseline.coordinate_names,
            self.points,
            baseline.points,
            fields_a=list(self.fields),
            fields_b=list(baseline.fields),
            ids_a=self.point_ids,
            ids_b=baseline.point_ids,
        )
        diff_fields = {}
        for name in self.fields:
            a_vals = self.fields[name]
            b_vals = baseline.fields[name]
            if not np.isfinite(a_vals).all() or not np.isfinite(b_vals).all():
                raise ValueError(f"Field '{name}' contains non-finite values; difference cannot be computed.")
            diff_fields[name] = a_vals - b_vals
        return SolutionDataset(
            points=self.points.copy(),
            coordinate_names=self.coordinate_names,
            fields=diff_fields,
            references={},
            title=label,
            source_description=f"{self.title} − {baseline.title}",
            steps=self.steps,
            point_ids=list(self.point_ids) if self.point_ids is not None else None,
            obstacle=self.obstacle,
            metadata={"baseline": baseline.title, "compared": self.title},
        )
