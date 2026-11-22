"""Self-contained scientific colormaps (256-entry LUTs) with zero matplotlib dependency."""
from __future__ import annotations

import numpy as np

# Piecewise-linear RGB control stops (0..1) chosen to closely match standard
# perceptually uniform and diverging scientific colormaps.
_STOPS: dict[str, tuple[tuple[float, float, float], ...]] = {
    "viridis": (
        (0.2670, 0.0049, 0.3294),
        (0.2826, 0.1409, 0.4575),
        (0.2539, 0.2653, 0.5299),
        (0.2068, 0.3718, 0.5531),
        (0.1636, 0.4711, 0.5581),
        (0.1276, 0.5669, 0.5506),
        (0.1347, 0.6586, 0.5176),
        (0.2669, 0.7488, 0.4406),
        (0.4775, 0.8214, 0.3182),
        (0.7414, 0.8734, 0.1496),
        (0.9932, 0.9062, 0.1439),
    ),
    "plasma": (
        (0.0504, 0.0298, 0.5280),
        (0.2546, 0.0139, 0.6154),
        (0.4176, 0.0006, 0.6584),
        (0.5627, 0.0515, 0.6415),
        (0.6928, 0.1651, 0.5645),
        (0.7982, 0.2802, 0.4695),
        (0.8814, 0.3925, 0.3832),
        (0.9492, 0.5178, 0.2957),
        (0.9883, 0.6523, 0.2114),
        (0.9886, 0.8096, 0.1454),
        (0.9400, 0.9752, 0.1313),
    ),
    "inferno": (
        (0.0015, 0.0005, 0.0139),
        (0.0874, 0.0446, 0.2248),
        (0.2582, 0.0386, 0.4064),
        (0.4163, 0.0902, 0.4329),
        (0.5783, 0.1480, 0.4044),
        (0.7357, 0.2159, 0.3302),
        (0.8650, 0.3168, 0.2261),
        (0.9545, 0.4687, 0.1199),
        (0.9876, 0.6453, 0.0399),
        (0.9644, 0.8438, 0.2734),
        (0.9884, 0.9984, 0.6449),
    ),
    "cividis": (
        (0.0000, 0.1351, 0.3048),
        (0.0000, 0.1932, 0.4384),
        (0.1481, 0.2598, 0.4227),
        (0.2640, 0.3055, 0.4237),
        (0.3626, 0.3545, 0.4362),
        (0.4534, 0.4063, 0.4496),
        (0.5488, 0.4609, 0.4566),
        (0.6500, 0.5194, 0.4493),
        (0.7581, 0.5839, 0.4188),
        (0.8758, 0.6557, 0.3563),
        (0.9958, 0.9093, 0.2178),
    ),
    "turbo": (
        (0.1899, 0.0718, 0.2322),
        (0.2706, 0.3147, 0.7460),
        (0.1618, 0.5697, 0.9830),
        (0.1059, 0.7859, 0.8427),
        (0.2749, 0.9275, 0.5425),
        (0.6352, 0.9896, 0.2389),
        (0.8995, 0.8794, 0.1869),
        (0.9932, 0.6283, 0.1904),
        (0.9467, 0.3522, 0.1008),
        (0.7662, 0.1476, 0.0209),
        (0.4796, 0.0158, 0.0106),
    ),
    "coolwarm": (
        (0.2298, 0.2987, 0.7537),
        (0.3827, 0.5067, 0.9168),
        (0.5529, 0.6889, 0.9954),
        (0.7204, 0.8165, 0.9762),
        (0.8654, 0.8654, 0.8654),
        (0.9589, 0.7698, 0.6780),
        (0.9580, 0.6028, 0.4818),
        (0.8692, 0.3783, 0.3003),
        (0.7057, 0.0156, 0.1502),
    ),
    "rdbu": (
        (0.0196, 0.1882, 0.3804),
        (0.1294, 0.4000, 0.6745),
        (0.2627, 0.5765, 0.7647),
        (0.5725, 0.7725, 0.8706),
        (0.9686, 0.9686, 0.9686),
        (0.9569, 0.6471, 0.5098),
        (0.8392, 0.3765, 0.3020),
        (0.6980, 0.0941, 0.1686),
        (0.4039, 0.0000, 0.1216),
    ),
    "grayscale": (
        (0.05, 0.05, 0.05),
        (0.50, 0.50, 0.50),
        (0.96, 0.96, 0.96),
    ),
}

COLORMAP_LABELS: dict[str, str] = {
    "viridis": "Viridis",
    "plasma": "Plasma",
    "inferno": "Inferno",
    "cividis": "Cividis",
    "turbo": "Turbo",
    "coolwarm": "Coolwarm (diverging)",
    "rdbu": "RdBu (diverging)",
    "grayscale": "Grayscale",
}

COLORMAPS: tuple[str, ...] = tuple(_STOPS.keys())

_LUT_CACHE: dict[str, np.ndarray] = {}


def colormap_lut(name: str = "viridis") -> np.ndarray:
    """Return a cached (256, 3) uint8 RGB lookup table for ``name``."""
    key = name.strip().lower() if isinstance(name, str) else "viridis"
    if key not in _STOPS:
        key = "viridis"
    cached = _LUT_CACHE.get(key)
    if cached is not None:
        return cached
    stops = np.asarray(_STOPS[key], dtype=np.float64)
    x_stops = np.linspace(0.0, 1.0, len(stops))
    x_lut = np.linspace(0.0, 1.0, 256)
    rgb = np.empty((256, 3), dtype=np.uint8)
    for ch in range(3):
        channel = np.interp(x_lut, x_stops, stops[:, ch])
        rgb[:, ch] = np.clip(np.round(channel * 255.0), 0, 255).astype(np.uint8)
    rgb.setflags(write=False)
    _LUT_CACHE[key] = rgb
    return rgb


def map_scalars_to_rgba(
    values: np.ndarray,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = False,
    log_scale: bool = False,
    mask: np.ndarray | None = None,
    alpha: int = 255,
    invalid_rgba: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> tuple[np.ndarray, float, float]:
    """Map an arbitrary-shape scalar array to ``(*shape, 4)`` uint8 RGBA values.

    Returns ``(rgba, effective_vmin, effective_vmax)``.
    """
    arr = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(arr)
    if mask is not None:
        valid = valid & (~np.asarray(mask, dtype=bool))

    if log_scale:
        work = np.where(valid, np.log10(np.maximum(np.abs(arr), 1e-12)), np.nan)
    else:
        work = np.where(valid, arr, np.nan)

    if np.any(valid):
        finite_vals = work[valid]
        lo = float(np.min(finite_vals)) if vmin is None else (
            float(np.log10(max(abs(vmin), 1e-12))) if log_scale else float(vmin)
        )
        hi = float(np.max(finite_vals)) if vmax is None else (
            float(np.log10(max(abs(vmax), 1e-12))) if log_scale else float(vmax)
        )
        if symmetric and not log_scale:
            bound = max(abs(lo), abs(hi))
            lo, hi = -bound, bound
    else:
        lo, hi = 0.0, 1.0

    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = 0.0, 1.0
    if hi <= lo:
        span = max(abs(lo) * 0.05, 1e-6)
        lo -= span
        hi += span

    norm = np.zeros_like(work, dtype=np.float64)
    if np.any(valid):
        norm[valid] = np.clip((work[valid] - lo) / (hi - lo), 0.0, 1.0)
    indices = np.clip(np.round(norm * 255.0).astype(np.int64), 0, 255)
    lut = colormap_lut(cmap)
    rgba = np.empty(arr.shape + (4,), dtype=np.uint8)
    rgba[..., :3] = lut[indices]
    rgba[..., 3] = np.uint8(max(0, min(255, int(alpha))))
    if not np.all(valid):
        rgba[~valid] = np.asarray(invalid_rgba, dtype=np.uint8)

    eff_vmin = (10.0**lo) if log_scale else lo
    eff_vmax = (10.0**hi) if log_scale else hi
    return rgba, float(eff_vmin), float(eff_vmax)
