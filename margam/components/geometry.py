"""Shared obstacle predicates, fluid-domain sampling, and exact fluid measures.

Keeps initial sampling, adaptive candidate generation, evaluation grids,
supplied-point validation, integral-loss volume measures, and visualization
obstacle overlays on one consistent geometry contract.
"""
from __future__ import annotations

import math
import numpy as np
from .sampling import sample_box


def obstacle_spec(config) -> dict | None:
    """Return the built-in solid obstacle descriptor for ``config``, or ``None``."""
    if config is None:
        return None
    shape = getattr(config, 'shape', '')
    if shape == 'SphereObstacle3D':
        return {
            'kind': 'sphere',
            'center': (
                float(getattr(config, 'sphere_x', 0.5)),
                float(getattr(config, 'sphere_y', 0.5)),
                float(getattr(config, 'sphere_z', 0.5)),
            ),
            'radius': float(getattr(config, 'sphere_radius', 0.2)),
        }
    if getattr(config, 'has_obstacle', False):
        return {
            'kind': 'cylinder',
            'center': (
                float(getattr(config, 'cylinder_x', 0.2)),
                float(getattr(config, 'cylinder_y', 0.2)),
            ),
            'radius': float(getattr(config, 'cylinder_radius', 0.05)),
        }
    return None


def fluid_mask(points, obstacle: dict | None, *, strict_interior: bool = False, tolerance: float = 0.0) -> np.ndarray:
    """Return a boolean mask selecting rows outside the solid obstacle.

    When ``strict_interior=True``, points strictly outside the obstacle surface
    (``r > radius``) are accepted; otherwise points on or outside the surface
    (``r >= radius - tolerance``) are accepted.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None, :]
    if not obstacle or len(pts) == 0:
        return np.ones(len(pts), dtype=bool)
    kind = obstacle.get('kind')
    radius = float(obstacle['radius'])
    center = obstacle['center']
    if kind == 'sphere':
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        dist2 = (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2 + (pts[:, 2] - cz) ** 2
    elif kind == 'cylinder':
        cx, cy = float(center[0]), float(center[1])
        dist2 = (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2
    else:
        return np.ones(len(pts), dtype=bool)
    if strict_interior:
        return dist2 > radius * radius
    thresh = max(0.0, radius - max(0.0, float(tolerance)))
    return dist2 >= thresh * thresh


def is_in_fluid(row, obstacle: dict | None, *, strict_interior: bool = True, tolerance: float = 0.0) -> bool:
    """Scalar convenience check for a single coordinate row."""
    if not obstacle:
        return True
    return bool(fluid_mask(np.asarray(row, dtype=np.float64)[None, :], obstacle, strict_interior=strict_interior, tolerance=tolerance)[0])


def fluid_spatial_measure(config) -> float:
    """Return the exact spatial area/volume of the fluid domain (excluding solid obstacles)."""
    lower, upper = getattr(config, 'spatial_time_bounds', config.bounds)
    if getattr(config, 'is_3d', False):
        dx, dy, dz = upper[0] - lower[0], upper[1] - lower[1], upper[2] - lower[2]
        vol = dx * dy * dz
        if getattr(config, 'shape', '') == 'SphereObstacle3D':
            vol -= (4.0 / 3.0) * math.pi * (float(config.sphere_radius) ** 3)
        if vol <= 0.0:
            raise ValueError('Fluid domain volume must remain strictly positive after excluding the sphere obstacle.')
        return vol
    if getattr(config, 'shape', '') in ('Interval', 'SpaceTime1D'):
        return float(upper[0] - lower[0])
    dx, dy = upper[0] - lower[0], upper[1] - lower[1]
    area = dx * dy
    if getattr(config, 'has_obstacle', False):
        area -= math.pi * (float(config.cylinder_radius) ** 2)
    if area <= 0.0:
        raise ValueError('Fluid domain area must remain strictly positive after excluding the cylinder obstacle.')
    return area


def sample_fluid_box(lower, upper, count: int, method: str, seed: int, obstacle: dict | None = None) -> list[list[float]]:
    """Sample up to ``count`` points strictly in the fluid region of a bounding box."""
    if not obstacle:
        return sample_box(lower, upper, count, method, seed)
    if count <= 0:
        return []
    # Estimate obstacle fraction so rejection sampling usually fills count in one pass.
    box_vol = 1.0
    obs_vol = 0.0
    if obstacle.get('kind') == 'sphere' and len(lower) >= 3:
        box_vol = max(1e-12, (upper[0] - lower[0]) * (upper[1] - lower[1]) * (upper[2] - lower[2]))
        obs_vol = (4.0 / 3.0) * math.pi * (float(obstacle['radius']) ** 3)
    elif obstacle.get('kind') == 'cylinder' and len(lower) >= 2:
        box_vol = max(1e-12, (upper[0] - lower[0]) * (upper[1] - lower[1]))
        obs_vol = math.pi * (float(obstacle['radius']) ** 2)
    fluid_frac = max(0.05, min(1.0, 1.0 - obs_vol / box_vol))
    oversample = min(100000, max(count, int(math.ceil(count / fluid_frac * 1.25))))
    collected: list[list[float]] = []
    # For structured/grid/hammersley/stratified initial draws, use the requested method on attempt 0,
    # then deterministic random retries only if needed to reach count for random sampling.
    max_attempts = 8 if method == 'random' else 1
    for attempt in range(max_attempts):
        need = count - len(collected)
        if need <= 0:
            break
        batch_n = oversample if attempt == 0 else min(100000, max(need, int(math.ceil(need / fluid_frac * 1.5))))
        attempt_seed = (int(seed) + attempt * 104729) % (2 ** 31)
        raw = sample_box(lower, upper, batch_n, method if attempt == 0 else 'random', attempt_seed)
        if not raw:
            continue
        mask = fluid_mask(raw, obstacle, strict_interior=True)
        for row, keep in zip(raw, mask):
            if keep:
                collected.append(row)
                if len(collected) >= count:
                    break
    if not collected:
        raise ValueError('No interior points remained outside the solid obstacle; increase point budget or shrink the obstacle.')
    return collected[:count]
