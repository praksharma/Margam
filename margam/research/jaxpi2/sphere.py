"""3D sphere-in-box geometry, spatial pools, cropped boundary reference, and step journal."""
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import numpy as np
import torch


@dataclass(frozen=True)
class Domain:
    """3D rectangular channel enclosing a centered sphere obstacle of radius r."""

    lower: tuple = (-8.0, -8.0, -8.0)
    upper: tuple = (24.0, 8.0, 8.0)
    radius: float = 0.5
    duration: float = 1.0

    def __post_init__(self):
        if len(self.lower) != 3 or len(self.upper) != 3:
            raise ValueError("Sphere domain must have three spatial axes.")
        if not all(math.isfinite(v) for v in (*self.lower, *self.upper, self.radius, self.duration)):
            raise ValueError("Domain parameters must be finite.")
        if (
            self.radius <= 0
            or self.duration <= 0
            or any(a >= -self.radius or b <= self.radius for a, b in zip(self.lower, self.upper))
        ):
            raise ValueError("Use a positive time span and a box strictly enclosing the sphere.")


@dataclass
class SpatialPools:
    """Pre-sampled 3D spatial point pools for fluid, near-sphere shell, wake, sphere, and 6 faces."""

    arrays: dict
    metadata: dict

    def identity(self):
        digest = hashlib.sha256()
        for name, array in sorted(self.arrays.items()):
            a = np.asarray(array)
            digest.update(name.encode())
            digest.update(str((a.shape, str(a.dtype))).encode())
            digest.update(a.tobytes())
        records = self.metadata.get("records", [])
        requests = [
            {
                k: r["request"][k]
                for k in (
                    "source_hash",
                    "label",
                    "count",
                    "method",
                    "seed",
                    "resolution",
                    "compute_mode",
                    "cpu_threads",
                )
                if k in r.get("request", {})
            }
            for r in records
        ]
        return dict(
            backend=self.metadata.get("backend", "jorvik-in-memory-spatial-pools"),
            arrays_sha256=digest.hexdigest(),
            requests=requests,
            versions=self.metadata.get(
                "versions",
                dict(jorvik=self.metadata.get("jorvik_version", "analytic")),
            ),
            sampler_source_sha256=self.metadata.get("sampler_source_sha256", "analytic-sphere-pools-v1"),
        )

    @classmethod
    def from_archive(cls, path):
        """Read a validated Jorvik spatial-pool NPZ archive and companion metadata.json."""
        path = Path(path)
        metadata = json.loads(path.with_name("metadata.json").read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError("Jorvik fixture checksum mismatch.")
        with np.load(path, allow_pickle=False) as data:
            return cls({name: data[name].copy() for name in data.files}, metadata)

    @classmethod
    def from_domain(
        cls,
        domain: Domain = Domain(),
        *,
        volume_count: int = 256,
        boundary_count: int = 64,
        near_outer_radius: float = 0.6,
        wake_upper_x: float = 8.0,
        wake_half_width: float = 1.0,
        seed: int = 42,
    ):
        """Construct deterministic analytic 3D spatial pools without external CAD files."""
        if volume_count < 1 or boundary_count < 1:
            raise ValueError("Pool counts must be positive integers.")
        rng = np.random.default_rng(seed)
        low = np.asarray(domain.lower, dtype=np.float64)
        high = np.asarray(domain.upper, dtype=np.float64)
        r_in = float(domain.radius)

        def sample_box_outside_sphere(lo, hi, count, max_radius=None):
            collected = []
            total = 0
            while total < count:
                pts = lo + rng.random((max(count * 2, 128), 3)) * (hi - lo)
                rad = np.linalg.norm(pts, axis=1)
                mask = rad > r_in
                if max_radius is not None:
                    mask &= rad < max_radius
                valid = pts[mask]
                if len(valid):
                    collected.append(valid)
                    total += len(valid)
            return np.concatenate(collected, axis=0)[:count]

        fluid = sample_box_outside_sphere(low, high, volume_count)
        near_lo = np.full(3, -near_outer_radius, dtype=np.float64)
        near_hi = np.full(3, near_outer_radius, dtype=np.float64)
        near = sample_box_outside_sphere(near_lo, near_hi, volume_count, max_radius=near_outer_radius)
        wake_lo = np.array([r_in, -wake_half_width, -wake_half_width], dtype=np.float64)
        wake_hi = np.array([wake_upper_x, wake_half_width, wake_half_width], dtype=np.float64)
        wake = sample_box_outside_sphere(wake_lo, wake_hi, volume_count)

        normals = rng.standard_normal((boundary_count, 3))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        sphere = normals * r_in

        arrays = {
            "fluid": fluid,
            "near": near,
            "wake": wake,
            "sphere": sphere,
            "sphere_normals": -normals,
        }
        for axis in range(3):
            for side in (0, 1):
                pts = low + rng.random((boundary_count, 3)) * (high - low)
                pts[:, axis] = low[axis] if side == 0 else high[axis]
                arrays[f"face_{axis}_{side}"] = pts

        metadata = dict(
            backend="analytic-sphere-pools",
            jorvik_version="analytic-1.0",
            sampler_source_sha256="analytic-sphere-pools-v1",
            domain=dict(lower=list(domain.lower), upper=list(domain.upper), radius=domain.radius),
            refinement=dict(
                near_outer_radius=near_outer_radius,
                wake_upper_x=wake_upper_x,
                wake_half_width=wake_half_width,
            ),
            records=[],
        )
        return cls(arrays, metadata)


class SphereSampler:
    """Space-time batch sampler for 3D flow past a sphere with near-wall and wake refinement."""

    def __init__(
        self,
        domain: Domain,
        initial_q: torch.Tensor,
        initial_values: torch.Tensor,
        *,
        residual_count: int = 4096,
        boundary_count: int = 2048,
        initial_count: int = 2048,
        seed: int = 42,
        near_fraction: float = 0.25,
        wake_fraction: float = 0.25,
        near_outer_radius: float = 0.6,
        wake_upper_x: float = 8.0,
        wake_half_width: float = 1.0,
        spatial_pools: SpatialPools | None = None,
    ):
        self.domain = domain
        self.initial_q, self.initial_values = initial_q.cpu(), initial_values.cpu()
        self.residual_count, self.boundary_count = residual_count, boundary_count
        self.initial_count = initial_count
        if any(type(n) is not int or n < 1 for n in (residual_count, boundary_count, initial_count)):
            raise ValueError("Sample budgets must be positive integers.")
        if not all(
            math.isfinite(v)
            for v in (near_fraction, wake_fraction, near_outer_radius, wake_upper_x, wake_half_width)
        ):
            raise ValueError("Refinement settings must be finite.")
        if min(near_fraction, wake_fraction) < 0 or near_fraction + wake_fraction >= 1:
            raise ValueError("Keep nonnegative fine fractions and a positive full-domain fraction.")
        self.near_count = round(residual_count * near_fraction)
        self.wake_count = round(residual_count * wake_fraction)
        if (
            (near_fraction > 0 and not self.near_count)
            or (wake_fraction > 0 and not self.wake_count)
            or self.near_count + self.wake_count >= residual_count
        ):
            raise ValueError("Increase the PDE budget so all requested sampling regions have points.")
        if self.near_count and not domain.radius < near_outer_radius <= min(
            *(-v for v in domain.lower), *domain.upper
        ):
            raise ValueError("Near-wall shell must enclose the sphere and fit inside the box.")
        self.wake_lower = self.initial_q.new_tensor(
            (
                domain.radius,
                max(domain.lower[1], -wake_half_width),
                max(domain.lower[2], -wake_half_width),
            )
        )
        self.wake_upper = self.initial_q.new_tensor(
            (
                min(domain.upper[0], wake_upper_x),
                min(domain.upper[1], wake_half_width),
                min(domain.upper[2], wake_half_width),
            )
        )
        if self.wake_count and (wake_half_width <= 0 or (self.wake_upper <= self.wake_lower).any()):
            raise ValueError("Wake refinement box must have positive extent in the fluid.")
        self.refinement = dict(
            near_fraction=near_fraction,
            wake_fraction=wake_fraction,
            near_outer_radius=near_outer_radius,
            wake_upper_x=wake_upper_x,
            wake_half_width=wake_half_width,
        )
        if spatial_pools is None:
            spatial_pools = SpatialPools.from_domain(
                domain,
                near_outer_radius=near_outer_radius,
                wake_upper_x=wake_upper_x,
                wake_half_width=wake_half_width,
                seed=seed,
            )
        metadata = spatial_pools.metadata
        expected_domain = dict(lower=list(domain.lower), upper=list(domain.upper), radius=domain.radius)
        expected_refinement = {
            k: self.refinement[k] for k in ("near_outer_radius", "wake_upper_x", "wake_half_width")
        }
        if metadata["domain"] != expected_domain or metadata["refinement"] != expected_refinement:
            raise ValueError("Spatial pools do not match the requested geometry and refinement regions.")
        self.pool_identity = spatial_pools.identity()
        names = ("fluid", "near", "wake", "sphere", *(f"face_{a}_{s}" for a in range(3) for s in (0, 1)))
        self.pools = {
            name: torch.tensor(spatial_pools.arrays[name], dtype=self.initial_q.dtype) for name in names
        }
        if any(
            v.ndim != 2 or v.shape[1] != 3 or not len(v) or not torch.isfinite(v).all()
            for v in self.pools.values()
        ):
            raise ValueError("Spatial pools must contain finite, nonempty (N,3) arrays.")
        self.generator = torch.Generator().manual_seed(seed)

    def identity(self):
        return dict(self.pool_identity, refinement=self.refinement)

    def pick(self, name: str, n: int) -> torch.Tensor:
        pool = self.pools[name]
        indices = torch.randint(len(pool), (n,), generator=self.generator)
        return pool[indices]

    def uniform(self, n: int, d: int) -> torch.Tensor:
        return torch.rand(n, d, generator=self.generator, dtype=self.initial_q.dtype)

    def fluid(self, n: int) -> torch.Tensor:
        return self.pick("fluid", n)

    def spacetime(self, xyz: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.uniform(len(xyz), 1) * self.domain.duration, xyz), dim=1)

    def draw(self) -> dict:
        coarse_count = self.residual_count - self.near_count - self.wake_count
        near = self.pick("near", self.near_count)
        wake = self.pick("wake", self.wake_count)
        xyz = torch.cat((self.fluid(coarse_count), near, wake))
        regions = torch.tensor([0] * coarse_count + [1] * self.near_count + [2] * self.wake_count)
        res = self.spacetime(xyz)
        order = res[:, 0].argsort()
        res, regions = res[order], regions[order]
        faces = {}
        for axis in range(3):
            for side in (0, 1):
                xyz_face = self.pick(f"face_{axis}_{side}", self.boundary_count)
                faces[(axis, side)] = self.spacetime(xyz_face)
        surface = self.pick("sphere", self.boundary_count)
        indices = torch.randint(len(self.initial_q), (self.initial_count,), generator=self.generator)
        return dict(
            res=res,
            res_regions=regions,
            faces=faces,
            sphere=self.spacetime(surface),
            initial=self.initial_q[indices],
            values=self.initial_values[indices],
        )


def _multilinear_interpolate_3d(grid_axes, values, coordinates):
    """Pure-NumPy 3D multilinear interpolator over rectilinear (t, u, v) grids."""
    coords = np.asarray(coordinates, dtype=np.float64)
    n_pts = len(coords)
    idx_low = []
    weights = []
    for dim, axis in enumerate(grid_axes):
        x = coords[:, dim]
        if (x < axis[0] - 1e-12).any() or (x > axis[-1] + 1e-12).any():
            raise ValueError("Interpolation coordinate out of bounds.")
        x_clamped = np.clip(x, axis[0], axis[-1])
        hi = np.searchsorted(axis, x_clamped, side="right")
        hi = np.clip(hi, 1, len(axis) - 1)
        lo = hi - 1
        span = axis[hi] - axis[lo]
        w = np.where(span > 0, (x_clamped - axis[lo]) / span, 0.0)
        idx_low.append(lo)
        weights.append(w)
    out = np.zeros((n_pts, values.shape[-1]), dtype=np.float64)
    for b0, b1, b2 in itertools.product((0, 1), repeat=3):
        i0 = idx_low[0] + b0
        i1 = idx_low[1] + b1
        i2 = idx_low[2] + b2
        w0 = weights[0] if b0 else (1.0 - weights[0])
        w1 = weights[1] if b1 else (1.0 - weights[1])
        w2 = weights[2] if b2 else (1.0 - weights[2])
        w = (w0 * w1 * w2)[:, None]
        out += w * values[i0, i1, i2]
    return out


class CropBoundaryReference:
    """Read outer box faces and corner pressure gauge for t in a declared local window.

    Raw layout is (snapshot, p/u/v/w, x, y, z). Gauge subtracts the pressure at the
    high-x/high-y/high-z corner each snapshot.
    """

    def __init__(self, array, axes, dt: float, duration: float):
        if not math.isfinite(dt) or not math.isfinite(duration) or dt <= 0 or duration <= 0:
            raise ValueError("Positive finite dt and window duration required.")
        last = math.ceil(duration / dt)
        if last >= len(array):
            raise ValueError("Reference does not cover requested window.")
        self.dt, self.duration = dt, duration
        self.axes = tuple(np.asarray(a, dtype=np.float64) for a in axes)
        self.times = np.arange(last + 1) * dt
        self.gauge = np.array(array[: last + 1, 0, -1, -1, -1], copy=True)
        self.faces = {}
        digest = hashlib.sha256()
        digest.update(np.asarray([dt, duration], dtype=np.float64).tobytes())
        for a in self.axes:
            digest.update(a.tobytes())
        digest.update(self.gauge.tobytes())
        try:
            from scipy.interpolate import RegularGridInterpolator
        except ImportError:
            RegularGridInterpolator = None
        for axis in range(3):
            for side in (0, 1):
                select = [slice(0, last + 1), slice(None), slice(None), slice(None), slice(None)]
                select[axis + 2] = 0 if side == 0 else -1
                raw = np.array(array[tuple(select)], copy=True)
                values = raw[:, [1, 2, 3, 0]].transpose(0, 2, 3, 1).copy()
                values[..., 3] -= self.gauge[:, None, None]
                digest.update(values.tobytes())
                tangential = [a for a in range(3) if a != axis]
                grid_axes = (self.times, *(self.axes[a] for a in tangential))
                if RegularGridInterpolator is not None:
                    self.faces[(axis, side)] = RegularGridInterpolator(
                        grid_axes, values, bounds_error=True
                    )
                else:
                    self.faces[(axis, side)] = (
                        lambda coords, ga=grid_axes, val=values: _multilinear_interpolate_3d(ga, val, coords)
                    )
        self.sha256 = digest.hexdigest()

    def face_values(self, q, axis: int, side: int):
        points = np.asarray(q, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
            raise ValueError("Finite txyz query required.")
        if (points[:, 0] < 0).any() or (points[:, 0] > self.duration).any():
            raise ValueError("Query outside declared time window.")
        expected = self.axes[axis][0 if side == 0 else -1]
        if not np.allclose(points[:, axis + 1], expected, atol=1e-10, rtol=0):
            raise ValueError("Query is not on requested boundary face.")
        coordinates = points[:, [0, *(a + 1 for a in range(3) if a != axis)]]
        return self.faces[(axis, side)](coordinates)


class CropSampler(SphereSampler):
    """Cropped-domain sphere sampler attaching interpolated outer face values and pressure anchor."""

    def __init__(self, *args, reference: CropBoundaryReference, **kwargs):
        self.reference = reference
        super().__init__(*args, **kwargs)
        if self.domain.duration != reference.duration or any(
            not np.isclose(
                [self.domain.lower[a], self.domain.upper[a]],
                reference.axes[a][[0, -1]],
                atol=1e-12,
                rtol=0,
            ).all()
            for a in range(3)
        ):
            raise ValueError("Boundary reference and sampler domain differ.")

    def identity(self):
        return dict(
            super().identity(),
            boundary_reference_sha256=self.reference.sha256,
            boundary_kind="six-face-velocity-dirichlet",
            pressure_gauge="high_xyz_corner_zero_each_time",
            snapshot_dt=self.reference.dt,
        )

    def draw(self) -> dict:
        batch = super().draw()
        batch["face_values"] = {
            face: torch.tensor(self.reference.face_values(q.numpy(), *face), dtype=q.dtype)
            for face, q in batch["faces"].items()
        }
        anchor = batch["faces"][(0, 1)].clone()
        anchor[:, 1:] = anchor.new_tensor(self.domain.upper)
        batch["pressure_anchor"] = anchor
        return batch


class StepJournal:
    """Append-only JSONL research records; checkpoints refer to a durable byte position."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.stream = self.path.open("xb")
        self.records = 0
        self.last_step = None

    def position(self):
        return dict(
            path=str(self.path),
            byte_offset=self.stream.tell(),
            records=self.records,
            last_step=self.last_step,
        )

    def append(self, event: dict):
        if self.last_step is not None and event["step"] != self.last_step + 1:
            raise ValueError("Journal steps must be consecutive.")
        payload = (json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n").encode()
        if self.stream.write(payload) != len(payload):
            raise OSError("Incomplete journal write; restore a valid checkpoint into a new run.")
        self.records += 1
        self.last_step = event["step"]
        return self.position()

    def sync(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())
        return self.position()

    def close(self):
        self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
