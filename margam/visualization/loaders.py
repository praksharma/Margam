"""Adapters converting Margam runs, studies, research arrays, and files to SolutionDataset."""
from __future__ import annotations

from collections import OrderedDict
import csv
import json
from pathlib import Path
from typing import Any
import numpy as np

from .dataset import SolutionDataset


def _obstacle_from_config(config) -> dict[str, Any] | None:
    if config is None:
        return None
    if getattr(config, "shape", "") == "SphereObstacle3D":
        return {
            "kind": "sphere",
            "center": (
                float(getattr(config, "sphere_x", getattr(config, "cylinder_x", 0.5))),
                float(getattr(config, "sphere_y", getattr(config, "cylinder_y", 0.5))),
                float(getattr(config, "sphere_z", 0.5)),
            ),
            "radius": float(getattr(config, "sphere_radius", getattr(config, "cylinder_radius", 0.2))),
        }
    if getattr(config, "has_obstacle", False):
        return {
            "kind": "cylinder",
            "center": (float(getattr(config, "cylinder_x", 0.2)), float(getattr(config, "cylinder_y", 0.2))),
            "radius": float(getattr(config, "cylinder_radius", 0.05)),
        }
    return None


def _build_live_evaluator(model, config):
    """Wrap a trained Margam model for debounced CPU/CUDA parametric/time inference."""
    if model is None or config is None or not hasattr(model, "predict"):
        return None
    coord_names = tuple(getattr(config, "coordinate_names", ("x", "y")))
    field_names = tuple(getattr(config, "field_names", ("u",)))
    param_axes = list(getattr(config, "parametric_axes", ()) or ())

    def evaluate(spatial_points: np.ndarray, *, time_value: float | None = None, params: dict[str, float] | None = None):
        import torch
        from ..components.prediction import predict_in_batches

        pts = np.asarray(spatial_points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts[:, None]
        n = len(pts)
        cols = []
        s_idx = 0
        param_map = dict(params or {})
        param_defaults = {p["name"]: float(p.get("default", 0.5 * (p["min"] + p["max"]))) for p in param_axes}
        for name in coord_names:
            if name in ("x", "y", "z"):
                cols.append(pts[:, s_idx : s_idx + 1])
                s_idx += 1
            elif name == "t":
                tv = float(time_value) if time_value is not None else float(getattr(config, "t_min", getattr(config, "y_min", 0.0)))
                cols.append(np.full((n, 1), tv, dtype=np.float64))
            else:
                pv = float(param_map.get(name, param_defaults.get(name, 0.0)))
                cols.append(np.full((n, 1), pv, dtype=np.float64))
        full = np.hstack(cols)
        dtype = getattr(model, "dtype", torch.get_default_dtype())
        if not isinstance(dtype, torch.dtype):
            dtype = torch.get_default_dtype()
        tensor = torch.tensor(full, dtype=dtype)
        pred = predict_in_batches(model, tensor, 2048).detach().cpu().numpy()
        if pred.ndim == 1:
            pred = pred[:, None]
        return {fname: pred[:, idx].astype(np.float64) for idx, fname in enumerate(field_names)}

    return evaluate


def from_output_result(
    data: dict[str, Any],
    *,
    title: str = "Solution",
    live_model: Any = None,
) -> SolutionDataset:
    """Build a :class:`SolutionDataset` from a Margam desktop output result dict."""
    config = data["config"]
    points = np.asarray(data["points"], dtype=np.float64)
    if points.ndim == 1:
        points = points[:, None]
    coord_names = tuple(getattr(config, "coordinate_names", tuple(f"x{i}" for i in range(points.shape[1]))))
    # Ensure coordinate_names matches point columns even when parametric columns are appended
    if len(coord_names) != points.shape[1]:
        spatial_base = ("x", "y", "z", "t")[: points.shape[1]]
        coord_names = tuple(spatial_base)

    field_names = tuple(data.get("fields") or getattr(config, "field_names", ("u",)))
    raw_vals = np.asarray(data["values"], dtype=np.float64)
    if raw_vals.ndim == 1:
        raw_vals = raw_vals[:, None]
    fields = {name: raw_vals[:, idx] for idx, name in enumerate(field_names)}

    references: dict[str, np.ndarray] = {}
    if data.get("reference") is not None:
        raw_ref = np.asarray(data["reference"], dtype=np.float64)
        if raw_ref.ndim == 1:
            raw_ref = raw_ref[:, None]
        for idx, name in enumerate(field_names):
            if idx < raw_ref.shape[1]:
                references[name] = raw_ref[:, idx]

    param_specs = [dict(p) for p in (getattr(config, "parametric_axes", ()) or ())]
    lower, upper = config.bounds
    spatial_bounds = [
        (float(lo), float(hi))
        for name, lo, hi in zip(coord_names, lower, upper)
        if name in ("x", "y", "z")
    ]
    time_bounds = None
    for name, lo, hi in zip(coord_names, lower, upper):
        if name == "t":
            time_bounds = (float(lo), float(hi))
            break

    return SolutionDataset(
        points=points,
        coordinate_names=coord_names,
        fields=fields,
        references=references,
        title=title,
        source_description=f"{getattr(config, 'shape', 'Domain')} · {len(points)} rows",
        steps=data.get("steps"),
        point_ids=data.get("point_ids"),
        obstacle=_obstacle_from_config(config),
        metadata={
            "shape": getattr(config, "shape", ""),
            "physics": getattr(config, "physics", ""),
            "method": getattr(config, "method", ""),
            "reference_name": data.get("reference_name"),
        },
        live_evaluator=_build_live_evaluator(live_model, config),
        param_specs=param_specs,
        spatial_bounds=spatial_bounds or None,
        time_bounds=time_bounds,
    )


def from_run_archive(path: str | Path) -> dict[str, SolutionDataset]:
    """Safely load all prediction/comparison outputs from a ``.margam-run.pt`` archive."""
    from ..ui.run_document import load_run
    from ..ui.model_document import decode_document

    archive = load_run(path)
    config, records, _ = decode_document(archive["model"])
    titles = {rec["id"]: rec["title"] for rec in records}
    outputs = archive.get("result", {}).get("outputs", {})
    live_model = None
    if archive.get("checkpoint") is not None and getattr(config, "parametric_axes", ()):
        checkpoint = archive["checkpoint"]

        class _LazyCheckpointModel:
            def __init__(self):
                self._model = None

            def _ensure(self):
                if self._model is None:
                    from ..components.prediction import prediction_model
                    self._model = prediction_model(config, checkpoint)
                return self._model

            @property
            def dtype(self):
                return self._ensure().dtype

            def predict(self, points):
                return self._ensure().predict(points)

        live_model = _LazyCheckpointModel()

    datasets: dict[str, SolutionDataset] = {}
    for identifier, payload in outputs.items():
        merged = dict(payload)
        merged["config"] = config
        label = f"{titles.get(identifier, identifier)} ({Path(path).name})"
        datasets[identifier] = from_output_result(merged, title=label, live_model=live_model)
    return datasets


def from_arrays(
    points: np.ndarray,
    fields: dict[str, np.ndarray],
    *,
    coordinate_names: tuple[str, ...] | None = None,
    references: dict[str, np.ndarray] | None = None,
    title: str = "Array solution",
    obstacle: dict[str, Any] | None = None,
    live_evaluator=None,
    param_specs: list[dict[str, Any]] | None = None,
) -> SolutionDataset:
    """Construct a :class:`SolutionDataset` directly from NumPy/PyTorch arrays."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[:, None]
    if coordinate_names is None:
        defaults = {1: ("x",), 2: ("x", "y"), 3: ("x", "y", "z"), 4: ("x", "y", "z", "t")}
        coordinate_names = defaults.get(pts.shape[1], tuple(f"x{i}" for i in range(pts.shape[1])))
    return SolutionDataset(
        points=pts,
        coordinate_names=tuple(coordinate_names),
        fields={k: np.asarray(v, dtype=np.float64).reshape(-1) for k, v in fields.items()},
        references={k: np.asarray(v, dtype=np.float64).reshape(-1) for k, v in (references or {}).items()},
        title=title,
        source_description=f"In-memory arrays · {len(pts)} points",
        obstacle=obstacle,
        live_evaluator=live_evaluator,
        param_specs=list(param_specs or []),
    )


def from_file(path: str | Path) -> SolutionDataset:
    """Load a 1D, 2D, 3D, or transient solution from ``.npz``, ``.csv``, or ``.nc``."""
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix == ".npz":
        return _load_npz(target)
    if suffix == ".csv":
        return _load_csv(target)
    if suffix == ".nc":
        return _load_netcdf(target)
    if suffix == ".pt":
        runs = from_run_archive(target)
        if not runs:
            raise ValueError("Run archive contains no saved prediction outputs.")
        return next(iter(runs.values()))
    raise ValueError(f"Unsupported solution file format: {suffix}. Use .margam-run.pt, .npz, .csv, or .nc.")


def _load_npz(path: Path) -> SolutionDataset:
    with np.load(path, allow_pickle=False) as archive:
        keys = list(archive.files)
        arrays = {k: np.asarray(archive[k]) for k in keys}

    # Case 1: explicit 'points' (N, D) array
    coord_names: list[str] = []
    if "points" in arrays and arrays["points"].ndim == 2:
        pts = arrays["points"].astype(np.float64)
        if "coordinate_names" in arrays:
            coord_names = [str(v) for v in arrays["coordinate_names"].tolist()]
        else:
            coord_names = list({1: ("x",), 2: ("x", "y"), 3: ("x", "y", "z"), 4: ("x", "y", "z", "t")}.get(pts.shape[1], tuple(f"x{i}" for i in range(pts.shape[1]))))
        used = {"points", "coordinate_names"}
    else:
        # Case 2: separate 1D/ND coordinate arrays x, y, z, t
        cols = []
        used = set()
        for cname in ("x", "y", "z", "t"):
            if cname in arrays and arrays[cname].ndim >= 1:
                cols.append(arrays[cname].astype(np.float64))
                coord_names.append(cname)
                used.add(cname)
        if not cols:
            raise ValueError("NPZ file must contain a 'points' array or coordinate arrays ('x', 'y', 'z', 't').")
        # If coordinate arrays are 1D axes of a regular meshgrid and field arrays match their product shape
        lengths = [c.size for c in cols]
        if all(c.ndim == 1 for c in cols) and len(set(lengths)) > 1:
            mesh = np.meshgrid(*cols, indexing="ij")
            pts = np.column_stack([m.reshape(-1) for m in mesh])
        else:
            pts = np.column_stack([c.reshape(-1) for c in cols])

    n = len(pts)
    fields: dict[str, np.ndarray] = {}
    references: dict[str, np.ndarray] = {}
    for k, arr in arrays.items():
        if k in used or arr.dtype.kind not in ("f", "i", "u"):
            continue
        if arr.size == n:
            flat = arr.astype(np.float64).reshape(-1)
            if k.startswith("ref_"):
                references[k[4:]] = flat
            elif k.startswith("ref "):
                references[k[4:]] = flat
            elif k.endswith("_ref"):
                references[k[:-4]] = flat
            elif not k.startswith("error"):
                fields[k] = flat
        elif arr.ndim == 2 and arr.shape[0] == n:
            if k in ("values", "pred", "prediction", "fields"):
                default_names = ("u", "v", "w", "p") if arr.shape[1] == 4 else ("u", "v", "p") if arr.shape[1] == 3 else tuple(f"f{i}" for i in range(arr.shape[1]))
                for idx in range(arr.shape[1]):
                    fields[default_names[idx] if idx < len(default_names) else f"f{idx}"] = arr[:, idx].astype(np.float64)
            elif k in ("reference", "ref", "truth"):
                default_names = ("u", "v", "w", "p") if arr.shape[1] == 4 else ("u", "v", "p") if arr.shape[1] == 3 else tuple(f"f{i}" for i in range(arr.shape[1]))
                for idx in range(arr.shape[1]):
                    references[default_names[idx] if idx < len(default_names) else f"f{idx}"] = arr[:, idx].astype(np.float64)

    if not fields:
        raise ValueError("NPZ file contains no field arrays matching the coordinate point count.")

    return SolutionDataset(
        points=pts,
        coordinate_names=tuple(coord_names),
        fields=fields,
        references=references,
        title=path.name,
        source_description=f"NPZ · {n} points",
    )


def _load_csv(path: Path) -> SolutionDataset:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV file has no header row.")
        headers = [h.strip() for h in reader.fieldnames if h is not None]
        raw_rows = list(reader)
    if not raw_rows:
        raise ValueError("CSV file contains no data rows.")

    coord_candidates = [h for h in headers if h.lower() in ("x", "y", "z", "t")]
    if not coord_candidates:
        raise ValueError("CSV file must include at least one coordinate column (x, y, z, or t).")
    coord_names = tuple(c.lower() for c in coord_candidates)
    n = len(raw_rows)
    pts = np.empty((n, len(coord_candidates)), dtype=np.float64)
    for col_i, h in enumerate(coord_candidates):
        pts[:, col_i] = [float(r[h]) for r in raw_rows]

    fields: dict[str, np.ndarray] = {}
    references: dict[str, np.ndarray] = {}
    point_ids = None
    skip = set(coord_candidates)
    for h in headers:
        if h in skip:
            continue
        low = h.lower()
        if low in ("point_id", "id"):
            point_ids = [r[h] for r in raw_rows]
            continue
        if low.startswith("error ") or low.startswith("error_"):
            continue
        try:
            vals = np.asarray([float(r[h]) for r in raw_rows], dtype=np.float64)
        except ValueError:
            continue
        if low.startswith("ref "):
            references[h[4:].strip()] = vals
        elif low.startswith("ref_"):
            references[h[4:].strip()] = vals
        else:
            fields[h] = vals

    if not fields:
        raise ValueError("CSV file contains no numeric field columns.")

    return SolutionDataset(
        points=pts,
        coordinate_names=coord_names,
        fields=fields,
        references=references,
        title=path.name,
        source_description=f"CSV · {n} rows",
        point_ids=point_ids,
    )


def _load_netcdf(path: Path) -> SolutionDataset:
    try:
        import xarray as xr
    except ImportError as exc:
        raise ValueError("Reading NetCDF (.nc) files requires the optional geometry extra (xarray).") from exc

    with xr.open_dataset(path) as ds:
        # Check coordinate variables or dims matching x, y, z, t
        coord_vars = [c for c in ("x", "y", "z", "t") if c in ds.coords or c in ds.data_vars]
        if not coord_vars:
            raise ValueError("NetCDF dataset contains no x/y/z/t coordinates.")
        data_vars = [v for v in ds.data_vars if v not in coord_vars]
        if not data_vars:
            raise ValueError("NetCDF dataset contains no field variables.")
        first = ds[data_vars[0]]
        # Broadcast coordinate arrays against the primary field's dims
        coord_arrays = []
        for c in coord_vars:
            arr = ds[c]
            if arr.dims == first.dims:
                coord_arrays.append(np.asarray(arr.values, dtype=np.float64).reshape(-1))
            elif set(arr.dims) <= set(first.dims):
                broadcast = arr.broadcast_like(first)
                coord_arrays.append(np.asarray(broadcast.values, dtype=np.float64).reshape(-1))
        pts = np.column_stack(coord_arrays)
        n = len(pts)
        fields = {}
        references = {}
        for v in data_vars:
            arr = np.asarray(ds[v].values, dtype=np.float64)
            if arr.size == n:
                if v.startswith("ref_"):
                    references[v[4:]] = arr.reshape(-1)
                else:
                    fields[v] = arr.reshape(-1)
        if not fields:
            raise ValueError("NetCDF dataset contains no fields matching coordinate dimensions.")
        return SolutionDataset(
            points=pts,
            coordinate_names=tuple(coord_vars[: pts.shape[1]]),
            fields=fields,
            references=references,
            title=path.name,
            source_description=f"NetCDF · {n} points",
        )


class StudyCollection:
    """Read-only parameter-study browser with lazy LRU caching of trial datasets."""

    def __init__(self, root: str | Path, *, max_cached: int = 16):
        from ..study_results import collect

        self.root = Path(root).absolute()
        self.collection = collect(self.root)
        self.max_cached = max(2, int(max_cached))
        self._cache: OrderedDict[tuple[int, str], SolutionDataset] = OrderedDict()

        study = self.collection["study"]
        self.study_id: str = study["study_id"]
        self.axes_meta: list[dict[str, Any]] = []
        self.records: list[dict[str, Any]] = self.collection["records"]

        # Extract unique ordered values per swept axis
        if self.records:
            num_axes = len(self.records[0]["choices"])
            for axis_i in range(num_axes):
                first = self.records[0]["choices"][axis_i]
                seen: list[Any] = []
                for rec in self.records:
                    val = rec["choices"][axis_i]["value"]
                    if val not in seen:
                        seen.append(val)
                self.axes_meta.append(
                    {
                        "index": axis_i,
                        "block": first["block"],
                        "parameter": first["parameter"],
                        "label": f"{first['block']} · {first['parameter']}",
                        "values": seen,
                    }
                )

        self.completed_indices: list[int] = [
            idx for idx, rec in enumerate(self.records) if rec["row"]["status"] == "completed"
        ]
        # Map canonical JSON choice tuple -> trial index
        self._choice_map: dict[tuple[str, ...], int] = {}
        for idx, rec in enumerate(self.records):
            key = tuple(json.dumps(c["value"], sort_keys=True) for c in rec["choices"])
            self._choice_map[key] = idx

    def trial_for_values(self, values: list[Any]) -> int | None:
        key = tuple(json.dumps(v, sort_keys=True) for v in values)
        return self._choice_map.get(key)

    def trial_choices(self, trial_index: int) -> list[dict[str, Any]]:
        return self.records[trial_index]["choices"]

    def trial_status(self, trial_index: int) -> str:
        return self.records[trial_index]["row"]["status"]

    def trial_archive_path(self, trial_index: int) -> Path:
        row = self.records[trial_index]["row"]
        return self.root / row["directory"] / "artifacts" / "run.margam-run.pt"

    def output_ids_for_trial(self, trial_index: int) -> list[str]:
        if self.trial_status(trial_index) != "completed":
            return []
        path = self.trial_archive_path(trial_index)
        if not path.is_file():
            return []
        datasets = from_run_archive(path)
        for oid, ds in datasets.items():
            self._put_cache((trial_index, oid), ds)
        return list(datasets.keys())

    def _put_cache(self, key: tuple[int, str], ds: SolutionDataset):
        self._cache[key] = ds
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_cached:
            self._cache.popitem(last=False)

    def load_trial_dataset(self, trial_index: int, output_id: str | None = None) -> SolutionDataset | None:
        if self.trial_status(trial_index) != "completed":
            return None
        if output_id is not None and (trial_index, output_id) in self._cache:
            self._cache.move_to_end((trial_index, output_id))
            return self._cache[(trial_index, output_id)]
        path = self.trial_archive_path(trial_index)
        if not path.is_file():
            return None
        datasets = from_run_archive(path)
        if not datasets:
            return None
        choices_str = ", ".join(f"{c['parameter']}={c['value']}" for c in self.trial_choices(trial_index))
        for oid, ds in datasets.items():
            ds.title = f"DP{trial_index + 1:04d} ({choices_str})"
            ds.source_description = f"Study {self.root.name} · DP{trial_index + 1:04d}"
            self._put_cache((trial_index, oid), ds)
        chosen = output_id if output_id in datasets else next(iter(datasets))
        return datasets[chosen]

    def parametric_response(
        self,
        axis_index: int,
        fixed_values: list[Any],
        *,
        field_name: str,
        metric: str = "max",
        probe_coord: tuple[float, ...] | None = None,
        output_id: str | None = None,
        time_index: int | None = None,
    ) -> tuple[list[Any], list[float]]:
        """Collect ``(axis_values, metric_values)`` along ``axis_index`` holding other axes fixed."""
        if not 0 <= axis_index < len(self.axes_meta):
            return [], []
        axis = self.axes_meta[axis_index]
        xs: list[Any] = []
        ys: list[float] = []
        for val in axis["values"]:
            combo = list(fixed_values)
            combo[axis_index] = val
            t_idx = self.trial_for_values(combo)
            if t_idx is None or self.trial_status(t_idx) != "completed":
                continue
            ds = self.load_trial_dataset(t_idx, output_id=output_id)
            if ds is None:
                continue
            try:
                if metric == "probe" and probe_coord is not None:
                    info = ds.probe_point(probe_coord, field_name, time_index=time_index)
                    y = float("nan") if info is None else info["value"]
                else:
                    stats = ds.statistics(field_name, time_index=time_index)
                    y = stats.get(metric)
                    y = float("nan") if y is None else float(y)
            except (KeyError, ValueError):
                continue
            if np.isfinite(y):
                xs.append(val)
                ys.append(float(y))
        return xs, ys
