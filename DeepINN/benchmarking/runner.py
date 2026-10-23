"""Compare recipes using fixed data per seed and explicit saved artefacts."""
import copy
import csv
import hashlib
import json
import platform
import random
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from ..components import PINNProblem, PointConstraint
from .metrics import solution_errors


@dataclass
class BenchmarkCase:
    """Problem factory plus evaluation coordinates and a reference solution.

    Factories run on CPU after seeding. The reference is a tensor or callable
    of evaluation points. Callers define their train/evaluation split; this
    runner does not invent a split or infer a PDE from numerical data.
    """
    name: str
    build_problem: Callable
    evaluation_points: object
    reference: object
    config: dict = field(default_factory=dict)


@dataclass
class Recipe:
    """Factory build(problem) -> compiled recipe with train/predict methods.

    A recipe exposes device and loss_history (one dictionary per update).
    It can construct one or several networks internally. Configuration is
    supplied explicitly for provenance; include architecture and optimiser.
    """
    name: str
    build: Callable
    config: dict = field(default_factory=dict)


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _tensor(value):
    value = torch.as_tensor(value).detach().cpu().clone()
    if not torch.is_floating_point(value):
        value = value.float()
    if not torch.isfinite(value).all():
        raise ValueError("Benchmark data must be finite.")
    return value


def _points(source):
    value = _tensor(source() if callable(source) else source)
    if value.ndim != 2 or not value.shape[0] or not value.shape[1]:
        raise ValueError("Points must be non-empty (N, D) tensors.")
    return value


def _target(source, points):
    return _tensor(source(points) if callable(source) else source)


def _snapshot(case):
    """Resolve stochastic samplers once; clone this data for every recipe."""
    problem = case.build_problem()
    collocation = _points(problem.collocation_points)
    residual_target = _target(problem.residual_target, collocation)
    constraints = []
    arrays = {"collocation_points": collocation, "residual_target": residual_target}
    names = []
    for index, condition in enumerate(problem.constraints):
        points = _points(condition.points)
        target = _target(condition.target, points)
        constraints.append(PointConstraint(condition.name, points, target, condition.operator))
        arrays[f"constraint_{index}_points"] = points
        arrays[f"constraint_{index}_target"] = target
        names.append(condition.name)
    evaluation = _points(case.evaluation_points)
    reference = _target(case.reference, evaluation)
    if reference.ndim == 1:
        reference = reference[:, None]
    if reference.ndim != 2 or reference.shape[0] != len(evaluation):
        raise ValueError("Reference must have one row per evaluation point.")
    if evaluation.shape[1] != collocation.shape[1]:
        raise ValueError("Evaluation and collocation points must have the same input dimension.")
    arrays.update(evaluation_points=evaluation, reference=reference)
    return PINNProblem(problem.residual, collocation, constraints, residual_target), arrays, names


def _fingerprint(arrays):
    digest = hashlib.sha256()
    for name, tensor in sorted(arrays.items()):
        array = tensor.contiguous().numpy()
        digest.update(json.dumps([name, str(array.dtype), list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _json(path, value):
    # A temporary sibling prevents partially written results on interruption.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    if not keys:
        keys = ["iteration", "total"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _sync(model):
    device = torch.device(model.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _environment():
    environment = {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda, "cpu_threads": torch.get_num_threads(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                  capture_output=True, text=True, check=True, timeout=5)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                                capture_output=True, text=True, check=True, timeout=5)
        environment.update(git_commit=revision.stdout.strip(), git_dirty=bool(status.stdout))
    except (OSError, subprocess.SubprocessError):
        environment.update(git_commit=None, git_dirty=None)
    return environment


def _save_summary(output, results):
    _json(output / "summary.json", results)
    _csv(output / "summary.csv", results)
    aggregates = []
    for name in dict.fromkeys(result["recipe"] for result in results):
        runs = [result for result in results if result["recipe"] == name]
        completed = [result for result in runs if result["status"] == "completed"]
        row = {"recipe": name, "runs_completed": len(completed),
               "runs_failed": len(runs) - len(completed)}
        for metric in ("rmse", "mae", "max_abs_error", "relative_l2",
                       "training_seconds", "prediction_seconds"):
            values = [result[metric] for result in completed if result[metric] is not None]
            row[metric + "_mean"] = float(np.mean(values)) if values else None
            row[metric + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        aggregates.append(row)
    _json(output / "aggregate.json", aggregates)
    _csv(output / "aggregate.csv", aggregates)


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError("Names must use letters, numbers, underscores or hyphens.")


def run_benchmark(case, recipes, *, seeds, iterations, output_dir, display_every=0):
    """Run each compiled recipe for a fixed number of updates per seed.

    The output directory must not exist. Data are materialised once per seed
    and saved as samples.npz, then deep-copied for each recipe. Factories are
    re-seeded before construction. Timing synchronises CUDA, includes no
    checkpoint/plot writing, and measures full training including history
    collection. Seeds aid repeatability, not bitwise determinism across devices.
    A failure is saved with its error and re-raised; it cannot count as success.
    """
    recipes, seeds = list(recipes), list(seeds)
    _name(case.name)
    for recipe in recipes:
        _name(recipe.name)
    if not recipes or len({r.name for r in recipes}) != len(recipes):
        raise ValueError("Provide at least one recipe with unique names.")
    if (not seeds or len(set(seeds)) != len(seeds) or
            any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**32 for s in seeds)):
        raise ValueError("Seeds must be distinct integers in [0, 2**32).")
    for name, value in (("iterations", iterations), ("display_every", display_every)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")
    manifest = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "case": case.name, "case_config": case.config,
        "iterations": iterations, "seeds": seeds,
        "recipes": [{"name": r.name, "config": r.config} for r in recipes],
        "environment": _environment(),
        "metric_reduction": "pooled over all evaluation points and output components",
    }
    json.dumps(manifest, allow_nan=False)  # Validate config before creating output.
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _json(output / "manifest.json", manifest)
    results = []
    for seed in seeds:
        _seed(seed)
        try:
            problem, arrays, condition_names = _snapshot(case)
        except Exception as error:
            _json(output / "failure.json", {"stage": "sampling", "seed": seed,
                  "error": f"{type(error).__name__}: {error}"})
            raise
        seed_dir = output / f"seed-{seed}"
        seed_dir.mkdir()
        np.savez_compressed(seed_dir / "samples.npz", **{k: v.numpy() for k, v in arrays.items()})
        fingerprint = _fingerprint(arrays)
        _json(seed_dir / "samples.json", {
            "sha256": fingerprint, "constraint_names": condition_names,
            "shapes": {k: list(v.shape) for k, v in arrays.items()},
        })
        for recipe in recipes:
            run_dir = seed_dir / recipe.name
            run_dir.mkdir()
            result = {"case": case.name, "recipe": recipe.name, "seed": seed,
                      "iterations": iterations, "samples_sha256": fingerprint,
                      "evaluation_points": len(arrays["evaluation_points"]), "status": "running"}
            try:
                _seed(seed)
                start = time.perf_counter()
                model = recipe.build(copy.deepcopy(problem))
                _sync(model)
                result["setup_seconds"] = time.perf_counter() - start
                device = torch.device(model.device)
                result["device"] = str(device)
                result["gpu"] = torch.cuda.get_device_name(device) if device.type == "cuda" else None
                start = time.perf_counter()
                model.train(iterations, display_every=display_every)
                _sync(model)
                result["training_seconds"] = time.perf_counter() - start
                history = [{"iteration": i+1, **row} for i, row in enumerate(model.loss_history)]
                if len(history) != iterations:
                    raise ValueError("Recipe must record one loss_history entry per update.")
                for row in history:
                    if any(not np.isfinite(float(v)) for v in row.values()):
                        raise ValueError("Recipe history contains a non-finite value.")
                _csv(run_dir / "loss_history.csv", history)
                _sync(model)
                start = time.perf_counter()
                prediction = model.predict(arrays["evaluation_points"].clone())
                _sync(model)
                result["prediction_seconds"] = time.perf_counter() - start
                prediction = prediction.detach().cpu()
                if prediction.ndim == 1:
                    prediction = prediction[:, None]
                reference = arrays["reference"]
                result.update(solution_errors(prediction, reference))
                np.savez_compressed(run_dir / "predictions.npz",
                                    points=arrays["evaluation_points"].numpy(),
                                    prediction=prediction.numpy(), reference=reference.numpy())
                result["final_training_loss"] = history[-1].get("total") if history else None
                result["status"] = "completed"
            except Exception as error:
                result.update(status="failed", error=f"{type(error).__name__}: {error}")
                _json(run_dir / "result.json", result)
                results.append(result)
                _save_summary(output, results)
                raise
            _json(run_dir / "result.json", result)
            results.append(result)
            _save_summary(output, results)
    return results
