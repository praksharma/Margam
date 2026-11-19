"""JAXPI2 training loop integration on Margam's shared research Trainer.

Supports PirateNet/ModifiedMlp/Mlp, SOAP/Adam with Schedule-Free and warmup/decay,
per-component causal gating, grad-norm balancing (with 1e-5 relative denominator floor),
pseudo-time stepping, time-marching with transfer learning, and multi-stage linearized NS.
"""
from collections import deque
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from functools import partial
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import torch
from ...components.problem import PINNProblem, PointConstraint
from ..trainer import Trainer
from .optim import Jaxpi2Optimizer, OptimConfig
from .physics import (
    component,
    compute_pts_weights,
    jaxpi2_causal_residual_losses,
    multi_stage_navier_stokes,
    navier_stokes,
    normal_gradient,
    pseudo_time_residuals,
)


@dataclass(frozen=True)
class Jaxpi2TrainingConfig:
    reynolds: float = 300.0
    # Causal settings (jaxpi2 ForwardIVP)
    causal_enabled: bool = True
    chunks: int = 16
    causal_epsilon: float = 1.0
    componentwise_causal: bool = True
    # Loss weighting settings (jaxpi2 PINN.compute_loss_weights)
    loss_weighting_strategy: str = "dynamic"  # "dynamic" or "constant"
    balance_start: int = 100
    balance_every: int = 1000
    momentum: float = 0.9
    grad_norm_eps: float = 1e-5
    ic_init_weight: float = 100.0
    init_weights: Optional[Dict[str, float]] = None
    # Pseudo-time stepping settings (jaxpi2 PINN.compute_pts_weights)
    pseudo_time_enabled: bool = False
    pseudo_time_strategy: str = "constant"  # "dynamic" or "constant"
    pts_start: int = 100
    pts_every: int = 1000
    pts_momentum: float = 0.9
    pts_init_weight: float = 1.0
    pts_shrink_enabled: bool = True
    pts_start_log_drop: float = 2.0
    pts_end_log_drop: float = 6.0
    pts_min_factor: float = 0.1
    # Optimizer settings
    optim: OptimConfig = field(default_factory=OptimConfig)
    # Time-marching transfer learning & IC pressure control
    transfer_learning: bool = True
    include_pressure_ic: bool = True

    def __post_init__(self):
        if any(
            type(v) is not int or v < 1
            for v in (self.chunks, self.balance_every, self.pts_every)
        ):
            raise ValueError("chunks, balance_every, and pts_every must be positive integers.")
        if any(type(v) is not int or v < 0 for v in (self.balance_start, self.pts_start)):
            raise ValueError("balance_start and pts_start must be non-negative integers.")
        if self.loss_weighting_strategy not in ("dynamic", "constant"):
            raise ValueError("loss_weighting_strategy must be 'dynamic' or 'constant'.")
        if self.pseudo_time_strategy not in ("dynamic", "constant"):
            raise ValueError("pseudo_time_strategy must be 'dynamic' or 'constant'.")
        if (
            not math.isfinite(self.reynolds)
            or self.reynolds <= 0
            or not math.isfinite(self.causal_epsilon)
            or self.causal_epsilon < 0
            or not (0.0 <= self.momentum < 1.0)
            or not (0.0 <= self.pts_momentum < 1.0)
            or not math.isfinite(self.grad_norm_eps)
            or self.grad_norm_eps < 0
            or not math.isfinite(self.ic_init_weight)
            or self.ic_init_weight <= 0
            or not math.isfinite(self.pts_init_weight)
            or self.pts_init_weight <= 0
        ):
            raise ValueError("Invalid physical, causal, weighting, or pseudo-time parameters.")


def should_update_on_schedule(step: int, start: int, every: int) -> bool:
    """Match jaxpi2.utils.create_update_scheduler: s = step - start; s >= 0 and s % every == 0."""
    s = step - start
    return s >= 0 and (s % every) == 0


def mean_norm_weights(
    terms: Mapping[str, torch.Tensor],
    parameters: Sequence[torch.nn.Parameter],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Unweighted per-loss gradient norms with exact mean reference (jaxpi v1 rule)."""
    norms = {}
    for name, term in terms.items():
        grads = torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
        norms[name] = torch.sqrt(
            sum((g.detach().square().sum() for g in grads if g is not None), term.new_zeros(()))
        )
    if any(not torch.isfinite(n) or n <= 0 for n in norms.values()):
        raise FloatingPointError("Zero/nonfinite loss gradient norm; exact JAX-PI ratio undefined.")
    reference = torch.stack(list(norms.values())).mean()
    targets = {name: (reference / n).item() for name, n in norms.items()}
    if not all(math.isfinite(v) for v in targets.values()):
        raise FloatingPointError("Nonfinite gradient-norm ratio.")
    return targets, {name: n.item() for name, n in norms.items()}


def jaxpi2_grad_norm_weights(
    terms: Mapping[str, torch.Tensor],
    parameters: Sequence[torch.nn.Parameter],
    eps_factor: float = 1e-5,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Per-loss gradient norm balancing matching jaxpi2 PINN.compute_loss_weights."""
    norms: Dict[str, torch.Tensor] = {}
    for name, term in terms.items():
        grads = torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
        norms[name] = torch.sqrt(
            sum((g.detach().square().sum() for g in grads if g is not None), term.new_zeros(()))
        )
    if any(not torch.isfinite(n) or n < 0 for n in norms.values()):
        raise FloatingPointError("Non-finite loss gradient norm in jaxpi2 balancing.")
    mean_grad_norm = torch.stack(list(norms.values())).mean()
    if not torch.isfinite(mean_grad_norm) or float(mean_grad_norm.item()) <= 0.0:
        raise FloatingPointError("Zero or non-finite mean gradient norm in jaxpi2 balancing.")
    targets = {
        name: float((mean_grad_norm / (n + eps_factor * mean_grad_norm)).item())
        for name, n in norms.items()
    }
    if not all(math.isfinite(v) for v in targets.values()):
        raise FloatingPointError("Non-finite adaptive loss weight target.")
    return targets, {name: float(n.item()) for name, n in norms.items()}


class Jaxpi2SphereTrainer(Trainer):
    """Sphere PINN trainer with JAXPI2 architecture, SOAP/Schedule-Free, and causal/pts rules."""

    RESIDUAL_KEYS = ("ru", "rv", "rw", "rc")
    VARIABLE_KEYS = ("u", "v", "w", "p")

    def __init__(
        self,
        network: torch.nn.Module,
        sampler,
        config: Jaxpi2TrainingConfig = Jaxpi2TrainingConfig(),
        *,
        history_limit: int = 100,
        history_sink=None,
        pde_fn=None,
    ):
        if config.causal_enabled and (
            config.chunks < 1 or sampler.residual_count % config.chunks != 0
        ):
            raise ValueError("Choose a positive chunk count dividing residual_count.")
        if type(history_limit) is not int or history_limit < 1:
            raise ValueError("History limit must be a positive integer.")

        self.history_limit = history_limit
        self.history_sink = history_sink
        self.log_position = None
        self._failed_update = None
        self.sampler = sampler
        self.config = config
        self.balance_history = deque(maxlen=history_limit)
        self.pts_history = deque(maxlen=history_limit)
        self.balance_count = 0
        self.pts_count = 0
        self.minimum_causal_weight = 1.0
        self.pts_weights = {
            k: float(config.pts_init_weight) for k in self.VARIABLE_KEYS
        }
        self._init_network = None
        self._prev_network = None

        self.batch = sampler.draw()
        constraints = self.make_constraints(self.batch)
        if pde_fn is None:
            pde_fn = partial(navier_stokes, reynolds=config.reynolds)
        self._pde_fn = pde_fn
        problem = PINNProblem(
            pde_fn,
            self.batch["res"],
            constraints=constraints,
            data_identity="sphere300-jaxpi2-profile-v1",
        )
        names = [c.name for c in constraints] + list(self.RESIDUAL_KEYS)
        weights = {}
        for name in names:
            if config.init_weights is not None and name in config.init_weights:
                weights[name] = float(config.init_weights[name])
            elif name.endswith("_ic") or name.endswith("_diff"):
                weights[name] = float(config.ic_init_weight)
            else:
                weights[name] = 1.0
        weights["pde"] = 0.0
        super().__init__(
            problem,
            network,
            weights=weights,
            checkpoint_id="sphere300-jaxpi2-core-v1",
        )

    def make_constraints(self, batch):
        c = []
        ic_fields = ("u", "v", "w", "p") if self.config.include_pressure_ic else ("u", "v", "w")
        for i, name in enumerate(ic_fields):
            c.append(PointConstraint(f"{name}_ic", batch["initial"], batch["values"][:, i : i + 1], component(i)))
        for i, name in enumerate(("u", "v", "w")):
            c.append(PointConstraint(f"{name}_in", batch["faces"][(0, 0)], float(i == 0), component(i)))
            c.append(PointConstraint(f"{name}_noslip", batch["sphere"], 0.0, component(i)))
            c.append(PointConstraint(f"{name}_out", batch["faces"][(0, 1)], 0.0, normal_gradient(i, 1)))
        c.append(PointConstraint("p_out", batch["faces"][(0, 1)], 0.0, component(3)))
        for axis in (1, 2):
            for side in (0, 1):
                for i, name in enumerate(("u", "v", "w")):
                    op = component(i) if i == axis else normal_gradient(i, axis + 1)
                    c.append(PointConstraint(f"{name}_slip_{axis}_{side}", batch["faces"][(axis, side)], 0.0, op))
        return c

    def compile(self, *, device: str = "cpu"):
        self._healthy()
        if getattr(self, "_compiled", False):
            raise RuntimeError("Create a new trainer for a new run; restore a checkpoint to continue.")
        full_weights = self.weights
        self.weights = {
            name: value
            for name, value in full_weights.items()
            if name not in self.RESIDUAL_KEYS
        }
        try:
            super().compile(
                optimizer=partial(Jaxpi2Optimizer, config=self.config.optim),
                lr=self.config.optim.learning_rate,
                device=device,
            )
        finally:
            self.weights = full_weights
        self.loss_history = deque(maxlen=self.history_limit)
        self.training_history = deque(maxlen=self.history_limit)
        self._init_network = deepcopy(self.network).to(self.device)
        for p in self._init_network.parameters():
            p.requires_grad_(False)
        self._prev_network = deepcopy(self.network).to(self.device)
        for p in self._prev_network.parameters():
            p.requires_grad_(False)
        return self

    def _healthy(self):
        if self._failed_update is not None:
            raise RuntimeError(
                "Trainer is invalid after a failed update. Restore a valid checkpoint or rebuild it. "
                + self._failed_update
            )

    def _before_training_step(self):
        self._healthy()
        self.batch = self.sampler.draw()
        self.collocation_point_sample = self._points(self.batch["res"])
        self.collocation_point_labels = self._target(0.0, self.collocation_point_sample)
        conditions = self.make_constraints(self.batch)
        self._constraints = []
        for condition in conditions:
            points = self._points(condition.points)
            self._constraints.append((condition, points, self._target(condition.target, points)))

    def _pts_vector(self) -> List[float]:
        return [float(self.pts_weights[k]) for k in self.VARIABLE_KEYS]

    def compute_losses(self):
        samples = self.loss_samples()
        losses = {name: self._term(*sample) for name, sample in samples.items() if name != "pde"}
        residuals = samples["pde"][0]
        if self.config.pseudo_time_enabled:
            q = self.collocation_point_sample
            sol_pred = self.network(q)
            with torch.no_grad():
                sol_prev = self._prev_network(q.detach())
            residuals = pseudo_time_residuals(residuals, sol_pred, sol_prev, self._pts_vector())
        if self.config.causal_enabled:
            residual_losses, gamma_min, _ = jaxpi2_causal_residual_losses(
                residuals,
                self.config.chunks,
                self.config.causal_epsilon,
                componentwise=self.config.componentwise_causal,
            )
            self.minimum_causal_weight = float(gamma_min.min().detach().item())
        else:
            residual_losses = residuals.square().mean(dim=0)
            self.minimum_causal_weight = 1.0
        losses.update(zip(self.RESIDUAL_KEYS, residual_losses.unbind()))
        losses["pde"] = residual_losses.sum()
        return losses

    def _eval_residuals_and_sol(self, net: torch.nn.Module, q: torch.Tensor):
        q_req = q.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            sol = net(q_req)
            res = self._pde_fn(q_req, sol)
        return res.detach(), sol.detach()

    def _update_pseudo_time_weights(self):
        q = self.collocation_point_sample.detach()
        res_pred, sol_pred = self._eval_residuals_and_sol(self.network, q)
        res_prev, sol_prev = self._eval_residuals_and_sol(self._prev_network, q)
        res0_pred, _ = self._eval_residuals_and_sol(self._init_network, q)
        targets = compute_pts_weights(
            res_pred,
            res_prev,
            res0_pred,
            sol_pred,
            sol_prev,
            shrink_enabled=self.config.pts_shrink_enabled,
            start_log_drop=self.config.pts_start_log_drop,
            end_log_drop=self.config.pts_end_log_drop,
            min_factor=self.config.pts_min_factor,
        )
        target_dict = {
            k: float(targets[i].item()) for i, k in enumerate(self.VARIABLE_KEYS)
        }
        updated = {
            k: self.config.pts_momentum * self.pts_weights[k]
            + (1.0 - self.config.pts_momentum) * target_dict[k]
            for k in self.VARIABLE_KEYS
        }
        if not all(math.isfinite(v) for v in updated.values()):
            raise FloatingPointError("Non-finite pseudo-time weight.")
        self.pts_weights.update(updated)
        self.pts_count += 1
        self.pts_history.append(
            dict(
                after_update=self.iter + 1,
                targets=target_dict,
                pts_weights=deepcopy(self.pts_weights),
            )
        )

    def update(self, terms, total):
        self._healthy()
        try:
            if self.config.pseudo_time_enabled:
                self._prev_network.load_state_dict(self.network.state_dict())
            super().update(terms, total)
            if (
                self.config.pseudo_time_enabled
                and self.config.pseudo_time_strategy == "dynamic"
                and should_update_on_schedule(self.iter, self.config.pts_start, self.config.pts_every)
            ):
                self._update_pseudo_time_weights()
            if (
                self.config.loss_weighting_strategy == "dynamic"
                and should_update_on_schedule(
                    self.iter, self.config.balance_start, self.config.balance_every
                )
            ):
                new_terms = self.compute_losses()
                active = {name: value for name, value in new_terms.items() if name != "pde"}
                targets, norms = jaxpi2_grad_norm_weights(
                    active,
                    tuple(self.network.parameters()),
                    eps_factor=self.config.grad_norm_eps,
                )
                updated = {
                    name: self.config.momentum * self.weights[name]
                    + (1.0 - self.config.momentum) * targets[name]
                    for name in active
                }
                if not all(math.isfinite(v) for v in updated.values()):
                    raise FloatingPointError("Non-finite adaptive loss weight.")
                self.weights.update(updated)
                self.balance_count += 1
                self.balance_history.append(
                    dict(
                        after_update=self.iter + 1,
                        norms=norms,
                        targets=targets,
                        weights=deepcopy(self.weights),
                    )
                )
        except BaseException as error:
            self._failed_update = f"{type(error).__name__}: {error}"
            raise

    def record_contributions(self, record, terms, weights=None):
        result = super().record_contributions(record, terms, weights)
        if self.history_sink is not None:
            event = (
                self.balance_history[-1]
                if self.balance_history and self.balance_history[-1]["after_update"] == self.iter
                else None
            )
            pts_event = (
                self.pts_history[-1]
                if self.pts_history and self.pts_history[-1]["after_update"] == self.iter
                else None
            )
            counts = (
                torch.bincount(self.batch["res_regions"], minlength=3).tolist()
                if "res_regions" in self.batch
                else [len(self.batch["res"])]
            )
            try:
                self.log_position = self.history_sink(
                    dict(
                        type="step",
                        step=self.iter,
                        losses=record,
                        balance=event,
                        pseudo_time=pts_event,
                        sample_counts=counts,
                    )
                )
            except BaseException as error:
                self._failed_update = f"Logging failed after update: {type(error).__name__}: {error}"
                raise
        return result

    @contextmanager
    def use_eval_params(self):
        """Temporarily switch network parameters to schedule-free averaged eval parameters."""
        self._require_compiled()
        if hasattr(self.optimizer, "use_eval_params"):
            with self.optimizer.use_eval_params():
                yield
        else:
            yield

    def eval_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return network state_dict with schedule-free evaluation parameters."""
        with self.use_eval_params():
            return deepcopy(self.network.state_dict())

    def transfer_from(self, prev_trainer: "Jaxpi2SphereTrainer", *, use_eval: bool = False):
        """Initialize network parameters from a previous time window (arXiv:2507.08972v2 p. 10).

        In jaxpi2/training.py:298-299, transfer learning copies model.state.params (use_eval=False)
        and resets the optimizer state and adaptive weights for the new window.
        """
        self._healthy()
        state_dict = prev_trainer.eval_state_dict() if use_eval else deepcopy(prev_trainer.network.state_dict())
        self.network.load_state_dict(state_dict, strict=True)
        if self._init_network is not None:
            self._init_network.load_state_dict(state_dict, strict=True)
        if self._prev_network is not None:
            self._prev_network.load_state_dict(state_dict, strict=True)
        if getattr(self, "_compiled", False):
            self.optimiser = Jaxpi2Optimizer(
                self.network.parameters(),
                lr=self.config.optim.learning_rate,
                config=self.config.optim,
            )
        return self

    def identity(self):
        digest = hashlib.sha256()
        for tensor in (self.sampler.initial_q, self.sampler.initial_values):
            digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            digest.update(tensor.contiguous().numpy().tobytes())
        pkg_root = Path(__file__).resolve().parents[2]
        for path in sorted(Path(__file__).parent.glob("*.py")):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        for relative in (
            "research/trainer.py",
            "methods/pinn.py",
            "components/differential.py",
            "components/problem.py",
            "components/losses.py",
        ):
            target = pkg_root / relative
            if target.is_file():
                digest.update(relative.encode())
                digest.update(target.read_bytes())
        core_cfg = getattr(getattr(self.network, "core", self.network), "config", None)
        net_meta = asdict(core_cfg) if core_cfg is not None else {"class": type(self.network).__qualname__}
        return dict(
            format="sphere-jaxpi2-adaptation",
            version=1,
            code_and_initial_sha256=digest.hexdigest(),
            torch_version=str(torch.__version__),
            sampling=self.sampler.identity(),
            network=net_meta,
            domain=asdict(self.sampler.domain),
            settings=asdict(self.config),
            dtype=str(self.dtype),
            device=str(self.device),
            counts=[
                self.sampler.residual_count,
                self.sampler.boundary_count,
                self.sampler.initial_count,
            ],
        )

    def training_state(self):
        self._healthy()
        return deepcopy(
            dict(
                identity=self.identity(),
                network=self.network.state_dict(),
                prev_network=self._prev_network.state_dict() if self._prev_network is not None else None,
                init_network=self._init_network.state_dict() if self._init_network is not None else None,
                optimizer=self.optimizer.state_dict(),
                iter=self.iter,
                weights=self.weights,
                pts_weights=self.pts_weights,
                balance_history=list(self.balance_history),
                pts_history=list(self.pts_history),
                loss_history=list(self.loss_history),
                balance_count=self.balance_count,
                pts_count=self.pts_count,
                log_position=self.log_position,
                minimum_causal_weight=self.minimum_causal_weight,
                training_history=list(self.training_history),
                sampler_rng=self.sampler.generator.get_state(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
            )
        )

    def load_training_state(self, state):
        if json.dumps(state["identity"], sort_keys=True) != json.dumps(self.identity(), sort_keys=True):
            raise ValueError("Checkpoint code, initial data, configuration or device differs.")
        if type(state["iter"]) is not int or state["iter"] < 0:
            raise ValueError("Invalid checkpoint update index.")
        try:
            self.network.load_state_dict(state["network"], strict=True)
            if state.get("prev_network") is not None and self._prev_network is not None:
                self._prev_network.load_state_dict(state["prev_network"], strict=True)
            if state.get("init_network") is not None and self._init_network is not None:
                self._init_network.load_state_dict(state["init_network"], strict=True)
            self.optimizer.load_state_dict(state["optimizer"])
            self.iter = state["iter"]
            self.weights = deepcopy(state["weights"])
            self.pts_weights = deepcopy(state["pts_weights"])
            self.balance_history = deque(deepcopy(state["balance_history"]), maxlen=self.history_limit)
            self.pts_history = deque(deepcopy(state["pts_history"]), maxlen=self.history_limit)
            self.loss_history = deque(deepcopy(state["loss_history"]), maxlen=self.history_limit)
            self.training_history = deque(deepcopy(state["training_history"]), maxlen=self.history_limit)
            self.balance_count = state["balance_count"]
            self.pts_count = state["pts_count"]
            self.log_position = deepcopy(state["log_position"])
            self.minimum_causal_weight = state["minimum_causal_weight"]
            self.sampler.generator.set_state(state["sampler_rng"])
            torch.set_rng_state(state["torch_rng"])
            if self.device.type == "cuda":
                if not state.get("cuda_rng"):
                    raise ValueError("CUDA checkpoint has no GPU RNG state.")
                torch.cuda.set_rng_state_all(state["cuda_rng"])
        except BaseException as error:
            self._failed_update = f"Checkpoint restore failed: {error}"
            raise
        self._failed_update = None
        return self


class Jaxpi2CropTrainer(Jaxpi2SphereTrainer):
    """Boundary-driven cropped sphere trainer with JAXPI2 methods on Margam's shared loop."""

    def make_constraints(self, batch):
        constraints = []
        ic_fields = ("u", "v", "w", "p") if self.config.include_pressure_ic else ("u", "v", "w")
        for i, name in enumerate(ic_fields):
            constraints.append(
                PointConstraint(f"{name}_ic", batch["initial"], batch["values"][:, i : i + 1], component(i))
            )
        for axis in range(3):
            for side in (0, 1):
                for i, name in enumerate(("u", "v", "w")):
                    constraints.append(
                        PointConstraint(
                            f"{name}_face_{axis}_{side}",
                            batch["faces"][(axis, side)],
                            batch["face_values"][(axis, side)][:, i : i + 1],
                            component(i),
                        )
                    )
        for i, name in enumerate(("u", "v", "w")):
            constraints.append(PointConstraint(f"{name}_noslip", batch["sphere"], 0.0, component(i)))
        constraints.append(PointConstraint("p_anchor", batch["pressure_anchor"], 0.0, component(3)))
        return constraints


class Jaxpi2MultiStageCropTrainer(Jaxpi2CropTrainer):
    """Stage k >= 1 linearized correction trainer on the boundary-driven sphere domain."""

    def __init__(
        self,
        network: torch.nn.Module,
        prev_solution: torch.nn.Module,
        stage_eps: float,
        sampler,
        config: Jaxpi2TrainingConfig = Jaxpi2TrainingConfig(causal_enabled=False),
        **kwargs,
    ):
        if not math.isfinite(stage_eps) or stage_eps <= 0.0:
            raise ValueError("stage_eps must be finite and positive.")
        self.prev_solution = prev_solution
        for p in self.prev_solution.parameters():
            p.requires_grad_(False)
        self.stage_eps = float(stage_eps)
        pde_fn = partial(
            multi_stage_navier_stokes,
            prev_fn=self.prev_solution,
            eps=self.stage_eps,
            reynolds=config.reynolds,
        )
        super().__init__(network, sampler, config, pde_fn=pde_fn, **kwargs)

    def compile(self, *, device: str = "cpu"):
        self.prev_solution.to(device)
        return super().compile(device=device)

    def _corr_target(self, q: torch.Tensor, target, comp_idx: int) -> torch.Tensor:
        with torch.no_grad():
            q_dev = q.to(next(self.prev_solution.parameters()).device, dtype=q.dtype)
            prev_val = self.prev_solution(q_dev)[:, comp_idx : comp_idx + 1].cpu()
        t_tensor = torch.as_tensor(target, dtype=q.dtype)
        if t_tensor.ndim == 0:
            t_tensor = t_tensor.expand_as(prev_val)
        return (t_tensor - prev_val) / self.stage_eps

    def make_constraints(self, batch):
        constraints = []
        ic_fields = ("u", "v", "w", "p") if self.config.include_pressure_ic else ("u", "v", "w")
        for i, name in enumerate(ic_fields):
            target = self._corr_target(batch["initial"], batch["values"][:, i : i + 1], i)
            constraints.append(
                PointConstraint(f"{name}_ic", batch["initial"], target, component(i))
            )
        for axis in range(3):
            for side in (0, 1):
                for i, name in enumerate(("u", "v", "w")):
                    q_face = batch["faces"][(axis, side)]
                    raw_target = batch["face_values"][(axis, side)][:, i : i + 1]
                    target = self._corr_target(q_face, raw_target, i)
                    constraints.append(
                        PointConstraint(f"{name}_face_{axis}_{side}", q_face, target, component(i))
                    )
        for i, name in enumerate(("u", "v", "w")):
            target = self._corr_target(batch["sphere"], 0.0, i)
            constraints.append(
                PointConstraint(f"{name}_noslip", batch["sphere"], target, component(i))
            )
        target_p = self._corr_target(batch["pressure_anchor"], 0.0, 3)
        constraints.append(
            PointConstraint("p_anchor", batch["pressure_anchor"], target_p, component(3))
        )
        return constraints
