"""3D Navier-Stokes residuals, causal gating, pseudo-time stepping, and multi-stage operators.

Matches sifanexisted/jaxpi2 (jaxpi/models.py, examples/taylor_green/models.py) and
arXiv:2507.08972v2 using Margam's shared autograd derivative helper.
"""
from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Callable, Tuple, Union
import torch
from torch import nn
from ...components.differential import derivative


@dataclass(frozen=True)
class Scales:
    """Dimensionless coordinate and field scaling for 3D flow around a sphere."""

    diameter: float
    speed: float
    density: float
    time_origin: float = 0.0
    pressure_reference: float = 0.0

    def __post_init__(self):
        if (
            not all(
                math.isfinite(v)
                for v in (
                    self.diameter,
                    self.speed,
                    self.density,
                    self.time_origin,
                    self.pressure_reference,
                )
            )
            or min(self.diameter, self.speed, self.density) <= 0
        ):
            raise ValueError("Physical scales must be positive.")

    def coordinates(self, dimensional_txyz: torch.Tensor) -> torch.Tensor:
        q = dimensional_txyz.clone()
        q[:, 0] = (q[:, 0] - self.time_origin) * self.speed / self.diameter
        q[:, 1:] = q[:, 1:] / self.diameter
        return q

    def fields(self, dimensional_uvwp: torch.Tensor) -> torch.Tensor:
        y = dimensional_uvwp.clone()
        y[:, :3] /= self.speed
        y[:, 3] = (y[:, 3] - self.pressure_reference) / (self.density * self.speed ** 2)
        return y


def navier_stokes(q: torch.Tensor, prediction: torch.Tensor, reynolds: float = 300.0) -> torch.Tensor:
    """Return 3D unsteady Navier-Stokes residuals (ru, rv, rw, rc) for q=(t,x,y,z), outputs=(u,v,w,p)."""
    if reynolds <= 0:
        raise ValueError("Reynolds number must be positive.")
    velocity = [prediction[:, i : i + 1] for i in range(3)]
    pressure = prediction[:, 3:4]
    spatial = [[derivative(v, q, axis) for axis in (1, 2, 3)] for v in velocity]
    momentum = []
    for i, v in enumerate(velocity):
        advection = sum(velocity[j] * spatial[i][j] for j in range(3))
        laplacian = sum(derivative(spatial[i][j], q, j + 1) for j in range(3))
        momentum.append(
            derivative(v, q, 0)
            + advection
            + derivative(pressure, q, i + 1)
            - laplacian / reynolds
        )
    continuity = sum(spatial[i][i] for i in range(3))
    return torch.cat((*momentum, continuity), dim=1)


def component(index: int):
    return lambda q, y: y[:, index : index + 1]


def normal_gradient(index: int, axis: int):
    return lambda q, y: derivative(y[:, index : index + 1], q, axis)


def vorticity_3d(q: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """Return 3D vorticity (vor_x, vor_y, vor_z) for q=(t,x,y,z) and prediction=(u,v,w,p)."""
    u, v, w = (prediction[:, i : i + 1] for i in range(3))
    u_y, u_z = derivative(u, q, 2), derivative(u, q, 3)
    v_x, v_z = derivative(v, q, 1), derivative(v, q, 3)
    w_x, w_y = derivative(w, q, 1), derivative(w, q, 2)
    return torch.cat((w_y - v_z, u_z - w_x, v_x - u_y), dim=1)


def causal_residual_losses(residuals: torch.Tensor, chunks: int, epsilon: float):
    """Shared-minimum causal gating across residual components on time-ordered chunks."""
    if len(residuals) % chunks:
        raise ValueError("Residual count must divide evenly into causal chunks.")
    components = residuals.shape[1]
    per_component = residuals.square().reshape(chunks, -1, components).mean(dim=1)
    prefix = torch.cat(
        (per_component.new_zeros((1, components)), per_component.detach().cumsum(dim=0)[:-1]),
        dim=0,
    )
    component_weights = torch.exp(-epsilon * prefix)
    shared_weights = component_weights.min(dim=1).values
    return (per_component * shared_weights[:, None]).mean(dim=0), shared_weights


def jaxpi2_causal_residual_losses(
    residuals: torch.Tensor,
    chunks: int,
    epsilon: float,
    *,
    componentwise: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Causally weighted residual losses matching jaxpi2 ForwardIVP._causal_losses.

    In jaxpi2 (jaxpi/models.py:625-659), each residual component is gated by its own
    earlier-chunk cumulative loss (componentwise=True), whereas compute_causal_weights
    for logging reports the minimum across components at each time chunk. Setting
    componentwise=False selects the jaxpi v1 shared-minimum rule.
    """
    if chunks < 1 or len(residuals) % chunks != 0:
        raise ValueError("Residual count must divide evenly into positive causal chunks.")
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("Causal tolerance epsilon must be finite and non-negative.")
    components = residuals.shape[1]
    per_component = residuals.square().reshape(chunks, -1, components).mean(dim=1)
    prefix = torch.cat(
        (per_component.new_zeros((1, components)), per_component.detach().cumsum(dim=0)[:-1]),
        dim=0,
    )
    component_weights = torch.exp(-epsilon * prefix).detach()
    min_weights = component_weights.min(dim=1).values
    gates = component_weights if componentwise else min_weights[:, None]
    losses = (per_component * gates).mean(dim=0)
    return losses, min_weights, component_weights


def pseudo_time_residuals(
    residuals: torch.Tensor,
    sol_pred: torch.Tensor,
    sol_prev: torch.Tensor,
    pts_weights: Union[torch.Tensor, Sequence[float]],
) -> torch.Tensor:
    """Shift PDE residuals by pseudo-time damping: r + w_pts * (sol - sol_prev)."""
    if residuals.shape != sol_pred.shape or residuals.shape != sol_prev.shape:
        raise ValueError("Residuals and solution matrices must have identical (N, C) shapes.")
    w = torch.as_tensor(pts_weights, dtype=residuals.dtype, device=residuals.device)
    if w.ndim != 1 or w.shape[0] != residuals.shape[1]:
        raise ValueError("pts_weights must have one scalar per residual component.")
    return residuals + w[None, :] * (sol_pred - sol_prev.detach())


def cosine_decay_from_loss(
    losses: torch.Tensor,
    losses0: torch.Tensor,
    *,
    start_log_drop: float = 3.0,
    end_log_drop: float = 5.0,
    min_factor: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Cosine shrink factor for pseudo-time weights based on log10 residual drop."""
    log_drop = torch.log10((losses0 + eps) / (losses + eps))
    p = torch.clamp((log_drop - start_log_drop) / (end_log_drop - start_log_drop), 0.0, 1.0)
    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + torch.cos(math.pi * p))


def compute_pts_weights(
    res_pred: torch.Tensor,
    res_prev: torch.Tensor,
    res0_pred: torch.Tensor,
    sol_pred: torch.Tensor,
    sol_prev: torch.Tensor,
    *,
    shrink_enabled: bool = True,
    start_log_drop: float = 3.0,
    end_log_drop: float = 5.0,
    min_factor: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Dynamic pseudo-time weight targets matching jaxpi2 PINN.compute_pts_weights."""
    r_cur = res_pred.detach()
    r_prev = res_prev.detach()
    r_0 = res0_pred.detach()
    s_cur = sol_pred.detach()
    s_prev = sol_prev.detach()

    losses0 = r_0.square().mean(dim=0)
    losses = r_cur.square().mean(dim=0)
    sol_diffs = s_cur - s_prev
    res_diffs = r_cur - r_prev

    if shrink_enabled:
        factors = cosine_decay_from_loss(
            losses,
            losses0,
            start_log_drop=start_log_drop,
            end_log_drop=end_log_drop,
            min_factor=min_factor,
            eps=eps,
        )
    else:
        factors = 1.0

    num = torch.linalg.vector_norm(res_diffs, dim=0)
    den = torch.linalg.vector_norm(sol_diffs, dim=0) + 1e-8
    weights = (num / den) * factors
    return torch.clamp(weights, 1e-2, 100.0).detach()


def linearized_navier_stokes_operator(
    q: torch.Tensor,
    diff_prediction: torch.Tensor,
    prev_prediction: torch.Tensor,
    *,
    reynolds: float = 300.0,
) -> torch.Tensor:
    """Linearized 3D Navier-Stokes operator f(diff; prev) around frozen prev_prediction.

    Matches MultiStage.f_net in jaxpi2/examples/taylor_green/models.py:196-238.
    """
    if reynolds <= 0:
        raise ValueError("Reynolds number must be positive.")
    v_diff = [diff_prediction[:, i : i + 1] for i in range(3)]
    p_diff = diff_prediction[:, 3:4]
    v_prev = [prev_prediction[:, i : i + 1] for i in range(3)]

    spatial_diff = [[derivative(v, q, axis) for axis in (1, 2, 3)] for v in v_diff]
    spatial_prev = [[derivative(v, q, axis).detach() for axis in (1, 2, 3)] for v in v_prev]
    v_prev_det = [v.detach() for v in v_prev]

    momentum = []
    for i in range(3):
        advection_lin = sum(
            v_prev_det[j] * spatial_diff[i][j] + spatial_prev[i][j] * v_diff[j]
            for j in range(3)
        )
        laplacian_diff = sum(derivative(spatial_diff[i][j], q, j + 1) for j in range(3))
        momentum.append(
            derivative(v_diff[i], q, 0)
            + advection_lin
            + derivative(p_diff, q, i + 1)
            - laplacian_diff / reynolds
        )
    continuity = sum(spatial_diff[i][i] for i in range(3))
    return torch.cat((*momentum, continuity), dim=1)


def multi_stage_navier_stokes(
    q: torch.Tensor,
    diff_prediction: torch.Tensor,
    prev_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    eps: float = 1e-3,
    reynolds: float = 300.0,
) -> torch.Tensor:
    """Correction stage residual f(diff; prev) + r(prev) / eps (arXiv:2507.08972v2 p. 17)."""
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("Stage scale eps must be finite and positive.")
    prev_pred = prev_fn(q)
    r_prev = navier_stokes(q, prev_pred, reynolds=reynolds).detach()
    f_diff = linearized_navier_stokes_operator(
        q, diff_prediction, prev_pred, reynolds=reynolds
    )
    return f_diff + r_prev / eps


class MultiStageSolution(nn.Module):
    """Composite multi-stage solution sum_{k} eps_k * stage_k(q)."""

    def __init__(self, stages: Sequence[nn.Module], eps_list: Sequence[float]):
        super().__init__()
        if len(stages) == 0 or len(stages) != len(eps_list):
            raise ValueError("MultiStageSolution requires matching non-empty stages and eps_list.")
        if any(not math.isfinite(e) or e <= 0 for e in eps_list):
            raise ValueError("All stage scaling factors eps must be finite and positive.")
        self.stages = nn.ModuleList(stages)
        self.eps_list = tuple(float(e) for e in eps_list)

    def forward_prev(self, q: torch.Tensor) -> torch.Tensor:
        if len(self.stages) < 2:
            raise ValueError("No previous stage exists when len(stages) < 2.")
        total = None
        for net, eps in zip(list(self.stages)[:-1], self.eps_list[:-1]):
            term = eps * net(q)
            total = term if total is None else total + term
        return total

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        total = None
        for net, eps in zip(self.stages, self.eps_list):
            term = eps * net(q)
            total = term if total is None else total + term
        return total


def reject_sampling(
    generator: torch.Generator,
    num_samples: int,
    pdf_fn: Callable[[torch.Tensor], torch.Tensor],
    dom: torch.Tensor,
    threshold: float = 0.5,
    batch_size: int = 4096,
) -> torch.Tensor:
    """Rejection-sample collocation points where residual magnitude pdf_fn is large."""
    if num_samples < 1 or batch_size < 1:
        raise ValueError("num_samples and batch_size must be positive integers.")
    dom = torch.as_tensor(dom)
    if dom.ndim != 2 or dom.shape[1] != 2:
        raise ValueError("dom must have shape (dim, 2) with (min, max) bounds.")
    dim = dom.shape[0]
    low = dom[:, 0]
    span = dom[:, 1] - dom[:, 0]

    accepted_batches = []
    num_accepted = 0
    while num_accepted < num_samples:
        u_coord = torch.rand((batch_size, dim), generator=generator, dtype=dom.dtype, device=dom.device)
        points = low + u_coord * span
        pdf_values = pdf_fn(points).detach()
        pdf_max = pdf_values.max()
        if float(pdf_max.item()) <= 0.0:
            accepted = points
        else:
            u_val = (
                torch.rand((batch_size,), generator=generator, dtype=dom.dtype, device=dom.device)
                * pdf_max
            )
            mask = (u_val <= pdf_values) & (pdf_values >= threshold * pdf_max)
            accepted = points[mask]
        if len(accepted) > 0:
            accepted_batches.append(accepted)
            num_accepted += len(accepted)
    return torch.cat(accepted_batches, dim=0)[:num_samples]
