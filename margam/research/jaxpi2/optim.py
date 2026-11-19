"""PyTorch conversion of JAXPI2 optimizers: SOAP, Adam, Schedule-Free, and LR schedules.

Matches sifanexisted/jaxpi2 (jaxpi/models.py, jaxpi/utils.py), haydn-jones/SOAP_JAX@v0.2.1
(soap_jax/soap.py), and optax.contrib.schedule_free. Implemented as a torch.optim.Optimizer
so it integrates directly with margam.research.Trainer.compile(optimizer=...).
"""
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Callable, List, Optional, Tuple, Union
import torch


@dataclass(frozen=True)
class OptimConfig:
    optimizer: str = "soap"
    lr_schedule: str = "exponential_decay"
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    learning_rate: float = 1e-3
    decay_rate: float = 0.9
    decay_steps: int = 2000
    warmup_steps: int = 2000
    staircase: bool = False
    end_learning_rate: float = 0.0
    schedule_free: bool = True
    grad_clip_norm: Optional[float] = 1.0
    precondition_frequency: int = 2
    max_precond_dim: int = 10000
    shampoo_beta: float = -1.0
    weight_decay: float = 0.0
    correct_bias: bool = True
    precondition_1d: bool = False
    weight_lr_power: float = 2.0
    qr_dtype: Optional[str] = "float32"


def create_lr_schedule(config: Union[OptimConfig, Mapping]) -> Callable[[int], float]:
    """Return a step-indexed learning-rate schedule matching jaxpi2/models.py."""
    if not isinstance(config, OptimConfig):
        config = OptimConfig(**dict(config))
    kind = config.lr_schedule.lower()
    peak = float(config.learning_rate)
    warmup = int(config.warmup_steps)
    decay_steps = int(config.decay_steps)

    if kind == "exponential_decay":
        decay_rate = float(config.decay_rate)
        staircase = bool(config.staircase)

        def exp_schedule(step: int) -> float:
            c = float(step)
            if warmup > 0 and c < warmup:
                return peak * (c / warmup)
            if decay_steps <= 0 or decay_rate == 0.0:
                return peak
            decreased = c - max(warmup, 0)
            if decreased <= 0.0:
                return peak
            p = decreased / decay_steps
            if staircase:
                p = math.floor(p)
            return peak * (decay_rate ** p)

        return exp_schedule

    if kind == "cosine_decay":
        end_val = float(config.end_learning_rate)
        span = decay_steps - warmup
        if span <= 0:
            raise ValueError(f"cosine_decay requires decay_steps ({decay_steps}) > warmup_steps ({warmup}).")
        alpha = 0.0 if peak == 0.0 else end_val / peak

        def cos_schedule(step: int) -> float:
            c = float(step)
            if warmup > 0 and c < warmup:
                return peak * (c / warmup)
            t = min(max(c - warmup, 0.0), float(span))
            cosine = 0.5 * (1.0 + math.cos(math.pi * t / span))
            return peak * ((1.0 - alpha) * cosine + alpha)

        return cos_schedule

    raise NotImplementedError(f"LR schedule {config.lr_schedule} not supported yet!")


def _resolve_qr_dtype(qr_dtype: Optional[Union[str, torch.dtype]], fallback: torch.dtype) -> torch.dtype:
    if qr_dtype is None:
        return fallback
    if isinstance(qr_dtype, torch.dtype):
        return qr_dtype
    if qr_dtype in ("float32", "torch.float32"):
        return torch.float32
    if qr_dtype in ("float64", "torch.float64"):
        return torch.float64
    raise ValueError(f"Unsupported qr_dtype: {qr_dtype}")


def init_conditioner(
    p: torch.Tensor,
    max_precond_dim: int,
    precondition_1d: bool,
    dtype: torch.dtype,
) -> List[Optional[torch.Tensor]]:
    if p.ndim <= 1:
        if p.ndim == 0 or not precondition_1d or p.shape[0] > max_precond_dim:
            return [None]
        return [torch.zeros((p.shape[0], p.shape[0]), dtype=dtype, device=p.device)]
    return [
        torch.zeros((s, s), dtype=dtype, device=p.device) if s <= max_precond_dim else None
        for s in p.shape
    ]


def update_preconditioner(
    grad: torch.Tensor,
    gg_list: List[Optional[torch.Tensor]],
    beta: float,
) -> List[Optional[torch.Tensor]]:
    if grad.ndim <= 1:
        if gg_list[0] is None:
            return [None]
        outer = grad[:, None] @ grad[None, :]
        gg0 = gg_list[0]
        return [gg0 + (1.0 - beta) * (outer - gg0)]
    new_gg: List[Optional[torch.Tensor]] = []
    for idx, gg in enumerate(gg_list):
        if gg is None:
            new_gg.append(None)
            continue
        axes = [a for a in range(grad.ndim) if a != idx]
        outer = torch.tensordot(grad, grad, dims=(axes, axes))
        new_gg.append(gg + (1.0 - beta) * (outer - gg))
    return new_gg


def project(grad: torch.Tensor, q_list: List[Optional[torch.Tensor]]) -> torch.Tensor:
    out = grad
    for mat in q_list:
        if mat is not None:
            out = torch.tensordot(out, mat, dims=([0], [0]))
        elif out.ndim > 1:
            out = out.permute(*range(1, out.ndim), 0)
    return out


def project_back(grad: torch.Tensor, q_list: List[Optional[torch.Tensor]]) -> torch.Tensor:
    out = grad
    for mat in q_list:
        if mat is not None:
            out = torch.tensordot(out, mat, dims=([0], [1]))
        elif out.ndim > 1:
            out = torch.moveaxis(out, 0, -1)
    return out


def get_orthogonal_matrix(
    gg: Optional[torch.Tensor],
    qr_dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if gg is None:
        return None
    use_fp64 = (qr_dtype == torch.float64) or (getattr(torch.version, "hip", None) is not None)
    work_dtype = torch.float64 if use_fp64 else qr_dtype
    gg_mat = gg.to(work_dtype) if gg.dtype != work_dtype else gg
    jitter = torch.tensor(1e-30, dtype=work_dtype, device=gg_mat.device)
    eye = torch.eye(gg_mat.shape[0], dtype=work_dtype, device=gg_mat.device)
    mat = gg_mat + jitter * eye
    try:
        _, eigh = torch.linalg.eigh(mat)
        if use_fp64:
            orth_err = (eigh.T @ eigh - eye).abs().max()
            if not bool((torch.isfinite(orth_err) & (orth_err <= 1e-2)).item()):
                raise RuntimeError("Non-orthogonal device eigh result")
    except RuntimeError:
        # ROCm rocSOLVER syevd can fail to converge on step-1 rank-1 or zero
        # preconditioner matrices that have repeated zero eigenvalues; fall back
        # to CPU LAPACK for the one-time step-1 eigenbasis initialization.
        mat_cpu = mat.detach().cpu()
        try:
            _, eigh_cpu = torch.linalg.eigh(mat_cpu)
        except RuntimeError:
            _, eigh_cpu = torch.linalg.eigh(mat_cpu.to(torch.float64))
            eigh_cpu = eigh_cpu.to(work_dtype)
        eigh = eigh_cpu.to(device=gg_mat.device, dtype=work_dtype)
    q = torch.flip(eigh, dims=[1])
    if q.dtype != gg.dtype:
        q = q.to(gg.dtype)
    return q


def get_orthogonal_matrix_qr(
    gg_list: List[Optional[torch.Tensor]],
    q_list: List[Optional[torch.Tensor]],
    exp_avg_sq: torch.Tensor,
    qr_dtype: torch.dtype,
) -> Tuple[List[Optional[torch.Tensor]], torch.Tensor]:
    final_q: List[Optional[torch.Tensor]] = []
    use_fp64_qr = (qr_dtype == torch.float64) or (getattr(torch.version, "hip", None) is not None)
    work_dtype = torch.float64 if use_fp64_qr else qr_dtype
    for ind, (m, o) in enumerate(zip(gg_list, q_list)):
        if m is None or o is None:
            final_q.append(None)
            continue
        m_mat = m.to(work_dtype) if m.dtype != work_dtype else m
        o_mat = o.to(work_dtype) if o.dtype != work_dtype else o
        est_eig = torch.diag(o_mat.T @ m_mat @ o_mat)
        sort_idx = torch.argsort(est_eig, descending=True, stable=True)
        exp_avg_sq = torch.index_select(exp_avg_sq, dim=ind, index=sort_idx)
        o_mat = o_mat[:, sort_idx]
        power_iter = m_mat @ o_mat
        if use_fp64_qr:
            p_max = power_iter.abs().max()
            scale = torch.where(p_max > 0.0, p_max, torch.ones_like(p_max))
            power_iter = power_iter / scale
        try:
            q_new, _ = torch.linalg.qr(power_iter)
            if use_fp64_qr:
                eye = torch.eye(q_new.shape[0], dtype=work_dtype, device=q_new.device)
                orth_err = (q_new.T @ q_new - eye).abs().max()
                if not bool((torch.isfinite(orth_err) & (orth_err <= 1e-2)).item()):
                    raise RuntimeError("Non-orthogonal device QR result")
        except RuntimeError:
            q_cpu, _ = torch.linalg.qr(power_iter.detach().cpu().to(torch.float64))
            q_new = q_cpu.to(device=m_mat.device, dtype=work_dtype)
        if q_new.dtype != m.dtype:
            q_new = q_new.to(m.dtype)
        final_q.append(q_new)
    return final_q, exp_avg_sq


class Jaxpi2Optimizer(torch.optim.Optimizer):
    """PyTorch Optimizer implementing JAXPI2's SOAP/Adam + Schedule-Free + ClipByGlobalNorm."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        config: Optional[Union[OptimConfig, Mapping]] = None,
    ):
        if config is None:
            config = OptimConfig(learning_rate=lr)
        elif not isinstance(config, OptimConfig):
            config = OptimConfig(**dict(config))
        kind = config.optimizer.lower()
        if kind not in ("soap", "adam"):
            raise NotImplementedError(f"Optimizer {config.optimizer} not supported yet!")
        if not (0.0 <= config.beta1 < 1.0) or not (0.0 <= config.beta2 < 1.0):
            raise ValueError("beta1 and beta2 must lie in [0, 1).")
        if config.schedule_free and config.beta1 <= 0.0:
            raise ValueError("schedule_free requires strictly positive beta1.")
        if config.eps <= 0.0 or config.precondition_frequency <= 0 or config.max_precond_dim <= 0:
            raise ValueError("eps, precondition_frequency, and max_precond_dim must be positive.")

        self.config = config
        self.lr_fn = create_lr_schedule(config)
        defaults = dict(
            lr=float(config.learning_rate),
            betas=(float(config.beta1), float(config.beta2)),
            eps=float(config.eps),
            weight_decay=float(config.weight_decay),
        )
        super().__init__(params, defaults)
        self._step_count = 0  # 0-based completed updates

    @property
    def current_lr(self) -> float:
        return float(self.lr_fn(self._step_count))

    def _init_param_state(self, p: torch.Tensor):
        state = self.state[p]
        if state:
            return state
        qr_dtype = _resolve_qr_dtype(self.config.qr_dtype, p.dtype)
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p)
        state["exp_avg_sq"] = torch.zeros_like(p)
        if self.config.optimizer.lower() == "soap":
            state["GG"] = init_conditioner(
                p, self.config.max_precond_dim, self.config.precondition_1d, qr_dtype
            )
            state["Q"] = init_conditioner(
                p, self.config.max_precond_dim, self.config.precondition_1d, qr_dtype
            )
        if self.config.schedule_free:
            state["z"] = p.detach().clone()
            state["sf_step_count"] = 1
            state["sf_weight_sum"] = 0.0
            state["sf_max_lr"] = 0.0
        return state

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("Closure-based optimization is not supported.")

        params_with_grad: List[torch.nn.Parameter] = []
        grads: List[torch.Tensor] = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad.detach())
                    self._init_param_state(p)

        if not params_with_grad:
            return None

        # 1. Optional global gradient norm clipping (applied when schedule_free is enabled
        # in jaxpi2/models.py:140-144).
        if self.config.schedule_free and self.config.grad_clip_norm is not None:
            max_norm = float(self.config.grad_clip_norm)
            sq_sum = sum((g.square().sum() for g in grads), grads[0].new_zeros(()))
            g_norm = torch.sqrt(sq_sum)
            if bool((g_norm >= max_norm).item()) and bool((g_norm > 0.0).item()):
                grads = [(g / g_norm.to(g.dtype)) * max_norm for g in grads]

        b1 = float(self.config.beta1)
        b2 = float(self.config.beta2)
        shampoo_beta = float(self.config.shampoo_beta) if self.config.shampoo_beta >= 0.0 else b2
        eps = float(self.config.eps)
        kind = self.config.optimizer.lower()
        lr_t = float(self.lr_fn(self._step_count))

        for p, grad in zip(params_with_grad, grads):
            state = self.state[p]
            count_inc = int(state["step"]) + 1
            state["step"] = count_inc

            if kind == "adam":
                exp_avg = b1 * state["exp_avg"] + (1.0 - b1) * grad
                exp_avg_sq = b2 * state["exp_avg_sq"] + (1.0 - b2) * grad.square()
                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq
                bc1 = 1.0 - (b1 ** count_inc)
                bc2 = 1.0 - (b2 ** count_inc)
                m_hat = exp_avg / bc1
                v_hat = exp_avg_sq / bc2
                direction = m_hat / (torch.sqrt(v_hat) + eps)
                base_update = -lr_t * direction
                if self.config.weight_decay != 0.0:
                    base_update = base_update - (lr_t * float(self.config.weight_decay)) * p
            else:
                # SOAP (SOAP_JAX v0.2.1 semantics)
                qr_dtype = _resolve_qr_dtype(self.config.qr_dtype, p.dtype)
                if count_inc == 1:
                    new_gg = update_preconditioner(grad, state["GG"], shampoo_beta)
                    new_q = [get_orthogonal_matrix(m, qr_dtype) for m in new_gg]
                    state["GG"] = new_gg
                    state["Q"] = new_q
                    base_update = torch.zeros_like(p)
                else:
                    old_q = state["Q"]
                    grad_proj = project(grad, old_q)
                    exp_avg = b1 * state["exp_avg"] + (1.0 - b1) * grad_proj
                    exp_avg_sq = b2 * state["exp_avg_sq"] + (1.0 - b2) * grad_proj.square()
                    norm_update = project_back(exp_avg / (torch.sqrt(exp_avg_sq) + eps), old_q)
                    if self.config.correct_bias:
                        effective_step = count_inc - 1
                        bc1 = 1.0 - (b1 ** effective_step)
                        bc2 = 1.0 - (b2 ** effective_step)
                        corr = math.sqrt(bc2) / bc1
                        norm_update = norm_update * corr
                    new_gg = update_preconditioner(grad, state["GG"], shampoo_beta)
                    if (count_inc - 1) % self.config.precondition_frequency == 0:
                        new_q, exp_avg_sq = get_orthogonal_matrix_qr(
                            new_gg, old_q, exp_avg_sq, qr_dtype
                        )
                        exp_avg = project(project_back(exp_avg, old_q), new_q)
                    else:
                        new_q = old_q
                    state["GG"] = new_gg
                    state["Q"] = new_q
                    state["exp_avg"] = exp_avg
                    state["exp_avg_sq"] = exp_avg_sq
                    base_update = -lr_t * norm_update
                    if self.config.weight_decay != 0.0:
                        base_update = base_update - (lr_t * float(self.config.weight_decay)) * p

            if self.config.schedule_free:
                sf_step = int(state["sf_step_count"])
                sf_lr = float(self.lr_fn(sf_step))
                max_lr = max(float(state["sf_max_lr"]), sf_lr)
                weight = max_lr ** float(self.config.weight_lr_power)
                next_total_weight = float(state["sf_weight_sum"]) + weight
                ck = 0.0 if next_total_weight == 0.0 else (weight / next_total_weight)

                z_old = state["z"]
                z_new = (z_old + base_update).to(p.dtype)
                prev_x = (p - (1.0 - b1) * z_old) / b1
                x_new = (1.0 - ck) * prev_x + ck * z_new
                p_new = b1 * x_new + (1.0 - b1) * z_new

                p.copy_(p_new)
                state["z"] = z_new
                state["sf_step_count"] = sf_step + 1
                state["sf_weight_sum"] = next_total_weight
                state["sf_max_lr"] = max_lr
            else:
                p.add_(base_update)

        self._step_count += 1
        for group in self.param_groups:
            group["lr"] = self.current_lr
        return None

    def get_eval_param(self, p: torch.nn.Parameter) -> torch.Tensor:
        """Return schedule-free evaluation parameter x = (y - (1 - b1) z) / b1."""
        if not self.config.schedule_free:
            return p.detach().clone()
        state = self.state.get(p)
        if not state or "z" not in state:
            return p.detach().clone()
        b1 = float(self.config.beta1)
        return ((p.detach() - (1.0 - b1) * state["z"]) / b1).clone()

    @contextmanager
    def use_eval_params(self):
        """Temporarily replace parameters with schedule-free evaluation parameters."""
        if not self.config.schedule_free:
            yield
            return
        saved = []
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    state = self.state.get(p)
                    if state and "z" in state:
                        saved.append((p, p.detach().clone()))
                        b1 = float(self.config.beta1)
                        p.copy_((p - (1.0 - b1) * state["z"]) / b1)
        try:
            yield
        finally:
            with torch.no_grad():
                for p, orig in saved:
                    p.copy_(orig)

    def state_dict(self):
        base = super().state_dict()
        base["jaxpi2_step_count"] = self._step_count
        return base

    def load_state_dict(self, state_dict):
        copied = dict(state_dict)
        step_count = copied.pop("jaxpi2_step_count", 0)
        super().load_state_dict(copied)
        self._step_count = int(step_count)
        for group in self.param_groups:
            group["lr"] = self.current_lr
            for p in group["params"]:
                state = self.state.get(p)
                if not state:
                    continue
                for key in ("exp_avg", "exp_avg_sq", "z"):
                    if key in state and isinstance(state[key], torch.Tensor):
                        state[key] = state[key].to(device=p.device, dtype=p.dtype)
                for key in ("GG", "Q"):
                    if key in state and isinstance(state[key], list):
                        state[key] = [
                            m.to(device=p.device) if isinstance(m, torch.Tensor) else None
                            for m in state[key]
                        ]
