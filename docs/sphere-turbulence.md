# 3D sphere flow and JAXPI2 turbulence methods

`margam.research.jaxpi2` ports the 3D turbulence PINN methodology of Wang, Sankaran,
Fan, Stinis and Perdikaris (2025), *Simulating Three-dimensional Turbulence with
Physics-informed Neural Networks* ([arXiv:2507.08972v2](https://arxiv.org/abs/2507.08972v2)),
and its companion repository `sifanexisted/jaxpi2` to PyTorch on top of
`margam.research.Trainer`, `PINNProblem`, `PointConstraint` and `derivative`.

It is a **research API** module (`from margam.research import jaxpi2`): no desktop
GUI block or 3D CAD viewport is required, and 3D unsteady flow experiments run
directly from Python scripts or Slurm jobs.

---

## What is included in `margam.research.jaxpi2`?

| Component | Classes / functions | Key details |
| --- | --- | --- |
| **Backbone architectures** | `PirateNet`, `PirateBlock`, `ModifiedMlp`, `Mlp`, `ArchConfig`, `create_arch`, `TimeScaledField`, `SphereField` | 3-layer residual adaptive blocks with dual $(U, V)$ gating and trainable residual skip gates $\alpha^{(l)}$ (`nonlinearity=0.0` per Eq. 9 or `1.0` per `jaxpi2` configs) |
| **Embeddings & dense layers** | `PeriodEmbs`, `FourierEmbs`, `DirectFactorizedLinear`, `LecunLinear`, `stable_swish`, `stable_gelu` | Exact periodic embeddings $\gamma_i(x_i) = [\cos(\omega_i x_i), \sin(\omega_i x_i)]$, trainable Random Fourier features, Random Weight Factorization $W = \operatorname{diag}(g)V$, and 3rd-derivative-stable Swish $x\sigma(x)$ |
| **SOAP + Schedule-Free optimizer** | `Jaxpi2Optimizer`, `OptimConfig`, `create_lr_schedule` | PyTorch port of `SOAP_JAX v0.2.1` (`init_step` eigendecomposition on step 1, projected Adam + sorted QR power-iteration refresh every $f=2$ steps), `optax.contrib.schedule_free` ($z_k, x_k, y_k$ with `use_eval_params()`), `clip_by_global_norm(1.0)`, and ROCm `rocSOLVER` hardening |
| **3D Navier–Stokes & causality** | `navier_stokes`, `vorticity_3d`, `jaxpi2_causal_residual_losses`, `pseudo_time_residuals`, `compute_pts_weights` | 3D unsteady $(u, v, w, p)$ residuals, per-component causal temporal gating (`componentwise=True`), and variable-keyed pseudo-time damping $r_k + \omega_k^{\text{pts}}(u_k(\theta) - u_k(\theta_{\text{prev}}))$ |
| **Multi-stage perturbation** | `linearized_navier_stokes_operator`, `multi_stage_navier_stokes`, `MultiStageSolution`, `reject_sampling` | Linearized 3D Navier–Stokes perturbation operator around frozen Stage-0 solution for Stage-$k$ correction networks |
| **3D sphere trainers & samplers** | `Jaxpi2SphereTrainer`, `Jaxpi2CropTrainer`, `Jaxpi2MultiStageCropTrainer`, `Jaxpi2TrainingConfig`, `Domain`, `SpatialPools`, `SphereSampler`, `CropBoundaryReference`, `CropSampler` | Subclasses of `margam.research.Trainer` with post-step mean gradient-norm balancing $\hat{\lambda}_i = \bar{g}/(g_i + 10^{-5}\bar{g})$, Schedule-Free evaluation extraction, bit-exact checkpoint continuation, and time-window transfer (`transfer_from`) |

---

## Fixed-point parity against `jaxpi2` and `SOAP_JAX v0.2.1`

Every component in `margam.research.jaxpi2` was verified against the pinned upstream
references `sifanexisted/jaxpi2` (`77a5c1315a056388271822c35ad512a5a192b60d`) and
`haydn-jones/SOAP_JAX@v0.2.1` (`45145bcadb20864e821b87999c306e885bed7171`) using
exact Flax-to-PyTorch parameter transplantation (`reference_parameter_map()`):

- **`float64` parity**: **628 / 628 checks passed** (maximum absolute difference
  `2.58e-8` across forward passes, 1st/2nd spatial-temporal derivatives, 3D
  Navier–Stokes residuals and vorticity, per-component causal gates, pseudo-time
  weights, and 5-step `SOAP + Schedule-Free + ClipByGlobalNorm` trajectories;
  `9.31e-10` on multi-stage linearized residuals scaled by $(1/\varepsilon)^2 = 10^6$).
- **`float32` parity**: **628 / 628 checks passed** (maximum non-multistage absolute
  difference `5.17e-5`; relative multi-stage difference $\le 5 \times 10^{-7}$).

### AMD ROCm / MI300X `rocSOLVER` hardening

When `PirateNet` uses `nonlinearity=0.0` ($\alpha^{(l)} = 0$ at initialization),
all `PirateBlock` internal dense layers have zero gradients on steps 1–2 (`GG = 0`)
and tiny preconditioner entries ($\sim 4.65 \times 10^{-21} < \sqrt{\text{FLT\_MIN}} = 1.084 \times 10^{-19}$)
on steps 3–5. On AMD Instinct MI300X (`torch+rocm6.2`):

1. `rocsolver_ssyevd` (`torch.linalg.eigh` in `float32`) raises `LinAlgError` on
   `GG = 0` due to repeated zero eigenvalues.
2. `rocsolver_sgeqrf` (`torch.linalg.qr` in `float32`) silently underflows on
   squared column norms $< \text{FLT\_MIN}$ and returns non-orthogonal matrices
   ($\|Q^\top Q - I\|_\infty > 4 \times 10^4$).

`get_orthogonal_matrix` and `get_orthogonal_matrix_qr` in `margam.research.jaxpi2.optim`
normalize `power_iter` by its maximum magnitude prior to Householder QR, promote
ROCm (and `qr_dtype="float64"`) eigensolver/QR factorizations to `torch.float64`,
verify $\|Q^\top Q - I\|_\infty \le 10^{-2}$, and fall back to CPU LAPACK `float64`
if needed (maintaining $\|Q^\top Q - I\|_\infty \le 8.35 \times 10^{-7}$ on MI300X).

---

## 3D flow past a sphere ($\text{Re}=300$) benchmark results

The problem solves 3D unsteady incompressible Navier–Stokes flow around a stationary
sphere of radius $r = 0.5$ centered at the origin inside $[-8, 24] \times [-8, 8]^2$
over dimensionless time $t^* \in [0, 0.90]$ at $\text{Re} = 300$. Outer box faces
supply velocity Dirichlet traces interpolated from the 3D CFD reference, the sphere
surface enforces no-slip $(u, v, w) = (0, 0, 0)$, and the high-$(x, y, z)$ corner
anchors pressure $p = 0$.

Full-batch 3D training with dense spatial pools (`16,384` points/region, `8,192`
points/surface, `32,768` near-sphere-enriched initial points, and `8,192` PDE
collocation points per step) exceeds the memory/compute budget of a 6 GB laptop
RTX 3060 and was executed on an **AMD Instinct MI300X (192 GB)**. Held-out errors
at $t^* = 0.90$ (`8,192` independent interior evaluation nodes; see
[recorded evidence](../assets/benchmarks/sphere300-jaxpi2-mi300x.json)):

| Run | Hardware & budget | Architecture & optimizer | Velocity rel $L^2$ ($t^*=0.90$) | Near-sphere / wake vel rel $L^2$ | Fixed-gauge / centered pressure rel $L^2$ | Sphere no-slip RMS |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| **1. Expert PINN baseline** | RTX 3060 · 20k steps · 2,048 PDE · sparse 4k pools | `ModifiedMlp` (w=256, 398k params) + `Adam` | 6.40% | 23.73% / 5.15% | 178.17% / 129.91% | 4.25% $U$ |
| **2. JAXPI2 sparse-pool intervention** | RTX 3060 · 20k steps · 2,048 PDE · sparse 4k pools | `PirateNet` (w=256, 530k params) + `SOAP` + `Schedule-Free` | 2.53% | 9.99% / 0.83% | 45.26% / — | 0.236% $U$ |
| **3. JAXPI2 MI300X Run 0 (20k milestone)** | MI300X · 20k steps · 8,192 PDE · dense 16k/32k pools | `PirateNet` (w=256, 530k params) + `SOAP` + `Schedule-Free` | **0.294%** | 0.338% / 0.607% | 3.36% / — | — |
| **4. JAXPI2 MI300X Run 0 (50k final)** | MI300X · 50k steps · 8,192 PDE · dense 16k/32k pools | `PirateNet` (w=256, 530k params) + `SOAP` + `Schedule-Free` | **0.295%** | **0.270%** / 0.613% | **1.49%** / **0.960%** | 0.259% $U$ |
| **5. JAXPI2 MI300X Run 1 (width 512)** | MI300X · 50k steps · 8,192 PDE · dense 16k/32k pools | `PirateNet` (w=512, 2.11M params) + `SOAP` + `Schedule-Free` | **0.298%** | 0.383% / 0.603% | 8.87% / 1.52% | **0.120% $U$** |
| **6. JAXPI2 MI300X Run 2 (2 windows)** | MI300X · $2\times 25\text{k}$ steps · 8,192 PDE · dense pools | `PirateNet` (w=512, 2.11M params) + `SOAP` + `Schedule-Free` | **0.289%** (`0.184%` at $t^*=0.45$) | **0.270%** / **0.599%** | 72.88% / 15.04% (gauge offset) | — |

> **Time-marching pressure handoff takeaway (Run 2):** In incompressible
> Navier–Stokes, pressure is determined up to a time-dependent spatial constant
> fixed by the pressure gauge anchor. Constraining Window 2's $t=0$ pressure to
> Window 1's 25k-step predicted pressure locked in a uniform gauge offset
> ($\Delta p \approx -0.0299$) against the corner anchor. Pass
> `include_pressure_ic=False` on `Jaxpi2TrainingConfig` for subsequent time
> windows so only velocity $(u, v, w)$ is constrained at the window interface and
> pressure is fixed solely by the window's gauge anchor.

---

## Python tutorials

### Tutorial 1: Use `PirateNet` and `Jaxpi2Optimizer` (`SOAP` + `Schedule-Free`) with `margam.research.Trainer`

You can pair `PirateNet` and `Jaxpi2Optimizer` with the standard `margam.research.Trainer`
on any 1D, 2D or 3D PDE problem, evaluate using Schedule-Free averaged weights
(`with trainer.use_eval_params(): ...`), and save/load verified research checkpoints:

```python
from functools import partial
import torch
from margam.research import (
    ArchConfig,
    Jaxpi2Optimizer,
    OptimConfig,
    PINNProblem,
    PointConstraint,
    Trainer,
    create_arch,
    derivative,
)

torch.manual_seed(7)

# 1. Build a PirateNet with Random Fourier features, RWF, and Swish activation
arch_cfg = ArchConfig(
    arch_name="PirateNet",
    input_dim=2,
    out_dim=1,
    num_layers=2,
    hidden_dim=32,
    activation="swish",
    nonlinearity=0.0,  # Residual blocks start as identity at step 0
    fourier_emb={"embed_scale": 1.0, "embed_dim": 32},
    reparam={"type": "weight_fact", "mean": 1.0, "stddev": 0.1},
)
network = create_arch(arch_cfg)

# 2. Configure SOAP + Schedule-Free + global gradient norm clipping
optim_cfg = OptimConfig(
    optimizer="soap",
    learning_rate=1e-3,
    warmup_steps=50,
    decay_steps=500,
    schedule_free=True,
    grad_clip_norm=1.0,
    precondition_frequency=2,
)

# 3. Define a 1D space-time advection problem u_t + u_x = 0 on [0, 1] x [0, 2*pi]
t_x = torch.rand(128, 2)
t_x[:, 1] *= 2.0 * torch.pi
t0_x = torch.column_stack([torch.zeros(32), torch.linspace(0.0, 2.0 * torch.pi, 32)])

problem = PINNProblem(
    residual=lambda q, u: derivative(u, q, axis=0) + derivative(u, q, axis=1),
    collocation_points=t_x,
    constraints=[PointConstraint("ic", t0_x, target=torch.sin(t0_x[:, 1:2]))],
)

trainer = Trainer(problem, network, weights={"ic": 10.0}, checkpoint_id="piratenet-soap-demo-v1").compile(
    optimizer=partial(Jaxpi2Optimizer, config=optim_cfg),
    lr=optim_cfg.learning_rate,
    device="cpu",
)
trainer.train(20, display_every=10)
print("Learned PirateBlock skip gates alpha:", network.alphas())

# 4. Evaluate with Schedule-Free averaged weights x_k = (y_k - (1 - beta1) z_k) / beta1
with trainer.use_eval_params():
    pred = trainer.predict(t0_x)
print("Initial profile RMSE (Schedule-Free eval weights):", (pred - torch.sin(t0_x[:, 1:2])).square().mean().sqrt().item())
```

### Tutorial 2: 3D cropped sphere flow with `Jaxpi2CropTrainer`, window transfer and multi-stage correction

`Jaxpi2CropTrainer` runs the full 3D sphere-in-box training pipeline (`PirateNet`
wrapped in `SphereField`, 3D unsteady `navier_stokes` residuals, per-component
causal temporal gating, dynamic pseudo-time stepping, and post-step mean
gradient-norm balancing) and supports analytic or Jorvik-archived spatial pools:

```python
import numpy as np
import torch
from margam.research.jaxpi2 import (
    ArchConfig,
    CropBoundaryReference,
    CropSampler,
    Domain,
    Jaxpi2CropTrainer,
    Jaxpi2MultiStageCropTrainer,
    Jaxpi2TrainingConfig,
    MultiStageSolution,
    OptimConfig,
    SpatialPools,
    SphereField,
    create_arch,
)

torch.manual_seed(19)
domain = Domain(lower=(-8.0, -8.0, -8.0), upper=(24.0, 8.0, 8.0), radius=0.5, duration=1.0)

# 1. Build synthetic 5D outer-box boundary reference (T, [p,u,v,w], Nx, Ny, Nz) and 3D spatial pools
axes = [np.linspace(a, b, 5) for a, b in zip(domain.lower, domain.upper)]
t_grid, x_grid, y_grid, z_grid = np.meshgrid(np.arange(3) * 0.5, *axes, indexing="ij")
ref_array = np.stack([np.zeros_like(t_grid), np.ones_like(t_grid), np.zeros_like(t_grid), np.zeros_like(t_grid)], axis=1)
reference = CropBoundaryReference(ref_array, axes, dt=0.5, duration=1.0)

pools = SpatialPools.from_domain(domain, volume_count=64, boundary_count=16, seed=42)
q_ic = torch.tensor([[0.0, 1.0, 0.7, 0.6], [0.0, 2.0, -0.7, 0.8], [0.0, 3.0, 0.9, -0.7]])
val_ic = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

sampler = CropSampler(
    domain,
    q_ic,
    val_ic,
    reference=reference,
    residual_count=16,
    boundary_count=4,
    initial_count=3,
    spatial_pools=pools,
)

# 2. Construct PirateNet + SphereField and train Window 1
net_w1 = SphereField(
    create_arch(
        ArchConfig(
            arch_name="PirateNet",
            input_dim=4,
            out_dim=4,
            num_layers=1,
            hidden_dim=16,
            activation="swish",
            fourier_emb={"embed_scale": 1.0, "embed_dim": 16},
            nonlinearity=0.0,
        )
    )
)
cfg_w1 = Jaxpi2TrainingConfig(
    reynolds=300.0,
    causal_enabled=True,
    chunks=4,
    balance_start=0,
    balance_every=5,
    pseudo_time_enabled=True,
    pseudo_time_strategy="dynamic",
    pts_start=1,
    pts_every=5,
    optim=OptimConfig(optimizer="soap", learning_rate=1e-3, warmup_steps=5, decay_steps=100, schedule_free=True),
)
trainer_w1 = Jaxpi2CropTrainer(net_w1, sampler, cfg_w1).compile(device="cpu")
trainer_w1.train(5, display_every=0)
print("Window 1 min causal gate:", trainer_w1.minimum_causal_weight, "pts weights:", trainer_w1.pts_weights)

# 3. Transfer to Window 2 with velocity-only IC handoff (include_pressure_ic=False)
net_w2 = SphereField(create_arch(net_w1.core.config))
cfg_w2 = Jaxpi2TrainingConfig(
    reynolds=300.0,
    causal_enabled=True,
    chunks=4,
    include_pressure_ic=False,  # Let Window 2 corner gauge anchor fix pressure without offset
    optim=cfg_w1.optim,
)
trainer_w2 = Jaxpi2CropTrainer(net_w2, sampler, cfg_w2).compile(device="cpu").transfer_from(trainer_w1)
trainer_w2.train(2, display_every=0)

# 4. Stage-1 Multi-Stage linearized Navier-Stokes perturbation correction around frozen Stage-0
stage0_frozen = SphereField(create_arch(net_w1.core.config))
stage0_frozen.load_state_dict(trainer_w1.eval_state_dict())
stage1_net = SphereField(
    create_arch(
        ArchConfig(
            arch_name="PirateNet",
            input_dim=4,
            out_dim=4,
            num_layers=1,
            hidden_dim=16,
            activation="swish",
            fourier_emb={"embed_scale": 2.0, "embed_dim": 16},
            nonlinearity=0.0,
        )
    )
)
ms_trainer = Jaxpi2MultiStageCropTrainer(
    stage1_net,
    stage0_frozen,
    stage_eps=1e-3,
    sampler=sampler,
    config=Jaxpi2TrainingConfig(reynolds=300.0, causal_enabled=False, optim=cfg_w1.optim),
).compile(device="cpu")
ms_trainer.train(2, display_every=0)

composite = MultiStageSolution([stage0_frozen, stage1_net], [1.0, 1e-3])
print("Composite multi-stage prediction shape:", composite(q_ic).shape)
```

---

## Verification

Run the unit and scientific regression test suite:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_jaxpi2.py' -v
```
