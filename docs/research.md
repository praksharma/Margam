# Use Margam from your research code

Write the new idea as ordinary PyTorch code, then use Margam to prepare data,
train and record losses. Your experiment can live in its own folder or Git
repository. No GUI block, registration file or Margam source edit is required.

`margam.research` is part of the existing package, with a script-facing `Trainer`.
It reuses the existing `PINN` training loop, problem definitions, constraints,
autograd helpers and prediction code. It is separate from desktop recipes and
model documents; it is not a second implementation of the numerical engine.

## A complete minimal experiment

Run this in your research project using an interpreter where Margam is installed:

```python
from functools import partial
import torch
from margam.research import Trainer, PINNProblem, PointConstraint, derivative

# Your bare PyTorch module. Existing Margam networks can also be imported.
network = torch.nn.Sequential(
    torch.nn.Linear(1, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)
)

# Your sampling function; it is evaluated once when compile() prepares data.
def my_points():
    return torch.quasirandom.SobolEngine(1, scramble=True, seed=7).draw(64)

# Your physics: u_xx + 2 = 0, with zero endpoint values.
def residual(x, u):
    return derivative(u, x, axis=0, order=2) + 2

problem = PINNProblem(
    residual=residual,
    collocation_points=my_points,
    constraints=[PointConstraint("boundary", torch.tensor([[0.], [1.]]))],
)

trainer = Trainer(problem, network, weights={"boundary": 10}).compile(
    optimizer=partial(torch.optim.AdamW, weight_decay=1e-4),
    lr=0.003,
    device="auto",
)
trainer.train(1000, display_every=100)
prediction = trainer.predict(torch.linspace(0, 1, 101)[:, None])
print(trainer.loss_history[-1])
```

The example samples only interior PDE rows. To apply the PDE at endpoints too,
explicitly include those coordinates in `my_points()`. Constraints remain separate.
For repeated experiments, seed PyTorch **before constructing the network** and
use seeded samplers. The runnable example does both.

## Which parts can I replace?

| Research change | Supply or override |
| --- | --- |
| Neural network | Any pointwise `torch.nn.Module` returning `(N, C)` values |
| Physics | `PINNProblem.residual(points, prediction)` |
| Collocation sampling | Tensor/array rows or a zero-argument function in `collocation_points` |
| Total scalar flow/flux | `IntegralConstraint` with explicit fixed quadrature weights |
| Boundary/initial/data points | `PointConstraint(name, points, target, operator)` |
| Derivative BC or custom operator | `operator(points, prediction)` on a point constraint |
| Optimiser | `compile(optimizer=factory)`; factory accepts `parameters, lr=...` |
| Learning-rate schedule | `compile(schedule=StepDecay(every=..., gamma=...))` |
| Loss reduction | `Trainer(..., loss=callable)` or a per-term mapping |
| New named objective terms | Subclass `Trainer` and override `compute_losses()` |
| Gradient update | `Trainer(..., step=function)` or subclass `update()` |

Most experiments need functions and a PyTorch module, not a subclass. Existing
components remain directly importable:

```python
from margam.nn import FullyConnected
from margam.nn.fourier import FourierNetwork
from margam.components.sampling import sample_box
from margam.components.differential import derivative
from margam.methods.self_adaptive import SelfAdaptivePINN
```

The catalogue is in [Components](components.md); existing advanced methods keep
their own supported combinations. This research trainer does not automatically
compose SA, LRA, Expert or other algorithms when you supply a new callback.

## Replace a loss

A loss is a normal callable receiving aligned prediction and target tensors and
returning one scalar tensor. It must keep parameter gradients attached.

```python
def my_loss(prediction, target):
    error = prediction - target
    return error.square().mean() + 0.01 * error.pow(4).mean()

trainer = Trainer(problem, network, loss={"pde": my_loss}).compile(
    optimizer=torch.optim.SGD, lr=0.001, device="cpu"
)
```

A callable replaces every term's MSE; a mapping replaces only the named terms.
Unlisted terms retain MSE. Unknown names are rejected. Callable targets and
scalar targets are aligned before your loss is called. PyTorch loss modules
move to the selected device; plain functions manage any tensors they capture.
Trainable loss modules need an explicit custom step managing their parameters.
Stateful loss modules cannot use the standard research checkpoint.

Custom losses currently use fixed points and mean mode. The existing integral
squared-error formulation remains available with `loss=None`; an arbitrary custom
loss is not silently interpreted as a physical integral.

To add a stateless extra objective term, reuse physical losses and add a scalar:

```python
class RegularisedPINN(Trainer):
    def compute_losses(self):
        terms = super().compute_losses()
        terms["regularisation"] = 1e-6 * sum(
            parameter.square().sum() for parameter in self.network.parameters()
        )
        return terms
```

Extra terms default to weight one; apply their coefficient in the returned
expression, as above. Keep `pde` and each named constraint present. `total` and
the `weighted:` prefix are reserved for logging.

## Constrain total flow or another scalar integral

Use `IntegralConstraint` when the **total** of a field/flux must match a target:

```python
from margam.research import IntegralConstraint

# Cross-section x=1, y in [0,2], for a scalar axial-velocity network u(x,y).
# Simpson weights include the physical cross-section measure (sum = 2).
flow = IntegralConstraint(
    "flow_rate",
    torch.tensor([[1., 0.], [1., 1.], [1., 2.]], dtype=torch.float64),
    target=4/3,
    quadrature_weights=[1/3, 4/3, 1/3],
)
# Add flow to PINNProblem.constraints alongside your PointConstraint instances,
# then compile/train with margam.research.Trainer as usual.
```

With default MSE, this adds one scalar term:

```text
Q_prediction = sum_i quadrature_weights[i] * q_theta(points[i])
L_flow_rate   = (Q_prediction - target)^2
```

The sum is formed **before** squaring. Opposing signed field values can cancel;
this is distinct from [integrating squared pointwise errors](integral-loss.md),
where they cannot. Weights are not normalized: their sum carries physical measure.
The constraint's inherited `measure` must stay **1**, including in integral loss
mode, so the physical measure is not applied twice. Set its objective coefficient
with `Trainer(..., weights={"flow_rate": coefficient})`; existing raw and
`weighted:flow_rate` histories retain their usual meaning.

For a vector network or a derivative flux, supply an explicit scalar pointwise
`operator(points, prediction)`. For example, `operator=lambda x, uvp: uvp[:, 0:1]`
selects axial velocity. Use the existing derivative helpers for derivative-based
fluxes; gradients through the operator and network remain attached. Normals,
orientation, units and the physical integrand are your explicit research code;
Margam does not infer them from geometry or boundary names.

Provide one finite, nonnegative quadrature weight per point, with positive finite
total measure, and one finite fixed scalar target. Zero-weight rows are allowed;
negative/signed quadrature rules, complex/bool weights, callable targets and vector
integrands are rejected. Weights may be `(N,)` or `(N,1)`. The operator must return
`(N,)` or `(N,1)` before reduction. A custom per-term loss receives the integrated
`(1,1)` prediction and scalar target, rather than the individual quadrature rows.

Points/samplers follow the usual compile-once rule. Compiled weights are detached
copies on the network's dtype/device; changing the source array does not silently
change a running experiment. Standard research continuation fingerprints actual
quadrature weights, coordinates and targets. Reconstruct them before loading;
changed quadrature is rejected. Unscheduled/scheduled research formats remain
1/2, and ordinary point-only checkpoints retain their existing identity. Functions
or constants captured by an operator are not automatically hashed: version
`checkpoint_id` or `problem.data_identity` when their semantics/data change.

This first implementation is a **research API** capability. Desktop trainers and
ordinary `PINN` reject the new constraint rather than interpreting it as a pointwise
BC; there is no new GUI block or imported-mesh/quadrature contract yet.

Run the complete example:

```sh
.venv/bin/python -m examples.research_integral_flow --device cpu
.venv/bin/python -m examples.research_integral_flow --device cuda
```

[The script](../examples/research_integral_flow.py) uses a scalar parabolic
axial-velocity ansatz, continuity, wall values and an explicit cross-section flow
constraint. Its ten-update [CPU](../assets/benchmarks/integral-constraint-cpu.json)
and [CUDA](../assets/benchmarks/integral-constraint-cuda.json) runs reproduce split
continuation exactly, including step decay. They are API/gradient checks, not a
full Navier–Stokes model or a solution-accuracy benchmark.

## Change the update without rebuilding the trainer

```python
def my_step(trainer, terms, total):
    total.backward()
    torch.nn.utils.clip_grad_norm_(
        trainer.network.parameters(), max_norm=1.0, error_if_nonfinite=True
    )
    trainer.optimizer.step()

trainer = Trainer(problem, network, step=my_step).compile(device="auto")
```

For each update Margam prepares fresh derivative graphs, zeroes the main
optimiser's gradients, computes and validates scalar losses, then calls your
step. Your function owns backward/update work; Margam owns iteration counting,
named history and prediction. Losses are recorded **before** the update;
prediction uses the weights after it. Constraint contributions use the weights
in effect before the custom update changes them.

An algorithm with extra optimisers, masks, trainable coefficients or schedules
can own those objects in a subclass or function closure. You must manage their
devices, extra zeroing and complete persistence. Parameters to be optimised by the
main optimiser should be registered on your network module. The default update
checks finite gradients before stepping; a custom step owns its own update checks.
Closure-based optimisers such as LBFGS need a different procedure and are rejected
by this initial interface.

## Decay the learning rate by update count

Use an explicit schedule for an ordinary fixed-data run:

```python
from margam.research import StepDecay

trainer = Trainer(
    problem, network, checkpoint_id="my-decayed-experiment-v1",
).compile(lr=0.003, device="auto", schedule=StepDecay(every=1000, gamma=0.5))
trainer.train(2500, display_every=100)
print(trainer.loss_history[-1]["learning_rate:0"])
print(trainer.optimizer.param_groups[0]["lr"])  # Rate for the next update
```

`every` counts completed optimizer updates, not epochs or calls to `train()`.
Zero-based update `k` uses `initial_lr * gamma ** (k // every)`. For example,
`every=2, gamma=0.5, lr=0.008` gives update rates
`0.008, 0.008, 0.004, 0.004, 0.002, ...`. Rates are computed from the initial
values, rather than repeatedly multiplying rounded values. Each parameter group
keeps its own initial rate. `gamma` must be finite in `(0, 1]`; `every` must be a
positive integer. Underflow to zero is rejected before an optimizer update.

The trainer applies the next rate only after the optimizer step succeeds. Each
history record contains the rate actually used as `learning_rate:0`,
`learning_rate:1`, and so on. Separate `train()` calls continue the same count;
`compile()` starts a new optimizer, schedule and history. Do not edit optimizer
rates manually or call another scheduler alongside this one: external rate changes
are rejected on the next update or checkpoint operation.

Both built-in schedules require the default research update. Custom update hooks,
metric-driven schedulers, warmup and arbitrary PyTorch scheduler objects still
own their timing and persistence. This option does not add a desktop block or
alter desktop trainers.

Full-batch scheduled checkpoints use research format 2, storing schedule settings, initial
rates, the completed count and actual per-update rate history with the optimizer.
Recreate the same schedule and initial rates before loading. Resume restores the
next rate without taking an extra scheduler step. Format 1 checkpoints remain
supported for unscheduled runs; a scheduled checkpoint cannot be loaded into an
unscheduled trainer. In-memory checkpoint optimizer tensors are copied on restore,
so continuing a run does not mutate the supplied snapshot.

Eight-update example runs on [CPU](../assets/benchmarks/research-step-decay-cpu.json)
and [CUDA](../assets/benchmarks/research-step-decay-cuda.json) both reproduce split
continuation exactly. With `lr=0.003, every=2, gamma=0.5`, their last update uses
`0.000375` and the next update would use `0.0001875`. These short checks establish
schedule and checkpoint behaviour, not solution accuracy.

### Cosine decay to a fixed floor

For a finite training budget, use the importable `CosineDecay` schedule:

```python
from margam.research import CosineDecay

trainer = Trainer(
    problem, network, checkpoint_id="my-cosine-experiment-v1",
).compile(
    lr=0.003, device="auto",
    schedule=CosineDecay(steps=10000, minimum_ratio=0.05, power=1.0),
)
trainer.train(10000, display_every=100)
```

For zero-based update `k`, each optimizer group's initial rate is multiplied by:

```text
minimum_ratio + (1-minimum_ratio) *
    ((1 + cos(pi * min(k, steps) / steps)) / 2) ** power
```

`steps` is the number of completed updates over which the decay occurs, not the
number of epochs or `train()` calls. The first update uses the initial rate.
Update `steps` and all later updates use `initial_rate * minimum_ratio`.
**Training continues at that floor** if you request more updates; there is no
implicit stopping, warm restart or cosine cycle. To continue an existing schedule,
keep its original horizon when rebuilding the trainer to load a checkpoint.

`steps` must be a positive integer, `minimum_ratio` must be finite in `(0, 1]`,
and `power` must be finite and positive. Defaults are `minimum_ratio=0.01` and
`power=1`. The floor is a ratio of each group's own initial rate, so groups with
different starting rates keep different floors. Zero rates and floating-point
underflow are rejected before updating the optimizer. `power` shapes the cosine
factor **before** adding the floor. This matches the rate formula in the separate
[SCALE cavity driver](scale-cavity-benchmark.md); it does not change that case's
network, sampling, correction or training protocol.

Rate logging, failed-update behavior and external-rate guards follow the same
contract as StepDecay. Research checkpoint format 2 stores the schedule kind,
horizon, floor ratio, power, initial group rates and actual rate history.
Changing any of those settings rejects continuation. Existing StepDecay format-2
and unscheduled format-1 checkpoint identities remain unchanged. Standard resume
continues the optimizer, schedule and Torch RNG rather than taking an extra decay
step. A weights-only warm restart requires an explicitly new experiment.

Run the existing custom-PyTorch example with cosine decay:

```sh
.venv/bin/python -m examples.research_custom --cosine-decay --device cpu --output build/research-cosine-cpu
.venv/bin/python -m examples.research_custom --cosine-decay --device cuda --output build/research-cosine-cuda
```

Choose distinct output folders for experiments. `--cosine-decay` and
`--step-decay` are mutually exclusive and both require the default update;
`--custom-step` owns its scheduling and persistence. The example's cosine
horizon is `--steps` (eight by default), with a 5% floor and power one. Its
[CPU](../assets/benchmarks/research-cosine-cpu-2026-11-14.json) and
[CUDA](../assets/benchmarks/research-cosine-cuda-2026-11-14.json) reports verify
exact split continuation, record selected source hashes and finish with a next
rate of `0.00015` from an initial `0.003`. These eight-update checks establish
schedule/lifecycle behavior, not solution accuracy. No desktop block or default
is changed.

## Bound pointwise training graphs

For deterministic fixed-data mean-MSE problems, compile with
`pointwise_microbatch=256` to accumulate smaller derivative graphs before one
optimizer update. This is an explicit pointwise contract covering the network,
residual and constraint operators, with research checkpoint format 3 and strict
chunk-size continuation. See [supported components, normalization and measured
CPU/CUDA workspace](research-microbatch.md). Defaults and desktop trainers stay
unchanged; custom losses/steps, adaptive masks and global integrals are excluded.

## Shapes, sampling and device

Points are nonempty `(N, D)` arrays/tensors. Scalar/vector residuals and constraint
predictions are `(N,)` or `(N, C)`. Return individual field derivatives explicitly
for coupled equations; do not differentiate a sum of unrelated vector fields.
Derivative helpers assume pointwise networks: batch-coupled layers can change
the meaning of a summed autograd derivative.

Zero-argument samplers run once per compile; training does not silently redraw
points. Margam copies inputs onto the network's dtype and selected device, owns
the derivative tensors and leaves your source arrays untouched. `predict()`
returns a tensor on that device, without gradients, and restores network mode.
`device="auto"` chooses CUDA when available. Python callbacks are ordinary trusted
research code, outside the desktop's restricted expression parser.

`compile()` resets optimiser state, samples and histories. `train(n)` performs
exactly `n` additional updates. Do not call compile between segments of a run
you intend to continue in memory.

## Save and continue your experiment

For fixed data, stateless loss code and the default update, research checkpoints
support **Adam, AdamW and SGD**, including their parameter groups and momentum.
Supply a versioned identity for your physics, network and loss implementation:

```python
trainer = Trainer(
    problem, network, loss=my_loss,
    checkpoint_id="my-paper-experiment-v1",
).compile(optimizer=torch.optim.AdamW, device="cpu")
trainer.train(500, display_every=0)
trainer.save_checkpoint("experiment.pt")

# In another process: recreate the same problem, network, loss and optimiser,
# set the same checkpoint_id, compile, then:
trainer.load_checkpoint("experiment.pt")
trainer.train(500, display_every=0)
```

Saved state contains tensors and primitives, never Python functions, modules or
pickled model objects. Loading uses `weights_only=True`; it does not import your
research code. It validates your declared identity, actual fixed points/targets,
weights, loss measures, network tensors, optimiser options and history. A failed
native restore rolls back weights, optimiser and Torch RNG state.

Increment `checkpoint_id` when you change physics, architecture semantics,
custom loss code or any behaviour not described by the saved numeric settings.
It is an explicit researcher declaration, **not an automatic source-code hash**.
The checkpoint cannot prove that two functions with the same ID behave identically.

Continuation retains Torch CPU and selected CUDA RNG state and currently requires
the **same device**, compatible PyTorch runtime and reconstructed components.
Loading replaces those global Torch RNG states. Python/NumPy RNG, external
mutable closures, external schedulers and arbitrary custom algorithm state are not captured.
The explicit `StepDecay` and `CosineDecay` schedules above are included in standard continuation.
A custom update hook, stateful loss or custom optimiser class must implement its
own complete state contract; the standard save operation rejects these cases.
Research checkpoints are separate from desktop saved models and run archives.

## Run the supplied external-style example

```sh
.venv/bin/python -m examples.research_custom --device cpu
.venv/bin/python -m examples.research_custom --device cuda --output build/research-cuda
.venv/bin/python -m examples.research_custom --step-decay --device cpu --output build/research-decay
.venv/bin/python -m examples.research_custom --custom-step --device cpu --output build/research-step
```

[The script](../examples/research_custom.py) defines a custom network, Sobol
sampler, residual and loss in ordinary Python, uses AdamW, and writes predictions
and a small JSON record. Default eight-update runs verify exact split continuation;
the custom-step version demonstrates clipping and deliberately does not create a
standard checkpoint. Output is under ignored `build/`; choose `--steps` for an
actual experiment. These short runs establish API/lifecycle behaviour, not accuracy.

The dedicated tests also execute a script from a temporary folder outside the
repository and verify that no GUI, MCP package or block registration is loaded.

## Inspect selected errors

`from margam.research import error_norms` provides the same scalar/vector diagnostic
as the Comparison report. Supply already aligned prediction/reference arrays and
explicit compatible fields. It returns finite/None values plus status metadata,
without changing tensors or gradients. See [error norm definitions and examples](use/error-norms.md).

## Sketched Gauss–Newton experiments

`from margam.research import DSGNARPINN` exposes the same independent
[DSGNAR trainer](dsgnar.md#script-use) used by desktop blocks. It accepts an
ordinary pointwise network and fixed PINNProblem, without GUI registration.
Its SVD update/state contract is separate from Trainer custom-step callbacks.

## 3D turbulence, PirateNet and SOAP (`margam.research.jaxpi2`)

`from margam.research import jaxpi2` provides the PyTorch port of
[arXiv:2507.08972v2](https://arxiv.org/abs/2507.08972v2) and `sifanexisted/jaxpi2`:
`PirateNet`, `ModifiedMlp`, `PeriodEmbs`, `FourierEmbs`, `DirectFactorizedLinear`,
`stable_swish`, `Jaxpi2Optimizer` (`SOAP` + `Schedule-Free` + global gradient norm
clipping + ROCm `rocSOLVER` hardening), 3D unsteady `navier_stokes`, per-component
causal gating, pseudo-time stepping, multi-stage linearized perturbation operators,
and `Jaxpi2CropTrainer`. See [3D sphere flow and JAXPI2 turbulence methods](sphere-turbulence.md)
for full tutorials and AMD Instinct MI300X $\text{Re}=300$ sphere-flow benchmarks.

## End-to-end research tutorials

### Tutorial 1: 2D manufactured PDE with mixed Dirichlet and Neumann operators

Solve $-\Delta u = 2\pi^2 \sin(\pi x)\cos(\pi y)$ on $[0, 1]^2$ with Dirichlet
conditions on $x=0$ and $x=1$ and zero normal-flux Neumann conditions on $y=0$
and $y=1$:

```python
import math
import torch
from margam.nn import FullyConnected
from margam.research import PINNProblem, PointConstraint, Trainer, derivative, error_norms

torch.manual_seed(0)

# Collocation and boundary points on [0, 1]^2
gx = torch.linspace(0.0, 1.0, 21)[1:-1]
X, Y = torch.meshgrid(gx, gx, indexing="ij")
interior = torch.column_stack([X.reshape(-1), Y.reshape(-1)])

s = torch.linspace(0.0, 1.0, 21)
left_right = torch.cat([
    torch.column_stack([torch.zeros_like(s), s]),
    torch.column_stack([torch.ones_like(s), s]),
])
bottom_top = torch.cat([
    torch.column_stack([s, torch.zeros_like(s)]),
    torch.column_stack([s, torch.ones_like(s)]),
])

def pde_residual(xy, u):
    u_xx = derivative(u, xy, axis=0, order=2)
    u_yy = derivative(u, xy, axis=1, order=2)
    forcing = 2.0 * (math.pi**2) * torch.sin(math.pi * xy[:, 0:1]) * torch.cos(math.pi * xy[:, 1:2])
    return u_xx + u_yy + forcing

problem = PINNProblem(
    residual=pde_residual,
    collocation_points=interior,
    constraints=[
        PointConstraint("dirichlet_x", left_right, target=0.0),
        PointConstraint(
            "neumann_y",
            bottom_top,
            target=0.0,
            operator=lambda xy, u: derivative(u, xy, axis=1, order=1),
        ),
    ],
)

net = FullyConnected([2, 32, 32, 1], "tanh", "Xavier normal",
                     input_bounds=([0.0, 0.0], [1.0, 1.0]))
trainer = Trainer(problem, net, weights={"dirichlet_x": 10.0, "neumann_y": 10.0}).compile(
    lr=2e-3, device="cpu"
)
trainer.train(600, display_every=200)

# Evaluate against exact u(x, y) = sin(pi x) cos(pi y)
pred = trainer.predict(interior)
exact = torch.sin(math.pi * interior[:, 0:1]) * torch.cos(math.pi * interior[:, 1:2])
print(error_norms(pred, exact, fields=["u"], norm="relative_l2"))
```

### Tutorial 2: Cosine learning-rate schedule with exact checkpoint continuation

Combine `CosineDecay` with `save_checkpoint` and `load_checkpoint` to split a
finite-horizon training budget across two separate stages with identical RNG and
schedule continuation:

```python
from pathlib import Path
import torch
from margam.nn import FullyConnected
from margam.research import CosineDecay, PINNProblem, PointConstraint, Trainer, derivative

def make_scheduled_trainer():
    torch.manual_seed(42)
    x_int = torch.linspace(0.0, 1.0, 66)[1:-1, None]
    x_bc = torch.tensor([[0.0], [1.0]])
    problem = PINNProblem(
        residual=lambda x, u: derivative(u, x, axis=0, order=2) + 2.0,
        collocation_points=x_int,
        constraints=[PointConstraint("bc", x_bc, target=0.0)],
    )
    net = FullyConnected([1, 24, 24, 1], "tanh", "Xavier normal",
                         input_bounds=([0.0], [1.0]))
    schedule = CosineDecay(steps=400, minimum_ratio=0.05, power=1.0)
    return Trainer(
        problem,
        net,
        weights={"bc": 10.0},
        checkpoint_id="tutorial-cosine-poisson-v1",
    ).compile(lr=3e-3, device="cpu", schedule=schedule)

ckpt_path = Path("build/tutorial-cosine-ckpt.pt")
ckpt_path.parent.mkdir(parents=True, exist_ok=True)

# Stage 1: train first 200 updates and save format-2 scheduled checkpoint
stage1 = make_scheduled_trainer()
stage1.train(200, display_every=0)
stage1.save_checkpoint(ckpt_path)

# Stage 2: reconstruct identical trainer, restore checkpoint, finish remaining 200 updates
stage2 = make_scheduled_trainer()
stage2.load_checkpoint(ckpt_path)
stage2.train(200, display_every=0)

print("Completed updates:", stage2.iter)
print("Final learning rate used:", stage2.loss_history[-1]["learning_rate:0"])
```

### Tutorial 3: Joint PDE and physical parameter identification in a custom module

Because `Trainer.compile()` optimizes all parameters registered on `self.network`,
you can wrap a neural network alongside trainable `torch.nn.Parameter` scalars to
identify unknown physical coefficients from noisy or sparse interior observations:

```python
import math
import torch
from margam.nn import FullyConnected
from margam.research import PINNProblem, PointConstraint, Trainer, derivative

class ParametricPoissonNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FullyConnected([1, 32, 32, 1], "tanh", "Xavier normal",
                                       input_bounds=([0.0], [1.0]))
        # Unknown diffusion scaling alpha (true value = 2.5), stored in log-space
        self.log_alpha = torch.nn.Parameter(torch.tensor(0.0))

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    def forward(self, x):
        return self.backbone(x)

torch.manual_seed(7)
model_net = ParametricPoissonNet()

# True solution: u(x) = sin(pi x), satisfying -alpha * u_xx = 2.5 * pi^2 * sin(pi x)
true_alpha = 2.5
x_col = torch.linspace(0.0, 1.0, 66)[1:-1, None]
x_obs = torch.linspace(0.0, 1.0, 21)[:, None]
u_obs = torch.sin(math.pi * x_obs)

def inverse_residual(x, u):
    u_xx = derivative(u, x, axis=0, order=2)
    rhs = true_alpha * (math.pi**2) * torch.sin(math.pi * x)
    return model_net.alpha * u_xx + rhs

problem = PINNProblem(
    residual=inverse_residual,
    collocation_points=x_col,
    constraints=[PointConstraint("observations", x_obs, target=u_obs)],
)

trainer = Trainer(problem, model_net, weights={"observations": 50.0}).compile(
    lr=5e-3, device="cpu"
)
trainer.train(800, display_every=200)
print(f"Identified alpha = {model_net.alpha.item():.4f} (true = {true_alpha:.4f})")
```

## PyTorch compilation

Margam's `Trainer.compile()` prepares training data and optimization; it does not
mean `torch.compile`. A network is already a PyTorch module. PyTorch also offers
[in-place `Module.compile`](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.compile),
which avoids replacing it with a wrapper that changes state-dictionary key names.
That alone does not establish compatibility with a PINN loss.

A 13 November check using Torch 2.14.1+cu130 and a small float64 FCNN found that
compiled CUDA inference matched eager values within tolerance and retained the
same state keys and weights. A native Poisson update failed with compiled backward
buffer/graph restrictions on both CPU `aot_eager` and CUDA Inductor. A separate
one-dimensional tanh network with donated buffers disabled reproduced the explicit
AOTAutograd double-backward error when differentiating its first coordinate
derivative. CPU Inductor was not qualified because the local C++ compiler was
unavailable. These were compatibility probes, not speed or accuracy benchmarks.

Do not compile a general training network as a drop-in change or suppress derivative
errors with `allow_unused=True`/zero derivatives. Generic compilation stays deferred.
The existing [DSGNAR functional benchmark](dsgnar.md) instead compiles complete
case-specific functional derivative blocks and has its own parity/checkpoint
contract. Extending that approach requires concrete derivative and continuation
checks. Compilation also has startup cost and can recompile for changed shapes;
see the [PyTorch compiler API](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html).

The [SCALE cavity benchmark](scale-cavity-benchmark.md) adds another specifically
qualified route: explicit sine/SiLU coordinate-derivative channels make the
complete cavity loss an ordinary first-order parameter-autograd graph. Compiled
losses and parameter gradients are checked against eager/native Scale calculations.
This research case does not enable general compiled-network training in desktop
blocks; its full-budget accuracy attempts and limitations are recorded separately.
