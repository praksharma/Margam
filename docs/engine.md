# Numerical API and benchmarking

[Documentation index](README.md) · [Method/recipe ownership](adding-a-pinn.md) ·
[Measured results](results.md) · [Validation](validation.md)

For custom networks, sampling, losses and optimisers in an external research
script, start with the [research API](research.md).

The method implementation is `margam/methods/pinn.py`. Shared problem definitions
live in `margam/components/problem.py`, with named weighted losses alongside them.
SA-PINN is a separate implementation in `margam/methods/self_adaptive.py`;
hard Dirichlet is in `margam/methods/hard_dirichlet.py`. UI recipes are separate
construction definitions under `margam/recipes/`.
Only strong-form residuals are in scope; weak/variational methods are not implemented.

```python
import torch
import margam as mg

problem = mg.PINNProblem(
    residual=lambda x, u: mg.utils.laplacian(u, x) + 2,
    collocation_points=lambda: torch.linspace(0, 1, 34)[1:-1, None],
    constraints=[mg.PointConstraint("boundary", torch.tensor([[0.], [1.]]))],
)
network = mg.nn.FullyConnected([1, 16, 16, 1], "tanh", "Xavier normal",
                               input_bounds=([0.], [1.]))
method = mg.PINN(problem, network).compile(lr=0.01, device="cpu")
method.train(1000, display_every=250)
values = method.predict(torch.linspace(0, 1, 101)[:, None])
```

Sampling happens once per compile. Point sources are tensors or zero-argument
functions returning non-empty `(N, D)` tensors. Residuals and constraint operators
return `(N, C)` or `(N,)`. Targets may be scalar, matching tensors, or callables of
points. Constraints can supply derivative operators. Each named term has its own
mean loss. Fixed term weights are summed for full-batch Adam training. LBFGS needs
a separate closure-based procedure and is explicitly rejected by this recipe.

`train(n)` runs exactly n additional updates. Compile resets optimiser and histories.
Loss history records unweighted named terms and the weighted total before each update.
Prediction restores the network's prior training/evaluation mode. FullyConnected
uses raw coordinates or fixed bounds; it does not normalise by batch extrema.
`device="auto"` selects CUDA when available, otherwise CPU.

`margam.benchmarking` exposes BenchmarkCase, Recipe, run_benchmark and solution_errors.
A case supplies a problem factory, evaluation points and reference values. Each
recipe builds its own compiled method. Per seed, the runner materialises data once
and gives independent copies to recipes. It records settings, Git state, sample
fingerprints, versions, timing, losses, predictions and errors. Recipe config should
include architecture and hyperparameters. Output directories must be new.

Artifacts include manifest.json, per-seed samples, per-recipe result.json,
loss_history.csv and predictions.npz, plus summary and aggregate CSV/JSON files.
Relative L2 is null for zero reference norm. Aggregates include success/failure
counts and sample standard deviations; one seed has no sample standard deviation.
Failures are recorded and re-raised. This analytical Poisson example is a baseline
correctness exercise, not evidence of broad PDE coverage or hardware equivalence.

The small [SA comparison](self-adaptive-pinn.md) has a separate runnable script,
`examples.compare_self_adaptive`, with optional provenance/settings JSON output.
It uses the template builder and existing mean-BC convention; it does not share
the historical baseline benchmark's exact collocation grid, learning rate or BC
weight. Compare results only under the [stated protocols](results.md).

## Integral squared-error loss

`PINN(problem, network, loss_formulation="integral")` uses `PINNProblem.measure`
and each `PointConstraint.measure` (positive finite scalars, default one). It sums
squared residual components, averages over points, and multiplies by the region
measure. Template builders set these measures from physical box dimensions.
The default `loss_formulation="mean"` preserves previous MSE behaviour. See
[integral loss](integral-loss.md) for importance correction and supported data.

## Differential operators tutorial (`margam.utils`)

`margam.utils` provides automatic-differentiation operators that preserve the
solution's dtype, device and higher-order derivative graphs (`create_graph=True`,
`retain_graph=True`). Unused coordinate inputs return zero tensors of matching
shape and dtype instead of `None`.

```python
import torch
import margam as mg

# Separate coordinate tensors allow partial/mixed derivatives per variable
x = torch.tensor([[1.0], [2.0]], dtype=torch.float64, requires_grad=True)
y = torch.tensor([[3.0], [1.0]], dtype=torch.float64, requires_grad=True)
xy = torch.tensor([[1.0, 3.0], [2.0, 1.0]], dtype=torch.float64, requires_grad=True)

# 1. Scalar gradient, Laplacian, directional derivative and mixed partials
u = x**3 * y**2                              # scalar output (N, 1)
grad_u = mg.utils.grad(u, x, y)              # (N, 2): [du/dx, du/dy]
lap_u = mg.utils.laplacian(u, x, y)          # (N, 1): d2u/dx2 + d2u/dy2
du_dxdy = mg.utils.partial(u, x, y)          # (N, 1): d2u/(dx dy)

normals = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
dn_u = mg.utils.normal_derivative(u, normals, x, y)  # (N, 1): grad(u) . n

# 2. Vector Jacobian, divergence, symmetric gradient and convective derivative
v = torch.cat([xy[:, 0:1]**2 + xy[:, 1:2], xy[:, 0:1] * xy[:, 1:2]**2], dim=1)
J_v = mg.utils.jac(v, xy)                    # (N, 2, 2): J[b, i, j] = dv_i / dxy_j
div_v = mg.utils.div(v, xy)                  # (N, 1): dv_0/dx + dv_1/dy
eps_v = mg.utils.sym_grad(v, xy)             # (N, 2, 2): 0.5 * (J + J^T)
conv_v = mg.utils.convective(v, v, xy)       # (N, 2): (v . nabla) v

# 3. Row-wise tensor divergence: div(sigma)_i = sum_j d(sigma_ij)/dx_j
div_eps = mg.utils.matrix_div(eps_v, xy)     # (N, 2)

# 4. 3D curl / rotation of a 3-component vector field
xyz = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64, requires_grad=True)
field_3d = torch.cat([xyz[:, 1:2] * xyz[:, 2:3],
                      xyz[:, 0:1] * xyz[:, 2:3],
                      xyz[:, 0:1] * xyz[:, 1:2]], dim=1)
curl_3d = mg.utils.rot(field_3d, xyz)        # (N, 3)
```

For single-axis $n$-th order derivatives on a combined coordinate tensor `xy`,
you can also use `from margam.components.differential import derivative`:

```python
from margam.components.differential import derivative

u_xx = derivative(u_xy, xy, axis=0, order=2)  # d2u/dx2 from combined (N, D) input
u_yy = derivative(u_xy, xy, axis=1, order=2)  # d2u/dy2
```

## Checkpointing and cross-device continuation

Every compiled `PINN` trainer exports a pure-data dictionary via
`training_state()` and restores it via `load_training_state(state)`. Checkpoints
are loaded with `weights_only=True` and verify network architecture, optimizer
hyperparameters, loss formulation, regional measures and a SHA-256 fingerprint of
the compiled training points and targets:

```python
import torch
import margam as mg

def make_model(device="cpu"):
    problem = mg.PINNProblem(
        residual=lambda x, u: mg.utils.laplacian(u, x) + 2.0,
        collocation_points=torch.linspace(0.0, 1.0, 34)[1:-1, None],
        constraints=[mg.PointConstraint("bc", torch.tensor([[0.0], [1.0]]))],
    )
    net = mg.nn.FullyConnected([1, 16, 16, 1], "tanh", "Xavier normal",
                               input_bounds=([0.0], [1.0]))
    return mg.PINN(problem, net).compile(lr=1e-2, device=device)

# Train 250 steps and save a weights-only checkpoint
model = make_model(device="cpu")
model.train(250, display_every=0)
torch.save(model.training_state(), "build/poisson-checkpoint.pt")

# Restore into a freshly compiled instance (same or different device) and continue
restored = make_model(device="cpu")
state = torch.load("build/poisson-checkpoint.pt", map_location="cpu", weights_only=True)
restored.load_training_state(state)
restored.train(250, display_every=0)
assert restored.iter == 500
```

## Reproducible multi-recipe benchmark runner (`margam.benchmarking`)

Use `BenchmarkCase`, `Recipe` and `run_benchmark` to compare multiple trainers or
architectures across a list of seeds with identical per-seed collocation samples:

```python
import math
import torch
import margam as mg
from margam.benchmarking import BenchmarkCase, Recipe, run_benchmark

eval_x = torch.linspace(0.0, 1.0, 101)[:, None]

case = BenchmarkCase(
    name="poisson-1d-sin",
    build_problem=lambda: mg.PINNProblem(
        residual=lambda x, u: mg.utils.laplacian(u, x) + (math.pi**2) * torch.sin(math.pi * x),
        collocation_points=lambda: torch.rand(64, 1),
        constraints=[mg.PointConstraint("bc", torch.tensor([[0.0], [1.0]]), 0.0)],
    ),
    evaluation_points=eval_x,
    reference=lambda x: torch.sin(math.pi * x),
    config={"pde": "u_xx + pi^2 sin(pi x) = 0"},
)

recipes = [
    Recipe(
        name="baseline-tanh",
        build=lambda problem: mg.PINN(
            problem,
            mg.nn.FullyConnected([1, 32, 32, 1], "tanh", "Xavier normal",
                                 input_bounds=([0.0], [1.0])),
            weights={"bc": 10.0},
        ).compile(lr=1e-3, device="cpu"),
        config={"network": "fcnn-32x2", "trainer": "baseline", "lr": 1e-3},
    ),
    Recipe(
        name="hard-dirichlet",
        build=lambda problem: mg.methods.HardDirichletPINN(
            problem,
            mg.nn.FullyConnected([1, 32, 32, 1], "tanh", "Xavier normal",
                                 input_bounds=([0.0], [1.0])),
            bounds=([0.0], [1.0]),
            value=0.0,
        ).compile(lr=1e-3, device="cpu"),
        config={"network": "fcnn-32x2", "trainer": "hard_dirichlet", "lr": 1e-3},
    ),
]

results = run_benchmark(
    case,
    recipes,
    seeds=[0, 1],
    iterations=300,
    output_dir="build/bench-poisson-1d",
)
for row in results:
    print(row["recipe"], "seed=", row["seed"], "rel_l2=", f"{row['relative_l2']:.4e}")
```

## Execute native saved studies

The [headless study CLI](use/study-cli.md) consumes the portable native graph/grid
format. Shared `DocumentGraph`/`prepare_graph` validation and the same isolated
numerical child preserve desktop semantics. Serial grid resume verifies and keeps
completed trials, restarting unfinished numerical attempts fresh. It shares the
existing training implementation and does not serialize custom research callbacks.
Individual outputs/checkpoints use existing native run archives.
