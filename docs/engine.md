# Numerical API and benchmarking

[Documentation index](README.md) · [Method/recipe ownership](adding-a-pinn.md) ·
[Measured results](results.md) · [Validation](validation.md)

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
