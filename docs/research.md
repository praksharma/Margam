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
| Boundary/initial/data points | `PointConstraint(name, points, target, operator)` |
| Derivative BC or custom operator | `operator(points, prediction)` on a point constraint |
| Optimiser | `compile(optimizer=factory)`; factory accepts `parameters, lr=...` |
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
mutable closures, schedulers and arbitrary custom algorithm state are not captured.
A custom update hook, stateful loss or custom optimiser class must implement its
own complete state contract; the standard save operation rejects these cases.
Research checkpoints are separate from desktop saved models and run archives.

## Run the supplied external-style example

```sh
.venv/bin/python -m examples.research_custom --device cpu
.venv/bin/python -m examples.research_custom --device cuda --output build/research-cuda
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
