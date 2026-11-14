# Pointwise research microbatching

Use `pointwise_microbatch` to bound the derivative graphs built during a fixed-data
research update. Every update still uses **all PDE and constraint rows**, adds
their gradients, and takes **one optimizer step**. The application never enables
this after a memory failure or changes the chunk size automatically.

```python
from margam.research import Trainer, CosineDecay

trainer = Trainer(
    problem, network, checkpoint_id="my-pointwise-experiment-v1",
).compile(
    device="auto", lr=0.001,
    pointwise_microbatch=256,
    schedule=CosineDecay(steps=10000, minimum_ratio=0.1),
)
trainer.train(10000, display_every=100)
```

`None` retains the ordinary full-batch research path. A positive integer sets
the maximum rows per network forward in a training chunk. The compiled size is
read-only: use a new `compile()` to start an experiment with a different size.
The default numerical update and GUI trainers are unchanged.

## What you declare when opting in

Your **network, PDE residual and constraint operators must be deterministic,
pointwise functions**, with each row independent of other rows. Callable point
sources still run once during compilation. Scalar or row-matching tensor targets
are compiled as usual; full target lengths are checked before slicing.

Supported in this first contract:

- The standard `margam.research.Trainer`, built-in mean MSE and fixed point sets.
- Scalar or vector residuals/outputs and derivative-based point constraints.
- Independent PDE/constraint row counts, uneven last chunks and fixed term weights.
- Adam, AdamW or SGD, including momentum and optional StepDecay/CosineDecay.

Custom losses, custom update callbacks/subclasses, adaptive sampling, integral
squared-error mode and global `IntegralConstraint` reductions are rejected.
BatchNorm, nonzero Dropout and RReLU layers are also rejected. Custom code using
cross-row reductions, hidden random operations or mutable state is outside this
contract even if it is not recognizable as one of those module types. Keep all
learned parameters registered on the network and version `checkpoint_id` when
your callable physics or network semantics change.

The option affects training updates. Direct `objective()`/`compute_losses()` and
ordinary research prediction still use their existing full-data behavior.
All compiled coordinates/targets and optimizer state remain resident on their
chosen device. Microbatching reduces graph workspace; it does not bound total
host/device memory or guarantee that a problem fits.

## Correct loss normalization

For each physical term with `N` rows, a chunk containing `n` rows contributes:

```text
chunk_contribution = fixed_term_weight * (n / N) * MSE(chunk)
```

Each term uses **its own** `N`. Boundary sets with five and three points therefore
keep their separate denominators, rather than being normalized by PDE rows or by
the number of chunks. MSE averages over the residual/output components just as
in the full-data path. Raw and weighted loss histories aggregate those pre-update
values. Derivative graphs are released after each backward contribution.

Only after all chunks and finite-gradient checks succeed does the optimizer step.
The scheduler and successful-update counter advance once. A failing loss chunk
or non-finite gradient clears partial gradients without committing an optimizer
step, schedule change or loss record. Adaptive hooks are rejected before they can
change the active samples.

Floating-point summation and matrix-operation shapes differ between full and
chunked updates. Tests establish mathematical agreement with explicit tolerances;
full and chunked trajectories are not required to be bitwise identical.

## Checkpoint continuation

Microbatched research checkpoints use **format 3** and record the exact chunk
size and deterministic pointwise-mean contract, alongside the existing data,
network, optimizer, schedule, history and Torch RNG identity. Rebuild with the
same chunk size before loading. Changing the size or switching to full batches
rejects standard continuation because it changes reduction order.

Same-device split continuation is exact in the qualified float32/float64
CPU/CUDA cases. Existing
full-batch research format 1 (unscheduled) and format 2 (scheduled) identities
remain unchanged. Research checkpoints remain separate from GUI run archives.

## Recorded workspace experiment

[Raw reports and source hashes](../assets/benchmarks/research-microbatch-2026-11-14.json)
record separate fresh processes on 14 November 2026. Each solves the same small
Poisson setup: 4,096 fixed PDE points, two endpoint BC points, FP64, a 32×32 Tanh
network (1,153 parameters), Adam and six total updates. All four runs reproduce
split continuation exactly. Their short budgets establish API behavior, not
solution accuracy.

| Device / mode | First complete update | Peak CUDA allocated | Peak CUDA reserved |
| --- | ---: | ---: | ---: |
| CPU / full | 0.359 s | Not measured | Not measured |
| CPU / 256 rows | 0.222 s | Not measured | Not measured |
| RTX 3060 Laptop / full | 0.212 s | 36.35 MiB | 42 MiB |
| RTX 3060 Laptop / 256 rows | 0.272 s | 17.57 MiB | 22 MiB |

The measured CUDA tensor-allocation peak fell **51.7%** in this configuration;
its first GPU update took longer. Timings describe a single cold update including
Adam-state initialization and lazy backend work. They are not repeated
steady-state throughput measurements or a general CPU/GPU ranking. Allocator
peaks cover PyTorch allocations/reservations, not total driver/system memory.
Construction, checkpoint copies, later updates and prediction costs are outside
that measurement window. CPU memory peaks are not measured.

Run each mode in its own process and a fresh folder:

```sh
.venv/bin/python -m examples.research_microbatch --device cuda --output build/microbatch-full
.venv/bin/python -m examples.research_microbatch --device cuda --microbatch 256 --output build/microbatch-256
```

Use `--device cpu` for the CPU workflow. `--points` and `--steps` set the data and
update budgets. The script rejects existing output folders and writes split/final
research checkpoints plus `result.json`. The [validation record](validation.md#pointwise-research-microbatching-14-november-2026)
covers gradient agreement, failure behavior, continuation and CLI artifacts.
