# Profiling DSGNAR batches and workspaces

This opt-in command measures the existing wave and multiscale research cases in
fresh numerical processes. It does not open or alter a GUI project, load a user's
trained run, change dependencies, or select new defaults.

```bash
python -m examples.profile_dsgnar --device cpu --output build/dsgnar-profile/cpu.json
python -m examples.profile_dsgnar --device cuda --output build/dsgnar-profile/cuda.json
```

Defaults are two measured attempts after one warm-up attempt, for each case and
batch size **16, 64, 128, 256**. Every configuration gets a fresh subprocess and a
five-minute worker timeout. Smaller diagnostics can use `--case wave --batches
16 64 --points 128 --boundary 64 --attempts 1`. Invalid settings and failed workers
do not overwrite an existing successful report.

## What is measured

The fixed cases use 2,048 interior and 512 boundary rows, float64, and the existing
SIREN 16×3 wave or Fourier 24×3 multiscale network. The wave also has 512 shared
initial locations for independent displacement and velocity losses. Effective
square sketches are 609 and 700 respectively.

- **Jacobian and sketch:** the entire blocked construction, including residual
  counting, differentiation, randomized transforms, bucket accumulation and checks.
- **Jacobian blocks:** a nested subset of the preceding duration. **Do not add it
  again** when summing phase costs.
- **SVD:** factorization of the sketch.
- **Full objective:** all baseline/probe/selected-candidate condition evaluations.
- **Regularized steps:** trust-region regularization solves.
- **Radius interpolation:** the monotone probe crossing calculation.

CUDA is synchronized at instrument boundaries. Instrumentation costs time and
changes the scheduling of asynchronous work. The report therefore also times an
**uninstrumented reference continuation** from the identical post-warmup
checkpoint. This is useful context, not a claim that two short measurements are a
stable production throughput estimate.

## State and memory checks

Each profiled continuation is compared with its uninstrumented control. Network
weights, residual/loss histories, dedicated sketch RNG and the remainder of the
checkpoint must match exactly. Existing last-bit SVD regularization diagnostics
use relative tolerance `5e-14`; this tolerance does not apply to trained weights
or loss histories. Global CPU/CUDA random generators must remain unchanged by
the measured training. Scoped hooks restore instance methods and numerical
functions on both normal exit and an injected failure.

CUDA reports actual allocator **peak allocated** and **peak reserved** bytes after
warm-up, plus resident allocation before timing. The control network is created
only after these measurements. These numbers exclude untracked driver/OS
allocations and compilation/initialization peaks before warm-up. CPU/host RSS is
the process lifetime high-water mark through the measured work, including imports
and warm-up; it is **not** an update-only allocation delta.

The report also gives the logical dense-Jacobian size and sketch-matrix size as
arithmetic references. They are not additional measured allocations and do not
represent SVD/autograd peak workspace. No available-memory snapshot is treated as
a reservation, quota or promise that a larger experiment fits.

## Interpreting batch comparisons

Larger batches reduce the number of blocked sketch/derivative operations. They
can increase derivative workspace, while an SVD or objective workspace may still
dominate the observed peak. Do not assume memory rises monotonically at every
batch increment.

Changing batch size changes the realized random sketch and floating-point
reduction grouping in this implementation. **Exact continuation is checked within
each batch setting, not across settings.** These measurements are not evidence
that the same optimization trajectory or final accuracy survives a batch change.
A saved checkpoint still requires matching batch settings.

Maintained measurements are [CPU](../assets/benchmarks/dsgnar-profile-cpu.json) and
[CUDA](../assets/benchmarks/dsgnar-profile-cuda.json). Each contains source
fingerprints, actual raw losses, measured phase durations and memory scopes.
They cover small concrete networks and sketches, not the paper's 4,000-square
H100/JAX workload. Source fusion, large-sketch workspaces, adaptive growth and
output/inference costs remain separate work.

## Observed laptop measurements

These short RTX 3060 Laptop measurements use the two existing float64 cases.
The reference column has no phase instrumentation; peaks come from the profiled
continuation. CPU and GPU profiling were separate processes on the same laptop,
not an exclusive-machine benchmark.

| Case | Batch rows | Reference s/attempt | Instrumented s/attempt | Peak allocated MiB |
| --- | ---: | ---: | ---: | ---: |
| wave | 16 | 4.122 | 4.161 | 94.8 |
| wave | 64 | 1.486 | 1.504 | 94.8 |
| wave | 128 | 0.964 | 0.977 | 94.8 |
| wave | 256 | 0.748 | 0.766 | 94.8 |
| multiscale | 16 | 2.921 | 3.385 | 103.4 |
| multiscale | 64 | 1.446 | 1.474 | 103.4 |
| multiscale | 128 | 1.023 | 1.043 | 103.4 |
| multiscale | 256 | 0.858 | 0.877 | 115.7 |

All **16 CPU/CUDA configurations** pass checkpoint parity and global-RNG checks.
At batch 16, Jacobian/sketch construction takes roughly 83–86% of measured GPU
attempt time; at 256, its share drops to roughly 26–29%. The absolute wave peak
stays flat across these batches, while multiscale rises at 256: another workspace
can dominate before derivative batches become the peak.

These results justify further attention to blocked sketch overhead. They do not
justify silently changing batch settings or claiming the paper’s throughput.
Short reference timings can also fluctuate relative to instrumented timings.
