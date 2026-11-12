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

Retained **pre-optimization baseline** measurements are [CPU](../assets/benchmarks/dsgnar-profile-cpu.json) and
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

## Occupied-bucket CPU update

The production CPU accumulator now updates only buckets occupied by the current
Jacobian batch. Within each hash it keeps the stable bucket sort and the same
ordered segment sums. Unique destination rows avoid colliding writes. It no
longer materializes a sketch-sized zero-filled reduction for every small batch.
Sampling, signs, normalization, hash order and dedicated RNG draws are unchanged.
No new numerical setting or checkpoint format is introduced.

The existing dense implementation remains the **production CUDA path**. The
[initial GPU occupied-row prototype](../assets/benchmarks/dsgnar-touched-gpu-prototype.json)
was mostly slower on the 609/700-sized sketches where extra indexing/kernel overhead appears to outweigh the saved empty-bucket
work. Large accumulation-kernel gains
alone do not justify regressing the common end-to-end GPU cases.

### Paired full-attempt checks

```bash
python -m examples.compare_dsgnar_sketch --device cpu --output build/dsgnar-profile/paired-cpu.json
python -m examples.compare_dsgnar_sketch --device cuda --output build/dsgnar-profile/paired-cuda.json
```

Each process creates one common dense-path warm-up checkpoint. The old and
production implementations continue independently from that same checkpoint;
branch order alternates across batch sizes. Previous model cycles and allocator
caches are released between branches.

[CPU](../assets/benchmarks/dsgnar-touched-cpu.json) and
[CUDA](../assets/benchmarks/dsgnar-touched-cuda.json) cover 16 paired full
continuations. All trained weights, loss histories, RNG and state match exactly,
with the established `5e-14` tolerance only for regularization diagnostics. CUDA
compares two equivalent dense paths, so its short-run timing differences are
noise, **not** a claimed GPU speedup.

The short CPU wave comparisons show roughly **1.06–1.42×** old/new elapsed ratios.
Multiscale results are mixed, including a slower sample, so the retained two-
attempt timings are not a universal end-to-end speed guarantee. Direct kernel
measurements below separate the targeted reduction from SVD/objective costs.

### Larger accumulation workspace

```bash
python -m examples.profile_countsketch --device cpu --size 700 --batches 16 64 128 256 --repeats 5 --output build/dsgnar-profile/count700-cpu.json
python -m examples.profile_countsketch --device cpu --output build/dsgnar-profile/count4000-cpu.json
python -m examples.profile_countsketch --device cuda --output build/dsgnar-profile/count4000-cuda.json
```

These are **accumulation-only** tests, including a 4,000×4,000 sketch in float32
and float64. Half of each hash's rows deliberately collide at bucket zero; the
remaining assignments are uniform. All output matrix/vector values match the
old reduction exactly. Matrix zeroing, SRCT, Jacobian construction and SVD are
excluded. Both output matrices remain resident during peak measurements.

The [700-row CPU kernel record](../assets/benchmarks/dsgnar-countsketch700-cpu.json)
shows about **2.2–7.7×** float64 kernel ratios across tested batches. The
[4,000-row CPU](../assets/benchmarks/dsgnar-countsketch4000-cpu.json) and
[experimental CUDA](../assets/benchmarks/dsgnar-countsketch4000-cuda.json) records
show why empty buckets can be costly at larger sizes. **These are not whole-
trainer speedups or 4,000-row SVD measurements.** The CUDA microbenchmark calls
the experimental helper explicitly; it does not select production GPU behavior.

To reproduce the rejected whole-attempt GPU experiment, add
`--experimental-touched` to the paired comparison command. This flag belongs to
the standalone benchmark, not to saved model settings or the GUI.
