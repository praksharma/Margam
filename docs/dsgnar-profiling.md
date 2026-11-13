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
configurable 30-minute worker timeout by default. Smaller diagnostics can use `--case wave --batches
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
H100/JAX workload. The completed large-SVD/sketch/probe follow-up is documented
below. Adaptive growth and output/inference costs are outside these measurements.

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

The existing dense implementation remains the **production CUDA path below 4,000 rows**. The later full-size qualification below enables occupied-row updates at 4,000 and above. The
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
with the established `5e-14` tolerance only for regularization diagnostics. These historical small-sketch CUDA records
compare two equivalent dense paths, so its short-run timing differences are
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

## Completed 4,000-row SVD and full-attempt profiling

The larger follow-up runs a **real 4,000×4,000 SVD inside complete DSGNAR attempts**.
It uses the existing 60×4 SIREN with 11,221 parameters, fixed frequencies 8/1,
separate wave displacement/velocity conditions, and one complete time interval.
This matches the large sketch dimension, not the paper's full architecture,
time-window schedule, accuracy or H100 hardware protocol.

CUDA uses **32,768 PDE points and 16,384 rows per boundary/initial condition**
(81,920 scalar residuals). CPU uses **4,096/1,024** (7,168 residuals), with the same
actual 4,000-row SVD. CPU/GPU total times therefore are **not** a same-data device
comparison. Each configuration has one warm-up, one measured complete attempt
and an uninstrumented same-checkpoint continuation. CUDA runs are sequential
so another profiling job does not compete for that GPU.

```bash
python -m examples.profile_dsgnar --case wave --device cuda --width 60 --depth 4 --sketch 4000 --batches 64 256 --points 32768 --boundary 16384 --attempts 1 --precision float64 --svd-memory --output build/dsgnar-profile/large-cuda64.json
python -m examples.profile_dsgnar --case wave --device cuda --width 60 --depth 4 --sketch 4000 --batches 256 --points 32768 --boundary 16384 --attempts 1 --precision float32 --svd-memory --output build/dsgnar-profile/large-cuda32.json
python -m examples.profile_dsgnar --case wave --device cpu --width 60 --depth 4 --sketch 4000 --batches 256 --points 4096 --boundary 1024 --attempts 1 --precision float64 --output build/dsgnar-profile/large-cpu64.json
```

`--precision float32` also works for the CPU command. Width, depth, requested and
effective sketch size, precision and point counts are recorded explicitly. A
requested sketch larger than the network is capped and reported, rather than
misrepresented as a larger SVD. Reports refuse publication if source/revision
fingerprints change during profiling, preserving any prior report.

### SVD memory scope

`--svd-memory` records the actual solver device, allocation immediately before
the SVD, the phase's allocator peak, and its growth. **Growth includes SVD outputs
and temporary workspace**, not just scratch storage. The profiler maintains its
own running maximum while resetting phase counters, so capturing an SVD peak
does not erase an earlier whole-attempt peak. These are PyTorch allocator figures;
CUDA context/driver allocations are outside their scope. CPU retains its honest
process-lifetime RSS high-water mark, without inventing a phase-specific peak.

### Large-only GPU dispatch and probe caching

The [full-count float32](../assets/benchmarks/dsgnar-large-touched-cuda32.json)
and [float64](../assets/benchmarks/dsgnar-large-touched-cuda64.json) old/candidate
comparisons gave **1.088× and 1.122× old/new elapsed ratios**, with exact checkpoint
parity. This complete-attempt evidence, rather than the earlier kernel-only
measurements, qualifies the occupied-row CUDA path **only for sketches ≥4,000**.
Smaller CUDA sketches keep the previous dense path. Boundary tests cover 3,999,
4,000 and 4,096 rows. The final
[production comparison](../assets/benchmarks/dsgnar-large-production-cuda64.json)
also checks the actual dispatch against the old accumulator. These short laptop
measurements are not universal hardware speed guarantees.

Probe selection now computes unchanged PCHIP coefficients **once per crossing**,
rather than recomputing them during all 60 bisection evaluations. Arithmetic,
plateau handling, bisection count, selected radii and RNG behavior are unchanged.
The [exact-radius comparison](../assets/benchmarks/dsgnar-probe-cache.json)
shows about **2–9×** interpolation-only ratios across 3–32 probes; endpoint
fast paths need no crossing calculation. Full
[CPU](../assets/benchmarks/dsgnar-probe-cache-cpu.json) and
[GPU](../assets/benchmarks/dsgnar-probe-cache-cuda.json) continuations compare the
cached implementation with the uncached reference and retain exact state parity.
These are not SVD speedups or paper-accuracy improvements.

```bash
python -m examples.compare_dsgnar_probes --output build/dsgnar-profile/probe-cache.json
python -m examples.compare_dsgnar_sketch --component probe --device cuda --batches 256 --output build/dsgnar-profile/probe-paired.json
python -m examples.compare_dsgnar_sketch --case wave --device cuda --width 60 --depth 4 --sketch 4000 --points 32768 --boundary 16384 --batches 256 --attempts 1 --output build/dsgnar-profile/large-paired.json
```

### Final measured full-attempt results

| Backend | Precision | Batch | PDE / condition rows | Attempt s | SVD s | Peak allocated MiB |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| CPU | float32 | 256 | 4,096 / 1,024 | 19.29 | 13.87 | — |
| CPU | float64 | 256 | 4,096 / 1,024 | 24.02 | 18.33 | — |
| CUDA | float32 | 256 | 32,768 / 16,384 | 14.14 | 2.97 | 561.1 |
| CUDA | float64 | 64 | 32,768 / 16,384 | 41.22 | 6.77 | 1050.8 |
| CUDA | float64 | 256 | 32,768 / 16,384 | 21.95 | 6.85 | 1050.8 |

Records: [CPU float32](../assets/benchmarks/dsgnar-large-cpu32.json),
[CPU float64](../assets/benchmarks/dsgnar-large-cpu64.json),
[CUDA float32](../assets/benchmarks/dsgnar-large-cuda32.json), and
[CUDA float64](../assets/benchmarks/dsgnar-large-cuda64.json).

Every row completes an actual attempt and passes same-checkpoint reference
continuation, global-RNG and source-stability checks. These are short timing
samples, not a promise of sustained throughput. The GPU SVD phase adds about
**434 MiB** in float32 and **863 MiB** in float64 above its pre-SVD allocation,
including outputs and temporary workspace. CPU figures use different point
counts and report lifetime RSS instead of an invented allocator peak.

The pre-optimization [CPU64](../assets/benchmarks/dsgnar-large-baseline-cpu64.json),
[CUDA32](../assets/benchmarks/dsgnar-large-baseline-cuda32.json), and
[CUDA64](../assets/benchmarks/dsgnar-large-baseline-cuda64.json) records remain
for context. Cross-process baseline/after timing differences are not as controlled
as the paired comparisons and should not be treated as exact optimization gains.
