# Recorded numerical results

[Documentation index](README.md) · [Raw evidence index](../assets/benchmarks/README.md)

These are measured runs, with their individual settings and limitations. Test
acceptance thresholds are catalogued separately in [validation](validation.md).
Do not compare errors or timings across different problems, grids, loss scales,
training budgets or software/hardware environments.

| Study | Evidence date / device | What it supports |
| --- | --- | --- |
| [Baseline vs SA interval Poisson](#baseline-vs-self-adaptive-pinn) | 27 October 2026 / CPU | Working pointwise adaptation; mixed three-seed comparison |
| [Heat: soft vs hard Dirichlet](#steady-heat-soft-and-hard-dirichlet) | 27 October 2026 / CPU | Lower field error for hard BC enforcement on this supported box problem |
| [Historical baseline Poisson](#historical-baseline-poisson-on-cuda) | 23 October 2026 / CUDA, retired runtime | Preserved older measurement; not current CUDA validation |

## Metric definitions

For M scalar evaluation values, error `e_i = prediction_i - reference_i`:

- RMSE = `sqrt(sum(e_i²)/M)`.
- Relative L2 = `sqrt(sum(e_i²)/sum(reference_i²))`; undefined for zero reference norm.
- Maximum absolute error = `max(abs(e_i))`.
- Reported sample standard deviation across seeds uses denominator `n-1`.

These field errors are evaluated after training. Loss records describe pre-update
collocation/constraint errors and may include global/adaptive weights. In particular,
SA's masked total is a changing objective, not a directly comparable accuracy score.
For coupled fields, use per-field errors rather than combining differently scaled
velocity and pressure values without explanation.

## Baseline vs self-adaptive PINN

[Raw JSON](../assets/benchmarks/self-adaptive-poisson-2026-10-27.json) ·
[Method, settings and limitations](self-adaptive-pinn.md)

**Problem:** `u_xx + 2 = 0`, x in [0,1], u(0)=u(1)=0. Exact reference: x(1−x).
32 fixed cell-centred interior grid points; endpoints as separate BC sets.
Tanh 16×16 hidden network with Xavier normal initialisation and fixed coordinate
normalisation. Both methods start from the same network for each seed and train
700 full-batch Adam updates at network lr=0.005. PDE weight 1; mean BC weight 10.
Evaluation: 201 uniform including-boundary points, CPU float32, one Torch thread.
Environment: Python 3.14.8, Torch 2.14.1+cpu, NumPy 2.5.2.

SA uses λ=1 initially, squared mask λ², ascent Adam lr=0.005, cap λ=1000, and PDE
adaptation only. No L-BFGS stage, resampling or additional optimisation budget.

| Network seed | Baseline RMSE | SA RMSE | Baseline relative L2 | SA relative L2 |
| --- | ---: | ---: | ---: | ---: |
| 7 | 0.00923862 | 0.000361239 | 0.0507283 | 0.00198353 |
| 11 | 0.000139327 | 0.000150259 | 0.000765033 | 0.000825056 |
| 19 | 0.000180799 | 0.000222133 | 0.000992749 | 0.00121971 |

| Across these three seeds | Baseline | SA-PINN |
| --- | ---: | ---: |
| Mean RMSE | 0.00318625 | 0.000244543 |
| Sample standard deviation of RMSE | 0.00524155 | 0.000107260 |
| Median RMSE | 0.000180799 | 0.000222133 |

SA substantially improves seed 7, while seeds 11 and 19 are slightly worse.
The mean improvement is dominated by seed 7; the median is worse. Three seeds on
this easy equation do not establish general superiority or statistical significance.

| Seed | Final λ minimum | Final λ mean | Final λ maximum |
| --- | ---: | ---: | ---: |
| 7 | 1.19377 | 1.26778 | 1.49433 |
| 11 | 1.19603 | 1.25758 | 1.45512 |
| 19 | 1.16187 | 1.22277 | 1.42879 |

Weights developed different values across points and no cap was reached. Raw JSON
also preserves maximum field error and final pre-update loss terms. These results
were repeated during the documentation update and match the earlier recorded
comparison. This is not a reproduction of the paper's stiff benchmark protocols.

Reproduce to a **new** evidence file:

```sh
.venv/bin/python -m examples.compare_self_adaptive --iterations 700 --seeds 7 11 19 --output benchmark-results/sa-poisson-reproduction.json
```

The report records Git HEAD `fe19aed…` and tracked changes because the reporting
extension was being added when it ran. Selected source hashes identify the exact
files used, including that script. This is not a claim of running a clean release
or a complete source snapshot. Seeds override the stored base config's network seed;
method names override its baseline method. Sampling settings stay fixed.

## Steady heat: soft and hard Dirichlet

[Detailed nine-run table](heat-accuracy.md) ·
[Raw JSON](../assets/benchmarks/heat-accuracy.json) · [Problem/reference](heat-demo.md)

**Problem:** Δu+2=0 on [0,1]², u=0 on all edges. 144 interior grid points, 24 per
side, tanh 32×32 network, network lr=0.005, 2500 Adam updates per run. Three network
seeds: 7, 17, 27. PDE weight 1. Evaluation: 41×41 points against the explicitly
truncated reference with 40 odd sine modes per axis. CPU float32, Torch 2.14.1+cpu.

| Method / mean BC weight | RMSE range over 3 seeds | Max absolute error range |
| --- | ---: | ---: |
| Baseline / 10 | 0.0185791–0.0204918 | 0.0796043–0.0821636 |
| Baseline / 100 | 0.0139774–0.0187275 | 0.0599912–0.0938645 |
| Hard Dirichlet / 1 | 0.000898833–0.000932801 | 0.00212636–0.00246223 |

Hard enforcement improves RMSE and maximum error for every paired seed in this
study and exactly enforces the supported constant boundary values. Increasing the
soft BC penalty alone is not uniformly beneficial: seed 17's maximum error worsens.
The reference is truncated, not an exact finite solution or rigorous error bound.
This result does not apply to unsupported mixed/spatial/imported hard BCs.

The existing record is preserved, not rerun during this documentation update. It
contains its timestamp, Torch version, grid, reference label, per-run settings,
errors and losses. It does not contain full runtime/source hashes; the artifact
was introduced in commit `0e37a1e`. That commit is provenance of the stored file,
not an inferred exact execution revision.

Reproduce to a fresh filename (the heat script itself allows overwriting):

```sh
.venv/bin/python -m examples.benchmark_heat_accuracy --output benchmark-results/heat-reproduction.json
```

## Historical baseline Poisson on CUDA

[Preserved manifest, records and aggregate](../assets/benchmarks/baseline-poisson-cuda-2026-10-23.json)

This archived experiment predates the current Python 3.14 environment. It reports
Python 3.9.25, Torch 2.0.1+cu118, NumPy 1.26.4, CUDA runtime 11.8 and an NVIDIA
GeForce RTX 3060 Laptop GPU, one CPU thread. Its manifest reports clean source
revision `b420c5565d0982f7dc1b8156e05cbc4412d19557`, created 23 October 2026.

**Problem:** the same interval Poisson equation/reference, but a different protocol:
32 interior `linspace(0,1,34)[1:-1]` points, a single two-endpoint BC constraint,
101 evaluation points, tanh 16×16 network, lr=0.01, PDE/BC weights 1/1,
1000 Adam updates. Seeds: 7,17,27. It is not the SA comparison above.

| Seed | RMSE | Relative L2 | Max absolute error | Training seconds (archived) |
| --- | ---: | ---: | ---: | ---: |
| 7 | 0.00209995 | 0.0115592 | 0.00267881 | 4.54272 |
| 17 | 0.000853738 | 0.00469944 | 0.000882775 | 3.83487 |
| 27 | 0.000105648 | 0.000581544 | 0.000207141 | 3.83324 |

The original manifest, summaries and aggregate values were copied unchanged into
the curated JSON envelope; their source-file hashes are included. Original full
runs remain locally under ignored `benchmark-results/baseline-poisson-b420c55/`.
The curated record does not include original point/prediction arrays or checkpoints.
It was not rerun or independently reproduced during this update.

These are historical measurements, not evidence that today's build was tested on
CUDA, nor a CPU/GPU speed comparison. Current CUDA tests still skip. The current
baseline script can generate a new run on a compatible device, but will not recreate
the retired software environment automatically:

```sh
.venv/bin/python -m examples.benchmark_poisson --device cuda --seeds 7 17 27 --iterations 1000
```

## What is not established

No current-build CUDA equivalence, cross-version optimiser equivalence, general
SA advantage, stiff-paper reproduction, mesh-convergence guarantee or accuracy on
arbitrary imported domains follows from these records. Couette and manufactured
PDEs currently have regression checks, not a committed multi-seed measured study.
Use [validation](validation.md) for those exact thresholds and [TODO](todo.md) for
remaining work.
