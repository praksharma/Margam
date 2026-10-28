# Recorded numerical results

[Documentation index](README.md) · [Raw evidence index](../assets/benchmarks/README.md)

These are measured runs, with their individual settings and limitations. Test
acceptance thresholds are catalogued separately in [validation](validation.md).
Do not compare errors or timings across different problems, grids, loss scales,
training budgets or software/hardware environments.

| Study | Evidence date / device | What it supports |
| --- | --- | --- |
| [Helmholtz Fourier/LRA](#helmholtz-with-fourier-lra-pinn) | 27 October 2026 / CPU | Three seeds on the paper problem using an existing Fourier network; changed training protocol |
| [LRA interval Poisson](#learning-rate-annealing-pinn) | 27 October 2026 / CPU | Equal-budget three-seed comparison and strong coefficient-cap sensitivity; baseline wins here |
| [Baseline vs SA interval Poisson](#baseline-vs-self-adaptive-pinn) | 27 October 2026 / CPU | Working pointwise adaptation; mixed three-seed comparison |
| [Heat: soft vs hard Dirichlet](#steady-heat-soft-and-hard-dirichlet) | 27 October 2026 / CPU | Lower field error for hard BC enforcement on this supported box problem |
| [Pointwise network compatibility](#pointwise-network-compatibility) | 27 October 2026 / CPU | Ten architectures × baseline/SA on a one-seed smoke problem; not a ranking |
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

## Pointwise network compatibility

[Raw settings/predictions/errors](../assets/benchmarks/pointwise-networks-2026-10-27.json) ·
[Architectures and derivative checks](neural-networks.md)

Unit interval Poisson `u_xx+2=0`, zero endpoints, reference x(1−x), 32 fixed grid
points, 201 including-boundary evaluation points, hidden widths 16,16, seed 7,
700 full-batch updates. CPU float32, one Torch thread. Baseline and SA share the
same initial network for each architecture; SA adapts PDE points only (rate 0.005,
initial λ=1, cap 1000), global PDE/mean-BC weights are 1/10.

Network lr is **0.003**, except SIREN **0.001**. Explicit smoke options:
axis Fourier/highway/modified frequencies [0,0.5,1]; angular Fourier MLP modes 3,
scale 0.1; multiscale Gaussian scales [0.1,0.5], 8 features; SIREN first/hidden
omega 10/10; multiplicative filter input scale 0.5. Other options retain their
validated defaults. These are not all UI defaults and not an equal-capacity or
hyperparameter-search-controlled comparison. Width entries have architecture-specific
meaning, so parameter counts differ. No efficiency ranking is made.

| Architecture | Baseline RMSE | SA RMSE |
| --- | ---: | ---: |
| Fully connected | 0.002182551 | 0.0004081843 |
| Fourier MLP (angular) | 7.359666e-05 | 6.320456e-05 |
| Fourier feature NN | 0.0001424609 | 0.0001722684 |
| Highway Fourier NN | 3.459548e-05 | 5.73967e-05 |
| Modified Fourier NN | 0.0005386058 | 0.000642377 |
| Multiscale Fourier NN | 5.359752e-05 | 6.415103e-05 |
| DGM network | 0.0001696879 | 0.0001521516 |
| SIREN | 0.000439797 | 0.0005324064 |
| Multiplicative Fourier NN | 4.697537e-05 | 5.107183e-05 |
| Multiplicative Gabor NN | 0.0002652292 | 0.0002539854 |

All twenty runs completed with RMSE below 0.003. This is evidence of usable
pointwise derivatives/training under those specific settings. One seed on an easy
problem does not establish architecture superiority, general SA improvement, or
accuracy on stiff/multiscale PDEs. It does not validate neural operators or CUDA.
The earlier SA study above used lr=0.005 and different Fourier-free defaults;
its values should not be substituted for this study's FCNN row.

Raw JSON includes actual evaluation coordinates/reference, predictions, per-run
complete configs, final pre-update loss records, field metrics, adaptive summaries,
environment and selected source hashes. It reports the pre-commit HEAD and modified
working tree honestly; hashes identify the selected files used, not a complete
source archive. Reproduce to a **new** file:

```sh
.venv/bin/python -m examples.benchmark_pointwise_networks --iterations 700 --seed 7 --output benchmark-results/pointwise-networks-new.json
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

## Learning-rate annealing PINN

[Raw settings, source hashes and all balancing events](../assets/benchmarks/lra-poisson-2026-10-27.json) · [Method and safeguards](lra-pinn.md)

Problem u_xx+2=0 on [0,1], endpoints zero, reference x(1−x). All variants use the
same seeded FCNN 16×16, 32 fixed cell-centred points, full-batch Adam rate 0.001,
2,000 updates, static PDE weight 1 and mean BC weight 1 (0.5 per endpoint).
LRA uses supplied v1 raw-gradient denominator, alpha 0.9, interval 10 and floor
1e-12. Only the coefficient cap changes between its two variants. Evaluation:
201 uniform points including endpoints, CPU float32, one Torch thread; Python
3.14.8, Torch 2.14.1+cpu, NumPy 2.5.2. No extra optimisation stage.

| Network seed | Baseline RMSE | LRA cap 100 RMSE | LRA cap 1,000,000 RMSE |
| --- | ---: | ---: | ---: |
| 7 | 0.000299148 | 0.00425180 | 0.176026 |
| 11 | 0.000164153 | 0.000276327 | 0.179469 |
| 19 | 0.000574412 | 0.000742293 | 0.158451 |

**Baseline is more accurate in all three runs.** Very small boundary gradients
produce large coefficients in the high-cap variant and the PDE fit stalls. Cap
100 substantially improves LRA here; the File → Examples case uses that cap.
The cap is a practical guard, not a paper equation or evidence of an optimal
setting for other problems. These are post-update field errors, not the changing
weighted objectives. Full event histories and final raw losses are preserved.

This small fixed-sample problem does not reproduce the paper's stiff benchmark,
minibatch protocol, aggregated-boundary grouping or proposed architecture. It
establishes reproducible implemented behaviour and parameter sensitivity, not a
ranking of PINN methods. The older SA table uses a different rate, BC weight and
budget and must not be compared directly. CUDA remains unvalidated.

## Helmholtz with Fourier LRA-PINN

[Exact settings, source hashes and balancing histories](../assets/benchmarks/helmholtz-fourier-lra-2026-10-27.json) · [Problem and differences from the paper](helmholtz.md)

Equation Δu+u=(1−17π²)sin(πx)sin(4πy), square [−1,1]², zero soft Dirichlet
edges, explicit reference sin(πx)sin(4πy). Existing axis Fourier NN with frequencies
1–4, tanh hidden widths 32×32; 1,024 fixed grid interior points, 64 points per edge.
LRA alpha 0.9, interval 10, floor 1e-12, cap 10,000; static PDE weight 1 and mean
BC weight 1 (0.25 per edge). Full-batch Adam rate 0.001 for 5,000 updates.

Errors after the last update, evaluated on a separate 101×101 grid including
edges. CPU float32, one Torch thread, Python 3.14.8, Torch 2.14.1+cpu. Training
wall time excludes construction/evaluation; other validation ran concurrently,
so these timings are approximate local observations, not performance comparisons.

| Network seed | RMSE | Relative L2 | Maximum absolute error | Dense-grid PDE residual RMS | Maximum boundary error | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.006596973 | 0.01332589 | 0.0193499 | 3.345981 | 0.005976811 | 37.38 |
| 11 | 0.007249426 | 0.01464384 | 0.01861102 | 2.952915 | 0.006694295 | 36.79 |
| 19 | 0.004117785 | 0.008317927 | 0.01758766 | 3.896296 | 0.006933656 | 38.33 |

Mean relative L2 across these three seeds: **1.210%**. This is a small
three-seed demonstration, not a ten-trial estimate or a comparison against other
architectures/methods. Known frequencies informed the feature choice.

Dense-grid residuals are higher than the final training-point residuals; the
locations differ, and the evaluation grid also includes boundary points where
training enforces values rather than PDE residuals. Good field accuracy does not
establish uniformly small PDE residuals. Boundary conditions are soft, so their
errors are measured rather than assumed zero. The raw JSON retains each edge's
maximum error and final pre-update physical losses. Changing weighted objectives
are not used as field-accuracy scores.

The paper's problem is retained; its architecture, stochastic/resampling protocol,
base-rate schedule and 40,000-update training budget are not. These results do
not reproduce or establish superiority to its reported values. CUDA is untested.

## Residual sampler workflow check — 28 October 2026

[CPU evidence](../assets/benchmarks/adaptive-sampling-cpu.json) records two 60-update
rectangle runs starting from 36 grid points. RAR adds 8 points after updates 20 and
40 (52 final points); importance sampling retains 36 points and applies inverse
probability weights. Both reproduce the uninterrupted trajectory exactly after
checkpointing at step 25. These checks establish schedule/continuation behaviour,
not comparative numerical accuracy. See the [sampler guide](adaptive-sampling.md).

## Integral loss workflow — 28 October 2026

[CPU record](../assets/benchmarks/integral-loss-cpu.json): on a 3 × 2 rectangle,
initial scalar PDE/BC losses match mean-MSE losses times the corresponding
area/edge length. Fixed and importance-sampled 40-update runs resume identically
from update 15. This verifies measure application and continuation, not solution
accuracy or convergence superiority. See [integral loss](integral-loss.md).

## Burgers workflow — 28 October 2026

[CPU record](../assets/benchmarks/burgers-cpu.json): the original 1D Burgers setup
with current Margam defaults is exercised for 100 updates. Checkpoint continuation
is identical, predictions/losses are finite, and nine spatial VTI frames carry
physical times from 0 to 1. No later-time exact solution or error metric is used;
this is not a reproduction of paper accuracy. See [Burgers](burgers.md).

## Expert PINN workflow — 28 October 2026

Short [CPU](../assets/benchmarks/expert-cpu.json) and
[CUDA](../assets/benchmarks/expert-cuda.json) Burgers runs exercise causal weights,
RWF, sum-gradient-norm updates every ten steps and Adam decay for 100 updates.
Both retain the exact same-device trajectory after checkpoint continuation. Full
configuration and balancing histories are recorded; no shock-reference error or
paper accuracy is claimed. See [Expert PINN](expert-pinn.md).
