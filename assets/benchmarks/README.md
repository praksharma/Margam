# Curated numerical evidence

Small, versioned measurement records supporting [the results catalogue](../../docs/results.md).
Full generated experiments belong in ignored `benchmark-results/`. Do not overwrite
these files to refresh a study: create a dated record and update the catalogue.

| Record | Status | Payload |
| --- | --- | --- |
| [self-adaptive-poisson-2026-10-27.json](self-adaptive-poisson-2026-10-27.json) | Repeated during documentation update, CPU | Six paired baseline/SA runs; complete base config, per-seed overrides, environment, selected source hashes, RMSE/relative L2/max error, final losses and λ summaries |
| [pointwise-networks-2026-10-27.json](pointwise-networks-2026-10-27.json) | CPU compatibility study, one seed | Ten pointwise architectures with baseline/SA; per-run configs, predictions, metrics, source hashes and environment; not an architecture ranking |
| [heat-accuracy.json](heat-accuracy.json) | Original 27 October CPU study preserved | Nine soft/hard BC runs with errors and losses; timestamp/Torch/grid/reference metadata; no full source/environment snapshot |
| [baseline-poisson-cuda-2026-10-23.json](baseline-poisson-cuda-2026-10-23.json) | Historical GPU record preserved, not rerun | Original manifest, three records and aggregate copied unchanged from a local archived run; source JSON hashes and preservation note |

Values are measurements under their stated settings, not expected constants or test
thresholds. Floating-point results can differ with software and device. Histories
are pre-update losses; field errors are after the final update. The SA objective
changes with λ and cannot be compared as an unweighted baseline loss.

The SA report's `tracked_changes=true` records the reporting-script edit present
at execution; its selected source hashes match the files preserved at documentation commit `4077a63`.
Later architecture changes can legitimately change current source hashes; do not
rewrite the older evidence to make them match.
Its base config is overridden by each record's method and network seed. Historical
GPU payload fields retain their original names and values. No new CUDA validation
is implied by preserving them.

The network compatibility record preserves actual scalar predictions/reference
values and selected source hashes for the network addition. Its explicit per-run
settings (including lower-frequency choices and SIREN rate) differ from UI defaults
and parameter counts differ, so it is a compatibility study rather than a ranking.

## LRA interval Poisson, 27 October 2026

[lra-poisson-2026-10-27.json](lra-poisson-2026-10-27.json) records three network
seeds × baseline/LRA cap100/LRA cap1e6, equal data/architecture/rate/budget,
post-training field errors, pre-update losses and complete balancing events.
Reproduce with `.venv/bin/python -m examples.compare_lra --iterations 2000`
(use `--output` with a new filename). See [interpretation](../../docs/results.md#learning-rate-annealing-pinn).
The supplied v1 algorithm and Margam's guards/grouping are distinguished in the
[method guide](../../docs/lra-pinn.md). Source hashes identify this working-tree
implementation even though the recorded parent commit predates this change.

## Helmholtz with existing Fourier/LRA, 27 October 2026

[helmholtz-fourier-lra-2026-10-27.json](helmholtz-fourier-lra-2026-10-27.json)
records three network seeds, 5,000 updates each, full configuration, selected source
hashes, field/residual errors and all LRA balancing events. The problem comes from
the supplied LRA paper; the Fourier architecture, fixed samples and training
protocol are Margam choices. Reproduce with `.venv/bin/python -m examples.run_helmholtz`
and use a fresh filename with `--output` when saving. See [problem guide](../../docs/helmholtz.md)
and [results](../../docs/results.md#helmholtz-with-fourier-lra-pinn).

- [Adaptive sampling CPU workflow](adaptive-sampling-cpu.json): 60-update rectangle
  runs for RAR and importance sampling, with event counts and exact checkpoint
  continuation checks. This is software correctness evidence, not an accuracy study.

- [Integral loss CPU workflow](integral-loss-cpu.json): non-unit-domain measure
  scaling and exact fixed/importance checkpoint continuation; no accuracy claim.

- [Burgers CPU workflow](burgers-cpu.json): original physical setup, short training,
  exact continuation and physical-time export checks; no accuracy reproduction.

- Expert PINN [CPU](expert-cpu.json) / [CUDA](expert-cuda.json): short independent
  Expert’s Guide v1 pipeline checks, with balancing histories and exact continuation.

- Research StepDecay [CPU](research-step-decay-cpu.json) /
  [CUDA](research-step-decay-cuda.json): eight-update custom-network AdamW runs,
  halving rates every two updates and checking exact split continuation. API
  lifecycle evidence, not a numerical accuracy benchmark.

- [CUDA regression audit — 12 November 2026](cuda-validation-2026-11-12.json):
  21 actual GPU-focused numerical/desktop/data cases pass without skips. Includes
  runtime and source fingerprints; a targeted audit, not a full-suite or
  paper-accuracy reproduction. Run `examples.validate_cuda` to reproduce.

- Research integral constraint [CPU](integral-constraint-cpu.json) /
  [CUDA](integral-constraint-cuda.json): explicit Simpson cross-section quadrature
  with a scalar parabolic axial-velocity ansatz and exact scheduled continuation.
  Numerical/API lifecycle evidence, not full-flow or accuracy reproduction.

- Isolated desktop workers [CPU](isolated-worker-cpu.json) /
  [CUDA](isolated-worker-cuda.json): separate headless processes, CPU retained
  state, unchanged weights during re-evaluation, archive reopening and exact
  continuation. Lifecycle evidence, not accuracy or capacity benchmarks.

- DSGNAR saved-project workflows [CPU](dsgnar-cpu.json) /
  [CUDA](dsgnar-cuda.json): fixed scalar Poisson/Burgers, CPU-retained states,
  model/loss/RNG continuation, references and explicit export. SVD regularization
  diagnostics allow only documented last-bit roundoff.
- DSGNAR [Poisson accuracy](dsgnar-poisson-accuracy.json) and
  [corrected periodic Burgers run](dsgnar-burgers-forward-probes.json) record
  actual errors, configurations, stopping and runtime. The Poisson case is not a
  paper benchmark; Burgers uses an independent reference grid. Read `target_met`
  and the scope before interpreting the result.
- [Compiled Burgers derivative validation](dsgnar-functional-validation.json)
  compares actual 1,447-parameter CUDA Jacobians/residuals with generic autograd.

- [Burgers qualifying checkpoint](dsgnar-burgers-target-checkpoint.json), fresh
  [GPU](dsgnar-burgers-reloaded-cuda.json) / [CPU](dsgnar-burgers-reloaded-cpu.json)
  reloads: about `2.56e-7` float32 relative L2 on the independent evaluation grid,
  below the paper's reported `4.75e-7` threshold. This does not reproduce its
  reference-grid protocol or 9.8-second H100 runtime.

- `dsgnar-cavity-ui-cpu.json`: actual six-attempt smooth-lid cavity UI save/open,
  exact fresh-process continuation and exports. Not accuracy evidence.
- `dsgnar-cavity-jacobian-cuda.json`: all five full 60×4 SIREN condition Jacobians
  checked against generic autograd on CUDA (11,347 actual parameters).
- `dsgnar-cavity-pilot.json`: independently implemented five-condition Re=100
  cavity training with a smaller 24×3 SIREN and sketch 700. Retains measured
  losses/history; reference-based evaluation is recorded separately.
- `dsgnar-cavity-reloaded-cpu129.json` and `dsgnar-cavity-reloaded-{cpu,cuda}257.json`:
  unchanged final-checkpoint inference against independent 129/257 references.
  Fine-grid velocity agreement is about 0.247%; not a paper accuracy reproduction.
- `dsgnar-cavity-compiled-smoke.json`: one actual CUDA attempt with five
  conditions/remainder batches; compiled and eager final losses agree.

- `dsgnar-multiscale-trained.json`: 106-attempt smaller Fourier-network solve;
  analytical relative L2 0.000475316.
- `dsgnar-wave-trained.json`: selected 96-attempt SIREN wave experiment;
  relative L2 0.243392 is **inaccurate**, not a paper reproduction.
- `dsgnar-wave-{adaptive,fixed-fourier}-attempt.json`: retained unsuccessful
  Fourier wave experiments; do not mistake small PDE losses for a valid solve.
- `dsgnar-{wave,multiscale}-reloaded-{cpu,cuda}.json`: independent unchanged-
  checkpoint inference with analytical errors and checkpoint fingerprints.
- `dsgnar-analytic-jacobian-cuda.json`: actual-device full-network parameter
  Jacobian checks for the wave initial-velocity/PDE/BC conditions and Poisson.
- `dsgnar-{wave,multiscale}-ui-cpu.json`: six-attempt native project lifecycle
  and exports; separate from analytical-accuracy experiments.

- `dsgnar-profile-{cpu,cuda}.json`: isolated batch/phase/workspace measurements
  with post-warmup uninstrumented reference continuations. Nested durations,
  allocator/RSS scope and batch-dependent sketches are explicit. Not paper speed
  or accuracy evidence.

- `dsgnar-touched-{cpu,cuda}.json`: old/production paired complete continuations
  with exact checkpoint parity; short timing ratios are noisy. Production CUDA
  retains the old dense path.
- `dsgnar-touched-gpu-prototype.json`: rejected occupied-row CUDA candidate;
  common small-sketch cases mostly regressed despite numerical parity.
- `dsgnar-countsketch700-cpu.json` and `dsgnar-countsketch4000-{cpu,cuda}.json`:
  exact accumulation-only comparisons with explicit collision/residency scope.
  CUDA calls the experimental helper; no SVD or end-to-end claim is made.

- `dsgnar-large-baseline-{cpu64,cuda32,cuda64}.json`: pre-optimization actual
  4,000-row SVD/full-attempt profiles; CPU has reduced point counts, CUDA full counts.
- `dsgnar-large-{cpu32,cpu64,cuda32,cuda64}.json`: final complete profiles in both
  precisions/backends; CUDA includes measured SVD allocation peaks and full counts.
- `dsgnar-large-touched-cuda{32,64}.json` and `dsgnar-large-production-cuda64.json`:
  full-count old/new accumulation qualification with exact checkpoint parity.
- `dsgnar-probe-cache.json` and `dsgnar-probe-cache-{cpu,cuda}.json`: exact radius
  and full-state comparisons against the uncached interpolation reference.

- [CUDA regression audit — 13 November 2026](cuda-validation-2026-11-13.json):
  23 executed GPU cases pass without skips, including actual SVD-memory capture
  and the 3,999/4,000/4,096-row dispatch boundary.

## Bounded pointwise inference

[prediction-batches-cpu.json](prediction-batches-cpu.json) and
[prediction-batches-cuda.json](prediction-batches-cuda.json) compare one-shot and
2,048-row forwards for 50,000 points, float64, a 128×6 FCNN (83,073 parameters).
These are inference memory/timing checks, not training or solution-accuracy
benchmarks. CUDA allocated peaks exclude driver memory; complete CPU outputs and
IPC still scale with total rows. Both records check weights/RNG and actual values.
See the [workflow](../../docs/use/isolated-jobs.md#bounded-prediction-batches).

[cuda-validation-prediction-batches-2026-11-13.json](cuda-validation-prediction-batches-2026-11-13.json)
records 24 executed CUDA passes, including native network batch parity, with source
and runtime metadata. Earlier dated audits remain historical evidence.

## Fixed importance candidates

[cuda-validation-fixed-importance-2026-11-13.json](cuda-validation-fixed-importance-2026-11-13.json)
records 25 executed CUDA passes, including fixed candidate-pool continuation for
baseline, hard-BC and LRA trainers. It preserves earlier audits and includes
runtime/source identity. This validates implementation behavior, not sampling
accuracy or convergence speed. See [adaptive sampling](../../docs/adaptive-sampling.md).
