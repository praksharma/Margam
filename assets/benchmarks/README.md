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
