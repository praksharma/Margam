# Curated numerical evidence

Small, versioned measurement records supporting [the results catalogue](../../docs/results.md).
Full generated experiments belong in ignored `benchmark-results/`. Do not overwrite
these files to refresh a study: create a dated record and update the catalogue.

| Record | Status | Payload |
| --- | --- | --- |
| [self-adaptive-poisson-2026-10-27.json](self-adaptive-poisson-2026-10-27.json) | Repeated during documentation update, CPU | Six paired baseline/SA runs; complete base config, per-seed overrides, environment, selected source hashes, RMSE/relative L2/max error, final losses and λ summaries |
| [heat-accuracy.json](heat-accuracy.json) | Original 27 October CPU study preserved | Nine soft/hard BC runs with errors and losses; timestamp/Torch/grid/reference metadata; no full source/environment snapshot |
| [baseline-poisson-cuda-2026-10-23.json](baseline-poisson-cuda-2026-10-23.json) | Historical GPU record preserved, not rerun | Original manifest, three records and aggregate copied unchanged from a local archived run; source JSON hashes and preservation note |

Values are measurements under their stated settings, not expected constants or test
thresholds. Floating-point results can differ with software and device. Histories
are pre-update losses; field errors are after the final update. The SA objective
changes with λ and cannot be compared as an unweighted baseline loss.

The SA report's `tracked_changes=true` records the reporting-script edit present
at execution; its selected source hashes can be checked against this commit's files.
Its base config is overridden by each record's method and network seed. Historical
GPU payload fields retain their original names and values. No new CUDA validation
is implied by preserving them.
