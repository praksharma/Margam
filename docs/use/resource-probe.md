# Measure a disposable training probe

The allocation-free [Run summary](resources.md) describes workload without
training. For a measured check, save a [parameter study](parameter-studies.md),
then run this from the existing Margam environment:

```bash
.venv/bin/margam-cli study probe my-study.margam-study \
  --trial 1 --updates 2 --device cpu --wall-time 60
```

Use a study with one Fixed parameter if you only need one configuration. `--trial`
is the **one-based design-point number**; paired and Cartesian studies use their
saved ordering. The command prints JSON and creates no results folder, run
archive or export files. It never adopts the probe's weights into the application.
There is currently no desktop probe button.

Choose `--device cuda` for CUDA, or `auto` for availability-based selection. If
omitted, the saved study's device intent applies. Explicit CUDA never silently
falls back to CPU. The numerical process uses the same environment as the caller.

For scripts, the same operation returns a plain report dictionary:

```python
from margam.execution.probe import probe_study
report = probe_study("my-study.margam-study", trial=1, updates=2, device="cpu")
```

## What runs

A new isolated process builds the selected native model and executes complete
training updates through the ordinary numerical worker. Gradient-based methods count
updates; DSGNAR counts attempts, including rejections. The original configured
training horizon, seeds, loss coefficients, architecture and sampling schedules
stay intact. Only the number of updates performed by this disposable job changes.
The probe is fresh training, not checkpoint continuation. It uses the native CLI
float32 default; DSGNAR's explicit precision setting still applies. The report
records the actual dtype, rather than assuming the dtype of an open checkpoint.

`--updates` defaults to 1, accepts 1–1000 and cannot exceed the trial's configured
budget. One update includes derivative evaluation, loss, backward work and the
method's normal update. Adam state, SA masks and Scale's previous network are
constructed by the native method. Scheduled balancing, NTK work and residual
sampling occur **only if the probe reaches their configured schedule**. Increase
the explicit probe budget to include later work; the command never shifts a
schedule forward to make a short probe appear representative.

Single-window native methods are supported through their existing validators.
Multiple Expert time windows are explicitly rejected: teacher-transfer and later
window peaks have not been qualified. Invalid/incompatible trials fail without
printing a measurement. No random replacement of missing imported points occurs.

## Read measured values

| JSON field | Meaning |
| --- | --- |
| `completed_updates`, `requested_updates`, `configured_updates` | Actual work, requested probe work and original training horizon; convergence can stop DSGNAR early |
| `process_peak_rss_bytes` | Whole child-process resident-memory high-water mark, including Python/Torch imports; `null` when unavailable |
| `cuda_peak_allocated_bytes` | PyTorch allocator peak covering construction and training after CUDA context initialization; `null` for CPU |
| `cuda_peak_reserved_bytes` | Peak reserved allocator pool over that same CUDA scope |
| `build_and_train_seconds` | Construction and observed training, synchronized on CUDA; excludes imports, CUDA context initialization, output evaluation and checkpoint transfer |
| `numerical_job_seconds` | Parent-observed transport duration, including packet preparation, child startup and result transfer; preflight is separate |
| `initial_pde_rows`, `final_pde_rows` | Actual collocation row counts before/after the observed updates, preserving connected boundary rows |
| `observed_sampling_events` | RAR/importance controller events actually reached; this does not count Expert's own fresh-batch mechanism |
| `requested_output_count`, `excluded` | Output requests were validated but are not evaluated or exported; exclusions are recorded explicitly |
| `study_id`, `trial_id`, `study_archive_sha256`, `source_sha256`, `runtime` | Definition, input archive, code and child runtime provenance |

The report also records the actual device/dtype, network family/widths, method,
adaptive settings and job limit. Byte values are measurements, **not a capacity
prediction or reservation**. CPU RSS cannot be reset between phases; its scope
is deliberately broader than the CUDA allocator measurement. Other processes,
CUDA driver/context memory and OS/scheduler quotas are not measured by the CUDA
allocator counters.

Outputs, exports, checkpoint capture/transfer, unobserved adaptive growth and
later scheduled work remain excluded. A short probe can succeed while a longer
run or output evaluation runs out of memory. No full-run peak is extrapolated.

## Limits and isolation

The child defaults to a **60-second launch-to-exit limit**. Use `--wall-time 0`
explicitly for unlimited time, or another limit up to seven days. Parent-side
input/output packet work is outside that deadline. The probe can itself exhaust
memory; this is neither an OS memory quota nor an automatic batch-sizing system.

Ctrl+C, a worker failure or timeout discards the probe. A runtime-source change
also refuses the measurement. The saved study, inputs and parent model/RNG remain
unchanged; private workspaces are cleaned up. No checkpoint or numerical output
is retained, and nothing is resumed automatically after failure.

## Reproduce the small workflow check

```bash
.venv/bin/python -m examples.profile_resource_probe \
  --devices cpu cuda --output build/resource-probe.json
```

The [recorded CPU/CUDA measurements](../../assets/benchmarks/resource-probes.json)
use two complete updates of the shipped Poisson study. They demonstrate measured
scope and isolation, not solution accuracy, a hardware speed comparison or a
universal fit threshold. Omit `cuda` on a CPU-only environment. Existing evidence
files are refused unless `--overwrite` is explicitly supplied.
