# Run a saved study without the GUI

Prepare and save a [portable study](parameter-studies.md), then execute its finite
grid from a terminal. No open Margam window, display or MCP service is required.
The first runner is **serial**: one isolated native numerical process at a time.
Each numerical attempt starts fresh. On POSIX systems, resume can verify and keep
completed trials while restarting unfinished ones in the same results folder.

For desktop controls using this same runner, see [Run saved study](run-study.md).

From the repository root and existing environment:

```bash
.venv/bin/margam-cli study inspect my-study.margam-study
.venv/bin/margam-cli study run my-study.margam-study --output results/my-study-run
```

The equivalent module command is `python -m margam.cli study ...`. If an existing
editable checkout lacks the command after an update, refresh only Margam:

```bash
uv pip install --python .venv/bin/python --no-deps -e .
```

Use a **new output directory** for each execution. Existing folders are refused;
there is no overwrite flag for study results. This protects earlier trials.
The stored portable Poisson example has twelve trials and 24,000 requested
updates: inspect its budgets before running the entire grid.

For batch packaging, see [export a study for Slurm](slurm.md). Export creates
a reviewable bundle; it does not submit a cluster job.

## Choose device, limit and explicit exports

Saved device and job-time-limit intent apply by default. Overrides are explicit
and recorded alongside that intent:

```bash
.venv/bin/margam-cli study run my-study.margam-study \
  --output results/cpu-run --device cpu --wall-time 300

.venv/bin/margam-cli study run my-study.margam-study \
  --output results/gpu-run --device cuda --export
```

`auto` uses CUDA when available, otherwise CPU. Explicit `cuda` never falls back
silently; otherwise valid trials become **incompatible** if CUDA is unavailable.
Only the first visible CUDA device is used. There is no multi-GPU scheduling yet.

The per-job limit is 0/unlimited or up to 604800 seconds. It covers the numerical
child from launch through exit, including startup, construction, training and
inference. Parent preparation, packet handling and explicit file exports are
outside that clock. Expired numerical jobs are reaped and discarded; the trial
records failure and the runner continues. It is not a memory quota or an overall
study time limit.

Every completed trial writes a native run archive. **`--export` additionally writes
files only for enabled, connected CSV Export and ParaView Export blocks.** It does
not add blocks, invent references, export every Solution Plot or open any view.
Without this flag, predictions still follow the existing explicit output requests
and are retained in the run archive, but no separate CSV/VTK files are produced.

## Find results and failure reasons

A run folder looks like this:

```text
my-study-run/
  study.margam-study       # exact frozen input archive
  execution.json          # frozen runtime, code and effective execution settings
  summary.json            # study status, provenance, per-trial outcomes/metrics
  DP0001/
    model.margam.json      # trial definition for inspection
    config.json           # resolved settings, with portable owned references
    events.jsonl          # native progress and method/sampler events
    trial.json            # status/reason, timings, hashes and export-block mapping
    artifacts/
      publication.json    # committed provenance and payload checksums
      run.margam-run.pt    # checkpoint, explicit results and owned input snapshots
      output-....csv       # only with --export and the matching block
      output-....pvd       # physical-time ParaView series, when requested
      output-..._frames/   # keep the frame folder beside its PVD
  DP0002/
    ...
```

`model.margam.json` and `config.json` are inspection records; they do not have
standalone companion data folders. **Open `artifacts/run.margam-run.pt` through
File → Open saved run…** to restore a trial's data, weights and outputs in the GUI.
The [study results browser](study-results.md) lists all reported outcomes and
verifies selected completed trials before native opening. Normal desktop checkpoint
continuation remains available. Open the root portable
study to restore the whole parameter table/base model instead.

Each trial has the same stable ID as its saved definition. A different output
folder is a new execution attempt. Device/time/export overrides and runtime code
versions are recorded separately; they do not change that definition's trial ID.
The runner records full resolved settings/seeds, the input archive/hash, actual
runtime/device, Python-source hashes, progress, artifact checksums and stage timings.
It copies data into a private workspace per trial and closes it afterward.

| Status | Meaning |
| --- | --- |
| `completed` | Native job and requested artifact publication succeeded; actual steps are recorded |
| `invalid` | Graph/configuration/required-data/output preflight failed; no numerical job was launched |
| `incompatible` | Explicit CUDA target is unavailable; no fallback or numerical job |
| `failed` | Numerical, source-identity or artifact stage failed; see `stage`, `reason` and optional `failure_kind` |
| `interrupted` | User stopped the current attempt; no further trial starts |
| `not_started` | Remaining trial after interruption or source-change abort |

Invalid/numerically failed trials do not stop otherwise valid combinations.
A runtime Python-source change aborts remaining trials and prevents publishing
uncertain results. Source checks run before launch, after numerical work and
before artifact publication. Do not edit runtime sources while a study runs.

A completed DSGNAR run may stop at its minimum radius before the requested attempt
budget. Actual `steps` is authoritative. The runner uses the same graph validation,
reference/point alignment, numerical factories and isolated workers as desktop;
it does not introduce a second training algorithm.

## Interpret diagnostics honestly

Failed or non-running trials have `metrics: null`; the runner creates no fake
objective. Completed outputs without a reference produce no accuracy metric.
Where an explicit reference exists, per-output relative L2 diagnostics retain
field names, row counts and zero-reference status. They use the existing discrete
error norm over all those fields, without interpolation, unit normalization or
quadrature. A zero reference can give an undefined value with an explicit status.
Raw/weighted training diagnostics remain in native history and `last_loss`.
These are not automatically comparable objectives across different physics.

Timings distinguish preflight, the isolated-job call, artifact preparation and
per-trial wall time. The job-call measurement includes process startup and
transport, not just synchronized training kernels. They are not paper-performance
or hardware-parity measurements. A successful short test is not solution accuracy.

## Interrupt and recover

Press **Ctrl-C** to stop. The active isolated job is killed/reaped and discarded;
completed earlier trial artifacts remain available, and remaining trials stay
`not_started`. It does not return a partial checkpoint from the interrupted job.
Artifacts are staged privately and published together only after validation and
requested exports succeed. A requested export failure marks the trial failed and
removes its staged artifacts. Already published artifacts survive interruption.

Exit codes are **0** for all trials completed, **1** for recorded failures/invalid
or incompatible combinations/source abort, **130** for interruption, and **2**
for command/setup errors. The summary is the authoritative outcome report; inspect
its trial reasons even when the process exits normally with a report.

### Continue the grid

```bash
.venv/bin/margam-cli study resume results/my-study-run
.venv/bin/margam-cli study resume results/my-study-run --retry-failed
```

Resume requires **version-2 results produced by the current runner**, on a POSIX
system with process-owned file locking. Older version-1 result folders remain
readable for inspection and individual native archives still open in the GUI;
they cannot be resumed as a grid. Portable definitions independently use version 1 for numeric grids or
version 2 for [frozen network alternatives](study-networks.md).

Before changing a summary or launching any job, resume verifies the frozen study,
Margam Python sources, recorded runtime/device, completed model/configuration and
every published artifact checksum. It uses the original effective device, time
limit and export choice; resume has no override flags. Changed inputs, code,
runtime, missing/altered artifacts or unsafe paths require a fresh execution.
This checks the recorded runtime fields, not a complete environment/container image.

- Completed trials are verified and kept without changing their files.
- Not-started, preparing, running and interrupted trials start a fresh attempt.
- Failed trials restart only with `--retry-failed`. Invalid/incompatible trials
  stay recorded; fixing the definition or environment requires a new run.
- Earlier attempts remain in place. A second attempt uses
  `DP0002/attempts/0002/`, including its own logs, inspection records and artifacts.
  The summary records current attempt, previous outcomes and execution sessions.
- Fully published artifacts can be adopted if an interruption happened before
  their completion reached the root summary. Partial staging is never a result.
- A completed grid resume performs verification only: no new jobs or session.

An exclusive `.study.lock` prevents cooperating runners from writing the same
folder. A numerical child retains that lock if its parent dies, until the child
exits. Do not remove/replace the lock file to bypass a running process. The empty
file can remain after exit; its presence alone does not mean the study is locked.
Do not edit a results folder while it is running. Filesystem locking must work
on the chosen storage; these records are integrity checks, not signed attestations
against deliberate replacement of an entire folder.

**This resumes the grid, not the interrupted optimizer step.** The active job's
partial weights are discarded. Mid-trial checkpoint recovery, GUI multi-trial
execution/collection, Slurm signals/preemption integration, concurrent scheduling
and search integrations remain future work. A hard OS/scheduler kill may leave
the last recorded state as running; resume restarts that trial unless it finds
a verified completed publication. Previously published artifacts remain intact. Current file/transport budgets and
isolated failures do not guarantee RAM/VRAM fit or prevent an OS memory kill.

## Reproduce the software workflow check

```bash
python -m unittest discover -s tests -p 'test_study_resume.py' -v
python -m examples.study_cli_workflow --device cpu --output build/study-cli-cpu-check
python -m examples.study_cli_workflow --device cuda --output build/study-cli-cuda-check
```

Choose fresh paths. These export an owned-reference Burgers study using a
state-isolated producer window, then run two/three updates through the **real
headless CLI**, verify native archives, CSV IDs/reference columns and eleven
physical-time ParaView frames. They validate software/data behavior, not trained
shock accuracy. See [recorded CPU/GPU evidence](../../assets/benchmarks/study-cli-2026-11-15.json).
