# Resource-probe report contract

The CLI/API returns a data-only JSON-compatible measurement dictionary. Keep its
source study separately for replay. The desktop's Copy JSON report copies that
whole dictionary; it is not a solver checkpoint.

## Versions and scope

| Version | Configuration | Measurement scope |
| --- | --- | --- |
| 1 | Single-window native model | Existing construction/update process RSS and CUDA allocator peaks |
| 2 | Expert with two or more windows | Construction, native window updates and reached teacher transfers, with ordered phase records |

Both versions keep `format: "margam-resource-probe"`, actual device/dtype, work
counts, measured timing/memory, runtime, `capacity_prediction: false` and
`checkpoint_retained: false`. `probe_study` adds frozen study/trial/archive and
source hashes, original configuration metadata and transport duration. Direct
worker audits store the full configuration and source manifest beside the report.

Version 2 supports the native 2–32-window limit. Probe work remains within the
original `config.iterations` stop horizon, capped at 100,000 for marching or
1,000 for single-window configurations. It does not replace that horizon with
the window-count × per-window budget or invent work at a future boundary.

## Version-2 fields

| Field | Meaning |
| --- | --- |
| `configured_windows`, `updates_per_window` | Original native marching schedule |
| `observed_windows`, `observed_transfers` | Windows updated and completed handoffs actually reached |
| `next_transfer_pending` | The probe ended at a non-final boundary before executing its next handoff |
| `teacher_count`, `cached_teacher_copies` | Retained frozen teachers and their native device copies at the end |
| `internal_window_snapshots_included` | True: handoff cost includes native completed-window CPU state snapshots |
| `measurement_overhead_included` | True: phase synchronization/bookkeeping affects measured time and can alter overlap/allocator reuse |
| `phases` | Ordered construction/training/transfer records, at most 64 |
| `peak_aggregation` | Explicit maximum/cumulative-counter semantics |

Each phase has exactly these fields:

- `kind`: `construction`, `window_training` or `teacher_transfer`;
- `window`: one-based window, the **destination** for a transfer;
- `start_update`, `end_update`: global update interval; a transfer advances no optimizer update;
- `time_interval`: physical bounds of that window, preserving native local scaling;
- `duration_seconds`: synchronized elapsed time for that observed phase;
- `process_peak_rss_bytes`: cumulative child-process high-water mark, or null when unavailable;
- `cuda_start_allocated_bytes`, `cuda_peak_allocated_bytes`, `cuda_end_allocated_bytes`;
- `cuda_start_reserved_bytes`, `cuda_peak_reserved_bytes`, `cuda_end_reserved_bytes`;
- `completed_window_snapshots`, `teacher_count`, `cached_teacher_copies`;
- `pde_rows`: actual collocation count at phase end.

CUDA counters are null for CPU execution. Phase allocator peaks include live state
from earlier phases and start/end baselines. The overall CUDA values equal the
**maximum** of each counter's phase peaks. Reserved pools persist; phase peaks
are not disjoint costs to add. RSS has no phase reset and cannot identify a
phase-local increment. CUDA context/driver and other processes remain outside
allocator measurements.

For three windows of two updates, a five-update report contains:

| Phase | Window | Global updates |
| --- | --- | --- |
| Construction | 1 | 0 → 0 |
| Training | 1 | 0 → 2 |
| Transfer | 1 → 2 | 2 → 2 |
| Training | 2 | 2 → 4 |
| Transfer | 2 → 3 | 4 → 4 |
| Training | 3 | 4 → 5 |

Ending at update 2 instead produces only construction/training, zero transfers
and pending=true. Normal native behavior in a manually configured final window
is preserved even when the original stop horizon differs from the planned window
schedule. Internal working snapshots/teachers are included; final archive capture,
output evaluation and exports are excluded. Failed/cancelled jobs return no
successful or partial measurement.

## Validation and numerical behavior

`margam.execution.probe_report.validate_window_report` checks phase order, exact
work ranges, time continuity, pending status, teacher/snapshot counts, row counts,
finite durations, bounded byte counters, cumulative RSS and maximum CUDA
aggregation. It imports neither Torch nor Qt. The native worker, API and desktop
use it; desktop review also binds the original window schedule and archive hash.
Reports use at most 256 KiB of desktop CLI capture, while ordinary study logs keep
their existing 64 KiB tail.

The observer is a trusted in-process profiler hook created only for disposable
work. It is absent from checkpoints and is not accepted as executable model-file
data. Ordinary training uses the same handoff body. Same-device CPU/CUDA checks
compare every checkpoint field, optimizer/sampling/balancing state, teachers,
piecewise predictions and RNG exactly for the tested scalar/coupled cases.
These numerical checks do not promise identical uninstrumented memory overlap,
wall time, cross-device reproducibility or paper accuracy.
