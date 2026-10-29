# Validation scope

[Documentation index](README.md) · [Measured results](results.md)

## Latest implementation suite

27 October 2026: **106 tests: 102 passed, 4 skips**, full local CPU suite in
offscreen mode after the GitHub dependency fix (145.8 seconds). Skips: three CUDA
checks and the optional VTK reader, which is not installed in the project environment.
The Qt import/startup/render preflight also passed locally. The Ubuntu workflow
change still needs confirmation on the next GitHub run.

Previous ParaView milestone: 100 tests, 97 passed and 3 CUDA skips, with VTK
available only in an isolated validation directory. Earlier milestones: Helmholtz
95 tests; LRA 93; networks 88; SA-only 78. Runtime: Python 3.14.8,
Torch 2.14.1+cpu, PySide6 6.12.0, NumPy 2.5.2. These counts are a dated record, not a permanently fixed test count.

```sh
.venv/bin/python -m unittest discover -s tests -v
QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s tests -p 'test_self_adaptive*.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_pinn_recipes.py' -v
```

Tests instantiate `Window(state_directory=False)` or a dedicated temporary state
folder, keeping personal preferences/recovery untouched. Offscreen tests release
clipboard data before Qt teardown. CUDA checks skip without a usable CUDA runtime;
CPU success does not establish current GPU behaviour.

## Numerical regression checks

These are acceptance thresholds in tests, **not measured benchmark tables**.
For exact current setups/assertions, follow the linked source.

| Check | Acceptance / meaning | Test source |
| --- | --- | --- |
| Manufactured scalar examples | Reference PDE residual max <1e-4; prediction RMSE <0.05 after 700 CPU updates (2,000 for Helmholtz) | [predefined examples](../tests/test_predefined_examples.py) |
| SA interval Poisson | RMSE <0.04 after 700 updates, with BC masks off and on | [SA numerical checks](../tests/test_self_adaptive.py) |
| SA update equations | Two-point exact λ gradient [1,4]; network descends while masks ascend | [SA numerical checks](../tests/test_self_adaptive.py) |
| SA resume | Identical loss history, masks and predictions for 30+20 resumed vs uninterrupted updates | [SA numerical checks](../tests/test_self_adaptive.py) |
| Couette baseline | Analytic residual/BC checks; each u/v/p RMSE <0.03 after 750 updates on 11×11 evaluation points | [coupled flow](../tests/test_incompressible_flow.py) |
| Rectangle reference truncation | Selected 40/80-mode values within 3e-6; zero edges and symmetry | [reference checks](../tests/test_rectangle_reference.py) |
| Independent reference comparison | Max difference <0.002 from finite differences on a translated non-square 31×21 grid | [reference checks](../tests/test_rectangle_reference.py) |
| Recipe representations | Identical numerical predictions from compact/expanded baseline configurations at equal seed/budget | [recipe lifecycle](../tests/test_pinn_recipes.py) |

SA also checks vector residual rows/targets, cap projection, optional constraint
masks, coupled-flow dispatch, rejection of nonfinite loss/gradient before updates,
and incompatible/corrupt adaptive checkpoints. These checks verify implemented
contracts; they do not reproduce stiff paper benchmarks or prove convergence.

## Desktop and persistence checks

The suite covers typed wiring, block insertion, owned parameters, copy/paste,
undo/redo, compact/expanded recipes, background training, cancellation/teardown,
explicit outputs/references, retained monitor histories, model/run save/open,
checkpoint resume, data mappings, CSV, planar GeoSampler and local recovery.
SA's desktop check saves both optimiser states, opens an archive, resumes updates
and inspects adaptive statistics. [SA workflow test](../tests/test_self_adaptive_ui.py).

UI startup or a screenshot demonstrates appearance/lifecycle, not numerical accuracy.
Screenshots are under [assets/screenshots](../assets/screenshots/README.md). Model format 12
reads 1–11; run archives use format 2 and restricted tensor loading. Data fingerprints
and architecture/options are checked for resume, but cross-version optimiser
compatibility and arbitrary PDE well-posedness are not guaranteed.

## Documentation/evidence verification

The documentation update repeated the six-run SA comparison and saved its complete
record. Historical heat and GPU measurements were preserved with explicit provenance
limits. Local Markdown targets, JSON structure/finite metrics, recorded source hashes,
reported tables and CLI help/output preservation were checked. No production numerical
algorithm changed as part of that documentation work.

## Pointwise network additions

All ten choices pass numerical first/second-derivative checks, per-row independence,
output shapes, real u/v/p training integration and exact adaptive checkpoint resume.
Additional checks cover exposed Fourier encodings, activations, skip/gate branches,
Gabor centres, legacy FCNN identity, setting-mismatch rejection and hard-Dirichlet
architecture metadata. [Numerical tests](../tests/test_pointwise_networks.py).
The [architecture UI test](../tests/test_network_ui.py) exercises every choice through
save/open, compact conversion, clipboard and actual archive/resume. Startup remains
Torch-free until numerical execution is requested.

The [twenty-run compatibility record](../assets/benchmarks/pointwise-networks-2026-10-27.json)
contains predictions as well as metrics. Their errors and rounded documentation
table were recomputed/checked; selected source hashes and local documentation links
were verified. This one-seed CPU study is separate from regression thresholds and
from any architecture-quality or CUDA-equivalence claim.

## Illustrated guide capture

The [walkthrough](visual-guide.md) uses nine native UI captures, individually
inspected for readable controls, complete menus and visible graph/output content.
The [manifest](../assets/screenshots/guide-captures.json) records application source
`e0fde23`, unchanged application tree, offscreen/light rendering and hashes/settings.
The helper trains one real SIREN/SA interval case for 700 CPU updates, verifies
configuration preservation across recipe conversion and that output views remain
closed until requested. Captured comparison RMSE is approximately 5.32e-4 on 201
points. This documents a working UI example, not another architecture benchmark.
Only documentation and capture tooling changed in this update; the earlier 88-test
implementation result remains a dated record rather than a claim of rerunning that
full suite for screenshots.

## LRA validation

`test_lra.py` verifies max/mean over parameter entries, raw denominator (excluding
existing lambda), alpha as new-estimate fraction, cap and zero-gradient hold,
nonfinite aborts, event phase across batches, exact checkpoint continuation and
rejection of changed settings/data/tampered events. All ten networks train for a
smoke budget; a real u/v/p model also trains. Those short runs do not establish
accuracy for each architecture or flow. `test_lra_ui.py` exercises real Qt recipe
insertion, parameters, expand/collapse, clipboard/undo, model round-trip, background
training, restricted run loading, resume and optional weight-summary action.

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_lra*.py' -v
```

The configured cap100 Poisson example also participates in the existing
700-update RMSE <0.05 example check. Measured 2,000-update comparisons, including
the high-cap failures to fit, are in [results](results.md#learning-rate-annealing-pinn).

## Helmholtz validation

`test_helmholtz.py` checks the supplied paper's square domain, sine modes, forcing
and positive mass sign, zero edge values, Fourier feature settings and rejection
of negative k². Its real Qt workflow uses the full configured samples with a short
25+5 update budget to verify background execution, explicit references, model/run
round-trips, compact recipes and resume. Those lifecycle updates do not establish
accuracy. The separate manufactured-example check trains 2,000 updates and tests
RMSE <0.05 over random points across the full square. Measured 5,000-update results
are in the [Helmholtz study](results.md#helmholtz-with-fourier-lra-pinn).

## ParaView export correctness

Five focused tests cover known-value scalar/u-v-p export, velocity vectors, grid
ordering, physical timestamps, XML escaping and failed-write preservation. Actual
Qt tests add the block explicitly, train with random samples, export independent
regular-grid predictions, and reopen model/run files. Both scalar and coupled-flow
exports are exercised. No files or views are opened by training. The optional real
VTK 9.7.1 test verifies readback geometry, fields and TIME_STEPS metadata. The bundled
21-frame synthetic dataset was independently read and checked at every point;
[dataset validation](../assets/examples/paraview-demo/validation.json) records this.
These are file/data correctness checks, not solution-accuracy measurements.
ParaView GUI playback was not run because ParaView is not installed on this host.

## Illustrated manual and Help

The manual adds five staging/link/configuration/privacy checks and one Qt help lifecycle
check. Sources remain untouched by staging, root README/catalogue links stay
distinct from the homepage, missing targets/section links are rejected, and F1
opens the local guide without importing Torch or Zensical. Missing builds and
browser failures have explicit messages. Desktop startup and block-context
checks also pass after the Help menu change. These targeted checks supplement
the preceding full numerical/export suite; it was not rerun for documentation.

A strict Zensical build and rendered link checker cover the manual's pages, images
and section targets. Browser preview confirmed navigation, search, result-page
opening, screenshot rendering and theme switching. The in-app test browser blocks
file URLs; direct-file offline search was therefore not exercised. The bundled
shim and offline configuration are verified locally. The 22 UI captures come
from a state-isolated helper that runs the default baseline tutorial and saves
both model and checkpoint. This is workflow evidence, not a new accuracy study.

## GitHub runner Qt dependencies

The Linux workflow installs Ubuntu `libegl1`, `libgl1` and `libopengl0` before
running the Python suite. PySide6 wheels depend on system graphics libraries even
when tests use `QT_QPA_PLATFORM=offscreen`; selecting that plugin does not remove
the dynamic-linker requirements.

The supplied failing GitHub log had 33 UI failures, each caused by the same missing
`libEGL.so.1` import, before application checks could run. The workflow now performs
a QtWidgets/QtTest import, QApplication startup and offscreen widget-render check
before running the full suite. It fails immediately with the dependency error
rather than repeating it through every desktop test. UI coverage remains enabled.

CUDA and optional VTK-reader skips remain conditional on those runtimes being
available; they are separate from missing Qt system libraries. A local passing
suite does not itself confirm a remote GitHub run; verify the next workflow run
after the change is pushed.

## Residual sampling verification — 28 October 2026

The full CPU/offscreen regression run passed **115 tests**, with **4 skips**
(three CUDA-only checks and the optional external VTK reader). Nine adaptive
sampling checks cover greedy selection and deduplication, the importance estimator
and gradient identity, derivative-aware chunked scoring, finite-value failures,
scalar/flow integration, phase-boundary checkpoint continuation across supported
trainers, legacy model defaults, and UI editing/clipboard/recipe/archive workflows.
No CUDA execution or comparative numerical accuracy claim is made for the new
samplers. The separate [CPU demo record](../assets/benchmarks/adaptive-sampling-cpu.json)
records point counts, residual statistics and exact continuation checks.

## Integral squared-error verification — 28 October 2026

The full CPU/offscreen suite passed **124 tests**, with **4 skips** for unavailable
CUDA/optional VTK checks. Eight integral-loss tests cover volume/edge-length
scaling, point-count independence, summed vector components and gradients,
squaring before integration, unit endpoint/gauge penalties, importance correction,
method/checkpoint combinations, legacy defaults and compact/UI persistence. A
separate final archive check covers importance sampling combined with integral
loss. The manual build checked all 47 rendered pages and local links. See the
[CPU workflow evidence](../assets/benchmarks/integral-loss-cpu.json); this establishes
software behaviour, not improved solution accuracy.

## Burgers space–time verification — 28 October 2026

The final full CPU/offscreen suite passed **130 tests**, with **4 skips** for
unavailable CUDA/optional VTK checks. Six Burgers checks cover original PDE/IC
setup, nonlinear/time/spatial derivatives, spatial-only Laplacian, absence of a
final-time condition, method/sampler continuation, measure scaling, model/archive
round trips, explicit tables, x/t/u CSV and physical-time PVD frame contents.
Existing flow, recipe and export tests also pass. The manual build checked all
49 rendered pages. XML metadata/field contents were read back; ParaView GUI
playback and shock accuracy against reference data were not exercised. See the
[100-update CPU record](../assets/benchmarks/burgers-cpu.json).

## CUDA environment verification — 28 October 2026

The existing root environment was changed from Torch 2.14.1+cpu to
**2.14.1+cu130** (CUDA runtime 13.0), keeping Python 3.14 and the same environment.
`uv pip check` passed. CUDA availability, a tensor matrix multiplication and
actual training were verified on an NVIDIA GeForce RTX 3060 Laptop GPU.

All **three existing CUDA checks passed without skips**: CPU/GPU loss agreement,
auto-device Poisson training/data placement, and GPU benchmark artifact creation.
Additional five-update Burgers CUDA checks passed for fixed sampling, RAR, and
importance sampling with integral loss. These were targeted GPU checks; the full
130-test run recorded above was the earlier CPU validation. The setup/device
commands are in [the command reference](commands.md#devices-and-cpu-setup).

## Expert PINN verification — 28 October 2026

The full CPU/offscreen regression suite passed **139 tests**, with **5 skips**:
four CUDA-only checks excluded from this CPU run and the unavailable optional
VTK reader. The Expert numerical suite separately passed all **8 tests** with
CUDA available, including real GPU training/checkpoint/archive checks. The final
Expert UI test also passed after diagnostics were added.

Checks cover the detached causal prefix and its gradient, paper v1 sum-L2-norm
ratios/old-weight EMA/zero-gradient handling, RWF initial values and input
derivatives, fresh temporal-stratum/IC geometry, Adam decay, same-device exact
continuation at balance boundaries, corrupted state rejection, legacy defaults,
compact expansion and network replacement. CPU-canonical target fingerprints
allow GPU archives to open on CPU.

Independent 100-update [CPU](../assets/benchmarks/expert-cpu.json) and
[CUDA](../assets/benchmarks/expert-cuda.json) workflow records each contain ten
balance events and exact same-device continuation. They establish implementation
behaviour, not agreement with JAX or published accuracy. The illustrated manual
build checks all 50 generated HTML pages and local links.


## Supplied evaluation data

`tests/test_evaluation_data.py` checks exact row/ID preservation, shuffled reference
alignment, finite/domain validation, NPZ integer precision and object rejection,
offline run archives, and reference/error arrays in regular and supplied exports.
Time-series checks read the emitted XML and verify physical times, per-frame
point counts and original source-row indices. Optional VTK-reader tests are
skipped when VTK is unavailable; no new ParaView GUI playback claim is made.

`tests/test_evaluation_ui.py` exercises separate source blocks, mapping preview,
preflight rejection, model/run persistence, live and archived weight reuse,
offline export and invalidation after core-physics edits.
`tests/test_prediction.py` checks checkpoint prediction against trained networks
across available architectures, baseline/SA/LRA/hard-BC/Scale/Expert methods and
Expert time windows, including deletion of the original training-data file.

The [CPU record](../assets/benchmarks/evaluation-points-cpu.json) and
[CUDA record](../assets/benchmarks/evaluation-points-cuda.json) independently verify
the four-point example after 100 updates, unchanged weights/history and no
optimizer updates for evaluation. These validate implementation and data flow,
not solution accuracy. See the [illustrated guide](evaluation-points.md).


## Prepared training-data blocks

`tests/test_training_data.py` checks manufactured triangle residuals and imported
Dirichlet/Neumann operators, explicit outward normals, coordinate bounds, NPZ
mappings, unsupported combinations, document validation and exact CPU continuation
for baseline/LRA/fixed-SA. Flipped normals invalidate the checkpoint identity.
`tests/test_training_data_ui.py` checks separate source inspectors, named-set
selection, validation before worker startup, model/run persistence, resume, undo
and offline archived export. CPU/CUDA evidence and actual UI captures are linked
in [the prepared-data guide](training-data.md). These are implementation checks,
not a solution-accuracy benchmark.


## Affine Fourier features

`tests/test_affine_fourier.py` checks notebook input equations, physical units,
fourth-order derivatives with/without normalization, projection updates, exact CPU
resume, changed-map rejection and historical fixed-Fourier loading.
`tests/test_affine_fourier_ui.py` checks editable mappings, save/load and compact
recipe roundtrips. Scale and pointwise-network suites also cover the new option.
New CPU/CUDA short-run evidence is linked from [the Scale guide](scale-pinn.md).


## gPINN

`tests/test_gpinn.py` verifies full-residual/forcing/callable-target derivatives,
backpropagation into weights, constant derivative zeros, disabled baseline identity,
non-finite update guards, RAR continuation, method incompatibilities and legacy
checkpoint migrations. Its CUDA case checks differentiable residual gradients and
resume. `tests/test_gpinn_ui.py` exercises coordinate controls, compact expansion,
models/run archives, named monitor histories and Burgers physical-time export.
See [records and exact scope](gpinn.md).


## RBA

`tests/test_rba.py` verifies the exact recurrence, initialization modes, stopped
attention gradients, squared residual weighting, unchanged boundary losses, all
network choices, fixed-point identity, failed-step atomicity and checkpoint guards.
It checks legacy Scale/Expert/gPINN migrations and includes a CUDA-to-CPU archive
identity case. `tests/test_rba_ui.py` checks controls, compact roundtrips, raw/weighted
monitor histories, attention statistics, resume and physical-time PVD export.
Evidence and exact scope are in [RBA](rba-pinn.md).


## Jorvik geometry documents

`tests/test_jorvik.py` uses the installed Jorvik exporter to check unequal NetCDF
dimensions, planar coordinate/normal validation, explicit dataset roles, document
creation, editor command/locking, undoable updates, BC cleanup, unchanged state on
failed updates, portable Save/Save As, relocation and snapshot tampering. Desktop
state tests also recover owned documents. The full CPU regression run completed
213 tests: 204 passed and 9 skipped. The local installed Jorvik version was built
from commit `795c5ff`. A separate isolated-process smoke check launched that editor
on an empty owned document and verified the lock was released on exit.

The [annulus interchange record](../assets/benchmarks/jorvik-exchange.json) checks
real CAD-derived samples, hole/outer normal orientation and exact CPU checkpoint
continuation. The manual build checked 60 rendered pages. See the illustrated
[Jorvik workflow](jorvik.md) for current scope and reproducible commands.
