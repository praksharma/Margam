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

The automatic-export follow-up additionally verifies reading generated arrays from
a saved project, preserving an existing NetCDF file when no generated samples are
available, prefilled interior/boundary roles and editable per-boundary names. The
real annulus example now saves and reloads a Jorvik document before automatically
exporting it. Ten focused persistence/integration tests and the 61-page manual
build passed after this change.

## Explicit PDE collocation connections

`tests/test_pde_collocation.py` verifies interior-only, boundary-only and combined
coordinates against the actual physical residual loss; independent BC rows;
arrow removal and model migration; fixed adaptive anchors and exact continuation;
and a training update for every File → Examples configuration. Updated UI tests
cover optional boundary wiring, imported point counts, integral quadrature without
the boundary arrow, refinement budgets and model/run archive round trips.

The full CPU regression run completed 219 tests: 210 passed, 9 skipped. Separate
CPU/CUDA continuation records cover baseline, importance sampling, gPINN, RBA and
Expert with connected boundary/initial PDE points. See the
[point-selection guide](modeling/physics.md#arrows-choose-where-the-pde-is-enforced)
for records, screenshots and reproducible commands. These checks establish
implementation behaviour and same-device continuation, not solution accuracy.


## Application device and expanded help

`tests/test_application_device.py` verifies local preference persistence, overriding
legacy per-block device values, unavailable-GPU behaviour, execution locking and
expanded parameter/dropdown/library descriptions. A separate CUDA run verifies CPU
training, weight-preserving GPU inference, GPU continuation and CPU inference using
only the application menu. The final focused persistence/help run passed nine cases
with its CUDA case skipped in the CPU-only run; that CUDA case passed separately.
The illustrated manual build checked 61 rendered pages.

```sh
CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest tests.test_help_ui tests.test_application_device tests.test_block_insertion tests.test_model_files_ui tests.test_evaluation_ui tests.test_desktop_state -v
QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest tests.test_application_device.ApplicationDeviceTests.test_training_inference_and_resume_follow_selection -v
```


## Complete Jorvik desktop workflow

`examples.jorvik_workflow` exercises the real desktop worker and saved Jorvik
annulus, mixed Dirichlet/Neumann operators checked with `u=log(2r)`, explicit 143-row
PDE collocation, stop/archive/reopen/resume with the original companion folder
hidden, exact same-device continuation versus uninterrupted training, unchanged
weights during inference, and exact supplied-point VTU fields
and IDs. CPU and CUDA records live in
[CPU](../assets/benchmarks/jorvik-workflow-cpu.json) and
[CUDA](../assets/benchmarks/jorvik-workflow-cuda.json).
This exposed and fixed missing owned files in pre-version-4 run archives.
`tests/test_jorvik_workflow.py` retains CPU lifecycle and unsafe extraction-path
regressions. Stop timing and final update count are scheduling-dependent.
No solution-accuracy claim is made. ParaView/VTK reader verification remains
unperformed here because neither is installed.

The complete CPU regression run after this change completed **224 tests: 214
passed and 10 skipped**. CUDA was hidden for that suite; the complete CUDA
workflow was run separately and passed. The final focused archive/workflow/output
suite also passed all five cases. The manual build checked 61 rendered pages.


## Named boundary loss coefficients

`tests/test_boundary_weights.py` checks unit-default compatibility, the manual
weighted objective and its exact parameter gradients, raw/weighted separation,
non-unit coefficients and exact 2+2 continuation for baseline, SA, LRA, Expert,
Scale, gPINN and RBA on CPU and CUDA. The CUDA case was run separately with CUDA
visible. SA diagnostics are checked against the pre-update masks, rather than
recomputed after their ascent update. Changing an active coefficient is rejected
on checkpoint load. UI checks cover control editing, undo/redo, model 27 and
version-26 migration, archive/resume and selected monitor data. Jorvik update tests
also verify coefficient removal/restoration when an alias is renamed/undone.

The reproducible [CPU](../assets/benchmarks/boundary-weights-cpu.json) and
[CUDA](../assets/benchmarks/boundary-weights-cuda.json) records use imported triangle
points, mixed BC operators, coefficients 2, 0.5 and 3, and four baseline updates.
The sum of recorded weighted BC contributions plus the PDE contribution reconciles
to the total within floating-point tolerance. These are implementation checks,
not numerical-accuracy claims. See the
[illustrated Jorvik controls](jorvik.md#weight-named-boundaries-and-inspect-their-contributions).

The complete CPU regression run completed **228 tests: 217 passed and 11
skipped**. The final focused coefficient/Jorvik/flow/archive suite completed
11 tests with 10 passed and one CUDA skip. A separate CUDA test exercised all
seven trainers and passed. The manual build checked 61 rendered pages; the
capture helper refreshed 22 real UI walkthrough images plus the Jorvik captures.


## Portable CSV/NPZ datasets

`tests/test_dataset_storage.py` covers fixed copies versus live external links,
failed-copy cleanup, cancellation without allocating an owned folder, 64 MiB
bounds, busy-editor guards, owned-source previews, undo/redo after Save,
legacy Reference data and the complete portable-data workflow. Existing external
link fingerprint checks remain explicit in `tests/test_data_ui.py`.

`examples.portable_data` uses real mapping dialogs for six training, boundary,
evaluation and reference sources (CSV and NPZ), removes their original files,
validates recovery, preserves weights through first Save and Save As, trains a
relocated model, and resumes a run archive after deleting the model data folder.
Same-device 2+2 continuation matches four uninterrupted updates exactly; supplied
IDs/order and a separately ordered reference are preserved and predictions export
through the existing CSV/VTU blocks. Both [CPU](../assets/benchmarks/portable-data-cpu.json)
and [CUDA](../assets/benchmarks/portable-data-cuda.json) records passed. These checks
exercise portability and correctness, not solution accuracy.

The complete CPU regression run completed **233 tests: 221 passed and 12
skipped**. Final focused storage/project-save checks completed 11 tests with ten
passed and one CUDA skip. The complete CUDA portability example was also run
separately and passed. The illustrated manual build checks 61 rendered pages;
its capture helper refreshed the existing 22 UI images and three new data-storage
captures. The existing root environment and dependencies were unchanged.


## Project data overview

`tests/test_project_data.py` checks per-block mapping inventory (including
unconnected sources), copied/linked CSV and NPZ counts, both reference block
forms, field/ID details, invalid mappings, unchanged model/history/results during
inspection, persistent session change warnings, missing owned files without
creating folders, Save As baselines, named-boundary navigation, stale/locked
editing guards and clean Qt teardown. A real Jorvik NetCDF fixture checks its
stored snapshot fingerprint independently of a session baseline and routes
review to the owning geometry block.

`examples.capture_project_data` captures an actual triangle file inventory with
one copied dataset, one missing linked file and another changed linked file.
The helper verifies model/history preservation. Refresh is intentionally manual;
older mappings without an accepted import fingerprint use session-only hashes,
which are not evidence of an older import-time file version. No numerical training or GPU claim is made by these
file/UI checks.

The complete CPU regression run completed **236 tests: 224 passed and 12
skipped**. The final focused overview/storage/workspace checks completed 11 tests
with ten passed and one CUDA skip; the final overview-only checks passed all
three cases, including reference mappings and clean teardown. The illustrated
manual build checked 61 rendered pages. No dependencies or environments changed.


## Clickable multi-issue validation

`tests/test_validation_report.py` checks simultaneous connection, hidden-layer,
PDE and multiple BC-expression errors; multiple missing source files; prepared
boundary normal requirements; early missing/mismatched legacy reference rejection;
explicit prepared output points; non-creating missing-folder checks; coupled-flow component expressions; nonblocking
unused-source warnings; correct block/boundary navigation; stale/deleted owners;
model/history/results preservation; programmatic no-popup behaviour; clean Qt
teardown; and the existing hard-Dirichlet, integral-quadrature and corner checks.
`examples.capture_validation` captures an actual invalid model and verifies no
training or model/history edits occur.

The same pre-existing sequence (compile graph, validate numerical configuration,
read prepared training sets, build explicit output requests) remains authoritative.
Output preparation now also checks legacy mapped reference coordinates and
requires explicit Evaluation points for prepared-domain outputs before training.
Independent active-data errors block early, while unused optional declarations
remain warnings. Owned input paths are resolved without creating missing folders.
Independent diagnostics reuse the existing expression parser, network validator,
prepared-data readers and method compatibility checks. They are additional UI
explanations rather than a parallel numerical executor. Checks blocked by missing
connections may appear only after those connections are fixed. Unused optional
declarations are advisory when the existing executor accepts the graph.

The complete CPU regression run completed **239 tests: 227 passed and 12
skipped**. Final focused validation/data-overview/storage/Jorvik checks completed
15 tests with 14 passed and one CUDA skip. Earlier focused input/output checks
also passed. The manual build checked 61 rendered pages. These are desktop and
preflight checks; no new GPU-performance or solution-accuracy claim is made.


## Persisted input fingerprints

Model format **28** stores accepted source mappings and bounded SHA-256 file
fingerprints separately from numerical parameters. Versions 1–27 remain readable;
older missing fingerprints are never fabricated. Fingerprint metadata validation
imports no Qt. All source formats retain their original numerical checkpoint
identity rules.

`tests/test_source_fingerprints.py` verifies strict source-slot/hash validation,
legacy loading, external file changes before reopening, fixed baselines on
Refresh/Save, warning-only valid changed inputs, explicit reacceptance as one
undo/redo operation, rejection of stale mapping bindings, copied inputs through
Save As and run archives, and unchanged detection after another save/reload.
Existing portable-data tests exercise fingerprints on all six training/evaluation/
reference import paths. The refreshed project-data screenshot records accepted
fingerprints, saves/reopens the project, then detects a missing file and changed
linked bytes without altering model/history during inspection.

The existing 239-test CPU regression run passed (227 passed, 12 skipped). The
three new fingerprint persistence cases passed separately. Final focused
fingerprint/storage/overview/Jorvik checks completed 15 tests with 14 passed and
one CUDA skip. Metadata decoding was also checked without importing Qt. The
manual build checked 61 rendered pages; no numerical-accuracy or GPU-performance
claim is made by these metadata checks.

## Simulink-inspired diagram tools (2 November 2026)

The Format menu provides six edge/centre alignment commands, equal-gap
horizontal/vertical distribution and deterministic left-to-right automatic layout.
Find in model searches existing instances and resolves IDs again before selection.
The interaction case checks graph/settings/result preservation, one-step undo and
redo, stable repeated layout, forward dependencies, cycle termination, no-overlap
distribution, busy locks and navigation after deletion. The startup test now
expects the additional Format menu. Screenshots are generated by
`examples.capture_diagram_tools`; no personal desktop state is used.

The initial CPU regression run completed 243 tests: 229 passed, 12 skipped, with
one newly exposed shutdown callback error and one outdated menu-list expectation.
The callback now guards deleted scenes, and the expectation includes Format.
Both affected cases passed on rerun. All 10 final focused editor, history,
persistence and validation tests passed. The manual
build checks 61 rendered pages. These are UI checks, not new numerical-accuracy
or CUDA-performance evidence.

Fit selection follow-up: the navigation, diagram-tools and startup/shutdown cases
all passed (3 tests). Coverage includes empty and wire-only selections, framing
selected blocks, zoom-label updates, busy-state navigation, and preservation of
model definitions, history and cached results. Ctrl+Shift+F is scoped to the canvas.


## Recent model navigation (2 November 2026)

File → Recent models keeps ten unique successful model opens/saves in local
desktop preferences. The tests cover ordering, persistence, duplicate filenames,
clear-list behaviour, missing-file keep/remove choices without file deletion,
failed save/open exclusion, dirty-cancel preservation of graph/results/history,
busy guards and state-disabled isolation. Recovery drafts and run archives do
not enter this list. All six focused recent-model, recovery, model-file, startup
and diagram-tool tests passed on CPU. The manual build checked 61 rendered pages;
this feature adds no numerical or GPU-performance claim. The state-isolated
diagram capture helper also records the recent-model menu using fixture filenames.

## Explainable graph and execution (2 November 2026)

Port hover descriptions explain the supported payload contracts. A selected wire
highlights its endpoints and displays its source, destination and dependency
meaning in the inspector. Blocks use two-line summaries and complete tooltips.
Settings search navigates using instance IDs and existing inspector controls,
including boundary coefficients, per-side samplers and flow component expressions.
It preserves model/results/history and handles removed instances and conditional
controls without changing options or expanding recipes.

Output badges and open-view labels distinguish unrun/disconnected, running,
evaluating, current, stopped/partial, failed, unavailable and cleared-by-edit
states. The CPU lifecycle case trains two updates, opens a monitor, changes the
physics, verifies curves are cleared, reopens a run archive and evaluates the
retained weights. Move/rename operations preserve availability. Badges remain
outside model definitions; they do not certify accuracy or unchanged external files.

The GUI Train/F5 action opens a run review; its Train new run action validates the
current graph before execution. Programmatic training and checkpoint continuation
retain their previous contracts. The plan uses initial sample builders with
explicit collocation selection, periodic partner rows and excluded gauge anchors.
Tests cover prepared triangle counts, missing boundary arrows, inspector-cache
independence, vector flow, first-window Expert counts and KdV. The summary builds
no network and performs no inference or optimizer updates.

This review exposed and fixed KdV's omitted boundary-PDE collocation: initial and
refreshed samples now use the shared row-selection function. Tests verify interior,
boundary and combined modes; interior-only behaviour is preserved. Mixed/boundary
KdV runs now follow their explicit arrows and retain normal strict checkpoint
identity checks rather than silently resuming against different coordinates.

The full CPU suite completed 247 tests: 235 passed and 12 skipped. A focused
19-test pass completed with 18 passed and one CUDA skip. The final three feature
cases, including native wire clicking and first-window review, passed again after
small navigation refinements. The manual validates 61 rendered pages. Four actual
UI screenshots and a capture manifest are generated by
`examples.capture_graph_explanations`; its two-update CPU fixture is not an
accuracy benchmark. No new GPU-performance claim is made.

## Notes, blank drafts and canvas organisation (2 November 2026)

Model format 29 adds plain-text annotation nodes with no ports. The annotation
case trains a two-update CPU fixture and verifies that note add/edit, note-only
copy/paste/cut/delete and undo/redo preserve the actual model and results. It
checks literal HTML escaping, live-editor synchronization, the 4096-character
bound, model saving/reopening, independent copied IDs and run-archive reopening.
The existing recovery case now includes an annotation and verifies its text after
restoration. Formats 1–28 remain readable; annotations themselves require 29.

Blank-model checks cover dirty-cancel preservation, busy guards, Ctrl+N, empty and
notes-only model saves and invalid/incomplete execution. Existing startup still
uses the Poisson template. Grid and dependency preferences persist in an isolated
local state directory without changing model definitions or history. Dependency
traversal covers branching/cycles, upstream/downstream modes and unchanged
selection/results. Native drag tests verify common-offset group snapping and one
undo step. An actual viewport context-menu event checks that selected groups keep
their selection and expose alignment/distribution actions with busy guards.

The CPU regression suite completed 250 tests: 238 passed and 12 skipped. A focused
16-test pass covered the new tools, library/context behaviour, document validation,
PDE collocation and previous graph explanations. All four final canvas/recovery
checks passed after the note-editor synchronization and bound checks were refined.
The user manual validates 61 rendered pages. Four actual UI screenshots and a
capture manifest are generated by `examples.capture_canvas_tools` without training,
inference or personal preferences. No new numerical-accuracy or GPU claim is made.


## Continuous-time inverse Burgers (2 November 2026)

This concrete inverse method jointly optimizes a replaceable solution network and
two scalar PDE coefficients using measured `u(x,t)` and residual MSE at the same
locations. Diffusion is represented by exp(log_lambda_2). There is no extra imposed
boundary/initial term and no reference answer in the objective. A pinned MIT-licensed
source manifest records the original dataset and deterministic 2,000-row selection.

Numerical checks verify the residual, analytical coefficient gradients, nonzero
coefficient updates, positive diffusion and identical same-device continuation.
They reject changed measurements/objective/guesses and corrupt coefficient history,
and verify rollback after an invalid optimizer moment. Prediction-only loading
restores the physical parameters. Model format 30 and old advanced-checkpoint
signature migrations preserve existing non-inverse workflows.

Desktop checks cover typed observation/PDE/trainer connections, compact/expanded
recipe equivalence, source fingerprints, copied data and project inventory,
model/run reopening, observed-loss monitoring, coefficient/history reporting,
inference-only evaluation, further optimizer updates and physical-time PVD export.
Additional import tests drive the actual observation mapping/copy dialog, verify
one-step undo/redo and independence from the deleted external original, and reject
missing bundled fixtures without replacing the current graph.

The final full CPU run completed 254 tests: 241 passed and 13 skipped. The final
10-test inverse/graph/device focus completed with 8 passed and 2 CUDA skips. Both
final inverse desktop cases, including the additional mapping case, passed
separately. The new CUDA-specific coefficient/resume test passed outside the
CPU-only run. Four-update CLI fixtures on CPU and CUDA each verified exact 2+2
versus 4 update weights, losses and parameter history; their values/configurations
are retained in assets/benchmarks/inverse-burgers-{cpu,cuda}.json. These checks do
not establish paper coefficient-recovery accuracy. The inverse guide records the
actual short-run values and implementation differences. Three screenshots use the
same explicit four-update CPU lifecycle, with local preferences disabled.

The manual validates 63 rendered pages. The Community methods page's duplicated
historical TODO material was replaced by a method/attribution index; detailed
retained source notices remain linked through THIRD_PARTY.md and the method guides.

## Research Python API

`tests/test_research.py` checks a real script executed outside the repository,
without GUI/MCP imports or block registration. It compares default updates
exactly with the existing baseline engine and custom loss/SGD updates directly
with PyTorch. It also covers custom PINN terms/update hooks, input ownership,
loss/gradient failures before an update, atomic checkpoint rejection/rollback,
and exact split continuation for Adam, AdamW, momentum SGD and Torch dropout RNG.
The CUDA case checks AdamW continuation and moving custom loss-module buffers.

`examples.research_custom` exercises a bare custom network, seeded Sobol sampler,
PDE loss and AdamW. Eight-update CPU and CUDA runs both matched split checkpoint
continuation exactly; a separate clipping-hook run completed. This is API and
lifecycle evidence, not a numerical accuracy study. See [Research API](research.md).

## Saved model comparison

`tests/test_model_comparison.py` covers stable IDs, nested settings/wires,
added/removed blocks, optional presentation, old-format defaults and unchanged
inputs. The native workflow compares two files and a current trained graph,
checks that weights/results/Undo and file bytes are preserved, exercises missing
and invalid files, and resolves deleted block IDs safely. Comparisons are model
metadata inspection, not numerical accuracy or external dataset equivalence.

## Independent block labels

`tests/test_block_labels.py` checks that renaming keeps types, IDs, parameters,
weights/results and freshness unchanged, with Undo/Redo and literal text. It
exercises automatic-name reset, explicit labels equal to automatic names,
recipe collapse/expand, copy/paste with distinct IDs, file/run round trips and
schema rejection. Draft recovery tests include labelled networks and notes.
Model format 31 preserves formats 1–30; labels are presentation data and model
comparison hides them unless presentation differences are requested.
