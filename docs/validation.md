# Validation scope

[Documentation index](README.md) · [Measured results](results.md)

## Latest targeted validation — 13 November 2026

The completed large-sketch DSGNAR changes pass **38 family tests: 33 passed,
5 CUDA-dependent skips**, plus **23 separate real-GPU cases without skips**.
Actual 4,000-square SVDs and complete same-checkpoint attempts are measured in
both precisions on CPU/CUDA. The GPU cases use the full point counts. See
[the complete qualification below](#full-size-dsgnar-profiling-and-probe-optimization)
for scope, memory measurements and numerical-parity evidence. This is targeted
coverage, not a fresh run of every unrelated test in the repository.

## Historical implementation suite

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

## Unused import cleanup

`tests/test_dataset_cleanup.py` verifies current/saved/cached and Undo/Redo/macro
reference protection, bounded-history eviction, Redo branch removal, stale reviews
and changed files. It checks that linked originals, geometry, unknown layouts,
symlinks and hard links remain untouched, and that inspection creates no owned
folders. The native workflow preserves model/history/result state during explicit
removal and refuses unknown Undo commands or stale project reviews.

## Selected scalar/vector error norms

`tests/test_error_norms.py` checks exact norm definitions, field selection,
zero-reference policies, overflow/underflow handling, strict shapes and detached
tensor inputs. Native tests verify read-only state, zero components, report
invalidation, block deletion and recomputation after a real run/archive reopen.
The affected outputs/reference/archive/freshness/benchmark/research checks pass;
a separate CUDA tensor probe confirms diagnostics do not change gradients.
These are diagnostic correctness/lifecycle checks, not paper accuracy results.

## Research update-count schedules

`tests/test_research_schedules.py` verifies update timing against manual PyTorch
StepLR updates, segmented training, per-group rates, compile resets and zero-update
calls. CPU and CUDA split runs retain exact network weights and loss/rate history
for Adam, momentum SGD and two-group AdamW with dropout RNG. Changed schedules,
corrupt rates/history, external rate edits, non-finite losses, zero-rate underflow
and custom-step composition are rejected. Reusing scheduled and unscheduled
in-memory checkpoints after further training leaves the supplied optimizer
tensors intact. All 18 research regression tests pass with CUDA available.

The runnable example's eight-update scheduled AdamW checks also pass on
[CPU](../assets/benchmarks/research-step-decay-cpu.json) and
[CUDA](../assets/benchmarks/research-step-decay-cuda.json). Both reproduce the split
run exactly, log a final-update rate of `0.000375`, and retain `0.0001875` for the
next update. These are lifecycle checks, not convergence or paper-accuracy claims.

## Crowded-diagram wire routing

`tests/test_wire_routing.py` covers deterministic clear/detour routes, backwards
and vertically separated ports, blocked escapes, invalid coordinates and the
64-block bound. Its native Qt workflows select a routed wire, drag the blocking
note, Undo/Redo its movement and deletion, exercise fallback explanations and
restore the preference from isolated desktop state. A real two-update CPU run
retains its trained model and result throughout the presentation changes; this
is lifecycle evidence, not an accuracy test.

`examples.capture_wire_routing` reproduces a heat graph with a note obscuring an
interior-to-PDE connection. The maintained before/after screenshots show the same
blocks and connections with only the view preference changed; all twelve wires
find clear routes in this fixture. The capture asserts unchanged serialized model
and Undo index, checks the painted stroke around the note, and records image
hashes. See [routing scope and screenshots](modeling/blocks.md#route-wires-around-blocks).

The affected canvas/navigation, dependency explanations, edit history, block-label
and saved-model checks are run alongside the routing tests. Default curved wires,
physics validation and model formats are preserved. Crossings/shared lanes remain
possible; neither implies a new connection. No CUDA numerical code changes.

## Stored physical-time result selection

`tests/test_result_times.py` checks unsorted coupled transient rows, duplicate IDs
and times, close-but-distinct float times, scalar predictions, reference/error
columns, original one-based row headers and honest full-data RMSE labels. It
checks a lazy 100,000-time model and the 200-row bound within a selected frame.
Steady tables hide the control; same-result refresh retains selection, new results
reset it, and invalidation releases stored arrays.

A real two-update CPU Burgers workflow filters an existing prediction while a
guard rejects any inference call. Model serialization, Undo index, result identity
and optimizer phase stay unchanged. CSV bytes match before/after selection; PVD
still contains all nine physical times. An archive reopens with complete data and
All stored rows selected, then supports filtering offline. The screenshot capture
uses this same short-run setup and records its hash; it is a lifecycle example,
not an accuracy benchmark. No numerical CUDA implementation changes.

## Retained adaptive point inspection

`tests/test_sample_points.py` checks page order/bounds, detached copies, RAR's
initial anchor positions, importance's trailing anchors, actual correction
weights, warm-up unit weights and invalid/non-finite data rejection. Native
RAR/importance runs compare every displayed coordinate/weight against the actual
retained tensors. Inspection is guarded against residual scoring, prediction and
resampling; model, Undo index, result identity, loss history and weights stay
unchanged. The same page reopens from an archive without sample generation.
Busy/editor state refuses reads; renaming preserves the snapshot, semantic edits
clear it, and deleting the owner closes its dialog.

The CUDA case checks that only a detached 200-row page is copied for display,
leaving the original CUDA tensor and gradients intact. The maintained capture
shows page two of a real 214-row RAR set after five CPU updates, including two
fixed boundary PDE anchors and six additions. This verifies retention and UI
lifecycle, not numerical accuracy.

## CUDA regression audit — 11 November 2026

The retained [CUDA audit](../assets/benchmarks/cuda-validation-2026-11-11.json)
passes **18 targeted checks with no skips**, using the existing root environment:
Python **3.14.8**, Torch **2.14.1+cu130**, compiled CUDA runtime **13.0**, NVIDIA
GeForce RTX 3060 Laptop GPU. No environment/dependency changes were made.

Coverage includes CPU/GPU loss agreement, automatic device/data placement and
Poisson training, GPU benchmark artifacts, per-boundary weighting across trainers,
Expert training and NTK/time marching, Scale-PINN, gPINN with RAR, RBA, inverse
Burgers coefficients, research components, schedules and scalar integral constraints, advisory CUDA memory queries, isolated CUDA checkpoint jobs, detached sampler-page
inspection, application-level training/inference/resume device switching and
portable copied-data workflows. These are the existing numerical and lifecycle
regressions; this does not claim a full-suite run or paper-accuracy reproduction.

The audit exposed a stale **test fixture** in the gPINN CUDA case: reducing the
common boundary budget had left paper-preset per-set overrides intact, making
the initial collocation set exceed its deliberately small RAR cap. The numerical
validator correctly rejected that configuration. The fixture now explicitly clears
those overrides and asserts two actual RAR events; all six gPINN tests pass,
including the CUDA derivative/checkpoint case. No solver validation was relaxed.

Reproduce with `examples.validate_cuda`; see [commands](commands.md#audit-actual-cuda-execution).
The saved JSON includes outcomes and wall times for individual cases, runtime/GPU
metadata, a base Git commit and SHA-256 fingerprints of numerical/UI source,
selected test sources and the runner. The fingerprints identify tested bytes even
when the audit is recorded before its containing commit. Tracebacks remain local
and do not enter the public JSON.

Five audit-runner checks pass separately. They verify ordinary success, skips,
expected/unexpected failures, skipped/failed subtests, empty suites, explicit selection
of the desktop GPU case, exclusion of its mocked-unavailable-GPU case, and the
no-CUDA CLI path. A missing GPU returns a non-successful unavailable report rather
than a green skipped suite; existing report files are preserved without
`--overwrite`. The old 27 October CPU-only prerequisite in TODO is now closed;
the older dated validation records remain historical.

## Research scalar integral constraints

`tests/test_integral_constraints.py` checks physical non-unit quadrature against
manual losses and gradients, cancellation before squaring, a scalar derivative
operator with nonzero parameter gradient, raw/weighted history and custom loss
inputs. Integral squared-error mode does not apply cross-section measure twice.
Invalid weights/targets and unsupported desktop/base trainers are rejected; a
vector integrand requires explicit field/flux selection. Compile snapshots preserve
source isolation and Python-list precision in the network dtype.

Fixed quadrature coordinates, weights and targets participate in continuation
identity, including changed same-total-measure weights. CPU and CUDA split runs
match uninterrupted weights/history exactly with step decay and restricted tensor
checkpoint loading. Existing point-only research and standard PINN behavior are
covered by the affected loss/training/sampling regression checks.

The ten-update scalar axial-flow example has [CPU](../assets/benchmarks/integral-constraint-cpu.json)
and [CUDA](../assets/benchmarks/integral-constraint-cuda.json) records: both retain
exact continuation, and report the actual post-update flow separately from
pre-update losses. The example is deliberately a parabolic ansatz/continuity
workflow, not a full flow solver or accuracy reproduction.

## Resource review and memory-allocation failures

`tests/test_resources.py` compares arithmetic parameter/buffer counts with actual
small instances of all supported networks over 1D/2D/space-time dimensions, plus
RWF, full/diagonal/Gaussian Fourier, affine expansion and inverse coefficients.
It flags a large full-Fourier point/feature combination without allocating its
feature array. It covers aggregate output/candidate/RAR/monitor counts, unknown
or zero free memory, and honest reference-scale captions.

An actual trained GUI retains its graph, Undo index, result, model and RNG while
a low-memory fixture produces an advisory warning with Train still enabled.
Injected host/Torch allocator failures exercise construction, checkpoint restore,
training and output stages, clear failed worker references and permit a subsequent
real successful run. Non-memory errors retain their original diagnostic. No real
RAM/VRAM exhaustion is attempted. Existing run/device/evaluation/automation
checks are run alongside this workflow.

The CUDA query case reads real driver availability and verifies unchanged Torch
CPU/CUDA RNG and allocated tensor bytes. The maintained screenshot explicitly
labels simulated host availability and is a UI illustration, not a capacity or
accuracy benchmark. [Resource guide](use/resources.md) documents the remaining
quota and probe limitations.

## Isolated desktop numerical jobs

`tests/test_isolated_jobs.py` verifies actual distinct child processes with no Qt
imports, direct-engine equivalence, parent RNG preservation, CPU retained tensors,
cooperative cancellation, forced discard, abnormal/native exits, malformed
events, bounded atomic files, rejection of arbitrary reference callbacks and
parent control-pipe closure. It checks late force-stop during result transfer and
that GUI controls recover so a subsequent run can succeed. CPU/CUDA continuation
uses fresh child processes and compares exact histories/weights after a split run.

The desktop regression paths include baseline/SA/LRA, Expert and sequential
windows, Scale, gPINN, RBA and inverse workflows, archives, supplied/owned data,
references, inspectors, device switching and automation. Tests that injected
GUI-local numerical mocks now inject at the supervisor boundary; comparisons with
private retained network states explicitly compare CPU tensors. Ordinary research
trainers and library numerical APIs remain direct PyTorch objects.

Maintained [CPU](../assets/benchmarks/isolated-worker-cpu.json) and
[CUDA](../assets/benchmarks/isolated-worker-cuda.json) records establish separate
headless execution, re-evaluation with unchanged weights, CPU storage, archive
reopening and exact three-plus-two-update continuation against an uninterrupted
run. The force-stop screenshot is an actual menu from a deliberately discarded
small job. No real memory exhaustion or numerical-accuracy reproduction is claimed.

The full CPU/offscreen discovery ran **338 cases**: 318 passed, 19 CUDA-dependent
cases skipped and one Jorvik workflow check failed because it expected a live
operator on the new retained-state handle. That check was corrected to construct
the physical operators from configuration; its two-case module then passed.
The final isolated-job module passed seven cases, including actual CUDA
continuation, and the refreshed CUDA audit passed 18 cases without skips.

## Post-isolation reliability pass

On 2026-11-11, a fresh complete CPU/offscreen discovery finished cleanly:
**338 tests, 319 passed and 19 CUDA-dependent skips** (753.329 seconds).
This supersedes the earlier full-run failure described above; no further runtime
changes were needed. The separate 18-case CUDA audit remains applicable to the
unchanged runtime and audited test sources.

The maintained Jorvik annulus saved-project workflow was also rerun independently
on **CPU and CUDA**. Both passed project save/reopen, cooperative stop after 50
updates, archive reopening, two-update continuation with exact uninterrupted-run
weights/history, evaluation without changing weights, project relocation and
explicit VTU export. Export checks preserved the three supplied coordinates,
point IDs and predicted values. The optional VTK reader was unavailable, so these
runs verified XML contents directly rather than claiming reader interoperability.
The existing [CPU](../assets/benchmarks/jorvik-workflow-cpu.json) and
[CUDA](../assets/benchmarks/jorvik-workflow-cuda.json) records remain unchanged
because the reruns produced identical results. These are lifecycle/correctness
checks, not numerical-accuracy benchmarks.

## DSGNAR implementation and accuracy checks

The independent fixed-point scalar trainer includes CountSketch/SRCT, SVD
trust-region steps, adaptive condition coefficients, stage switching, explicit
precision and strict data/state validation. A faulty inverse interpolation at
repeated-ratio plateaus was found during accuracy work and replaced by a forward
monotone PCHIP crossing. Tests compare its roots against SciPy independently.
Zero-grace defaults preserve earlier format-32 models/checkpoints; the reference
training schedule's 80-attempt grace is an explicit option.

The targeted method/reference/desktop selection passes **20 tests**, including
actual CPU/CUDA training, analytic parameter Jacobians, DCT/adjoint identities,
CountSketch collisions, singular systems, failed/rejected-step rollback,
float32/float64 continuation, old-state migration, isolated workers, optional
views, archives and exports. The [real GPU audit](../assets/benchmarks/cuda-validation-2026-11-12.json)
contains 20 executed passes without skips and source/runtime fingerprints.
The final complete discovery ran **358 tests: 337 passed and 21 CUDA-dependent
checks skipped**, covering the interpolation and migration changes. The separate
GPU audit above executes the maintained CUDA selection without skips.

Maintained [CPU](../assets/benchmarks/dsgnar-cpu.json) and
[CUDA](../assets/benchmarks/dsgnar-cuda.json) native workflows compare exact
four-plus-two versus uninterrupted fresh-process network weights, loss histories
and sketch RNG. Last-bit SVD regularization diagnostics use `5e-14` relative
tolerance; Qt-host direct-engine parity is separate. Tensor dtype is preserved
by the shared Laplacian, and baseline/trial objective evaluation uses the same
batching. Fixed target functions are evaluated canonically on CPU before transfer.

### Accuracy and checkpoint reloads

The [Poisson analytical validation](../assets/benchmarks/dsgnar-poisson-accuracy.json) reaches **5.512e-09 relative L2** on 1,001 evaluation points after 74 attempts. Its native project and trained run archive under `build/dsgnar-accuracy/` are saved and reopened successfully.

The [corrected periodic Burgers run](../assets/benchmarks/dsgnar-burgers-forward-probes.json) reaches **2.558e-07 relative L2 in float32**, with 1,447 parameters, sketch 700 and the full 32,768 PDE / 16,384 initial rows. It first meets the paper's `4.75e-7` threshold at attempt 123 and stops at 127 attempts after 624.4 seconds.

An independently generated Cole–Hopf reference is checked against initial and
periodic conditions, the PDE, and doubled quadrature resolution. The uniform
257×65 evaluation grid differs from the authors' protocol. This is a numerical
threshold match, **not** a full protocol or H100-speed reproduction. The paper's
9.8-second runtime is not matched by this portable PyTorch port on the RTX 3060.
Earlier unsuccessful prototype records are retained and marked as superseded
when they used the incorrect inverse interpolation.

The [retained checkpoint](../assets/benchmarks/dsgnar-burgers-target-checkpoint.json)
is reloaded in fresh [GPU](../assets/benchmarks/dsgnar-burgers-reloaded-cuda.json)
and [CPU](../assets/benchmarks/dsgnar-burgers-reloaded-cpu.json) processes without
updates. Their relative errors are `2.558e-7` and `2.548e-7`; weights, iteration and
sketch RNG remain unchanged. Stage two begins at attempt 102, and complete replay
verifies that an 80-attempt grace would not alter this run's decisions.
The [compiled derivative check](../assets/benchmarks/dsgnar-functional-validation.json)
compares paper-sized 1,447-parameter CUDA PDE/IC Jacobians and probe residuals
against generic autograd. Backend/compiler identity is explicit for benchmark
continuation; weights-only warm starts are labelled separately.

## DSGNAR smooth-lid cavity

The new **Lid-driven cavity · 2D · smooth lid · Re=100** UI example uses actual
incompressible u/v/p physics. CPU/GPU checks cover isolated training, exact
fresh-process 2+1 versus uninterrupted continuation, recipe transformations,
model/run save/open, evaluation and CSV/steady VTI export. Host-process numerical
parity uses a separate double-precision tolerance, as in the existing workflow.

The independent research problem follows Appendix C.7's five scalar conditions
without a pressure gauge. Analytic polynomial/Couette checks verify advection,
viscous and pressure terms, divergence, pressure-offset invariance and the smooth
lid. Tests compare functional/generic parameter Jacobians including remainder
batches, plus exact checkpoint continuation. The independent finite-difference
reference solver checks its discrete Poisson normalization/sign and steady flow.
The [real-GPU 60×4 Jacobian/reference-refinement record](../assets/benchmarks/dsgnar-cavity-jacobian-cuda.json)
contains explicit measured discrepancies.

The [training record](../assets/benchmarks/dsgnar-cavity-pilot.json) retains a
115-attempt, 1,350-parameter float64 solve. Fresh CPU/GPU reloads give **0.247%**
relative velocity difference against an independent 257×257 reference. Reference
resolution changes matter (129 versus 257 differs by **0.152%**); this does not
match the paper's full training/evaluation protocol or reported accuracy. See
[the cavity guide](dsgnar.md#smooth-lid-cavity-re100) for exact settings and commands.

Final cavity regression discovery ran **361 tests: 340 passed, 21 CUDA-dependent
checks skipped** (777.5 seconds). The separate real-GPU audit passed **all 20
selected tests without skips**, including the expanded cavity desktop workflow.
The optional compiled/eager cavity smoke comparison also passes; compiler
specializations are locally bounded and restored. The complete manual builds
and checks **71 rendered pages** without broken local links or images.

## DSGNAR standing wave and multiscale Poisson

Three analytical-method regressions cover both exact PDE identities, separate
wave displacement/velocity losses, no final-time boundary, constant-folded
Poisson forcing, NumPy/Torch reference agreement, functional/generic parameter
Jacobians, exact continuation and changed-data rejection. Both Fourier and SIREN
architecture choices are asserted explicitly. The real-GPU parameter-Jacobian
audit checks 609- and 1,681-parameter networks, including remainder rows.

Native lifecycle checks include both new examples, compact/expanded recipes,
model/run persistence, isolated training, exact fresh-process continuation,
analytical comparisons, CSV and physical-time PVD for the wave. The wave execution
summary labels its speed and separate initial-velocity condition; the initial
geometric set is added to wired PDE points once.

The [analytical training guide](dsgnar-analytic-examples.md) reports **0.0475%**
relative L2 for multiscale Poisson and **24.3%** for the selected single-window
wave attempt. The latter remains an inaccurate solve. Two failed Fourier wave
experiments are retained, including adaptive suppression of the displacement
condition. Independent CPU/GPU checkpoint inference preserves weights, iteration
and sketch RNG. No paper accuracy or H100-speed reproduction is claimed.

The 364-test CPU discovery found one preset persistence mismatch (342 passed,
21 CUDA-dependent skips): a tuple of Fourier frequencies became a JSON list on
reload. The preset now uses JSON-native lists. The entire affected three-test
example module passes after the correction, including its all-example round-trip
and manufactured-solution training checks. The analytical-method selection also
passes its three tests after the final architecture-selection changes. The final
real-GPU audit passes **20 selected tests without skips**, with a matching runtime
source fingerprint. The expanded DSGNAR selection passed **26 tests**, and the
manual builds/checks **72 rendered pages**. No unresolved test failure remains;
the full discovery was not repeated after this targeted serialization correction.

## Isolated DSGNAR profiling

Three profiler regressions pass: complete CPU-attempt checkpoint parity; scoped
method/numerical-function cleanup and unchanged state after an injected failure;
and fresh subprocess/report behavior, including a real unavailable-CUDA worker
failure that preserves an existing report. Timing instrumentation does not belong
in a live GUI job.

The maintained CPU and CUDA profiles cover **16 configurations** across two
existing cases and batches 16,64,128,256. Each includes a complete post-warmup
continuation and an uninstrumented same-checkpoint control. Trained state/losses/
private RNG match exactly; regularization diagnostics alone retain the existing
`5e-14` tolerance. Global CPU/CUDA RNG remains unchanged. CUDA allocator peaks
and process-lifetime RSS have explicit, different scopes. No production numerical
code, device defaults or batch settings changed. See
[phase/workspace evidence](dsgnar-profiling.md) for interpretation and limitations.

## CountSketch occupied-bucket CPU optimization

The production CPU path omits empty-bucket reductions; production CUDA retains
its previous dense path after an occupied-row prototype mostly regressed the
small measured sketches. Sorted within-hash summation, normalization and RNG
draws are unchanged. Exact float32/float64 matrix/vector checks cover collisions,
empty batches and initialized accumulators on CPU and actual CUDA.

The DSGNAR family selection runs **33 tests: 30 pass and 3 CUDA-dependent checks
skip**. The separate real-GPU audit passes **21 tests without skips**, including
the new reference kernel check and all native recipe/checkpoint workflows. The
paired old/production full-attempt records cover **16 CPU/CUDA continuations**
with exact trained state/loss/RNG parity; regularization diagnostics alone use
the existing `5e-14` tolerance. Comparison hooks restore on injected failure.

The 700- and 4,000-row accumulation-only measurements also match old sums exactly.
Their collision pattern, resident outputs and excluded SRCT/Jacobian/SVD work
are explicit. Kernel ratios are not whole-trainer speedups; short end-to-end
comparisons are noisy and include a slower CPU sample. Larger paper SVD and GPU
fusion remain future work. The manual builds/checks **73 pages** successfully.

## Full-size DSGNAR profiling and probe optimization

Actual **4,000×4,000 SVDs** are measured inside complete CPU/CUDA attempts in
float32 and float64, with 11,221-parameter networks. GPU uses the full 32,768 PDE
and 16,384 condition rows; CPU uses reduced 4,096/1,024 counts, stated explicitly.
Both same-checkpoint continuation and global-RNG checks pass. SVD phase allocator
resets preserve an explicit running whole-attempt maximum; phase growth includes
outputs/workspace, not all CUDA driver allocation. Source changes during a long
profile reject publication without overwriting an earlier report.

Full-count occupied-row GPU qualification passes exact trained-state/loss/RNG
comparisons in both precisions. Its measured old/new elapsed ratios are 1.088×
and 1.122×; the actual production-dispatch double-precision comparison also
passes. CUDA remains dense below 4,000, with 3,999/4,000/4,096 boundary coverage.

PCHIP coefficient caching preserves the original floating-point operation order
and 60 bisection steps. Exact uncached-reference radii cover increasing/plateau/
negative-clipped curves and fast paths; the independent SciPy oracle remains.
CPU/GPU full continuations match the uncached reference. Neither optimization
changes the mathematical objective, sampling, precision defaults or checkpoint
format. See [complete timing/workspace evidence](dsgnar-profiling.md#completed-4000-row-svd-and-full-attempt-profiling).

Final DSGNAR regression selection: **38 tests, 33 passed and 5 CUDA-dependent
skips**. The separate [13 November GPU audit](../assets/benchmarks/cuda-validation-2026-11-13.json)
executes **23 passes without skips**, including the new SVD-peak test and dispatch
boundary. Report fingerprints match the measured source. All five final large
profile configurations pass complete checkpoint/RNG controls; paired probe and
large-accumulator comparisons also pass. The **73-page manual** builds and checks
without broken local links/images.

The five audit-reporting/skip/failure-handling regressions also pass after
extending the CUDA manifest. No full unrelated-suite rerun was used to inflate
this targeted qualification.

## Bounded native inference

`tests/test_prediction_batches.py` has eight focused checks, all passed with real
CUDA available. They cover every native network factory on CPU/CUDA, checkpoint
predictors, preserved network mode/weights/RNG, duplicate coordinates and text IDs,
float64 coordinate/reference precision, multi-window physical-time order and
out-of-horizon rejection. A real isolated coupled-flow job evaluates 5,001 rows
through both output and raw prediction paths. Invalid points, inconsistent batch
fields and a late nonfinite batch fail without publishing partial results.
A batch-dependent research model remains one-shot by default.

The standalone 50,000-row float64 measurement uses a 128×6 FCNN with 83,073
parameters. It verifies 25 forwards bounded to 2,048 rows, CPU output assembly,
checkpoint/RNG preservation and prediction parity. Recorded RTX 3060 Laptop
CUDA allocated peaks are 172,905,984 bytes one-shot and 24,062,464 bytes batched;
maximum observed value difference is zero. These are short inference measurements,
not training-memory bounds, numerical accuracy evidence or a general promise of
bitwise equality. Full host/IPC storage still grows with requested output rows.
[CPU record](../assets/benchmarks/prediction-batches-cpu.json),
[CUDA record](../assets/benchmarks/prediction-batches-cuda.json),
[user guide](use/isolated-jobs.md#bounded-prediction-batches).

The [targeted CUDA audit](../assets/benchmarks/cuda-validation-prediction-batches-2026-11-13.json)
passes all 24 selected checks with no skips, including the added network-batching
case. Its runtime/test hash matches this implementation and regression sources.
The documentation build checks 73 rendered pages and their local links/assets.

The complete CPU-mode regression run completed 384 tests: 359 passed and 25 were
skipped (`CUDA_VISIBLE_DEVICES=''`, offscreen Qt). The separate real-CUDA audit
above supplies GPU coverage; the eight focused batching checks also passed in a
CUDA-enabled run. No test failures occurred.

## Independent geometry clipboard snapshots

`tests/test_clipboard_files.py` passes six CPU checks. A real offscreen desktop
workflow creates an owned Jorvik document and unequal-size NetCDF samples, applies
them, copies the geometry/interior/boundary subgraph and closes the source window.
Pasting into another project preserves both internal wires and all sample rows;
the document, adjacent automatic export and applied snapshot are independent.
Repeated pastes survive edits to previous copies. Cut/Undo/Redo and model save/load
retain the correct files. Blank geometry copying stays lazy; missing applied
geometry documents fail rather than becoming empty copies.

Additional checks cover active mapping/fingerprint rebasing without changing
literal labels, mismatched/missing/extra/traversal payloads, source and destination
symlinks, oversized files and an injected disk failure with rollback. A rejected
paste leaves the graph, history and owned-file set unchanged. No numerical training
or geometry editor is needed by these new tests.

The related CPU-mode clipboard, Jorvik exchange/workflow, canvas/labels,
project-data cleanup/storage, model-file and run-archive suite completed 31 tests:
30 passed and one CUDA-only test was skipped. The finalized six-case clipboard
suite was run separately afterward and passed. The manual build checks 73 rendered
pages and their local navigation/assets. This is file/desktop lifecycle validation,
not a new numerical or GPU accuracy claim.
