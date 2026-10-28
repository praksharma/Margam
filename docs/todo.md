# Current TODO

See roadmap.md for the agreed design and stages. Completed historical DeepINN
checkboxes are not claims about current Margam support.

Audit on 27 October 2026: the current implementation items below are addressed.
The remaining unchecked entries are explicitly deferred scope or external
validation prerequisites; they are not presented as completed work. Current
execution is one supported problem root, not arbitrary multi-model simulation.


- [x] Catalogue implemented methods, sampling and optimisation components with
  API/UI support and limitations in components.md.
- [x] Reuse GeoSampler Hammersley and recursive stratified unit-domain samplers,
  with exact counts, seeded shifts/strata and physical interval/rectangle bounds.
  Existing grid/random sequences are preserved; SciPy-only methods are not exposed.
- [x] Add built-in intervals, squares and rectangles with grid/random meshless sampling.
- [x] Create named rectangle-side point sets, then assign constant Dirichlet/Neumann BCs.
- [x] Connect PDE points to the Poisson residual preset; defer additional pointwise inputs.
- [x] Expose restricted scalar PDE expressions in the UI, as selected by the user.
  Coordinates, u, c, arithmetic, derivatives and basic functions are supported.
- [x] Extend the field contract for the user-selected steady 2D incompressible
  case: u/v/p, editable momentum/continuity residuals, component BCs, pressure
  gauge, explicit references, table views and multi-column CSV/run persistence.
- [x] Add constant Dirichlet/Neumann settings, analytical normals and an explicit rectangle corner policy.
- [x] Add restricted custom scalar BC operators (including Robin expressions)
  and coordinate-dependent targets. Retain a Dirichlet anchor requirement.
- [x] Add CSV/NPZ point adapters with explicit coordinate, unit-normal and scalar
  value mappings and row previews. Interior imports are coordinates only; imported
  boundary normals/targets are supported. NetCDF/other formats need adapters.
- [x] Define typed editor ports and reject incompatible/occupied inputs before
  adding wires. Validate stored wire endpoints again before training.
- [x] Define executable scalar graph semantics: follow one PINN root's typed
  residual/BC/network dependencies, require a common geometry source and assemble
  the training recipe from owned block parameters rather than the inspector cache.
  Multiple roots/coupled systems remain an explicit future extension.
- [x] Add block/wire deletion and fit-to-model navigation.
- [x] Add versioned model Save / Save As / Open, including graph/configuration,
  block IDs/settings/positions, named sampling recipes, atomic replacement,
  validated loading, dirty state and Save/Discard/Cancel protection.
- [x] Add explicit saved-run archives with model definitions, requested predictions,
  monitor records, network/Adam state, validated CPU loading and resume updates.
  Model JSON files remain definitions only; run archives use .margam-run.pt.
- [x] Add bounded model undo/redo for add/move/connect/delete and settings.
- [x] Connect the baseline engine through background training with progress and cancellation.
- [x] Add a runnable Poisson visual template and verify the analytical interval solution.
- [x] Reuse GeoSampler-style Qt layout/settings and per-session lock-file recovery
  patterns. Save core dock layout/theme/library groups; recover unsaved model
  drafts without reopening plot views or loading numerical state.
- [x] Add an explicit planar GeoSampler schema-v2 NPZ bridge with label/axis
  selection, source provenance, unit-normal checks and no CAD runtime/viewport.
- [ ] Deferred: full 3D/non-planar GeoSampler integration and richer region schemas.
- [ ] External prerequisite: validate GPU training when a working NVIDIA driver
  and CUDA-enabled PyTorch build are available. Audit on 27 October: installed
  torch 2.14.1+cpu, torch.version.cuda=None, CUDA unavailable; nvidia-smi cannot
  communicate with the driver. CPU checks do not satisfy this item.

## Menus and explicit outputs

- [x] Add File / Edit / View / Run / Help menus for implemented actions. File
  offers model Open/Save/Save As, template actions and Quit.
- [x] Allow training without any plot/results block.
- [x] Make loss plots and prediction plots explicit optional output blocks, with
  per-instance history/evaluation settings.
- [x] Create/show output views only when explicitly opened through their blocks.
  Unconnected blocks can show a placeholder; only connected outputs request data.
- [x] Separate reference sources and comparisons from ordinary predictions; the
  current sources cover analytical intervals and equal-Dirichlet rectangular
  truncated-series references.
- [x] Remove automatic analytical-reference construction from the generic worker.
- [x] Keep operational status, progress, validation and failures available without
  numerical output blocks.
- [x] Add Training Monitor blocks with bounded recorded histories.
- [x] Support bottom docking, floating, hide/reopen, and removal with the owning
  block. Keep view visibility separate from data retention.
- [x] Add explicit imported scalar reference data and restricted reference
  expressions; matching evaluation coordinates required, no implicit interpolation.
- [x] Add a dedicated scalar/u-v-p CSV Export request block; files are written only on
  explicit export after training. The separate opt-in ParaView Export block now supports regular-grid VTI output; richer integration remains deferred.

## Left library and sampling options

- [x] Replace the flat left-hand list with a searchable, collapsible component
  library: Geometry, Sampling, Neural networks, Loss / Physics, PINN recipes, Training,
  and Results. Template actions remain separate in File/the toolbar. Search matches
  block/category names and restores collapsed groups when cleared. Only implemented
  blocks are listed; Enter and double-click add a block.
- [x] Group sampling by purpose: interior/PDE points and boundary point sets.
  Within each, show implemented methods and their supported settings. Sampling
  method is distinct from the target region; assigning a BC is a later operation.
- [x] Keep basic built-in interval/rectangle sampling in Margam: grid and seeded
  uniform random, domain/side selection, point budget or points-per-side, seed,
  actual count and preview. A simple problem must not require GeoSampler.
- [x] Let the template interior and each named rectangle side use independent
  sampling recipes, methods, counts and seeds. Intervals retain fixed endpoints. Arbitrary additional sampler
  instances still depend on future graph execution work.
- [x] Share implemented block status, parameters, compatible ports and limits
  across the library, inspector and Help. Property sampler choices come from the
  implemented sampler catalogue. Keep widgets explicit; no generic plugin registry.
- [x] Define the GeoSampler ownership/import boundary in geosampler-contract.md;
  keep CAD authoring/selection outside Margam. The planar bridge is implemented.
- [x] Keep library, ports, conditions, data imports and output requests expressed
  as scalar fields/operators, independent of temperature. Heat remains a preset;
  vector/coupled schemas are an explicit future extension.

## Simulink interaction evidence

- [x] Collect official, release-labelled documentation for library insertion,
  parameter editing, wiring, navigation, outputs, diagnostics and persistence.
  See [simulink-evidence.md](simulink-evidence.md) for sources, explicit Margam
  adaptations, implementation order and acceptance checks.
- [x] Add library drag/drop and searchable canvas quick insert, sharing the
  supported UI catalogue and existing block creation path. Placement respects
  pan/zoom; empty-canvas double-click and canvas-only Tab open quick insert.
  This does not generalise the template-backed core execution model.
- [x] Define contextual double-click behaviour: core/reference blocks use the
  shared inspector; output blocks open/reuse their owned views. No separate
  parameter-dialog state is introduced.
- [x] Store core parameters on their block instances, including independent network
  initialisation seeds; editing/copying one instance does not overwrite another.
- [x] Support output branching from one method/reference to distinct consumers,
  with one source per typed input and no implicit view creation.
- [x] Add undoable graph/parameter edits, including grouped drag moves, restored
  block IDs/settings/wires, per-editor value coalescing and a 100-command limit.
  Training locks history. Template changes start fresh; run data/views are excluded.
- [ ] External prerequisite: validate interaction parity hands-on against a
  licensed Simulink session. No MATLAB executable was found on PATH on 27 October;
  current evidence and native Qt tests do not constitute a licensed live comparison.

## Output/menu milestone (26 October 2026)

The first output-ownership stage in simulink-evidence.md is implemented and
verified. See ui-prototype.md for actual menu actions, output lifecycle and limits.
The searchable, collapsible library and independent rectangle-side sampling
recipes are implemented. The component catalogue records implemented capabilities;
supported catalogue descriptions are shared by the library, inspector and Help.

## Block-owned rectangle previews and context menus

- [x] Add block-specific right-click menus for Properties, Delete and Help, plus
  only the view actions each supported block actually provides.
- [x] Remove the default domain/sample figure pane; keep operational Messages
  separate and initially hidden.
- [x] Add optional Rectangle → Preview geometry / Preview samples windows, floating
  by default and owned by that rectangle block.
- [x] Reuse hidden previews and update them on settings changes; remove them on
  owner deletion, shape change or template reset.
- [x] Keep these sample previews limited to rectangles. Do not create a general
  geometry renderer or implicit 3D preview path without a justified requirement.
- [x] Add Undo/Redo actions to block context menus.
- [x] Add undoable block Copy/Cut/Paste with fresh IDs, internal wires and
  canvas-scoped shortcuts. Pasted core settings remain template-backed.

- [x] Add pointer-centred wheel zoom, mouse/Space panning, zoom controls and fit-to-model.

## Heat demo and navigation milestone

- [x] Add a separately loadable steady heat demo with monitor, prediction,
  Reference and Comparison blocks fully connected.
- [x] Add a labelled rectangular sine-series reference and verify it against
  finite differences and increased truncation order.
- [x] Show 2D prediction/reference/error with shared prediction/reference colours,
  RMSE and maximum absolute error only when comparison is explicitly requested.
- [x] Quantify rectangular training error for three seeds and two soft BC weights,
  and add/verify an optional hard-Dirichlet method that improves field error in
  every tested seed. See heat-accuracy.md; no field plots were changed.

## ParaView file export

- [x] Add a separate opt-in ParaView Export block for regular-grid predicted fields;
  retain CSV Export and never create files or launch viewers by default.
- [x] Write uniform-grid VTI scalar/u-v-p data and a derived velocity vector.
- [x] Support physical-time PVD collections and per-snapshot TimeValue metadata
  in the writer API, with a known-value playback dataset and real VTK reader checks.
- [ ] Deferred: actual transient PINN solver/UI integration and physical-time selection.
- [ ] Deferred: embedding/launching ParaView and direct viewer integration.

## Future evaluation and export at supplied points

- [ ] Let users choose evaluation locations from GeoSampler/prepared input data
  or supplied FEM mesh nodes, alongside the existing regular export grid. These
  are evaluation points and need not be the PINN training samples.
- [ ] Evaluate predictions at those exact spatial coordinates. Preserve original
  point/node IDs, ordering, coordinate units, boundary/region labels and provenance,
  so results can be matched back to the source without nearest-neighbour guessing.
- [ ] Preserve supplied mesh connectivity and node-versus-cell field association
  when available. Do not invent connectivity for a point cloud or silently export
  scattered nodes as uniform ImageData. Select an appropriate VTK representation
  (e.g. PolyData for points or UnstructuredGrid when cells are supplied).
- [ ] Accept optional reference/FEM values at the same nodes, with explicit field
  mapping and units, and export prediction, reference and pointwise error arrays.
  Do not require an exact solution. Define scalar/component errors and optional
  vector norms; handle absent references and zero denominators explicitly.
- [ ] Preserve physical times and point IDs across transient snapshots; support
  ParaView playback of the supplied-node or mesh results. Never use optimiser
  iterations as physical timesteps.
- [ ] Keep this broader input → evaluation → export system a separate implementation
  milestone. Current VTI exports still evaluate on regular line/rectangle grids.

Existing in-app heatmaps are left as they are. Further field-plot development is
paused; operational training monitors and basic rectangle previews remain useful.

- [x] Add a toolbar/View Pan toggle with open/closed hand cursors and Escape to
  return to block selection. Existing temporary pan gestures remain supported.

## Problem catalogue and PDE presets

- [x] Group all runnable desktop problem templates under File → Examples.
- [x] Retain editable PDE residual and add alternative Predefined PDE block.
- [x] Add manufactured interval Poisson, reaction–diffusion,
  advection–diffusion and 2D sine Poisson examples with explicit references.
- [x] Check manufactured equations, CPU training, graph assembly and model format.

- [x] Add palette-aware vector icons to the toolbar, menus and collapsible block
  library; retain labels and keep demo shortcuts in File → Examples.

## PINN recipes

- [x] Define the baseline recipe in one file with shared components, exposed
  settings, ports and internal connections.
- [x] Insert compact or expanded; expand by double-click/context action and
  collapse a baseline trainer with its private network.
- [x] Preserve network/training parameters, external connections, history,
  clipboard and saved-model compatibility; verify actual training equivalence.

## Self-adaptive PINN

- [x] Read the supplied McClenny/Braga-Neto paper and verify the pointwise mask
  objective and ascent/descent updates against primary sources.
- [x] Add a separate method and one-file compact/expanded SA recipe; fixed samples,
  squared mask, configurable ascent rate/initial weight/cap, optional adaptive BCs.
- [x] Add a runnable File → Examples problem, explicit weight summary, full adaptive
  checkpoint/resume, equation/gradient checks and an equal-budget CPU comparison.
- [ ] Reproduce a stiff benchmark protocol from the paper before claiming an
  advantage there; L-BFGS and GP-resampled minibatches remain unsupported.

## Documentation and evidence

- [x] Add a documentation index and results catalogue; separate measured studies,
  regression thresholds, historical records and unreproduced protocols.
- [x] Save the repeated SA comparison with settings, environment, source hashes,
  full-precision errors, final losses and weight summaries.
- [x] Preserve historical GPU metadata separately from current CPU validation.
- [x] Consolidate setup/run commands, update menu/persistence/recipe guidance,
  and document the concrete process for adding and recording a new method.

## Pointwise networks

- [x] Inspect NVIDIA layers/full model equations; adapt angular/Fourier, highway,
  modified/multiscale Fourier, DGM, SIREN and Fourier/Gabor filter networks.
- [x] Add architecture controls to the shared network block and retain them in
  both recipe forms, model migrations, copy/undo and validated checkpoint resume.
- [x] Check first/second derivatives, physical scaling, batch independence, gating,
  original FCNN identity, real scalar/flow training and all-architecture archives.
- [x] Record a twenty-run CPU compatibility study with configs, predictions and
  provenance; retain Apache notices and keep neural operators out of scope.

## Illustrated documentation

- [x] Add a coherent network/PINN screenshot walkthrough with native menus,
  architecture-specific settings, compact/expanded views and explicit losses and
  interval reference comparison; keep numerical evidence separate.
- [x] Provide repeatable state-isolated screenshot capture with a real CPU run,
  configuration assertions, source/settings manifest and no silent overwrites.

## Learning-rate annealing PINN

- [x] Implement supplied Wang/Teng/Perdikaris v1 Algorithm 1 independently of its architecture.
- [x] Add one-file compact/expanded recipe, editable balance controls and optional summary.
- [x] Reuse all existing pointwise networks and scalar/coupled builders; validate checkpoint continuation.
- [x] Record equal-budget CPU baseline/LRA results including coefficient-cap sensitivity.
- [ ] Deferred: reproduce a paper benchmark with its full sampling/training protocol; current small Poisson evidence is not that reproduction.

- [x] Separate PINN method, Optimisation and Trainer inspector groups in compact
  and expanded recipes; distinguish network Adam descent and SA mask Adam ascent.

- [x] Split library categories into PINN recipes (compact constructions) and Training
  (expanded trainers). Optimisation remains inspector settings until standalone blocks exist.

- [x] Group five 1D problems in File → Examples → 1D examples, with explicit
  2D labels on the parent-menu cases.

- [x] Add the LRA paper 2D Helmholtz problem with existing Fourier features, explicit
  forcing/reference and soft BCs; verify desktop/persistence and record CPU results.
  Paper-architecture and full stochastic benchmark reproduction remain outside this example.

## Future studies, iterator blocks and laptop/HPC sweeps

See [study/sweep vision](roadmap.md#future-vision-studies-component-alternatives-and-sweeps).
These are proposed features; current execution still runs one supported problem.

- [ ] Define a saved study containing alternative PINN recipes/components and
  parameter choices, with inspectable resolved configurations for every trial.
- [ ] Design iterator/alternative blocks for concrete existing components. Make
  Cartesian, paired and conditional combination rules explicit; preview trial
  counts and budgets before running.
- [ ] Preflight each trial's component/parameter/data compatibility. Record skipped
  combinations and reasons in a dedicated study output; continue valid trials.
  Distinguish incompatible, invalid, failed, interrupted, pruned and completed
  outcomes. Never fabricate objective values for non-running combinations.
- [ ] Add a shared GUI-independent execution entry point, so identical study/model
  definitions run from the laptop GUI or a headless command on an HPC node.
- [ ] Persist stable trial IDs, full resolved settings, seeds, code/data provenance,
  resource settings, status/reason, logs, requested metrics, exports and checkpoints.
  Keep evaluation data and comparison protocols explicit across trials.
- [ ] Add bounded local CPU/GPU scheduling, cancellation and checkpoint-safe resume;
  isolate failures and outputs so one trial cannot overwrite or corrupt another.
- [ ] Add Slurm sbatch study export/execution and result collection, including
  explicit resource requests, input/artifact locations, optional job arrays,
  preemption/restart handling and reopening results in the GUI.
- [ ] Assess Hydra, Optuna and W&B sweep integrations against the study contract.
  Keep configuration, search, execution and tracking separable; select tools later.
  Preserve offline operation and avoid mandatory tracking-service credentials.
- [ ] Add correctness tests for expansion, conditional settings, compatibility
  reports, failure isolation, repeatability, resume and desktop/headless parity
  before pursuing sweep-driven solution accuracy.

## User-friendly illustrated help

- [x] Add a searchable local manual with short task-based walkthroughs and
  separate reference/project sections, following the PyniteGUI documentation approach.
- [x] Capture 22 current Qt screenshots with captions and full-size links; verify
  a complete first-PINN workflow and preserve personal UI state during capture.
- [x] Add F1 / Help → User manual, with build guidance and clear browser errors.
- [x] Keep search/fonts local for offline reading; document optional build tools
  and verify rendered links, screenshots and help behaviour.

- [x] Add a GitHub Pages workflow that builds/checks and publishes the generated
  manual, with build-only PR validation and documented GitHub Actions source setting.

## Residual-driven sampling (28 October 2026)

- [x] Separate RAR and residual importance sampler blocks; keep fixed sampling default.
- [x] Scheduled, seeded candidate scoring with derivatives and bounded scoring chunks.
- [x] Capped duplicate-free RAR additions; fixed-budget importance draws with inverse-probability MSE.
- [x] Baseline, hard-BC and LRA integration; fixed BCs and independent output evaluation.
- [x] Model migration, active-point/weight history checkpointing and exact resume tests.
- [x] Document formulas, DeepXDE differences, screenshots and CPU workflow evidence.
- [ ] GeoSampler-provided admissible candidate sets for imported/non-box geometries.
- [ ] Define SA-PINN mask and Adam-moment transfer before combining it with moving points.
- [ ] Consider a sampler-owned live point-set view when there is a concrete need.
- [ ] Revisit XPINN after a practical GeoSampler subdomain/interface sampling workflow exists.

- [x] Cross-check importance loss correction against NVIDIA Sym's area/density
  convention for equal-weight pools, including non-unit volume and gradients.
- [ ] Consider field-gradient importance measures and fixed candidate pools as
  explicit options; retain residual sampling semantics for existing saved models.
