# Current TODO

See roadmap.md for the agreed design and stages. Completed historical DeepINN
checkboxes are not claims about current Margam support.

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
- [ ] Extend the explicit scalar expression contract to coupled/vector PDE systems
  when field and coordinate schemas are defined.
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
- [ ] Integrate GeoSampler for complex geometry later; no 3D viewport in this project.
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
- [ ] Add dedicated result export blocks (ParaView remains deferred). Saved-run
  archives are implemented independently of field export blocks.

## Left library and sampling options

- [x] Replace the flat left-hand list with a searchable, collapsible component
  library: Geometry, Sampling, Neural networks, Loss / Physics, Optimisation / PINN,
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
- [ ] Reserve advanced geometry-aware sampling, CAD face/edge selection and richer
  region workflows for GeoSampler. Define the integration boundary rather than
  rebuilding its full toolbox inside Margam.
- [ ] Keep library, ports, conditions and output views independent of heat-transfer
  terminology. Poisson is a demonstration template; future scalar/vector fields,
  multiple outputs and other physics must not inherit temperature-only assumptions.

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
- [ ] Validate interaction parity hands-on against a selected Simulink release;
  the current evidence collection is documentation-based.

## Output/menu milestone (26 October 2026)

The first output-ownership stage in simulink-evidence.md is implemented and
verified. See ui-prototype.md for actual menu actions, output lifecycle and limits.
The searchable, collapsible library and independent rectangle-side sampling
recipes are implemented. The component catalogue records implemented capabilities;
structured catalogue metadata driving all library descriptions remains future work.

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
- [ ] Improve and quantify 2D training accuracy across seeds/settings; soft boundary
  constraints and finite training can leave visible errors, which remain reported.

## Deferred: ParaView

- [ ] Later, decide a ParaView-compatible result export format and implement it as
  an explicit output request. Preserve coordinates, field names and available
  prediction/reference/error data; define units and export scope first.
- [ ] Revisit ParaView integration only when requested. No ParaView setup, export
  implementation or integration is part of the current work.

Existing in-app heatmaps are left as they are. Further field-plot development is
paused; operational training monitors and basic rectangle previews remain useful.
