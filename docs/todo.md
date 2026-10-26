# Current TODO

See roadmap.md for the agreed design and stages. Completed historical DeepINN
checkboxes are not claims about current Margam support.

- [ ] Catalogue implemented methods, sampling and optimisation components with limitations.
- [ ] Reuse GeoSampler unit-domain samplers and expose supported settings.
- [x] Add built-in intervals, squares and rectangles with grid/random meshless sampling.
- [x] Create named rectangle-side point sets, then assign constant Dirichlet/Neumann BCs.
- [x] Connect PDE points to the Poisson residual preset; defer additional pointwise inputs.
- [ ] Expose arbitrary PDE residual functions through the UI.
- [x] Add constant Dirichlet/Neumann settings, analytical normals and an explicit rectangle corner policy.
- [ ] Add broader BC operators and spatially varying targets.
- [ ] Add prepared-data adapters with explicit coordinate/label mapping and previews.
- [ ] Define typed ports, connection validation and executable graph semantics.
- [x] Add block/wire deletion and fit-to-model navigation.
- [ ] Add saved models/results, undo/redo and zoom/pan controls.
- [x] Connect the baseline engine through background training with progress and cancellation.
- [x] Add a runnable Poisson visual template and verify the analytical interval solution.
- [ ] Reuse GeoSampler dock persistence and recovery patterns where appropriate.
- [ ] Integrate GeoSampler for complex geometry later; no 3D viewport in this project.
- [ ] Validate a compatible GPU PyTorch build in the Python 3.14 environment.

## Menus and explicit outputs

- [x] Add File / Edit / View / Run / Help menus for implemented actions. File
  currently offers template reset and Quit; model Open/Save remains future work.
- [x] Allow training without any plot/results block.
- [x] Make loss plots and prediction plots explicit optional output blocks, with
  per-instance history/evaluation settings.
- [x] Create/show output views only when explicitly opened through their blocks.
  Unconnected blocks can show a placeholder; only connected outputs request data.
- [x] Separate reference sources and comparisons from ordinary predictions; the
  current explicit analytical demonstration source supports intervals only.
- [x] Remove automatic analytical-reference construction from the generic worker.
- [x] Keep operational status, progress, validation and failures available without
  numerical output blocks.
- [x] Add Training Monitor blocks with bounded recorded histories.
- [x] Support bottom docking, floating, hide/reopen, and removal with the owning
  block. Keep view visibility separate from data retention.
- [ ] Add result export blocks and general imported/user-function reference sources.

## Left library and sampling options

- [ ] Replace the flat left-hand list with a searchable, grouped component library:
  Geometry / Data, Sampling, Physics / Conditions, Networks, PINN Methods, and
  Outputs / Comparison. Keep runnable templates distinct from reusable components.
- [ ] Group sampling by purpose: interior/PDE points and boundary point sets.
  Within each, show implemented methods and their supported settings. Sampling
  method is distinct from the target region; assigning a BC is a later operation.
- [ ] Keep basic built-in interval/rectangle sampling in Margam: grid and seeded
  uniform random, domain/side selection, point budget or points-per-side, seed,
  actual count and preview. A simple problem must not require GeoSampler.
- [ ] Let distinct point sets use independent sampling recipes, methods, counts
  and seeds; do not force every boundary and the interior to share one selection.
- [ ] Drive library entries and property options from the component catalogue,
  including status, compatible targets, parameters and limitations. Do not list
  planned sampling methods as executable options.
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
- [ ] Add drag/drop library insertion and canvas quick insert using the same
  catalogue and block-instance model.
- [ ] Define contextual double-click behaviour and shared inspector/dialog state.
- [ ] Add instance-owned settings and typed ports before generalising graph execution.
- [ ] Add output branching and undoable graph/parameter edits.
- [ ] Validate interaction parity hands-on against a selected Simulink release;
  the current evidence collection is documentation-based.

## Output/menu milestone (26 October 2026)

The first output-ownership stage in simulink-evidence.md is implemented and
verified. See ui-prototype.md for actual menu actions, output lifecycle and limits.
The searchable/grouped library and independent core sampling recipes remain next.
