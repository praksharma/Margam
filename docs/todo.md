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

- [ ] Add a standard File / Edit / View / Help menu bar. Organise model open/save,
  editing, panel visibility/layout, and help actions under the relevant menus;
  expose only implemented actions as available.
- [ ] Allow training without any plot/results block. The current Poisson validator
  requires a Solution block; remove that requirement as output blocks are introduced.
- [ ] Make loss plots, solution plots and result export explicit optional output
  blocks, with their own settings and evaluation points where appropriate.
- [ ] Create/show result views only for connected output blocks. Do not pre-create
  result panels or switch to them automatically when no such output is requested.
- [ ] Separate reference data/functions from predictions. Add an optional reference
  source and comparison block; compute RMSE/error plots only when explicitly
  connected to compatible reference values at the evaluation points.
- [ ] Remove automatic analytical-reference construction from the generic training
  worker. The Poisson demo may provide it through an explicitly visible optional
  reference block, rather than making it a property of every solve.
- [ ] Keep operational status, progress, validation and failures available without
  requiring numerical output blocks; distinguish these from requested result plots.
