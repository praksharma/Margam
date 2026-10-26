# Simulink interaction evidence and Margam specification

Collected 26 October 2026 from official MathWorks documentation, currently
labelled R2026b. This is documentation-based research, not a hands-on validation
in a licensed Simulink installation. Behaviours below are distinguished from
Margam proposals; release-specific details should not be treated as timeless.
No application code was changed as part of this evidence collection.

## Important corrections to our earlier assumptions

1. Modern Simulink uses a task-oriented toolstrip. A conventional File / Edit /
   View / Help menu is a Margam choice, not a literal copy of the current shell.
   The editor reference maps older menus to newer commands. [Editor reference](https://www.mathworks.com/help/simulink/slref/simulinkeditor.html)
2. A Scope is not a compulsory bottom results pane. Current scopes can have
   standalone windows or be grouped in a scope container. MATLAB-desktop scope
   docking was removed in R2025a. Our bottom-docked plots remain an adaptation.
   [Scope reference, including version history](https://www.mathworks.com/help/simulink/slref/scope.html)
3. “No Scope block means no possible data viewing” is too strong: Floating Scope,
   Scope Viewer and logging/inspection provide other routes. Margam's explicit
   output blocks are a deliberate initial contract. [Viewing routes](https://www.mathworks.com/help/simulink/view-simulation-results.html)
4. An analytical solution is not required for comparisons: the documented
   comparison workflow accepts logged or imported data. [Compare data](https://www.mathworks.com/help/simulink/ug/compare-simulation-data.html)

## Evidence register

Each row contains a documented observation. The next section contains our design
choices; those choices are not claims about Simulink.

| ID | Topic | Documented evidence | Primary source |
| --- | --- | --- | --- |
| E01 | Application shell | Commands are organised in Simulation, Debug, Modeling, Format and Apps toolstrip tabs. Save, undo and redo are available in the quick access toolbar. | [Simulink Editor](https://www.mathworks.com/help/simulink/slref/simulinkeditor.html) |
| E02 | Library | The Library Browser has a library tree and keyword search. It docks beside the model by default. Blocks can be dragged in or added through a context menu. | [Library Browser](https://www.mathworks.com/help/simulink/slref/librarybrowser.html) |
| E03 | Quick insertion | Double-click empty canvas, search, choose a block with arrow keys and press Enter. Results include descriptive details. Quick insertion can also extend an unfinished connection. | [Add Blocks to Models](https://www.mathworks.com/help/simulink/ug/add-blocks-to-models.html) |
| E04 | Parameter dialogs | Double-clicking an ordinary block opens its parameter dialog. Parameter editing is also available through the Property Inspector. | [Edit Block Parameters](https://www.mathworks.com/help/simulink/ug/edit-block-parameters.html) |
| E05 | Inspector | The inspector follows the selection, is dockable/undockable, and applies values when set. Double-click behaviour is contextual; subsystems and model blocks are not ordinary parameter-dialog cases. | [Property Inspector](https://www.mathworks.com/help/simulink/slref/propertyinspector.html) |
| E06 | Connections | Drag an output port to an input port. An output can branch to several destinations. Ctrl-drag from an existing line makes a branch. Some blocks, such as Scope, can add input ports for multiple signals. | [Connect Blocks](https://www.mathworks.com/help/simulink/ug/connect-blocks.html) |
| E07 | Navigation/editing/execution | The modelling shortcut reference documents fit, zoom, pan, clipboard, undo/redo, subsystem creation, model update and simulation controls. Selected mappings are recorded below. | [Modelling shortcuts and mouse actions](https://www.mathworks.com/help/simulink/ug/summary-of-mouse-and-keyboard-actions.html) |
| E08 | Hierarchy | A subsystem groups blocks behind an input/output interface. Subsystems can nest. Double-click enters one; navigation returns to its parent. A copied subsystem is independent; referenced components support reuse separately. | [Group Blocks into Subsystems](https://www.mathworks.com/help/simulink/ug/creating-subsystems.html) |
| E09 | Plot windows | Scopes show connected signals, support multiple displays, and can be docked together in a scope container or undocked. Closed scopes can later display retained data; retention limits are configurable. | [Scope](https://www.mathworks.com/help/simulink/slref/scope.html) |
| E10 | Alternative viewing routes | Scope, Floating Scope/Scope Viewer, Record and the Simulation Data Inspector offer different viewing/logging routes. A physical Scope connection is not the only route. | [View Data During Simulation](https://www.mathworks.com/help/simulink/view-simulation-results.html) |
| E11 | References and comparison | The Data Inspector compares selected signals or runs from logged/imported data, using tolerances and alignment constraints. Comparison plots show the signals and their difference. | [Compare Simulation Data](https://www.mathworks.com/help/simulink/ug/compare-simulation-data.html) |
| E12 | Diagnostics | Diagnostics are classified as errors, warnings and information, with stages for operations such as loading, updating and simulation. The viewer may be standalone or docked. A message can link to a highlighted source block. | [Diagnostic Viewer](https://www.mathworks.com/help/simulink/slref/diagnosticviewer.html) |
| E13 | Saving | Models normally save as SLX. In R2026b, personal model-view settings default to external JSON preferences; view state can instead be stored in the model. | [Save Models](https://www.mathworks.com/help/simulink/ug/save-models.html) |
| E14 | Beginner workflow | The tutorial walks through a blank model, browsing/inserting blocks, setting parameters, wiring, saving and running. It is a useful sequence for reviewing the overall interaction. | [Create a Simple Model](https://www.mathworks.com/help/simulink/gs/create-a-simple-model.html) |

## Additional context-menu evidence

MathWorks documents element-specific context menus with block parameters,
properties, help and editing actions. This supports block-owned context actions;
it does not establish a universal sample-plot command. See [Find Simulink Context
Menu Options](https://www.mathworks.com/help/simulink/ug/find-simulink-context-menu-options.html).
Margam's rectangle-specific preview actions are an explicit local adaptation.

## Proposed Margam interaction contract

These are proposed acceptance targets, not completed implementation claims.
They preserve the interactions while defining PINN-specific meaning.

| Area | Margam behaviour | Evidence / rationale |
| --- | --- | --- |
| Main window | Keep a large canvas, left library, right inspector and status area. Let panels move or hide. Use File / Edit / View / Help plus concise execution controls initially. | E01, E02, E05; conventional menus are our choice. |
| Library | Searchable tree: Geometry / Data, Sampling, Physics / Conditions, Networks, PINN Methods, Outputs / Comparison. Search aliases and descriptions from the catalogue. Keep templates separately discoverable. | E02–E03; category names are PINN-specific. |
| Insertion | Drag a library entry to an exact location. Double-click empty canvas for quick search. Add a new instance with its own identity and settings. | E02–E03. |
| Selection | Single selection drives the inspector. Multiple selection exposes common/group actions rather than silently editing an arbitrary instance. | E05; multiple-selection policy is our decision. |
| Open block | Double-click a computational block for parameters; double-click an output block for its view; double-click a subsystem to enter it. Use the same underlying parameter model in dialog and inspector. | E04–E05, E08–E09. |
| Ports | Distinguish geometry, point sets, fields/model handles, residual/constraint definitions, diagnostics and evaluated predictions. Label inputs and check compatibility. One source per ordinary input; collection inputs must be explicit. | E06 motivates directed interfaces; these payloads and cardinalities are our contract. |
| Wiring | Support explicit fan-out, visible branch points and predictable line routing. Moving blocks preserves connections. Failed connections explain the incompatible types or dimensions. | E06; PINN validation details are our choice. |
| Sampling | Separate a target region/point set from a method. Each interior or boundary set owns its recipe, count and seed. Basic domains work locally; advanced geometry stays with GeoSampler. | Prior Margam decisions, not a claim about Simulink sampling. |
| Outputs | Training can run without any plot block. Training Monitor, Field/Solution Plot, Export and Comparison are explicit requests connected to compatible data. | Inspired by E09–E11; intentionally narrower initial policy than Simulink's viewing routes. |
| View placement | Open a requested view explicitly from its block, initially in bottom-docked tabs; allow detach/redock. Hide a view without deleting its request or stopping training. Removing the block removes its output request. | E09 inspires independent views; bottom docking and removal semantics are our choices. |
| Reference | An optional source supplies analytical, measured or numerical reference values. Comparison pairs them with predictions at compatible evaluation points and units. No exact solution is assumed. | E11; spatial evaluation and unit rules are PINN-specific. |
| Run/stop | Validate the graph before training; report progress and errors independently of outputs. Cancellation preserves coherent run state. Make fresh training versus continuation explicit. | E07, E12 inspire controls; training lifecycle is our choice. |
| Diagnostics | Link errors to their block/port/property. Keep operational diagnostics separate from plots and requested metrics. | E12. |
| Hierarchy | Later, group a selection into a subsystem with explicit ports and navigable contents. Distinguish independent copies from linked reusable components. | E08. |
| Persistence | Store graph identity, settings and edges in a versioned model. Decide separately whether to save data, trained weights and view preferences. Do not adopt SLX solely to mimic appearance. | E13 informs separation; Margam format remains undecided. |

### Important semantic boundary

Simulink's signal-oriented dynamic simulation is not automatically a PINN training
execution model. A Margam connection can represent a problem definition or a
trained-model handle, not a value changing at each simulation timestep. Training
iterations are not necessarily physical time. Loss histories, sampled coordinates,
spatial fields and vector predictions need their own schemas and output views.

Do not infer that the canvas is PyTorch's differentiation graph, that every block
runs once per epoch, or that a subsystem changes optimisation behaviour. Those
semantics must be specified independently. The reusable numerical method remains
in its method file; templates assemble it from shared components.

## Candidate familiar shortcuts

Documented Windows/Linux Simulink mappings from E07; adopt only after resolving
Qt/text-editor focus conflicts. These are reference mappings, not current Margam
bindings. macOS modifiers differ.

| Action | Simulink reference |
| --- | --- |
| New / Open / Save / Close | Ctrl+N / Ctrl+O / Ctrl+S / Ctrl+W |
| Cut / Copy / Paste | Ctrl+X / Ctrl+C / Ctrl+V |
| Undo / Redo | Ctrl+Z / Ctrl+Y |
| Fit model | Space |
| Zoom in / out / 100% | Ctrl+Plus / Ctrl+Minus / Ctrl+0 |
| Pan | Middle-button drag or hold Space and drag |
| Create subsystem | Ctrl+G |
| Enter selected subsystem / parent | Enter / Esc |
| Update model | Ctrl+D |
| Start / Stop | F5 or Ctrl+T / Ctrl+Shift+T |

For Margam, use model update as validation/assembly, not a claim that it shares
Simulink compilation. Do not intercept these keys while editing text or numeric
properties. Zoom should be centred predictably; automatic fit should not undo a
user's deliberate pan/zoom on each selection change.

## Illustrated reference pages to revisit

The official pages include diagrams/screenshots for reviewing placement and
interaction. Links point to the originals; no Simulink screenshot assets were
copied into the repository. Their illustrations are documentation evidence,
not a captured session on this machine.

| What to inspect | Reference page / relevant section |
| --- | --- |
| Overall editor and first-model sequence | E14: blank editor, Library Browser, quick insert and connected model |
| Library tree, search and docking | E02: docked and standalone Library Browser figures |
| Search popup with descriptions | E03: Quick Insert details pane |
| Parameter editing | E04: Property Inspector and block dialog examples |
| Output-to-input wiring and branching | E06: connection and branch illustrations |
| Independent plot presentation | E09: Scope tab and scope-container sections |
| Grouped model hierarchy | E08: creating a subsystem from selected blocks |
| Messages linked to blocks | E12: diagnostic messages and highlighted source |

## Implementation order and acceptance checks

1. **Output ownership first.** Remove the mandatory Solution block from execution
   validation. A model with no requested output runs without opening plots. A
   connected Training Monitor opens only when requested. Hide/reopen its view
   without cancelling training; define retained-history limits separately.
2. **Instance-based model and library.** Replace the singleton template settings
   with explicit block instances, typed ports and per-set sampling recipes.
   Two boundary samplers can have different settings without overwriting each
   other. Drag/drop and quick insert create the same kind of model instance.
3. **Editing and navigation.** Add undoable create/move/connect/delete/parameter
   actions, clipboard operations, pan/zoom, and contextual block opening. Inspector
   and dialogs reflect the same state; text inputs keep normal editing shortcuts.
4. **Persistence and menus.** Add model save/open, dirty state and recovery;
   organise implemented actions under the requested menus. Reopen a model with
   identical graph/settings without requiring local view preferences to be shared.
5. **Diagnostics and hierarchy.** Selecting an error locates the responsible
   element. Later, subsystem grouping preserves external connections and allows
   entry/return navigation, without changing the numerical problem implicitly.
6. **General physics and comparison.** Predictions work without a reference;
   comparison requires a declared compatible source. Test scalar and vector field
   schemas and multiple outputs without temperature-specific assumptions.

These steps refine the current working Poisson UI rather than requiring an
all-at-once rewrite. See todo.md for the active backlog and ui-prototype.md for
what is actually implemented today.

## Evidence gaps

- No direct observation of a live licensed Simulink session in this research.
- No exact pixel dimensions, drag timing, focus behaviour or performance claims.
- Input defaults, routing details and window behaviour can differ by release/platform.
- PINN graph schemas, method execution semantics, checkpointing and cancellation
  are Margam design work; the sources do not settle them.
- Before claiming visual/behavioural parity, run a hands-on comparison checklist
  for the selected Simulink release and the actual target desktop platform.
