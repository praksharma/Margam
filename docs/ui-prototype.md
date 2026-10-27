# Baseline PINN desktop workflow

Run from the repository root with the existing root environment:

```sh
source .venv/bin/activate
python -m margam.ui
```

## Layout and menus

The block library is on the left, model canvas in the centre, and inspector on the
right. No figure or sample panel is created by default. Operational Messages have
an independent dock, hidden until requested from View or shown for a training error.
Rectangle geometry/sample previews are local, explicitly opened windows; they float
by default and can be docked by the user. Output views have their own dock widgets,
which can be grouped or detached/redocked. No global panel owns the figures and
there is no 3D viewport. Source attribution is in ../THIRD_PARTY.md.

| Menu | Implemented actions |
| --- | --- |
| File | Reset Poisson template (Ctrl+N), Heat demo, Quit (Ctrl+Q) |
| Edit | Delete selected (Delete, canvas focus), select all blocks (Ctrl+A, canvas focus) |
| View | Panel visibility, zoom in/out, 100%, Fit model, Reset panel layout, Open selected output, Theme |
| Run | Validate (Ctrl+D), Train (F5), Stop (Ctrl+Shift+T) |
| Help | Quick start and About |

Open/Save and Undo/Redo are implemented for model definitions/edits as described below.
Panel visibility/layout is not yet persisted. Fitting is explicit; selecting a
block or resizing an inspector does not automatically reset the camera.

## Core model

The Poisson template has seven required blocks: domain, interior sampler, boundary
points, PDE residual, boundary condition, neural network, and PINN method. Output
blocks are optional. The underlying method remains `margam/methods/pinn.py`.

Geometry, sampling, constant source, BCs, hidden-layer sizes, Adam settings and
weights are edited in the selected block's inspector. Built-in interval/rectangle
sampling needs no imported geometry or mesh. Grid/random sampling, corner policy,
normal conventions and numerical limits are documented in components.md.

Right-clicking any supported block offers Properties, Delete block and Help.
Rectangle adds Preview geometry and Preview samples. Output blocks add Open
monitor, Open plot or Open comparison, according to their actual type. There is
no universal Plot action. Copy/paste is not offered until implemented.

Double-clicking a computational block exposes its inspector. Multiple selection
has no arbitrary single-block parameter editor. Full parameter dialogs and general
instance-owned core configurations remain future work.

## Optional rectangle previews

Right-click a Rectangle block and choose Preview geometry or Preview samples.
Geometry shows the bounds and outline; samples shows the currently configured
interior/boundary point sets. These two views are keyed to that block and reused
when reopened. Bounds/sampling edits update existing views without opening new
ones. Closing hides a view. Switching to Interval, deleting its owner or resetting
the model closes/removes its previews. These actions do not evaluate a neural model.

The implementation is deliberately `RectanglePreview`, not a general geometry
renderer or plot registry. Interval and unrelated blocks have no sample-preview
action. Complex geometry and advanced sampling visualisation belong to GeoSampler.
Rectangle previews are available as read-only views during training.

## Optional output blocks

| Block | Inputs | Output settings and behaviour |
| --- | --- | --- |
| Training Monitor | PINN method | Live losses; retained-update limit 25–100000, default 10000 |
| Solution Plot | PINN method | Scalar predictions only; evaluation resolution 2–201 per axis |
| Reference | No input | Explicit interval analytical / rectangular truncated-series Poisson source |
| Comparison | Model input above; Reference input below | Prediction/reference curves and RMSE at its own evaluation points |

Outputs have stable instance identities and independent settings. A reference
source does nothing unless connected to a requested Comparison. The rectangular reference supports equal constant Dirichlet conditions on all
sides and uses 40 odd sine modes per axis (truncated). Interval references remain
analytical. Other rectangle BCs are rejected for comparison; ordinary prediction
views work without references.
It is not a general symbolic or imported-data reference editor.

Connect an output block before training, then double-click it or use its specific
right-click Open action to open its view.
Selecting it and choosing View → Open selected output is another route. A view
can be opened before, during or after training. New or hidden views are never
opened automatically by training. Connected outputs still collect/evaluate their
requested data if their views are closed. An unconnected output is ignored; a
comparison connected on only one input fails validation rather than assuming a
reference. The trained model is evaluated separately for each prediction request.

Close a view to hide it; reopen the same block to reuse the view and retained
results. Removing the block deletes its view and request. Changing the graph or
model/evaluation settings invalidates the current run. Changing a monitor's
retention limit trims its existing display history without retraining; previously
trimmed data cannot be recovered by increasing the limit.

## Execution and diagnostics

Training runs in a background thread in batches of up to 25 updates. Progress,
current total loss and failures are operational information available without any
output block. Only connected monitors retain UI loss history. Their plots show
weighted total, raw PDE loss and mean raw BC loss on a logarithmic axis; losses
are measured before updates. The numerical recipe retains its own loss history
independently of the bounded view history.

Without prediction/comparison requests, the worker never calls prediction or
constructs a reference. A completed/partial numerical model is retained in the
finished worker until a model edit/new run, rather than passing a network to plot
widgets. There is no export/continuation interface for that model yet.

Settings/graph edits are locked during training, while block selection and opening
output views remain available. Stop requests cancellation between batches. Closing
the main window waits for the worker to stop. Errors release the controls and show
operational Messages; they do not create result views. CPU thread count is one for
these small examples. CUDA remains unverified in this environment.

## Analytical check

Choose Interval on [0,1], source 2, zero Dirichlet endpoints, 32 grid interior
points, hidden layers 16,16, learning rate .01 and 1000 iterations. Add Reference
and Comparison, connect the method to its upper model input and Reference to its
lower reference input, then train and explicitly open the Comparison. The reference
is u=x(1-x). A separate Solution Plot shows predictions alone, not an assumed exact
solution. A connected rectangular Comparison shows prediction/reference/error heatmaps,
RMSE and maximum absolute error against its explicit truncated-series reference.
Standalone rectangle Solution Plots show predictions alone.

## Validation and remaining work

Tests cover output-free execution, no implicit reference, comparison validation,
independent evaluation resolutions, rectangle-only context actions and preview
cleanup, monitor retention, closed-view recording,
detach/redock, deletion, menu presence, worker failures and close/cancel behaviour.
Screenshots are under assets/screenshots. General graph execution, instance-owned core settings,
multiple arbitrary sampling blocks, saved results/checkpoints, arbitrary PDE/reference
sources and export remain in todo.md.

## Canvas navigation

Wheel zoom is centred on the mouse pointer, limited to 10–400%. Middle-button drag
or Space+left-drag pans. Tapping Space fits all blocks. Toolbar/View actions offer
zoom in/out, 100% and Fit model. Ctrl+Plus, Ctrl+Minus and Ctrl+0 are canvas-scoped
shortcuts, so property text inputs retain their own key handling. Fitting can choose
a smaller scale if needed to include a large diagram. Panning and zoom do not move
blocks, alter training settings, or reset simply because selection changes.

## Ready-wired steady heat demo

Toolbar/File → Heat demo loads the same baseline method with a configured square,
constant source, zero edge temperature rise and connected monitor, prediction,
Reference and Comparison blocks. No view is forced open. The current configuration
uses 144 interior points, 24 samples per side, hidden layers 32,32, Adam learning
rate .005, 2500 updates, PDE weight 1 and mean BC weight 10. These are editable demo
settings, not universal defaults or an accuracy guarantee. See heat-demo.md.

## Collapsible block library

The left dock groups the existing blocks under Geometry, Sampling, Neural networks,
Loss / Physics, Optimisation / PINN, and Results. Each group collapses independently.
Search matches category and block names without case sensitivity, temporarily
expands matches, and restores previous expansion states when cleared. Double-click
a block or press Enter to add it; category headings only toggle expansion. The
whole library, including search, is disabled while training. The dock can still be
hidden/reopened from View. No new solver, loss or optimiser implementations are
implied by these categories: the PINN method still owns Adam settings and the PDE
and BC blocks define its existing loss terms.

## Independent boundary sampling

The Sampling library group separates Interior sampler from Boundary points. The
interior inspector exposes grid/random, budget and seed (also used for network
initialisation). The Boundary points inspector selects a named rectangle side and
exposes that side's method, exact count and independent seed. Limits match the
sampler contract: 100000 interior points, 10000 on each side. Method choices and
tooltips use the implemented `SAMPLERS` mapping. Changing one side does not change
the interior or another side. The boundary block shows the total actual point count;
the Rectangle's optional sample preview uses these recipes. BC type/target assignment
remains in its own block. Interval endpoints have no sampling controls. Shape changes
retain rectangle recipes, while template reset clears overrides. These are recipes
inside the current template-backed core, not arbitrary independently executable
sampler block instances or a general graph compiler.

## Drag/drop and quick insertion

Library leaves can be dragged into the canvas; headings cannot. Only a known
block title in Margam's block MIME payload is accepted, not text/files. The block
centre lands at the drop's scene coordinate, accounting for pan and zoom. New
blocks remain unconnected. Core duplicates are permitted for editing but continue
to fail the existing template validation; optional outputs never auto-open.

Double-click an empty canvas location or press Tab with canvas focus for the Add
block popup. Double-clicks on blocks retain their existing Properties/Open view
actions. Tab inserts at the visible canvas centre; mouse insertion uses the clicked
position. Popup search uses the same groups and titles as the dock. Enter in search
adds the first matching block; Down focuses the results for arrow-key selection and
Enter insertion. An empty result does nothing. Escape or clicking outside dismisses
the popup. Training and template reset dismiss pending insertion; training locks
both insertion paths. Property-editor Tab remains ordinary focus navigation.

`block_catalogue.py` is a small shared list of supported UI titles and kinds,
not a general solver/plugin registry or a replacement for the component capability
document. All insertion paths call `Window.add_node`, so output instance settings,
selection, graph invalidation and validation rules remain consistent. Current core
settings are still template-backed; arbitrary independent core instances and typed
execution are future work. Verified native Qt drop events, placement under camera
transforms, keyboard routing, invalid payloads and popup lifecycle are covered by
`tests/test_block_insertion.py`. A real platform drag gesture should also be checked
on the target desktop; automated offscreen events do not prove OS drag appearance.

## Typed editor ports and connection checks

Source-only Geometry, Neural network, and Reference blocks have no artificial
input socket. Terminal output blocks have no output socket. PINN method inputs
are separate pde/boundary/network ports; Comparison keeps separate model/reference
inputs. Tooltips identify each port's semantic role and one-source input rule.

| Source output | Accepted destination input |
| --- | --- |
| Geometry | Interior sampler / Boundary points: Geometry |
| Interior points | PDE residual: Interior points |
| Boundary point sets | Boundary condition: Boundary point sets |
| PDE residual | PINN method: PDE residual |
| Boundary conditions | PINN method: Boundary conditions |
| Neural network | PINN method: Neural network |
| PINN run | Training Monitor / Solution Plot: PINN run; Comparison: PINN run |
| Reference source | Comparison: Reference source |

`ports.py` holds the concrete editor contracts. Direction, compatible semantic
type, same-model ownership, self-connections and occupied destinations are checked
before creating a wire, for both gestures and template/helper connections. Each
input accepts one source; outputs may branch to distinct consumers. Delete the
old wire to replace an input source. A rejected native gesture reports its reason
in Properties, status and the Messages log without opening a plot or clearing
results. Dropping into empty space cancels without changing the model. Training
and deletion cancel pending wires, preventing stale endpoints. Full pre-training
validation checks endpoints again and still requires the exact seven-block core.

These types describe editor roles, not tensor shapes, units or general executable
graph semantics. A PINN run is the method's model/history output contract, not a
simulated physical-time signal. Core configuration remains owned by the template.
Outputs retain instance-specific settings and explicit view ownership. The existing
`connect_nodes` helper infers the unique compatible target socket; an explicit
input name can be supplied. Unsupported helper connections raise ValueError.

Double-clicking core/reference blocks selects them and opens the same inspector
used by Properties. There is no independent parameter-dialog state to synchronise.
Double-clicking an output opens/reuses its owned dock. Tests cover these actions,
per-output branching, rejected native wire gestures, endpoint checks, preservation
of run data on rejection, and cleanup of incomplete gestures.

## Undo and redo

Edit and block context menus offer Undo/Redo with the next operation's name.
With canvas focus, Ctrl+Z undoes and Ctrl+Y or Ctrl+Shift+Z redoes. Property/text
editors keep their local shortcuts; use the Edit menu to replay model history
while an inspector field has focus. Actions are disabled while training, including
through public undo/redo wrappers. Templates start a fresh, empty history.

The bounded Qt undo stack retains at most 100 commands. Additions, valid
connections, deletion of selected nodes/wires, grouped mouse-drag moves, template
settings (including BC and per-side sampling recipes), output settings and output
names are reversible. Consecutive changes from the same inspector control are
coalesced until another command or inspector reconstruction; programmatic setting
calls are separate steps. A new edit after undo discards the previous redo branch.
No-op values, rejected connections, navigation and selection do not create steps.
Direct low-level scene/item manipulation is not an editor command.

Deleted nodes keep their IDs and instance settings. Undo deletion restores the
same nodes and their incident wires together. Deleting an owner removes its
optional views; undo restores its model state without reopening those views.
Opening, hiding, docking and closing views, camera movement and library search are
outside model history. Layout-only block moves and output renaming preserve run
data. Semantic graph/settings edits and their replay invalidate old results;
undo never restores model weights, predictions or discarded monitor records.
Increasing a restored retention limit does not recover already trimmed samples.
A restored view can be explicitly opened again; no plot is created by history.

## Model files and unsaved changes

File → Open model (Ctrl+O), Save model (Ctrl+S) and Save model as (Ctrl+Shift+S)
use `.margam.json` documents. Version 1 is readable JSON with a fixed format marker,
version, template configuration, node IDs/kinds/names/positions/settings, and wire
source/target IDs plus target port name. Named boundary recipes and output-instance
settings are included. No Python callables, executable code, pickle data, trained
weights, numerical results, undo commands or view preferences are stored.

Opening validates the whole structure, known kinds/ports, compatible types,
unique IDs, finite data and single-source inputs, then stages the graph before
replacing the current document. Files are capped at 4 MiB, 2000 nodes and 10000
wires. Unknown versions, duplicate JSON keys, missing fields, invalid endpoints and
unsupported settings are rejected. Open failure leaves the graph, filename, history
and existing results untouched. Drafts may have missing/duplicate core blocks or
invalid physics/settings; they can still be saved and reopened, but pre-training
validation reports them. Core display names remain derived from the template;
output names preserve their user-defined labels.

Saving validates the serialised structure and writes and flushes data into a temporary
file beside the destination before atomic replacement. A failed write/replacement
does not mark the document clean or replace the previous file. Pending inspector
values are committed before saving or deciding whether to discard. The filename
and unsaved indicator update only after success. Undo/redo tracks the clean save
position. Programmatic low-level scene edits remain outside this dirty tracking,
just as they are outside model undo.

Unsaved UI edits prompt Save / Discard / Cancel before File Open, user-facing
template actions, or closing. Cancelling Save As cancels the pending replacement.
The existing direct template-building helpers reset without an interactive prompt;
menu/toolbar actions route through the unsaved guard. Templates are clean Untitled
models and reset history. Opening clears history/run data and leaves optional views
closed, with the loaded graph fitted to the canvas. Save/Open/template actions are
locked during training; a close request waits for safe worker cancellation before
the unsaved decision. Panel persistence, recent files and crash recovery remain TODO.
