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

Open/Save and undo/redo are not implemented and are not offered as working actions.
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
Screenshots are under assets/screenshots. General graph execution, typed core ports,
per-set sampling instances, a grouped library, persistence, arbitrary PDE/reference
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
