# Baseline PINN desktop workflow

Run from the repository root with the existing root environment:

```sh
source .venv/bin/activate
python -m margam.ui
```

## Layout and menus

The block library is on the left, model canvas in the centre, and inspector on the
right. Domain/sample previews and operational messages use the bottom dock. No
result plots are pre-created. Optional output views have their own dock widgets,
which can be tabbed at the bottom or detached/redocked elsewhere. There is no 3D
viewport. Source attribution is in ../THIRD_PARTY.md.

| Menu | Implemented actions |
| --- | --- |
| File | Reset Poisson template (Ctrl+N), Quit (Ctrl+Q) |
| Edit | Delete selected (Delete, canvas focus), select all blocks (Ctrl+A, canvas focus) |
| View | Panel visibility, Fit model, Reset panel layout, Open selected output, Theme |
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

Double-clicking a computational block exposes its inspector. Multiple selection
has no arbitrary single-block parameter editor. Full parameter dialogs and general
instance-owned core configurations remain future work.

## Optional output blocks

| Block | Inputs | Output settings and behaviour |
| --- | --- | --- |
| Training Monitor | PINN method | Live losses; retained-update limit 25–100000, default 10000 |
| Solution Plot | PINN method | Scalar predictions only; evaluation resolution 2–201 per axis |
| Reference | No input | Explicit analytical interval Poisson demonstration source |
| Comparison | Model input above; Reference input below | Prediction/reference curves and RMSE at its own evaluation points |

Outputs have stable instance identities and independent settings. A reference
source does nothing unless connected to a requested Comparison. The current demo
reference rejects rectangles; ordinary prediction views work without references.
It is not a general symbolic or imported-data reference editor.

Connect an output block before training, then double-click it to open its view.
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
solution. Rectangle plots show a predicted field and value range, not reference error.

## Validation and remaining work

Tests cover output-free execution, no implicit reference, comparison validation,
independent evaluation resolutions, monitor retention, closed-view recording,
detach/redock, deletion, menu presence, worker failures and close/cancel behaviour.
Screenshots are under assets/screenshots. General graph execution, typed core ports,
per-set sampling instances, a grouped library, persistence, arbitrary PDE/reference
sources and export remain in todo.md.
