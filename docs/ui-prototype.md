# Desktop guide

Despite this file's historical name, this describes the current connected builder.
[Documentation index](README.md) · [Illustrated walkthrough](visual-guide.md) ·
[Commands](../README.md) · [Capabilities](components.md)

## Start with a problem

```sh
.venv/bin/python -m margam.ui
```

Choose **File → Examples** for the basic Poisson reset, steady Heat demo,
incompressible Couette demo, manufactured scalar problems, or the self-adaptive
interval example. Example loading replaces the graph after Save / Discard / Cancel
for unsaved edits. The initial/reset graph has no output blocks; examples explicitly
wire their requested outputs. See [problem definitions](problem-examples.md).

The library is on the left, canvas in the middle and selected-block properties on
the right. Operational Messages are initially hidden and can be opened from View;
training errors show them. Results and rectangle previews have their own optional
docks and can float or dock. No default figure pane or 3D viewport is created.

| Menu | Implemented actions |
| --- | --- |
| File | Model Open/Save/Save As; run Open/Save; Examples; Insert PINN recipe; Quit |
| Edit | Undo/Redo, block Copy/Cut/Paste, Delete selected, Select all blocks |
| View | Panels, zoom/100%/Fit/Pan, Reset panel layout, Open selected output, Theme |
| Run | Validate, Train, Stop, Resume checkpoint |
| Help | Quick start and About |

Light/dark themes and vector icons retain text labels. Core panel layout, window
geometry, theme and collapsed library groups persist independently of models.

## Build the graph

The expanded problem consists of domain → interior sampler → PDE residual,
domain → boundary points → boundary condition, a selected pointwise network, and a trainer consuming
the PDE/BC/network inputs. Compact recipes hide the network/trainer pair.

Settings belong to the blocks. Execution follows one method root's typed inputs;
the inspector's cached configuration is not the source of truth. Interior and
boundary sets must use the same domain source. Unconnected or duplicate core
blocks remain editable drafts but prevent training. Multiple method roots and
arbitrary cyclic simulations are not supported.

| Block | Edit in its properties |
| --- | --- |
| Domain | Interval/rectangle shape and physical bounds |
| Interior sampler | Grid, random, Hammersley or stratified method; budget and seed; optional imported points |
| Boundary points | Rectangle named side, its independent sampler/count/seed; optional point/normal mapping; interval endpoints are fixed |
| PDE residual | Restricted scalar expression or steady u/v/p residuals; blank scalar expression retains constant-source Poisson |
| Predefined PDE | Equation family, coefficient, coordinate forcing and supported diffusivity/viscosity |
| Boundary condition | Per-set/component Dirichlet, Neumann or custom operator/target; flow pressure gauge |
| Neural network | Pointwise architecture, architecture-specific options, hidden widths and independent seed |
| PINN method | Baseline/hard-Dirichlet variant, Adam settings and loss weights |
| Self-adaptive trainer | Network and mask rates, initial mask/cap, optional adaptive BCs, global loss weights |

Predefined PDE and PDE residual are alternatives on the same typed input. A preset
supplies equations, not BCs or reference solutions. Expression syntax and point/data
limits are in [components](components.md). No Python eval/code import is offered.
Neumann values are outward normal derivatives, not automatically conductivity-scaled
flux. Built-in rectangle Dirichlet values must agree at shared corners; imported
boundary consistency and outward normal orientation remain the data provider's job.

### Insert, connect and navigate

Library groups collapse separately. Search matches titles/categories and restores
prior collapsed groups when cleared. Add a block by dragging it onto the canvas,
double-clicking it in the library, or selecting it and pressing Enter. Category
headings only expand/collapse. Double-click empty canvas space or press Tab with
canvas focus for quick insertion. Search and Enter selects the first match; Down
moves into the matches. Escape/click outside cancels. New blocks are unconnected.

Drag from an output port to a compatible input. Hover to see port roles. Inputs
accept one source; outputs may fan out. Incompatible/occupied inputs and self-connections are rejected immediately. Delete an existing wire before replacing its source.
Rejected/cancelled wiring leaves current results intact. Pan/zoom is respected by
insertion and does not change block coordinates.

| Navigation | Control |
| --- | --- |
| Pointer-centred zoom | Wheel, 10–400%; toolbar/View zoom actions |
| Persistent pan | Pan toggle, left-drag; open/closed hand cursors; Escape returns to selection |
| Temporary pan | Middle-drag or Space+left-drag |
| Fit | Tap Space or Fit model |
| Actual size | 100% or Ctrl+0 on the canvas |
| Zoom shortcuts | Ctrl+Plus / Ctrl+Minus on the canvas |

Selection or inspector resizing does not reset the camera. View → Reset panel
layout restores dock placement. Navigation remains available during training.

### Compact and expanded recipes

File → Insert PINN recipe offers baseline and self-adaptive compact/expanded forms.
The library's named PINN entries insert compact forms; trainer entries are separate.
Insertion does not replace the existing method root. Collapse an existing supported
trainer/private network, or remove the old pair before wiring a new recipe.

Double-click a compact recipe, use its inspector button, or right-click → Expand
into components. Its components appear on the same canvas. Edit or replace the
supported neural network through the typed input. Right-click the trainer →
Collapse as Baseline PINN / Self-adaptive PINN to hide the pair, retaining settings
and external connections. Shared networks cannot be collapsed; hard Dirichlet is
not relabelled as a baseline recipe. Individual layout offsets are not stored in
the compact form, but Undo restores the previous layout. [Recipe guide](pinn-recipes.md).

## Request results explicitly

| Output block | What it requests |
| --- | --- |
| Training Monitor | Live raw physical losses and weighted total; retained history limit 25–100000 |
| Solution Plot | Scalar predictions or coupled u/v/p table, evaluation resolution 2–201 per axis |
| Reference | Explicit supported preset, coordinate expression or mapped reference values; no request by itself |
| Comparison | Model **and** Reference inputs; predictions/reference and error metrics at its own evaluation points |
| CSV Export | Coordinate/field predictions only; no reference/error columns; written by its export action |

Connect outputs before training. Double-click the block, use its own right-click
Open action, or select an output and choose View → Open selected output. Views may
open before/during/after a run; adding, connecting, training or loading never opens
them automatically. Closed monitors still record connected requests. Unconnected
outputs are ignored; a partially connected Comparison fails validation.

Closing a view hides it; reopening reuses it. Removing a block removes its request
and view. Monitor retention changes trim retained data; increasing the limit cannot
recover discarded records. Prediction/comparison requests are evaluated separately.
With no prediction requests, training does not create field arrays or a reference.

Reference expressions supply one scalar expression or `u; v; p` expressions for
flow. Imported reference coordinates must match the requested evaluation grid;
row order may differ, but no interpolation occurs. Choose expression or imported
data, not both. Blank reference settings use only a supported preset. The rectangle
Poisson reference is a labelled 40-odd-mode-per-axis truncated series, not a universal
exact solution. Couette reference is valid for the unchanged Couette setup.

Existing scalar interval/rectangle plots remain available; coupled fields use tables
(first 200 rows shown) and full CSV. No new field heatmaps are being developed.
For SA-PINN, the monitor total includes adaptive masks and can increase while field
error falls. Right-click the completed SA block for Adaptive weight summary; this
reports λ rather than λ² and also works for loaded run archives.

## Train, stop and diagnose

Validate checks graph structure and physics/data settings. Train runs in a worker
thread, reporting batches of up to 25 full-batch updates. Settings and graph edits
are locked; selection and explicit output views remain available. Stop requests
cancellation between batches and retains the partial run. Closing waits for the
worker before the unsaved-model decision. Failures release controls and show
Messages without creating result views. These small examples use one CPU thread.

Losses are measured **before** updates; final predictions are measured after the
completed updates. Raw losses, weighted total and independent field-error metrics
have different meanings. Compare runs with the same problem/evaluation protocol;
see [results](results.md) and [validation scope](validation.md).

## Edit history and persistence

Canvas Ctrl+Z / Ctrl+Y (or Ctrl+Shift+Z) and Edit/context actions undo/redo model
changes, up to 100 commands. Insertion, grouped moves, wires, deletion, core/output
settings, and recipe expansion/collapse are reversible. Templates reset history.
Property editors retain ordinary text shortcuts. Copy/Cut/Paste applies to selected
blocks/internal wires, creates fresh IDs on paste, and is one undoable edit.

Model/graph/evaluation edits clear stale numerical state. Moving blocks and renaming
outputs preserve it. Undo restores the graph, not prior training results. An asterisk
marks unsaved edits; saving marks the current undo position clean.

| Persistence | Contents and limits |
| --- | --- |
| Model JSON (`.margam.json`, current format 10, reads 1–9) | Owned graph settings, IDs, positions and expressions; no weights, results, layout or undo history; incomplete drafts can be saved |
| Run archive (`.margam-run.pt`, current format 2) | Model, requested predictions/metrics, retained monitor histories, network and Adam state; SA also includes masks and ascent optimiser |
| Desktop preferences/recovery | Layout/theme/library groups and per-session unsaved model drafts; no trained numerical state |

File model/run actions validate before replacing the current document, and offer
Save / Discard / Cancel where needed. Atomic writes preserve the earlier file on
failure. Model JSON is capped at 4 MiB / 2000 nodes / 10000 wires. Run files and
expanded archives are capped at 256 MiB. Runs use restricted `weights_only=True`
loading and CPU tensor mapping; invalid data leaves the current model intact.

Run → Resume checkpoint asks for additional updates and resumes on CPU. Matching
architecture/Adam settings, dtype and data fingerprint are required; SA additionally
checks masks, ascent settings, term ordering and global weights. Total updates are
limited to 100000. Archived predictions remain inspectable if external sources
moved, but actual resume requires reconstructing compatible samples/targets/normals.
No cross-version or GPU-equivalence guarantee follows from CPU resume tests.

Prepared CSV/NPZ paths are absolute references, not embedded source arrays. Use the
mapping preview to select coordinates, optional outward unit normals and values.
PDE point imports are coordinates only. Off-box normal-dependent BC operators need
explicit normals. [GeoSampler contract](geosampler-contract.md) describes the planar
bridge and its bounding-box/evaluation limitations.

On Linux, layout and recovery normally live under `~/.local/share/Margam`.
Per-session recovery files are locked; active sessions are skipped, abandoned drafts
can be recovered as unsaved models, and clean close removes that session's draft.
Recovery does not reopen plots. [Current limits](components.md) and [TODO](todo.md)
list unsupported/deferred behaviour.

All network options are documented in the [network guide](neural-networks.md),
including smooth-derivative requirements, equal-width restrictions and the recorded
compatibility study. Changing architecture resets its options as one undoable edit.
