# Baseline PINN desktop workflow

Run from the repository root with the existing root environment:

```sh
source .venv/bin/activate
python -m margam.ui
```

## Layout and blocks

Left: block library. Centre: model canvas. Right: editable properties for the
selected block. Bottom: Domain & samples, Losses, Solution and Messages tabs.
There is no 3D viewport. The light/dark theme is adapted from GeoSampler/PyniteGUI;
see ../THIRD_PARTY.md for provenance.

| Block | Editable properties / responsibility |
| --- | --- |
| Rectangle / Interval | Domain shape and bounds |
| Interior sampler | Grid/random, interior point budget, seed |
| Boundary points | Named sets; points per rectangle side; interval endpoints |
| PDE residual | Scalar Poisson preset Δu+c=0, constant c |
| Boundary condition | Constant Dirichlet or outward Neumann derivative per side |
| Neural network | Hidden-layer sizes; fully connected tanh, Xavier normal |
| PINN method | Baseline full-batch Adam, iterations, learning rate, weights, device |
| Solution | Numerical results below the canvas |

Sampling produces point sets independently of boundary-condition assignment.
Grid rectangle counts use complete Cartesian layers, so a budget of 70 produces
64 points. Grid boundary samples exclude corners; interval boundaries are exactly
the two endpoints. The preview displays actual sampled coordinates, with display
subsampling for large clouds; training still receives all generated coordinates.

Boundary targets are u for Dirichlet and ∂u/∂n for Neumann. Normals are analytical
for built-in domains. Neumann here does not mean conductivity-scaled flux. Pure
Neumann problems are rejected pending compatibility and gauge handling. Adjacent
rectangle Dirichlet values must agree at shared corners. No general corner
regularity claim is made for mixed conditions.

## Train

The initial rectangle template is ready to run. Select blocks to configure the
model, click Validate, then Train PINN. No external geometry or mesh is needed.
For the original analytical Poisson check, choose Interval on [0,1], source 2,
zero Dirichlet endpoints, 32 interior grid points, hidden layers 16,16, learning
rate .01 and 1000 iterations. The default weights are one for PDE and mean BC.

Each side has a separate named loss. Its weight is mean_boundary_weight divided
by the number of sides. Loss plots show weighted total, raw PDE mean and the raw
average of boundary means, on a logarithmic axis. Losses precede their updates.

The UI worker calls the existing `methods/pinn.py` recipe in batches of up to 25
updates, emits numeric progress and evaluates the final network. The UI thread
never receives a network or live tensors. CPU thread count is set to one for these
small examples. One job runs at a time; model edits are locked during execution.

Stop requests cancellation between batches and displays the partial prediction.
Closing during a job waits for cancellation rather than destroying a running
thread. Errors appear in Messages and release the controls. Every new run creates
a fresh network/optimiser; continuation is not exposed in this UI yet.

Interval results include an analytical reference for supported constant-source
D/D and D/N combinations, and RMSE. Rectangle results are a predicted field on a
41×41 evaluation grid, with a labelled value range, not an analytical error check.
Numerical quality still depends on samples, weights, architecture and training.

## Connections and model edits

The supported template needs one of each block and its prescribed dependency
connections. Validation rejects missing/extra connections, duplicate blocks and
unsupported configurations. Connections therefore determine whether the template
can run; they are not silently ignored. This first implementation assembles that
single supported template, not arbitrary graphs or custom PINN methods.

Drag blocks, connect output to input, select wires/blocks and Delete selected.
Fit model fits the canvas; Reset Poisson template restores defaults. Added library
blocks must form a valid supported model before training. Configuration changes
clear old plots. Graph changes invalidate the previous solution. Layout resets,
model settings and numerical results are currently in memory only.

## Validation and remaining work

`tests/test_poisson_template.py` checks sampling, normal orientation, separation of
point sets from BC assignment, invalid problems and interval convergence.
`tests/test_ui_training.py` exercises real background interval/rectangle solves,
responsive event processing, graph rejection, errors and close/cancel behaviour.
Screenshots live in assets/screenshots and are generated with the real Qt widgets
in offscreen mode. The current environment has CPU PyTorch; GPU validation remains
outstanding.

Future work: saved models/results, arbitrary PDE functions and data adapters,
typed ports and general graph execution, additional methods, undo/redo, broader
sampling and GeoSampler integration. The UI currently exposes only the two
implemented small-domain sampling methods listed in components.md.
