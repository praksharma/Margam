# Margam

A visual framework for physics-informed neural networks (formerly DeepINN).
The desktop builder runs the baseline strong-form PINN on built-in intervals
and rectangles, with editable block settings, background training and optional
block-owned output views.
The initial runnable template solves scalar Poisson problems.

See [AGENTS.md](AGENTS.md) for agent working rules, including commit attribution.

## Development rules

- Keep all project work in this repository, never in a separate ChatGPT folder.
- Use the existing root `.venv`. Check its Python version and dependencies first.
  Do not create feature-specific environments or replace an existing environment
  without an explicit reason and authorization. Current development uses Python 3.14.
- Keep source in `margam/`, tests in `tests/`, runnable examples in `examples/`,
  documentation and planning in `docs/`, and static resources in `assets/`.
  Screenshots belong in `assets/screenshots/`. Keep the repository root minimal.
- Keep the `dev` branch untouched unless explicitly authorized. Check the branch
  before making changes.
- Make small, verified commits at completed milestones.
- Build concrete supported workflows first. Do not add generic extension frameworks
  or universal plotting actions without a demonstrated need.
- Keep each PINN method in one file, reusing shared components where appropriate.
  Update the component/method catalogue when adding or changing capabilities.

## Open the project and activate the environment

Run all commands below from the repository root. On this machine:

```sh
cd /home/hell/Desktop/Margam
source .venv/bin/activate
python --version
```

The existing root `.venv` uses Python 3.14. To leave it when finished:

```sh
deactivate
```

## First-time setup or dependency updates

Reuse `.venv` when it exists. **Only on a fresh checkout without `.venv`**, create it:

```sh
uv venv --python 3.14 .venv
```

Install the project dependencies and register the package in editable mode:

```sh
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python --no-deps -e .
```

Editable mode means source edits take effect without reinstalling the package.
Check the installed environment with:

```sh
uv pip check --python .venv/bin/python
uv pip list --python .venv/bin/python
```

The environment is managed with `uv`; it does not need its own `pip` module.

## Launch the desktop builder

After activation:

```sh
python -m margam.ui
```

Or launch without activating the environment:

```sh
.venv/bin/python -m margam.ui
```

The layout keeps the block library on the left, the connected model in the centre,
selected-block properties on the right, and no default figure pane.
Operational Messages are a separate panel, initially hidden (open it from View).
Use **View → Theme → Light / Dark** to switch themes. Panels can be resized or docked.
Menus are File, Edit, View, Run and Help; only implemented actions are exposed.
File currently provides the basic-template reset, Heat demo and Quit; model Open/Save remains future work.

### Navigate the canvas

- Mouse wheel: zoom around the pointer (10–400%).
- Middle-button drag or hold Space and left-drag: pan without moving blocks.
- Tap Space or click **Fit model**: fit the entire diagram.
- Toolbar / View: **Zoom in**, **Zoom out**, **100%**.
- With canvas focus: Ctrl+Plus / Ctrl+Minus / Ctrl+0 for zoom in/out/100%.

Navigation changes only the view; it does not change the PINN or its settings.

### Ready-wired heat demo

Click **Heat demo** on the toolbar or choose **File → Heat demo**. This loads the
steady rectangular heat/Poisson example with **Training Monitor**, **Solution Plot**,
**Reference**, and **Comparison** already connected. Click **Train PINN** and open
those blocks by double-click or their right-click Open action. Views stay optional.

The demo uses unit conductivity, a constant source and zero edge temperature rise.
The Comparison shows prediction, reference and error heatmaps, RMSE and maximum
error. Its rectangle reference is an explicitly labelled **truncated sine series**
(40 odd modes per axis), valid for equal constant Dirichlet values on all sides.
It is not a universal exact solution. Ordinary prediction plots still use no reference.
The basic **Reset Poisson template** continues to load a model without output blocks.
See [heat-demo.md](docs/heat-demo.md) for equations, parameters and reference checks.

### Train your first model

1. The app opens a complete rectangle Poisson template, Δu + 2 = 0, with zero
   Dirichlet conditions on all four sides. No geometry file or mesh is needed.
2. Select **Rectangle** to change the bounds or choose **Interval**. Right-click
   a rectangle block for **Preview geometry** or **Preview samples**. These local
   2D windows open only on request and float by default; they are not available
   for intervals or arbitrary geometry. Closing one hides it; reopening reuses it.
3. Select **Interior sampler** to choose grid/random sampling, point budget and seed.
   Select **Boundary points**, choose a named side, then set its own grid/random
   method, point count and seed. Changing one side leaves the other sides and
   interior unchanged. Interval boundaries are fixed endpoints. Grid seeds are
   retained for switching methods; grids themselves are deterministic.
4. Select **PDE residual** to change the constant source. This first UI supports a
   Poisson preset; arbitrary PDE editing and pointwise forcing arrays are future work.
5. Select **Boundary condition** to assign each named side a Dirichlet value or
   outward normal derivative (Neumann). These are not conductivity-scaled fluxes.
   At least one Dirichlet side is required. Adjacent Dirichlet values on a rectangle
   must agree at shared corners; corner sampling is excluded.
6. Select **Neural network** to edit hidden-layer sizes, or **PINN method** for Adam
   learning rate, iteration count, PDE/BC weights and device.
7. Click **Validate**, then **Train PINN**. A model with no output blocks trains
   without creating predictions or plot views. Progress and errors remain available.
8. To request losses, add **Training Monitor** and connect the PINN method to it.
   To request predictions, add **Solution Plot** and connect the PINN method to it.
   Double-click an output block, right-click it for its Open action, or select it
   and choose **View → Open selected output** to open its view. Merely adding, connecting or training does not open it.

For an analytical correctness check, choose **Interval**, keep bounds `[0, 1]`,
source `2`, zero Dirichlet endpoints, hidden layers `16, 16`, learning rate `0.01`
and `1000` iterations. Set the interior grid budget to `32`. Add **Reference** and
**Comparison** blocks. Connect PINN method → Comparison's upper (model) input and
Reference → its lower input. The explicit interval reference is analytical; the Comparison reports RMSE. An ordinary Solution Plot never adds
that reference, even for intervals. Rectangle prediction plots need no reference.

Block connections are checked against the supported Poisson template before
execution. The seven core blocks are required; output and reference blocks are
optional. Unconnected output blocks are ignored; partially connected comparisons
are rejected. Drag blocks to rearrange them and drag output to input to connect.
The left library has collapsible **Geometry**, **Sampling**, **Neural networks**,
**Loss / Physics**, **Optimisation / PINN**, and **Results** groups. Click an arrow
to collapse a group; search by block or category name. Clearing search restores
your collapsed groups. Double-click a block or select it and press Enter to add it.
Optimiser settings currently belong to the PINN method; PDE and boundary-condition
blocks supply its loss terms.

Use **Edit → Delete selected** for blocks/wires. Double-clicking the library adds
blocks; core duplicates and unsupported wiring prevent training. Output instances
have independent names, evaluation resolution or retained-history limits.
**Reset Poisson template** restores the runnable default; **Fit model** fits the
canvas. This is a template-backed workflow, not yet a general graph compiler.

Training runs in a background thread. Settings and graph edits are locked during
execution, but output views can still be opened, hidden or moved. Drag a view's
panel title to detach or redock it. Closing a view hides it; its connected block
continues requesting data. Removing the block removes the request and its view.
Monitors retain only their configured number of updates; reopening a hidden view
shows retained data. Training never forces plots open or switches to them.

**Stop** retains the partial run. Closing the main window requests cancellation
and waits for the worker. The numerical model remains in memory after a run until
a model edit or new run; there is no model export/continuation UI yet. Changing
the graph or model/evaluation settings clears stale results and requires another
run. Models/results are not saved automatically; persistence and arbitrary PDE
execution remain future work. Use **View → Reset panel layout** to restore docks.

Right-click menus expose only supported actions: Properties, Delete block, Help,
and the view actions belonging to that block. Output blocks offer Open monitor,
Open plot or Open comparison. Double-clicking a computational block still opens
its inspector. Rectangle preview windows close when the shape changes to Interval,
the owner is deleted, or the template is reset. There is no general sample renderer
or implicit 3D plotting path.

See the [UI guide](docs/ui-prototype.md) for conventions and current limitations.

## Run tests

Run the full numerical, benchmark and desktop lifecycle suite:

```sh
python -m unittest discover -s tests -v
```

Run one area at a time:

```sh
python -m unittest discover -s tests -p 'test_pinn.py' -v
python -m unittest discover -s tests -p 'test_benchmarking.py' -v
python -m unittest discover -s tests -p 'test_ui.py' -v
python -m unittest discover -s tests -p 'test_block_context.py' -v
python -m unittest discover -s tests -p 'test_ui_training.py' -v
```

The UI lifecycle test uses Qt's offscreen mode internally. For an entirely
headless test invocation:

```sh
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -v
```

CUDA-specific tests skip when a usable CUDA device is unavailable.

## Solve the baseline Poisson example

```sh
python -m examples.baseline_poisson --device cpu
```

This trains the baseline PINN for 1,000 updates and prints the solution RMSE
against `u = x(1-x)`. To choose CUDA automatically when available:

```sh
python -m examples.baseline_poisson --device auto
```

## Run a repeatable benchmark

```sh
python -m examples.benchmark_poisson --device cpu --seeds 7 17 27 --iterations 1000
```

Each invocation creates a new folder under `benchmark-results/`. To choose a
specific output folder, use a path that does not already exist:

```sh
python -m examples.benchmark_poisson --device cpu --seeds 7 17 27 --iterations 1000 --output benchmark-results/my-first-run
```

Outputs include settings and environment metadata, sampled points, losses,
predictions, per-run errors/timings, and aggregate CSV/JSON summaries. Existing
output folders are rejected to protect previous runs.

See available example options:

```sh
python -m examples.baseline_poisson --help
python -m examples.benchmark_poisson --help
```

## CPU and GPU execution

The current local environment has CPU-only PyTorch. Confirm device availability:

```sh
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

For a CPU installation on a fresh environment, install the numerical dependencies
from the PyTorch CPU index before the remaining setup commands above:

```sh
uv pip install --python .venv/bin/python 'torch>=2.9,<3' 'numpy>=2,<3' --index-url https://download.pytorch.org/whl/cpu
```

GPU installation depends on the machine and driver; install a compatible CUDA
PyTorch build before using these commands:

```sh
python -m examples.baseline_poisson --device cuda
python -m examples.benchmark_poisson --device cuda --seeds 7 17 27 --iterations 1000
```

`--device cuda` requires a usable CUDA installation; `--device auto` falls back to
CPU. The old PyTorch 2.0.1/Python 3.9 dependency set is retired.

## Numerical API

Import the package as `margam`. There is no `DeepINN` import alias.
`PINNProblem`, `PointConstraint`, and `PINN` retain the working baseline API.
The older `Model(domain, network)` API and borrowed geometry stack are retired.

## Project layout and documentation

- `margam/methods/`: method implementations, currently the baseline PINN.
- `margam/components/`: problem definitions and named loss aggregation.
- `margam/nn/`, `margam/backend/`, `margam/utils/`: networks, optimisation helpers,
  and differentiation operators.
- `margam/benchmarking/`: reproducible comparison runner and metrics.
- `margam/ui/`: Qt canvas, ports, connections, panels, and themes.
- [Roadmap](docs/roadmap.md), [TODO](docs/todo.md), [UI guide](docs/ui-prototype.md).
- [Simulink evidence and interaction specification](docs/simulink-evidence.md).
- [Component catalogue](docs/components.md), [numerical engine](docs/engine.md),
  [cleanup record](docs/cleanup.md),
  [GeoSampler reuse](docs/geosampler-reuse.md).
- [Third-party attribution](THIRD_PARTY.md) and [licence](LICENSE).

Existing benchmark results remain in the ignored `benchmark-results/` folder.
