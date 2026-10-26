# Margam

A visual framework for physics-informed neural networks (formerly DeepINN).
The baseline strong-form PINN and benchmark engine work; the desktop builder
is an interactive layout prototype and is not connected to training yet.

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

Double-click a library item to add a block, drag blocks to move them, and drag an
output port onto an input port to connect. Select a block to inspect it. Use
**Appearance → Light / Dark** to switch themes. Closing the window exits the app.
The rectangle preview is illustrative; graph execution and training are not yet
connected to the UI.

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
- [Numerical engine](docs/engine.md), [cleanup record](docs/cleanup.md),
  [GeoSampler reuse](docs/geosampler-reuse.md).
- [Third-party attribution](THIRD_PARTY.md) and [licence](LICENSE).

Existing benchmark results remain in the ignored `benchmark-results/` folder.
