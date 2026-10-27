# Margam

Margam (formerly DeepINN) is a visual builder for physics-informed neural networks.
Connect geometry, sampling, PDEs, boundary conditions and a PINN method, then train.
Results are requested by output blocks; no plot or reference solution is assumed.

Implemented methods are **baseline**, **hard Dirichlet**, and **self-adaptive PINN**.
Baseline and self-adaptive methods also have compact/expanded recipes. Built-in
problems cover scalar interval/rectangle physics and steady 2D incompressible flow
(u, v, p). The current executor runs one supported problem root at a time.

[Documentation index](docs/README.md) · [Desktop guide](docs/ui-prototype.md) ·
[Methods and components](docs/components.md) · [Recorded results](docs/results.md) ·
[Remaining work](docs/todo.md)

## Run the desktop

Use the single repository-root `.venv`. All commands below run from the repository
root; on this machine it is `/home/hell/Desktop/Margam`.

```sh
cd /home/hell/Desktop/Margam
.venv/bin/python --version
.venv/bin/python -m margam.ui
```

Alternatively, activate it with `source .venv/bin/activate`, run
`python -m margam.ui`, and use `deactivate` when finished. Development uses Python
3.14. Do not create environments inside UI or feature folders.

For a fresh checkout **without** `.venv`, create it once and install:

```sh
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python --no-deps -e .
uv pip check --python .venv/bin/python
uv pip list --python .venv/bin/python
```

The editable install picks up source changes. The environment uses `uv` and does
not require a `pip` module inside `.venv`. Keep an existing environment in place.

## First run

1. Choose **File → Examples**. Start with **Poisson · interval**, **Heat demo**, or
   **Self-adaptive PINN · interval Poisson**. Loading an example offers the usual
   Save / Discard / Cancel choice for unsaved edits.
2. Select blocks to edit their owned settings. Built-in intervals/rectangles need
   no geometry file or mesh. Sampling and BC assignments are separate.
3. Use **Predefined PDE** for a supported equation family, or **PDE residual** for
   restricted mathematical expressions. Choose one; a PDE preset supplies neither
   BCs nor a reference solution.
4. **Validate**, then **Train PINN**. **Stop** retains a partial run. Training locks
   model edits but leaves navigation and optional result views available.
5. Double-click a connected **Training Monitor**, **Solution Plot** or **Comparison**
   to open its view. Comparison also needs an explicit **Reference**. **CSV Export**
   writes predictions only when its export action is requested.

The startup/reset rectangle Poisson graph has no output blocks. Examples provide
explicit output wiring. Scalar results use the existing plots; coupled u/v/p
results use tables and full CSV. [Problem catalogue](docs/problem-examples.md).

### Choose a network

Select **Neural network → Architecture** for the original FCNN, angular Fourier MLP,
Fourier features, highway/modified/multiscale Fourier, DGM, SIREN, or multiplicative
Fourier/Gabor. The same choices work with baseline and SA-PINN. Each has its own
settings, preserved inside compact recipes. No neural operators or PhysicsNeMo
runtime dependency are included. [Network guide](docs/neural-networks.md).

### Compact or expanded PINNs

Use **File → Insert PINN recipe** to insert baseline or self-adaptive PINN compact
or expanded. A compact recipe contains its network and trainer. Double-click it
or use **Expand into components** to edit them on the canvas. Right-click the
trainer to collapse it again. Network widths/seed, loss weights and training
settings survive the round trip. Insertion does not replace existing method roots.

Keep **one file per recipe** in `margam/recipes/` and **one per numerical method**
in `margam/methods/`; reuse shared components. Physics examples are separate.
Hard Dirichlet is available through the original trainer's Method variant; it does
not yet have its own compact recipe. [Recipe guide](docs/pinn-recipes.md).

### Navigation and editing

| Action | Control |
| --- | --- |
| Zoom around the pointer | Mouse wheel; toolbar/View zoom actions |
| Pan with a hand cursor | Pan toggle, then left-drag; Escape leaves Pan mode |
| Temporary pan | Middle-drag or Space+left-drag |
| Fit / actual size | Tap Space or Fit model / 100% |
| Insert a block | Library drag/drop, double-click, or Enter; empty-canvas double-click or Tab for search |
| Edit wiring | Drag an output to a compatible input; delete an occupied wire before replacing it |
| Undo / redo | Ctrl+Z / Ctrl+Y or Ctrl+Shift+Z on the canvas; Edit menu |
| Copy / cut / paste blocks | Ctrl+C / Ctrl+X / Ctrl+V on the canvas |

Library categories collapse independently; clearing search restores their state.
View controls panels, light/dark themes and Reset panel layout. Vector icons keep
text labels. Right-click actions belong to the selected block; only Rectangle has
geometry/sample previews. No global figure pane or 3D viewport is provided.

### Save models and results

| File | Contents | Action |
| --- | --- | --- |
| `.margam.json` | Model graph and settings; no trained weights/results | File → Save model / Save model as / Open model |
| `.margam-run.pt` | Model, requested results, retained monitor history, network and optimiser state | File → Save run and checkpoint / Open saved run |
| CSV | Requested scalar x/y/u or coupled x/y/u/v/p predictions; no reference/error columns | Connected CSV Export block → Export CSV |

**Run → Resume checkpoint** adds updates. SA-PINN archives include point weights
and their separate Adam state. Opening a run leaves views closed. Model edits clear
stale results/checkpoints; moving blocks and renaming outputs preserve them.
External CSV/NPZ sources remain dependencies for actual resume. Layout/theme and
unsaved-draft recovery live separately in the user's Margam data directory.
[Persistence and data details](docs/ui-prototype.md).

## Commands and recorded evidence

```sh
# Numerical, persistence and desktop regression suite
.venv/bin/python -m unittest discover -s tests -v

# Fully headless invocation (desktop tests also set this internally)
QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests -v

# Focused SA-PINN numerical and desktop checks
.venv/bin/python -m unittest discover -s tests -p 'test_self_adaptive*.py' -v

# Baseline analytical interval example
.venv/bin/python -m examples.baseline_poisson --device cpu

# Full baseline benchmark artifacts: output folder must be new
.venv/bin/python -m examples.benchmark_poisson --device cpu --seeds 7 17 27 --iterations 1000

# Equal-budget baseline / SA comparison; save a new evidence file
.venv/bin/python -m examples.compare_self_adaptive --iterations 700 --seeds 7 11 19 --output benchmark-results/sa-poisson-new.json

# Pointwise network compatibility study: ten architectures × baseline/SA, one seed
.venv/bin/python -m examples.benchmark_pointwise_networks --iterations 700 --seed 7 --output benchmark-results/pointwise-networks-new.json

# Heat study (9 runs); choose a fresh output filename to preserve recorded evidence
.venv/bin/python -m examples.benchmark_heat_accuracy --output benchmark-results/heat-study-new.json

# Explicit planar GeoSampler preparation: destination directory must be new
.venv/bin/python -m examples.prepare_geosampler_2d --input /absolute/path/samples.npz --output /absolute/path/new-prepared-directory --axes x,y --labels interior wall
```

Each example supports `--help`. Baseline benchmark directories include environment,
settings, sample fingerprints, losses, predictions and aggregate metrics. The SA
comparison's optional JSON includes settings, environment, selected source hashes,
per-seed errors and weight summaries. It refuses to overwrite an existing file.
Full local experiments belong in ignored `benchmark-results/`; curated evidence
belongs in `assets/benchmarks/` and is indexed by [recorded results](docs/results.md).

Latest implementation validation: **88 tests, 85 passed, 3 CUDA skips** on the
27 October 2026 CPU environment. [Validation scope](docs/validation.md).
The SA smoke comparison improved one of three seeds and was slightly worse for the
other two; it is not a reproduction of the paper's stiff benchmarks. The heat study
found lower error for the hard-Dirichlet method on its supported boundary problem.
Do not compare either study's error or timing across different problems/protocols.

## Device availability

```sh
.venv/bin/python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The current environment has CPU-only Torch; CUDA checks skip. The results catalogue
also preserves an older GPU measurement, explicitly labelled as historical rather
than current GPU validation. On a compatible CUDA installation, use `--device cuda`
or `--device auto` in the baseline scripts. `auto` falls back to CPU.

On a fresh CPU environment, the numerical dependencies can be installed before
the remaining setup commands using:

```sh
uv pip install --python .venv/bin/python 'torch>=2.9,<3' 'numpy>=2,<3' --index-url https://download.pytorch.org/whl/cpu
```

GPU installation requires a machine-compatible driver and Torch build; do not
replace the working environment merely to read historical results.

## Project organisation

| Location | Purpose |
| --- | --- |
| `margam/methods/` | Distinctive numerical method implementations |
| `margam/recipes/` | One compact/expanded construction definition per PINN recipe |
| `margam/components/`, `nn/`, `backend/`, `utils/` | Shared data, losses, samplers, networks, optimisation and differentiation |
| `margam/templates/` | Configured physics examples and predefined equations |
| `margam/ui/`, `benchmarking/` | Desktop builder and experiment runner |
| `tests/`, `examples/` | Verification and runnable commands |
| `docs/`, `assets/` | Documentation, curated evidence and screenshots |

The package is `margam`; there is no legacy `DeepINN` import alias. Start with the
[numerical API](docs/engine.md) or [adding a method](docs/adding-a-pinn.md).
Follow [AGENTS.md](AGENTS.md): work in this repo, keep `dev` untouched, make verified
commits and keep source/results organised. Attribution: [THIRD_PARTY.md](THIRD_PARTY.md);
licence: [LICENSE](LICENSE).

Full 3D/non-planar geometry, ParaView, transient UI physics, multiple simultaneous
method roots, standalone optimiser blocks and neural-operator architectures remain
outside current support. [TODO](docs/todo.md) records deferred work.
