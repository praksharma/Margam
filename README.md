# DeepINN
[![DeepINN CI](https://github.com/praksharma/DeepINN/actions/workflows/main.yml/badge.svg)](https://github.com/praksharma/DeepINN/actions/workflows/main.yml) [![docker_container](https://github.com/praksharma/DeepINN/actions/workflows/docker.yml/badge.svg)](https://github.com/praksharma/DeepINN/actions/workflows/docker.yml) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/a5c43d9b9e6a45759061ac654bdc1e3f)](https://www.codacy.com/gh/praksharma/DeepINN/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=praksharma/DeepINN&amp;utm_campaign=Badge_Grade)![Travis (.org) branch](https://app.travis-ci.com/praksharma/DeepINN.svg?branch=main)
[![Documentation Status](https://readthedocs.org/projects/deepinn/badge/?version=latest)](https://deepinn.readthedocs.io/en/latest/index.html?badge=latest) [![License](https://img.shields.io/badge/License-AGPL_v3-red.svg)](https://github.com/praksharma/DeepINN/blob/main/LICENSE)

[DeepINN](https://github.com/praksharma/DeepINN) is a deep-learning framework for solving forward  and inverse problem involving PDEs using physics-informed neural networks (PINNs).

* The geometry module has been borrowed from [TorchPhysics](https://github.com/boschresearch/torchphysics).
* [TODO list](todo.md).

# Contribution
Create a `venv` in the root of the repo. Here the assumption is that the `python` is symlink to `python3`.
```sh
python -m venv .venv
```
Activate the environment.
```sh
source .venv/bin/activate
```
Confirm that the Python path is updated.
```sh
which python
```
The `STDOUT` should point to the .venv directory. Now, upgrade the pip.
```sh
python -m pip install --upgrade pip
```
Install the required packages.
```sh
pip install -r requirements.txt
```
If you want to build the docs using the same environment, then install the relevant dependencies.
```sh
pip install -r docs/requirements.txt
```

# Modular baseline PINN

Each PINN method has one file containing its method-specific preparation,
loss evaluation, training procedure, and prediction. The first recipe is
`DeepINN/methods/pinn.py` (`PINN`). Shared components remain reusable:

- `components/problem.py`: `PINNProblem`, sampling sources, and named
  `PointConstraint` conditions, including optional derivative operators.
- `components/losses.py`: fixed weighted aggregation of named loss terms.
- `nn/`, `geometry/`, and `utils/differentialoperators.py`: networks,
  geometry/sampling, and automatic differentiation operators.
- `model.py`: compatibility adapter for `Model(domain, network)`.

Add a future method in `methods/<method>.py`, reuse the shared components,
and keep that method's distinctive behaviour together there. This baseline
introduces no registry or switches for hypothetical methods. Multi-network
methods can define their own recipe interface when implemented.

The user supplies a **strong-form PDE residual** `residual(x, prediction)`.
Weak-form and variational methods are outside this project's scope.

```python
import torch
import DeepINN as dp

# u_xx + 2 = 0, u(0) = u(1) = 0; exact solution u = x(1-x).
def residual(x, u):
    return dp.utils.laplacian(u, x) + 2

problem = dp.PINNProblem(
    residual=residual,
    collocation_points=lambda: torch.linspace(0, 1, 34)[1:-1, None],
    constraints=[dp.PointConstraint("boundary", torch.tensor([[0.], [1.]]))],
)
network = dp.nn.FullyConnected(
    [1, 16, 16, 1], "tanh", "Xavier normal", input_bounds=([0.], [1.]))
pinn = dp.PINN(problem, network, weights={"pde": 1, "boundary": 1})
pinn.compile(lr=0.01)  # CUDA when available; device="cpu" forces CPU
pinn.train(iterations=1000, display_every=250)
values = pinn.predict(torch.linspace(0, 1, 101)[:, None])
```

The default `device="auto"` selects CUDA when available and otherwise CPU.
Use `device="cuda"` to require GPU execution (failure is explicit if unavailable).
Network parameters, fixed bounds, points and targets move to the selected device.

The baseline samples once per `compile()`, applies a mean squared loss to each
term, sums the weighted terms, and performs full-batch Adam updates. Each
`train(n)` performs exactly `n` additional steps; recompiling resets the
optimiser and histories. `loss_history` contains individual unweighted terms
and the weighted `total`, measured before each update. `display_every=0`
disables progress output. Set seeds before network construction and sampling
for repeatable runs. A custom optimiser factory is accepted; closure-based
optimisers such as LBFGS require their own training procedure.

Point sources are `(N, D)` tensors or zero-argument sampler functions. For an
existing geometry sampler, use `lambda: sampler.sample_points().as_tensor`.
Residuals and constraint operators return `(N, C)` (or `(N,)` for scalar
outputs). Targets default to scalar zero; tensors must match the output shape
and callable targets receive the sampled points. Use multiple named constraints
for boundary/initial conditions and observations. Each named term has its own
mean loss, so splitting one condition into several changes the objective unless
weights compensate. Existing `Model` calls retain a single combined boundary
term through `PINNProblem.from_domain()`.

**Numerical correction:** `FullyConnected` no longer normalises using each
batch's extrema. It uses raw coordinates by default, or optional fixed
`input_bounds`. This avoids batch-dependent solutions/derivatives and undefined
single-point predictions. Old notebooks still accept the same constructor
arguments, but numerical results will change. Fixed bounds follow the model's
device and dtype. Use smooth activations such as tanh for second derivatives.

# Testing

The existing CI uses Python 3.9 and PyTorch 2.0.1. Keep NumPy below 2 for this
legacy PyTorch environment. GPU setup for the Dell (PyTorch CUDA 11.8 build):

```sh
uv venv --python 3.9 .venv
uv pip install --python .venv/bin/python "torch==2.0.1+cu118" --index-url https://download.pytorch.org/whl/cu118
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python test.py
.venv/bin/python geometry_test.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m examples.baseline_poisson --device cuda
```

The tests cover a strong-form Poisson solve against an analytical solution,
legacy geometry/domain integration, exact training counts and continuation,
named weights, shape validation, derivative conditions, dtype preservation,
and batch-independent network evaluation. The example reports solution RMSE.
GPU tests verify device placement, CPU/GPU loss agreement and Poisson convergence
on CUDA. They skip on CPU-only machines. For CPU-only installation, use the
PyTorch CPU index (`https://download.pytorch.org/whl/cpu`) with `torch==2.0.1`.
This is a baseline correctness check, not a benchmark suite.

## Reusable benchmark runner

Run the baseline on the Dell GPU with three independent seeds:

```sh
.venv/bin/python -m examples.benchmark_poisson --device cuda --seeds 7 17 27 --iterations 1000
```

Results go to a new timestamped folder under `benchmark-results/` (ignored by
Git). Set `--output <new-folder>` to choose another location. Existing output
folders are rejected to protect earlier runs.

`DeepINN.benchmarking` exposes `BenchmarkCase`, `Recipe`, and `run_benchmark`.
A case supplies a `build_problem()` factory returning a strong-form
`PINNProblem`, evaluation points, and a reference solution. A recipe supplies
`build(problem)` returning a compiled method with `device`, `train`, `predict`,
and `loss_history`. Its factory owns network construction, so the runner does
not require every method to have one network. Baseline method logic remains in
`methods/pinn.py`.

```python
from DeepINN.benchmarking import BenchmarkCase, Recipe, run_benchmark

case = BenchmarkCase("my_problem", build_problem, evaluation_points, reference_solution)
recipes = [Recipe("baseline", build_baseline, config={"lr": 0.001})]
results = run_benchmark(case, recipes, seeds=[7, 17, 27], iterations=1000,
                        output_dir="benchmark-results/my-first-comparison")
```

For each seed, the runner materialises all samplers and target functions once,
saves the data, and gives every recipe an independent copy. It resets Python,
NumPy and PyTorch seeds before each recipe factory. Fixed evaluation tensors
allow the same reference mesh across seeds; callable evaluation sources are
sampled per seed. The case defines the training/evaluation split explicitly;
the runner does not infer one from data. Seeds improve repeatability but do not
guarantee identical floating-point results across hardware.

Each output folder contains:

- `manifest.json`: case/recipe settings, seeds, library versions, Git revision
  and dirty state, and relevant numeric backend settings. Supply architecture,
  optimiser, and other hyperparameters in recipe `config` for a useful record.
- `seed-<n>/samples.npz` and `samples.json`: training coordinates and targets,
  evaluation coordinates, references, constraint names, and a data fingerprint.
- `seed-<n>/<recipe>/loss_history.csv`: named training losses per update.
- `seed-<n>/<recipe>/predictions.npz`: evaluation points, predictions and reference.
- `seed-<n>/<recipe>/result.json`: device/GPU, setup/training/prediction times,
  RMSE, MAE, maximum absolute error and relative L2 error.
- `summary.json` / `summary.csv`: individual run results.
- `aggregate.json` / `aggregate.csv`: per-recipe means and sample standard
  deviations across successful seeds, with success/failure counts. A single
  seed has no sample standard deviation (`null` in JSON).

Metrics pool all evaluation points and output components. Relative L2 is
undefined for an all-zero reference and is recorded as `null`. Training losses
are recorded before their corresponding updates; solution errors use the final
network. CUDA timings synchronise the device, include history collection, and
exclude writing artefacts. They are end-to-end measurements, not kernel timings.
Failures are saved and re-raised, rather than silently omitted from comparisons.
The example is an analytical Poisson check; no external dataset has been chosen.

## Docker image
Pull the image with suitable tagname. The image is available [here](https://hub.docker.com/r/prakhars962/deepinn).

```sh
docker pull prakhars962/deepinn:tagname
```
### CPU Only
The image opens a jupyter server by default. 
```sh
docker run -p 8888:8888 prakhars962/deepinn:pre-release
```

You can override the jupyter server entrypoint using the following command.
```sh
docker run -it --entrypoint /bin/bash prakhars962/deepinn:pre-release
```
### GPU passthrough
First install `nvidia-docker` using this [guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#step-2-install-nvidia-container-toolkit). 

Now run the container with `nvidia-docker`.
```sh
nvidia-docker run -it --entrypoint /bin/bash prakhars962/deepinn:pre-release
```
This command will bind the `pwd` to `/workspace/tutorials` and open a jupyter-lab with GPU support.
```sh
nvidia-docker run -v $(pwd):/workspace/tutorials -p 8888:8888  prakhars962/deepinn:pre-release
```
Alternatively, one can run interactive session.
```sh
nvidia-docker run -v $(pwd):/workspace/tutorials -it --entrypoint /bin/bash  prakhars962/deepinn:pre-release
```

### Tagless copy
Each time you pull the updated image, docker will create a tagless copy of the old one. 
```sh
╰─ docker images                              
REPOSITORY            TAG           IMAGE ID       CREATED             SIZE
prakhars962/deepinn   pre-release   886808706155   4 minutes ago       6.99GB
prakhars962/deepinn   <none>        0bb744f6159e   38 minutes ago      6.99GB
prakhars962/deepinn   <none>        4ffbb67f8447   About an hour ago   6.8GB
prakhars962/deepinn   <none>        fe16ca34f9d9   About an hour ago   6.8GB
```
The only solution is to delete them one by one using the IMAGE_ID.
```sh
docker image rm -f IMAGE_ID
```

### UML diagram
The UML diagram is generated using `pyreverse`. This provides an overview of the codebase.

```sh
cd UML\ diagrams
pyreverse -o pdf ../DeepINN/
```