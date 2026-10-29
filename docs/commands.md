# Development and experiment commands

[User manual](index.md) · [Installation](start/install.md) · [Validation](validation.md)

Run commands from the repository root using its existing `.venv`. These are
reference commands; the [results catalogue](results.md) records measured studies
and their individual settings. Supported methods and networks belong in the
[component reference](components.md), not this command list.

## Tests and experiments

```sh
# Numerical, persistence and desktop regression suite
.venv/bin/python -m unittest discover -s tests -v

# Fully headless invocation (desktop tests also set this internally)
QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests -v

# Focused LRA numerical and desktop checks
.venv/bin/python -m unittest discover -s tests -p 'test_lra*.py' -v

# Focused SA-PINN numerical and desktop checks
.venv/bin/python -m unittest discover -s tests -p 'test_self_adaptive*.py' -v

# Baseline analytical interval example
.venv/bin/python -m examples.baseline_poisson --device cpu

# Full baseline benchmark artifacts: output folder must be new
.venv/bin/python -m examples.benchmark_poisson --device cpu --seeds 7 17 27 --iterations 1000

# Equal-budget baseline / SA comparison; save a new evidence file
.venv/bin/python -m examples.compare_self_adaptive --iterations 700 --seeds 7 11 19 --output benchmark-results/sa-poisson-new.json
.venv/bin/python -m examples.compare_lra --iterations 2000 --seeds 7 11 19 --output benchmark-results/lra-poisson-new.json
.venv/bin/python -m examples.run_helmholtz --iterations 5000 --seeds 7 11 19 --output benchmark-results/helmholtz-new.json

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
belongs in `assets/benchmarks/` and is indexed by [recorded results](results.md).

## Devices and CPU setup

```sh
.venv/bin/python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

A CUDA-enabled Torch build and a compatible NVIDIA driver are both required for
GPU training. A CPU-only build reports `torch.version.cuda == None`; selecting
`cuda` in Margam cannot add GPU support to that build. Use the application-wide
**Device** menu: Automatic uses CUDA when available, CPU otherwise. CPU and GPU
are explicit choices. It controls training, resumed training and inference across
all blocks. Command-line examples retain their explicit device options.

### Replace CPU-only Torch in the existing environment

For a driver compatible with CUDA 13.0, install the CUDA build into the same root
`.venv`; this does not create another environment:

```sh
uv pip install --python .venv/bin/python --torch-backend cu130 'torch>=2.9,<3' --reinstall-package torch
uv pip check --python .venv/bin/python
.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device')"
```

Choose a build compatible with your hardware/driver using the
[official PyTorch installation guide](https://pytorch.org/get-started/locally/).
Restart Margam after replacing an already imported Torch build. Baseline scripts
also accept `--device cuda` or `--device auto`.

For a CPU-only installation, keep the same environment and explicitly select
the CPU backend instead:

```sh
uv pip install --python .venv/bin/python --torch-backend cpu 'torch>=2.9,<3' 'numpy>=2,<3'
```

## Documentation

Build, preview and screenshot-capture commands are in
[documentation maintenance](documentation.md).

## Residual-driven sampling

```bash
python -m examples.adaptive_sampling
python -m unittest discover -s tests -p 'test_adaptive_sampling*.py' -v
python -m examples.capture_adaptive_sampling --overwrite
```

See [adaptive sampling](adaptive-sampling.md) for block wiring and estimator choices.

## Integral loss formulation

```bash
python -m examples.integral_loss
python -m unittest discover -s tests -p 'test_integral_loss*.py' -v
python -m examples.capture_integral_loss --overwrite
```

See [integral loss](integral-loss.md) for measures, components and supported samplers.

## Burgers space–time example

```bash
python -m examples.burgers
python -m examples.burgers --steps 100 --export results/burgers.pvd
python -m unittest discover -s tests -p 'test_burgers*.py' -v
python -m examples.capture_burgers --overwrite
```

See [Burgers](burgers.md) for the initial condition and physical-time export.

## Expert PINN

```bash
python -m examples.expert_pinn --device cpu
python -m examples.expert_pinn --device cuda
python -m unittest discover -s tests -p 'test_expert*.py' -v
python -m examples.capture_expert --overwrite
```

See [Expert PINN](expert-pinn.md) for the paper version, formulas and limitations.

## Expert example suite

```bash
python -m examples.expert_suite --device cpu
python -m examples.expert_suite --device cuda
python -m examples.capture_expert_suite --overwrite
```

See [case contracts and recorded results](expert-examples.md).

## Expert NTK and time marching

```bash
python -m examples.expert_extensions --device cpu
python -m examples.expert_extensions --device cuda
python -m examples.capture_expert_extensions --overwrite
```

See [controls and correctness evidence](expert-training.md).

## Scale-PINN

```bash
python -m examples.scale_pinn --device cpu
python -m examples.scale_pinn --device cuda
python -m examples.capture_scale --overwrite
```

See [Scale-PINN controls and recorded checks](scale-pinn.md).


## Supplied-point evaluation and UI captures

```bash
python -m examples.evaluation_points --device cpu --output-dir /tmp/margam-evaluation-cpu
python -m examples.evaluation_points --device cuda --output-dir /tmp/margam-evaluation-cuda
python -m examples.capture_evaluation --overwrite
```

These validate supplied-point prediction/export and capture the actual isolated
UI. See [the guide](evaluation-points.md); CUDA requires an available CUDA device.


## Prepared non-rectangular training example

```bash
python -m examples.training_data --device cpu --output-dir /tmp/margam-training-data-cpu
python -m examples.training_data --device cuda --output-dir /tmp/margam-training-data-cuda
python -m examples.capture_training_data --overwrite
```

See [Prepared training data](training-data.md) for mappings, mixed conditions and
current scope. CUDA requires an available CUDA device.


## Gradient-enhanced PINN

```bash
python -m examples.gpinn --device cpu --steps 6 --output /tmp/gpinn-cpu.json
python -m examples.gpinn --device cuda --steps 6 --output /tmp/gpinn-cuda.json
python -m examples.capture_gpinn --overwrite
```

See [gPINN](gpinn.md) for the method, starting examples and recorded checks.


## Residual-based attention

```bash
python -m examples.rba --device cpu --steps 6 --output /tmp/rba-cpu.json
python -m examples.rba --device cuda --steps 6 --output /tmp/rba-cuda.json
python -m examples.capture_rba --overwrite
```

See [RBA](rba-pinn.md) for fixed-point identity and initialization contracts.

## Jorvik geometry integration

Install a sibling Jorvik checkout in the existing environment:

```sh
uv pip install --python .venv/bin/python ../Jorvik
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m examples.jorvik_exchange --output assets/benchmarks/jorvik-exchange.json
QT_QPA_PLATFORM=offscreen .venv/bin/python -m examples.capture_jorvik --overwrite
```

See [geometry documents and NetCDF updates](jorvik.md).

## PDE collocation wiring screenshots

```sh
QT_QPA_PLATFORM=offscreen .venv/bin/python -m examples.capture_collocation --overwrite
```

See [PDE point selection](modeling/physics.md#arrows-choose-where-the-pde-is-enforced).

```sh
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m examples.pde_collocation --device cpu --output build/pde-collocation-cpu.json
.venv/bin/python -m examples.pde_collocation --device cuda --output build/pde-collocation-cuda.json
```

## Device and setting-help captures

```sh
QT_QPA_PLATFORM=offscreen .venv/bin/python -m examples.capture_device_help --overwrite
```

See [application device and expanded descriptions](use/results.md#choose-the-application-device).
