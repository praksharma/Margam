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

## Documentation

Build, preview and screenshot-capture commands are in
[documentation maintenance](documentation.md).
