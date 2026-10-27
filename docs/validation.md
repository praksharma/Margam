# Validation scope

[Documentation index](README.md) · [Measured results](results.md)

## Latest implementation suite

27 October 2026: **93 tests: 90 passed, 3 CUDA skips**, full suite after LRA was
added (109.8 seconds). The preceding network milestone was 88 tests (85 passed,
3 CUDA skips); SA-only was 78 tests (75 passed, 3 CUDA skips). Runtime: Python 3.14.8, Torch 2.14.1+cpu, PySide6 6.12.0,
NumPy 2.5.2. These counts are a dated record, not a permanently fixed test count.

```sh
.venv/bin/python -m unittest discover -s tests -v
QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s tests -p 'test_self_adaptive*.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_pinn_recipes.py' -v
```

Tests instantiate `Window(state_directory=False)` or a dedicated temporary state
folder, keeping personal preferences/recovery untouched. Offscreen tests release
clipboard data before Qt teardown. CUDA checks skip without a usable CUDA runtime;
CPU success does not establish current GPU behaviour.

## Numerical regression checks

These are acceptance thresholds in tests, **not measured benchmark tables**.
For exact current setups/assertions, follow the linked source.

| Check | Acceptance / meaning | Test source |
| --- | --- | --- |
| Manufactured scalar examples | Reference PDE residual max <1e-4; prediction RMSE <0.05 after 700 CPU updates | [predefined examples](../tests/test_predefined_examples.py) |
| SA interval Poisson | RMSE <0.04 after 700 updates, with BC masks off and on | [SA numerical checks](../tests/test_self_adaptive.py) |
| SA update equations | Two-point exact λ gradient [1,4]; network descends while masks ascend | [SA numerical checks](../tests/test_self_adaptive.py) |
| SA resume | Identical loss history, masks and predictions for 30+20 resumed vs uninterrupted updates | [SA numerical checks](../tests/test_self_adaptive.py) |
| Couette baseline | Analytic residual/BC checks; each u/v/p RMSE <0.03 after 750 updates on 11×11 evaluation points | [coupled flow](../tests/test_incompressible_flow.py) |
| Rectangle reference truncation | Selected 40/80-mode values within 3e-6; zero edges and symmetry | [reference checks](../tests/test_rectangle_reference.py) |
| Independent reference comparison | Max difference <0.002 from finite differences on a translated non-square 31×21 grid | [reference checks](../tests/test_rectangle_reference.py) |
| Recipe representations | Identical numerical predictions from compact/expanded baseline configurations at equal seed/budget | [recipe lifecycle](../tests/test_pinn_recipes.py) |

SA also checks vector residual rows/targets, cap projection, optional constraint
masks, coupled-flow dispatch, rejection of nonfinite loss/gradient before updates,
and incompatible/corrupt adaptive checkpoints. These checks verify implemented
contracts; they do not reproduce stiff paper benchmarks or prove convergence.

## Desktop and persistence checks

The suite covers typed wiring, block insertion, owned parameters, copy/paste,
undo/redo, compact/expanded recipes, background training, cancellation/teardown,
explicit outputs/references, retained monitor histories, model/run save/open,
checkpoint resume, data mappings, CSV, planar GeoSampler and local recovery.
SA's desktop check saves both optimiser states, opens an archive, resumes updates
and inspects adaptive statistics. [SA workflow test](../tests/test_self_adaptive_ui.py).

UI startup or a screenshot demonstrates appearance/lifecycle, not numerical accuracy.
Screenshots are under [assets/screenshots](../assets/screenshots/). Model format 11
reads 1–10; run archives use format 2 and restricted tensor loading. Data fingerprints
and architecture/options are checked for resume, but cross-version optimiser
compatibility and arbitrary PDE well-posedness are not guaranteed.

## Documentation/evidence verification

The documentation update repeated the six-run SA comparison and saved its complete
record. Historical heat and GPU measurements were preserved with explicit provenance
limits. Local Markdown targets, JSON structure/finite metrics, recorded source hashes,
reported tables and CLI help/output preservation were checked. No production numerical
algorithm changed as part of that documentation work.

## Pointwise network additions

All ten choices pass numerical first/second-derivative checks, per-row independence,
output shapes, real u/v/p training integration and exact adaptive checkpoint resume.
Additional checks cover exposed Fourier encodings, activations, skip/gate branches,
Gabor centres, legacy FCNN identity, setting-mismatch rejection and hard-Dirichlet
architecture metadata. [Numerical tests](../tests/test_pointwise_networks.py).
The [architecture UI test](../tests/test_network_ui.py) exercises every choice through
save/open, compact conversion, clipboard and actual archive/resume. Startup remains
Torch-free until numerical execution is requested.

The [twenty-run compatibility record](../assets/benchmarks/pointwise-networks-2026-10-27.json)
contains predictions as well as metrics. Their errors and rounded documentation
table were recomputed/checked; selected source hashes and local documentation links
were verified. This one-seed CPU study is separate from regression thresholds and
from any architecture-quality or CUDA-equivalence claim.

## Illustrated guide capture

The [walkthrough](visual-guide.md) uses nine native UI captures, individually
inspected for readable controls, complete menus and visible graph/output content.
The [manifest](../assets/screenshots/guide-captures.json) records application source
`e0fde23`, unchanged application tree, offscreen/light rendering and hashes/settings.
The helper trains one real SIREN/SA interval case for 700 CPU updates, verifies
configuration preservation across recipe conversion and that output views remain
closed until requested. Captured comparison RMSE is approximately 5.32e-4 on 201
points. This documents a working UI example, not another architecture benchmark.
Only documentation and capture tooling changed in this update; the earlier 88-test
implementation result remains a dated record rather than a claim of rerunning that
full suite for screenshots.

## LRA validation

`test_lra.py` verifies max/mean over parameter entries, raw denominator (excluding
existing lambda), alpha as new-estimate fraction, cap and zero-gradient hold,
nonfinite aborts, event phase across batches, exact checkpoint continuation and
rejection of changed settings/data/tampered events. All ten networks train for a
smoke budget; a real u/v/p model also trains. Those short runs do not establish
accuracy for each architecture or flow. `test_lra_ui.py` exercises real Qt recipe
insertion, parameters, expand/collapse, clipboard/undo, model round-trip, background
training, restricted run loading, resume and optional weight-summary action.

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_lra*.py' -v
```

The configured cap100 Poisson example also participates in the existing
700-update RMSE <0.05 example check. Measured 2,000-update comparisons, including
the high-cap failures to fit, are in [results](results.md#learning-rate-annealing-pinn).
