# Heat-demo numerical check

27 October 2026. CPU, float32, 2500 Adam updates, 144 interior grid points,
24 points per side, tanh 32×32 network. Evaluation: 41×41 coordinates, compared
with the explicitly truncated 40-odd-mode rectangular series. No plots were changed.

| Method | BC weight | Seed | RMSE | Maximum absolute error |
| --- | ---: | ---: | ---: | ---: |
| baseline | 10 | 7 | 0.0204918 | 0.0796043 |
| baseline | 10 | 17 | 0.0185791 | 0.0821636 |
| baseline | 10 | 27 | 0.0203726 | 0.0808272 |
| baseline | 100 | 7 | 0.0139774 | 0.0637816 |
| baseline | 100 | 17 | 0.0187275 | 0.0938645 |
| baseline | 100 | 27 | 0.014494 | 0.0599912 |
| hard_dirichlet | 1 | 7 | 0.00090309 | 0.00246223 |
| hard_dirichlet | 1 | 17 | 0.000932801 | 0.00241451 |
| hard_dirichlet | 1 | 27 | 0.000898833 | 0.00212636 |

The hard-Dirichlet method reduced field RMSE and maximum error for all three tested
seeds and enforces the supported boundary values exactly. It is available as an
explicit optional variant; the existing heat template remains baseline. Raising
only the soft boundary penalty helped two seeds but worsened seed 17's maximum
error, so a larger penalty was not made a universal default.

This is a narrow correctness/accuracy study, not a mesh-convergence or broad PDE
benchmark. The hard variant still has a non-zero collocation residual (reported
in the raw data) and is restricted to equal constant Dirichlet boxes. It does not
establish GPU equivalence or accuracy on unrelated physics.

Reproduce with the root environment:

```bash
python -m examples.benchmark_heat_accuracy --output assets/benchmarks/heat-accuracy.json
```

Raw settings/losses/errors: [heat-accuracy.json](../assets/benchmarks/heat-accuracy.json).
