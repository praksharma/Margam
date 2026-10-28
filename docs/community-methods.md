# Community methods and attribution

Margam brings established community methods into one visual PINN workflow.
The tools below credit the projects used as implementation references and
inspiration. They do not imply that Margam invented these algorithms, or that
the referenced software projects necessarily originated the underlying mathematics.

| Margam feature | Community reference | How it is used here |
| --- | --- | --- |
| Burgers initial-value problem | [Raissi, Perdikaris and Karniadakis — original PINN example](https://maziarraissi.github.io/PINNs/) | Original PDE, viscosity and initial/boundary setup; current Margam training defaults and independently written PyTorch implementation. |
| Residual refinement (RAR) | [DeepXDE RAR demonstration](https://deepxde.readthedocs.io/en/v1.12.2/demos/pinn_forward/burgers.rar.html) | Greedy refinement by adding high-residual candidate points. Margam implements its own scheduled updates, point cap and duplicate checks. |
| Importance sampling for PINNs | NVIDIA Modulus / Modulus Sym, subsequently PhysicsNeMo Sym: [sampling theory](https://docs.nvidia.com/physicsnemo/26.03/physicsnemo-sym/user_guide/theory/advanced_schemes.html#importance-sampling) and [dataset implementation](https://github.com/NVIDIA/physicsnemo-sym/blob/main/physicsnemo/sym/dataset/continuous.py) | Inverse-probability correction of the sampled PDE loss. Margam currently scores PDE residuals on fresh box candidate pools; the NVIDIA reference supports other importance measures and a fixed cloud. |
| Integral squared-error loss for PINNs | NVIDIA Modulus / Modulus Sym, subsequently PhysicsNeMo Sym: [pointwise loss implementation](https://github.com/NVIDIA/physicsnemo-sym/blob/main/physicsnemo/sym/loss/loss.py) | Physical domain/boundary measures in the squared-residual loss, with Monte Carlo approximation for random samples and equal-cell quadrature for grids. |

**NVIDIA Modulus** is the community provenance name used here. The source and
theory links point to PhysicsNeMo Sym versions inspected during implementation;
they preserve the relevant sampling/loss reference rather than linking to a
renamed product's general landing page.

## Adaptation and code reuse

These three Margam implementations are independently written. No DeepXDE or
NVIDIA sampler/loss source was copied, and neither package is needed at runtime.
References guide the mathematics and behaviour; they are not claims of identical
implementations or reproduced research results.

- [Adaptive sampling](adaptive-sampling.md) describes RAR scheduling and the
  importance estimator, including differences from DeepXDE and NVIDIA's sampler.
- [Integral loss](integral-loss.md) describes physical measures, vector-component
  sums and quadrature restrictions. It integrates squared pointwise error;
  constraints on total integrated quantities are a separate feature.
- [Third-party notices](../THIRD_PARTY.md) distinguish these references from
  actual reused code, including attributed NVIDIA network adaptations.
- [Results and checks](results.md) records what was measured and tested in Margam.

When adding a community method, record its paper/project, link the exact reference,
state whether code was copied or independently implemented, and describe deliberate
differences and current limits in its guide.
