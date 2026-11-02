# Community methods and attribution

Margam brings concrete implementations into one builder while keeping their
origins and limits visible. Algorithm inspiration does not imply copied source.
Exact source notices and retained licences are collected in
[Source attribution](../THIRD_PARTY.md); each method guide records its supported
composition, implementation differences and validation evidence.

| Area | Implementation reference and guide |
| --- | --- |
| Baseline PINNs | Raissi, Perdikaris and Karniadakis; [forward Burgers](burgers.md) |
| Self-adaptive weights | McClenny and Braga-Neto; [SA-PINN](self-adaptive-pinn.md) |
| Gradient-statistic annealing | Wang, Teng and Perdikaris; [LRA](lra-pinn.md) |
| Residual-adaptive refinement | DeepXDE; [sampling implementation and differences](adaptive-sampling.md) |
| Importance sampling | NVIDIA Modulus/PhysicsNeMo; [sampling](adaptive-sampling.md) |
| Integral mean-square loss | NVIDIA Modulus/PhysicsNeMo; [integral loss](integral-loss.md) |
| Pointwise architectures | NVIDIA-derived and independent components; [networks and provenance](neural-networks.md) |
| Expert's Guide pipeline | Wang, Sankaran, Wang and Perdikaris; [Expert PINN](expert-pinn.md) |
| Sequential residual correction | Chiu and colleagues; [Scale-PINN](scale-pinn.md) |
| Residual derivatives | Yu, Lu, Meng and Karniadakis; [gPINN](gpinn.md) |
| Residual-based attention | Anagnostopoulos and colleagues; [RBA](rba-pinn.md) |
| Inverse coefficient identification | Raissi, Perdikaris and Karniadakis; [inverse Burgers](inverse-burgers.md) |

## Inverse Burgers

The mathematical reference is the original continuous-time identification example
in [Part II §2.1](https://arxiv.org/html/1711.10566#S2.SS1). Margam's method is an
independent PyTorch implementation using shared network/Adam components. The
bundled observations derive from the authors' dataset, pinned at
`932f50a2d8ef4e80d1456bbae6887a73ff5166ef`; its MIT notice and selection provenance
are retained under `assets/training/inverse_burgers`. No upstream training code
or paper PDF is copied. The current preset and four-update checks are not paper
accuracy reproductions.

## Scale notebook input mapping

The separate Affine Fourier NN independently implements observed affine input
equations and trainable sine projections from SCALE-PINN notebooks. Source hashes,
exact mappings and architecture differences are recorded in
[the Scale guide](scale-pinn.md#affine-inputs-and-trainable-sine-features). No upstream
code/data was copied; the shared trunk is Margam's existing implementation.


## Gradient-enhanced PINN (gPINN)

Implemented independently from Yu, Lu, Meng and Karniadakis (CMA 393, 114823,
2022), [paper](https://arxiv.org/abs/2111.02801),
[authors' Apache-2.0 repository](https://github.com/lu-group/gpinn). The additional
losses differentiate the full scalar residual in coordinate directions. Source
hashes, scope and deliberate example differences are in [the guide](gpinn.md).
No upstream code, datasets, TensorFlow or DeepXDE dependency is bundled.


## Residual-based attention PINN

Implemented independently from Anagnostopoulos, Toscano, Stergiopulos and
Karniadakis, [CMA 421, 116805 (2024)](https://doi.org/10.1016/j.cma.2024.116805).
The [authors' MIT repository](https://github.com/soanagno/rba-pinns) verifies the
detached recurrence, squared weighted residual and initialization modes. Source
hashes and differences are in [the RBA guide](rba-pinn.md). No upstream code or
data is bundled. This is pointwise weighting, not Transformer attention.
