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
| Finite Basis domain decomposition | Moseley, Markham and Nissen-Meyer; Dolean, Heinlein, Mishra and Moseley; [FBPINN](fbpinn.md) |
| 3D turbulence & PirateNet / SOAP | Wang, Sankaran, Fan, Stinis and Perdikaris; [3D sphere flow & JAXPI2](sphere-turbulence.md) |

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

## Doubly-sketched Gauss–Newton (DSGNAR)

Independent PyTorch implementation from Webb, Jerad and Cartis,
[arXiv:2607.02194v1](https://arxiv.org/abs/2607.02194). The
[DSGNAR guide](dsgnar.md) records the inspected repository revision, algorithm
ambiguities, implementation choices and validation scope. No upstream code/data
is copied; the reference checkout has no declared top-level source licence.


## Finite Basis PINN (FBPINN)

Independent PyTorch implementation of overlapping partition-of-unity domain
decomposition and active-subdomain marching from Moseley, Markham and
Nissen-Meyer, *Advances in Computational Mathematics* **49**, 62 (2023),
[arXiv:2107.07871](https://arxiv.org/abs/2107.07871), together with the optional
two-level coarse-to-fine multilevel hierarchy from Dolean, Heinlein, Mishra and
Moseley, *Computer Methods in Applied Mechanics and Engineering* **429**, 117116
(2024), [arXiv:2306.05486](https://arxiv.org/abs/2306.05486). No upstream JAX
code or datasets are copied. The [FBPINN guide](fbpinn.md) documents the default
$C^2$ squared raised-cosine window alongside paper sigmoid ($C^\infty$) and
raised-cosine ($C^1$) windows, boundary handling, progressive freezing rules and
measured equal-budget benchmarks.


## Burgers reference quadrature

The optional [Burgers reference example](burgers.md#optional-reference-data-example)
uses an independent Cole–Hopf implementation. The setup follows
[Raissi et al.](https://maziarraissi.github.io/PINNs/); the Gaussian/Hermite integral
is documented by [Burkardt](https://people.sc.fsu.edu/~jburkardt/py_src/burgers_exact/burgers_exact.html).
No upstream solver code or arrays were copied into the generated reference CSV.
Comparison against the already retained MIT-licensed inverse observation fixture
validates this evaluator, not the short-run PINN accuracy.

## 3D turbulence, PirateNet and SOAP (`margam.research.jaxpi2`)

`margam.research.jaxpi2` is a PyTorch research-API implementation of Wang,
Sankaran, Fan, Stinis and Perdikaris (2025), *Simulating Three-dimensional
Turbulence with Physics-informed Neural Networks*
([arXiv:2507.08972v2](https://arxiv.org/abs/2507.08972v2)), audited against
`sifanexisted/jaxpi2` (`77a5c1315a056388271822c35ad512a5a192b60d`) and
`haydn-jones/SOAP_JAX@v0.2.1` (`45145bcadb20864e821b87999c306e885bed7171`).
No JAX/Flax/Optax code or CFD arrays are bundled in the repository. See
[3D sphere flow and JAXPI2 turbulence methods](sphere-turbulence.md) for the
628-check fixed-point parity verification, AMD ROCm `rocSOLVER` eigensolver/QR
hardening, and $\text{Re}=300$ sphere-flow MI300X results.
