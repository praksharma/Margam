# Self-adaptive PINN (SA-PINN)

Implemented from McClenny and Braga-Neto, *Self-Adaptive Physics-Informed Neural
Networks using a Soft Attention Mechanism*, arXiv:2009.04544v5 (18 June 2024).
The implementation follows section 3, equations (10)–(20), and the Adam
discussion on page 6 of the supplied v5 paper.

Primary references:
- [Paper](https://arxiv.org/abs/2009.04544)
- [Authors' implementation](https://github.com/levimcclenny/SA-PINNs)
- [Helmholtz implementation](https://github.com/levimcclenny/SA-PINNs/blob/master/Helmholtz/helmholtz.py)

No code was cloned or copied. The authors' Helmholtz source was consulted to
confirm its squared multiplicative mask and mean reduction. This is an independent
PyTorch implementation, with Margam's existing component and boundary conventions.

## Method

Each fixed PDE sample i has its own positive initial weight λᵢ. With the squared
mask m(λ)=λ², a physical term is `mean_i(λ_i² * mean_c(error_ic²))`.
One mask is shared across the residual components at a point for coupled u/v/p.
Boundary/constraint terms can optionally receive their own per-point masks.
Unmasked terms retain ordinary MSE. Margam's existing global PDE/mean-BC weights
(and pressure-gauge weight for flow) multiply the corresponding terms.

For each update, both gradients are evaluated at the same pre-update state:

1. Minimise the masked objective with respect to neural-network parameters using Adam.
2. Maximise that same objective with respect to λ using a separate Adam optimiser.
3. Project λ to [0, maximum] to bound the polynomial mask numerically.

This is pointwise adaptation, not one learned scalar per loss group. Points are
sampled once per compile, and ordering/counts remain fixed throughout training and
resume. Network/point preparation and physical sample evaluation reuse the baseline;
its training loop does not contain an SA branch.

Defaults: λ=1, mask ascent rate 0.005, maximum λ=1000, PDE masks enabled, BC masks
disabled. The paper permits fixed or adaptive subsets and constant or random initial
weights; this implementation supports configurable positive constant initialisation.
The cap follows the paper's instruction to bound polynomial masks against overflow.
It is a practical projection, not an additional theoretical mask family. The cap
must also fit the network dtype's squared range. Nonfinite losses or gradients stop
before either optimiser steps.

The paper writes half-sums in equations (11)–(13); Margam retains mean-normalised
losses, consistent with the authors' Helmholtz code, plus Margam's existing global
weights. Consequently, rates and results are not a literal paper reproduction.

## Desktop and recipe

- One method file: `margam/methods/self_adaptive.py`.
- One recipe file: `margam/recipes/self_adaptive.py`.
- File → Insert PINN recipe → Self-adaptive PINN · compact / expanded.
- File → Examples → Self-adaptive PINN · interval Poisson is a ready-wired problem.
- Compact settings expose steps, network learning rate, device and mask ascent rate.
- Expand to edit the network, mask initial value/cap, BC adaptation and loss weights.
- Right-click the SA trainer to collapse it, retaining all parameters/connections.
- After training, right-click either SA block → Adaptive weight summary for λ
  minimum/mean/maximum per term. It also works after opening a saved run.

Optional Training Monitor records raw per-term MSE and the masked, globally weighted
total. Because λ grows, total loss can increase while prediction error decreases.
Use the explicitly wired Reference/Comparison to assess accuracy. Views stay closed
until opened. No field heatmaps were changed.

Model format 10 reads formats 1–9. Run archives retain both Adam states, masks,
term ordering, adaptive settings, sample fingerprint and histories. Resume rejects
changed samples, rates, caps, term ordering or global loss weights. Archive inspection
validates shapes without reopening external data files; actual resume reconstructs
samples and checks the fingerprint.

## Numerical checks and limits

Tests check the pointwise gradient analytically on a two-point problem, opposite
network/mask update directions, scalar/vector targets, optional constraint masks,
cap projection, nonfinite failure, corrupted checkpoint rejection, exact uninterrupted
versus resumed training, and interval prediction error. Desktop tests exercise both
recipe forms, save/open, clipboard, actual training, archive/reload and resume.

The [results catalogue](results.md#baseline-vs-self-adaptive-pinn) is the canonical
comparison table, including per-seed errors, aggregate/median statistics, final λ
summaries, environment, and interpretation. The [raw evidence file](../assets/benchmarks/self-adaptive-poisson-2026-10-27.json)
preserves full-precision values and reproduction settings. A repeated six-run
comparison matched our earlier values: SA improves one seed and is slightly worse
on the other two. The better mean is dominated by that one seed; the median is worse.

```sh
.venv/bin/python -m examples.compare_self_adaptive --iterations 700 --seeds 7 11 19 --output benchmark-results/sa-poisson-new.json
```

Use a new filename; existing evidence is never overwritten by this command.
The [validation guide](validation.md) separates regression thresholds from actual
measurements. The paper's Burgers/Allen–Cahn/Helmholtz benchmark protocols were not
reproduced. Optional L-BFGS, GP-based resampling, minibatches, random mask
initialisation, alternative mask families and transient UI problems are not
implemented. Existing interval/rectangle scalar physics and steady u/v/p are
supported; no 3D or ParaView work is included.
