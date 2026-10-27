# Ready-wired steady heat demo

Load with **File → Examples → Heat demo**. The separate
basic template remains available through Reset Poisson template.

## Problem and configuration

The baseline equation is Δu + c = 0 on the unit square, with c=2 and u=0 on every
edge. Here u is dimensionless temperature rise, conductivity is one, and the source
is constant. This is a steady problem, not a transient heat-equation solver.

The demo uses the existing `methods/pinn.py` recipe, configured in
`templates/heat.py`: 144 grid interior points, 24 boundary samples per side, a tanh
network with hidden layers 32,32, Adam learning rate .005 and 2500 updates. The PDE
weight is 1 and mean BC weight is 10. Geometry, conditions and training settings
remain editable in their blocks. Boundary conditions are imposed softly by losses;
finite training does not imply exact satisfaction or a prescribed accuracy.

## Output wiring

```text
PINN method ──→ Training Monitor
            ├─→ Solution Plot
            └─→ Comparison ←── Reference
```

The monitor records all 2500 default updates within its 10000-update retention
limit. Views open only by explicit block action. The prediction block has no
implicit reference. Comparison requests its own evaluation points and presents
prediction, reference and signed error fields, plus RMSE and max |error|.
Prediction/reference maps share a colour scale; the error map has its own range.

## Rectangular reference

For rectangle widths Lx,Ly and a common constant Dirichlet value B, the reference
subtracts B and expands the constant forcing in odd double-sine modes. Let
ξ=x-x_min and η=y-y_min. The infinite-series coefficients are

```text
A_mn = 16 c / [π² m n ((mπ/Lx)² + (nπ/Ly)²)],  m,n odd
u = B + Σ A_mn sin(mπξ/Lx) sin(nπη/Ly).
```

The implementation retains 40 odd modes per axis (indices 1,3,...,79), evaluates
in float64 chunks, and labels the source as truncated. It is a numerical reference
from an analytical expansion, not an exact finite formula or universal error bound.

The separation-of-variables solution formula is documented in [University of
Maryland: rectangular Dirichlet Poisson equation](https://math.umd.edu/~petersd/464/html/PoissonD.html).
The constant-source odd coefficients above are our specialization of that formula;
no solver code from the page was copied.

Supported rectangle references require Dirichlet conditions on every side with
the same constant target. Mixed/Neumann rectangle references are not provided;
remove the Comparison if using a problem with no supported reference. Interval
comparisons retain their existing analytical reference.

`tests/test_rectangle_reference.py` checks zero boundaries, symmetry and selected
40-versus-80-mode values, and compares against an independent five-point finite-
difference solve on a translated, non-square rectangle with a nonzero boundary.
These are correctness checks at stated resolutions, not a rigorous truncation bound.
`tests/test_heat_demo_ui.py` verifies the wired demo, background run, monitor history,
reference metrics, comparison rendering and return to the basic template.

Measured soft/hard BC performance is recorded in the [accuracy study](heat-accuracy.md)
and [results catalogue](results.md). These measurements are separate from the
reference correctness checks and are not general accuracy guarantees.
