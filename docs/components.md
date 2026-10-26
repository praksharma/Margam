# Implemented components and templates

## Sampling

`margam/components/sampling.py` exposes `sample_box(lower, upper, count, method, seed)`.
It returns coordinate rows for intervals and rectangles, independent of Qt, CAD
and PyTorch. Supported methods:

| ID | Behaviour | Count |
| --- | --- | --- |
| `grid` | Cell-centred Cartesian grid, excludes boundaries | Interval: exact; rectangle: floor(sqrt(budget))² |
| `random` | Uniform random coordinates, seeded local generator | Exact budget |

Coordinates use the supplied physical bounds. Boundary sampling is separate;
the Poisson template maps 1D samples onto each rectangle side, excluding corners
for grid sampling. Interval boundaries are the two endpoints. These are initial
small-domain samplers; the broader GeoSampler catalogue is not yet imported.

## Poisson template

`margam/templates/poisson.py` supplies `PoissonConfig`, `sample_problem`,
`build_pinn`, and `interval_reference`. It assembles the existing baseline method,
not a separate training implementation. The scalar equation is Δu + c = 0 with
configurable constant c, on an interval or axis-aligned rectangle.

Each named side has a constant Dirichlet value u or an outward normal derivative
∂u/∂n (Neumann). Neumann values are derivatives, not conductivity-scaled physical
flux. Boundary target arrays and extra pointwise PDE inputs are not supported.
At least one Dirichlet side is required; pure Neumann problems need compatibility
and gauge handling. Adjacent Dirichlet targets must agree at rectangle corners.
Mixed corner regularity is not guaranteed by this validation.

The PDE mean loss has its own weight. Each boundary mean has boundary_weight /
number_of_sides, so the boundary weight controls the average across named sides.
Tanh networks, Adam, fixed samples and full-batch updates use the baseline recipe.
An analytical interval solution is available for supported BC combinations; no
analytical rectangle error estimate is claimed.

Validated by `tests/test_poisson_template.py`: sampling bounds/counts/repeatability,
outward normal sign, unsupported conditions and interval Poisson convergence.
