# Implemented components and templates

[Documentation index](README.md) · [Measured results](results.md) · [Validation](validation.md)

## Capability index

This is the implemented component catalogue. Check this index before adding a
method or sampler; record new capabilities and their limits here in the same change.
A Python API capability is not automatically an executable UI block option.

| Component | Implementation | Current support and limits |
| --- | --- | --- |
| Learning-rate annealing PINN | `margam/methods/learning_rate_annealing.py`; recipe `margam/recipes/learning_rate_annealing.py` | Max PDE / mean raw constraint parameter-gradient balancing, detached scalar coefficients, EMA, cap/floor and exact resume; existing networks; no base-rate scheduler; see [LRA](lra-pinn.md) |
| Self-adaptive PINN | `margam/methods/self_adaptive.py`; recipe `margam/recipes/self_adaptive.py` | Fixed-sample per-point squared masks, Adam descent/ascent, optional BC adaptation and full resume; no L-BFGS or GP resampling; see self-adaptive-pinn.md |
| Baseline recipe | `margam/recipes/baseline.py` | One definition for compact/expanded network + trainer; Adam only, replaceable supported tanh network; physics and outputs external; see pinn-recipes.md |
| Hard Dirichlet PINN | `margam/methods/hard_dirichlet.py` | Equal constant Dirichlet boxes; exact boundary transform; baseline Adam reused; mixed/imported/spatial BCs rejected |
| Steady incompressible template | `margam/templates/incompressible_flow.py` | Explicit u/v/p momentum/continuity residuals, component BCs and pressure gauge; no transient/turbulence model |
| Baseline PINN | `margam/methods/pinn.py` | Strong-form residuals; fixed samples; full-batch updates; named constraints; no adaptive resampling, weak form or closure-based LBFGS |
| Problem / constraint | `margam/components/problem.py` | Point arrays or callable sources; residual and optional constraint operators; scalar/matching tensor/callable targets; arbitrary user functions currently require Python |
| Grid / random / Hammersley / stratified | `margam/components/sampling.py` | Intervals and axis-aligned rectangles; physical coordinates; no CAD runtime or adaptive samplers |
| Loss aggregation | `margam/components/losses.py` | Fixed weighted sum of named scalar terms; Python supports finite non-negative weights; UI currently requires positive PDE/BC weights |
| Loss metric / optimiser | `margam/backend/utils.py` | Built-in MSE and Adam only; method accepts an optimiser factory in Python; no standalone optimiser block, scheduler or UI optimiser selection |
| Pointwise neural networks | `margam/nn/`; [network guide](neural-networks.md) | Original FCNN plus angular/Fourier, highway/modified/multiscale Fourier, DGM, SIREN and multiplicative Fourier/Gabor; architecture-specific settings; no neural operators |
| Helmholtz preset/example | `margam/templates/predefined_pdes.py`, `margam/templates/examples.py` | Positive mass term Δu+k²u=f; explicit 2D paper problem on [−1,1]² with existing Fourier NN and LRA; soft zero BCs; see [Helmholtz guide](helmholtz.md) |
| Poisson template | `margam/templates/poisson.py` | Scalar constant-source interval/rectangle problem; constant Dirichlet or outward derivative BCs; independent rectangle-side sampling |
| Heat demo | `margam/templates/heat.py` | Configures the same Poisson method and explicit outputs; no transient heat solver |
| Output contracts | `margam/components/outputs.py` | Explicit prediction, comparison and monitor requests; scalar plots or u/v/p tables; explicit CSV export; reference required only for comparison; ParaView deferred |

The dock, drag/drop and quick-insert popup share the supported title/kind list in
`margam/ui/block_catalogue.py`. That list is specific to the current builder; it
does not describe arbitrary solver capabilities. Sampler property choices come
from `SAMPLERS` in the sampler module and include descriptive tooltips. Implemented capabilities and limits are shared across the current library,
inspector and Help;
this document does not imply a generic component registry.

## Sampling

`margam/components/sampling.py` exposes `sample_box(lower, upper, count, method, seed)`.
It returns coordinate rows for intervals and rectangles, independent of Qt, CAD
and PyTorch. Supported methods:

| ID | Behaviour | Count |
| --- | --- | --- |
| `grid` | Cell-centred Cartesian grid, excludes boundaries | Interval: exact; rectangle: floor(sqrt(budget))² |
| `random` | Uniform random coordinates, seeded local generator | Exact budget |
| `hammersley` | Seeded randomly shifted Hammersley points | Exact budget |
| `stratified` | Seeded equal-volume recursive strata | Exact budget |

Coordinates use the supplied physical bounds. Boundary sampling is separate;
the Poisson template maps 1D samples onto each rectangle side, excluding corners
for grid sampling. Interval boundaries are the two endpoints.

Rectangle-side recipes are `BoundarySampling(points, method, seed)` values stored
in `PoissonConfig.boundary_sampling`, keyed by `left`, `right`, `bottom`, or `top`.
Each side is independent of `sampling` and `seed` (the interior settings). For an
unconfigured side, `boundary_recipe(side)` uses a grid, `boundary_points` as the
default count, and stable seeds 8/9/10/11 respectively. This preserves existing
all-grid templates and their point counts. Unlike earlier versions, changing the
interior method or seed no longer changes boundary samples. An explicit side recipe
also overrides the default count. Grid seeds are stored but unused. `network_seed` independently controls network initialisation. In the low-level
config, `network_seed=None` falls back to `seed`; UI network blocks store an
explicit network seed.

Supported budgets are 1–100000 interior points and 1–10000 per rectangle side.
Side sampling is 1D, so every implemented method returns the exact requested side count.
Random uses the sampler's half-open coordinate range; grid samples exclude corners.
The Boundary points inspector selects a named set then edits only that recipe.
Switching to an interval retains rectangle recipes but uses only its two endpoints;
switching back restores them. Template reset clears overrides. Boundary conditions
remain a separate assignment step. The optional Rectangle sample preview and the
trainer both use `sample_problem`, so they consume the same recipes.

```python
from margam.templates.poisson import BoundarySampling, PoissonConfig, sample_problem

config = PoissonConfig(points=70, sampling='random', seed=42)
config.boundary_sampling['left'] = BoundarySampling(30, 'random', 123)
config.boundary_sampling['top'] = BoundarySampling(12, 'grid', 11)
interior, boundaries = sample_problem(config)
# 70 interior; 30 left; 12 top; 16 each on right and bottom.
```

These are small-domain samplers; the broader GeoSampler catalogue is not yet imported.

## Poisson template

`margam/templates/poisson.py` supplies `PoissonConfig`, `sample_problem`,
`build_pinn`, and `interval_reference`. It assembles baseline, hard-Dirichlet,
self-adaptive or LRA methods from shared physics; it does not duplicate their trainers. The scalar equation is Δu + c = 0 with
configurable constant c, on an interval or axis-aligned rectangle.

Each named side has a constant Dirichlet value u or an outward normal derivative
∂u/∂n (Neumann). Neumann values are derivatives, not conductivity-scaled physical
flux. Scalar boundary targets can be constant, coordinate expressions or explicitly
mapped CSV/NPZ values. Additional pointwise PDE input arrays are not supported.
At least one Dirichlet side is required; pure Neumann problems need compatibility
and gauge handling. Adjacent Dirichlet targets must agree at rectangle corners.
Mixed corner regularity is not guaranteed by this validation.

The PDE mean loss has its own weight. Each boundary mean has boundary_weight /
number_of_sides, so the boundary weight controls the average across named sides.
Tanh networks, Adam, fixed samples and full-batch updates use the baseline recipe.
An analytical interval solution is available for supported BC combinations; a rectangular reference is available through an explicit truncated sine series
for equal constant Dirichlet boundaries.

Validated by `tests/test_poisson_template.py`: sampling bounds/counts/repeatability,
outward normal sign, unsupported conditions and interval Poisson convergence.

## Explicit output requests

`margam/components/outputs.py` defines OutputRequest and ReferenceSource without Qt.
`evaluate_outputs(model, config, requests)` evaluates only declared prediction or
comparison requests; an empty request list or monitor-only list never calls predict.
Each request has its own identifier and 2–201 evaluation points per coordinate axis.
Views support finite scalar outputs and the explicit u/v/p field contract. Comparison requires an explicit
callable reference with finite values matching those evaluation points. This first
contract does not provide interpolation, coordinate conversion or unit conversion.

The generic UI worker no longer imports or constructs the analytical interval
reference. References are made explicitly by a Reference block using the Poisson
template helper: analytical intervals or a truncated series for supported rectangles. Monitors have independent bounded UI histories; their visibility
is separate from recording. See ui-prototype.md for menus and view lifecycle.

## Editor connection contracts

`margam/ui/ports.py` defines the current semantic port types: geometry, interior
points, boundary points, residual, constraints, network, PINN run, and reference.
These prevent incompatible editor wires; they do not add tensor-shape validation,
units or a general execution engine. Every destination has one source; method and
reference outputs can branch to distinct consumers. See ui-prototype.md for the
full port table. Numerical execution still assembles the validated Poisson template
through `build_pinn`, not by evaluating arbitrary block graphs.

## Restricted scalar expressions

`components/expressions.py` interprets an allowlisted AST directly. It accepts
finite constants, x/y, u (operators only), c, pi, nx/ny, arithmetic, constant powers
from -8 through 8, sin/cos/exp/tanh/sqrt/abs, D(value, coordinate), laplace(value)
and dn(value). Expressions are capped at 512 characters, 128 AST nodes and nesting
20. Attributes, indexing, imports, Python execution, comprehensions and arbitrary
calls are rejected. These structural limits are not a mathematical well-posedness
proof: singular or non-finite evaluations still fail numerical validation/training.

PDE expressions define a scalar residual equal to zero. BC Custom expressions
define the operator equated to the target, e.g. u + dn(u); target expressions depend
on coordinates/constants only. The current scalar template retains a Dirichlet
anchor and checks adjacent rectangular Dirichlet target values at corners. Preset
references are unavailable for custom physics. Other vector systems and additional pointwise input
arrays remain separate future work. Tests cover manufactured residuals, derivatives,
normal signs, Robin operators, corner compatibility and actual custom-PDE training.

GeoSampler-derived `hammersley` and `stratified` samplers are also available for
interior and side recipes. Both return exact counts, are seeded and map unit designs
onto the supplied physical bounds. Hammersley uses a random toroidal shift;
stratified uses an exact-N equal-volume recursive partition. Grid/random retain
their previous sequences. No SciPy or Gmsh dependency was introduced. See
THIRD_PARTY.md for the upstream revision and licence.

The current block capability descriptions in `ui/block_catalogue.py` drive library
selection details, tooltips, the inspector summary and Help, including parameters,
port roles and implementation limits. Sampler dropdowns still derive from SAMPLERS.

## Prepared point/reference data

`components/prepared_data.py` reads CSV with explicit header names or NPZ with
`allow_pickle=False` and explicit array keys. It validates finite (N,D) coordinates,
optional unit normals with matching rows, and optional scalar targets or declared
three-component flow reference values. Files
and expanded NPZs are capped at 64 MiB; 100000 rows maximum. DataMappingDialog shows
a mapped row preview before applying it. PDE points import coordinates only.
Boundary imports can supply targets and normals; normal-dependent off-box
operators without explicit normals are rejected. Imported targets and expression targets cannot both be active.
No boundary geometry, orientation, physical units or CAD topology is inferred.

Imported references match and reorder explicit evaluation coordinates with
scale-aware floating-point tolerance; duplicates/mismatches fail instead of
silently interpolating. External paths remain dependencies, not embedded arrays
in model JSON. Run checkpoints fingerprint the compiled points, targets and imported normals and
reject changed external data. Current model files migrate prior defaults; v2 run files
include fingerprints, while v1 builtin-only checkpoints can be migrated.

## Method variants and CSV output

`methods/hard_dirichlet.py` defines HardDirichletPINN and its bounded-domain output
transform: boundary value + product(4*t_i*(1-t_i))*network(x). It enforces equal
constant Dirichlet targets for intervals/rectangles exactly, without reference
solution data. The shared baseline training loop is reused. Mixed operators,
imported boundary sets and spatial targets are rejected for this variant; use the
baseline for those cases. The variant is chosen explicitly in the method block.

CSV Export is an explicit scalar output request. Prediction data is evaluated
only when connected; opening the block writes CSV only after a user-selected path.
It never creates a plot view or automatic output file. File replacement is atomic.
The initial CSV columns are x[,y],u; no units, extra fields or ParaView assumptions
are inferred. Export requests participate in saved models and run archives.

## Steady incompressible 2D flow

The selected coupled contract is fields u,v,p over coordinates x,y, with positive
kinematic viscosity nu and density-normalised pressure. The residual preset is
u*u_x + v*u_y + p_x - nu*laplace(u),
u*v_x + v*v_y + p_y - nu*laplace(v), and u_x+v_y. Restricted expressions expose
u/v/p components without Python execution. Targets remain coordinate-only.
`templates/incompressible_flow.py` assembles a 3-output network and selects the
baseline, self-adaptive or LRA method; it is a physics template, not a duplicate trainer.

Each named set has optional u/v/p None/Dirichlet/Neumann/Custom operators. Each
velocity component currently requires a Dirichlet anchor. A separate pressure point
and positive gauge weight remove pressure's constant ambiguity. The PDE loss is
the mean across the three residual components; velocity/pressure boundary terms
remain named, and the pressure gauge has its own term/weight. Checks cover syntax,
finite settings and shared built-in corner values, not general flow well-posedness.

The Couette demo explicitly prescribes u=y, v=0 and a pressure gauge. Its optional
Reference source is valid only for that unchanged residual/BC preset; edited flow
requires an explicit compatible expression/data reference. Vector results retain
field names, per-field RMSE/max errors, table presentation and full CSV output.
No extra heatmap, 3D renderer, turbulence model, transient time or automatic units
are introduced. Tests verify analytic residuals/constraints, actual training,
pressure-gauge validation and UI/model/run/CSV round trips.

Flow inputs and residuals must be consistently scaled or nondimensionalised. No
automatic units or equation balancing are inferred; vector comparisons report
per-field errors so velocity and density-normalised pressure remain distinguishable.

Imported normal arrays participate in checkpoint identities, even when coordinates
and targets are unchanged. Earlier imported-normal checkpoints without that identity
remain viewable but may require restarting training; no cross-build checkpoint
compatibility is claimed beyond validated state/data contracts.
