# Tutorials and code snippets

Whether you prefer building models visually on the desktop canvas or running
experiments from Python scripts, this cookbook collects end-to-end tutorials and
copy-pasteable snippets across Margam.

## Tutorial map by topic

| Topic | Tutorial & snippets |
| --- | --- |
| **1. First 1D Poisson PINN (GUI + Python)** | [Your first PINN — desktop walkthrough & Python script](first-pinn.md#7-run-the-same-1d-poisson-problem-from-python) |
| **2. PDE & boundary expression syntax** | [Expression cheat sheet, 2D mixed Dirichlet/Neumann/Robin, and 1D transient heat](../modeling/physics.md#expression-syntax-cheat-sheet-and-snippets) |
| **3. All 10 neural network architectures** | [FCNN, Fourier, Affine Fourier, Modified Fourier, Highway, Multiscale, SIREN, DGM, MFN](../modeling/networks.md#quick-snippets-configuring-all-10-pointwise-architectures) |
| **4. Switching PINN trainers** | [Hard Dirichlet, SA-PINN, LRA, gPINN, RBA, Scale-PINN, Expert PINN, DSGNAR](../modeling/networks.md#quick-snippets-switching-pinn-trainers-on-the-same-problem) |
| **5. Sampling, RAR & importance sampling** | [Box samplers, per-side boundary recipes, RAR, and importance sampling](../modeling/sampling.md#sampling-and-prepared-data-python-snippets) |
| **6. Non-rectangular `Prepared2D` domains** | [Generate and train a custom triangle domain with outward unit normals](../training-data.md#python-tutorial-generate-and-train-a-custom-prepared2d-domain-from-scratch) |
| **7. Supplied evaluation points & IDs** | [Evaluate at unsorted coordinates and align shuffled reference CSVs](../evaluation-points.md#python-tutorial-evaluate-at-unsorted-supplied-points-with-shuffled-reference-ids) |
| **8. Burgers equation (1D + time)** | [Train space–time Burgers, compare against Cole–Hopf reference, export PVD](../burgers.md#python-tutorial-train-burgers-evaluate-colehopf-reference-and-export-pvd) |
| **9. Inverse Burgers identification** | [Jointly learn unknown PDE coefficients $\lambda_1, \lambda_2$ from observations](../inverse-burgers.md#python-tutorial-run-inverse-burgers-inspect-coefficient-history-and-resume) |
| **10. 2D Helmholtz with Fourier LRA-PINN** | [Train oscillatory 2D Helmholtz and inspect LRA adaptive weights](../helmholtz.md#python-tutorial-train-2d-helmholtz-with-fourier-features-and-lra-balancing) |
| **11. Expert PINN suite (7 problems)** | [Allen–Cahn with NTK & time windows, and 2D Stokes flow past a cylinder](../expert-examples.md#python-tutorial-run-any-expert-pinn-suite-case-with-ntk-or-time-marching) |
| **12. Scale-PINN sequential correction** | [KdV and high-Re lid-driven cavity with affine Fourier features](../scale-pinn.md#python-tutorial-run-scale-pinn-presets-and-inspect-sequential-correction-diagnostics) |
| **13. Gradient-enhanced gPINN** | [Scalar Burgers + RAR and coupled steady 2D Couette flow](../gpinn.md#python-tutorial-scalar-and-coupled-steady-flow-gpinn-with-rar-refinement) |
| **14. Residual-based attention (RBA)** | [Zero-start vs normalized-first-residual attention masks](../rba-pinn.md#python-tutorial-train-rba-pinn-in-both-initialization-modes-and-inspect-masks) |
| **15. DSGNAR (`float64` Gauss–Newton)** | [Doubly-sketched Gauss–Newton trust-region solver in double precision](../dsgnar.md#python-tutorial-run-dsgnar-presets-and-inspect-trust-region-step-diagnostics) |
| **16. Autograd operators & benchmarking** | [`margam.utils` differential operators, checkpointing, and `run_benchmark`](../engine.md#differential-operators-tutorial-margamutils) |
| **17. Custom PyTorch research (`Trainer`)** | [Custom modules, `CosineDecay` continuation, and inverse parameter learning](../research.md#end-to-end-research-tutorials) |
| **18. CSV, ParaView VTI/VTU/PVD & run diff** | [Programmatic VTI/VTU/PVD export and `.margam-run.pt` prediction comparison](../use/export.md#programmatic-export-and-run-comparison-snippets) |
| **19. Headless studies & Slurm bundles** | [Inspect, run, resume, collect `.margam-study` grids and export Slurm jobs](../use/study-cli.md#programmatic-study-inspection-execution-collection-and-slurm-export) |
| **20. Scalar & vector error norms** | [Sweep all six discrete error norms and zero-reference policies](../use/error-norms.md#snippet-sweep-all-six-discrete-norms-and-both-zero-reference-policies) |
| **21. Diff `.margam.json` model files** | [Compare two saved model definitions by stable block ID](../use/compare-models.md#python-tutorial-diff-two-margamjson-model-definitions-in-a-script) |
| **22. Automate the GUI via MCP** | [Drive an open Margam window with the official Python MCP client](../mcp.md#python-tutorial-automate-a-live-margam-window-with-the-official-mcp-client) |
| **23. 3D sphere flow & JAXPI2 (`PirateNet` + `SOAP`)** | [`PirateNet`, `SOAP` + `Schedule-Free`, 3D Navier–Stokes, and MI300X sphere benchmarks](../sphere-turbulence.md#python-tutorials) |

---

## Quick-start snippets

### 1. Train and evaluate a 2D manufactured Poisson problem

Solve $-\Delta u = 2\pi^2 \sin(\pi x)\sin(\pi y)$ on $[0, 1]^2$ with homogeneous
Dirichlet boundaries and an exact hard-Dirichlet output transform:

```python
import torch
from margam.templates.examples import example_config
from margam.templates.poisson import build_pinn

cfg = example_config("poisson_sine_2d")
cfg.iterations = 500
cfg.device = "cpu"

model = build_pinn(cfg)
model.train(cfg.iterations, display_every=100)

# Evaluate against exact u(x, y) = sin(pi x) sin(pi y) on a 51x51 grid
axis = torch.linspace(0.0, 1.0, 51)
xy = torch.cartesian_prod(axis, axis)
exact = torch.sin(torch.pi * xy[:, 0:1]) * torch.sin(torch.pi * xy[:, 1:2])
pred = model.predict(xy)

rel_l2 = ((pred - exact).norm() / exact.norm()).item()
print(f"2D Poisson relative L2 error: {rel_l2:.4e}")
```

### 2. Custom PDE and Robin/Neumann boundary conditions with allowlisted expressions

Use restricted mathematical expressions (`D(u, x)`, `laplace(u)`, `dn(u)`, `nx`, `ny`)
without writing custom autograd code:

```python
from margam.templates.poisson import PoissonConfig, build_pinn

cfg = PoissonConfig(
    shape="Rectangle",
    x_min=0.0,
    x_max=1.0,
    y_min=0.0,
    y_max=1.0,
    points=144,
    boundary_points=24,
    pde_points="both",
    source=1.0,
    pde_expression="-laplace(u) + c*u - ((pi*pi + 1)*sin(pi*x) - 2 + c*y*y)",
    bc_types={"left": "Dirichlet", "bottom": "Dirichlet", "right": "Neumann", "top": "Custom"},
    bc_targets={"left": "y*y", "bottom": "sin(pi*x)", "right": "-pi", "top": "sin(pi*x) + 2.0"},
    bc_operators={"top": "u + 0.5*dn(u)"},
    bc_weights={"left": 2.0, "bottom": 2.0, "right": 1.0, "top": 1.0},
    layers=(32, 32),
    lr=5e-3,
    iterations=300,
    device="cpu",
)
model = build_pinn(cfg)
model.train(cfg.iterations, display_every=100)
```

### 3. Custom PyTorch network and physics with `margam.research.Trainer`

When you want full control over your `torch.nn.Module`, sampling, residual
callable, learning-rate schedule, or total-flux `IntegralConstraint`:

```python
import torch
from margam.research import CosineDecay, IntegralConstraint, PINNProblem, PointConstraint, Trainer, derivative

torch.manual_seed(7)
network = torch.nn.Sequential(
    torch.nn.Linear(1, 32),
    torch.nn.Tanh(),
    torch.nn.Linear(32, 32),
    torch.nn.Tanh(),
    torch.nn.Linear(32, 1),
)

# Solve u_xx + 2 = 0 on [0, 1] with u(0) = u(1) = 0 and integral_0^1 u(x) dx = 1/6
x_int = torch.linspace(0.0, 1.0, 66)[1:-1, None]
x_quad = torch.tensor([[0.0], [0.5], [1.0]])

problem = PINNProblem(
    residual=lambda x, u: derivative(u, x, axis=0, order=2) + 2.0,
    collocation_points=x_int,
    constraints=[
        PointConstraint("endpoints", torch.tensor([[0.0], [1.0]]), target=0.0),
        IntegralConstraint("total_flux", x_quad, target=1.0 / 6.0, quadrature_weights=[1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0]),
    ],
)

trainer = Trainer(problem, network, weights={"endpoints": 10.0, "total_flux": 5.0}).compile(
    lr=3e-3,
    device="cpu",
    schedule=CosineDecay(steps=500, minimum_ratio=0.05),
)
trainer.train(500, display_every=100)
```

### 4. Steady 2D incompressible Navier–Stokes flow (`u, v, p`)

Train a coupled 2D velocity–pressure flow problem on a rectangle with velocity
boundary conditions and a pressure anchor point:

```python
import torch
from margam.templates.incompressible_flow import flow_demo_config
from margam.templates.poisson import build_pinn

flow_cfg = flow_demo_config()
flow_cfg.points = 100
flow_cfg.boundary_points = 20
flow_cfg.iterations = 200
flow_cfg.device = "cpu"

flow_model = build_pinn(flow_cfg)
flow_model.train(flow_cfg.iterations, display_every=50)

# Predict (u, v, p) at interior coordinates
uvp = flow_model.predict(torch.tensor([[0.5, 0.25], [0.5, 0.75]]))
print("Predicted [u, v, p] rows:\n", uvp)
```
