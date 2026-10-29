# PDEs and boundary conditions

Keep three choices separate: where points are located, what PDE they must
satisfy, and what conditions apply on the boundary. A PDE preset does not choose
a reference solution or boundary conditions for you.

## Choose a preset or an expression

Use **Predefined PDE** for a tested family, or **PDE residual** for a restricted
mathematical expression. Both fill the same PDE input on the trainer: choose
one producer, not two residual blocks connected sequentially.

![The Helmholtz preset exposes squared wavenumber and its forcing expression](../../assets/screenshots/manual-helmholtz-pde.png){ .portrait }

*The Helmholtz preset exposes squared wavenumber and its forcing expression. [Open full size](../../assets/screenshots/manual-helmholtz-pde.png).*

The current families include Poisson, reaction–diffusion, 1D advection–diffusion,
Helmholtz and steady 2D incompressible flow. Coefficient meanings and signs differ;
read the equation displayed in Properties. The [example catalogue](../problem-examples.md)
shows complete configurations. Loading an example is the simplest way to start.

Custom expressions use an allowlisted mathematical language, not Python scripts.
For example, `laplace(u)+c` is a scalar Poisson residual. `D(u,x)` denotes a
derivative. See the [expression and component reference](../components.md) before
writing a different operator; unsupported syntax is rejected rather than executed.

## Assign each named boundary

Select **Boundary condition**, choose a side, and assign its supported type and
target. Dirichlet specifies a field value; Neumann specifies its outward normal
derivative. A custom operator can express supported Robin-like conditions.

![Boundary-condition settings assign physics to the independently sampled sides](../../assets/screenshots/manual-boundary-condition.png){ .portrait }

*Boundary-condition settings assign physics to the independently sampled sides. [Open full size](../../assets/screenshots/manual-boundary-condition.png).*

Normal orientation matters. The scalar Neumann value is an outward derivative,
not automatically a conductivity-scaled flux. Current scalar problems require a
Dirichlet anchor. Adjacent Dirichlet values must agree at shared rectangle corners.
The dedicated hard-Dirichlet method has stricter equal-constant box conditions.

For steady flow, boundary settings apply to `u`, `v` and `p`, and a pressure gauge
fixes the pressure reference. Start from **2D incompressible flow** for a complete
working graph; scalar and vector contracts are not interchangeable.

## Reference values are optional

A Reference block is needed only when you request a Comparison. Examples may
supply an explicit expression; your own problem may have no exact solution.
Do not add a reference merely to make the graph look complete.

**Next:** [Choose networks and PINN methods](networks.md).

For physical time dependence, start with [Burgers — 1D + time](../burgers.md).
It uses `(x,t)` coordinates, two spatial BC traces and a separate initial profile,
with no final-time condition. Its preset uses `u_t + u*u_x - nu*u_xx = 0`.


## Arrows choose where the PDE is enforced

The PDE block accepts **Collocation points** and **Additional boundary points
(optional)**. This applies to both PDE residual and Predefined PDE.

- An interior source connected to Collocation points enforces the PDE there.
- A boundary source connected to the additional input adds those boundary rows to
  the PDE loss. Without that arrow, boundary rows are not added.
- A boundary source can instead connect directly to Collocation points for a
  boundary-only residual; remove the unused interior sampler in that graph.
- The independent arrow from Boundary data/points to Boundary condition controls
  BC losses. Connecting a boundary source to the PDE does not remove its BC loss.

![Interior and boundary arrows feed the PDE independently](../../assets/screenshots/manual-collocation-both.png)

*Both arrows: residual evaluated on the combined point rows; boundary operators
are still evaluated separately through the Boundary condition block.*

![Without the extra boundary arrow the PDE uses only interior points](../../assets/screenshots/manual-collocation-interior.png)

*No boundary arrow to the PDE means no extra boundary PDE rows.*

New templates and File → Examples include the boundary-to-PDE arrow. Existing
saved models retain their original wiring and sampling behaviour. The low-level
`PoissonConfig` API keeps `pde_points='interior'` for compatibility; scripts can
explicitly select `'both'` or `'boundary'`. Example scripts select their intended
mode explicitly. Points are concatenated without deduplication: shared corners or
repeated input rows retain their sampling multiplicity. The PDE term is one mean
squared residual over the selected rows, not an additional BC term. Preset forcing
is evaluated at the added rows too. For transient problems, the connected boundary
source may also contain initial points. Periodic examples include paired faces;
pressure gauges and mean-velocity anchors are not geometric boundary point sets.

RAR retains the boundary PDE anchors while growing the interior set. Importance
sampling resamples interior rows with inverse-probability weights and retains
boundary anchors at unit weight. RAR's maximum is the total active PDE-row budget,
including connected boundary anchors. Fixed SA/RBA masks include all selected PDE
rows. Expert causal weighting bins added rows by their physical time, preserving
complete coordinate tuples, and Scale refreshes the selected point sets together.

**Integral loss is different:** its PDE term is a domain-volume quadrature.
Disconnect the extra boundary PDE arrow for that mode; otherwise validation explains
the incompatibility. Boundary operators keep their own boundary measure. Margam
never assigns volume quadrature weights to extra boundary points automatically.

Boundary collocation is an explicit modelling choice. Imported points are not
geometrically filtered, and users may omit nonsmooth corners or incompatible
initial traces from the PDE dataset. There is no universal paper convention: for
example, the [upstream gPINN Burgers residual-only setup](https://github.com/lu-group/gpinn/blob/main/src/burgers.py) explicitly uses zero
boundary/initial samples. The graph supports either choice.


The [CPU](../../assets/benchmarks/pde-collocation-cpu.json) and
[CUDA](../../assets/benchmarks/pde-collocation-cuda.json) checks verify selected
PDE row counts and exact same-device continuation for baseline, importance, gPINN,
RBA and Expert training. These are correctness checks, not accuracy benchmarks.

```sh
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m examples.pde_collocation --device cpu --output build/pde-collocation-cpu.json
.venv/bin/python -m examples.pde_collocation --device cuda --output build/pde-collocation-cuda.json
```
