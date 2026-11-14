# Fixed Burgers reference samples

Evaluation-only values for `u_t + u*u_x - (0.01/pi)*u_xx = 0`,
`x in [-1,1]`, `t in [0,1]`, initial `-sin(pi*x)`, zero spatial walls.
Changing a model does not regenerate these fixed values.

`reference.csv` has 2,827 rows: 257 x coordinates at each of 11 physical times.
Map coordinates `x,t`, ID `point_id`; use no value column for Evaluation points
and `u` for Reference data. Row order is time first, then increasing x.
These are not training observations or a copied author dataset.

Generated independently by `examples.burgers_reference` using
`margam.components.burgers_reference.burgers_reference`, NumPy float64 and
128-node Cole–Hopf Gauss–Hermite quadrature, checked against 256 nodes.
[source.json](source.json) records the CSV/source hashes and convergence result.
The helper also agrees with the existing inverse-example author observations;
see the [guide](../../../docs/burgers.md) for validation and limitations.

Sources: [original PINN problem](https://maziarraissi.github.io/PINNs/) and
[Burkardt's integral-solution reference](https://people.sc.fsu.edu/~jburkardt/py_src/burgers_exact/burgers_exact.html).
No upstream solver source or reference arrays were copied for this dataset.

Regenerate into a fresh directory from the root environment:

```bash
python -m examples.burgers_reference --output results/burgers-reference
```
