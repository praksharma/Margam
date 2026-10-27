# Problem examples and predefined equations

File → Examples is the catalogue of complete runnable desktop graphs. Existing
Poisson, heat and steady incompressible flow examples remain available. Four
additional manufactured cases exercise the new Predefined PDE block:

| Example | Equation | Reference on the unit domain |
| --- | --- | --- |
| Interval Poisson | uₓₓ + 2 = 0 | x(1−x) |
| Interval reaction–diffusion | −uₓₓ + u = (π²+1)sin(πx) | sin(πx) |
| Interval advection–diffusion | uₓ − 0.1uₓₓ = 1 | x |
| Rectangle sine Poisson | Δu + 2π²sin(πx)sin(πy) = 0 | sin(πx)sin(πy) |

Dirichlet endpoints/edges follow the reference. The advection case has unequal
endpoint values. Reaction–diffusion and sine Poisson use the existing hard
Dirichlet method; the others use the baseline method. Training Monitor, Solution
Plot, Comparison and CSV Export are wired explicitly; no view opens automatically.

Predefined PDE and PDE residual are alternatives, not sequential stages. The
former chooses a tested family (Poisson, reaction–diffusion, interval
advection–diffusion, or steady incompressible flow); the latter retains editable
restricted expressions. Both produce the same typed residual connection. The
supported graph has one PDE producer. A preset supplies neither BCs nor a
reference solution. Source/coefficient, coordinate forcing and viscosity settings
belong to the selected block. Forcing uses the existing restricted mathematics
parser; no Python scripts are executed.

Poisson uses Δu+c+f=0; reaction–diffusion uses −Δu+cu=f;
advection–diffusion uses cuₓ−νuₓₓ=f with positive ν and an interval domain.
Flow retains the existing steady u/v/p equations and rectangular domain contract.
Changing a family does not automatically replace BCs, outputs or reference
expressions: update these to describe the intended problem.

Model format 10 persists block settings and compact recipes and reads earlier formats 1–9.
The tests check manufactured residuals and train each new example for 700 CPU
updates with RMS prediction error below 0.05. This is a regression check for these
configured cases, not a guarantee for arbitrary coefficients or domains.

File → Examples also includes Self-adaptive PINN · interval Poisson, using its
separate trainer and the same explicit interval reference. See self-adaptive-pinn.md.
