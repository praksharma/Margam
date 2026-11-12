"""Concrete standing-wave problem (DSGNAR paper Appendix C.3)."""
import math
INITIAL='sin(pi*x)+0.5*sin(4*pi*x)'
REFERENCE='sin(pi*x)*cos(c*pi*t)+0.5*sin(4*pi*x)*cos(4*c*pi*t)'
DESCRIPTION='Initial displacement: sin(πx)+0.5 sin(4πx). Initial velocity: ∂u/∂t=0 on the same initial points. Zero endpoint displacement; no final-time condition.'


def validate_wave(c):
    if c.shape!='SpaceTime1D' or c.physics!='scalar' or c.method not in ('baseline','dsgnar'):raise ValueError('Standing wave supports baseline or DSGNAR on 1D space–time.')
    if c.interior_data or c.boundary_data or c.adaptive_sampling or c.loss_formulation!='mean':raise ValueError('Standing wave uses fixed built-in samples and mean loss.')
    if (c.x_min,c.x_max,c.y_min)!=(0.,1.,0.) or not math.isfinite(c.source) or c.source<=0:raise ValueError('Standing wave needs x in [0,1], initial t=0 and positive wave speed c.')
    if any(c.bc_types[s]!='Dirichlet' or c.bc_values[s]!=0 or c.bc_operators.get(s) for s in c.sides) or any(c.bc_targets.get(s,'')!=(INITIAL if s=='bottom' else '') for s in c.sides):raise ValueError('Standing-wave preset owns its initial displacement/velocity and zero endpoint conditions.')


def residual(c,x,u):
    from ..components.differential import derivative
    return derivative(u,x,1,2)-c.source**2*derivative(u,x,0,2)


def build_wave(c):
    import torch
    from ..components.differential import derivative
    from .poisson import sample_problem
    from ..components.problem import PINNProblem,PointConstraint
    from ..components.collocation import collocation_rows
    from ..nn.factory import build_network
    from ..methods.pinn import PINN
    validate_wave(c);points,sets=sample_problem(c)
    initial=torch.tensor(sets['bottom']);boundary=torch.tensor(sets['left']+sets['right'])
    problem=PINNProblem(lambda x,u:residual(c,x,u),torch.tensor(collocation_rows(c,points,sets)),[
        PointConstraint('initial',initial,lambda x:torch.sin(math.pi*x[:,0:1])+.5*torch.sin(4*math.pi*x[:,0:1])),
        PointConstraint('initial_velocity',initial,0.,lambda x,u:derivative(u,x,1)),
        PointConstraint('boundary',boundary)],data_identity='standing-wave-two-mode-v1')
    from ..components.losses import apply_boundary_identity
    apply_boundary_identity(problem,c)
    return PINN(problem,build_network(c),weights=c.loss_weights).compile(lr=c.lr,device=c.device)
