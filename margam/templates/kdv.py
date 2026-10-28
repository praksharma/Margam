"""Periodic Korteweg–de Vries case from Scale-PINN, independently implemented."""
import math
from ..components.differential import derivative


def validate_kdv(c):
    if c.shape!='SpaceTime1D' or c.physics!='scalar' or c.method not in ('baseline','scale'):raise ValueError('KdV currently supports scalar 1D space–time with baseline or Scale training.')
    if c.sampling!='random' or c.interior_data or c.boundary_data or c.adaptive_sampling or c.loss_formulation!='mean':raise ValueError('KdV uses built-in random samples and mean MSE.')
    if not math.isfinite(c.viscosity) or c.viscosity<=0:raise ValueError('KdV dispersion coefficient must be positive.')


def residual(c,x,u):return derivative(u,x,1)+c.source*u*derivative(u,x,0)+c.viscosity*derivative(u,x,0,3)


def samples(c,phase):
    from ..components.sampling import sample_box
    seed=(c.seed+104729*phase)%2**31
    points=sample_box(*c.bounds,c.points,'random',seed)
    initial=[[row[0],c.y_min] for row in sample_box([c.x_min],[c.x_max],c.boundary_points,'random',(seed+1)%2**31)]
    periodic=[[c.x_min,row[0]] for row in sample_box([c.y_min],[c.y_max],c.boundary_points,'random',(seed+2)%2**31)]
    return points,dict(initial=initial,periodic_x=periodic)


def conditions(c,network,sets):
    import torch
    from ..components.problem import PointConstraint
    def periodic(x,u):
        other=x.detach().clone();other[:,0]=c.x_max;other.requires_grad_(True);paired=network(other)
        return torch.cat([u-paired]+[derivative(u,x,0,order)-derivative(paired,other,0,order) for order in (1,2)],1)
    return [PointConstraint('initial',torch.tensor(sets['initial']),lambda x:torch.cos(torch.pi*x[:,0:1])),PointConstraint('periodic_x',torch.tensor(sets['periodic_x']),0.,periodic)]


def build_kdv(c):
    import torch
    from ..components.problem import PINNProblem
    from ..nn.factory import build_network
    from ..methods.pinn import PINN
    validate_kdv(c);network=build_network(c);points,sets=samples(c,0)
    problem=PINNProblem(lambda x,u:residual(c,x,u),torch.tensor(points),conditions(c,network,sets))
    def refresh(phase):
        points,sets=samples(c,phase);return points,conditions(c,network,sets)
    problem.refresh_samples=refresh
    return PINN(problem,network,weights=c.loss_weights).compile(lr=c.lr,device=c.device)
