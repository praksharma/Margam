"""Seven concrete Expert's Guide examples, independently implemented from equations.

Torus Taylor–Green and ramped cylinder startup are labelled self-contained
variations. No JAX code/datasets or executable external data are used.
"""
import math
import random

CASES={
 'expert_allen_cahn':dict(label='Allen–Cahn · 1D + time',shape='SpaceTime1D',fields=('u',),equation='uₜ − 0.0001 uₓₓ + 5u³ − 5u = 0',names=('initial','periodic_x')),
 'expert_advection':dict(label='Advection · 1D + time',shape='SpaceTime1D',fields=('u',),equation='uₜ + c uₓ = 0',names=('initial','periodic_x')),
 'expert_stokes':dict(label='Stokes · 2D channel/cylinder',shape='Channel2D',fields=('u','v','p'),equation='−νΔ(u,v) + ∇p = 0; div(u,v)=0',names=('inlet','walls','cylinder','outlet_traction')),
 'expert_ks':dict(label='Kuramoto–Sivashinsky · 1D + time',shape='SpaceTime1D',fields=('u',),equation='uₜ + αu uₓ + βuₓₓ + γuₓₓₓₓ = 0',names=('initial','periodic_x')),
 'expert_cavity':dict(label='Lid-driven cavity · 2D · Re=3200',shape='Rectangle',fields=('u','v','p'),equation='(u,v)·∇(u,v) + ∇p − νΔ(u,v)=0; div=0',names=('lid','walls','pressure_gauge')),
 'expert_torus':dict(label='Navier–Stokes torus · 2D + time · Taylor–Green variation',shape='SpaceTime2D',fields=('u','v','w'),equation='ωₜ + uωₓ + vωᵧ − νΔω = 0; div=0; ω=vₓ−uᵧ',names=('initial','periodic_x','periodic_y','mean_velocity')),
 'expert_cylinder':dict(label='Navier–Stokes cylinder · 2D + time · startup variation',shape='ChannelTime2D',fields=('u','v','p'),equation='(u,v)ₜ +(u,v)·∇(u,v)+∇p−νΔ(u,v)=0; div=0',names=('initial','inlet','walls','cylinder','outlet_traction')),
}


def case_config(key):
    from .poisson import PoissonConfig
    from ..recipes.expert import DEFAULTS
    from copy import deepcopy
    if key not in CASES:raise ValueError('Unknown Expert example.')
    c=PoissonConfig()
    for name,value in DEFAULTS.items():setattr(c,name,tuple(value) if name=='layers' else deepcopy(value))
    c.pde_preset=key;c.physics='paper_case';c.shape=CASES[key]['shape'];c.sampling='random';c.boundary_sampling={}
    c.points=256;c.boundary_points=32;c.iterations=2000;c.source=0.;c.expert_settings['balance_every']=100
    if key in ('expert_allen_cahn','expert_advection','expert_ks'):
        c.x_min=-1. if key=='expert_allen_cahn' else 0.;c.x_max=1. if key=='expert_allen_cahn' else 2*math.pi
        c.y_min=0.;c.y_max=1.
        if key=='expert_allen_cahn':c.viscosity=.0001;c.source=5.
        elif key=='expert_advection':c.source=50.
        else:c.source=100/16;c.viscosity=100/16**2;c.fourth_order=100/16**4
    elif key=='expert_cavity':c.viscosity=1/3200;c.bc_values['top']=1.
    elif key=='expert_torus':
        c.x_min=c.y_min=0.;c.x_max=c.y_max=2*math.pi;c.t_min=0.;c.t_max=10.;c.viscosity=.01
    else:
        c.x_min=c.y_min=0.;c.x_max=2.2;c.y_max=.41;c.viscosity=.001;c.bc_values['left']=.3 if key=='expert_stokes' else 1.5
        c.t_min=0.;c.t_max=1.;c.startup_time=.1
    return c


def validate_case(c):
    if c.shape!=CASES[c.pde_preset]['shape']:raise ValueError('This Expert example needs '+CASES[c.pde_preset]['shape']+'.')
    if c.physics!='paper_case':raise ValueError('Invalid Expert example physics contract.')
    if c.method not in ('expert','baseline'):raise ValueError('These Expert examples support Expert or baseline training.')
    if c.interior_data or c.boundary_data or c.adaptive_sampling or c.loss_formulation!='mean':raise ValueError('Expert suite currently uses built-in random samples and mean loss; imported/adaptive quadrature is not supported.')
    if c.sampling!='random':raise ValueError('Expert suite geometry sampling requires random points.')
    if not math.isfinite(c.viscosity) or c.viscosity<=0:raise ValueError('Viscosity/diffusion coefficient must be positive.')
    if not math.isfinite(c.fourth_order) or c.fourth_order<=0 or not math.isfinite(c.startup_time) or c.startup_time<=0:raise ValueError('Fourth-order coefficient and startup time must be positive.')
    if not all(math.isfinite(c.bc_values.get(side,0.)) for side in ('left','top')):raise ValueError('Inlet and lid velocities must be finite.')
    if not math.isfinite(c.pressure_weight) or c.pressure_weight<=0:raise ValueError('Pressure gauge weight must be positive.')
    if c.has_obstacle:
        if not math.isfinite(c.cylinder_radius) or c.cylinder_radius<=0 or not all(math.isfinite(v) for v in (c.cylinder_x,c.cylinder_y)) or not(c.x_min+c.cylinder_radius<c.cylinder_x<c.x_max-c.cylinder_radius and c.y_min+c.cylinder_radius<c.cylinder_y<c.y_max-c.cylinder_radius):raise ValueError('The circle must lie strictly inside the channel.')


def inside_fluid(c,row):return not c.has_obstacle or (row[0]-c.cylinder_x)**2+(row[1]-c.cylinder_y)**2>c.cylinder_radius**2


def samples(c,phase=0):
    """Uniform spatial draws with rejection outside the disk and equal time strata."""
    lower,upper=c.bounds;rng=random.Random((c.seed+104729*phase)%2**31)
    chunks=c.expert_settings['chunks'] if c.method=='expert' and c.is_time_dependent else 1
    def draw(count,fixed=None,initial=False,chunked=False):
        rows=[];fixed=fixed or {};attempts=0
        while len(rows)<count:
            attempts+=1
            if attempts>count*1000:raise ValueError('Unable to sample the fluid region.')
            row=[a+(1e-7+(1-2e-7)*rng.random())*(b-a) for a,b in zip(lower,upper)]
            if c.is_time_dependent and chunked:
                group=min(chunks-1,len(rows)//(count//chunks));row[-1]=lower[-1]+(group+rng.random())*(upper[-1]-lower[-1])/chunks
            for index,value in fixed.items():row[index]=value
            if initial:row[-1]=lower[-1]
            if inside_fluid(c,row):rows.append(row)
        return rows
    n=c.boundary_points;key=c.pde_preset
    interior=draw(c.points,chunked=True);sets={}
    if c.is_time_dependent:sets['initial']=draw(n,initial=True)
    if key in ('expert_allen_cahn','expert_advection','expert_ks','expert_torus'):
        sets['periodic_x']=draw(n,{0:c.x_min})
        if key=='expert_torus':
            sets['periodic_y']=draw(n,{1:c.y_min})
            sets['mean_velocity']=draw(min(n,8),{0:c.x_min,1:c.y_min})
    elif key=='expert_cavity':
        sets['lid']=draw(n,{1:c.y_max})
        sets['walls']=draw(n,{0:c.x_min})+draw(n,{0:c.x_max})+draw(n,{1:c.y_min})
        sets['pressure_gauge']=[[c.x_min,c.y_min]]
    else:
        sets['inlet']=draw(n,{0:c.x_min});sets['outlet_traction']=draw(n,{0:c.x_max})
        sets['walls']=draw(n,{1:c.y_min})+draw(n,{1:c.y_max})
        rows=[]
        for _ in range(n):
            angle=2*math.pi*rng.random();row=[c.cylinder_x+c.cylinder_radius*math.cos(angle),c.cylinder_y+c.cylinder_radius*math.sin(angle)]
            if c.is_time_dependent:row.append(c.t_min+rng.random()*(c.t_max-c.t_min))
            rows.append(row)
        sets['cylinder']=rows
    return interior,sets


def derivative(value,x,axis,order=1):
    import torch
    for _ in range(order):
        if not value.requires_grad:return x.new_zeros((len(x),1))
        gradient=torch.autograd.grad(value.sum(),x,create_graph=True,retain_graph=True,allow_unused=True)[0]
        value=x.new_zeros((len(x),1)) if gradient is None else gradient[:,axis:axis+1]
    return value


def vorticity(x,u):return derivative(u[:,1:2],x,0)-derivative(u[:,0:1],x,1)


def residual(c,x,values):
    import torch
    d=lambda value,axis,order=1:derivative(value,x,axis,order)
    u=values[:,0:1];key=c.pde_preset
    if key=='expert_allen_cahn':return d(u,1)-c.viscosity*d(u,0,2)+c.source*(u**3-u)
    if key=='expert_advection':return d(u,1)+c.source*d(u,0)
    if key=='expert_ks':return d(u,1)+c.source*u*d(u,0)+c.viscosity*d(u,0,2)+c.fourth_order*d(u,0,4)
    v=values[:,1:2];div=d(u,0)+d(v,1)
    if key=='expert_torus':
        w=values[:,2:3]
        return torch.cat((d(w,2)+u*d(w,0)+v*d(w,1)-c.viscosity*(d(w,0,2)+d(w,1,2)),div,w-vorticity(x,values)),1)
    p=values[:,2:3];mx=d(p,0)-c.viscosity*(d(u,0,2)+d(u,1,2));my=d(p,1)-c.viscosity*(d(v,0,2)+d(v,1,2))
    if key!='expert_stokes':mx=mx+u*d(u,0)+v*d(u,1);my=my+u*d(v,0)+v*d(v,1)
    if key=='expert_cylinder':mx=mx+d(u,2);my=my+d(v,2)
    return torch.cat((mx,my,div),1)


def make_constraints(c,network,sets,*,time_origin=None):
    import torch
    from ..components.problem import PointConstraint
    result=[];key=c.pde_preset
    for name,rows in sets.items():
        target=0.;operator=None
        if name.startswith('periodic'):
            axis=0 if name=='periodic_x' else 1
            order=3 if key=='expert_ks' else 2 if key=='expert_torus' else 1 if key=='expert_allen_cahn' else 0
            def operator(x,u,axis=axis,order=order):
                other=x.detach().clone();other[:,axis]=c.bounds[1][axis];other.requires_grad_(True);paired=network(other)
                terms=[u-paired]
                for degree in range(1,order+1):
                    for field in range(u.shape[1]):terms.append(derivative(u[:,field:field+1],x,axis,degree)-derivative(paired[:,field:field+1],other,axis,degree))
                return torch.cat(terms,1)
        elif name=='initial':
            if key=='expert_allen_cahn':target=lambda x:x[:,0:1]**2*torch.cos(torch.pi*x[:,0:1])
            elif key=='expert_advection':target=lambda x:torch.sin(x[:,0:1])
            elif key=='expert_ks':target=lambda x:torch.cos(x[:,0:1])*(1+torch.sin(x[:,0:1]))
            elif key=='expert_torus':
                def target(x):
                    u=torch.sin(x[:,0:1])*torch.cos(x[:,1:2]);v=-torch.cos(x[:,0:1])*torch.sin(x[:,1:2]);w=2*torch.sin(x[:,0:1])*torch.sin(x[:,1:2])
                    return torch.cat((u,v,w),1)
                operator=lambda x,u:u
            else:operator=lambda x,u:u[:,:2] # Quiescent fluid startup; pressure is not an initial datum.
        elif name=='mean_velocity':
            def operator(x,u):
                # Periodic curl/div determine velocity only up to a time-varying
                # spatial constant. Preserve the variation's zero mean velocity.
                fractions=torch.arange(16,dtype=x.dtype,device=x.device)/16
                a=c.x_min+(c.x_max-c.x_min)*fractions;b=c.y_min+(c.y_max-c.y_min)*fractions
                yy,xx=torch.meshgrid(b,a,indexing='ij')
                xy=torch.stack((xx.flatten(),yy.flatten()),1).repeat(len(x),1)
                time=x[:,2:3].repeat_interleave(256,dim=0)
                values=network(torch.cat((xy,time),1))[:,:2]
                return values.reshape(len(x),256,2).mean(1)
        elif name=='pressure_gauge':operator=lambda x,u:u[:,2:3]
        elif name=='outlet_traction':operator=lambda x,u:torch.cat((c.viscosity*derivative(u[:,0:1],x,0)-u[:,2:3],c.viscosity*derivative(u[:,1:2],x,0)),1)
        else:
            operator=lambda x,u:u[:,:2]
            if name=='lid':target=lambda x:torch.cat((torch.full_like(x[:,0:1],c.bc_values['top']),torch.zeros_like(x[:,0:1])),1)
            if name=='inlet':
                def target(x):
                    velocity=4*c.bc_values['left']*(x[:,1:2]-c.y_min)*(c.y_max-x[:,1:2])/(c.y_max-c.y_min)**2
                    if key=='expert_cylinder':velocity=velocity*(1-torch.exp(-(x[:,-1:]-(c.t_min if time_origin is None else time_origin))/c.startup_time))
                    return torch.cat((velocity,torch.zeros_like(velocity)),1)
        result.append(PointConstraint(name,torch.tensor(rows),target,operator))
    return result


def build_case(c,*,initial_teacher=None,time_origin=None):
    import torch
    from ..components.problem import PINNProblem
    from ..nn.factory import build_network
    from ..methods.pinn import PINN
    network=build_network(c);points,sets=samples(c)
    problem=PINNProblem(lambda x,u:residual(c,x,u),torch.tensor(points),make_constraints(c,network,sets,time_origin=time_origin))
    def refresh(phase):
        points,sets=samples(c,phase)
        return points,make_constraints(c,network,sets,time_origin=time_origin)
    problem.refresh_samples=refresh
    if c.method=='expert':
        from ..methods.expert import ExpertPINN
        model=ExpertPINN(problem,network,config=c,weights=c.loss_weights,initial_teacher=initial_teacher)
    else:model=PINN(problem,network,weights=c.loss_weights)
    return model.compile(lr=c.lr,device=c.device)


def condition_description(c):
    return {
      'expert_allen_cahn':'Initial: x² cos(πx). Periodic solution and first spatial derivative. No final-time condition.',
      'expert_advection':'Initial: sin(x). Periodic solution. No final-time condition.',
      'expert_ks':'Initial: cos(x)(1+sin(x)). Periodic solution and spatial derivatives through order three. No final-time condition.',
      'expert_stokes':'Parabolic inlet velocity; no slip on channel walls/cylinder; zero outlet traction ν∂x(u,v)−(p,0). Pressure is fixed by outlet traction.',
      'expert_cavity':'Uniform moving top lid; other walls stationary. Corner points are excluded because the lid/wall velocities are discontinuous there. One pressure gauge.',
      'expert_torus':'Self-contained Taylor–Green variation: u=sin(x)cos(y), v=−cos(x)sin(y), ω=2sin(x)sin(y) initially. Periodic spatial fields/derivatives; zero mean velocity (16×16 quadrature). Not the paper dataset.',
      'expert_cylinder':'Self-contained quiescent-fluid startup variation: initial u=v=0, inlet ramp 1−exp(−(t−t0)/startup_time). No slip on walls/cylinder; zero outlet traction. No initial pressure constraint. Not the paper pre-simulated initial field.'
    }[c.pde_preset]
