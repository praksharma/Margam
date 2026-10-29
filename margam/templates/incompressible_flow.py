"""Steady 2D incompressible flow with explicit velocity BCs and a pressure gauge.

Density-normalised pressure and kinematic viscosity; this is not a transient solver.
"""
import math
import json
from .poisson import PoissonConfig,FLOW_RESIDUALS,flow_boundary_defaults,sample_problem
from ..components.expressions import Expression
from ..components.prepared_data import read_points,normal_fingerprint


def validate_flow_config(config):
    if config.shape!='Rectangle' or config.method not in ('baseline','self_adaptive','lra','scale'):raise ValueError('2D flow requires a rectangle/bounding box and a baseline, self-adaptive, LRA or Scale method.')
    if not math.isfinite(config.viscosity) or config.viscosity<=0:raise ValueError('Kinematic viscosity must be positive and finite.')
    if len(config.flow_residuals)!=3:raise ValueError('Flow needs momentum-x, momentum-y and continuity residuals.')
    for text in config.flow_residuals:Expression(text,2,fields=('u','v','p'))
    if len(config.pressure_point)!=2 or not all(math.isfinite(v) for v in config.pressure_point) or not config.x_min<=config.pressure_point[0]<=config.x_max or not config.y_min<=config.pressure_point[1]<=config.y_max:raise ValueError('Pressure gauge point must lie inside the bounding box.')
    if not math.isfinite(config.pressure_value) or not math.isfinite(config.pressure_weight) or config.pressure_weight<=0:raise ValueError('Pressure gauge target/weight must be finite, with a positive weight.')
    for side in config.sides:
        for field,spec in config.flow_boundary[side].items():
            if field not in ('u','v','p') or spec['type'] not in ('None','Dirichlet','Neumann','Custom'):raise ValueError('Unsupported flow boundary field/type.')
            if spec['type']!='None':Expression(spec['target'],2,target=True,fields=('u','v','p'))
            if spec['type']=='Custom':Expression(spec['operator'],2,fields=('u','v','p'))
    for field in ('u','v'):
        if not any(config.flow_boundary[side].get(field,{}).get('type')=='Dirichlet' for side in config.sides):raise ValueError('Each velocity component currently needs a Dirichlet anchor.')
    if any(source['values'] for source in config.boundary_data.values()):raise ValueError('Flow point imports use coordinates/normals; set component targets explicitly in the BC editor.')
    normals={'left':(-1.,0.),'right':(1.,0.),'bottom':(0.,-1.),'top':(0.,1.)}
    # Reject conflicting prescribed component values at built-in shared corners.
    import torch
    for a,b in [('left','bottom'),('left','top'),('right','bottom'),('right','top')]:
        if config.boundary_data.get(a) or config.boundary_data.get(b):continue
        point=torch.tensor([[config.x_min if a=='left' else config.x_max,config.y_min if b=='bottom' else config.y_max]])
        for field in ('u','v','p'):
            specs=[config.flow_boundary[side].get(field) for side in (a,b)]
            if all(spec and spec['type']=='Dirichlet' for spec in specs):
                values=[float(Expression(spec['target'],2,target=True,fields=('u','v','p'))(point,source=config.source,viscosity=config.viscosity,normal=normals[side])) for side,spec in zip((a,b),specs)]
                if not all(math.isfinite(v) for v in values) or not math.isclose(*values,rel_tol=1e-6,abs_tol=1e-6):raise ValueError('Conflicting '+field+' targets at the '+a+'/'+b+' corner.')

    gauge=torch.tensor([config.pressure_point])
    for side in config.sides:
        spec=config.flow_boundary[side].get('p')
        axis=0 if side in ('left','right') else 1
        edge={'left':config.x_min,'right':config.x_max,'bottom':config.y_min,'top':config.y_max}[side]
        if spec and spec['type']=='Dirichlet' and math.isclose(config.pressure_point[axis],edge,abs_tol=1e-9):
            value=float(Expression(spec['target'],2,target=True,fields=('u','v','p'))(gauge,source=config.source,viscosity=config.viscosity,normal=normals[side]))
            if not math.isfinite(value) or not math.isclose(value,config.pressure_value,rel_tol=1e-6,abs_tol=1e-6):raise ValueError('Pressure gauge conflicts with the prescribed '+side+' pressure.')


def build_flow_pinn(config):
    import torch
    from ..components import PINNProblem,PointConstraint
    from ..methods import PINN
    from ..nn.factory import build_network
    config.validate()
    torch.manual_seed(config.seed if config.network_seed is None else config.network_seed)
    interior,boundaries=sample_problem(config)
    expressions=[Expression(text,2,fields=('u','v','p')) for text in config.flow_residuals]
    def residual(x,fields):return torch.cat([expr(x,fields,source=config.source,viscosity=config.viscosity) for expr in expressions],dim=1)
    normals={'left':(-1.,0.),'right':(1.,0.),'bottom':(0.,-1.),'top':(0.,1.)};constraints=[];normal_identities={}
    for side,rows in boundaries.items():
        normal=normals[side]
        if config.boundary_data.get(side):
            _,imported_normals,_=read_points(config.boundary_data[side],2)
            if imported_normals is not None:normal=imported_normals.tolist();normal_identities[side]=normal_fingerprint(imported_normals)
            elif any(spec['type']=='Neumann' or spec['type']=='Custom' and Expression(spec['operator'],2,fields=('u','v','p')).uses_normal or spec['type']!='None' and Expression(spec['target'],2,target=True,fields=('u','v','p')).uses_normal for spec in config.flow_boundary[side].values()):
                axis=0 if side in ('left','right') else 1
                edge={'left':config.x_min,'right':config.x_max,'bottom':config.y_min,'top':config.y_max}[side]
                if any(abs(row[axis]-edge)>1e-7 for row in rows):raise ValueError('Off-box flow boundaries require explicit unit outward normals.')
        for field,spec in config.flow_boundary[side].items():
            if spec['type']=='None':continue
            text=field if spec['type']=='Dirichlet' else 'dn('+field+')' if spec['type']=='Neumann' else spec['operator']
            operation=Expression(text,2,fields=('u','v','p'));target=Expression(spec['target'],2,target=True,fields=('u','v','p'))
            def operator(x,u,expr=operation,normal=normal):
                n=tuple(x.new_tensor(normal)[:,i:i+1] for i in range(2)) if isinstance(normal[0],list) else normal
                return expr(x,u,source=config.source,viscosity=config.viscosity,normal=n)
            def value(x,expr=target,normal=normal):
                n=tuple(x.new_tensor(normal)[:,i:i+1] for i in range(2)) if isinstance(normal[0],list) else normal
                return expr(x,source=config.source,viscosity=config.viscosity,normal=n)
            constraints.append(PointConstraint(side+':'+field,torch.tensor(rows),value,operator))
    constraints.append(PointConstraint('pressure_gauge',torch.tensor([config.pressure_point]),config.pressure_value,lambda x,u:u[:,2:3]))
    from ..components.collocation import collocation_rows
    problem=PINNProblem(residual,torch.tensor(collocation_rows(config,interior,boundaries)),constraints,data_identity=json.dumps(normal_identities,sort_keys=True) if normal_identities else '')
    from ..components.losses import apply_problem_measures
    apply_problem_measures(problem,config)
    network=build_network(config)
    weights=config.loss_weights
    if config.method=='self_adaptive':
        from ..methods.self_adaptive import SelfAdaptivePINN
        return SelfAdaptivePINN(problem,network,weights=weights,loss_formulation=config.loss_formulation).compile(lr=config.lr,device=config.device,adaptive_lr=config.sa_lr,initial_weight=config.sa_initial,max_weight=config.sa_max,adapt_boundary=config.sa_boundary)
    if config.method=='lra':
        from ..methods.learning_rate_annealing import LearningRateAnnealingPINN
        return LearningRateAnnealingPINN(problem,network,weights=weights,loss_formulation=config.loss_formulation).compile(lr=config.lr,device=config.device,alpha=config.lra_alpha,update_interval=config.lra_interval,epsilon=config.lra_epsilon,max_weight=config.lra_max)
    return PINN(problem,network,weights=weights,loss_formulation=config.loss_formulation).compile(lr=config.lr,device=config.device)


def flow_demo_config():
    return PoissonConfig(physics='incompressible_2d',source=0.,shape='Rectangle',points=144,boundary_points=24,layers=(32,32),iterations=1500,lr=.005,boundary_weight=10.)


def make_flow_reference(config):
    from ..components.outputs import ReferenceSource
    config.validate()
    if tuple(config.flow_residuals)!=FLOW_RESIDUALS or config.flow_boundary!=flow_boundary_defaults():raise ValueError('The flow preset reference supports only the explicit Couette demonstration. Use a compatible expression/data source for edited physics.')
    pressure=config.pressure_value
    return ReferenceSource('Analytical Couette flow (u=y, v=0, constant p)',lambda points:[[point[1],0.,pressure] for point in points])
