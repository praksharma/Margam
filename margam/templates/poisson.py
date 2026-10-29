"""Editable Poisson template; method-specific training remains in methods/pinn.py."""
from dataclasses import dataclass, field
import math
import json
from ..components.sampling import sample_box, SAMPLERS
from ..components.scale_config import scale_defaults,validate_scale_settings,validate_scale_problem
from ..components.expert_config import expert_defaults,validate_expert_settings,validate_expert_problem
from ..components.expressions import Expression
from ..components.prepared_data import read_points, validate_mapping, normal_fingerprint


FLOW_RESIDUALS=('u*D(u,x)+v*D(u,y)+D(p,x)-nu*laplace(u)',
                'u*D(v,x)+v*D(v,y)+D(p,y)-nu*laplace(v)',
                'D(u,x)+D(v,y)')


def flow_boundary_defaults():
    return {side:{field:{'type':'Dirichlet','target':'y' if field=='u' else '0','operator':''}
                  for field in ('u','v')} for side in ('left','right','bottom','top')}


@dataclass(frozen=True)
class BoundarySampling:
    """One rectangle side's fixed sampling recipe, independent of interior points."""
    points: int = 16
    method: str = 'grid'
    seed: int = 8


@dataclass
class PoissonConfig:
    shape: str = 'Rectangle'
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    t_min: float = 0.
    t_max: float = 1.
    cylinder_x: float = .2
    cylinder_y: float = .2
    cylinder_radius: float = .05
    fourth_order: float = 100/16**4
    startup_time: float = .1
    points: int = 64
    boundary_points: int = 16  # Default count for sides without an explicit recipe.
    adaptive_sampling: dict = field(default_factory=dict)
    sampling: str = 'grid'
    seed: int = 7
    source: float = 2.0
    layers: tuple = (16, 16)
    iterations: int = 1000
    lr: float = .01
    device: str = 'cpu'
    loss_formulation: str = 'mean'
    pde_weight: float = 1.0
    boundary_weight: float = 1.0
    bc_types: dict = field(default_factory=lambda: dict.fromkeys(('left', 'right', 'bottom', 'top'), 'Dirichlet'))
    bc_values: dict = field(default_factory=lambda: dict.fromkeys(('left', 'right', 'bottom', 'top'), 0.0))
    boundary_sampling: dict = field(default_factory=dict)
    pde_expression: str = ''
    bc_targets: dict = field(default_factory=dict)
    bc_operators: dict = field(default_factory=dict)
    interior_data: dict = field(default_factory=dict)
    boundary_data: dict = field(default_factory=dict)
    network_seed: int | None = None
    method: str = 'baseline'
    expert_settings: dict = field(default_factory=expert_defaults)
    physics: str = 'scalar'
    viscosity: float = .01
    flow_residuals: tuple = FLOW_RESIDUALS
    flow_boundary: dict = field(default_factory=flow_boundary_defaults)
    pressure_point: tuple = (0.,0.)
    pressure_value: float = 0.
    pressure_weight: float = 1.
    pde_preset: str = ''
    pde_forcing: str = '0'
    sa_lr: float = .005
    sa_initial: float = 1.
    sa_max: float = 1000.
    sa_boundary: bool = False
    lra_alpha: float = .9
    lra_interval: int = 10
    lra_epsilon: float = 1e-12
    lra_max: float = 1e6
    network_type: str = 'fcnn'
    network_options: dict = field(default_factory=dict)
    gpinn_weights: dict = field(default_factory=lambda: dict(x=.01,y=.01,t=.01))
    scale_settings: dict = field(default_factory=scale_defaults)

    @property
    def is_time_dependent(self):return self.shape in ('SpaceTime1D','SpaceTime2D','ChannelTime2D')

    @property
    def is_paper_case(self):return self.pde_preset.startswith('expert_')

    @property
    def has_obstacle(self):return self.shape in ('Channel2D','ChannelTime2D')

    @property
    def coordinate_names(self):return ('x','y','t') if self.shape in ('SpaceTime2D','ChannelTime2D') else ('x','t') if self.is_time_dependent else ('x','y')[:len(self.bounds[0])]

    def constraint_name(self,side):return 'initial' if self.is_time_dependent and side=='bottom' else side

    def side_label(self,side):return 'initial (t start)' if self.is_time_dependent and side=='bottom' else side

    @property
    def field_names(self):
        if self.is_paper_case:
            from .expert_examples import CASES
            return CASES[self.pde_preset]['fields']
        return ('u','v','p') if self.physics=='incompressible_2d' else ('u',)

    @property
    def constraint_names(self):
        if self.pde_preset=='kdv_1d':return ('initial','periodic_x')
        if self.is_paper_case:
            from .expert_examples import CASES
            return CASES[self.pde_preset]['names']
        if self.physics=='scalar':return tuple(self.constraint_name(s) for s in self.sides)
        return tuple(side+':'+field for side in self.sides for field,spec in self.flow_boundary[side].items() if spec['type']!='None')+('pressure_gauge',)

    @property
    def loss_weights(self):
        if self.pde_preset=='kdv_1d':return {'pde':self.pde_weight,'initial':self.boundary_weight/2,'periodic_x':self.boundary_weight/2}
        if self.is_paper_case:return {'pde':self.pde_weight,**{name:self.pressure_weight if name=='pressure_gauge' else self.boundary_weight/len(self.constraint_names) for name in self.constraint_names}}
        if self.physics=='scalar':return {'pde':self.pde_weight,**{self.constraint_name(s):self.boundary_weight/len(self.sides) for s in self.sides}}
        names=self.constraint_names
        return {'pde':self.pde_weight,**{name:self.boundary_weight/(len(names)-1) for name in names if name!='pressure_gauge'},'pressure_gauge':self.pressure_weight}

    @property
    def sides(self):
        if self.shape=='Prepared2D':return tuple(self.boundary_data)
        return ('left','right','bottom') if self.is_time_dependent else ('left', 'right') if self.shape == 'Interval' else ('left', 'right', 'bottom', 'top')

    @property
    def bounds(self):
        if self.shape in ('SpaceTime2D','ChannelTime2D'):return [self.x_min,self.y_min,self.t_min],[self.x_max,self.y_max,self.t_max]
        if self.shape == 'Interval':
            return [self.x_min], [self.x_max]
        return [self.x_min, self.y_min], [self.x_max, self.y_max]

    def boundary_recipe(self, side):
        """Unconfigured sides use grid, the template count, and stable side seeds."""
        sides = ('left', 'right', 'bottom', 'top')
        if side not in sides:
            raise ValueError('Unknown boundary side.')
        return self.boundary_sampling.get(
            side, BoundarySampling(self.boundary_points, 'grid', 8 + sides.index(side)))

    def validate_sampling(self):
        if self.interior_data:
            validate_mapping(self.interior_data)
            if self.interior_data['normals'] or self.interior_data['values']:raise ValueError('PDE point imports accept coordinates only; additional pointwise inputs remain deferred.')
        from ..components.training_data import valid_boundary_name
        if not isinstance(self.boundary_data,dict) or len(self.boundary_data)>64 or any(not valid_boundary_name(name) for name in self.boundary_data) or self.shape!='Prepared2D' and set(self.boundary_data)-{'left','right','bottom','top'}:raise ValueError('Unknown or invalid imported boundary set.')
        if self.shape=='Prepared2D' and (not self.interior_data or not self.boundary_data):raise ValueError('Prepared 2D requires interior data and at least one named boundary set.')
        for source in self.boundary_data.values():validate_mapping(source)
        if self.shape not in ('Interval', 'Rectangle','Prepared2D','SpaceTime1D','SpaceTime2D','Channel2D','ChannelTime2D'):
            raise ValueError('Choose Interval, Rectangle or 1D space–time.')
        lower, upper = self.bounds
        if not all(math.isfinite(v) for v in (*lower, *upper)) or any(b <= a for a,b in zip(lower,upper)):
            raise ValueError('Domain maximums must be greater than minimums.')
        if self.sampling not in SAMPLERS:
            raise ValueError('Unknown sampler.')
        for value, limit, label in [(self.points,100000,'Interior points'),(self.boundary_points,10000,'Boundary points')]:
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError(f'{label} must be between 1 and {limit}.')
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError('Seed must be a non-negative 32-bit integer.')
        if not isinstance(self.boundary_sampling, dict) or set(self.boundary_sampling) - {'left', 'right', 'bottom', 'top'}:
            raise ValueError('Boundary sampling recipes must use named rectangle sides.')
        for side, recipe in self.boundary_sampling.items():
            if not isinstance(recipe, BoundarySampling):
                raise ValueError(f'{side}: expected a BoundarySampling recipe.')
            if recipe.method not in SAMPLERS:
                raise ValueError(f'{side}: unknown boundary sampler.')
            if type(recipe.points) is not int or not 1 <= recipe.points <= 10000:
                raise ValueError(f'{side}: point count must be between 1 and 10000.')
            if type(recipe.seed) is not int or not 0 <= recipe.seed < 2**31:
                raise ValueError(f'{side}: seed must be a non-negative 32-bit integer.')

    def validate(self):
        if self.pde_preset:
            from .predefined_pdes import resolve_predefined,validate_predefined
            resolve_predefined(self);validate_predefined(self)
        self.validate_sampling()
        from ..components.adaptive_sampling import validate_adaptive_sampling
        validate_adaptive_sampling(self.adaptive_sampling)
        from ..components.losses import LOSS_FORMULATIONS
        if not isinstance(self.loss_formulation,str) or self.loss_formulation not in LOSS_FORMULATIONS:raise ValueError('Unknown loss formulation.')
        if self.loss_formulation=='integral':
            if self.interior_data or self.boundary_data:raise ValueError('Integral loss on imported points requires explicit quadrature weights; use built-in geometry samples for now.')
            if self.adaptive_sampling.get('method')=='rar':raise ValueError('Integral loss cannot treat RAR anchors as uniform quadrature. Use fixed uniform-domain sampling or importance sampling.')
        if self.adaptive_sampling:
            if self.interior_data:raise ValueError('Adaptive candidate generation currently supports built-in domains only; imported interior points need an explicit candidate source.')
            if self.method=='self_adaptive':raise ValueError('Adaptive sampling cannot yet move SA-PINN pointwise masks. Use baseline, hard-BC or LRA training.')
            initial=math.isqrt(self.points)**2 if len(self.bounds[0])==2 and self.sampling=='grid' else self.points
            if self.adaptive_sampling['method']=='rar' and initial>self.adaptive_sampling['maximum']:raise ValueError('RAR maximum must cover the initial interior set.')
        if type(self.sa_boundary) is not bool or any(not math.isfinite(v) or v<=0 for v in (self.sa_lr,self.sa_initial,self.sa_max)) or self.sa_initial>self.sa_max:raise ValueError('Invalid self-adaptive rate, initial weight or cap.')
        if type(self.lra_alpha) not in (int,float) or type(self.lra_epsilon) not in (int,float) or type(self.lra_max) not in (int,float) or not math.isfinite(self.lra_alpha) or not 0<self.lra_alpha<=1 or type(self.lra_interval) is not int or not 1<=self.lra_interval<=100000 or not math.isfinite(self.lra_epsilon) or not 0<self.lra_epsilon<=1 or not math.isfinite(self.lra_max) or not 1<=self.lra_max<=1e12:raise ValueError('Invalid LRA smoothing, interval, gradient floor or coefficient cap.')
        if self.physics not in ('scalar','incompressible_2d','paper_case'):raise ValueError('Unknown physics contract.')
        if self.pde_preset=='kdv_1d':
            from .kdv import validate_kdv
            validate_kdv(self)
        if self.network_type=='affine_fourier':
            from ..components.affine_inputs import affine_map
            from ..components.network_config import network_options
            affine_map(network_options(self.network_type,self.network_options)['input_transform'],self.coordinate_names)
        from ..components.gpinn_config import validate_gpinn_weights,validate_gpinn_problem
        validate_gpinn_weights(self.gpinn_weights)
        if self.method=='gpinn':validate_gpinn_problem(self)
        validate_scale_settings(self.scale_settings)
        if self.method=='scale':validate_scale_problem(self)
        validate_expert_settings(self.expert_settings)
        if self.method=='expert':validate_expert_problem(self)
        if self.method not in ('baseline','hard_dirichlet','self_adaptive','lra','expert','scale','gpinn'):raise ValueError('Unknown PINN method.')
        if self.method=='hard_dirichlet' and (self.boundary_data or any(self.bc_targets.values()) or any(self.bc_types.get(s)!='Dirichlet' for s in self.sides) or len({self.bc_values[s] for s in self.sides})!=1):raise ValueError('Hard Dirichlet requires equal constant Dirichlet values on all built-in sides.')
        if self.network_seed is not None and (type(self.network_seed) is not int or not 0<=self.network_seed<2**31):raise ValueError('Network seed must be a non-negative 32-bit integer.')
        if type(self.iterations) is not int or not 1 <= self.iterations <= 100000:
            raise ValueError('Iterations must be between 1 and 100000.')
        if not self.layers or len(self.layers) > 6 or any(type(n) is not int or not 1 <= n <= 256 for n in self.layers):
            raise ValueError('Use 1–6 hidden layers, with 1–256 neurons each.')
        from ..components.network_config import validate_network
        validate_network(self.network_type,self.network_options,self.layers)
        if not math.isfinite(self.lr) or not 0 < self.lr <= 1:
            raise ValueError('Learning rate must be greater than zero and at most one.')
        if self.device not in ('cpu','auto','cuda'):
            raise ValueError('Unknown device.')
        if not math.isfinite(self.source):
            raise ValueError('PDE source must be finite.')
        if any(not math.isfinite(w) or w <= 0 for w in (self.pde_weight,self.boundary_weight)):
            raise ValueError('Loss weights must be finite and positive.')
        if self.pde_preset=='kdv_1d':return
        if self.is_paper_case:
            from .expert_examples import validate_case
            validate_case(self);return
        if self.shape in ('SpaceTime2D','Channel2D','ChannelTime2D'):raise ValueError('This geometry currently requires its concrete Expert example preset.')
        if self.shape=='Prepared2D' and (self.physics!='scalar' or self.method not in ('baseline','lra','self_adaptive','gpinn') or self.is_paper_case):raise ValueError('Prepared 2D currently supports scalar baseline, LRA, gPINN and fixed-point SA-PINN; box transforms and fresh samplers do not apply.')
        if self.physics=='incompressible_2d':
            from .incompressible_flow import validate_flow_config
            validate_flow_config(self);return
        if self.is_time_dependent:
            if self.method=='hard_dirichlet':raise ValueError('The all-sides hard-Dirichlet transform does not apply to an initial-value problem.')
            if self.bc_types['bottom']!='Dirichlet':raise ValueError('The initial set requires a Dirichlet value/profile.')
            if not self.pde_expression:raise ValueError('Choose the Burgers preset or enter an explicit time-dependent residual.')
        dimension=len(self.bounds[0])
        if self.pde_expression:Expression(self.pde_expression,dimension,coordinates=self.coordinate_names)
        for side in self.sides:
            if self.bc_targets.get(side):Expression(self.bc_targets[side],dimension,target=True,coordinates=self.coordinate_names)
            if self.bc_types.get(side)=='Custom':Expression(self.bc_operators.get(side,''),dimension,coordinates=self.coordinate_names)
            if self.bc_types.get(side) not in ('Dirichlet','Neumann','Custom') or not math.isfinite(self.bc_values.get(side,float('nan'))):
                raise ValueError(f'Assign a valid type and finite target to {side}.')
        if not any(self.bc_types[s] == 'Dirichlet' for s in self.sides):
            raise ValueError('Pure Neumann/custom-only problems need a compatibility check and solution gauge; use at least one Dirichlet side for now.')
        # Sampling excludes corners, but incompatible endpoint values still make
        # the prescribed continuous Dirichlet problem ambiguous at those corners.
        if self.shape in ('Rectangle','SpaceTime1D'):
            corners=[('left','bottom'),('right','bottom')] if self.is_time_dependent else [('left','bottom'),('left','top'),('right','bottom'),('right','top')]
            for a,b in corners:
                if self.bc_types[a] == self.bc_types[b] == 'Dirichlet' and not self.boundary_data.get(a) and not self.boundary_data.get(b):
                    if self.bc_targets.get(a) or self.bc_targets.get(b):
                        import torch
                        corner=torch.tensor([[self.x_min if a=='left' else self.x_max,self.y_min if b=='bottom' else self.y_max]])
                        values=[float(Expression(self.bc_targets[s],2,target=True,coordinates=self.coordinate_names)(corner,source=self.source,viscosity=self.viscosity,normal={'left':(-1.,0.),'right':(1.,0.),'bottom':(0.,-1.),'top':(0.,1.)}[s])) if self.bc_targets.get(s) else self.bc_values[s] for s in (a,b)]
                        agree=all(math.isfinite(v) for v in values) and math.isclose(*values,rel_tol=1e-6,abs_tol=1e-6)
                    else:agree=self.bc_values[a]==self.bc_values[b]
                    if not agree:raise ValueError(f'Dirichlet targets on {a} and {b} conflict at their shared corner.')


def sample_problem(config):
    """Return interior and named boundary coordinates, before assigning physics."""
    config.validate_sampling()
    if config.shape=='Prepared2D':
        from ..components.training_data import read_training_sets
        interior,boundaries=read_training_sets(config,check_physics=False)
        return interior[0].tolist(),{name:rows[0].tolist() for name,rows in boundaries.items()}
    lower, upper = config.bounds
    interior = read_points(config.interior_data,len(lower))[0].tolist() if config.interior_data else sample_box(lower, upper, config.points, config.sampling, config.seed)
    if config.shape == 'Interval':
        return interior, {side:read_points(config.boundary_data[side],1)[0].tolist() if config.boundary_data.get(side) else [[config.x_min if side=='left' else config.x_max]] for side in config.sides}
    boundaries = {}
    for side in config.sides:
        if config.boundary_data.get(side):
            boundaries[side]=read_points(config.boundary_data[side],2)[0].tolist();continue
        vertical = side in ('left','right')
        a,b = (config.y_min,config.y_max) if vertical else (config.x_min,config.x_max)
        recipe = config.boundary_recipe(side)
        rows = sample_box([a], [b], recipe.points, recipe.method, recipe.seed)
        fixed = {'left':config.x_min,'right':config.x_max,'bottom':config.y_min,'top':config.y_max}[side]
        boundaries[side] = [[fixed,t[0]] if vertical else [t[0],fixed] for t in rows]
    return interior, boundaries


def build_pinn(config,*,initial_teacher=None,time_origin=None):
    from ..components.adaptive_sampling import configure_adaptive_sampling
    config.validate()
    if config.method=='scale':
        from copy import deepcopy
        from ..methods.scale import ScalePINN
        baseline=deepcopy(config);baseline.method='baseline'
        model=_build_pinn(baseline)
        return ScalePINN(model.problem,model.network,config=config).compile(lr=config.lr,device=config.device)
    if config.method=='expert' and config.expert_settings['windows']>1:
        from ..methods.time_marching import TimeMarchingPINN
        return TimeMarchingPINN(config)
    return configure_adaptive_sampling(_build_pinn(config,initial_teacher=initial_teacher,time_origin=time_origin),config)


def _build_pinn(config,*,initial_teacher=None,time_origin=None):
    """Assemble the existing baseline recipe; torch is loaded only on execution."""
    config.validate()
    if config.pde_preset=='kdv_1d':
        from .kdv import build_kdv
        return build_kdv(config)
    if config.is_paper_case:
        from .expert_examples import build_case
        return build_case(config,initial_teacher=initial_teacher,time_origin=time_origin)
    if config.physics=='incompressible_2d':
        from .incompressible_flow import build_flow_pinn
        return build_flow_pinn(config)
    import torch
    from ..components import PINNProblem, PointConstraint
    from ..methods import PINN
    from ..nn.factory import build_network
    from ..utils import laplacian
    imported_sets=None
    if config.shape=='Prepared2D':
        from ..components.training_data import read_training_sets
        imported_interior,imported_sets=read_training_sets(config)
        interior=imported_interior[0].tolist();boundaries={name:rows[0].tolist() for name,rows in imported_sets.items()}
    else:interior, boundaries = sample_problem(config)
    torch.manual_seed(config.seed if config.network_seed is None else config.network_seed)
    normals = {'left': (-1.,0.), 'right': (1.,0.), 'bottom': (0.,-1.), 'top': (0.,1.)}
    constraints = [];normal_identities={}
    for side, rows in boundaries.items():
        imported=imported_sets[side] if imported_sets is not None else read_points(config.boundary_data[side],len(config.bounds[0])) if config.boundary_data.get(side) else None
        normal=normals.get(side,(0.,0.))[:len(rows[0])]
        if imported is not None:
            if imported[1] is not None:normal_identities[side]=normal_fingerprint(imported[1])
            normal=imported[1].tolist() if imported[1] is not None else normal
            # Without imported normals a side label is meaningful only on that
            # box side. Do not silently use analytical normals for arbitrary CAD.
            needs_normal=config.bc_types[side]=='Neumann' or config.bc_types[side]=='Custom' and Expression(config.bc_operators[side],len(rows[0]),coordinates=config.coordinate_names).uses_normal or bool(config.bc_targets.get(side)) and Expression(config.bc_targets[side],len(rows[0]),target=True,coordinates=config.coordinate_names).uses_normal
            if imported[1] is None and needs_normal:
                axis=0 if side in ('left','right') else 1
                edge={'left':config.x_min,'right':config.x_max,'bottom':config.y_min,'top':config.y_max}[side]
                if any(abs(row[axis]-edge)>1e-7 for row in rows):raise ValueError('Imported off-box boundary sets require explicit unit outward normals.')
        operator = None
        if config.bc_types[side] == 'Neumann':
            def operator(x, u, normal=normal):
                derivative = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
                return (derivative * x.new_tensor(normal)).sum(dim=1,keepdim=True)
        if config.bc_types[side]=='Custom':
            expression=Expression(config.bc_operators[side],len(rows[0]),coordinates=config.coordinate_names)
            def operator(x,u,expression=expression,normal=normal):
                return expression(x,u,source=config.source,viscosity=config.viscosity,normal=tuple(x.new_tensor(normal)[:,i:i+1] for i in range(x.shape[1])) if isinstance(normal[0],list) else normal)
        target=config.bc_values[side]
        if config.bc_targets.get(side):
            expression=Expression(config.bc_targets[side],len(rows[0]),target=True,coordinates=config.coordinate_names)
            def target(x,expression=expression,normal=normal):
                return expression(x,source=config.source,viscosity=config.viscosity,normal=tuple(x.new_tensor(normal)[:,i:i+1] for i in range(x.shape[1])) if isinstance(normal[0],list) else normal)
        if imported is not None and imported[2] is not None:
            if config.bc_targets.get(side):raise ValueError('Choose imported target values or a target expression, not both.')
            target=torch.tensor(imported[2])
        constraints.append(PointConstraint(config.constraint_name(side), torch.tensor(rows), target, operator))
    expression=Expression(config.pde_expression,len(config.bounds[0]),coordinates=config.coordinate_names) if config.pde_expression else None
    def residual(x, u):
        return expression(x,u,source=config.source,viscosity=config.viscosity) if expression else laplacian(u,x) + config.source
    problem = PINNProblem(residual,torch.tensor(interior),constraints,data_identity=json.dumps(normal_identities,sort_keys=True) if normal_identities else '')
    lower,upper = config.bounds
    from ..components.losses import apply_problem_measures
    apply_problem_measures(problem,config)
    network = build_network(config)
    weights = config.loss_weights
    if config.method=='hard_dirichlet':
        from ..methods.hard_dirichlet import HardDirichletPINN
        return HardDirichletPINN(problem,network,bounds=(lower,upper),value=config.bc_values['left'],weights=weights,loss_formulation=config.loss_formulation).compile(lr=config.lr,device=config.device)
    if config.method=='expert':
        from ..methods.expert import ExpertPINN
        return ExpertPINN(problem,network,config=config,weights=weights,loss_formulation=config.loss_formulation,initial_teacher=initial_teacher).compile(lr=config.lr,device=config.device)
    if config.method=='self_adaptive':
        from ..methods.self_adaptive import SelfAdaptivePINN
        return SelfAdaptivePINN(problem,network,weights=weights,loss_formulation=config.loss_formulation).compile(lr=config.lr,device=config.device,adaptive_lr=config.sa_lr,initial_weight=config.sa_initial,max_weight=config.sa_max,adapt_boundary=config.sa_boundary)
    if config.method=='gpinn':
        from dataclasses import asdict
        from ..methods.gpinn import GradientEnhancedPINN
        return GradientEnhancedPINN(problem,network,coordinates=config.coordinate_names,gradient_weights=config.gpinn_weights,weights=weights,configuration=asdict(config)).compile(lr=config.lr,device=config.device)
    if config.method=='lra':
        from ..methods.learning_rate_annealing import LearningRateAnnealingPINN
        return LearningRateAnnealingPINN(problem,network,weights=weights,loss_formulation=config.loss_formulation).compile(lr=config.lr,device=config.device,alpha=config.lra_alpha,update_interval=config.lra_interval,epsilon=config.lra_epsilon,max_weight=config.lra_max)
    return PINN(problem,network,weights=weights,loss_formulation=config.loss_formulation).compile(lr=config.lr,device=config.device)


def has_custom_physics(config):
    return bool(config.interior_data or config.boundary_data or config.pde_expression or any(config.bc_targets.values()) or
                any(config.bc_types[s]=='Custom' for s in config.sides))


def interval_reference(config, points):
    """Exact solution for constant-source interval problems with D/D or D/N BCs."""
    if config.shape != 'Interval' or has_custom_physics(config):
        return None
    length = config.x_max-config.x_min
    lv,rv = config.bc_values['left'],config.bc_values['right']
    if config.bc_types['left'] == 'Neumann':
        slope = -lv
        offset = rv + .5*config.source*length**2 - slope*length
    else:
        offset = lv
        slope = (rv-lv+.5*config.source*length**2)/length if config.bc_types['right']=='Dirichlet' else rv+config.source*length
    return [-.5*config.source*(p[0]-config.x_min)**2+slope*(p[0]-config.x_min)+offset for p in points]


def make_interval_reference(config):
    """Create an explicit demo reference source; never selected by the trainer."""
    from functools import partial
    from copy import deepcopy
    from ..components.outputs import ReferenceSource
    config.validate()
    if has_custom_physics(config):raise ValueError('Preset references do not apply to custom expressions.')
    if config.shape != 'Interval':
        raise ValueError('The demo analytical reference supports intervals only; prediction plots do not require it.')
    return ReferenceSource('Analytical interval Poisson', partial(interval_reference, deepcopy(config)))


def rectangle_reference(config, points, terms=40):
    """Constant-source rectangular Poisson reference, truncated odd sine modes.

    Equal constant Dirichlet values on every side. The infinite series solves
    the equation exactly; this explicitly labelled finite reference uses 40
    odd modes per axis by default, not an exact closed-form solution.
    """
    import numpy as np
    config.validate()
    if has_custom_physics(config):raise ValueError('Preset references do not apply to custom expressions.')
    if config.shape != 'Rectangle' or any(config.bc_types[s]!='Dirichlet' for s in config.sides):
        raise ValueError('Rectangle reference requires Dirichlet conditions on every side.')
    if len({config.bc_values[s] for s in config.sides}) != 1:
        raise ValueError('Rectangle reference requires equal constant boundary values.')
    if type(terms) is not int or not 1 <= terms <= 160:
        raise ValueError('Reference terms must be between 1 and 160.')
    coordinates=np.asarray(points,dtype=float)
    if coordinates.ndim!=2 or coordinates.shape[1]!=2 or not np.isfinite(coordinates).all():
        raise ValueError('Reference points must be finite (N,2) coordinates.')
    width,height=config.x_max-config.x_min,config.y_max-config.y_min
    modes=np.arange(1,2*terms,2,dtype=float)
    denominator=modes[:,None]*modes[None,:]*((np.pi*modes[:,None]/width)**2+(np.pi*modes[None,:]/height)**2)
    coefficients=(16*config.source/np.pi**2)/denominator
    result=[]
    for start in range(0,len(coordinates),1024):
        chunk=coordinates[start:start+1024]
        sx=np.sin(np.pi*(chunk[:,0:1]-config.x_min)/width*modes)
        sy=np.sin(np.pi*(chunk[:,1:2]-config.y_min)/height*modes)
        result.extend((config.bc_values['left']+((sx@coefficients)*sy).sum(axis=1)).tolist())
    return result


def make_poisson_reference(config):
    """Explicit demo reference for supported interval or rectangle conditions."""
    from functools import partial
    from copy import deepcopy
    from ..components.outputs import ReferenceSource
    if config.is_paper_case:raise ValueError('Expert examples need an explicit reference expression or reference data; no Poisson reference is assumed.')
    if config.physics=='incompressible_2d':
        from .incompressible_flow import make_flow_reference
        return make_flow_reference(config)
    if has_custom_physics(config):
        raise ValueError('Preset references do not apply to custom expressions. Remove Reference/Comparison or use an explicit compatible source.')
    if config.shape=='Interval':return make_interval_reference(config)
    config.validate()
    if config.shape!='Rectangle' or any(config.bc_types[s]!='Dirichlet' for s in config.sides):
        raise ValueError('Rectangle reference requires equal constant Dirichlet conditions on all sides; prediction plots need no reference.')
    return ReferenceSource('Rectangle sine series · 40 odd modes/axis (truncated)',partial(rectangle_reference,deepcopy(config)))
