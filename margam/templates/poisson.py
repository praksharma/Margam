"""Editable Poisson template; method-specific training remains in methods/pinn.py."""
from ..components.dsgnar_config import dsgnar_defaults
from dataclasses import dataclass, field
import math
import json
import re
from ..components.sampling import sample_box, SAMPLERS
from ..components.scale_config import scale_defaults,validate_scale_settings,validate_scale_problem
from ..components.expert_config import expert_defaults,validate_expert_settings,validate_expert_problem
from ..components.expressions import Expression
from ..components.prepared_data import read_points, validate_mapping, normal_fingerprint


RESERVED_PARAM_NAMES = frozenset({
    'x', 'y', 'z', 't', 'u', 'v', 'w', 'p', 'c', 'nu', 'pi', 'nx', 'ny', 'nz',
    'D', 'dn', 'laplace', 'sin', 'cos', 'exp', 'tanh', 'sqrt', 'abs',
    'True', 'False', 'None', 'and', 'or', 'not', 'in', 'is', 'if', 'else',
    'for', 'while', 'def', 'class', 'return', 'lambda', 'import', 'from',
})


def validate_parametric_axes(axes):
    """Validate continuous parametric axes: up to 4 named finite [min, max] intervals."""
    if not isinstance(axes, (list, tuple)) or len(axes) > 4:
        raise ValueError('Continuous parametric axes must be a sequence of at most 4 parameter definitions.')
    seen = set()
    normalized = []
    for entry in axes:
        if not isinstance(entry, dict) or set(entry) != {'name', 'min', 'max', 'default'}:
            raise ValueError('Each parametric axis needs exact keys: name, min, max, default.')
        name = entry['name']
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,31}', name):
            raise ValueError(f'Invalid parametric axis name: {name!r}. Use letters, digits, and underscores starting with a letter.')
        if name in RESERVED_PARAM_NAMES:
            raise ValueError(f'Parametric axis name {name!r} collides with a reserved coordinate, field, or math symbol.')
        if name in seen:
            raise ValueError(f'Duplicate parametric axis name: {name!r}.')
        seen.add(name)
        lo, hi, default = entry['min'], entry['max'], entry['default']
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (lo, hi, default)):
            raise ValueError(f'Parametric axis {name!r} bounds and default must be finite numbers.')
        lo, hi, default = float(lo), float(hi), float(default)
        if not (lo < hi):
            raise ValueError(f'Parametric axis {name!r} requires min < max.')
        if not (lo <= default <= hi):
            raise ValueError(f'Parametric axis {name!r} default must lie within [min, max].')
        normalized.append({'name': name, 'min': lo, 'max': hi, 'default': default})
    return tuple(normalized)


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
    z_min: float = 0.0
    z_max: float = 1.0
    t_min: float = 0.
    t_max: float = 1.
    cylinder_x: float = .2
    cylinder_y: float = .2
    cylinder_radius: float = .05
    sphere_x: float = .5
    sphere_y: float = .5
    sphere_z: float = .5
    sphere_radius: float = .2
    parametric_axes: tuple = ()
    fourth_order: float = 100/16**4
    startup_time: float = .1
    pde_points: str = 'interior'  # Legacy/direct API default; UI graphs choose by wiring.
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
    bc_weights: dict = field(default_factory=dict)
    bc_types: dict = field(default_factory=lambda: dict.fromkeys(('left', 'right', 'bottom', 'top'), 'Dirichlet'))
    bc_values: dict = field(default_factory=lambda: dict.fromkeys(('left', 'right', 'bottom', 'top'), 0.0))
    boundary_sampling: dict = field(default_factory=dict)
    pde_expression: str = ''
    bc_targets: dict = field(default_factory=dict)
    bc_operators: dict = field(default_factory=dict)
    interior_data: dict = field(default_factory=dict)
    boundary_data: dict = field(default_factory=dict)
    network_seed: int | None = None
    inverse_settings: dict = field(default_factory=lambda:dict(lambda_1=0.,log_lambda_2=-6.))
    dsgnar_settings: dict = field(default_factory=dsgnar_defaults)
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
    rba_settings: dict = field(default_factory=lambda: dict(decay=.999,rate=.001,initialization='zero'))
    gpinn_weights: dict = field(default_factory=lambda: dict(x=.01,y=.01,t=.01))
    scale_settings: dict = field(default_factory=scale_defaults)

    @property
    def is_3d(self):return self.shape in ('Box3D','SpaceTime3D','SphereObstacle3D')

    @property
    def is_time_dependent(self):return self.shape in ('SpaceTime1D','SpaceTime2D','ChannelTime2D','SpaceTime3D')

    @property
    def is_paper_case(self):return self.pde_preset.startswith('expert_')

    @property
    def has_obstacle(self):return self.shape in ('Channel2D','ChannelTime2D','SphereObstacle3D')

    @property
    def parametric_names(self):return tuple(ax['name'] for ax in self.parametric_axes)

    @property
    def spatial_dim(self):return 3 if self.is_3d else 1 if self.shape in ('Interval','SpaceTime1D') else 2

    @property
    def spatial_time_bounds(self):
        if self.shape=='SpaceTime3D':return [self.x_min,self.y_min,self.z_min,self.t_min],[self.x_max,self.y_max,self.z_max,self.t_max]
        if self.is_3d:return [self.x_min,self.y_min,self.z_min],[self.x_max,self.y_max,self.z_max]
        if self.shape in ('SpaceTime2D','ChannelTime2D'):return [self.x_min,self.y_min,self.t_min],[self.x_max,self.y_max,self.t_max]
        if self.shape == 'Interval':
            return [self.x_min], [self.x_max]
        return [self.x_min, self.y_min], [self.x_max, self.y_max]

    @property
    def coordinate_names(self):
        if self.shape=='SpaceTime3D':base=('x','y','z','t')
        elif self.is_3d:base=('x','y','z')
        elif self.shape in ('SpaceTime2D','ChannelTime2D'):base=('x','y','t')
        elif self.is_time_dependent:base=('x','t')
        elif self.shape=='Interval':base=('x',)
        else:base=('x','y')
        return base+self.parametric_names

    def constraint_name(self,side):return 'initial' if self.shape=='SpaceTime1D' and side=='bottom' else side

    def side_label(self,side):return 'initial (t start)' if (self.shape=='SpaceTime1D' and side=='bottom') or side=='initial' else side

    @property
    def field_names(self):
        if self.is_paper_case:
            from .expert_examples import CASES
            return CASES[self.pde_preset]['fields']
        return ('u','v','p') if self.physics=='incompressible_2d' else ('u',)

    @property
    def constraint_names(self):
        if self.method=='inverse_burgers':return ('observations',)
        if self.pde_preset=='wave_1d':return ('initial','initial_velocity','boundary')
        if self.pde_preset=='kdv_1d':return ('initial','periodic_x')
        if self.is_paper_case:
            from .expert_examples import CASES
            return CASES[self.pde_preset]['names']
        if self.physics=='scalar':return tuple(self.constraint_name(s) for s in self.sides)
        return tuple(side+':'+field for side in self.sides for field,spec in self.flow_boundary[side].items() if spec['type']!='None')+('pressure_gauge',)

    @property
    def loss_weights(self):
        if self.method=='inverse_burgers':return dict(pde=self.pde_weight,observations=self.boundary_weight)
        if self.pde_preset=='kdv_1d':weights={'pde':self.pde_weight,'initial':self.boundary_weight/2,'periodic_x':self.boundary_weight/2}
        elif self.pde_preset=='wave_1d':weights={'pde':self.pde_weight,**{name:self.boundary_weight/3 for name in self.constraint_names}}
        elif self.is_paper_case:weights={'pde':self.pde_weight,**{name:self.pressure_weight if name=='pressure_gauge' else self.boundary_weight/len(self.constraint_names) for name in self.constraint_names}}
        elif self.physics=='scalar':weights={'pde':self.pde_weight,**{self.constraint_name(s):self.boundary_weight/len(self.sides) for s in self.sides}}
        else:
            names=self.constraint_names
            weights={'pde':self.pde_weight,**{name:self.boundary_weight/(len(names)-1) for name in names if name!='pressure_gauge'},'pressure_gauge':self.pressure_weight}

        return {name:value*self.bc_weights.get(name,1.) if name!='pde' else value for name,value in weights.items()}

    @property
    def sides(self):
        if self.shape=='Prepared2D':return tuple(self.boundary_data)
        if self.shape=='Box3D':return ('left','right','bottom','top','front','back')
        if self.shape=='SpaceTime3D':return ('left','right','bottom','top','front','back','initial')
        if self.shape=='SphereObstacle3D':return ('left','right','bottom','top','front','back','sphere')
        return ('left','right','bottom') if self.is_time_dependent else ('left', 'right') if self.shape == 'Interval' else ('left', 'right', 'bottom', 'top')

    @property
    def bounds(self):
        lo, hi = self.spatial_time_bounds
        if self.parametric_axes:
            lo = list(lo) + [float(ax['min']) for ax in self.parametric_axes]
            hi = list(hi) + [float(ax['max']) for ax in self.parametric_axes]
        return lo, hi

    def boundary_recipe(self, side):
        """Unconfigured sides use grid, the template count, and stable side seeds."""
        sides = ('left', 'right', 'bottom', 'top', 'front', 'back', 'sphere', 'initial')
        if side not in sides:
            raise ValueError('Unknown boundary side.')
        return self.boundary_sampling.get(
            side, BoundarySampling(self.boundary_points, 'grid', 8 + sides.index(side)))

    def validate_sampling(self):
        from ..components.losses import validate_boundary_weights
        validate_boundary_weights(self.bc_weights)
        if self.pde_points not in ('interior','boundary','both'):raise ValueError('Choose interior, boundary or both PDE collocation sets.')
        if self.interior_data:
            validate_mapping(self.interior_data)
            if self.interior_data['normals'] or self.interior_data['values'] and self.method!='inverse_burgers':raise ValueError('PDE point imports accept coordinates only; additional pointwise inputs remain deferred.')
        from ..components.training_data import valid_boundary_name
        if not isinstance(self.boundary_data,dict) or len(self.boundary_data)>64 or any(not valid_boundary_name(name) for name in self.boundary_data) or self.shape!='Prepared2D' and set(self.boundary_data)-{'left','right','bottom','top','front','back','sphere','initial'}:raise ValueError('Unknown or invalid imported boundary set.')
        if self.shape=='Prepared2D' and ((self.pde_points!='boundary' and not self.interior_data) or not self.boundary_data):raise ValueError('Prepared 2D requires interior data and at least one named boundary set.')
        for source in self.boundary_data.values():validate_mapping(source)
        if self.shape not in ('Interval', 'Rectangle','Prepared2D','SpaceTime1D','SpaceTime2D','Channel2D','ChannelTime2D','Box3D','SpaceTime3D','SphereObstacle3D'):
            raise ValueError('Choose Interval, Rectangle, 1D space–time, or a supported 3D domain.')
        validate_parametric_axes(self.parametric_axes)
        if self.parametric_axes:
            if self.shape not in ('Interval','Rectangle','SpaceTime1D','Box3D','SpaceTime3D','SphereObstacle3D'):
                raise ValueError('Continuous parametric axes currently support built-in Interval, Rectangle, 1D space–time, and 3D box/sphere domains.')
            if len(self.bounds[0])>8:
                raise ValueError('Total spatial, time, and continuous parameter dimensionality cannot exceed 8.')
        lower, upper = self.bounds
        if not all(math.isfinite(v) for v in (*lower, *upper)) or any(b <= a for a,b in zip(lower,upper)):
            raise ValueError('Domain maximums must be greater than minimums.')
        if self.shape=='SphereObstacle3D':
            if not all(math.isfinite(v) for v in (self.sphere_x,self.sphere_y,self.sphere_z,self.sphere_radius)) or self.sphere_radius<=0:
                raise ValueError('Sphere obstacle centre must be finite and radius must be positive.')
            if not (self.x_min+self.sphere_radius<self.sphere_x<self.x_max-self.sphere_radius and
                    self.y_min+self.sphere_radius<self.sphere_y<self.y_max-self.sphere_radius and
                    self.z_min+self.sphere_radius<self.sphere_z<self.z_max-self.sphere_radius):
                raise ValueError('Sphere obstacle must lie strictly inside the 3D box bounds.')
        if self.sampling not in SAMPLERS:
            raise ValueError('Unknown sampler.')
        for value, limit, label in [(self.points,100000,'Interior points'),(self.boundary_points,10000,'Boundary points')]:
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError(f'{label} must be between 1 and {limit}.')
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError('Seed must be a non-negative 32-bit integer.')
        if not isinstance(self.boundary_sampling, dict) or set(self.boundary_sampling) - {'left', 'right', 'bottom', 'top', 'front', 'back', 'sphere', 'initial'}:
            raise ValueError('Boundary sampling recipes must use named rectangle sides or 3D faces.')
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
            from copy import deepcopy
            from .predefined_pdes import resolve_predefined,validate_predefined
            validate_predefined(self)
            candidate=deepcopy(self)
            resolve_predefined(candidate)
            candidate._validate_resolved()
            self.physics=candidate.physics
            self.pde_expression=candidate.pde_expression
            self.flow_residuals=candidate.flow_residuals
            return
        self._validate_resolved()

    def _validate_resolved(self):
        self.validate_sampling()
        from ..components.adaptive_sampling import validate_adaptive_sampling
        validate_adaptive_sampling(self.adaptive_sampling)
        from ..components.losses import LOSS_FORMULATIONS
        if not isinstance(self.loss_formulation,str) or self.loss_formulation not in LOSS_FORMULATIONS:raise ValueError('Unknown loss formulation.')
        if self.loss_formulation=='integral' and self.pde_points!='interior':raise ValueError('Domain integral loss needs interior quadrature only; disconnect boundary points from the PDE block.')
        if self.adaptive_sampling and self.pde_points=='boundary':raise ValueError('Boundary-only PDE points cannot use an interior adaptive sampler.')
        if self.loss_formulation=='integral':
            if self.interior_data or self.boundary_data:raise ValueError('Integral loss on imported points requires explicit quadrature weights; use built-in geometry samples for now.')
            if self.adaptive_sampling.get('method')=='rar':raise ValueError('Integral loss cannot treat RAR anchors as uniform quadrature. Use fixed uniform-domain sampling or importance sampling.')
        if self.adaptive_sampling:
            if self.adaptive_sampling.get('score')=='field_gradient':
                if self.is_time_dependent:raise ValueError('Field-gradient importance currently supports steady spatial domains only.')
                if self.adaptive_sampling.get('field',0)>=len(self.field_names):raise ValueError('Importance scoring field is not an output of this problem.')
            if self.interior_data:raise ValueError('Adaptive candidate generation currently supports built-in domains only; imported interior points need an explicit candidate source.')
            if self.method=='self_adaptive':raise ValueError('Adaptive sampling cannot yet move SA-PINN pointwise masks. Use baseline, hard-BC or LRA training.')
            initial=math.isqrt(self.points)**2 if len(self.bounds[0])==2 and self.sampling=='grid' else self.points
            if self.pde_points=='both':
                from ..components.collocation import boundary_rows
                _,sets=sample_problem(self);initial+=len(boundary_rows(self,sets))
            if initial>100000:raise ValueError('Adaptive PDE collocation supports at most 100000 active rows including boundary anchors.')
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
        from ..components.rba_config import validate_rba_settings,validate_rba_problem
        validate_rba_settings(self.rba_settings)
        if self.method=='rba':validate_rba_problem(self)
        validate_gpinn_weights(self.gpinn_weights)
        if self.method=='gpinn':validate_gpinn_problem(self)
        validate_scale_settings(self.scale_settings)
        if self.method=='scale':validate_scale_problem(self)
        validate_expert_settings(self.expert_settings)
        if self.method=='expert':validate_expert_problem(self)
        from ..components.dsgnar_config import validate_dsgnar_settings,validate_dsgnar_problem
        validate_dsgnar_settings(self.dsgnar_settings)
        if self.method=='dsgnar':validate_dsgnar_problem(self)
        if self.method not in ('baseline','hard_dirichlet','self_adaptive','lra','expert','scale','gpinn','rba','inverse_burgers','dsgnar'):raise ValueError('Unknown PINN method.')
        if self.parametric_axes and (self.physics!='scalar' or self.is_paper_case or self.pde_preset in ('kdv_1d','wave_1d') or self.method in ('expert','scale','dsgnar','inverse_burgers')):
            raise ValueError('Continuous parametric axes currently support scalar PDEs with baseline, hard Dirichlet, self-adaptive, LRA, or RBA trainers.')
        if self.method=='hard_dirichlet' and (self.shape=='SphereObstacle3D' or self.boundary_data or any(self.bc_targets.values()) or any(self.bc_types.get(s,'Dirichlet')!='Dirichlet' for s in self.sides) or len({self.bc_values.get(s,0.) for s in self.sides})!=1):raise ValueError('Hard Dirichlet requires equal constant Dirichlet values on all built-in box sides.')
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
        if self.method=='inverse_burgers':
            from ..components.inverse_config import validate_inverse
            validate_inverse(self);return
        if self.pde_preset=='wave_1d':
            from .wave import validate_wave
            validate_wave(self);return
        if self.pde_preset=='kdv_1d':return
        if self.is_paper_case:
            from .expert_examples import validate_case
            validate_case(self);return
        if self.shape in ('SpaceTime2D','Channel2D','ChannelTime2D'):raise ValueError('This geometry currently requires its concrete Expert example preset.')
        if self.shape=='Prepared2D' and (self.physics!='scalar' or self.method not in ('baseline','lra','self_adaptive','gpinn','rba') or self.is_paper_case):raise ValueError('Prepared 2D currently supports scalar baseline, LRA, gPINN, RBA and fixed-point SA-PINN; box transforms and fresh samplers do not apply.')
        if self.physics=='incompressible_2d':
            from .incompressible_flow import validate_flow_config
            validate_flow_config(self);return
        if self.is_time_dependent:
            if self.method=='hard_dirichlet':raise ValueError('The all-sides hard-Dirichlet transform does not apply to an initial-value problem.')
            init_side='initial' if self.shape=='SpaceTime3D' else 'bottom'
            if self.bc_types.get(init_side,'Dirichlet')!='Dirichlet':raise ValueError('The initial set requires a Dirichlet value/profile.')
            if not self.pde_expression:raise ValueError('Choose the Burgers preset or enter an explicit time-dependent residual.')
        dimension=len(self.bounds[0])
        if self.pde_expression:Expression(self.pde_expression,dimension,coordinates=self.coordinate_names)
        for side in self.sides:
            extra_3d = self.is_3d and side in ('front','back','sphere','initial')
            bc_type=self.bc_types.get(side,'Dirichlet' if extra_3d else None)
            bc_val=self.bc_values.get(side,0. if extra_3d else float('nan'))
            if self.bc_targets.get(side):Expression(self.bc_targets[side],dimension,target=True,coordinates=self.coordinate_names)
            if bc_type=='Custom':Expression(self.bc_operators.get(side,''),dimension,coordinates=self.coordinate_names)
            if bc_type not in ('Dirichlet','Neumann','Custom') or not math.isfinite(bc_val):
                raise ValueError(f'Assign a valid type and finite target to {side}.')
        if not any(self.bc_types.get(s,'Dirichlet') == 'Dirichlet' for s in self.sides):
            raise ValueError('Pure Neumann/custom-only problems need a compatibility check and solution gauge; use at least one Dirichlet side for now.')
        # Sampling excludes corners, but incompatible endpoint values still make
        # the prescribed continuous Dirichlet problem ambiguous at those corners.
        if self.shape in ('Rectangle','SpaceTime1D'):
            corners=[('left','bottom'),('right','bottom')] if self.is_time_dependent else [('left','bottom'),('left','top'),('right','bottom'),('right','top')]
            for a,b in corners:
                if self.bc_types[a] == self.bc_types[b] == 'Dirichlet' and not self.boundary_data.get(a) and not self.boundary_data.get(b):
                    if self.bc_targets.get(a) or self.bc_targets.get(b):
                        import torch
                        p_defaults=[float(ax['default']) for ax in self.parametric_axes]
                        corner=torch.tensor([[self.x_min if a=='left' else self.x_max,self.y_min if b=='bottom' else self.y_max,*p_defaults]])
                        values=[float(Expression(self.bc_targets[s],dimension,target=True,coordinates=self.coordinate_names)(corner,source=self.source,viscosity=self.viscosity,normal={'left':(-1.,0.),'right':(1.,0.),'bottom':(0.,-1.),'top':(0.,1.)}[s])) if self.bc_targets.get(s) else self.bc_values[s] for s in (a,b)]
                        agree=all(math.isfinite(v) for v in values) and math.isclose(*values,rel_tol=1e-6,abs_tol=1e-6)
                    else:agree=self.bc_values[a]==self.bc_values[b]
                    if not agree:raise ValueError(f'Dirichlet targets on {a} and {b} conflict at their shared corner.')


def _sample_sphere_surface(cx, cy, cz, radius, count, seed=7):
    """Deterministic golden-angle Fibonacci lattice on a 3D sphere surface."""
    golden = math.pi * (3.0 - math.sqrt(5.0))
    phase = (seed % 997) * 0.137
    pts = []
    for i in range(count):
        y_u = 1.0 - 2.0 * (i + 0.5) / count
        r_ring = math.sqrt(max(0.0, 1.0 - y_u * y_u))
        theta = golden * i + phase
        pts.append([cx + radius * math.cos(theta) * r_ring, cy + radius * y_u, cz + radius * math.sin(theta) * r_ring])
    return pts


def sample_problem(config):
    """Return interior and named boundary coordinates, before assigning physics."""
    config.validate_sampling()
    if config.shape=='Prepared2D':
        from ..components.training_data import read_training_sets
        interior,boundaries=read_training_sets(config,check_physics=False)
        return interior[0].tolist(),{name:rows[0].tolist() for name,rows in boundaries.items()}
    lower, upper = config.bounds
    if config.interior_data:
        interior = read_points(config.interior_data,len(lower))[0].tolist()
    else:
        from ..components.geometry import obstacle_spec, sample_fluid_box
        interior = sample_fluid_box(lower, upper, config.points, config.sampling, config.seed, obstacle=obstacle_spec(config))
    dim = len(lower)
    if config.shape == 'Interval':
        boundaries = {}
        for side in config.sides:
            if config.boundary_data.get(side):
                boundaries[side] = read_points(config.boundary_data[side], dim)[0].tolist()
                continue
            fixed_x = config.x_min if side == 'left' else config.x_max
            if dim == 1:
                boundaries[side] = [[fixed_x]]
            else:
                recipe = config.boundary_recipe(side)
                p_rows = sample_box(lower[1:], upper[1:], recipe.points, recipe.method, recipe.seed)
                boundaries[side] = [[fixed_x, *row] for row in p_rows]
        return interior, boundaries
    if config.is_3d:
        boundaries = {}
        for side in config.sides:
            if config.boundary_data.get(side):
                boundaries[side]=read_points(config.boundary_data[side],dim)[0].tolist();continue
            recipe = config.boundary_recipe(side)
            if side == 'sphere':
                sp_pts = _sample_sphere_surface(config.sphere_x,config.sphere_y,config.sphere_z,config.sphere_radius,recipe.points,recipe.seed)
                if dim > 3:
                    p_rows = sample_box(lower[3:], upper[3:], len(sp_pts), recipe.method, (recipe.seed + 17) % (2**31))
                    # Ensure parameter rows match sphere point count even for grid rounding
                    if len(p_rows) < len(sp_pts):
                        p_rows = sample_box(lower[3:], upper[3:], len(sp_pts), 'hammersley', (recipe.seed + 17) % (2**31))
                    boundaries[side] = [sp + p_rows[i] for i, sp in enumerate(sp_pts)]
                else:
                    boundaries[side] = sp_pts
                continue
            if side == 'initial':
                free_axes = [0, 1, 2] + list(range(4, dim))
                rows = sample_box([lower[i] for i in free_axes], [upper[i] for i in free_axes], recipe.points, recipe.method, recipe.seed)
                boundaries[side] = [r[:3] + [config.t_min] + r[3:] for r in rows]
                continue
            fixed_map = {'left':(0,config.x_min),'right':(0,config.x_max),'bottom':(1,config.y_min),'top':(1,config.y_max),'front':(2,config.z_min),'back':(2,config.z_max)}
            axis, fixed_val = fixed_map[side]
            free_axes = [i for i in range(dim) if i != axis]
            sub_lo = [lower[i] for i in free_axes]
            sub_hi = [upper[i] for i in free_axes]
            rows = sample_box(sub_lo, sub_hi, recipe.points, recipe.method, recipe.seed)
            out = []
            for r in rows:
                pt = [0.0] * dim
                pt[axis] = fixed_val
                for k, ax_i in enumerate(free_axes):
                    pt[ax_i] = r[k]
                out.append(pt)
            boundaries[side] = out
        return interior, boundaries
    boundaries = {}
    for side in config.sides:
        if config.boundary_data.get(side):
            boundaries[side]=read_points(config.boundary_data[side],dim)[0].tolist();continue
        axis = 0 if side in ('left','right') else 1
        fixed = {'left':config.x_min,'right':config.x_max,'bottom':config.y_min,'top':config.y_max}[side]
        free_axes = [i for i in range(dim) if i != axis]
        recipe = config.boundary_recipe(side)
        rows = sample_box([lower[i] for i in free_axes], [upper[i] for i in free_axes], recipe.points, recipe.method, recipe.seed)
        if dim == 2:
            boundaries[side] = [[fixed,t[0]] if axis==0 else [t[0],fixed] for t in rows]
        else:
            out = []
            for r in rows:
                pt = [0.0] * dim
                pt[axis] = fixed
                for k, ax_i in enumerate(free_axes):
                    pt[ax_i] = r[k]
                out.append(pt)
            boundaries[side] = out
    return interior, boundaries


def build_pinn(config,*,initial_teacher=None,time_origin=None):
    from ..components.adaptive_sampling import configure_adaptive_sampling
    config.validate()
    if config.method=='inverse_burgers':
        from ..methods.inverse_burgers import build_inverse
        return build_inverse(config)
    if config.method=='dsgnar':
        from copy import deepcopy
        from dataclasses import asdict
        from ..methods.dsgnar import DSGNARPINN
        baseline=deepcopy(config);baseline.method='baseline'
        import torch
        previous_dtype=torch.get_default_dtype()
        try:
            torch.set_default_dtype(getattr(torch,config.dsgnar_settings['precision']))
            model=_build_pinn(baseline)
        finally:torch.set_default_dtype(previous_dtype)
        return DSGNARPINN(model.problem,model.network,weights=config.loss_weights,settings=config.dsgnar_settings,configuration=asdict(config)).compile(lr=config.lr,device=config.device)
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
    if config.pde_preset=='wave_1d':
        from .wave import build_wave
        return build_wave(config)
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
    normals = {
        'left': (-1.,0.,0.,0.), 'right': (1.,0.,0.,0.),
        'bottom': (0.,-1.,0.,0.), 'top': (0.,1.,0.,0.),
        'front': (0.,0.,-1.,0.), 'back': (0.,0.,1.,0.),
        'initial': (0.,0.,0.,-1.),
    }
    constraints = [];normal_identities={}
    for side, rows in boundaries.items():
        dim = len(rows[0])
        bc_type = config.bc_types.get(side, 'Dirichlet')
        bc_val = config.bc_values.get(side, 0.0)
        imported=imported_sets[side] if imported_sets is not None else read_points(config.boundary_data[side],len(config.bounds[0])) if config.boundary_data.get(side) else None
        if side == 'sphere' and config.shape == 'SphereObstacle3D':
            sx,sy,sz,sr = config.sphere_x,config.sphere_y,config.sphere_z,config.sphere_radius
            normal = [[-(p[0]-sx)/sr, -(p[1]-sy)/sr, -(p[2]-sz)/sr] + [0.]*(dim-3) for p in rows]
        else:
            base_n = normals.get(side, (0.,)*dim)
            normal = base_n[:dim] + (0.,)*max(0, dim-len(base_n))
        if imported is not None:
            if imported[1] is not None:normal_identities[side]=normal_fingerprint(imported[1])
            normal=imported[1].tolist() if imported[1] is not None else normal
            # Without imported normals a side label is meaningful only on that
            # box side. Do not silently use analytical normals for arbitrary CAD.
            needs_normal=bc_type=='Neumann' or bc_type=='Custom' and Expression(config.bc_operators[side],dim,coordinates=config.coordinate_names).uses_normal or bool(config.bc_targets.get(side)) and Expression(config.bc_targets[side],dim,target=True,coordinates=config.coordinate_names).uses_normal
            if imported[1] is None and needs_normal:
                axis=0 if side in ('left','right') else 1 if side in ('bottom','top') else 2
                edge={'left':config.x_min,'right':config.x_max,'bottom':config.y_min,'top':config.y_max,'front':config.z_min,'back':config.z_max}[side]
                if any(abs(row[axis]-edge)>1e-7 for row in rows):raise ValueError('Imported off-box boundary sets require explicit unit outward normals.')
        operator = None
        if bc_type == 'Neumann':
            def operator(x, u, normal=normal):
                derivative = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0] if u.requires_grad else None
                if derivative is None:
                    return u.new_zeros((len(x), 1))
                return (derivative * derivative.new_tensor(normal)).sum(dim=1, keepdim=True)
        if bc_type=='Custom':
            expression=Expression(config.bc_operators[side],dim,coordinates=config.coordinate_names)
            def operator(x,u,expression=expression,normal=normal):
                return expression(x,u,source=config.source,viscosity=config.viscosity,normal=tuple(x.new_tensor(normal)[:,i:i+1] for i in range(x.shape[1])) if isinstance(normal[0],list) else normal)
        target=bc_val
        if config.bc_targets.get(side):
            expression=Expression(config.bc_targets[side],dim,target=True,coordinates=config.coordinate_names)
            def target(x,expression=expression,normal=normal):
                return expression(x,source=config.source,viscosity=config.viscosity,normal=tuple(x.new_tensor(normal)[:,i:i+1] for i in range(x.shape[1])) if isinstance(normal[0],list) else normal)
        if imported is not None and imported[2] is not None:
            if config.bc_targets.get(side):raise ValueError('Choose imported target values or a target expression, not both.')
            target=torch.tensor(imported[2])
        constraints.append(PointConstraint(config.constraint_name(side), torch.tensor(rows), target, operator))
    expr_text = config.pde_expression or ('laplace(u) + c' if config.parametric_axes else '')
    expression=Expression(expr_text,len(config.bounds[0]),coordinates=config.coordinate_names) if expr_text else None
    def residual(x, u):
        return expression(x,u,source=config.source,viscosity=config.viscosity) if expression else laplacian(u,x) + config.source
    from ..components.collocation import collocation_rows
    problem = PINNProblem(residual,torch.tensor(collocation_rows(config,interior,boundaries)),constraints,data_identity=json.dumps(normal_identities,sort_keys=True) if normal_identities else '')
    from ..components.losses import apply_problem_measures
    apply_problem_measures(problem,config)
    network = build_network(config)
    weights = config.loss_weights
    if config.method=='hard_dirichlet':
        from ..methods.hard_dirichlet import HardDirichletPINN
        return HardDirichletPINN(problem,network,bounds=config.spatial_time_bounds,value=config.bc_values['left'],weights=weights,loss_formulation=config.loss_formulation).compile(lr=config.lr,device=config.device)
    if config.method=='expert':
        from ..methods.expert import ExpertPINN
        return ExpertPINN(problem,network,config=config,weights=weights,loss_formulation=config.loss_formulation,initial_teacher=initial_teacher).compile(lr=config.lr,device=config.device)
    if config.method=='self_adaptive':
        from ..methods.self_adaptive import SelfAdaptivePINN
        return SelfAdaptivePINN(problem,network,weights=weights,loss_formulation=config.loss_formulation).compile(lr=config.lr,device=config.device,adaptive_lr=config.sa_lr,initial_weight=config.sa_initial,max_weight=config.sa_max,adapt_boundary=config.sa_boundary)
    if config.method=='rba':
        from dataclasses import asdict
        from ..methods.rba import ResidualAttentionPINN
        return ResidualAttentionPINN(problem,network,settings=config.rba_settings,weights=weights,configuration=asdict(config)).compile(lr=config.lr,device=config.device)
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
                any(config.bc_types.get(s,'Dirichlet')=='Custom' for s in config.sides))


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
