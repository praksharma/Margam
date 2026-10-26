"""Editable Poisson template; method-specific training remains in methods/pinn.py."""
from dataclasses import dataclass, field
import math
from ..components.sampling import sample_box, SAMPLERS


@dataclass
class PoissonConfig:
    shape: str = 'Rectangle'
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    points: int = 64
    boundary_points: int = 16
    sampling: str = 'grid'
    seed: int = 7
    source: float = 2.0
    layers: tuple = (16, 16)
    iterations: int = 1000
    lr: float = .01
    device: str = 'cpu'
    pde_weight: float = 1.0
    boundary_weight: float = 1.0
    bc_types: dict = field(default_factory=lambda: dict.fromkeys(('left', 'right', 'bottom', 'top'), 'Dirichlet'))
    bc_values: dict = field(default_factory=lambda: dict.fromkeys(('left', 'right', 'bottom', 'top'), 0.0))

    @property
    def sides(self):
        return ('left', 'right') if self.shape == 'Interval' else ('left', 'right', 'bottom', 'top')

    @property
    def bounds(self):
        if self.shape == 'Interval':
            return [self.x_min], [self.x_max]
        return [self.x_min, self.y_min], [self.x_max, self.y_max]

    def validate_sampling(self):
        if self.shape not in ('Interval', 'Rectangle'):
            raise ValueError('Choose Interval or Rectangle.')
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

    def validate(self):
        self.validate_sampling()
        if type(self.iterations) is not int or not 1 <= self.iterations <= 100000:
            raise ValueError('Iterations must be between 1 and 100000.')
        if not self.layers or len(self.layers) > 6 or any(type(n) is not int or not 1 <= n <= 256 for n in self.layers):
            raise ValueError('Use 1–6 hidden layers, with 1–256 neurons each.')
        if not math.isfinite(self.lr) or not 0 < self.lr <= 1:
            raise ValueError('Learning rate must be greater than zero and at most one.')
        if self.device not in ('cpu','auto','cuda'):
            raise ValueError('Unknown device.')
        if not math.isfinite(self.source):
            raise ValueError('PDE source must be finite.')
        if any(not math.isfinite(w) or w <= 0 for w in (self.pde_weight,self.boundary_weight)):
            raise ValueError('Loss weights must be finite and positive.')
        for side in self.sides:
            if self.bc_types.get(side) not in ('Dirichlet','Neumann') or not math.isfinite(self.bc_values.get(side,float('nan'))):
                raise ValueError(f'Assign a valid type and finite target to {side}.')
        if all(self.bc_types[s] == 'Neumann' for s in self.sides):
            raise ValueError('Pure Neumann problems need a compatibility check and solution gauge; use at least one Dirichlet side for now.')
        # Sampling excludes corners, but incompatible endpoint values still make
        # the prescribed continuous Dirichlet problem ambiguous at those corners.
        if self.shape == 'Rectangle':
            for a,b in [('left','bottom'),('left','top'),('right','bottom'),('right','top')]:
                if self.bc_types[a] == self.bc_types[b] == 'Dirichlet' and self.bc_values[a] != self.bc_values[b]:
                    raise ValueError(f'Dirichlet targets on {a} and {b} conflict at their shared corner.')


def sample_problem(config):
    """Return interior and named boundary coordinates, before assigning physics."""
    config.validate_sampling()
    lower, upper = config.bounds
    interior = sample_box(lower, upper, config.points, config.sampling, config.seed)
    if config.shape == 'Interval':
        return interior, {'left': [[config.x_min]], 'right': [[config.x_max]]}
    boundaries = {}
    for i,side in enumerate(config.sides):
        vertical = side in ('left','right')
        a,b = (config.y_min,config.y_max) if vertical else (config.x_min,config.x_max)
        rows = sample_box([a],[b],config.boundary_points,config.sampling,config.seed+i+1)
        fixed = {'left':config.x_min,'right':config.x_max,'bottom':config.y_min,'top':config.y_max}[side]
        boundaries[side] = [[fixed,t[0]] if vertical else [t[0],fixed] for t in rows]
    return interior, boundaries


def build_pinn(config):
    """Assemble the existing baseline recipe; torch is loaded only on execution."""
    config.validate()
    import torch
    from ..components import PINNProblem, PointConstraint
    from ..methods import PINN
    from ..nn import FullyConnected
    from ..utils import laplacian
    interior, boundaries = sample_problem(config)
    torch.manual_seed(config.seed)
    normals = {'left': (-1.,0.), 'right': (1.,0.), 'bottom': (0.,-1.), 'top': (0.,1.)}
    constraints = []
    for side, rows in boundaries.items():
        operator = None
        if config.bc_types[side] == 'Neumann':
            normal = normals[side][:len(rows[0])]
            def operator(x, u, normal=normal):
                derivative = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
                return (derivative * x.new_tensor(normal)).sum(dim=1,keepdim=True)
        constraints.append(PointConstraint(side, torch.tensor(rows), config.bc_values[side], operator))
    def residual(x, u):
        return laplacian(u,x) + config.source
    problem = PINNProblem(residual,torch.tensor(interior),constraints)
    lower,upper = config.bounds
    network = FullyConnected([len(lower),*config.layers,1], 'tanh', 'Xavier normal',input_bounds=(lower,upper))
    weights = {'pde':config.pde_weight, **{s:config.boundary_weight/len(config.sides) for s in config.sides}}
    return PINN(problem,network,weights=weights).compile(lr=config.lr,device=config.device)


def interval_reference(config, points):
    """Exact solution for constant-source interval problems with D/D or D/N BCs."""
    if config.shape != 'Interval':
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
    if config.shape=='Interval':return make_interval_reference(config)
    config.validate()
    if config.shape!='Rectangle' or any(config.bc_types[s]!='Dirichlet' for s in config.sides):
        raise ValueError('Rectangle reference requires equal constant Dirichlet conditions on all sides; prediction plots need no reference.')
    return ReferenceSource('Rectangle sine series · 40 odd modes/axis (truncated)',partial(rectangle_reference,deepcopy(config)))
