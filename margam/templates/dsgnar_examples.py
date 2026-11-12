"""Modest correctness presets, not reproductions of the paper's accuracy."""
from .wave import INITIAL,REFERENCE
MULTISCALE_REFERENCE='0.25*(sin(4*pi*x)*sin(4*pi*y)+sin(8*pi*x)*sin(8*pi*y)+sin(12*pi*x)*sin(12*pi*y)+sin(16*pi*x)*sin(16*pi*y))'
import math
# Numeric constant folding keeps the ordinary restricted-expression AST bounded.
MULTISCALE_FORCING='+'.join(f'{.5*(w*math.pi)**2!r}*sin({w*math.pi!r}*x)*sin({w*math.pi!r}*y)' for w in (4,8,12,16))
DSGNAR_EXAMPLES={
 'dsgnar_wave':('Wave · 1D + time · two standing modes','Wave speed 2; initial displacement and zero initial velocity, fixed endpoints; analytical reference.',REFERENCE),
 'dsgnar_multiscale':('Multiscale Poisson · 2D · four spatial frequencies','Modes 4,8,12,16; zero boundary values; analytical reference.',MULTISCALE_REFERENCE),
 'dsgnar_poisson_1d':('Poisson · 1D · DSGNAR','Fixed points, double precision and sketched SVD steps.','x*(1-x)'),
 'dsgnar_cavity':('Lid-driven cavity · 2D · smooth lid · Re=100','Parabolic moving lid; stationary walls. Fixed samples, pressure gauge and grouped flow losses; variation of paper Appendix C.7.',''),
 'dsgnar_burgers':('Burgers · 1D + time · DSGNAR','Original Burgers physics with a small independent DSGNAR preset.',''),
}


def dsgnar_example_config(key):
    from .examples import example_config
    from .burgers import burgers_config
    if key not in DSGNAR_EXAMPLES:raise ValueError('Unknown DSGNAR example.')
    if key=='dsgnar_wave':
        from .poisson import PoissonConfig
        c=PoissonConfig(shape='SpaceTime1D',pde_preset='wave_1d',source=2.,sampling='random');c.bc_targets={'bottom':INITIAL}
    elif key=='dsgnar_multiscale':
        from .poisson import PoissonConfig
        c=PoissonConfig(pde_preset='poisson',source=0.,pde_forcing=MULTISCALE_FORCING,sampling='random')
    elif key=='dsgnar_cavity':
        from .incompressible_flow import flow_demo_config
        c=flow_demo_config();c.pde_preset='incompressible_2d';c.viscosity=.01;c.source=0.
        for side in c.flow_boundary:
            for name,spec in c.flow_boundary[side].items():spec['target']='4*x*(1-x)' if side=='top' and name=='u' else '0'
        c.network_type='siren';c.network_options={'first_omega':1.,'omega':1.}
    else:c=burgers_config() if key=='dsgnar_burgers' else example_config('interval_poisson')
    c.method='dsgnar';c.network_seed=c.seed;c.points=32;c.boundary_points=8;c.boundary_sampling={}
    c.layers=(8,8);c.iterations=50;c.pde_weight=.01;c.boundary_weight=1.
    if key in ('dsgnar_wave','dsgnar_multiscale'):
        c.points=256;c.boundary_points=32;c.layers=(16,16,16);c.iterations=200
        c.network_type='fourier';c.network_options={'frequencies':[.5,1.,2.,4.] if key=='dsgnar_wave' else [2.,4.,6.,8.]}
        c.dsgnar_settings.update(sketch_size=128,batch_size=8,probes=8)
        if key=='dsgnar_wave':
            c.boundary_weight=3.;c.points=512;c.boundary_points=64;c.network_type='siren'
            c.network_options={'first_omega':8.,'omega':1.}
            c.dsgnar_settings.update(sketch_size=700,reweight_exponent=0.,stage_grace=80)
    if key=='dsgnar_cavity':
        c.points=128;c.boundary_points=16;c.layers=(16,16,16);c.iterations=150
        c.dsgnar_settings.update(sketch_size=128,batch_size=8,probes=8)
    return c
