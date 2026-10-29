"""Concrete scalar gPINN demonstrations, separate from its reusable recipe."""
GPINN_EXAMPLES={
    'gpinn_poisson_2d':('Poisson sine · 2D · gPINN','Manufactured sine variation; scalar residual gradients in x and y.','sin(pi*x)*sin(pi*y)'),
    'gpinn_burgers':('Burgers · 1D + time · gPINN','Scalar x/t residual gradients; soft initial/boundary conditions.',''),
    'gpinn_burgers_rar':('Burgers + RAR · 1D + time · gPINN','Residual-gradient losses with physical-residual refinement.',''),
}


def gpinn_example_config(key):
    from .poisson import PoissonConfig,BoundarySampling
    from .burgers import burgers_config
    if key not in GPINN_EXAMPLES:raise ValueError('Unknown gPINN example.')
    if key=='gpinn_poisson_2d':
        c=PoissonConfig(source=0.,pde_preset='poisson',pde_forcing='2*pi*pi*sin(pi*x)*sin(pi*y)',points=144,boundary_points=24,layers=(32,32),iterations=2000,lr=.001,boundary_weight=10.)
        c.gpinn_weights=dict(x=1e-5,y=1e-5,t=0.)
    else:
        c=burgers_config();c.points=256;c.gpinn_weights=dict(x=1e-4,y=0.,t=1e-4)
        if key.endswith('_rar'):
            from ..components.adaptive_sampling import sampling_defaults
            c.adaptive_sampling=sampling_defaults('rar');c.adaptive_sampling.update(interval=100,candidates=512,batch=16,maximum=4096)
    c.method='gpinn';return c
