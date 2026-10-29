"""Concrete Scale examples; shared physics with independently chosen networks."""
from copy import deepcopy
import math
from .expert_examples import case_config
from ..recipes.scale import DEFAULTS

CAVITY_REYNOLDS=(400,3200,7500,10000,20000)
SCALE_EXAMPLES={
    'scale_allen_cahn':('Allen–Cahn · 1D + time · Scale-PINN','Sequential correction on the existing periodic Allen–Cahn problem.'),
    **{'scale_cavity_'+str(re):('Lid-driven cavity · 2D · Re='+str(re)+' · Scale-PINN','Independent steady cavity starting preset; ν=γ=1/Re. No accuracy reproduction claim.') for re in CAVITY_REYNOLDS},
    'scale_ks':('Kuramoto–Sivashinsky · 1D + time · Scale-PINN','Periodic fourth-order problem on [0,2π]×[0,0.4].'),
    'scale_kdv':('Korteweg–de Vries · 1D + time · Scale-PINN','Periodic dispersive problem; ν=0.022², cosine initial data.'),
}


def scale_example_config(key):
    from .poisson import PoissonConfig,BoundarySampling
    if key not in SCALE_EXAMPLES:raise ValueError('Unknown Scale example.')
    c=PoissonConfig() if key=='scale_kdv' else case_config('expert_cavity' if key.startswith('scale_cavity_') else 'expert_ks' if key=='scale_ks' else 'expert_allen_cahn')
    for name,value in DEFAULTS.items():setattr(c,name,tuple(value) if name=='layers' else deepcopy(value))
    c.iterations=2000;c.points=256;c.boundary_points=32;c.scale_settings['gamma']=c.viscosity
    if key.startswith('scale_cavity_'):
        reynolds=int(key.rsplit('_',1)[1]);c.viscosity=1/reynolds;c.scale_settings.update(gamma=c.viscosity,tau_sc=.06,tau_alpha=1.);c.boundary_weight=15.;c.pde_weight=3.;c.lr=.0005
        if reynolds in (7500,10000,20000):c.scale_settings.update(tau_sc=.095,tau_alpha=.5);c.boundary_weight=10.
    elif key=='scale_ks':
        c.y_max=.4;c.scale_settings.update(tau_sc=.2,tau_alpha=1.5,gamma=100/16**2);c.boundary_weight=1000.
    elif key=='scale_kdv':
        c.pde_preset='kdv_1d';c.physics='scalar';c.shape='SpaceTime1D';c.x_min=-1.;c.x_max=1.;c.y_min=0.;c.y_max=1.;c.source=1.;c.viscosity=.022**2;c.sampling='random'
        c.boundary_sampling={side:BoundarySampling(32,'random',i+1) for i,side in enumerate(c.sides)}
        c.scale_settings.update(tau_sc=.1,tau_alpha=20.,gamma=.022);c.boundary_weight=2000.
    else:c.boundary_weight=100.
    # Notebook input equations, translated from (t,x) into Margam's named (x,t) coordinates.
    if key.startswith('scale_cavity_'):transform='x; y; x-1; y-1';features=128;cycles=1.
    elif key=='scale_ks':transform=f't; x/(2*pi); t-{c.y_max:g}';features=32;cycles=2.
    elif key=='scale_kdv':transform='t; x; t-1';features=32;cycles=1.
    else:transform='t; x';features=32;cycles=1.
    c.network_options.update(input_transform=transform,features=features,cycles=cycles,normalize_inputs=False)
    return c
