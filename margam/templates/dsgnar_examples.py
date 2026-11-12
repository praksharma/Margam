"""Modest correctness presets, not reproductions of the paper's accuracy."""
DSGNAR_EXAMPLES={
 'dsgnar_poisson_1d':('Poisson · 1D · DSGNAR','Fixed points, double precision and sketched SVD steps.','x*(1-x)'),
 'dsgnar_cavity':('Lid-driven cavity · 2D · smooth lid · Re=100','Parabolic moving lid; stationary walls. Fixed samples, pressure gauge and grouped flow losses; variation of paper Appendix C.7.',''),
 'dsgnar_burgers':('Burgers · 1D + time · DSGNAR','Original Burgers physics with a small independent DSGNAR preset.',''),
}


def dsgnar_example_config(key):
    from .examples import example_config
    from .burgers import burgers_config
    if key not in DSGNAR_EXAMPLES:raise ValueError('Unknown DSGNAR example.')
    if key=='dsgnar_cavity':
        from .incompressible_flow import flow_demo_config
        c=flow_demo_config();c.pde_preset='incompressible_2d';c.viscosity=.01;c.source=0.
        for side in c.flow_boundary:
            for name,spec in c.flow_boundary[side].items():spec['target']='4*x*(1-x)' if side=='top' and name=='u' else '0'
        c.network_type='siren';c.network_options={'first_omega':1.,'omega':1.}
    else:c=burgers_config() if key=='dsgnar_burgers' else example_config('interval_poisson')
    c.method='dsgnar';c.network_seed=c.seed;c.points=32;c.boundary_points=8;c.boundary_sampling={}
    c.layers=(8,8);c.iterations=50;c.pde_weight=.01;c.boundary_weight=1.
    if key=='dsgnar_cavity':
        c.points=128;c.boundary_points=16;c.layers=(16,16,16);c.iterations=150
        c.dsgnar_settings.update(sketch_size=128,batch_size=8,probes=8)
    return c
