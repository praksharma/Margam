"""Modest correctness presets, not reproductions of the paper's accuracy."""
DSGNAR_EXAMPLES={
 'dsgnar_poisson_1d':('Poisson · 1D · DSGNAR','Fixed points, double precision and sketched SVD steps.','x*(1-x)'),
 'dsgnar_burgers':('Burgers · 1D + time · DSGNAR','Original Burgers physics with a small independent DSGNAR preset.',''),
}


def dsgnar_example_config(key):
    from .examples import example_config
    from .burgers import burgers_config
    if key not in DSGNAR_EXAMPLES:raise ValueError('Unknown DSGNAR example.')
    c=burgers_config() if key=='dsgnar_burgers' else example_config('interval_poisson')
    c.method='dsgnar';c.network_seed=c.seed;c.points=32;c.boundary_points=8;c.boundary_sampling={}
    c.layers=(8,8);c.iterations=50;c.pde_weight=.01;c.boundary_weight=1.
    return c
