"""Fixed-point scalar examples for the residual-based attention trainer."""
RBA_EXAMPLES={
    'rba_allen_cahn':('Allen–Cahn · 1D + time · RBA','Fixed collocation points; periodic field/derivative and soft initial condition.',''),
    'rba_helmholtz_2d':('Helmholtz · 2D · RBA','Existing sine-forced Helmholtz physics with detached residual-history attention.','sin(pi*x)*sin(4*pi*y)'),
}


def rba_example_config(key):
    from .examples import example_config
    from .expert_examples import case_config
    if key not in RBA_EXAMPLES:raise ValueError('Unknown RBA example.')
    if key=='rba_allen_cahn':
        c=case_config('expert_allen_cahn');c.network_type='fcnn';c.network_options={};c.layers=(32,32,32);c.lr=.001
    else:c=example_config('helmholtz_lra_2d')
    c.method='rba';c.iterations=2000;c.points=256;c.boundary_points=32
    return c
