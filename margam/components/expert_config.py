"""Settings for Margam's independently implemented Expert's Guide v1 recipe."""
import math


def expert_defaults():
    return dict(chunks=8,causal_epsilon=1.,balance_every=1000,momentum=.9,
                decay_rate=.9,decay_steps=2000,gradient_floor=1e-12,weight_cap=1e6)


def validate_expert_settings(settings):
    if not isinstance(settings,dict) or set(settings)!=set(expert_defaults()):raise ValueError('Incomplete Expert PINN settings.')
    for key,limit in (('chunks',128),('balance_every',100000),('decay_steps',100000)):
        if type(settings[key]) is not int or not 1<=settings[key]<=limit:raise ValueError('Invalid Expert PINN '+key+'.')
    for key,low,high in (('causal_epsilon',0.,100.),('momentum',0.,.9999),('decay_rate',.001,1.),('gradient_floor',1e-16,1.),('weight_cap',1.,1e12)):
        value=settings[key]
        if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:raise ValueError('Invalid Expert PINN '+key+'.')


def validate_expert_problem(config):
    validate_expert_settings(config.expert_settings)
    if not config.is_time_dependent or config.physics!='scalar':raise ValueError('The first Expert PINN workflow requires a scalar 1D space–time problem.')
    if config.loss_formulation!='mean':raise ValueError('Expert PINN currently follows the paper mean-squared-error formulation.')
    if config.interior_data or config.boundary_data or config.adaptive_sampling:raise ValueError('Expert PINN uses its own fresh uniform batches; imported data and residual-adaptive sampler blocks are not yet compatible.')
    if config.sampling!='random' or any(config.boundary_recipe(s).method!='random' for s in config.sides):raise ValueError('Expert PINN requires random interior and boundary/initial sampling recipes.')
    if config.points%config.expert_settings['chunks']:raise ValueError('Expert PINN interior count must be divisible by its causal chunk count.')
