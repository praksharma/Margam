"""Gradient-residual weights; selected coordinate terms are explicit and scalar."""
import math


def gpinn_defaults():return dict(x=.01,y=.01,t=.01)


def validate_gpinn_weights(weights):
    if not isinstance(weights,dict) or set(weights)!=set(gpinn_defaults()) or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1e6 for v in weights.values()):raise ValueError('gPINN needs finite x/y/t gradient weights between 0 and 1e6; zero disables a direction.')


def gpinn_terms(config):return tuple('pde_grad_'+name for name in config.coordinate_names if config.gpinn_weights[name]>0)


def validate_gpinn_problem(config):
    validate_gpinn_weights(config.gpinn_weights)
    if config.physics!='scalar' or config.is_paper_case or config.pde_preset=='kdv_1d':raise ValueError('gPINN currently supports scalar interval, rectangle, prepared 2D and 1D space–time equations.')
    if config.loss_formulation!='mean':raise ValueError('gPINN currently uses mean squared losses; integral gradient quadrature is not implemented.')
    if config.adaptive_sampling.get('method')=='importance':raise ValueError('gPINN supports fixed points or RAR; importance-weighted gradient losses remain deferred.')
    if set(gpinn_terms(config))&set(config.constraint_names):raise ValueError('Boundary names conflict with gPINN gradient-loss terms.')
