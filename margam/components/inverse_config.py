"""Concrete two-coefficient continuous-time Burgers identification contract."""
import math


def inverse_defaults():return dict(lambda_1=0.,log_lambda_2=-6.)


def validate_inverse_settings(value):
    if not isinstance(value,dict) or set(value)!={'lambda_1','log_lambda_2'} or any(type(v) not in (int,float) or not math.isfinite(v) for v in value.values()):raise ValueError('Inverse Burgers needs finite initial lambda_1 and log_lambda_2.')
    if not -100<=value['lambda_1']<=100 or not -20<=value['log_lambda_2']<=5:raise ValueError('Initial lambda_1 must be in [-100,100] and log_lambda_2 in [-20,5].')


def validate_inverse(config):
    validate_inverse_settings(config.inverse_settings)
    if config.shape!='SpaceTime1D' or config.physics!='scalar' or config.pde_preset or config.pde_expression:raise ValueError('Inverse Burgers uses its dedicated PDE block on 1D space–time (x,t).')
    if config.loss_formulation!='mean' or config.adaptive_sampling or config.boundary_data or config.bc_weights or config.pde_points!='interior':raise ValueError('Inverse Burgers uses fixed observation/PDE rows and mean losses; separate BCs and adaptive samplers are not composed.')
    if not config.interior_data or not config.interior_data.get('values') or config.interior_data.get('normals'):raise ValueError('Map observation coordinates and measured u values, without normals.')


def read_observations(config):
    from .prepared_data import read_points
    import numpy as np
    points,normals,values=read_points(config.interior_data,2)
    if values is None or normals is not None:raise ValueError('Observations require measured scalar u values, not normals.')
    lower,upper=map(np.asarray,config.bounds);tolerance=np.maximum((upper-lower)*1e-7,1e-12)
    if ((points<lower-tolerance)|(points>upper+tolerance)).any():raise ValueError('Observation coordinates lie outside the configured (x,t) bounds.')
    return points,values
