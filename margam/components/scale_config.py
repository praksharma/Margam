"""Sequential-correction controls, independent of networks and physics presets."""
import math


def scale_defaults():return dict(tau_sc=.45,tau_alpha=1.5,gamma=.0001,strength=1.,fresh=True)


def validate_scale_settings(settings):
    if not isinstance(settings,dict) or set(settings)!=set(scale_defaults()):raise ValueError('Incomplete Scale-PINN settings.')
    for key in ('tau_sc','tau_alpha','gamma','strength'):
        value=settings[key]
        if type(value) not in (int,float) or not math.isfinite(value) or value<0 or key.startswith('tau') and value==0:raise ValueError('Invalid Scale-PINN '+key+'.')
    if type(settings['fresh']) is not bool:raise ValueError('Scale fresh sampling must be a boolean.')


def validate_scale_problem(c):
    validate_scale_settings(c.scale_settings)
    if c.loss_formulation!='mean' or c.adaptive_sampling or c.interior_data or c.boundary_data:raise ValueError('Scale-PINN currently requires mean loss and built-in samples; adaptive/imported quadrature needs a separate contract.')
    if c.field_names not in (('u',),('u','v','p')):raise ValueError('Scale-PINN supports scalar equations or momentum-x/momentum-y/continuity with u/v/p; no auxiliary-vorticity mapping is defined.')
    if c.physics=='incompressible_2d':
        from ..templates.poisson import FLOW_RESIDUALS
        if tuple(c.flow_residuals)!=FLOW_RESIDUALS:raise ValueError('Scale-PINN requires the standard momentum-x/momentum-y/continuity ordering.')
    if c.scale_settings['fresh'] and (c.sampling!='random' or not c.is_paper_case and any(c.boundary_recipe(side).method!='random' for side in c.sides)):raise ValueError('Fresh Scale batches require random interior and boundary/initial sampling recipes.')
