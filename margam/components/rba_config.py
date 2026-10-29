"""Residual-based attention settings independent of network architecture."""
import math


def rba_defaults():return dict(decay=.999,rate=.001,initialization='zero')


def validate_rba_settings(settings):
    if not isinstance(settings,dict) or set(settings)!=set(rba_defaults()):raise ValueError('Incomplete RBA settings.')
    if type(settings['decay']) not in (int,float) or not math.isfinite(settings['decay']) or not 0<=settings['decay']<=.999999:raise ValueError('RBA decay must be between zero and 0.999999.')
    if type(settings['rate']) not in (int,float) or not math.isfinite(settings['rate']) or not 1e-8<=settings['rate']<=1000:raise ValueError('RBA attention rate must be between 1e-8 and 1000.')
    if settings['initialization'] not in ('zero','normalized'):raise ValueError('RBA initialization must be zero or normalized.')


def validate_rba_problem(c):
    validate_rba_settings(c.rba_settings)
    if c.field_names!=('u',) or c.physics!='scalar' and c.pde_preset!='expert_allen_cahn' or c.is_paper_case and c.pde_preset!='expert_allen_cahn' or c.pde_preset=='kdv_1d':raise ValueError('RBA currently supports scalar equations and the concrete Allen–Cahn example.')
    if c.loss_formulation!='mean' or c.adaptive_sampling:raise ValueError('RBA currently requires mean loss and fixed points; moving masks requires an explicit transfer rule.')
    if 'pde_weighted' in c.constraint_names:raise ValueError('pde_weighted is reserved for RBA diagnostics.')
