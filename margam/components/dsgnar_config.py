"""Paper-based DSGNAR controls, shared by scripts and desktop validation."""
import math


def dsgnar_defaults():
    return dict(sketch_size=32,hashes=4,batch_size=16,probes=8,initial_radius=1.,
                min_radius=1e-8,max_radius=1000.,target_ratio=.075,stage2_ratio=.5,
                reweight_exponent=.05,history_window=30,precision='float64',seed=7)


def validate_dsgnar_settings(s):
    if not isinstance(s,dict) or set(s)!=set(dsgnar_defaults()):raise ValueError('Incomplete DSGNAR settings.')
    for key,lo,hi in [('sketch_size',1,4096),('hashes',1,8),('batch_size',1,256),('probes',3,32),('history_window',3,1000),('seed',0,2**31-1)]:
        if type(s[key]) is not int or not lo<=s[key]<=hi:raise ValueError(f'DSGNAR {key} must be an integer between {lo} and {hi}.')
    for key in ('initial_radius','min_radius','max_radius','target_ratio','stage2_ratio','reweight_exponent'):
        if type(s[key]) not in (int,float) or not math.isfinite(s[key]):raise ValueError('DSGNAR controls must be finite numbers.')
    if not 0<s['min_radius']<s['initial_radius']<=s['max_radius']<=1e8:raise ValueError('DSGNAR requires 0 < minimum radius < initial radius <= maximum radius <= 1e8.')
    if not 0<s['target_ratio']<s['stage2_ratio']<1:raise ValueError('DSGNAR ratios must satisfy 0 < first stage < second stage < 1.')
    if not 0<=s['reweight_exponent']<=1:raise ValueError('DSGNAR reweighting exponent must be between zero and one.')
    if s['precision'] not in ('float32','float64'):raise ValueError('DSGNAR precision must be float32 or float64.')


def validate_dsgnar_problem(c):
    validate_dsgnar_settings(c.dsgnar_settings)
    if c.physics!='scalar' or c.is_paper_case or c.shape not in ('Interval','Rectangle','SpaceTime1D'):
        raise ValueError('DSGNAR currently supports built-in scalar interval, rectangle and 1D space-time problems.')
    if c.loss_formulation!='mean' or c.adaptive_sampling or c.interior_data or c.boundary_data:
        raise ValueError('DSGNAR currently requires mean loss and fixed built-in points; imported or adaptive samples are not supported.')
