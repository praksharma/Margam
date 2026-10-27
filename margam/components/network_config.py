"""Concrete pointwise network choices/settings, deliberately independent of Torch/Qt."""
from copy import deepcopy
import math

NETWORK_NAMES = {
    'fcnn':'Fully connected', 'fourier_mlp':'Fourier MLP (angular)',
    'fourier':'Fourier feature NN', 'highway_fourier':'Highway Fourier NN',
    'modified_fourier':'Modified Fourier NN', 'multiscale_fourier':'Multiscale Fourier NN',
    'dgm':'DGM network', 'siren':'SIREN', 'fourier_filter':'Multiplicative Fourier NN',
    'gabor_filter':'Multiplicative Gabor NN',
}
NETWORK_FIELDS = ('layers','network_seed','network_type','network_options')


def choice(label,default,choices):return dict(label=label,kind='choice',default=default,choices=choices)
def number(label,default,minimum,maximum,kind='float'):return dict(label=label,kind=kind,default=default,minimum=minimum,maximum=maximum)
def values(label,default,minimum,maximum):return dict(label=label,kind='list',default=default,minimum=minimum,maximum=maximum)
def flag(label,default=False):return dict(label=label,kind='bool',default=default)

ACTIVATION=choice('Activation','tanh',('tanh','silu','sin'))
SKIP=flag('Skip connections')
FOURIER={
    'encoding':choice('Frequency encoding','axis',('axis','full','diagonal','gaussian')),
    'frequencies':values('Frequencies (cycles)',[0.,1.,2.,3.],-64.,64.),
    'features':number('Gaussian features',16,1,128,'int'),
    'scale':number('Frequency scale',1.,.001,64.),
    'activation':ACTIVATION,'skip_connections':SKIP,
}
OPTION_SCHEMAS = {
    'fcnn':{},
    'fourier_mlp':{'modes':number('Angular modes',4,1,32,'int'),'scale':number('Angular frequency scale',1.,.001,64.),'activation':ACTIVATION,'skip_connections':SKIP},
    'fourier':FOURIER,
    'highway_fourier':dict(FOURIER,transform_fourier=flag('Fourier transform gate',True),project_fourier=flag('Fourier carry projector',False)),
    'modified_fourier':FOURIER,
    'multiscale_fourier':{'scales':values('Gaussian scales',[1.,2.],.001,64.),'features':number('Features per scale',16,1,128,'int'),'activation':ACTIVATION,'skip_connections':SKIP},
    'dgm':{'activation':ACTIVATION},
    'siren':{'first_omega':number('First-layer omega',30.,.01,100.),'omega':number('Hidden-layer omega',30.,.01,100.)},
    'fourier_filter':{'input_scale':number('Filter input scale',1.,.001,64.)},
    'gabor_filter':{'input_scale':number('Filter input scale',1.,.001,64.),'alpha':number('Gabor gamma shape',6.,.01,32.),'beta':number('Gabor gamma rate',1.,.01,32.)},
}


def network_options(kind,options):
    if not isinstance(kind,str) or kind not in NETWORK_NAMES:raise ValueError('Choose a supported pointwise neural network.')
    schema=OPTION_SCHEMAS[kind]
    if not isinstance(options,dict) or not set(options)<=set(schema):raise ValueError('Unknown network option fields.')
    result={key:deepcopy(spec['default']) for key,spec in schema.items()};result.update(deepcopy(options))
    for key,value in result.items():
        spec=schema[key];option_kind=spec['kind']
        if option_kind=='bool':valid=type(value) is bool
        elif option_kind=='choice':valid=isinstance(value,str) and value in spec['choices']
        else:
            rows=value if option_kind=='list' else [value]
            valid=isinstance(rows,(list,tuple)) and 1<=len(rows)<=16 and all(type(v) in ((int,) if option_kind=='int' else (int,float)) and math.isfinite(v) and spec['minimum']<=v<=spec['maximum'] for v in rows)
        if not valid:raise ValueError('Invalid network setting: '+spec['label'])
    if kind=='multiscale_fourier' and len(result['scales'])>4:raise ValueError('Use at most four Gaussian scales.')
    return result


def validate_network(kind,options,layers):
    resolved=network_options(kind,options)
    if not isinstance(layers,(list,tuple)) or not 1<=len(layers)<=6 or any(type(n) is not int or not 1<=n<=256 for n in layers):raise ValueError('Use 1–6 hidden layers, with 1–256 neurons each.')
    if kind in ('highway_fourier','modified_fourier','dgm','fourier_filter','gabor_filter') or resolved.get('skip_connections'):
        if layers and len(set(layers))!=1:raise ValueError('This network requires equal hidden widths (for example 32, 32).')
    return resolved
