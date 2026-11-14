"""Independent Cole–Hopf quadrature for the original sine/zero-wall Burgers case.

Fixed benchmark: nu=0.01/pi, x in [-1,1], t in [0,1], u(x,0)=-sin(pi*x).
This is evaluation data, not a new PDE/trainer or an inferred reference.
"""
from functools import lru_cache
import math
import numpy as np

VISCOSITY=.01/math.pi


@lru_cache(maxsize=3)
def _rule(order):
    nodes,weights=np.polynomial.hermite.hermgauss(order)
    nodes.setflags(write=False);weights.setflags(write=False)
    return nodes,weights


def burgers_reference(points,*,order=128):
    """Return float64 u values in input row order, using bounded quadrature chunks.

    The heat-kernel Cole–Hopf integrals share their positive Gaussian weights.
    Their ratio is evaluated after a per-row exponential shift. Exact initial
    and wall limits are assigned from the specified benchmark conditions.
    """
    if type(order) is not int or order not in (64,128,256):raise ValueError('Choose 64, 128 or 256 Hermite nodes.')
    values=np.asarray(points)
    if values.dtype.kind not in 'fiu' or values.ndim!=2 or values.shape[1]!=2 or not 1<=len(values)<=100000:
        raise ValueError('Burgers reference needs 1–100000 real (x,t) rows.')
    values=values.astype(np.float64)
    if not np.isfinite(values).all() or (values[:,0]<-1).any() or (values[:,0]>1).any() or (values[:,1]<0).any() or (values[:,1]>1).any():
        raise ValueError('Burgers reference is restricted to x in [-1,1] and t in [0,1].')
    nodes,weights=_rule(order);result=np.empty((len(values),1),dtype=np.float64)
    for start in range(0,len(values),512):
        x,t=values[start:start+512].T
        shifted=x[:,None]-2*np.sqrt(VISCOSITY*t[:,None])*nodes
        phase=np.pi*shifted
        exponent=-np.cos(phase)/(2*np.pi*VISCOSITY)
        mass=weights*np.exp(exponent-exponent.max(axis=1,keepdims=True))
        result[start:start+len(x),0]=-(mass*np.sin(phase)).sum(axis=1)/mass.sum(axis=1)
    initial=values[:,1]==0;result[initial,0]=-np.sin(np.pi*values[initial,0])
    result[np.abs(values[:,0])==1,0]=0.
    if not np.isfinite(result).all():raise FloatingPointError('Non-finite Burgers reference quadrature.')
    return result
