"""Small meshless box samplers; no Qt, CAD kernel or tensor backend required."""
import itertools
import math
import random

SAMPLERS = {
    'grid': 'Cell-centred grid (complete Cartesian layers)',
    'random': 'Uniform random (seeded)',
    'hammersley': 'Randomly shifted Hammersley (exact count, seeded)',
    'stratified': 'Equal-volume recursive strata (exact count, seeded)',
}


def sample_box(lower, upper, count, method='grid', seed=7):
    """Return coordinate rows; grid may return fewer than the requested budget."""
    lower, upper = tuple(lower), tuple(upper)
    dim = len(lower)
    if not 1 <= dim <= 8 or dim != len(upper):
        raise ValueError('Sampling supports 1D to 8D box domains.')
    if not all(math.isfinite(v) for v in (*lower, *upper)) or any(b <= a for a, b in zip(lower, upper)):
        raise ValueError('Domain bounds must be finite and strictly increasing.')
    if type(count) is not int or not 1 <= count <= 100000:
        raise ValueError('Point budget must be between 1 and 100000.')
    if method not in SAMPLERS:
        raise ValueError('Choose a supported sampling method.')
    if method in ('hammersley','stratified'):
        unit = structured_unit_points(method,count,dim,seed)
    elif method == 'random':
        rng = random.Random(seed)
        unit = [[rng.random() for _ in lower] for _ in range(count)]
    else:
        if dim == 1:
            side = count
        elif dim == 2:
            side = math.isqrt(count)
        else:
            side = max(1, int(round(count ** (1.0 / dim))))
            while (side + 1) ** dim <= count:
                side += 1
            while side > 1 and side ** dim > count:
                side -= 1
        unit = itertools.product(((i + .5) / side for i in range(side)), repeat=dim)
    return [[a + t * (b-a) for a, b, t in zip(lower, upper, row)] for row in unit]


def structured_unit_points(method,count,dimension,seed):
    """Adapted from Jorvik c532cece sampling.py; AGPL-3.0-only.

    Retains the upstream Hammersley shift and exact-N recursive strata algorithms;
    extended to arbitrary box dimensions up to 8D. See THIRD_PARTY.md.
    """
    import numpy as np
    rng=np.random.default_rng(seed)
    if method=='hammersley':
        primes=(2,3,5,7,11,13,17)
        result=np.empty((count,dimension));result[:,0]=(np.arange(count)+.5)/count
        for axis in range(1,dimension):
            base=primes[axis-1]
            indices=np.arange(count,dtype=np.int64);values=np.zeros(count);factor=1./base
            while np.any(indices):
                values+=(indices%base)*factor;indices//=base;factor/=base
            result[:,axis]=values
        return (result+rng.random(dimension))%1
    leaves=[(np.zeros(dimension),np.ones(dimension),count)];output=[]
    while leaves:
        low,high,number=leaves.pop()
        if number==1:
            output.append(low+rng.random(dimension)*(high-low));continue
        axis=int(np.argmax(high-low));left=number//2
        middle=low[axis]+(high[axis]-low[axis])*left/number
        upper,lower=high.copy(),low.copy();upper[axis],lower[axis]=middle,middle
        leaves.extend([(lower,high.copy(),number-left),(low.copy(),upper,left)])
    return np.asarray(output)
