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
    if len(lower) not in (1, 2) or len(lower) != len(upper):
        raise ValueError('Sampling supports intervals and rectangles.')
    if not all(math.isfinite(v) for v in (*lower, *upper)) or any(b <= a for a, b in zip(lower, upper)):
        raise ValueError('Domain bounds must be finite and strictly increasing.')
    if type(count) is not int or not 1 <= count <= 100000:
        raise ValueError('Point budget must be between 1 and 100000.')
    if method not in SAMPLERS:
        raise ValueError('Choose a supported sampling method.')
    if method in ('hammersley','stratified'):
        unit = structured_unit_points(method,count,len(lower),seed)
    elif method == 'random':
        rng = random.Random(seed)
        unit = [[rng.random() for _ in lower] for _ in range(count)]
    else:
        side = math.isqrt(count) if len(lower) == 2 else count
        unit = itertools.product(((i + .5) / side for i in range(side)), repeat=len(lower))
    return [[a + t * (b-a) for a, b, t in zip(lower, upper, row)] for row in unit]


def structured_unit_points(method,count,dimension,seed):
    """Adapted from GeoSampler c532cece sampling.py; AGPL-3.0-only.

    Retains the upstream Hammersley shift and exact-N recursive strata algorithms;
    restricted here to the interval/rectangle contract. See THIRD_PARTY.md.
    """
    import numpy as np
    rng=np.random.default_rng(seed)
    if method=='hammersley':
        result=np.empty((count,dimension));result[:,0]=(np.arange(count)+.5)/count
        if dimension==2:
            indices=np.arange(count,dtype=np.int64);values=np.zeros(count);factor=.5
            while np.any(indices):
                values+=(indices%2)*factor;indices//=2;factor/=2
            result[:,1]=values
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
