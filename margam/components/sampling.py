"""Small meshless box samplers; no Qt, CAD kernel or tensor backend required."""
import itertools
import math
import random

SAMPLERS = {
    'grid': 'Cell-centred grid (complete Cartesian layers)',
    'random': 'Uniform random (seeded)',
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
    if method == 'random':
        rng = random.Random(seed)
        unit = [[rng.random() for _ in lower] for _ in range(count)]
    else:
        side = math.isqrt(count) if len(lower) == 2 else count
        unit = itertools.product(((i + .5) / side for i in range(side)), repeat=len(lower))
    return [[a + t * (b-a) for a, b, t in zip(lower, upper, row)] for row in unit]
