"""Loss aggregation shared by recipes; individual terms stay inspectable."""
import math


def weighted_loss(terms, weights=None):
    """Sum scalar losses with finite, non-negative weights (default one)."""
    weights = {} if weights is None else weights
    unknown = set(weights) - set(terms)
    if unknown:
        raise ValueError(f"Unknown loss terms: {sorted(unknown)}")
    if not terms:
        raise ValueError("At least one loss term is required.")
    result = None
    for name, loss in terms.items():
        weight = float(weights.get(name, 1.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Invalid weight for {name}: {weight}")
        term = weight * loss
        result = term if result is None else result + term
    return result
