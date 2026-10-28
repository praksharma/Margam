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


LOSS_FORMULATIONS = {'mean':'Mean squared error', 'integral':'Domain/boundary integral (Monte Carlo)'}


def apply_problem_measures(problem,config):
    """Concrete box measures; imported quadrature is deliberately not inferred."""
    lower,upper=config.bounds
    problem.measure=math.prod(b-a for a,b in zip(lower,upper))
    side_measures=dict.fromkeys(config.sides,1.) if len(lower)==1 else dict(left=upper[1]-lower[1],right=upper[1]-lower[1],bottom=upper[0]-lower[0],top=upper[0]-lower[0])
    if config.is_time_dependent:side_measures['initial']=upper[0]-lower[0]
    for condition in problem.constraints:
        side=condition.name.split(':',1)[0]
        condition.measure=side_measures.get(side,1.) # Pressure gauge is a point penalty.
    return problem
