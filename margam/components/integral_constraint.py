"""Explicit scalar quadrature constraint for the script-facing research trainer."""
from dataclasses import dataclass
from .problem import PointConstraint


@dataclass
class IntegralConstraint(PointConstraint):
    """Penalise (sum_i quadrature_weights[i] * operator(x_i, u_i) - target)^2.

    The operator is pointwise and scalar; omit it for a scalar network output.
    Weights are fixed nonnegative physical quadrature weights, not probabilities
    and not normalized. The target is one fixed scalar total. measure must be 1:
    physical measure is already represented by the quadrature weights.
    Supported by margam.research.Trainer; ordinary desktop methods reject it.
    """
    quadrature_weights: object = None
