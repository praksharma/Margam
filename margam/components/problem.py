"""Strong-form physics and point constraints, independent of training methods."""
from dataclasses import dataclass, field
from typing import Callable, Sequence, Union
import torch

TensorSource = Union[torch.Tensor, Callable[[], torch.Tensor]]


@dataclass
class PointConstraint:
    """Named condition: points, target, and optional operator(x, prediction).

    Points are tensors or zero-argument samplers. Targets are tensors, scalars,
    or functions of points. An operator can express derivative conditions.
    """
    name: str
    points: TensorSource
    target: Union[torch.Tensor, float, Callable] = 0.0
    operator: Callable = None
    measure: float = 1.0  # Boundary length, or unit counting measure for a point.


@dataclass
class PINNProblem:
    """User-supplied strong-form residual(x, prediction) and sampling data.

    Residuals have one row per point and are compared with residual_target
    (zero by default). The baseline samples once per compile().
    """
    residual: Callable
    collocation_points: TensorSource
    constraints: Sequence[PointConstraint] = field(default_factory=tuple)
    residual_target: Union[torch.Tensor, float, Callable] = 0.0
    data_identity: str = ""  # Optional immutable identity for imported operator data.

    measure: float = 1.0  # Domain volume for integral-of-squared-residual loss.
