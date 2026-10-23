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

    @classmethod
    def from_domain(cls, domain):
        """Adapt the original domain.Generic sampling interface."""
        points, labels = domain.sample_collocation_labels()
        boundary_points, boundary_labels = domain.sample_boundary_labels()
        if len(boundary_points) != len(boundary_labels):
            raise ValueError("Boundary points and labels must have equal lengths.")
        # Retain the legacy single BC term and its sample-count weighting.
        constraints = []
        if boundary_points:
            constraints.append(PointConstraint(
                "boundary", torch.cat(boundary_points), torch.cat(boundary_labels)
            ))
        return cls(domain.pde, points, constraints, labels)
