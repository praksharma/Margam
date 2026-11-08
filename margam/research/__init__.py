"""Script-first research components; no desktop or block registration required."""
from .trainer import Trainer
from ..components.problem import PINNProblem, PointConstraint
from ..components.sampling import sample_box
from ..components.differential import derivative
from ..utils import laplacian
from ..components.losses import weighted_loss
from ..components.error_norms import error_norms

__all__ = ['Trainer', 'PINNProblem', 'PointConstraint', 'sample_box',
           'derivative', 'laplacian', 'weighted_loss', 'error_norms']
