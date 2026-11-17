"""Shared components; numerical dependencies load only when requested."""
from importlib import import_module
from typing import TYPE_CHECKING

_EXPORTS = {
    'PINNProblem': '.problem',
    'PointConstraint': '.problem',
    'IntegralConstraint': '.integral_constraint',
    'weighted_loss': '.losses',
    'sample_box': '.sampling',
    'derivative': '.differential',
    'error_norms': '.error_norms',
}

__all__ = tuple(_EXPORTS)

if TYPE_CHECKING:
    from .differential import derivative
    from .error_norms import error_norms
    from .integral_constraint import IntegralConstraint
    from .losses import weighted_loss
    from .problem import PINNProblem, PointConstraint
    from .sampling import sample_box


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted({*globals(), *__all__} - {'import_module', 'TYPE_CHECKING', '_EXPORTS'})
