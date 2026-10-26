"""Shared components; numerical dependencies load only when requested."""
from importlib import import_module

_EXPORTS = {'PINNProblem': '.problem', 'PointConstraint': '.problem', 'weighted_loss': '.losses'}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
