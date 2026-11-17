"""Margam: PINN methods, shared components, and a native desktop builder."""
from importlib import import_module
from typing import TYPE_CHECKING
from .__about__ import __version__

_EXPORTS = {
    "PINN": (".methods", "PINN"),
    "PINNProblem": (".components", "PINNProblem"),
    "PointConstraint": (".components", "PointConstraint"),
}
_SUBPACKAGES = ("nn", "utils", "components", "methods", "benchmarking", "research")

__all__ = ("__version__", *_EXPORTS, *_SUBPACKAGES)

if TYPE_CHECKING:
    from .components import PINNProblem, PointConstraint
    from .methods import PINN
    from . import benchmarking, components, methods, nn, research, utils


def __getattr__(name):
    if name in _EXPORTS:
        module, attribute = _EXPORTS[name]
        value = getattr(import_module(module, __name__), attribute)
    elif name in _SUBPACKAGES:
        value = import_module("." + name, __name__)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value


def __dir__():
    return sorted({*globals(), *__all__} - {"import_module", "TYPE_CHECKING", "_EXPORTS", "_SUBPACKAGES"})
