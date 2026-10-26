"""Margam: PINN methods, shared components, and a native desktop builder."""
from importlib import import_module
from .__about__ import __version__

_EXPORTS = {"PINN": (".methods", "PINN"), "PINNProblem": (".components", "PINNProblem"),
            "PointConstraint": (".components", "PointConstraint")}

def __getattr__(name):
    if name in _EXPORTS:
        module, attribute = _EXPORTS[name]
        value = getattr(import_module(module, __name__), attribute)
    elif name in {"nn", "utils", "components", "methods", "benchmarking"}:
        value = import_module("." + name, __name__)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
