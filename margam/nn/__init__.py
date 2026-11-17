"""Neural network architectures and pointwise builders."""
from importlib import import_module
from typing import TYPE_CHECKING
from .FCNN import BaseNetwork, FullyConnected
from .utils import activation, initialiser

_EXPORTS = {
    'build_network': '.factory',
    'FourierNetwork': '.fourier',
    'AffineFourierNetwork': '.affine_fourier',
    'HighwayFourierNetwork': '.highway',
    'ModifiedFourierNetwork': '.modified_fourier',
    'MultiscaleFourierNetwork': '.multiscale_fourier',
    'DGMNetwork': '.dgm',
    'SirenNetwork': '.siren',
    'MultiplicativeFilterNetwork': '.multiplicative_filter',
}

__all__ = (
    'BaseNetwork',
    'FullyConnected',
    'activation',
    'initialiser',
    'build_network',
    'FourierNetwork',
    'AffineFourierNetwork',
    'HighwayFourierNetwork',
    'ModifiedFourierNetwork',
    'MultiscaleFourierNetwork',
    'DGMNetwork',
    'SirenNetwork',
    'MultiplicativeFilterNetwork',
)

if TYPE_CHECKING:
    from .affine_fourier import AffineFourierNetwork
    from .dgm import DGMNetwork
    from .factory import build_network
    from .fourier import FourierNetwork
    from .highway import HighwayFourierNetwork
    from .modified_fourier import ModifiedFourierNetwork
    from .multiscale_fourier import MultiscaleFourierNetwork
    from .multiplicative_filter import MultiplicativeFilterNetwork
    from .siren import SirenNetwork


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted({*globals(), *__all__} - {'import_module', 'TYPE_CHECKING', '_EXPORTS'})
