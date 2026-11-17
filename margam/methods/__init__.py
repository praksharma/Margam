"""One file per PINN method, assembled from shared components."""
from importlib import import_module
from typing import TYPE_CHECKING
from .pinn import PINN

_EXPORTS = {
    'HardDirichletPINN': '.hard_dirichlet',
    'SelfAdaptivePINN': '.self_adaptive',
    'LearningRateAnnealingPINN': '.learning_rate_annealing',
    'ExpertPINN': '.expert',
    'TimeMarchingPINN': '.time_marching',
    'ScalePINN': '.scale',
    'GradientEnhancedPINN': '.gpinn',
    'ResidualAttentionPINN': '.rba',
    'InverseBurgersPINN': '.inverse_burgers',
    'DSGNARPINN': '.dsgnar',
}

__all__ = (
    'PINN',
    'HardDirichletPINN',
    'SelfAdaptivePINN',
    'LearningRateAnnealingPINN',
    'ExpertPINN',
    'TimeMarchingPINN',
    'ScalePINN',
    'GradientEnhancedPINN',
    'ResidualAttentionPINN',
    'InverseBurgersPINN',
    'DSGNARPINN',
)

if TYPE_CHECKING:
    from .dsgnar import DSGNARPINN
    from .expert import ExpertPINN
    from .gpinn import GradientEnhancedPINN
    from .hard_dirichlet import HardDirichletPINN
    from .inverse_burgers import InverseBurgersPINN
    from .learning_rate_annealing import LearningRateAnnealingPINN
    from .rba import ResidualAttentionPINN
    from .scale import ScalePINN
    from .self_adaptive import SelfAdaptivePINN
    from .time_marching import TimeMarchingPINN


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted({*globals(), *__all__} - {'import_module', 'TYPE_CHECKING', '_EXPORTS'})
