# Fetch the version
from .__about__ import __version__

from . import backend
from .config import Config

from .geometry import spaces
from .geometry import domains
from .geometry import samplers

from . import nn

from . import utils
from . import constraint

from . import domain

from .model import Model

from .components import PINNProblem, PointConstraint
from .methods import PINN
