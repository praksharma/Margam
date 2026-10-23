"""Compatibility entry point for the original domain/network API."""
from .components import PINNProblem
from .methods import PINN


class Model(PINN):
    """Adapt a legacy domain to the baseline recipe, retaining compile/train."""
    def __init__(self, domain, network):
        self.domain = domain
        super().__init__(None, network)

    def compile(self, *args, **kwargs):
        self.problem = PINNProblem.from_domain(self.domain)
        return super().compile(*args, **kwargs)
