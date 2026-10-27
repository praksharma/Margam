"""Strong-form PINN with exact equal-constant box Dirichlet values.

This method owns its output transform; the baseline full-batch recipe is reused.
It does not support arbitrary boundary functions or mixed boundary operators.
"""
import torch
from .pinn import PINN


class BoxDirichletNetwork(torch.nn.Module):
    def __init__(self,network,lower,upper,value):
        super().__init__();self.network=network
        self.register_buffer('lower',torch.as_tensor(lower,dtype=next(network.parameters()).dtype))
        self.register_buffer('span',torch.as_tensor(upper,dtype=self.lower.dtype)-self.lower)
        self.register_buffer('value',torch.as_tensor(value,dtype=self.lower.dtype))
        if len(lower) not in (1,2) or (self.span<=0).any():raise ValueError('Hard Dirichlet needs valid interval/rectangle bounds.')

    def forward(self,x):
        t=(x-self.lower)/self.span
        factor=(4*t*(1-t)).prod(dim=1,keepdim=True)
        return self.value+factor*self.network(x)


class HardDirichletPINN(PINN):
    def __init__(self,problem,network,*,bounds,value,weights=None):
        super().__init__(problem,BoxDirichletNetwork(network,*bounds,value),weights=weights)
