"""Independent implementation of Expert's Guide v1 exponential row factorization."""
import torch
from torch import nn
from torch.nn import functional as F


class FactorizedLinear(nn.Module):
    def __init__(self,layer,mean=1.,std=.1):
        super().__init__()
        # Preserve the supplied initialized effective weight exactly up to rounding.
        log_scale=torch.empty(layer.out_features,device=layer.weight.device,dtype=layer.weight.dtype).normal_(mean,std)
        self.log_scale=nn.Parameter(log_scale)
        self.direction=nn.Parameter(layer.weight.detach().clone()/log_scale.exp()[:,None])
        self.bias=nn.Parameter(layer.bias.detach().clone()) if layer.bias is not None else None
        self.in_features=layer.in_features;self.out_features=layer.out_features

    def forward(self,x):return F.linear(x,self.log_scale.exp()[:,None]*self.direction,self.bias)


def factorize_linear_layers(module,mean,std):
    for name,child in list(module.named_children()):
        if isinstance(child,nn.Linear):setattr(module,name,FactorizedLinear(child,mean,std))
        else:factorize_linear_layers(child,mean,std)
    return module
