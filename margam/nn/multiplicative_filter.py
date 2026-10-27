# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Margam: tensor I/O, fixed-domain scaling and PINN derivatives.
# Source revisions and modifications: THIRD_PARTY.md; license: assets/LICENSE-PHYSICSNEMO.
"""Pointwise Fourier/Gabor filters; no FFT, grid or neural-operator dependency."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .pointwise import PointwiseNetwork,linear


class FourierFilter(nn.Module):
    def __init__(self,inputs,width,depth,options):
        super().__init__();self.scale=options['input_scale']/math.sqrt(depth+1)
        self.frequency=nn.Parameter(torch.empty(inputs,width));self.phase=nn.Parameter(torch.empty(width))
        nn.init.xavier_uniform_(self.frequency);nn.init.uniform_(self.phase,-math.pi,math.pi)

    def forward(self,x):return torch.sin(2*math.pi*(x@(self.scale*self.frequency))+self.phase)


class GaborFilter(FourierFilter):
    def __init__(self,inputs,width,depth,options):
        super().__init__(inputs,width,depth,options)
        self.center=nn.Parameter(torch.empty(inputs,width));nn.init.uniform_(self.center,-1,1)
        gamma=torch.distributions.Gamma(options['alpha'],options['beta']).sample((width,)).clamp_min(1e-6)
        self.raw_gamma=nn.Parameter(gamma+torch.log(-torch.expm1(-gamma)))

    def forward(self,x):
        gamma=F.softplus(self.raw_gamma)+torch.finfo(x.dtype).eps
        # Sum of squares is smooth at the centre; norm().square() is not safe
        # for second derivatives at exactly zero distance.
        distance=(x.unsqueeze(-1)-self.center).square().sum(dim=-2)
        phase=2*math.pi*(x@(self.scale*self.frequency*gamma.sqrt()))+self.phase
        return torch.exp(-.5*distance*gamma)*torch.sin(phase)


class MultiplicativeFilterNetwork(PointwiseNetwork):
    def __init__(self,inputs,outputs,layers,bounds,options,gabor=False):
        super().__init__(inputs,bounds);width=layers[0];depth=len(layers)
        cls=GaborFilter if gabor else FourierFilter
        self.filters=nn.ModuleList(cls(inputs,width,depth,options) for _ in range(depth+1))
        self.layers=nn.ModuleList(linear(width,width) for _ in range(depth))
        self.output=linear(width,outputs)

    def forward(self,x):
        x=self.coordinates(x);state=self.filters[0](x)
        for layer,filter in zip(self.layers,self.filters[1:]):state=layer(state)*filter(x)
        return self.output(state)
