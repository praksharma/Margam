# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Margam: tensor I/O, fixed-domain scaling and PINN derivatives.
# Source revisions and modifications: THIRD_PARTY.md; license: assets/LICENSE-PHYSICSNEMO.
"""Fixed Fourier features and angular Fourier MLP, not Fourier operators."""
import itertools
import math
import torch
from torch import nn
from .pointwise import PointwiseNetwork,SmoothTrunk,linear


class FourierFeatures(nn.Module):
    def __init__(self,dimension,encoding='axis',frequencies=(0.,1.,2.,3.),features=16,scale=1.):
        super().__init__();frequencies=torch.as_tensor(frequencies,dtype=torch.get_default_dtype())
        if encoding=='gaussian':matrix=torch.randn(dimension,features)*scale
        elif encoding=='axis':matrix=torch.stack([torch.eye(dimension)*frequency for frequency in frequencies]).reshape(-1,dimension).T*scale
        elif encoding=='full':matrix=torch.tensor(list(itertools.product(frequencies.tolist(),repeat=dimension)),dtype=torch.get_default_dtype()).T*scale
        elif encoding=='diagonal':matrix=frequencies[None,:].repeat(dimension,1).repeat_interleave(dimension,dim=1)*scale
        else:raise ValueError('Unknown Fourier encoding.')
        self.register_buffer('frequencies',matrix.contiguous());self.output_size=2*matrix.shape[1]

    def forward(self,x):
        phase=2*math.pi*(x@self.frequencies)
        return torch.cat((phase.sin(),phase.cos()),dim=-1)


class AngularFeatures(nn.Module):
    def __init__(self,dimension,modes=4,scale=1.):
        super().__init__();self.output_size=2*dimension*modes
        self.register_buffer('frequencies',torch.exp(torch.linspace(0,math.pi,modes))*scale)

    def forward(self,x):
        phase=(x.unsqueeze(-2)*self.frequencies[:,None]).flatten(-2)
        return torch.cat((phase.sin(),phase.cos()),dim=-1)


def features_from_options(dimension,options):
    return FourierFeatures(dimension,**{key:options[key] for key in ('encoding','frequencies','features','scale')})


class FourierNetwork(PointwiseNetwork):
    def __init__(self,inputs,outputs,layers,bounds,options,angular=False):
        super().__init__(inputs,bounds)
        self.features=AngularFeatures(inputs,options['modes'],options['scale']) if angular else features_from_options(inputs,options)
        self.trunk=SmoothTrunk(inputs+self.features.output_size,layers,options['activation'],options['skip_connections'])
        self.output=linear(layers[-1],outputs)

    def forward(self,x):
        x=self.coordinates(x)
        return self.output(self.trunk(torch.cat((x,self.features(x)),dim=-1)))
