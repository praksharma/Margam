# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Margam: tensor I/O, fixed-domain scaling and PINN derivatives.
# Source revisions and modifications: THIRD_PARTY.md; license: assets/LICENSE-PHYSICSNEMO.
"""Shared smooth pointwise layers for the adapted PINN architectures."""
import torch
from torch import nn


class PointwiseNetwork(nn.Module):
    def __init__(self,input_size,bounds):
        super().__init__()
        lower,upper=(torch.as_tensor(row,dtype=torch.get_default_dtype()) for row in bounds)
        if lower.shape!=(input_size,) or upper.shape!=lower.shape or not torch.isfinite(lower).all() or not torch.isfinite(upper).all() or (upper<=lower).any():
            raise ValueError('Network bounds must be finite and match the input dimension.')
        self.register_buffer('input_lower',lower);self.register_buffer('input_upper',upper)

    def coordinates(self,x):return (x-self.input_lower)/(self.input_upper-self.input_lower)


def activate(x,name):
    if name=='tanh':return torch.tanh(x)
    if name=='silu':return torch.nn.functional.silu(x)
    if name=='sin':return torch.sin(x)
    raise ValueError('Use a supported smooth activation.')


def linear(inputs,outputs,bias=True):
    layer=nn.Linear(inputs,outputs,bias=bias)
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:nn.init.zeros_(layer.bias)
    return layer


class SmoothTrunk(nn.Module):
    def __init__(self,inputs,layers,activation='tanh',skip_connections=False):
        super().__init__();self.activation=activation;self.skip=skip_connections
        sizes=[inputs,*layers];self.layers=nn.ModuleList(linear(a,b) for a,b in zip(sizes,sizes[1:]))

    def forward(self,x):
        previous=None
        for i,layer in enumerate(self.layers):
            x=activate(layer(x),self.activation)
            if self.skip and i%2==0:
                if previous is None:previous=x
                else:x,previous=x+previous,x
        return x
