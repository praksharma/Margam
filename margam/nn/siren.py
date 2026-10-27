# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Margam: tensor I/O, fixed-domain scaling and PINN derivatives.
# Source revisions and modifications: THIRD_PARTY.md; license: assets/LICENSE-PHYSICSNEMO.
"""Sine layers and NVIDIA SIREN first/hidden/linear-output initialisation."""
import math
import torch
from torch import nn
from .pointwise import PointwiseNetwork


class SirenNetwork(PointwiseNetwork):
    def __init__(self,inputs,outputs,layers,bounds,options):
        super().__init__(inputs,bounds);self.first_omega=options['first_omega'];self.omega=options['omega']
        sizes=[inputs,*layers,outputs];self.layers=nn.ModuleList(nn.Linear(a,b) for a,b in zip(sizes,sizes[1:]))
        for index,layer in enumerate(self.layers):
            if index==0:limit=1/layer.in_features
            elif index==len(self.layers)-1:limit=math.sqrt(6/layer.in_features)
            else:limit=math.sqrt(6/layer.in_features)/self.omega
            nn.init.uniform_(layer.weight,-limit,limit)
            nn.init.uniform_(layer.bias,-math.sqrt(1/layer.in_features),math.sqrt(1/layer.in_features))

    def forward(self,x):
        x=self.coordinates(x)
        for i,layer in enumerate(self.layers[:-1]):x=torch.sin((self.first_omega if i==0 else self.omega)*layer(x))
        return self.layers[-1](x)
