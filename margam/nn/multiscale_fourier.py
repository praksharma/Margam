# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Margam: tensor I/O, fixed-domain scaling and PINN derivatives.
# Source revisions and modifications: THIRD_PARTY.md; license: assets/LICENSE-PHYSICSNEMO.
"""Multi-scale Gaussian Fourier embeddings, shared MLP, concatenated readout."""
import torch
from torch import nn
from .pointwise import PointwiseNetwork,SmoothTrunk,linear
from .fourier import FourierFeatures


class MultiscaleFourierNetwork(PointwiseNetwork):
    def __init__(self,inputs,outputs,layers,bounds,options):
        super().__init__(inputs,bounds)
        self.features=nn.ModuleList(FourierFeatures(inputs,'gaussian',features=options['features'],scale=scale) for scale in options['scales'])
        self.trunk=SmoothTrunk(inputs+2*options['features'],layers,options['activation'],options['skip_connections'])
        self.output=linear(layers[-1]*len(self.features),outputs)

    def forward(self,x):
        x=self.coordinates(x)
        branches=[self.trunk(torch.cat((x,features(x)),dim=-1)) for features in self.features]
        return self.output(torch.cat(branches,dim=-1))
