# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Margam: tensor I/O, fixed-domain scaling and PINN derivatives.
# Source revisions and modifications: THIRD_PARTY.md; license: assets/LICENSE-PHYSICSNEMO.
"""NVIDIA-style highway Fourier network with shared transform/carry projections."""
import torch
from .pointwise import PointwiseNetwork,SmoothTrunk,linear,activate
from .fourier import features_from_options


class HighwayFourierNetwork(PointwiseNetwork):
    def __init__(self,inputs,outputs,layers,bounds,options):
        super().__init__(inputs,bounds);self.options=options;self.features=features_from_options(inputs,options)
        size=inputs+self.features.output_size;width=layers[0]
        self.transform=linear(size if options['transform_fourier'] else inputs,width)
        self.carry=linear(size if options['project_fourier'] else inputs,width)
        self.trunk=SmoothTrunk(size,layers,options['activation'],False)
        self.output=linear(width,outputs)

    def forward(self,x):
        raw=self.coordinates(x);encoded=torch.cat((raw,self.features(raw)),dim=-1)
        gate=torch.sigmoid(self.transform(encoded if self.options['transform_fourier'] else raw))
        carry=self.carry(encoded if self.options['project_fourier'] else raw)
        x=encoded;previous=None
        for i,layer in enumerate(self.trunk.layers):
            x=activate(layer(x),self.options['activation'])*gate+carry*(1-gate)
            if self.options['skip_connections'] and i%2==0:
                if previous is None:previous=x
                else:x,previous=x+previous,x
        return self.output(x)
