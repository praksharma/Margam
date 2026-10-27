# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Margam: tensor I/O, fixed-domain scaling and PINN derivatives.
# Source revisions and modifications: THIRD_PARTY.md; license: assets/LICENSE-PHYSICSNEMO.
"""Modified Fourier features with two encoders and multiplicative hidden mixing."""
import torch
from .pointwise import PointwiseNetwork,SmoothTrunk,linear,activate
from .fourier import features_from_options


class ModifiedFourierNetwork(PointwiseNetwork):
    def __init__(self,inputs,outputs,layers,bounds,options):
        super().__init__(inputs,bounds);self.options=options;self.features=features_from_options(inputs,options)
        size=inputs+self.features.output_size;width=layers[0]
        self.encoder_u=linear(size,width);self.encoder_v=linear(size,width)
        self.trunk=SmoothTrunk(size,layers,options['activation'],False)
        self.output=linear(width,outputs)

    def forward(self,x):
        x=self.coordinates(x);x=torch.cat((x,self.features(x)),dim=-1)
        u=activate(self.encoder_u(x),self.options['activation']);v=activate(self.encoder_v(x),self.options['activation'])
        x=activate(self.trunk.layers[0](x),self.options['activation']);previous=None
        for i,layer in enumerate(self.trunk.layers[1:],1):
            z=activate(layer(x),self.options['activation']);x=u+(v-u)*z
            if self.options['skip_connections'] and i%2==0:
                if previous is None:previous=x
                else:x,previous=x+previous,x
        return self.output(x)
