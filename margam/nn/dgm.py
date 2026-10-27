# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Margam: tensor I/O, fixed-domain scaling and PINN derivatives.
# Source revisions and modifications: THIRD_PARTY.md; license: assets/LICENSE-PHYSICSNEMO.
"""Deep Galerkin gated network architecture, independent of the PINN trainer."""
from torch import nn
from .pointwise import PointwiseNetwork,linear,activate


class DGMNetwork(PointwiseNetwork):
    def __init__(self,inputs,outputs,layers,bounds,options):
        super().__init__(inputs,bounds);self.activation=options['activation'];width=layers[0]
        self.start=linear(inputs,width)
        self.cells=nn.ModuleList(nn.ModuleDict({key:nn.ModuleList([linear(inputs,width),linear(width,width,bias=False)]) for key in ('z','g','r','h')}) for _ in layers[1:])
        self.output=linear(width,outputs)

    def forward(self,x):
        x=self.coordinates(x);state=activate(self.start(x),self.activation)
        for cell in self.cells:
            z,g,r=[activate(cell[key][0](x)+cell[key][1](state),self.activation) for key in ('z','g','r')]
            h=activate(cell['h'][0](x)+cell['h'][1](state*r),self.activation)
            state=(1-g)*h+z*state
        return self.output(state)
