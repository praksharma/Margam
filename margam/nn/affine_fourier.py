"""Independent trainable sinusoidal features after explicit affine input expansion.

Equations observed in SCALE-PINN notebooks; no upstream implementation copied.
The downstream trunk stays replaceable and does not reproduce their output heads.
"""
import math
import torch
from torch import nn
from .pointwise import PointwiseNetwork,SmoothTrunk,linear
from ..components.affine_inputs import affine_map


class AffineFourierNetwork(PointwiseNetwork):
    def __init__(self,inputs,outputs,layers,bounds,options,coordinates):
        super().__init__(inputs,bounds)
        matrix,offset=affine_map(options['input_transform'],coordinates)
        self.register_buffer('affine_matrix',torch.tensor(matrix,dtype=torch.get_default_dtype()).T.contiguous())
        self.register_buffer('affine_offset',torch.tensor(offset,dtype=torch.get_default_dtype()))
        self.normalize=options['normalize_inputs'];self.cycles=options['cycles']
        self.projection=nn.Linear(len(offset),options['features'])
        nn.init.kaiming_uniform_(self.projection.weight,a=0.,nonlinearity='relu');nn.init.zeros_(self.projection.bias)
        self.trunk=SmoothTrunk(options['features'],layers,options['activation'],options['skip_connections'])
        self.output=linear(layers[-1],outputs)

    def affine_inputs(self,x):
        if self.normalize:x=self.coordinates(x)
        return x@self.affine_matrix+self.affine_offset

    def features(self,x):return torch.sin(2*math.pi*self.cycles*self.projection(self.affine_inputs(x)))

    def forward(self,x):return self.output(self.trunk(self.features(x)))
