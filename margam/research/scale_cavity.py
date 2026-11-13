"""Independent paper-case SCALE cavity network and explicit coordinate derivatives.

Paper section 4.5.1/Table A1. No upstream implementation is copied. These five
Taylor channels keep parameter differentiation first-order, so the complete
case-specific loss can be compiled without AOTAutograd double backward.
"""
import math
import torch
from torch import nn


def linear_jet(jet,layer):
    value=jet@layer.weight.T
    if layer.bias is not None:
        value=torch.cat(((value[0]+layer.bias)[None],value[1:]),dim=0)
    return value


def activate_jet(jet,kind):
    z,dx,dy,dxx,dyy=jet.unbind(0)
    if kind=='sine':
        value=torch.sin(2*math.pi*z);first=2*math.pi*torch.cos(2*math.pi*z);second=-(2*math.pi)**2*value
    elif kind=='silu':
        sigmoid=torch.sigmoid(z);base=sigmoid*(1-sigmoid)
        value=z*sigmoid;first=sigmoid+z*base;second=2*base+z*base*(1-2*sigmoid)
    else:raise ValueError('Unknown jet activation.')
    return torch.stack((value,first*dx,first*dy,second*dx.square()+first*dxx,second*dy.square()+first*dyy))


class ScaleCavityNetwork(nn.Module):
    """Trainable affine Fourier features, shared SiLU trunk, three SiLU heads."""
    def __init__(self,width=32):
        super().__init__();self.width=width
        self.projection=nn.Linear(4,4*width)
        self.shared=nn.ModuleList((nn.Linear(4*width,width),nn.Linear(width,width)))
        self.heads=nn.ModuleList(nn.ModuleList([nn.Linear(width,width) for _ in range(3)]+[nn.Linear(width,1,bias=False)]) for _ in range(3))
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.kaiming_uniform_(layer.weight,nonlinearity='relu')
                if layer.bias is not None:nn.init.zeros_(layer.bias)

    def forward(self,points):
        x,y=points[:,0:1],points[:,1:2]
        value=torch.sin(2*math.pi*self.projection(torch.cat((x,y,x-1,y-1),1)))
        for layer in self.shared:value=torch.nn.functional.silu(layer(value))
        outputs=[]
        for head in self.heads:
            h=value
            for layer in head[:-1]:h=torch.nn.functional.silu(layer(h))
            outputs.append(head[-1](h))
        return torch.cat(outputs,1)

    def jets(self,points):
        x,y=points[:,0:1],points[:,1:2];value=torch.cat((x,y,x-1,y-1),1)
        dx=torch.zeros_like(value);dx[:,0]=1;dx[:,2]=1
        dy=torch.zeros_like(value);dy[:,1]=1;dy[:,3]=1
        h=activate_jet(linear_jet(torch.stack((value,dx,dy,torch.zeros_like(value),torch.zeros_like(value))),self.projection),'sine')
        for layer in self.shared:h=activate_jet(linear_jet(h,layer),'silu')
        outputs=[]
        for head in self.heads:
            branch=h
            for layer in head[:-1]:branch=activate_jet(linear_jet(branch,layer),'silu')
            outputs.append(linear_jet(branch,head[-1]))
        return torch.cat(outputs,dim=2)


def cavity_terms(network,previous,points,boundary,targets,*,reynolds,tau_sc,tau_alpha):
    """Paper sum-equation MSE; only boundary velocity labels enter the objective."""
    current=network.jets(points)
    with torch.no_grad():old=previous.jets(points)
    values,dx,dy,dxx,dyy=current.unbind(0);lap=dxx+dyy
    u,v=values[:,0],values[:,1];nu=1/reynolds
    residual=torch.stack((u*dx[:,0]+v*dy[:,0]+dx[:,2]-nu*lap[:,0],u*dx[:,1]+v*dy[:,1]+dy[:,2]-nu*lap[:,1],dx[:,0]+dy[:,1]),1)
    change=(values-old[0])/tau_sc
    velocity_change=change[:,:2]-nu/tau_alpha*(lap[:,:2]-old[3,:,:2]-old[4,:,:2])
    correction=torch.cat((velocity_change,change[:,2:3]),1)
    interior=(~boundary).to(values.dtype);wall=boundary.to(values.dtype)
    corrected=((residual+correction).square().sum(1)*interior).sum()/interior.sum()
    raw=(residual.square().sum(1)*interior).sum()/interior.sum()
    bc=((values[:,:2]-targets).square()*wall[:,None]).sum(0)/wall.sum()
    return corrected,bc[0],bc[1],raw,(correction.square().mean())
