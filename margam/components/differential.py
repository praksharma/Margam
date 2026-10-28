"""Pointwise autograd derivatives shared by the concrete PINN equations."""

def derivative(value,x,axis,order=1):
    import torch
    for _ in range(order):
        if not value.requires_grad:return x.new_zeros((len(x),1))
        gradient=torch.autograd.grad(value.sum(),x,create_graph=True,retain_graph=True,allow_unused=True)[0]
        value=x.new_zeros((len(x),1)) if gradient is None else gradient[:,axis:axis+1]
    return value
