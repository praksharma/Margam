"""Deterministic pointwise mean-MSE accumulation for the research Trainer only.

Opting in declares the network, residual and constraint operators independent
across rows. This does not implement stochastic minibatch optimization.
"""
import torch
from ..components.losses import weighted_loss


def validate_network(network):
    for module in network.modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
            raise ValueError('Pointwise microbatching cannot use batch-normalization or batch-dependent networks.')
        if (isinstance(module,torch.nn.modules.dropout._DropoutNd) and module.p>0) or isinstance(module,torch.nn.RReLU):
            raise ValueError('Pointwise microbatching requires deterministic networks; stochastic layers are unsupported.')


def accumulate(trainer,size):
    """Backpropagate each chunk's exact row fraction; return detached diagnostics.

    Each physical term retains its own denominator, including uneven final
    chunks and different PDE/constraint row counts. Optimizer/schedule work
    happens once, in Trainer, after every contribution has succeeded.
    """
    specifications=[('pde',trainer.collocation_point_sample,trainer.collocation_point_labels,trainer.problem.residual,True)]
    specifications.extend((condition.name,points,target,condition.operator,condition.operator is not None)
        for condition,points,target in trainer._constraints)
    terms={};differentiable=False
    for name,points,target,operator,needs_grad in specifications:
        count=len(points);aggregate=points.new_zeros(())
        if target.ndim>0 and len(target)!=count:
            raise ValueError('Pointwise targets must match the complete point-set row count before chunking.')
        for start in range(0,count,size):
            stop=min(start+size,count)
            x=points[start:stop].detach().requires_grad_(needs_grad)
            values=trainer.network(x)
            if operator is not None:values=operator(x,values)
            labels=target if target.ndim==0 else target[start:stop]
            loss=trainer._term(values,labels,len(x))
            if not torch.isfinite(loss).item():raise FloatingPointError('Non-finite research chunk loss; no optimizer update was taken.')
            fraction=len(x)/count
            contribution=loss*fraction*trainer.weights.get(name,1.)
            if not torch.isfinite(contribution).item():raise FloatingPointError('Non-finite research chunk objective; no optimizer update was taken.')
            if contribution.requires_grad:
                differentiable=True;contribution.backward()
            aggregate=aggregate+loss.detach()*fraction
            # Release this graph before constructing the next chunk.
            del values,loss,contribution,x
        terms[name]=aggregate
    total=weighted_loss(terms,trainer.weights)
    if not torch.isfinite(total).item():raise FloatingPointError('Non-finite research objective; no optimizer update was taken.')
    if not differentiable:raise ValueError('The objective must retain its parameter gradients.')
    return terms,total
