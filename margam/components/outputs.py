"""Explicit prediction and comparison requests, independent of Qt and PDE presets."""
from dataclasses import dataclass
from typing import Callable, Optional
import math


@dataclass(frozen=True)
class ReferenceSource:
    name: str
    evaluate: Callable


@dataclass(frozen=True)
class OutputRequest:
    identifier: str
    kind: str
    resolution: int = 101
    reference: Optional[ReferenceSource] = None


def evaluate_outputs(model, config, requests):
    """Evaluate only requested outputs. Training alone does not call predict()."""
    results = {}
    for request in requests:
        if request.kind == 'monitor':
            continue
        if request.kind not in ('solution', 'comparison', 'export', 'paraview'):
            raise ValueError('Unknown output request.')
        if request.kind == 'comparison' and request.reference is None:
            raise ValueError('Comparison requires an explicit reference source.')
        if type(request.resolution) is not int or not 2 <= request.resolution <= 201:
            raise ValueError('Evaluation resolution must be between 2 and 201.')
        lower, upper = config.bounds
        count = request.resolution
        points=evaluation_points(config,count)
        import torch
        import numpy as np
        prediction=model.predict(torch.tensor(points)).detach().cpu().numpy()
        fields=tuple(getattr(config,'field_names',('u',)))
        if prediction.ndim==1:prediction=prediction[:,None]
        if prediction.shape!=(len(points),len(fields)) or not np.isfinite(prediction).all():
            raise ValueError('Predictions must be finite and match the explicit output field contract.')
        values=prediction[:,0].tolist() if len(fields)==1 else prediction.tolist()
        data = {'points': points, 'values': values, 'size': count, 'config': config,
                'steps': model.iter, 'reference': None, 'rmse': None}
        if len(fields)>1:data['fields']=list(fields)
        if request.kind == 'comparison':
            reference=np.asarray(request.reference.evaluate(points),dtype=float)
            if reference.ndim==1:reference=reference[:,None]
            if reference.shape!=prediction.shape or not np.isfinite(reference).all():
                raise ValueError('Reference values must be finite and match evaluation coordinates and fields.')
            error=prediction-reference
            data['reference']=reference[:,0].tolist() if len(fields)==1 else reference.tolist()
            data['reference_name']=request.reference.name
            data['max_error']=float(np.max(np.abs(error)))
            data['rmse']=float(np.sqrt(np.mean(error**2)))
            if len(fields)>1:
                data['rmse_by_field']={name:float(np.sqrt(np.mean(error[:,i]**2))) for i,name in enumerate(fields)}
                data['max_error_by_field']={name:float(np.max(np.abs(error[:,i]))) for i,name in enumerate(fields)}
        results[request.identifier] = data
    return results


def evaluation_points(config,count):
    """Bounded regular evaluation grids; discard solid-cylinder locations."""
    import itertools
    lower,upper=config.bounds
    if count**len(lower)>100000:raise ValueError('Evaluation exceeds 100000 locations; reduce the per-axis resolution.')
    axes=[[a+i*(b-a)/(count-1) for i in range(count)] for a,b in zip(lower,upper)]
    points=[list(reversed(row)) for row in itertools.product(*reversed(axes))]
    if getattr(config,'has_obstacle',False):
        from ..templates.expert_examples import inside_fluid
        points=[point for point in points if inside_fluid(config,point)]
    return points
