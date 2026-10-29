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
    point_data: object = None
    reference_data: object = None


def evaluate_outputs(model, config, requests):
    """Evaluate only requested outputs. Training alone does not call predict()."""
    results = {}
    for request in requests:
        if request.kind == 'monitor':
            continue
        if request.kind not in ('solution', 'comparison', 'export', 'paraview'):
            raise ValueError('Unknown output request.')
        if request.kind == 'comparison' and request.reference is None and request.reference_data is None:
            raise ValueError('Comparison requires an explicit reference source.')
        if request.point_data is None and (type(request.resolution) is not int or not 2 <= request.resolution <= 201):
            raise ValueError('Evaluation resolution must be between 2 and 201.')
        lower, upper = config.bounds
        count = request.resolution
        if request.point_data is None:points=evaluation_points(config,count)
        else:
            from .evaluation_data import validate_points
            points=validate_points(request.point_data.points,config,request.point_data.point_ids).tolist()
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
        if request.point_data is not None:
            data.update(size=None,evaluation=dict(request.point_data.provenance))
            if request.point_data.point_ids is not None:data['point_ids']=list(request.point_data.point_ids)
        if len(fields)>1:data['fields']=list(fields)
        if request.kind == 'comparison' or request.reference is not None or request.reference_data is not None:
            if request.reference_data is not None:
                from .evaluation_data import align_reference
                reference=align_reference(request.reference_data,points,data.get('point_ids'),config);data['reference_provenance']=dict(request.reference_data.provenance)
            else:reference=np.asarray(request.reference.evaluate(points),dtype=float)
            if reference.ndim==1:reference=reference[:,None]
            if reference.shape!=prediction.shape or not np.isfinite(reference).all():
                raise ValueError('Reference values must be finite and match evaluation coordinates and fields.')
            error=prediction-reference
            data['reference']=reference[:,0].tolist() if len(fields)==1 else reference.tolist()
            data['reference_name']=request.reference.name if request.reference is not None else 'Supplied reference data'
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
    if config.shape=='Prepared2D':raise ValueError('Prepared geometry requires an Evaluation points connection; a bounding-box grid is not its domain.')
    lower,upper=config.bounds
    if count**len(lower)>100000:raise ValueError('Evaluation exceeds 100000 locations; reduce the per-axis resolution.')
    axes=[[a+i*(b-a)/(count-1) for i in range(count)] for a,b in zip(lower,upper)]
    points=[list(reversed(row)) for row in itertools.product(*reversed(axes))]
    if getattr(config,'has_obstacle',False):
        from ..templates.expert_examples import inside_fluid
        points=[point for point in points if inside_fluid(config,point)]
    return points
