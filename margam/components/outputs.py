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
        if request.kind not in ('solution', 'comparison'):
            raise ValueError('Unknown output request.')
        if request.kind == 'comparison' and request.reference is None:
            raise ValueError('Comparison requires an explicit reference source.')
        if type(request.resolution) is not int or not 2 <= request.resolution <= 201:
            raise ValueError('Evaluation resolution must be between 2 and 201.')
        lower, upper = config.bounds
        count = request.resolution
        xs = [lower[0] + i*(upper[0]-lower[0])/(count-1) for i in range(count)]
        if len(lower) == 1:
            points = [[x] for x in xs]
        else:
            ys = [lower[1] + i*(upper[1]-lower[1])/(count-1) for i in range(count)]
            points = [[x, y] for y in ys for x in xs]
        import torch
        values = model.predict(torch.tensor(points)).detach().cpu().reshape(-1).tolist()
        if len(values) != len(points) or not all(math.isfinite(v) for v in values):
            raise ValueError('The current output view requires finite scalar predictions.')
        data = {'points': points, 'values': values, 'size': count, 'config': config,
                'steps': model.iter, 'reference': None, 'rmse': None}
        if request.kind == 'comparison':
            reference = request.reference.evaluate(points)
            if reference is None or len(reference) != len(values) or not all(math.isfinite(v) for v in reference):
                raise ValueError('Reference values must be finite and match the evaluation points.')
            data['reference'] = list(reference)
            data['reference_name'] = request.reference.name
            data['rmse'] = math.sqrt(sum((a-b)**2 for a,b in zip(values, reference))/len(values))
        results[request.identifier] = data
    return results
