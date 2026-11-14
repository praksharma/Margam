"""Explicit update-count schedules for the fixed-data research trainer."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StepDecay:
    """Multiply the initial rate by gamma every `every` completed updates.

    Update k (zero based) uses initial_lr * gamma ** (k // every).
    This is an update schedule, not an epoch/validation-metric scheduler.
    """
    every: int
    gamma: float = .5

    def __post_init__(self):
        if type(self.every) is not int or self.every < 1:
            raise ValueError('StepDecay every must be a positive integer.')
        if type(self.gamma) not in (int, float) or not math.isfinite(self.gamma) or not 0 < self.gamma <= 1:
            raise ValueError('StepDecay gamma must be finite and in (0, 1].')

    def settings(self):
        return dict(kind='step_decay',every=self.every,gamma=self.gamma)

    def rates(self, base_lrs, completed):
        if type(completed) is not int or completed < 0:
            raise ValueError('Completed updates must be a non-negative integer.')
        rates=[base * self.gamma ** (completed // self.every) for base in base_lrs]
        if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
            raise ValueError('Scheduled learning rate must remain finite and positive; decay underflowed.')
        return rates


@dataclass(frozen=True)
class CosineDecay:
    """Decay each initial group rate to a positive relative floor over `steps`.

    Update k uses base_lr * (minimum_ratio + (1-minimum_ratio) *
    ((1+cos(pi*min(k/steps, 1)))/2)**power). Rates stay at the floor
    after the horizon; training does not stop or restart automatically.
    """
    steps: int
    minimum_ratio: float = .01
    power: float = 1.

    def __post_init__(self):
        if type(self.steps) is not int or self.steps<1:
            raise ValueError('CosineDecay steps must be a positive integer.')
        if type(self.minimum_ratio) not in (int,float) or not math.isfinite(self.minimum_ratio) or not 0<self.minimum_ratio<=1:
            raise ValueError('CosineDecay minimum_ratio must be finite and in (0, 1].')
        if type(self.power) not in (int,float) or not math.isfinite(self.power) or self.power<=0:
            raise ValueError('CosineDecay power must be finite and positive.')

    def settings(self):
        return dict(kind='cosine_decay',steps=self.steps,minimum_ratio=self.minimum_ratio,power=self.power)

    def rates(self,base_lrs,completed):
        if type(completed) is not int or completed<0:
            raise ValueError('Completed updates must be a non-negative integer.')
        fraction=min(completed,self.steps)/self.steps
        ratio=self.minimum_ratio+(1-self.minimum_ratio)*(.5*(1+math.cos(math.pi*fraction)))**self.power
        rates=[base*ratio for base in base_lrs]
        if any(not math.isfinite(rate) or rate<=0 for rate in rates):
            raise ValueError('Scheduled learning rate must remain finite and positive; decay underflowed.')
        return rates
