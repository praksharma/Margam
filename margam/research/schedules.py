"""Explicit update-count decay for the fixed-data research trainer."""
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

    def rates(self, base_lrs, completed):
        if type(completed) is not int or completed < 0:
            raise ValueError('Completed updates must be a non-negative integer.')
        rates=[base * self.gamma ** (completed // self.every) for base in base_lrs]
        if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
            raise ValueError('Scheduled learning rate must remain finite and positive; decay underflowed.')
        return rates
