"""Script-first research components; no desktop or block registration required."""
from ..methods.dsgnar import DSGNARPINN
from ..methods.fbpinn import FiniteBasisNetwork, FiniteBasisPINN
from .trainer import Trainer
from .schedules import StepDecay, CosineDecay
from . import jaxpi2
from .jaxpi2 import (
    ArchConfig,
    Jaxpi2CropTrainer,
    Jaxpi2MultiStageCropTrainer,
    Jaxpi2Optimizer,
    Jaxpi2SphereTrainer,
    Jaxpi2TrainingConfig,
    ModifiedMlp,
    MultiStageSolution,
    OptimConfig,
    PirateBlock,
    PirateNet,
    create_arch,
    jaxpi2_causal_residual_losses,
    jaxpi2_grad_norm_weights,
    linearized_navier_stokes_operator,
    multi_stage_navier_stokes,
    navier_stokes,
    pseudo_time_residuals,
    stable_swish,
    vorticity_3d,
)
from ..components.integral_constraint import IntegralConstraint
from ..components.problem import PINNProblem, PointConstraint
from ..components.sampling import sample_box
from ..components.differential import derivative
from ..utils import laplacian
from ..components.losses import weighted_loss
from ..components.error_norms import error_norms

__all__ = [
    'DSGNARPINN',
    'FiniteBasisNetwork',
    'FiniteBasisPINN',
    'Trainer',
    'StepDecay',
    'CosineDecay',
    'IntegralConstraint',
    'PINNProblem',
    'PointConstraint',
    'sample_box',
    'derivative',
    'laplacian',
    'weighted_loss',
    'error_norms',
    'jaxpi2',
    'ArchConfig',
    'PirateBlock',
    'PirateNet',
    'ModifiedMlp',
    'create_arch',
    'stable_swish',
    'Jaxpi2Optimizer',
    'OptimConfig',
    'Jaxpi2TrainingConfig',
    'Jaxpi2SphereTrainer',
    'Jaxpi2CropTrainer',
    'Jaxpi2MultiStageCropTrainer',
    'MultiStageSolution',
    'navier_stokes',
    'vorticity_3d',
    'jaxpi2_causal_residual_losses',
    'jaxpi2_grad_norm_weights',
    'pseudo_time_residuals',
    'linearized_navier_stokes_operator',
    'multi_stage_navier_stokes',
]
