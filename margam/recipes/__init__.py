"""One file per executable PINN recipe; physics examples live in templates."""
from . import dsgnar, baseline, self_adaptive, learning_rate_annealing, expert, scale, gradient_enhanced, residual_attention, inverse_burgers

# Concrete supported recipes; no dynamic imports or user-defined execution paths.
COMPACT_RECIPES = {'dsgnar_recipe':dsgnar,'inverse_recipe':inverse_burgers,'recipe':baseline, 'sa_recipe':self_adaptive,'lra_recipe':learning_rate_annealing,'expert_recipe':expert,'scale_recipe':scale,'gpinn_recipe':gradient_enhanced,'rba_recipe':residual_attention}
TRAINER_RECIPES = {'dsgnar_method':dsgnar,'inverse_method':inverse_burgers,'method':baseline, 'sa_method':self_adaptive,'lra_method':learning_rate_annealing,'expert_method':expert,'scale_method':scale,'gpinn_method':gradient_enhanced,'rba_method':residual_attention}
