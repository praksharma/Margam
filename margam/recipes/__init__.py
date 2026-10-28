"""One file per executable PINN recipe; physics examples live in templates."""
from . import baseline, self_adaptive, learning_rate_annealing, expert

# Concrete supported recipes; no dynamic imports or user-defined execution paths.
COMPACT_RECIPES = {'recipe':baseline, 'sa_recipe':self_adaptive,'lra_recipe':learning_rate_annealing,'expert_recipe':expert}
TRAINER_RECIPES = {'method':baseline, 'sa_method':self_adaptive,'lra_method':learning_rate_annealing,'expert_method':expert}
