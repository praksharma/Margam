"""One file per executable PINN recipe; physics examples live in templates."""
from . import baseline, self_adaptive

# Concrete supported recipes; no dynamic imports or user-defined execution paths.
COMPACT_RECIPES = {'recipe':baseline, 'sa_recipe':self_adaptive}
TRAINER_RECIPES = {'method':baseline, 'sa_method':self_adaptive}
