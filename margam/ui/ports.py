"""Concrete port contracts for the current template-backed builder.

These describe editor connections, not tensor shapes or a general graph executor.
"""
from dataclasses import dataclass
from ..recipes import baseline


@dataclass(frozen=True)
class PortSpec:
    name: str
    label: str
    data_type: str


INPUTS = {
    'lra_recipe':tuple(PortSpec(*spec) for spec in baseline.INPUTS),
    'lra_method':(PortSpec('pde','PDE residual','residual'),PortSpec('boundary','Boundary conditions','constraints'),PortSpec('network','Neural network','network')),
    'sa_recipe':tuple(PortSpec(*spec) for spec in baseline.INPUTS),
    'sa_method':(PortSpec('pde','PDE residual','residual'),PortSpec('boundary','Boundary conditions','constraints'),PortSpec('network','Neural network','network')),
    'recipe':tuple(PortSpec(*spec) for spec in baseline.INPUTS),
    'domain': (),
    'rar': (PortSpec('input', 'Geometry', 'geometry'),),
    'importance': (PortSpec('input', 'Geometry', 'geometry'),),
    'interior': (PortSpec('input', 'Geometry', 'geometry'),),
    'boundary': (PortSpec('input', 'Geometry', 'geometry'),),
    'predefined_pde': (PortSpec('input','Interior points','interior_points'),),
    'pde': (PortSpec('input', 'Interior points', 'interior_points'),),
    'bc': (PortSpec('input', 'Boundary point sets', 'boundary_points'),),
    'network': (),
    'method': (PortSpec('pde', 'PDE residual', 'residual'),
               PortSpec('boundary', 'Boundary conditions', 'constraints'),
               PortSpec('network', 'Neural network', 'network')),
    'paraview': (PortSpec('input', 'PINN run', 'pinn_run'),),
    'export': (PortSpec('input', 'PINN run', 'pinn_run'),),
    'solution': (PortSpec('input', 'PINN run', 'pinn_run'),),
    'monitor': (PortSpec('input', 'PINN run', 'pinn_run'),),
    'reference': (),
    'comparison': (PortSpec('model', 'PINN run', 'pinn_run'),
                   PortSpec('reference', 'Reference source', 'reference')),
}
OUTPUTS = {
    'lra_recipe':PortSpec(*baseline.OUTPUT),
    'lra_method':PortSpec(*baseline.OUTPUT),
    'sa_recipe':PortSpec(*baseline.OUTPUT),
    'sa_method':PortSpec(*baseline.OUTPUT),
    'recipe':PortSpec(*baseline.OUTPUT),
    'domain': PortSpec('output', 'Geometry', 'geometry'),
    'rar': PortSpec('output', 'Interior points', 'interior_points'),
    'importance': PortSpec('output', 'Interior points', 'interior_points'),
    'interior': PortSpec('output', 'Interior points', 'interior_points'),
    'boundary': PortSpec('output', 'Boundary point sets', 'boundary_points'),
    'predefined_pde': PortSpec('output','PDE residual','residual'),
    'pde': PortSpec('output', 'PDE residual', 'residual'),
    'bc': PortSpec('output', 'Boundary conditions', 'constraints'),
    'network': PortSpec('output', 'Neural network', 'network'),
    'method': PortSpec('output', 'PINN run', 'pinn_run'),
    'paraview': None,
    'export': None,
    'solution': None,
    'monitor': None,
    'reference': PortSpec('output', 'Reference source', 'reference'),
    'comparison': None,
}

# Expert uses the same concrete physics/network contracts as the baseline recipe.
INPUTS['expert_recipe']=INPUTS['recipe']
INPUTS['expert_method']=INPUTS['method']
OUTPUTS['expert_recipe']=OUTPUTS['recipe']
OUTPUTS['expert_method']=OUTPUTS['method']

INPUTS['scale_recipe']=INPUTS['recipe']
INPUTS['scale_method']=INPUTS['method']
OUTPUTS['scale_recipe']=OUTPUTS['recipe']
OUTPUTS['scale_method']=OUTPUTS['method']

# Optional supplied-point sources never connect to geometry/training inputs.
INPUTS['evaluation_points']=()
INPUTS['reference_data']=()
OUTPUTS['evaluation_points']=PortSpec('output','Evaluation points','evaluation_points')
OUTPUTS['reference_data']=PortSpec('output','Reference fields','reference')
OUTPUTS['comparison']=PortSpec('output','Compared fields','comparison_data')
for _kind in ('solution','comparison','export','paraview'):
    INPUTS[_kind]=INPUTS[_kind]+(PortSpec('points','Evaluation points (optional)','evaluation_points'),)
for _kind in ('export','paraview'):
    INPUTS[_kind]=INPUTS[_kind]+(PortSpec('comparison','Compared fields (optional)','comparison_data'),)

# Prepared training sources replace their corresponding built-in samplers.
INPUTS['interior_data']=INPUTS['interior']
INPUTS['boundary_data']=INPUTS['boundary']
OUTPUTS['interior_data']=OUTPUTS['interior']
OUTPUTS['boundary_data']=OUTPUTS['boundary']

INPUTS['gpinn_recipe']=INPUTS['recipe']
INPUTS['gpinn_method']=INPUTS['method']
OUTPUTS['gpinn_recipe']=OUTPUTS['recipe']
OUTPUTS['gpinn_method']=OUTPUTS['method']
