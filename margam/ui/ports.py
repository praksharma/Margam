"""Concrete port contracts for the current template-backed builder.

These describe editor connections, not tensor shapes or a general graph executor.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PortSpec:
    name: str
    label: str
    data_type: str


INPUTS = {
    'domain': (),
    'interior': (PortSpec('input', 'Geometry', 'geometry'),),
    'boundary': (PortSpec('input', 'Geometry', 'geometry'),),
    'pde': (PortSpec('input', 'Interior points', 'interior_points'),),
    'bc': (PortSpec('input', 'Boundary point sets', 'boundary_points'),),
    'network': (),
    'method': (PortSpec('pde', 'PDE residual', 'residual'),
               PortSpec('boundary', 'Boundary conditions', 'constraints'),
               PortSpec('network', 'Neural network', 'network')),
    'solution': (PortSpec('input', 'PINN run', 'pinn_run'),),
    'monitor': (PortSpec('input', 'PINN run', 'pinn_run'),),
    'reference': (),
    'comparison': (PortSpec('model', 'PINN run', 'pinn_run'),
                   PortSpec('reference', 'Reference source', 'reference')),
}
OUTPUTS = {
    'domain': PortSpec('output', 'Geometry', 'geometry'),
    'interior': PortSpec('output', 'Interior points', 'interior_points'),
    'boundary': PortSpec('output', 'Boundary point sets', 'boundary_points'),
    'pde': PortSpec('output', 'PDE residual', 'residual'),
    'bc': PortSpec('output', 'Boundary conditions', 'constraints'),
    'network': PortSpec('output', 'Neural network', 'network'),
    'method': PortSpec('output', 'PINN run', 'pinn_run'),
    'solution': None,
    'monitor': None,
    'reference': PortSpec('output', 'Reference source', 'reference'),
    'comparison': None,
}
