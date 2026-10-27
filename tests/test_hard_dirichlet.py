"""The optional method enforces supported Dirichlet values without exact-solution data."""
import unittest
import torch
from margam.templates.poisson import PoissonConfig,build_pinn,interval_reference


class HardDirichletTests(unittest.TestCase):
    def test_exact_boundary_and_interval_accuracy(self):
        torch.set_num_threads(1)
        config=PoissonConfig(shape='Interval',method='hard_dirichlet',iterations=500)
        model=build_pinn(config)
        torch.testing.assert_close(model.predict(torch.tensor([[0.],[1.]])),torch.zeros(2,1),rtol=0,atol=0)
        model.train(500,display_every=0)
        x=torch.linspace(0,1,41)[:,None]
        expected=torch.tensor(interval_reference(config,x.tolist()))[:,None]
        self.assertLess((model.predict(x)-expected).square().mean().sqrt().item(),.005)
        config=PoissonConfig(method='hard_dirichlet');config.bc_values=dict.fromkeys(config.sides,3.)
        model=build_pinn(config)
        points=torch.tensor([[0.,.2],[1.,.4],[.5,0.],[.7,1.]])
        torch.testing.assert_close(model.predict(points),torch.full((4,1),3.),rtol=0,atol=0)
        config.bc_types['left']='Neumann'
        with self.assertRaisesRegex(ValueError,'Hard Dirichlet'):config.validate()
