"""Manufactured Couette residual/BC checks and real coupled-field training."""
import unittest
import torch
from margam.templates.incompressible_flow import flow_demo_config,build_flow_pinn,make_flow_reference


class IncompressibleFlowTests(unittest.TestCase):
    def test_couette_residual_constraints_and_training(self):
        torch.set_num_threads(1);config=flow_demo_config();model=build_flow_pinn(config)
        x=torch.tensor([[.2,.3],[.7,.8]],requires_grad=True)
        fields=torch.cat((x[:,1:2],0*x[:,0:1],0*x[:,0:1]),dim=1)
        torch.testing.assert_close(model.problem.residual(x,fields),torch.zeros(2,3),rtol=0,atol=0)
        for condition in model.problem.constraints:
            points=condition.points.clone().requires_grad_(True)
            fields=torch.cat((points[:,1:2],0*points[:,0:1],0*points[:,0:1]),dim=1)
            actual=condition.operator(points,fields)
            target=condition.target(points) if callable(condition.target) else points.new_full((len(points),1),condition.target)
            torch.testing.assert_close(actual,target,rtol=0,atol=0)
        model.train(750,display_every=0)
        points=torch.tensor([[i/10,j/10] for j in range(11) for i in range(11)])
        reference=torch.tensor(make_flow_reference(config).evaluate(points.tolist()))
        error=(model.predict(points)-reference).square().mean(dim=0).sqrt()
        self.assertTrue((error<.03).all(),error.tolist())
        config.flow_residuals=('u','v','p')
        with self.assertRaisesRegex(ValueError,'preset reference'):make_flow_reference(config)

    def test_gauge_and_geometry_validation(self):
        config=flow_demo_config();config.pressure_point=(2,2)
        with self.assertRaisesRegex(ValueError,'gauge point'):config.validate()
        config=flow_demo_config();config.flow_boundary['left']['p']={'type':'Dirichlet','target':'1','operator':''}
        with self.assertRaisesRegex(ValueError,'gauge conflicts'):config.validate()
        config=flow_demo_config();config.shape='Interval'
        with self.assertRaisesRegex(ValueError,'rectangle'):config.validate()
