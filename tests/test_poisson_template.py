"""Physical sampling, Neumann orientation and template convergence checks."""
import unittest
import torch
from margam.templates.poisson import PoissonConfig, sample_problem, build_pinn, interval_reference


class PoissonTemplateTests(unittest.TestCase):
    def test_rectangle_samples_and_budget(self):
        config=PoissonConfig(x_min=-2,x_max=3,y_min=4,y_max=7,points=70,boundary_points=9)
        interior, sides=sample_problem(config)
        self.assertEqual(len(interior),64)
        self.assertTrue(all(-2<x<3 and 4<y<7 for x,y in interior))
        self.assertEqual(set(sides),{'left','right','bottom','top'})
        self.assertTrue(all(x==-2 for x,y in sides['left']))
        self.assertTrue(all(y==7 for x,y in sides['top']))
        config.sampling='random'
        self.assertEqual(sample_problem(config),sample_problem(config))
        self.assertEqual(len(sample_problem(config)[0]),70)

    def test_mixed_neumann_reference_and_operator(self):
        config=PoissonConfig(shape='Interval')
        config.bc_types['left']='Neumann'
        config.bc_values['left']=1
        config.bc_values['right']=0
        model=build_pinn(config)
        condition=model.problem.constraints[0]
        x=torch.tensor([[0.]],requires_grad=True)
        u=x**2+3*x
        torch.testing.assert_close(condition.operator(x,u),torch.tensor([[-3.]]))
        self.assertAlmostEqual(interval_reference(config,[[0.],[1.]])[0],2.)
        self.assertAlmostEqual(interval_reference(config,[[0.],[1.]])[1],0.)

    def test_interval_matches_baseline_solution(self):
        torch.set_num_threads(1)
        config=PoissonConfig(shape='Interval',points=32)
        model=build_pinn(config)
        model.train(config.iterations,display_every=0)
        x=torch.linspace(0,1,101)[:,None]
        reference=torch.tensor(interval_reference(config,x.tolist()))[:,None]
        self.assertLess(((model.predict(x)-reference)**2).mean().sqrt().item(),.02)

    def test_boundary_points_exist_before_condition_assignment(self):
        config=PoissonConfig()
        config.bc_types={}
        interior,boundaries=sample_problem(config)
        self.assertEqual(len(interior),64)
        self.assertEqual(len(boundaries),4)
        with self.assertRaises(ValueError):build_pinn(config)

    def test_reject_undefined_problems(self):
        config=PoissonConfig(x_max=0)
        with self.assertRaises(ValueError):config.validate()
        config=PoissonConfig()
        config.bc_types=dict.fromkeys(config.sides,'Neumann')
        with self.assertRaisesRegex(ValueError,'Pure Neumann'):config.validate()
        config=PoissonConfig()
        config.bc_values['top']=2
        with self.assertRaisesRegex(ValueError,'corner'):config.validate()
