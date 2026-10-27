"""Pointwise squared-mask min/max equations, restart fidelity and convergence."""
from copy import deepcopy
import unittest
import torch
from margam.components import PINNProblem,PointConstraint
from margam.methods.self_adaptive import SelfAdaptivePINN
from margam.templates.poisson import PoissonConfig,build_pinn


class SelfAdaptiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_pointwise_objective_and_opposite_update_directions(self):
        network=torch.nn.Linear(1,1,bias=False)
        with torch.no_grad():network.weight.fill_(1.)
        model=SelfAdaptivePINN(PINNProblem(lambda x,u:u,torch.tensor([[1.],[2.]])),network).compile(lr=.1,adaptive_lr=.01)
        errors=model.point_errors()
        torch.testing.assert_close(errors['pde'],torch.tensor([[1.],[4.]]))
        self.assertEqual(model.objective(errors).item(),2.5)
        model.train(1,display_every=0)
        self.assertLess(network.weight.item(),1.)
        self.assertTrue((model.adaptive_weights['pde']>1.).all())
        torch.testing.assert_close(model.adaptive_weights['pde'].grad,torch.tensor([[1.],[4.]]))
        self.assertEqual(model.loss_history[0],{'pde':2.5,'total':2.5})
        self.assertEqual(model.iter,1)

    def test_constraints_targets_vector_rows_and_cap(self):
        network=torch.nn.Linear(1,2)
        points=torch.tensor([[.2],[.7]])
        problem=PINNProblem(lambda x,u:torch.cat((u[:,0:1],2*u[:,1:2]),1),points,[PointConstraint('value',points,torch.zeros(2,2))])
        model=SelfAdaptivePINN(problem,network).compile(adapt_boundary=True,max_weight=1.)
        self.assertEqual(set(model.adaptive_weights),{'pde','value'})
        self.assertEqual(tuple(model.point_errors()['pde'].shape),(2,1))
        model.train(3,display_every=0)
        self.assertTrue(all((w==1.).all() for w in model.adaptive_weights.values()))
        # The real u/v/p builder must dispatch to SA rather than silently using baseline.
        flow=build_pinn(PoissonConfig(physics='incompressible_2d',method='self_adaptive',points=16,boundary_points=4,layers=(8,8),sa_boundary=True))
        self.assertIsInstance(flow,SelfAdaptivePINN)
        self.assertIn('pressure_gauge',flow.adaptive_weights)
        flow.train(5,display_every=0)
        self.assertTrue(torch.isfinite(flow.predict(torch.tensor([[.2,.3]]))).all())
        problem=PINNProblem(lambda x,u:u,points,residual_target=torch.ones(2,1))
        scalar=torch.nn.Linear(1,1)
        model=SelfAdaptivePINN(problem,scalar).compile()
        torch.testing.assert_close(model.point_errors()['pde'],(scalar(points)-1.).square())
        for kwargs in ({'initial_weight':0.},{'initial_weight':2.,'max_weight':1.},{'adaptive_lr':float('nan')},{'adapt_boundary':1}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):SelfAdaptivePINN(problem,scalar).compile(**kwargs)

    def test_resume_exact_and_reject_changed_data_or_settings(self):
        config=PoissonConfig(shape='Interval',method='self_adaptive',sa_boundary=True,lr=.005)
        a=build_pinn(config);a.train(30,display_every=0);state=a.training_state()
        b=build_pinn(config);b.load_training_state(state)
        a.train(20,display_every=0);b.train(20,display_every=0)
        self.assertEqual(a.loss_history,b.loss_history)
        for name in a.adaptive_weights:torch.testing.assert_close(a.adaptive_weights[name],b.adaptive_weights[name],rtol=0,atol=0)
        torch.testing.assert_close(a.predict(torch.tensor([[.3],[.7]])),b.predict(torch.tensor([[.3],[.7]])),rtol=0,atol=0)
        config.sa_lr=.01
        with self.assertRaisesRegex(ValueError,'settings'):build_pinn(config).load_training_state(state)
        config.sa_lr=.005;config.seed=8;config.sampling='random'
        with self.assertRaisesRegex(ValueError,'data/targets'):build_pinn(config).load_training_state(state)
        for mutate in (lambda s:s['adaptive']['weights']['pde'].fill_(float('nan')),lambda s:s['adaptive']['weights'].update(pde=torch.ones(1,1)),lambda s:s['adaptive']['optimiser']['param_groups'][0].update(maximize=False)):
            bad=deepcopy(state);mutate(bad)
            with self.assertRaises(ValueError):b.load_training_state(bad)
        with self.assertRaises(ValueError):build_pinn(PoissonConfig(shape='Interval')).load_training_state(state)

    def test_nonfinite_does_not_step_and_manufactured_accuracy(self):
        network=torch.nn.Linear(1,1)
        model=SelfAdaptivePINN(PINNProblem(lambda x,u:u*float('nan'),torch.ones(2,1)),network).compile()
        before=deepcopy(network.state_dict())
        with self.assertRaises(FloatingPointError):model.train(1,display_every=0)
        self.assertEqual(model.iter,0)
        for k,v in network.state_dict().items():torch.testing.assert_close(v,before[k])
        self.assertTrue((model.adaptive_weights['pde']==1.).all())
        # A finite forward loss can still have a nonfinite derivative at zero.
        with torch.no_grad():
            network.weight.zero_();network.bias.zero_()
        model=SelfAdaptivePINN(PINNProblem(lambda x,u:torch.sqrt(u),torch.ones(2,1)),network).compile()
        with self.assertRaisesRegex(FloatingPointError,'gradient'):model.train(1,display_every=0)
        self.assertEqual(model.iter,0)
        self.assertTrue((model.adaptive_weights['pde']==1.).all())
        for adaptive_bc in (False,True):
            config=PoissonConfig(shape='Interval',method='self_adaptive',sa_boundary=adaptive_bc,points=32,lr=.005,boundary_weight=10.)
            model=build_pinn(config);model.train(700,display_every=0)
            x=torch.linspace(0,1,101)[:,None]
            self.assertLess((model.predict(x)-x*(1-x)).square().mean().sqrt().item(),.04)
            self.assertGreater(model.adaptive_summary()['pde']['maximum'],1.)
