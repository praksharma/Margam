"""Paper-v1 gradient equation, safeguards, method reuse and exact continuation."""
from copy import deepcopy
import unittest
import torch
from margam.components import PINNProblem,PointConstraint
from margam.methods.learning_rate_annealing import LearningRateAnnealingPINN
from margam.templates.poisson import PoissonConfig,build_pinn

class LRATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_raw_gradient_ratio_and_ema(self):
        network=torch.nn.Linear(1,1)
        model=LearningRateAnnealingPINN(PINNProblem(lambda x,u:u,torch.ones(2,1),[PointConstraint('bc',torch.zeros(1,1))]),network,weights={'pde':2.,'bc':.5}).compile(alpha=.9,max_weight=100.)
        a,b=tuple(network.parameters())
        # PDE gradient [6,8], constraint [1,0] including unused bias.
        terms={'pde':3*a.sum()+4*b.sum(),'bc':2*a.sum()}
        model.lra_weights['bc']=5.
        weights,event=model._candidate_weights(terms)
        self.assertEqual(event['pde_max'],8.)
        self.assertEqual(event['means']['bc'],.5)
        self.assertEqual(event['targets']['bc'],16.)
        self.assertAlmostEqual(weights['bc'],.1*5+.9*16)
        # Existing lambda=5 is not part of the denominator.
        weights,event=model._candidate_weights({'pde':terms['pde'],'bc':a.sum()*0})
        self.assertIsNone(event['targets']['bc']);self.assertEqual(weights['bc'],5.)
        weights,event=model._candidate_weights({'pde':terms['pde']*100,'bc':terms['bc']})
        self.assertEqual(event['targets']['bc'],100.)
        self.assertAlmostEqual(weights['bc'],90.5)

    def test_batch_phase_exact_resume_and_tamper_rejection(self):
        # Existing format-10 definitions without LRA fields still load.
        from dataclasses import asdict
        from margam.ui.model_document import decode_document
        legacy=asdict(PoissonConfig())
        for key in ('lra_alpha','lra_interval','lra_epsilon','lra_max'):legacy.pop(key)
        for version in (7,8,9,10):
            restored,_,_=decode_document(dict(format='margam-model',version=version,config=legacy,nodes=[],wires=[]))
            self.assertEqual(restored.lra_interval,10)
            self.assertEqual(restored.method,'baseline')
        config=PoissonConfig(shape='Interval',method='lra',points=16,layers=(8,8),lr=.001)
        continuous=build_pinn(config);continuous.train(42,display_every=0)
        a=build_pinn(config);a.train(25,display_every=0);state=a.training_state()
        b=build_pinn(config);b.load_training_state(state)
        a.train(17,display_every=0);b.train(17,display_every=0)
        self.assertEqual(continuous.loss_history,a.loss_history)
        self.assertEqual(a.loss_history,b.loss_history)
        self.assertEqual(a.lra_history,b.lra_history)
        self.assertEqual([e['step'] for e in b.lra_history],[1,11,21,31,41])
        x=torch.tensor([[.3],[.7]])
        torch.testing.assert_close(continuous.predict(x),b.predict(x),rtol=0,atol=0)
        for mutate in (lambda s:s['lra']['weights'].update(left=float('nan')),lambda s:s['lra']['history'][0].update(step=2),lambda s:s['lra']['history'][0]['targets'].update(right=123.),lambda s:s['lra']['settings'].update(alpha=.1)):
            bad=deepcopy(state);mutate(bad)
            with self.assertRaises(ValueError):build_pinn(config).load_training_state(bad)
        config.lra_interval=5
        with self.assertRaisesRegex(ValueError,'settings'):build_pinn(config).load_training_state(state)
        config.lra_interval=10;config.sampling='random';config.seed=9
        with self.assertRaisesRegex(ValueError,'data/targets'):build_pinn(config).load_training_state(state)
        with self.assertRaises(ValueError):build_pinn(PoissonConfig(shape='Interval')).load_training_state(state)

    def test_nonfinite_no_update_and_settings(self):
        network=torch.nn.Linear(1,1)
        model=LearningRateAnnealingPINN(PINNProblem(lambda x,u:u*float('nan'),torch.ones(2,1)),network).compile()
        before=deepcopy(network.state_dict())
        with self.assertRaises(FloatingPointError):model.train(1,display_every=0)
        self.assertEqual(model.iter,0);self.assertFalse(model.lra_history);self.assertFalse(model.optimiser.state)
        for name,value in network.state_dict().items():torch.testing.assert_close(value,before[name],rtol=0,atol=0)
        with torch.no_grad():network.weight.zero_();network.bias.zero_()
        model=LearningRateAnnealingPINN(PINNProblem(lambda x,u:torch.sqrt(u),torch.ones(2,1)),network).compile()
        with self.assertRaisesRegex(FloatingPointError,'gradient'):model.train(1,display_every=0)
        self.assertEqual(model.iter,0);self.assertFalse(model.optimiser.state)
        for settings in ({'alpha':True},{'alpha':0},{'update_interval':True},{'epsilon':float('nan')},{'max_weight':.1}):
            with self.subTest(settings=settings),self.assertRaises(ValueError):model.compile(**settings)

    def test_existing_architectures_and_flow(self):
        from margam.components.network_config import NETWORK_NAMES
        for kind in NETWORK_NAMES:
            with self.subTest(network=kind):
                model=build_pinn(PoissonConfig(shape='Interval',method='lra',network_type=kind,points=8,layers=(8,8)))
                model.train(2,display_every=0)
                self.assertTrue(torch.isfinite(model.predict(torch.tensor([[.3],[.7]]))).all())
        model=build_pinn(PoissonConfig(physics='incompressible_2d',method='lra',points=16,boundary_points=4,layers=(8,8)))
        model.train(3,display_every=0)
        self.assertIn('pressure_gauge',model.lra_weights)
        self.assertEqual(model.predict(torch.tensor([[.2,.3]])).shape,(1,3))
