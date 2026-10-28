"""Paper equations, factorization, fresh causal batches and exact continuation."""
from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import torch
from margam.templates.examples import example_config
from margam.templates.poisson import build_pinn
from margam.methods.expert import ExpertPINN
from margam.nn.weight_factorization import FactorizedLinear


def small(device='cpu'):
    c=example_config('expert_burgers');c.device=device;c.points=32;c.boundary_points=4;c.boundary_sampling={};c.sampling='random'
    from margam.templates.poisson import BoundarySampling
    c.boundary_sampling={s:BoundarySampling(4,'random',10+i) for i,s in enumerate(c.sides)}
    c.layers=(8,8);c.network_options.update(features=4);c.expert_settings.update(chunks=4,balance_every=2,decay_steps=10)
    return c


class ExpertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_detached_causal_prefix_weights(self):
        losses=torch.tensor([1.,2.,3.],dtype=torch.float64,requires_grad=True)
        value,weights=ExpertPINN.causal_objective(losses,.5)
        expected=torch.tensor([1.,math.exp(-.5),math.exp(-1.5)],dtype=torch.float64)
        torch.testing.assert_close(weights,expected);self.assertFalse(weights.requires_grad)
        torch.testing.assert_close(torch.autograd.grad(value,losses)[0],expected/3)
        _,uniform=ExpertPINN.causal_objective(losses,0.);torch.testing.assert_close(uniform,torch.ones(3,dtype=torch.float64))

    def test_sum_gradient_norm_balancing_old_weight_ema_and_zero_gradient(self):
        model=build_pinn(small());parameter=next(model.network.parameters())
        with torch.no_grad():parameter.flatten()[0]=2.
        scalar=parameter.flatten()[0]
        groups=dict(pde=scalar.square(),initial=2*scalar.square(),boundary=3*scalar.square())
        weights,event=model._balanced(groups)
        for name,target in dict(pde=6.,initial=3.,boundary=2.).items():
            self.assertAlmostEqual(event['targets'][name],target,places=6)
            self.assertAlmostEqual(weights[name],.9+.1*target,places=6)
        groups['pde']=scalar*0
        weights,event=model._balanced(groups)
        self.assertIsNone(event['targets']['pde']);self.assertEqual(weights['pde'],1.)

    def test_factorization_preserves_initial_function_and_derivatives(self):
        torch.manual_seed(4);plain=torch.nn.Linear(2,3).double();fact=FactorizedLinear(plain,1.,.1)
        x=torch.tensor([[.3,.7],[-.2,.9]],dtype=torch.float64,requires_grad=True)
        a=plain(x).tanh();b=fact(x).tanh()
        torch.testing.assert_close(a,b,rtol=1e-14,atol=1e-14)
        first_a=torch.autograd.grad(a.sum(),x,create_graph=True)[0];first_b=torch.autograd.grad(b.sum(),x,create_graph=True)[0]
        torch.testing.assert_close(first_a,first_b,rtol=1e-13,atol=1e-13)
        torch.testing.assert_close(torch.autograd.grad(first_a.sum(),x)[0],torch.autograd.grad(first_b.sum(),x)[0],rtol=1e-13,atol=1e-13)
        self.assertTrue((fact.log_scale.exp()>0).all())
        fact(x).tanh().square().sum().backward();self.assertIsNotNone(fact.log_scale.grad);self.assertIsNotNone(fact.direction.grad)

    def test_fresh_samples_remain_in_their_temporal_strata(self):
        model=build_pinn(small());before=model.collocation_point_sample.clone();fingerprint=model.data_fingerprint()
        for phase in (0,1,7):
            model._fresh_samples(phase);points=model.collocation_point_sample
            for chunk,rows in enumerate(points.reshape(4,8,2)):
                self.assertTrue(((rows[:,1]>=chunk/4)&(rows[:,1]<=(chunk+1)/4)).all())
                self.assertTrue(((rows[:,0]>=-1)&(rows[:,0]<=1)).all())
            self.assertEqual(model.data_fingerprint(),fingerprint)
            initial=next(points for condition,points,_ in model._constraints if condition.name=='initial')
            self.assertTrue((initial[:,1]==0).all())
        self.assertFalse(torch.equal(before,model.collocation_point_sample))

    def test_exact_phase_resume_decay_and_reject_corrupt_states(self):
        c=small();full=build_pinn(c);full.train(7,display_every=0)
        for phase in (0,2,3):
            partial=build_pinn(c);partial.train(phase,display_every=0);state=partial.training_state()
            resumed=build_pinn(c);resumed.load_training_state(state);resumed.train(7-phase,display_every=0)
            self.assertEqual(resumed.loss_history,full.loss_history)
            self.assertEqual(resumed.balance_history,full.balance_history)
            torch.testing.assert_close(resumed.collocation_point_sample,full.collocation_point_sample,rtol=0,atol=0)
            for name,value in full.network.state_dict().items():torch.testing.assert_close(value,resumed.network.state_dict()[name],rtol=0,atol=0)
        self.assertAlmostEqual(full.optimiser.param_groups[0]['lr'],c.lr*.9**(.6))
        state=full.training_state()
        for case in ('ratio','phase','coefficients','rate','settings'):
            bad=deepcopy(state)
            if case=='ratio':bad['expert']['history'][0]['targets']['pde']+=1.
            elif case=='phase':bad['expert']['history'].pop()
            elif case=='coefficients':bad['expert']['coefficients']['pde']=0.
            elif case=='rate':bad['optimiser']['param_groups'][0]['lr']=.1
            else:bad['expert']['settings']['configuration']['expert_settings']['momentum']=.5
            with self.subTest(case=case),self.assertRaises(ValueError):build_pinn(c).load_training_state(bad)

    def test_archive_and_nonfinite_update(self):
        from margam.ui.run_document import save_run,load_run
        c=small();m=build_pinn(c);m.train(3,display_every=0)
        data=dict(format='margam-run',version=2,model=dict(format='margam-model',version=16,config=asdict(c),nodes=[],wires=[]),result=dict(outputs={},steps=3,cancelled=False,device='cpu'),monitors={},checkpoint=m.training_state())
        with TemporaryDirectory() as directory:
            path=Path(directory)/'expert.pt';save_run(path,data);loaded=load_run(path)
            n=build_pinn(c);n.load_training_state(loaded['checkpoint']);n.train(2,display_every=0);m.train(2,display_every=0)
            self.assertEqual(n.loss_history,m.loss_history)
        before=deepcopy(m.network.state_dict());phase=m.iter
        with patch.object(m.problem,'residual',side_effect=lambda x,u:u*float('nan')):
            with self.assertRaises(FloatingPointError):m.train(1,display_every=0)
        self.assertEqual(m.iter,phase)
        for name,value in before.items():torch.testing.assert_close(value,m.network.state_dict()[name],rtol=0,atol=0)

    def test_unsupported_combinations_and_legacy_defaults(self):
        from margam.ui.model_document import decode_document
        for key,value in (('shape','Rectangle'),('sampling','grid'),('points',31),('loss_formulation','integral')):
            c=small();setattr(c,key,value)
            with self.subTest(key=key),self.assertRaises(ValueError):build_pinn(c)
        c=small();c.adaptive_sampling=dict(method='rar')
        with self.assertRaises(ValueError):build_pinn(c)
        from margam.templates.poisson import PoissonConfig
        config=asdict(PoissonConfig());config.pop('expert_settings')
        c,_,_=decode_document(dict(format='margam-model',version=15,config=config,nodes=[],wires=[]))
        self.assertEqual(c.expert_settings['momentum'],.9)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA GPU required')
    def test_cuda_training_and_checkpoint(self):
        c=small('cuda');m=build_pinn(c);m.train(3,display_every=0);state=m.training_state()
        n=build_pinn(c);n.load_training_state(state);n.train(2,display_every=0);m.train(2,display_every=0)
        self.assertTrue(m.collocation_point_sample.is_cuda);self.assertEqual(m.loss_history,n.loss_history)
        cpu=small('cpu');self.assertEqual(build_pinn(cpu).data_fingerprint(),build_pinn(c).data_fingerprint())
        from margam.ui.run_document import save_run,load_run
        data=dict(format='margam-run',version=2,model=dict(format='margam-model',version=16,config=asdict(c),nodes=[],wires=[]),result=dict(outputs={},steps=5,cancelled=False,device='cuda'),monitors={},checkpoint=m.training_state())
        with TemporaryDirectory() as directory:
            path=Path(directory)/'gpu-expert.pt';save_run(path,data);loaded=load_run(path)
            reopened=build_pinn(c);reopened.load_training_state(loaded['checkpoint'])
            self.assertEqual(reopened.loss_history,m.loss_history)
        self.assertTrue(all(math.isfinite(v) for row in m.loss_history for v in row.values()))
