"""Selection mathematics, scheduling, derivative scoring and exact resume."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import torch
from margam.components.adaptive_sampling import sampling_defaults,importance_probabilities
from margam.components.problem import PINNProblem
from margam.methods.pinn import PINN
from margam.templates.poisson import PoissonConfig,build_pinn


def config(sampler='rar',**options):
    settings=sampling_defaults(sampler);settings.update(interval=2,candidates=19,chunk=4)
    if sampler=='rar':settings.update(batch=3,maximum=15)
    values=dict(shape='Interval',points=6,layers=(8,),iterations=9,adaptive_sampling=settings);values.update(options)
    return PoissonConfig(**values)


class AdaptiveSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_rar_selects_largest_residuals_without_touching_boundaries(self):
        model=build_pinn(config());controller=model.sampling_controller
        original=model.collocation_point_sample.clone();boundaries=[points.clone() for _,points,_ in model._constraints]
        candidates=[[.1],[.2],[.3],[.4],[.5],[.6],[.7],[.8],[.9]]
        controller.settings['candidates']=len(candidates)
        controller.settings['batch']=3
        model.iter=2
        with patch('margam.components.adaptive_sampling.sample_box',return_value=candidates),patch.object(controller,'scores',return_value=torch.arange(9,dtype=torch.float64)):
            controller.before_step(model)
            controller.before_step(model) # Same phase is not applied twice.
        torch.testing.assert_close(model.collocation_point_sample[:6],original,rtol=0,atol=0)
        torch.testing.assert_close(model.collocation_point_sample[6:],torch.tensor([[.9],[.8],[.7]]),rtol=0,atol=0)
        self.assertEqual(len(controller.history),1)
        for expected,(_,actual,_) in zip(boundaries,model._constraints):torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        # Repeated candidate coordinates and existing points cannot create anchors twice.
        model.iter=4
        with patch('margam.components.adaptive_sampling.sample_box',return_value=[[.9]]*9),patch.object(controller,'scores',return_value=torch.ones(9)):
            controller.before_step(model)
        self.assertEqual(len(model.collocation_point_sample),9)

    def test_importance_estimates_uniform_pool_loss_and_gradient(self):
        scores=torch.tensor([0.,1.,4.],dtype=torch.float64)
        probabilities=importance_probabilities(scores,2.,.9)
        self.assertTrue((probabilities>0).all())
        self.assertAlmostEqual(probabilities.sum().item(),1.)
        parameter=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
        errors=(parameter*torch.tensor([1.,2.,3.],dtype=torch.float64)).square()
        expected=(probabilities*errors/(3*probabilities)).sum()
        actual=errors.mean()
        torch.testing.assert_close(expected,actual)
        torch.testing.assert_close(torch.autograd.grad(expected,parameter,retain_graph=True)[0],torch.autograd.grad(actual,parameter)[0])
        torch.testing.assert_close(importance_probabilities(torch.zeros(3),2.,.9),torch.full((3,),1/3))
        model=build_pinn(config('importance'));model.train(3,display_every=0)
        self.assertEqual(len(model.collocation_point_sample),6)
        samples=model.loss_samples();values,target,rows=samples['pde'];values,target=model._aligned(values,target,rows)
        manual=((values-target).square().mean(1,keepdim=True)*model.sampling_controller.weights).mean()
        torch.testing.assert_close(model.compute_losses()['pde'],manual)

    def test_derivative_scores_are_chunked_detached_and_nonfinite_atomic(self):
        model=build_pinn(config());points=torch.linspace(.1,.9,13)[:,None]
        score=model.sampling_controller.scores(model,points)
        x=points.clone().requires_grad_();expected=model.problem.residual(x,model.network(x)).detach().double().abs().flatten()
        torch.testing.assert_close(score,expected,rtol=1e-6,atol=1e-7)
        self.assertFalse(score.requires_grad)
        self.assertTrue(all(p.grad is None for p in model.network.parameters()))
        original=model.collocation_point_sample.clone();model.iter=2
        with patch.object(model.problem,'residual',side_effect=lambda x,u:u*float('nan')):
            with self.assertRaisesRegex(FloatingPointError,'Non-finite'):model.sampling_controller.before_step(model)
        torch.testing.assert_close(model.collocation_point_sample,original)
        self.assertEqual(model.sampling_controller.history,[])

    def test_all_supported_trainers_resume_identically_at_update_boundaries(self):
        for sampler in ('rar','importance'):
            for trainer in ('baseline','hard_dirichlet','lra'):
                for phase in (2,3):
                    with self.subTest(sampler=sampler,trainer=trainer,phase=phase):
                        c=config(sampler,method=trainer)
                        continuous=build_pinn(c);continuous.train(9,display_every=0)
                        interrupted=build_pinn(c);interrupted.train(phase,display_every=0)
                        resumed=build_pinn(c);resumed.load_training_state(interrupted.training_state());resumed.train(9-phase,display_every=0)
                        self.assertEqual(resumed.loss_history,continuous.loss_history)
                        self.assertEqual(resumed.sampling_controller.history,continuous.sampling_controller.history)
                        torch.testing.assert_close(resumed.collocation_point_sample,continuous.collocation_point_sample,rtol=0,atol=0)
                        for key,value in resumed.network.state_dict().items():torch.testing.assert_close(value,continuous.network.state_dict()[key],rtol=0,atol=0)
                        self.assertLessEqual(len(resumed.collocation_point_sample),15)

    def test_flow_residuals_and_rectangle_bounds(self):
        for sampler in ('rar','importance'):
            c=config(sampler,shape='Rectangle',physics='incompressible_2d')
            c.points=9;c.boundary_points=3;c.layers=(8,)
            model=build_pinn(c);model.train(5,display_every=0)
            self.assertEqual(len(model.sampling_controller.history),2)
            self.assertTrue(torch.isfinite(model.collocation_point_sample).all())
            self.assertEqual(model.collocation_point_sample.shape[1],2)

    def test_rejects_unsupported_and_corrupt_checkpoint_data(self):
        for method in ('rar','importance'):
            c=config(method,method='self_adaptive')
            with self.assertRaisesRegex(ValueError,'pointwise masks'):build_pinn(c)
            c=config(method);c.interior_data=dict(path='missing.csv',coordinates='x',normals='',values='')
            with self.assertRaisesRegex(ValueError,'candidate source'):build_pinn(c)
            c=config(method);model=build_pinn(c);model.train(5,display_every=0);state=model.training_state()
            for corrupt in ('bounds','phase','nonfinite','settings','missing'):
                bad=deepcopy(state)
                if corrupt=='bounds':bad['sampling']['points'][0,0]=2.
                elif corrupt=='phase':bad['sampling']['history'][0]['step']=1
                elif corrupt=='nonfinite':bad['sampling']['points'][0,0]=float('nan')
                elif corrupt=='settings':bad['sampling']['settings']['interval']=9
                else:bad.pop('sampling')
                with self.subTest(method=method,corrupt=corrupt),self.assertRaises(ValueError):build_pinn(c).load_training_state(bad)
            changed=config(method);changed.points=7
            with self.assertRaisesRegex(ValueError,'data/targets'):build_pinn(changed).load_training_state(state)

    def test_archive_roundtrip_uses_restricted_loader(self):
        from margam.ui.run_document import save_run,load_run
        for sampler in ('rar','importance'):
            c=config(sampler);model=build_pinn(c);model.train(5,display_every=0)
            archive=dict(format='margam-run',version=2,model=dict(format='margam-model',version=13,config=asdict(c),nodes=[],wires=[]),
                         result=dict(outputs={},steps=5,cancelled=False,device='cpu'),monitors={},checkpoint=model.training_state())
            with TemporaryDirectory() as directory:
                path=Path(directory)/'adaptive.pt';save_run(path,archive);loaded=load_run(path)
                resumed=build_pinn(c);resumed.load_training_state(loaded['checkpoint'])
                resumed.train(4,display_every=0);model.train(4,display_every=0)
                self.assertEqual(resumed.loss_history,model.loss_history)

    def test_legacy_models_remain_fixed_and_invalid_settings_rejected(self):
        from margam.ui.model_document import decode_document
        old=asdict(PoissonConfig());old.pop('adaptive_sampling')
        c,_,_=decode_document(dict(format='margam-model',version=12,config=old,nodes=[],wires=[]))
        self.assertEqual(c.adaptive_sampling,{})
        for key,value in (('interval',True),('candidates',0),('batch',100001),('maximum',1),('chunk',99999)):
            c=config();c.adaptive_sampling[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):build_pinn(c)
