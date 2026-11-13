"""Fixed candidates retain finite-pool correction, boundary anchors and exact resume."""
from copy import deepcopy
import random,unittest
from unittest.mock import patch
import torch
from margam.components.adaptive_sampling import sampling_defaults,importance_probabilities,validate_adaptive_sampling
from margam.templates.poisson import PoissonConfig,build_pinn


def config(device='cpu',method='baseline',pool='fixed'):
    settings=sampling_defaults('importance');settings.update(interval=2,candidates=19,chunk=4)
    if pool is not None:settings['pool']=pool
    return PoissonConfig(shape='Interval',points=8,boundary_points=2,layers=(5,),pde_points='both',method=method,adaptive_sampling=settings,device=device)


class FixedImportancePoolTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_coordinates_stay_fixed_scores_refresh_and_random_state_is_private(self):
        from margam.components.sampling import sample_box
        with patch('margam.components.adaptive_sampling.sample_box',wraps=sample_box) as sampled:
            model=build_pinn(config());controller=model.sampling_controller;pool=controller.candidate_pool.clone();seen=[]
            actual=controller.scores
            def score(m,points):seen.append(points.clone());return actual(m,points)
            with patch.object(controller,'scores',side_effect=score):model.train(7,0)
            self.assertEqual(sum(call.args[2]==19 for call in sampled.call_args_list),1)
        self.assertEqual(len(seen),3)
        for points in seen:torch.testing.assert_close(points,pool,rtol=0,atol=0)
        model.iter=8;torch_rng=torch.get_rng_state().clone();python_rng=random.getstate()
        controller.before_step(model)
        self.assertTrue(torch.equal(torch_rng,torch.get_rng_state()));self.assertEqual(python_rng,random.getstate())
        self.assertFalse(controller.candidate_pool.requires_grad)
        torch.testing.assert_close(controller.candidate_pool,pool,rtol=0,atol=0)
        self.assertTrue(set(map(tuple,model.collocation_point_sample[:-2].tolist()))<=set(map(tuple,pool.tolist())))

    def _resume(self,device):
        for method in ('baseline','hard_dirichlet','lra'):
            for phase in (2,3):
                with self.subTest(device=device,method=method,phase=phase):
                    c=config(device,method);full=build_pinn(c);full.train(7,0)
                    part=build_pinn(c);anchors=part.collocation_point_sample[-2:].clone();part.train(phase,0)
                    saved=part.training_state();continued=build_pinn(c);continued.load_training_state(saved);continued.train(7-phase,0)
                    self.assertEqual(full.loss_history,continued.loss_history)
                    self.assertEqual(full.sampling_controller.history,continued.sampling_controller.history)
                    for name,value in full.network.state_dict().items():torch.testing.assert_close(value,continued.network.state_dict()[name],rtol=0,atol=0)
                    torch.testing.assert_close(full.collocation_point_sample,continued.collocation_point_sample,rtol=0,atol=0)
                    torch.testing.assert_close(full.sampling_controller.weights,continued.sampling_controller.weights,rtol=0,atol=0)
                    torch.testing.assert_close(continued.collocation_point_sample[-2:],anchors,rtol=0,atol=0)
                    torch.testing.assert_close(continued.sampling_controller.weights[-2:],anchors.new_ones(2,1),rtol=0,atol=0)
                    torch.testing.assert_close(continued.sampling_controller.candidate_pool,saved['sampling']['candidate_pool'],rtol=0,atol=0)

    def test_cpu_all_supported_trainers_exact_resume(self):self._resume('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_all_supported_trainers_exact_resume(self):self._resume('cuda')

    def test_checkpoint_rejects_changed_pool_and_outside_draw_without_mutating_model(self):
        c=config();model=build_pinn(c);model.train(3,0);state=model.training_state()
        for corruption in ('pool','missing','dtype','draw'):
            with self.subTest(corruption=corruption):
                bad=deepcopy(state)
                if corruption=='pool':bad['sampling']['candidate_pool'][0,0]+=.01
                elif corruption=='missing':bad['sampling'].pop('candidate_pool')
                elif corruption=='dtype':bad['sampling']['candidate_pool']=bad['sampling']['candidate_pool'].to(torch.float16)
                else:bad['sampling']['points'][0,0]=.123456
                target=build_pinn(c);before=deepcopy(target.network.state_dict())
                with self.assertRaises(ValueError):target.load_training_state(bad)
                self.assertEqual(target.iter,0)
                for name,value in before.items():torch.testing.assert_close(value,target.network.state_dict()[name],rtol=0,atol=0)
        legacy=build_pinn(config(pool=None))
        with self.assertRaises(ValueError):legacy.load_training_state(state)

    def test_legacy_and_explicit_refresh_have_identical_trajectories_and_old_state(self):
        old=build_pinn(config(pool=None));new=build_pinn(config(pool='refresh'))
        old.train(7,0);new.train(7,0)
        self.assertEqual(old.loss_history,new.loss_history)
        state=old.training_state();self.assertNotIn('candidate_pool',state['sampling']);self.assertNotIn('pool',state['sampling']['settings'])
        resumed=build_pinn(config(pool='refresh'));resumed.load_training_state(state);resumed.train(1,0);old.train(1,0)
        self.assertEqual(old.loss_history,resumed.loss_history)
        for settings in (dict(sampling_defaults('rar'),pool='fixed'),dict(sampling_defaults('importance'),pool='unknown')):
            with self.assertRaises(ValueError):validate_adaptive_sampling(settings)

    def test_finite_pool_expected_loss_and_gradient_and_atomic_score_failure(self):
        c=config();model=build_pinn(c);controller=model.sampling_controller
        x=controller.candidate_pool[:,0].double();probability=importance_probabilities(x,2.,.9)
        parameter=torch.tensor(1.7,dtype=torch.float64,requires_grad=True);error=(parameter*x-1).square()
        expected=(probability*(error/(len(x)*probability))).sum();uniform=error.mean()
        torch.testing.assert_close(expected,uniform)
        torch.testing.assert_close(torch.autograd.grad(expected,parameter,retain_graph=True)[0],torch.autograd.grad(uniform,parameter)[0])
        before=model.collocation_point_sample.clone();pool=controller.candidate_pool.clone();model.iter=2
        with patch.object(model.problem,'residual',side_effect=lambda x,u:u*float('nan')):
            with self.assertRaises(FloatingPointError):controller.before_step(model)
        self.assertEqual(controller.history,[]);self.assertIsNone(controller.weights)
        torch.testing.assert_close(model.collocation_point_sample,before,rtol=0,atol=0)
        torch.testing.assert_close(controller.candidate_pool,pool,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
