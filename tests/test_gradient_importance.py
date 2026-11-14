"""Selected spatial-field gradients, unbiased correction and exact sampler state."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import torch
from margam.components.adaptive_sampling import sampling_defaults,importance_probabilities,validate_adaptive_sampling,score_statistics
from margam.templates.poisson import PoissonConfig,build_pinn


def config(device='cpu',method='baseline',pool='fixed',flow=False):
    settings=sampling_defaults('importance')
    settings.update(interval=2,candidates=19,chunk=4,pool=pool,score='field_gradient',field=1 if flow else 0)
    return PoissonConfig(shape='Rectangle' if flow else 'Interval',physics='incompressible_2d' if flow else 'scalar',
        points=9,boundary_points=3,layers=(5,),pde_points='both',method=method,adaptive_sampling=settings,device=device)


class Polynomial(torch.nn.Module):
    def __init__(self):
        super().__init__();self.scale=torch.nn.Parameter(torch.tensor(2.))
    def forward(self,x):
        a,b=x[:,0:1],x[:,1:2]
        return torch.cat((self.scale*(a*a+3*b),self.scale*(a*b),self.scale*(a-4*b)),dim=1)


class GradientImportanceTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_selected_field_scores_match_analytic_gradient_without_residual_or_parameter_grads(self):
        model=build_pinn(config(flow=True));model.network=Polynomial();controller=model.sampling_controller
        points=torch.tensor([[.1,.2],[.3,.4],[.5,.7],[.9,.8],[.2,.6]])
        rng=torch.get_rng_state().clone()
        for field in (0,1,2):
            controller.settings['field']=field
            with patch.object(model.problem,'residual',side_effect=AssertionError('Field scores do not evaluate the PDE')),torch.no_grad():
                actual=controller.scores(model,points)
            x,y=points.double().unbind(1)
            expected=(torch.stack((4*x,torch.full_like(y,6)),1) if field==0 else torch.stack((2*y,2*x),1) if field==1 else torch.stack((torch.full_like(x,2),torch.full_like(y,-8)),1)).norm(dim=1)
            torch.testing.assert_close(actual,expected,rtol=1e-7,atol=1e-7)
            self.assertFalse(actual.requires_grad);self.assertEqual(actual.device.type,'cpu')
            self.assertIsNone(model.network.scale.grad)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertIsNone(points.grad)

    def test_zero_and_nonfinite_scores_and_field_guards_are_atomic(self):
        model=build_pinn(config());controller=model.sampling_controller
        class Constant(torch.nn.Module):
            def forward(self,x):return torch.ones((len(x),1),device=x.device)
        model.network=Constant();scores=controller.scores(model,model.collocation_point_sample)
        torch.testing.assert_close(scores,torch.zeros_like(scores))
        torch.testing.assert_close(importance_probabilities(scores,2.,.9),torch.full_like(scores,1/len(scores)))
        model.network=torch.nn.Linear(1,1)
        with torch.no_grad():model.network.weight.zero_();model.network.bias.fill_(3)
        torch.testing.assert_close(controller.scores(model,model.collocation_point_sample),torch.zeros_like(scores))
        class Invalid(torch.nn.Module):
            def __init__(self,kind):super().__init__();self.kind=kind
            def forward(self,x):return x*float('nan') if self.kind=='values' else torch.sqrt(x*0)
        for kind in ('values','derivatives'):
            model.network=Invalid(kind);model.iter=2
            original=model.collocation_point_sample.clone();pool=controller.candidate_pool.clone()
            with self.assertRaises(FloatingPointError):controller.before_step(model)
            torch.testing.assert_close(original,model.collocation_point_sample,rtol=0,atol=0)
            torch.testing.assert_close(pool,controller.candidate_pool,rtol=0,atol=0)
            self.assertEqual(controller.history,[]);self.assertIsNone(controller.weights)
        for change in (dict(score='unknown'),dict(field=True),dict(field=-1),dict(field=3),dict(score='residual',field=0)):
            with self.subTest(change=change),self.assertRaises(ValueError):validate_adaptive_sampling(config().adaptive_sampling|change)
        c=config();c.adaptive_sampling['field']=1
        with self.assertRaisesRegex(ValueError,'not an output'):build_pinn(c)
        c=config();c.shape='SpaceTime1D';c.pde_preset='burgers_1d'
        with self.assertRaisesRegex(ValueError,'steady'):build_pinn(c)
        with self.assertRaises(ValueError):validate_adaptive_sampling(sampling_defaults('rar')|dict(score='field_gradient'))

    def test_gradient_probabilities_preserve_expected_pde_objective_and_parameter_gradient(self):
        model=build_pinn(config());controller=model.sampling_controller
        points=controller.candidate_pool;scores=controller.scores(model,points)
        probability=importance_probabilities(scores,2.,.9)
        x=points.detach().requires_grad_();residual=model.problem.residual(x,model.network(x))
        target=model._target(model.problem.residual_target,x)
        error=(residual-target).square().mean(1).double()
        expected=(probability*error/(len(points)*probability)).sum();uniform=error.mean()
        torch.testing.assert_close(expected,uniform)
        parameters=tuple(model.network.parameters())
        first=torch.autograd.grad(expected,parameters,retain_graph=True,allow_unused=True)
        second=torch.autograd.grad(uniform,parameters,allow_unused=True)
        for a,b in zip(first,second):
            if a is None:self.assertIsNone(b)
            else:torch.testing.assert_close(a,b)
        model.train(3,0)
        values,target,rows=model.loss_samples()['pde'];values,target=model._aligned(values,target,rows)
        torch.testing.assert_close(model.compute_losses()['pde'],((values-target).square().mean(1,keepdim=True)*controller.weights).mean())
        self.assertEqual(set(controller.history[0]),{'step','points','mean_gradient','max_gradient'})
        self.assertEqual(score_statistics(controller.history[0])[0],'field-gradient magnitude')

    def _continuation(self,device):
        for method in ('baseline','hard_dirichlet','lra'):
            for pool in ('refresh','fixed'):
                for phase in (2,3):
                    with self.subTest(device=device,method=method,pool=pool,phase=phase):
                        c=config(device,method,pool);full=build_pinn(c);full.train(7,0)
                        part=build_pinn(c);part.train(phase,0);saved=part.training_state()
                        resumed=build_pinn(c);resumed.load_training_state(saved);resumed.train(7-phase,0)
                        self.assertEqual(full.loss_history,resumed.loss_history)
                        self.assertEqual(full.sampling_controller.history,resumed.sampling_controller.history)
                        for name,value in full.network.state_dict().items():torch.testing.assert_close(value,resumed.network.state_dict()[name],rtol=0,atol=0)
                        torch.testing.assert_close(full.collocation_point_sample,resumed.collocation_point_sample,rtol=0,atol=0)
                        torch.testing.assert_close(full.sampling_controller.weights,resumed.sampling_controller.weights,rtol=0,atol=0)
                        torch.testing.assert_close(resumed.sampling_controller.weights[-2:],torch.ones((2,1),device=device),rtol=0,atol=0)
        # A selected coupled field and integral loss are independently exercised.
        for flow,integral in ((True,False),(False,True)):
            c=config(device,flow=flow)
            if integral:c.loss_formulation='integral';c.pde_points='interior';c.x_max=3
            model=build_pinn(c);model.train(3,0);saved=model.training_state()
            resumed=build_pinn(c);resumed.load_training_state(saved);model.train(2,0);resumed.train(2,0)
            self.assertEqual(model.loss_history,resumed.loss_history)
            for name,value in model.network.state_dict().items():torch.testing.assert_close(value,resumed.network.state_dict()[name],rtol=0,atol=0)

    def test_cpu_continuation_all_supported_trainers_and_pools(self):self._continuation('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_continuation_all_supported_trainers_and_pools(self):self._continuation('cuda')

    def test_checkpoint_rejects_changed_measure_field_and_mislabelled_history_before_mutation(self):
        c=config(flow=True);model=build_pinn(c);model.train(3,0);state=model.training_state()
        for change in ('measure','field','history'):
            bad=deepcopy(state)
            if change=='measure':bad['sampling']['settings'].pop('score');bad['sampling']['settings'].pop('field')
            elif change=='field':bad['sampling']['settings']['field']=2
            else:bad['sampling']['history'][0]['mean_residual']=bad['sampling']['history'][0].pop('mean_gradient')
            target=build_pinn(c);before=deepcopy(target.network.state_dict())
            with self.subTest(change=change),self.assertRaises(ValueError):target.load_training_state(bad)
            self.assertEqual(target.iter,0)
            for name,value in before.items():torch.testing.assert_close(value,target.network.state_dict()[name],rtol=0,atol=0)
        # Explicit residual is the same legacy choice, including checkpoint schema.
        old=config();old.adaptive_sampling.pop('score');old.adaptive_sampling.pop('field')
        explicit=deepcopy(old);explicit.adaptive_sampling['score']='residual'
        a=build_pinn(old);b=build_pinn(explicit);a.train(3,0);b.train(3,0)
        self.assertEqual(a.loss_history,b.loss_history);self.assertNotIn('score',b.training_state()['sampling']['settings'])
        b.load_training_state(a.training_state());a.train(2,0);b.train(2,0)
        self.assertEqual(a.loss_history,b.loss_history)


if __name__=='__main__':unittest.main()
