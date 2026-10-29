"""gPINN residual derivatives, targets, boundary separation, RAR and restart."""
from copy import deepcopy
from dataclasses import asdict
import math,unittest
import torch
from margam.components import PINNProblem
from margam.components.differential import derivative
from margam.methods.gpinn import GradientEnhancedPINN
from margam.templates.poisson import PoissonConfig,build_pinn
from margam.templates.examples import example_config
from margam.components.gpinn_config import gpinn_defaults


class GPINNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_derivative_of_full_residual_includes_forcing_and_callable_target(self):
        class Cubic(torch.nn.Module):
            def __init__(self):super().__init__();self.a=torch.nn.Parameter(torch.tensor(2.,dtype=torch.float64))
            def forward(self,x):return self.a*x**3
        x=torch.tensor([[.2],[.7]],dtype=torch.float64)
        problem=PINNProblem(lambda x,u:derivative(u,x,0,2)+torch.sin(x),x,residual_target=lambda x:x*x)
        m=GradientEnhancedPINN(problem,Cubic(),coordinates=('x',),gradient_weights=dict(x=.25,y=0.,t=0.),weights={'pde':2.}).compile(device='cpu')
        samples=m.pde_samples(x.clone().requires_grad_());expected=12+torch.cos(x)-2*x
        torch.testing.assert_close(samples['pde_grad_x'][0],expected)
        losses=m.compute_losses();torch.testing.assert_close(losses['pde_grad_x'],expected.square().mean())
        self.assertEqual(m.weights['pde_grad_x'],.5)
        # The gradient penalty must backpropagate through the derivative itself.
        grad=torch.autograd.grad(losses['pde_grad_x'],m.network.a)[0]
        torch.testing.assert_close(grad,12*expected.mean())
        m.train(1,0);self.assertTrue(torch.isfinite(m.network.a))

    def test_zero_weights_equal_baseline_and_constant_derivative_is_zero(self):
        c=PoissonConfig(shape='Interval',points=8,layers=(4,4),device='cpu')
        baseline=build_pinn(c);c.method='gpinn';c.gpinn_weights=dict.fromkeys(('x','y','t'),0.)
        enhanced=build_pinn(c);baseline.train(4,0);enhanced.train(4,0);self.assertEqual(baseline.loss_history,enhanced.loss_history)
        for name,value in baseline.network.state_dict().items():torch.testing.assert_close(value,enhanced.network.state_dict()[name],rtol=0,atol=0)
        n=torch.nn.Linear(1,1);p=PINNProblem(lambda x,u:derivative(u,x,0),torch.tensor([[.2],[.7]]))
        model=GradientEnhancedPINN(p,n,coordinates=('x',),gradient_weights=gpinn_defaults()).compile(device='cpu')
        self.assertEqual(model.compute_losses()['pde_grad_x'].item(),0.);model.train(2,0);model.compile(device='cpu')
        self.assertEqual(model.iter,0)

    def test_examples_rar_and_exact_resume(self):
        for key in ('gpinn_poisson_2d','gpinn_burgers','gpinn_burgers_rar'):
            with self.subTest(case=key):
                c=example_config(key);c.points=8;c.boundary_points=2;c.layers=(4,4)
                from margam.templates.poisson import BoundarySampling
                c.boundary_sampling={side:BoundarySampling(2,c.boundary_recipe(side).method,c.boundary_recipe(side).seed) for side in c.sides}
                if c.adaptive_sampling:c.adaptive_sampling.update(interval=2,candidates=20,batch=3,maximum=20,chunk=8)
                full=build_pinn(c);full.train(5,0);part=build_pinn(c);part.train(2,0);resumed=build_pinn(c);resumed.load_training_state(part.training_state());resumed.train(3,0)
                self.assertEqual(full.loss_history,resumed.loss_history)
                for name,value in full.network.state_dict().items():torch.testing.assert_close(value,resumed.network.state_dict()[name],rtol=0,atol=0)
                self.assertEqual(set(full.loss_history[0]),{'pde','total',*c.constraint_names,*('pde_grad_'+n for n in c.coordinate_names)})
                if c.adaptive_sampling:self.assertEqual(len(full.collocation_point_sample),20)  # 8 interior + 6 boundary/initial anchors + 6 RAR rows.
                altered=deepcopy(c);altered.gpinn_weights['x']*=2
                with self.assertRaisesRegex(ValueError,'gPINN checkpoint'):build_pinn(altered).load_training_state(part.training_state())
        from margam.components.network_config import NETWORK_NAMES
        for kind in NETWORK_NAMES:
            with self.subTest(network=kind):
                c=PoissonConfig(shape='Interval',method='gpinn',network_type=kind,points=8,layers=(4,4));m=build_pinn(c);m.train(1,0)
                self.assertTrue(all(math.isfinite(v) for v in m.loss_history[-1].values()))
        from margam.templates.prepared_triangle import prepared_triangle_config
        c=prepared_triangle_config();c.method='gpinn';c.layers=(4,4);build_pinn(c).train(2,0)

    def test_invalid_settings_and_nonfinite_gradients_do_not_step(self):
        c=example_config('gpinn_burgers')
        for value in (-1.,float('nan'),True):
            bad=deepcopy(c);bad.gpinn_weights['x']=value
            with self.assertRaises(ValueError):bad.validate()
        bad=deepcopy(c);bad.pde_preset='';bad.physics='incompressible_2d'
        with self.assertRaises(ValueError):bad.validate()
        bad=deepcopy(c);bad.loss_formulation='integral'
        with self.assertRaises(ValueError):bad.validate()
        from margam.components.adaptive_sampling import sampling_defaults
        bad=deepcopy(c);bad.adaptive_sampling=sampling_defaults('importance')
        with self.assertRaisesRegex(ValueError,'importance'):bad.validate()
        from margam.components.adaptive_sampling import configure_adaptive_sampling
        manual=build_pinn(c);configure_adaptive_sampling(manual,bad)
        with self.assertRaisesRegex(ValueError,'fixed points or RAR'):manual.compute_losses()
        class Zero(torch.nn.Module):
            def __init__(self):super().__init__();self.a=torch.nn.Parameter(torch.tensor(0.))
            def forward(self,x):return self.a*x
        for weights in (gpinn_defaults(),dict.fromkeys(('x','y','t'),0.)):
            n=Zero();m=GradientEnhancedPINN(PINNProblem(lambda x,u:u.sqrt(),torch.ones(2,1)),n,coordinates=('x',),gradient_weights=weights).compile(device='cpu')
            with self.assertRaises(FloatingPointError):m.train(1,0)
            self.assertEqual(m.iter,0);self.assertEqual(n.a.item(),0.)

    def test_legacy_other_method_signatures_gain_default_gpinn_settings(self):
        for key in ('scale_cavity_400','expert_burgers'):
            c=example_config(key);c.points=8;c.boundary_points=2;c.layers=(4,4);c.network_options['features']=2
            if c.method=='expert':c.expert_settings.update(chunks=2,balance_every=2)
            m=build_pinn(c);m.train(2,0);state=m.training_state();payload='scale' if c.method=='scale' else 'expert'
            state[payload]['settings']['configuration'].pop('gpinn_weights')
            n=build_pinn(c);n.load_training_state(state);m.train(1,0);n.train(1,0);self.assertEqual(m.loss_history,n.loss_history)

        c=example_config('expert_burgers');c.points=8;c.boundary_points=2;c.layers=(4,4);c.network_options['features']=2
        c.expert_settings.update(chunks=2,balance_every=2,windows=2,window_steps=2)
        m=build_pinn(c);m.train(3,0);state=m.training_state()
        def legacy(value):
            if isinstance(value,dict):
                value.pop('gpinn_weights',None)
                for child in value.values():legacy(child)
            elif isinstance(value,(list,tuple)):
                for child in value:legacy(child)
        legacy(state);n=build_pinn(c);n.load_training_state(state);m.train(1,0);n.train(1,0);self.assertEqual(m.loss_history,n.loss_history)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_derivatives_and_checkpoint(self):
        c=example_config('gpinn_burgers_rar');c.device='cuda';c.points=8;c.boundary_points=2;c.layers=(4,4);c.adaptive_sampling.update(interval=2,candidates=20,batch=2,maximum=20)
        m=build_pinn(c);m.train(3,0);n=build_pinn(c);n.load_training_state(m.training_state());m.train(2,0);n.train(2,0)
        self.assertEqual(m.loss_history,n.loss_history)
        for name,value in m.network.state_dict().items():torch.testing.assert_close(value,n.network.state_dict()[name],rtol=1e-5,atol=1e-7)
