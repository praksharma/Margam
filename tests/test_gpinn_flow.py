"""Component-wise residual gradients for concrete steady u/v/p flow."""
from copy import deepcopy
import unittest
import torch
from margam.components.problem import PINNProblem
from margam.methods.gpinn import GradientEnhancedPINN
from margam.templates.poisson import build_pinn
from margam.templates.examples import example_config


class FlowGPINNTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_opposite_components_do_not_cancel_and_targets_are_differentiated(self):
        class Cubic(torch.nn.Module):
            def __init__(self):super().__init__();self.a=torch.nn.Parameter(torch.tensor(1.7,dtype=torch.float64))
            def forward(self,x):return self.a*torch.cat((x[:,0:1]**3,-x[:,0:1]**3,x[:,1:2]**2),dim=1)
        points=torch.tensor([[.2,.3],[.4,.7],[.8,.5]],dtype=torch.float64)
        def target(x):return torch.cat((x[:,0:1],-x[:,0:1],x[:,1:2]),dim=1)
        net=Cubic();p=PINNProblem(lambda x,u:u,points,residual_target=target)
        model=GradientEnhancedPINN(p,net,coordinates=('x','y'),gradient_weights=dict(x=.2,y=.3,t=0.),residual_components=3).compile(device='cpu')
        terms=model.compute_losses();a=net.a;x=points[:,0];y=points[:,1]
        expected_x=(2*(3*a*x*x-1)**2/3).mean();expected_y=((2*a*y-1)**2/3).mean()
        torch.testing.assert_close(terms['pde_grad_x'],expected_x);torch.testing.assert_close(terms['pde_grad_y'],expected_y)
        self.assertGreater(terms['pde_grad_x'].item(),0)
        actual_grad=torch.autograd.grad(terms['pde_grad_x']+terms['pde_grad_y'],a,retain_graph=True)[0]
        expected_grad=torch.autograd.grad(expected_x+expected_y,a)[0]
        torch.testing.assert_close(actual_grad,expected_grad)

    def _resume(self,device,rar=False):
        c=example_config('gpinn_couette_2d');c.device=device;c.points=9;c.boundary_points=3;c.layers=(5,5)
        if rar:
            from margam.components.adaptive_sampling import sampling_defaults
            c.adaptive_sampling=sampling_defaults('rar');c.adaptive_sampling.update(interval=2,candidates=17,batch=2,maximum=100,chunk=4)
        full=build_pinn(c);full.train(5,0)
        part=build_pinn(c);part.train(3,0);state=part.training_state()
        self.assertEqual(state['gpinn']['residual_components'],3)
        resumed=build_pinn(c);resumed.load_training_state(state);resumed.train(2,0)
        self.assertEqual(full.loss_history,resumed.loss_history)
        for name,value in full.network.state_dict().items():torch.testing.assert_close(value,resumed.network.state_dict()[name],rtol=0,atol=0)
        self.assertEqual(tuple(full.predict([[.2,.4]]).shape),(1,3))
        self.assertIn('pressure_gauge',full.loss_history[-1]);self.assertIn('pde_grad_y',full.loss_history[-1])
        if rar:
            self.assertEqual(full.sampling_controller.history,resumed.sampling_controller.history)
            torch.testing.assert_close(full.collocation_point_sample,resumed.collocation_point_sample,rtol=0,atol=0)
        wrong=deepcopy(state);wrong['gpinn']['residual_components']=1
        with self.assertRaisesRegex(ValueError,'checkpoint'):build_pinn(c).load_training_state(wrong)

    def test_cpu_fixed_and_rar_flow_exact_resume(self):
        for rar in (False,True):self._resume('cpu',rar)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_fixed_and_rar_flow_exact_resume(self):
        for rar in (False,True):self._resume('cuda',rar)

    def test_zero_gradient_weights_reproduce_baseline_flow(self):
        c=example_config('gpinn_couette_2d');c.points=9;c.boundary_points=3;c.layers=(5,);c.device='cpu';c.gpinn_weights=dict(x=0.,y=0.,t=0.)
        enhanced=build_pinn(c);c.method='baseline';baseline=build_pinn(c)
        enhanced.train(3,0);baseline.train(3,0)
        self.assertEqual(enhanced.loss_history,baseline.loss_history)
        for name,value in enhanced.network.state_dict().items():torch.testing.assert_close(value,baseline.network.state_dict()[name],rtol=0,atol=0)

    def test_couette_reference_satisfies_each_flow_equation_and_its_gradients(self):
        class Exact(torch.nn.Module):
            def __init__(self):super().__init__();self.a=torch.nn.Parameter(torch.tensor(1.))
            def forward(self,x):return torch.cat((self.a*x[:,1:2],0*x[:,0:1],0*x[:,0:1]),1)
        c=example_config('gpinn_couette_2d');c.points=9;c.boundary_points=3;c.layers=(5,);c.device='cpu'
        configured=build_pinn(c)
        exact=GradientEnhancedPINN(configured.problem,Exact(),coordinates=('x','y'),gradient_weights=c.gpinn_weights,weights=c.loss_weights,residual_components=3).compile(device='cpu')
        for name,loss in exact.compute_losses().items():self.assertEqual(loss.item(),0.,name)

    def test_scope_and_component_count_are_explicit(self):
        c=example_config('gpinn_couette_2d');c.shape='SpaceTime1D'
        with self.assertRaises(ValueError):c.validate()
        p=PINNProblem(lambda x,u:torch.cat((u,u),1),torch.ones(2,2))
        m=GradientEnhancedPINN(p,torch.nn.Linear(2,1),coordinates=('x','y'),gradient_weights=dict(x=.1,y=.1,t=0.),residual_components=3).compile(device='cpu')
        with self.assertRaisesRegex(ValueError,'component count'):m.compute_losses()
        with self.assertRaisesRegex(ValueError,'steady'):GradientEnhancedPINN(p,torch.nn.Linear(2,1),coordinates=('x','t'),gradient_weights=dict(x=.1,y=.1,t=0.),residual_components=3)


if __name__=='__main__':unittest.main()
