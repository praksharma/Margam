"""Integral of a field versus integral of squared error; gradients and strict data identity."""
from copy import deepcopy
import io
import unittest
import torch
from margam.research import Trainer,PINNProblem,IntegralConstraint,derivative
from margam.methods.pinn import PINN
from examples.research_integral_flow import make_trainer,AxialVelocity


class IntegralConstraintTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_nonunit_flux_matches_manual_loss_and_gradient_update(self):
        model=make_trainer();terms=model.compute_losses();a=model.network.amplitude
        expected=(4*a/3-4/3).square()
        torch.testing.assert_close(terms['flow_rate'],expected,rtol=1e-14,atol=1e-14)
        actual=torch.autograd.grad(terms['flow_rate'],a)[0]
        torch.testing.assert_close(actual,torch.autograd.grad(expected,a)[0])
        samples=model.loss_samples();self.assertEqual(samples['flow_rate'][0].shape,(1,1));self.assertEqual(samples['flow_rate'][2],1)
        # Physical cross-section measure is two, not normalized away.
        self.assertAlmostEqual(model._integral_weights['flow_rate'].sum().item(),2.,places=14)
        model.weights['flow_rate']=3.;before=model.objective()[1].item();model.train(1,0)
        self.assertAlmostEqual(model.loss_history[0]['total'],before)
        self.assertAlmostEqual(model.loss_history[0]['weighted:flow_rate'],3*model.loss_history[0]['flow_rate'])

    def test_cancellation_and_derivative_scalar_operator(self):
        points=torch.tensor([[-1.],[1.]],dtype=torch.float64)
        net=torch.nn.Linear(1,1,bias=False).double();net.weight.data.fill_(1.)
        problem=PINNProblem(lambda x,u:u*0,points,[IntegralConstraint('net_flux',points,quadrature_weights=[1.,1.])])
        model=Trainer(problem,net).compile(device='cpu')
        self.assertEqual(model.compute_losses()['net_flux'].item(),0.)
        self.assertEqual(model.network(points).square().mean().item(),1.)
        # Integrating a field derivative retains parameter gradients.
        points=torch.tensor([[1.,0.],[1.,.5],[1.,1.]],dtype=torch.float64)
        constraint=IntegralConstraint('slope',points,target=.3,operator=lambda x,u:derivative(u,x,1),quadrature_weights=[1/6,4/6,1/6])
        model=Trainer(PINNProblem(lambda x,u:u*0,points,[constraint]),AxialVelocity()).compile(device='cpu')
        # Integral of du/dy over [0,1] is amplitude, retaining nonzero parameter gradients.
        value=model.compute_losses()['slope'];expected=(model.network.amplitude-.3).square()
        torch.testing.assert_close(value,expected)
        torch.testing.assert_close(torch.autograd.grad(value,model.network.amplitude)[0],torch.autograd.grad(expected,model.network.amplitude)[0])

    def test_integral_mode_does_not_apply_cross_section_measure_twice(self):
        original=make_trainer();problem=original.problem
        problem.measure=7.
        mean=Trainer(problem,AxialVelocity()).compile(device='cpu')
        integral=Trainer(problem,AxialVelocity(),loss_formulation='integral').compile(device='cpu')
        torch.testing.assert_close(mean.compute_losses()['flow_rate'],integral.compute_losses()['flow_rate'])
        custom=Trainer(problem,AxialVelocity(),loss={'flow_rate':torch.nn.L1Loss()}).compile(device='cpu')
        expected=(4*custom.network.amplitude/3-4/3).abs()
        torch.testing.assert_close(custom.compute_losses()['flow_rate'],expected)

    def test_invalid_quadrature_targets_and_unsupported_trainers(self):
        for quadrature in (None,[-1.,1.,1.],[0.,0.,0.],[1.,2.],[1.,float('nan'),1.],[True,True,True],[1j,1j,1j]):
            with self.subTest(quadrature=quadrature):
                model=make_trainer();condition=model.problem.constraints[-1];condition.quadrature_weights=quadrature
                with self.assertRaises(ValueError):model.compile(device='cpu')
                with self.assertRaises(RuntimeError):model.predict([[.5,1.]])
        for target in ([1.,2.],float('inf'),lambda x:0.,1j,True):
            model=make_trainer();model.problem.constraints[-1].target=target
            with self.assertRaises(ValueError):model.compile(device='cpu')
        model=make_trainer();model.problem.constraints[-1].measure=2
        with self.assertRaises(ValueError):model.compile(device='cpu')
        with self.assertRaisesRegex(ValueError,'research.Trainer'):PINN(model.problem,AxialVelocity()).compile(device='cpu')
        model=make_trainer();condition=model.problem.constraints[-1];condition.operator=lambda x,u:torch.cat([u,u],dim=1)
        with self.assertRaisesRegex(ValueError,'one scalar'):model.train(1,0)
        self.assertEqual(model.iter,0)
        for operator in (lambda x,u:0.,lambda x,u:u.to(torch.complex128)):
            model=make_trainer();model.problem.constraints[-1].operator=operator
            with self.assertRaisesRegex(ValueError,'real floating tensor'):model.train(1,0)
            self.assertEqual(model.iter,0)

    def continuation(self,device):
        full=make_trainer(device);full.train(7,0)
        split=make_trainer(device);split.train(3,0);state=split.training_state()
        stream=io.BytesIO();torch.save(state,stream);stream.seek(0)
        restored=make_trainer(device);restored.load_training_state(torch.load(stream,weights_only=True,map_location='cpu'));restored.train(4,0)
        self.assertEqual(full.loss_history,restored.loss_history)
        for key,value in full.network.state_dict().items():self.assertTrue(torch.equal(value,restored.network.state_dict()[key]))
        return state

    def test_cpu_resume_and_changed_weights_points_or_targets_are_rejected(self):
        state=self.continuation('cpu')
        for change in ('weights','points','target'):
            model=make_trainer()
            if change=='weights':model=make_trainer(quadrature=[.25,1.5,.25])
            elif change=='points':model._constraints[-1][1][1,1]=.9
            else:model=make_trainer(target=1.)
            before=deepcopy(model.network.state_dict())
            with self.assertRaises(ValueError):model.load_training_state(state)
            self.assertEqual(model.iter,0)
            for key,value in before.items():self.assertTrue(torch.equal(value,model.network.state_dict()[key]))
        # The compiled quadrature is a detached snapshot, independent of source edits.
        weights=torch.tensor([1/3,4/3,1/3],dtype=torch.float64,requires_grad=True)
        model=make_trainer(quadrature=weights);fingerprint=model.data_fingerprint()
        with torch.no_grad():weights.fill_(8.)
        self.assertEqual(model.data_fingerprint(),fingerprint);self.assertFalse(model._integral_weights['flow_rate'].requires_grad)
        precise=[.123456789012345,1.,.25]
        self.assertEqual(make_trainer(quadrature=precise)._integral_weights['flow_rate'][:,0].tolist(),precise)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_integral_gradient_and_exact_resume(self):
        self.continuation('cuda')
        model=make_trainer('cuda');terms=model.compute_losses();terms['flow_rate'].backward()
        self.assertTrue(model.network.amplitude.grad.is_cuda and torch.isfinite(model.network.amplitude.grad))
