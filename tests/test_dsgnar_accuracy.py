"""Check the independent benchmark reference and paper-sized periodic network."""
import importlib.util
import unittest
import numpy as np
import torch
from examples.dsgnar_accuracy import burgers_reference,PeriodicMLP


class AccuracyReferenceTests(unittest.TestCase):
    def test_periodic_network_parameter_count_and_derivative_matching(self):
        with torch.random.fork_rng():network=PeriodicMLP().double()
        self.assertEqual(sum(p.numel() for p in network.parameters()),1447)
        left=torch.tensor([[-1.,.3]],dtype=torch.float64,requires_grad=True)
        right=torch.tensor([[1.,.3]],dtype=torch.float64,requires_grad=True)
        a,b=network(left),network(right)
        torch.testing.assert_close(a,b,atol=1e-14,rtol=1e-14)
        da=torch.autograd.grad(a,left)[0];db=torch.autograd.grad(b,right)[0]
        torch.testing.assert_close(da,db,atol=1e-13,rtol=1e-13)

    def test_functional_burgers_jacobian_matches_generic_parameter_gradients(self):
        from copy import deepcopy
        from examples.dsgnar_accuracy import FunctionalBurgersTrainer
        from margam.methods.dsgnar import DSGNARPINN
        from margam.components import PINNProblem,PointConstraint
        from margam.components.differential import derivative
        from margam.components.dsgnar_config import dsgnar_defaults
        torch.set_num_threads(1)
        with torch.random.fork_rng():network=PeriodicMLP(width=4,depth=2).double()
        points=torch.tensor([[-.8,.1],[.2,.6],[.9,.8]],dtype=torch.float64)
        initial=points.clone();initial[:,1]=0
        def residual(x,u):return derivative(u,x,1)+u*derivative(u,x,0)-.01*derivative(u,x,0,2)
        problem=PINNProblem(residual,points,[PointConstraint('initial',initial,lambda x:-torch.sin(torch.pi*x[:,0:1]))])
        settings=dsgnar_defaults();settings['batch_size']=2
        generic=DSGNARPINN(problem,network,settings=settings).compile(device='cpu')
        functional=FunctionalBurgersTrainer(problem,deepcopy(network),settings=settings).compile(device='cpu')
        for a,b in zip(generic._conditions(),functional._conditions()):
            left=list(generic._jacobian_batches(a));right=list(functional._jacobian_batches(b))
            for before,after in zip(generic._batches(a),functional._batches(b)):torch.testing.assert_close(before,after,rtol=1e-12,atol=1e-12)
            self.assertEqual(len(left),len(right))
            for (ja,ra),(jb,rb) in zip(left,right):
                torch.testing.assert_close(ja,jb,atol=1e-12,rtol=1e-12)
                torch.testing.assert_close(ra,rb,atol=1e-13,rtol=1e-13)

    @unittest.skipUnless(importlib.util.find_spec('scipy'),'Optional SciPy reference quadrature required')
    def test_cole_hopf_initial_periodic_quadrature_and_pde(self):
        x=np.linspace(-1,1,101);points=np.column_stack((x,np.zeros_like(x)))
        np.testing.assert_allclose(burgers_reference(points),-np.sin(np.pi*x),atol=1e-14,rtol=1e-14)
        points=np.array([[-1.,.6],[1.,.6],[.2,.4],[-.6,.7]])
        reference=burgers_reference(points);self.assertAlmostEqual(reference[0],reference[1],places=14)
        np.testing.assert_allclose(reference,burgers_reference(points,512),atol=3e-14,rtol=3e-14)
        x,t=.35,.4;h=1e-4
        values=burgers_reference(np.array([[x,t],[x+h,t],[x-h,t],[x,t+h],[x,t-h]]))
        u,plus,minus,forward,backward=values
        residual=(forward-backward)/(2*h)+u*(plus-minus)/(2*h)-.01*(plus-2*u+minus)/h**2
        self.assertLess(abs(residual),2e-6)
