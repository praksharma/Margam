"""Concrete smooth-lid Navier–Stokes derivative and reference regressions."""
from copy import deepcopy
import importlib.util
import unittest
import numpy as np
import torch
from examples.dsgnar_cavity import CavitySiren,FunctionalCavityTrainer,make_problem,residuals,lid_target,sample_points
from margam.methods.dsgnar import DSGNARPINN
from margam.components.dsgnar_config import dsgnar_defaults


class CavityTests(unittest.TestCase):
    def test_five_conditions_and_functional_jacobian_match_generic(self):
        torch.set_num_threads(1)
        with torch.random.fork_rng():network=CavitySiren(4,2).double()
        interior,wall=sample_points(5,9,dtype=torch.float64)
        settings=dsgnar_defaults();settings.update(batch_size=3,sketch_size=8,probes=3)
        a=DSGNARPINN(make_problem(interior,wall),network,settings=settings).compile(device='cpu')
        b=FunctionalCavityTrainer(make_problem(interior,wall),deepcopy(network),settings=settings).compile(device='cpu')
        self.assertEqual(a.names,['pde','momentum_y','continuity','boundary_u','boundary_v'])
        for left,right in zip(a._conditions(),b._conditions()):
            for (ja,ra),(jb,rb) in zip(a._jacobian_batches(left),b._jacobian_batches(right)):
                torch.testing.assert_close(ja,jb,rtol=2e-12,atol=2e-12);torch.testing.assert_close(ra,rb,rtol=2e-13,atol=2e-13)
        a.train(2,0);b.train(2,0)
        torch.testing.assert_close(a.predict(interior),b.predict(interior),rtol=2e-10,atol=2e-10)
        resumed=FunctionalCavityTrainer(make_problem(interior,wall),deepcopy(network),settings=settings).compile(device='cpu')
        resumed.load_training_state(b.training_state());b.train(1,0);resumed.train(1,0)
        torch.testing.assert_close(b._flat(),resumed._flat(),rtol=0,atol=0)

    def test_manufactured_couette_pressure_gauge_and_lid(self):
        x=torch.tensor([[.2,.3],[.7,.8]],dtype=torch.float64,requires_grad=True)
        fields=torch.cat((x[:,1:2],x[:,0:1]*0,x[:,0:1]*0+3),1)
        for r in residuals(x,fields):torch.testing.assert_close(r,torch.zeros_like(r),atol=0,rtol=0)
        polynomial=torch.cat((x[:,0:1]**2*x[:,1:2],-x[:,0:1]*x[:,1:2]**2,x[:,0:1]**3+2*x[:,1:2]**3),1)
        mx,my,div=residuals(x,polynomial);a,b=x[:,0:1],x[:,1:2]
        torch.testing.assert_close(mx,a**3*b**2+3*a**2-.02*b,rtol=1e-13,atol=1e-13)
        torch.testing.assert_close(my,a**2*b**3+6*b**2+.02*a,rtol=1e-13,atol=1e-13)
        torch.testing.assert_close(div,torch.zeros_like(div),rtol=0,atol=0)
        shifted=fields+fields.new_tensor([0,0,7])
        for a,b in zip(residuals(x,fields),residuals(x,shifted)):torch.testing.assert_close(a,b,atol=0,rtol=0)
        walls=torch.tensor([[0,1],[1,1],[.5,1],[0,.5],[.5,0]],dtype=torch.float64)
        torch.testing.assert_close(lid_target(walls).flatten(),torch.tensor([0,0,1,0,0],dtype=torch.float64))
        for policy in ('uniform','corner_enriched'):
            p,w=sample_points(33,17,sampling=policy);self.assertEqual(p.shape,(33,2));self.assertEqual(w.shape,(17,2))
            self.assertTrue(torch.all(((w==0)|(w==1)).any(dim=1)));self.assertTrue(torch.all((p>=0)&(p<=1)))

    @unittest.skipUnless(importlib.util.find_spec('scipy'),'Optional SciPy reference solver required')
    def test_independent_poisson_and_steady_reference(self):
        from examples.dsgnar_cavity_reference import poisson,solve
        n=33;h=1/(n-1);x=np.linspace(0,1,n);xx,yy=np.meshgrid(x,x);exact=np.sin(np.pi*xx)*np.sin(np.pi*yy)
        # Discrete sine eigenfunction: test indexing, normalization and sign.
        omega=4*(1-np.cos(np.pi*h))/h**2*exact
        np.testing.assert_allclose(poisson(omega,h),exact,atol=5e-14,rtol=5e-14)
        result=solve(17,tolerance=1e-6,max_time=60)
        self.assertTrue(result['converged']);self.assertLess(result['steady_residual'],1e-6)
        np.testing.assert_allclose(result['u'][-1],4*x[::2]*(1-x[::2]),atol=0)
        self.assertLess(result['u'][8,8],0);self.assertLess(result['psi'][8,8],0)
