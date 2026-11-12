"""Wave initial velocity, analytical PDE identities and functional Jacobians."""
from copy import deepcopy
import unittest
import numpy as np
import torch
from examples.dsgnar_analytic import build,reference,forcing
from margam.methods.dsgnar import DSGNARPINN
from margam.templates.examples import example_config
from margam.templates.poisson import build_pinn
from margam.components.expressions import Expression


class AnalyticTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_analytical_equations_initial_velocity_and_no_final_boundary(self):
        for case in ('wave','multiscale'):
            c=example_config('dsgnar_'+case);c.device='cpu';c.points=9;c.boundary_points=4;c.layers=(4,);c.dsgnar_settings.update(sketch_size=8,batch_size=3,probes=3)
            model=build_pinn(c);x=torch.rand(11,2,dtype=torch.float64,requires_grad=True);u=reference(case,x)
            self.assertLess(model.problem.residual(x,u).abs().max().item(),1e-9)
            if case=='wave':
                self.assertEqual(model.names,['pde','initial','initial_velocity','boundary'])
                initial=next(row for row in model._constraints if row[0].name=='initial_velocity')
                points=initial[1].clone().requires_grad_(True);values=reference(case,points)
                torch.testing.assert_close(initial[0].operator(points,values),torch.zeros_like(values),rtol=0,atol=0)
                boundary=next(row for row in model._constraints if row[0].name=='boundary')[1]
                self.assertTrue(torch.all((boundary[:,0]==0)|(boundary[:,0]==1)))
                self.assertEqual(len(model.collocation_point_sample),9+3*4)
                c.bc_targets['bottom']='0'
                with self.assertRaisesRegex(ValueError,'owns its initial'):c.validate()
            else:
                expected=Expression(c.pde_forcing,2,target=True)(x)
                torch.testing.assert_close(expected,forcing(x),rtol=2e-14,atol=2e-12)

    def test_functional_generic_jacobians_and_checkpoint_continuation(self):
        for case,architecture in [('wave','fourier'),('multiscale','fourier'),('wave','siren')]:
            model=build(case,network=architecture,width=4,depth=2,points=7,boundary=5,sketch=8,batch=3,probes=3)
            self.assertEqual(model.network._margam_spec['type'],architecture)
            generic=DSGNARPINN(model.problem,deepcopy(model.network),weights=model.initial_weights,settings=model.settings,configuration=model.configuration).compile(device='cpu')
            for a,b in zip(model._conditions(),generic._conditions()):
                for (ja,ra),(jb,rb) in zip(model._jacobian_batches(a),generic._jacobian_batches(b)):
                    torch.testing.assert_close(ja,jb,rtol=2e-12,atol=2e-11);torch.testing.assert_close(ra,rb,rtol=2e-13,atol=2e-12)
            model.train(2,0)
            resumed=build(case,network=architecture,width=4,depth=2,points=7,boundary=5,sketch=8,batch=3,probes=3);resumed.load_training_state(model.training_state())
            model.train(1,0);resumed.train(1,0);torch.testing.assert_close(model._flat(),resumed._flat(),rtol=0,atol=0)
            changed=build(case,network=architecture,width=4,depth=2,points=8,boundary=5,sketch=8,batch=3,probes=3)
            with self.assertRaises(ValueError):changed.load_training_state(model.training_state())

    def test_numpy_torch_analytical_references_agree(self):
        x=np.random.default_rng(3).random((43,2))
        for case in ('wave','multiscale'):np.testing.assert_allclose(reference(case,x),reference(case,torch.tensor(x)).numpy(),rtol=3e-14,atol=3e-14)
