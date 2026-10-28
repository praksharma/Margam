"""High-Re case definitions, KdV mathematics and periodic derivative matching."""
import math
from copy import deepcopy
from tempfile import TemporaryDirectory
from pathlib import Path
import unittest
import torch
from margam.templates.scale_examples import SCALE_EXAMPLES,CAVITY_REYNOLDS,scale_example_config
from margam.templates.poisson import build_pinn
from margam.templates.kdv import residual,samples,conditions
from margam.components.outputs import evaluate_outputs,OutputRequest
from margam.components.paraview_export import export_spacetime_pvd


class ScaleExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_reynolds_and_smoothing_are_consistent_independent_presets(self):
        self.assertEqual(CAVITY_REYNOLDS,(400,3200,7500,10000,20000))
        for re in CAVITY_REYNOLDS:
            c=scale_example_config('scale_cavity_'+str(re));c.validate()
            self.assertEqual(c.viscosity,1/re);self.assertEqual(c.scale_settings['gamma'],1/re)
            self.assertEqual(c.bc_values['top'],1.);self.assertEqual(c.bounds,([0.,0.],[1.,1.]))
        c=scale_example_config('scale_ks');self.assertEqual(c.y_max,.4)
        self.assertEqual(c.source,100/16);self.assertEqual(c.viscosity,100/16**2);self.assertEqual(c.fourth_order,100/16**4)

    def test_kdv_third_derivative_sign_and_physical_time(self):
        c=scale_example_config('scale_kdv');c.validate()
        x=torch.tensor([[.2,.3],[-.7,.6]],dtype=torch.float64,requires_grad=True)
        u=torch.sin(x[:,0:1])+x[:,1:2]
        expected=1+u*torch.cos(x[:,0:1])-c.viscosity*torch.cos(x[:,0:1])
        torch.testing.assert_close(residual(c,x,u),expected,rtol=1e-12,atol=1e-12)
        self.assertEqual(c.viscosity,.022**2);self.assertEqual(c.scale_settings['gamma'],.022)

    def test_kdv_initial_and_all_periodic_derivatives(self):
        c=scale_example_config('scale_kdv');c.boundary_points=5
        class Analytic(torch.nn.Module):
            def forward(self,x):return torch.cos(torch.pi*x[:,0:1])*torch.exp(-x[:,1:2])
        network=Analytic();points,sets=samples(c,0)
        self.assertTrue(all(row[-1]==0. for row in sets['initial']))
        self.assertEqual(set(sets),{'initial','periodic_x'});self.assertNotEqual(points,samples(c,1)[0])
        constraints=conditions(c,network,sets)
        x=constraints[1].points.double().requires_grad_(True)
        error=constraints[1].operator(x,network(x));self.assertEqual(error.shape,(5,3))
        torch.testing.assert_close(error,torch.zeros_like(error),atol=1e-12,rtol=0)
        x=constraints[0].points.double();torch.testing.assert_close(constraints[0].target(x),torch.cos(torch.pi*x[:,0:1]))

    def test_every_case_trains_resumes_and_exports(self):
        with TemporaryDirectory() as directory:
            for key in SCALE_EXAMPLES:
                with self.subTest(key=key):
                    c=scale_example_config(key);c.device='cpu';c.points=8;c.boundary_points=2;c.layers=(4,4);c.network_options['features']=2
                    full=build_pinn(c);full.train(4,0);part=build_pinn(c);part.train(2,0)
                    resumed=build_pinn(c);resumed.load_training_state(part.training_state());resumed.train(2,0)
                    self.assertEqual(full.loss_history,resumed.loss_history)
                    for name,value in full.network.state_dict().items():torch.testing.assert_close(value,resumed.network.state_dict()[name],rtol=0,atol=0)
                    data=evaluate_outputs(full,c,[OutputRequest('out','paraview',3)])['out']
                    if c.is_time_dependent:
                        path=Path(directory)/(key+'.pvd');export_spacetime_pvd(path,data);self.assertTrue(path.exists())
                    self.assertTrue(all(math.isfinite(v) for row in full.loss_history for v in row.values()))

    def test_kdv_rejects_undefined_training_contracts(self):
        for attribute,value in (('viscosity',0.),('shape','Rectangle'),('method','expert'),('sampling','grid')):
            c=scale_example_config('scale_kdv');setattr(c,attribute,value)
            with self.assertRaises(ValueError):build_pinn(c)
