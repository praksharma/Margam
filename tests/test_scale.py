"""Scale equations, detached spatial derivatives and previous-iterate persistence."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import torch
from margam.templates.examples import example_config
from margam.templates.poisson import build_pinn,PoissonConfig
from margam.methods.scale import ScalePINN,spatial_laplacian


def small(key='scale_allen_cahn',device='cpu'):
    c=example_config(key);c.device=device;c.points=8;c.boundary_points=2;c.layers=(4,4);c.network_options['features']=2;c.iterations=4
    return c


class ScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_scalar_correction_excludes_time_and_detaches_old_parameters(self):
        c=small();c.scale_settings.update(tau_sc=2.,tau_alpha=4.,gamma=3.)
        class Polynomial(torch.nn.Module):
            def __init__(self,value):super().__init__();self.a=torch.nn.Parameter(torch.tensor(value,dtype=torch.float64))
            def forward(self,x):return self.a*(x[:,0:1]**2+x[:,1:2]**2)
        model=object.__new__(ScalePINN);model.config=c;model.options=c.scale_settings;model.previous=Polynomial(1.)
        for parameter in model.previous.parameters():parameter.requires_grad_(False)
        current=Polynomial(2.);x=torch.tensor([[.2,.3],[.4,.7]],dtype=torch.float64,requires_grad=True)
        value=model.correction(x,current(x));expected=(x[:,0:1]**2+x[:,1:2]**2)/2-1.5
        torch.testing.assert_close(value,expected)
        gradient=torch.autograd.grad(value.sum(),current.a)[0]
        torch.testing.assert_close(gradient,expected.sum());self.assertIsNone(model.previous.a.grad)
        torch.testing.assert_close(spatial_laplacian(x,current(x),(0,)),torch.full((2,1),4.,dtype=x.dtype))

    def test_flow_correction_maps_pressure_to_continuity_without_pressure_smoothing(self):
        c=small('scale_cavity_3200');c.scale_settings.update(tau_sc=2.,tau_alpha=4.,gamma=3.)
        class Polynomial(torch.nn.Module):
            def __init__(self,value):super().__init__();self.a=torch.nn.Parameter(torch.tensor(value,dtype=torch.float64))
            def forward(self,x):return self.a*torch.cat((x[:,0:1]**2,x[:,1:2]**2,x[:,0:1]**2+x[:,1:2]**2),1)
        model=object.__new__(ScalePINN);model.config=c;model.options=c.scale_settings;model.previous=Polynomial(1.)
        for parameter in model.previous.parameters():parameter.requires_grad_(False)
        x=torch.tensor([[.2,.3]],dtype=torch.float64,requires_grad=True);u=Polynomial(2.)(x)
        expected=torch.cat((x[:,0:1]**2/2-1.5,x[:,1:2]**2/2-1.5,(x[:,0:1]**2+x[:,1:2]**2)/2),1)
        torch.testing.assert_close(model.correction(x,u),expected)

    def test_zero_correction_value_still_has_stabilizing_gradient_at_first_step(self):
        c=small();c.scale_settings['fresh']=False;m=build_pinn(c)
        x=m.collocation_point_sample.detach().requires_grad_(True);correction=m.correction(x,m.network(x))
        torch.testing.assert_close(correction,torch.zeros_like(correction),atol=1e-6,rtol=0)
        gradients=torch.autograd.grad(correction.sum(),tuple(m.network.parameters()),allow_unused=True)
        self.assertTrue(any(g is not None and g.abs().sum()>0 for g in gradients))

    def test_disabled_correction_is_bitwise_baseline_and_snapshot_is_pre_update(self):
        c=small();c.scale_settings.update(strength=0.,fresh=False);base=deepcopy(c);base.method='baseline'
        m=build_pinn(c);reference=build_pinn(base);initial=deepcopy(m.network.state_dict());m.train(1,0)
        for name,value in initial.items():torch.testing.assert_close(value,m.previous.state_dict()[name],rtol=0,atol=0)
        previous=deepcopy(m.network.state_dict());m.train(2,0);reference.train(3,0)
        self.assertEqual(m.loss_history,reference.loss_history)
        for name,value in m.network.state_dict().items():torch.testing.assert_close(value,reference.network.state_dict()[name],rtol=0,atol=0)

    def test_named_fresh_samples_and_exact_resume_both_cases(self):
        for key in ('scale_allen_cahn','scale_cavity_3200'):
            with self.subTest(key=key):
                c=small(key);full=build_pinn(c);full.train(5,0)
                for phase in (0,1,3):
                    m=build_pinn(c);m.train(phase,0);n=build_pinn(c);n.load_training_state(m.training_state());n.train(5-phase,0)
                    self.assertEqual(full.loss_history,n.loss_history);self.assertEqual(full.diagnostics,n.diagnostics)
                    for name,value in full.previous.state_dict().items():torch.testing.assert_close(value,n.previous.state_dict()[name],rtol=0,atol=0)
                self.assertNotEqual(full.diagnostics[-1]['physical_pde'],full.diagnostics[-1]['corrected_pde'])

    def test_nonfinite_and_corrupt_previous_state_rejected(self):
        c=small();m=build_pinn(c);m.train(2,0);state=m.training_state()
        for change in (lambda s:s['scale']['previous'].pop(next(iter(s['scale']['previous']))),lambda s:s['scale']['settings']['configuration']['scale_settings'].update(tau_sc=0),lambda s:s['scale']['diagnostics'].pop()):
            bad=deepcopy(state);change(bad)
            with self.assertRaises(ValueError):build_pinn(c).load_training_state(bad)
        before=deepcopy(m.previous.state_dict());m.problem.residual=lambda x,u:u*float('nan')
        with self.assertRaises(FloatingPointError):m.train(1,0)
        self.assertEqual(m.iter,2)
        for name,value in before.items():torch.testing.assert_close(value,m.previous.state_dict()[name],rtol=0,atol=0)
        c=example_config('expert_torus');c.method='scale'
        with self.assertRaises(ValueError):build_pinn(c)

    def test_standard_flow_refreshes_component_boundary_coordinates(self):
        from margam.templates.poisson import BoundarySampling
        c=PoissonConfig(physics='incompressible_2d',method='scale',sampling='random',points=8,boundary_points=3,layers=(4,4),device='cpu')
        c.boundary_sampling={side:BoundarySampling(3,'random',i+1) for i,side in enumerate(c.sides)}
        m=build_pinn(c);before={condition.name:points.clone() for condition,points,_ in m._constraints};m.train(2,0)
        after={condition.name:points for condition,points,_ in m._constraints}
        self.assertFalse(torch.equal(before['left:u'],after['left:u']))
        torch.testing.assert_close(before['pressure_gauge'],after['pressure_gauge'],rtol=0,atol=0)

    def test_previous_expert_window_archives_migrate_scale_defaults(self):
        c=example_config('expert_burgers');c.device='cpu';c.points=8;c.boundary_points=2;c.layers=(4,4);c.network_options['features']=2
        c.expert_settings.update(chunks=2,windows=2,window_steps=2,balance_every=2)
        m=build_pinn(c);m.train(3,0);state=m.training_state();state['marching']['configuration'].pop('scale_settings')
        state['expert']['settings']['configuration'].pop('scale_settings')
        for previous in state['marching']['completed']:previous['expert']['settings']['configuration'].pop('scale_settings')
        n=build_pinn(c);n.load_training_state(state);m.train(1,0);n.train(1,0);self.assertEqual(m.loss_history,n.loss_history)

    def test_archive_and_legacy_defaults(self):
        from margam.ui.run_document import save_run,load_run
        from margam.ui.model_document import decode_document
        c=small();m=build_pinn(c);m.train(2,0)
        data=dict(format='margam-run',version=2,model=dict(format='margam-model',version=19,config=asdict(c),nodes=[],wires=[]),result=dict(outputs={},steps=2,cancelled=False,device='cpu'),monitors={},checkpoint=m.training_state())
        with TemporaryDirectory() as directory:
            path=Path(directory)/'scale.pt';save_run(path,data);loaded=load_run(path);n=build_pinn(c);n.load_training_state(loaded['checkpoint']);n.train(1,0);m.train(1,0);self.assertEqual(m.loss_history,n.loss_history)
        old=asdict(PoissonConfig());old.pop('scale_settings')
        decoded,_,_=decode_document(dict(format='margam-model',version=18,config=old,nodes=[],wires=[]));self.assertEqual(decoded.scale_settings['strength'],1.)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_derivatives_checkpoint_and_cpu_archive_validation(self):
        c=small(device='cuda');m=build_pinn(c);m.train(3,0);n=build_pinn(c);n.load_training_state(m.training_state());m.train(1,0);n.train(1,0)
        for name,value in m.network.state_dict().items():torch.testing.assert_close(value,n.network.state_dict()[name],rtol=1e-5,atol=1e-7)
        self.assertTrue(m.previous.affine_matrix.is_cuda)
        from margam.ui.run_document import save_run,load_run
        data=dict(format='margam-run',version=2,model=dict(format='margam-model',version=19,config=asdict(c),nodes=[],wires=[]),result=dict(outputs={},steps=4,cancelled=False,device='cuda'),monitors={},checkpoint=m.training_state())
        with TemporaryDirectory() as directory:
            path=Path(directory)/'scale-cuda.pt';save_run(path,data);loaded=load_run(path);reopened=build_pinn(c);reopened.load_training_state(loaded['checkpoint']);self.assertEqual(reopened.loss_history,m.loss_history)

