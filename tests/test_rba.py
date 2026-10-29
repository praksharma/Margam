"""RBA recurrence, detached weighting, identity, failure atomicity and checkpoints."""
from copy import deepcopy
from dataclasses import asdict
from tempfile import TemporaryDirectory
from pathlib import Path
import unittest,torch
from margam.components import PINNProblem,PointConstraint
from margam.methods.rba import ResidualAttentionPINN
from margam.templates.examples import example_config
from margam.templates.poisson import build_pinn,PoissonConfig
from margam.components.rba_config import rba_defaults


def small(key='rba_allen_cahn',mode='zero',device='cpu'):
    c=example_config(key);c.points=8;c.boundary_points=2;c.layers=(4,4);c.device=device;c.rba_settings['initialization']=mode
    if c.network_type!='fcnn':c.network_options['features']=2
    return c


class RBATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_exact_recurrence_initialization_detach_and_squared_weight(self):
        n=torch.nn.Linear(1,1).double()
        with torch.no_grad():n.weight.fill_(1.);n.bias.fill_(1.)
        x=torch.tensor([[1.],[2.]],dtype=torch.float64);settings=dict(decay=.5,rate=.2,initialization='zero')
        m=ResidualAttentionPINN(PINNProblem(lambda x,u:u,x,[PointConstraint('bc',x,0.)]),n,settings=settings,weights={'pde':2.,'bc':3.}).compile(device='cpu')
        error=n(x);candidate=m.candidate_attention(error)
        expected=torch.tensor([[.2*2/3],[.2]],dtype=torch.float64);torch.testing.assert_close(candidate,expected)
        self.assertFalse(candidate.requires_grad);loss=(candidate*error).square().mean()
        grad=torch.autograd.grad(loss,n.weight)[0];torch.testing.assert_close(grad,(2*expected.square()*error.detach()*x).mean().reshape(1,1))
        raw=m.compute_losses();m.train(1,0);torch.testing.assert_close(m.attention,expected)
        self.assertAlmostEqual(m.loss_history[0]['total'],2*loss.item()+3*raw['bc'].item())
        self.assertAlmostEqual(m.loss_history[0]['bc'],raw['bc'].item());self.assertFalse(hasattr(m,'adaptive_optimiser'))
        normalized=deepcopy(settings);normalized['initialization']='normalized'
        other=ResidualAttentionPINN(PINNProblem(lambda x,u:u,x),torch.nn.Linear(1,1).double(),settings=normalized).compile(device='cpu')
        first=other.candidate_attention(torch.tensor([[2.],[3.]],dtype=torch.float64));torch.testing.assert_close(first,torch.tensor([[2/3],[1.]],dtype=torch.float64))
        other.attention=first;other.iter=1
        torch.testing.assert_close(other.candidate_attention(torch.zeros_like(first)),.5*first)

    def test_exact_resume_both_modes_examples_and_all_networks(self):
        for key in ('rba_allen_cahn','rba_helmholtz_2d'):
            for mode in ('zero','normalized'):
                c=small(key,mode);full=build_pinn(c);full.train(5,0)
                for phase in (0,1,3):
                    with self.subTest(key=key,mode=mode,phase=phase):
                        partial=build_pinn(c);partial.train(phase,0);resumed=build_pinn(c);resumed.load_training_state(partial.training_state());resumed.train(5-phase,0)
                        self.assertEqual(full.loss_history,resumed.loss_history);torch.testing.assert_close(full.attention,resumed.attention,rtol=0,atol=0)
                        for name,value in full.network.state_dict().items():torch.testing.assert_close(value,resumed.network.state_dict()[name],rtol=0,atol=0)
        from margam.components.network_config import NETWORK_NAMES
        for kind in NETWORK_NAMES:
            with self.subTest(network=kind):build_pinn(PoissonConfig(shape='Interval',method='rba',network_type=kind,points=8,layers=(4,4))).train(1,0)
        from margam.templates.prepared_triangle import prepared_triangle_config
        c=prepared_triangle_config();c.method='rba';c.layers=(4,4);build_pinn(c).train(2,0)

    def test_mask_identity_checkpoint_corruption_and_moving_points_rejected(self):
        c=small('rba_helmholtz_2d');m=build_pinn(c);m.train(2,0);state=m.training_state()
        for mutate in (lambda s:s['rba']['attention'].fill_(float('nan')),lambda s:s['rba']['attention'].fill_(100.),lambda s:s['rba'].update(attention=torch.zeros(1,1)),lambda s:s['rba']['settings']['settings'].update(rate=.2)):
            bad=deepcopy(state);mutate(bad)
            with self.assertRaises(ValueError):build_pinn(c).load_training_state(bad)
        m.collocation_point_sample=m.collocation_point_sample.flip(0)
        with self.assertRaisesRegex(ValueError,'identities'):m.train(1,0)
        from margam.components.adaptive_sampling import sampling_defaults
        for kind in ('rar','importance'):
            bad=deepcopy(c);bad.adaptive_sampling=sampling_defaults(kind)
            with self.assertRaisesRegex(ValueError,'fixed points'):bad.validate()
        bad=deepcopy(c);bad.loss_formulation='integral'
        with self.assertRaises(ValueError):bad.validate()
        bad=deepcopy(c);bad.rba_settings['decay']=1.
        with self.assertRaises(ValueError):bad.validate()

    def test_nonfinite_gradient_does_not_commit_attention_or_update(self):
        class Zero(torch.nn.Module):
            def __init__(self):super().__init__();self.a=torch.nn.Parameter(torch.tensor(0.))
            def forward(self,x):return self.a*x
        n=Zero();m=ResidualAttentionPINN(PINNProblem(lambda x,u:u.sqrt(),torch.ones(2,1)),n,settings=rba_defaults()).compile(device='cpu')
        with self.assertRaises(FloatingPointError):m.train(1,0)
        self.assertEqual(m.iter,0);self.assertEqual(m.attention.sum().item(),0.);self.assertEqual(n.a.item(),0.)
        # Perfect zero residuals remain finite and do not invent nonzero masks.
        n=Zero();m=ResidualAttentionPINN(PINNProblem(lambda x,u:0*u,torch.ones(2,1)),n,settings=rba_defaults()).compile(device='cpu');m.train(2,0)
        self.assertEqual(m.attention.sum().item(),0.);self.assertEqual(m.loss_history[-1]['total'],0.)

    def test_legacy_other_method_signatures_migrate_unused_rba_setting(self):
        for key in ('scale_cavity_400','expert_burgers','gpinn_burgers'):
            c=example_config(key);c.points=8;c.boundary_points=2;c.layers=(4,4)
            if c.network_type!='fcnn':c.network_options['features']=2
            if c.method=='expert':c.expert_settings.update(chunks=2,balance_every=2,windows=2,window_steps=2)
            m=build_pinn(c);m.train(3,0);state=m.training_state()
            def legacy(value):
                if isinstance(value,dict):
                    value.pop('rba_settings',None)
                    for child in value.values():legacy(child)
                elif isinstance(value,(list,tuple)):
                    for child in value:legacy(child)
            legacy(state);resumed=build_pinn(c);resumed.load_training_state(state);m.train(1,0);resumed.train(1,0);self.assertEqual(m.loss_history,resumed.loss_history)

        # Low-level gPINN API without a serialized template remains resumable too.
        from margam.methods.gpinn import GradientEnhancedPINN
        from margam.components.gpinn_config import gpinn_defaults
        problem=PINNProblem(lambda x,u:u,torch.ones(2,1))
        direct=GradientEnhancedPINN(problem,torch.nn.Linear(1,1),coordinates=('x',),gradient_weights=gpinn_defaults()).compile(device='cpu');direct.train(1,0)
        other=GradientEnhancedPINN(problem,torch.nn.Linear(1,1),coordinates=('x',),gradient_weights=gpinn_defaults()).compile(device='cpu');other.load_training_state(direct.training_state());self.assertEqual(other.loss_history,direct.loss_history)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_continuation_and_cpu_archive_identity(self):
        from margam.ui.run_document import save_run,load_run
        c=small(device='cuda');m=build_pinn(c);m.train(3,0);n=build_pinn(c);n.load_training_state(m.training_state());m.train(2,0);n.train(2,0)
        self.assertEqual(m.loss_history,n.loss_history);torch.testing.assert_close(m.attention,n.attention,rtol=0,atol=0)
        state=m.training_state();cpu=deepcopy(c);cpu.device='cpu';resumed=build_pinn(cpu);resumed.load_training_state(state);resumed.train(1,0)
        data=dict(format='margam-run',version=3,model=dict(format='margam-model',version=23,config=asdict(c),nodes=[],wires=[]),result=dict(outputs={},steps=5,cancelled=False,device='cuda'),monitors={},checkpoint=state)
        with TemporaryDirectory() as directory:
            path=Path(directory)/'rba.pt';save_run(path,data);loaded=load_run(path);self.assertEqual(loaded['checkpoint']['rba']['attention'].device.type,'cpu')
