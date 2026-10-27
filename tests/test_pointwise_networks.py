"""Pointwise contracts, physical derivatives, source equations and restart identity."""
from copy import deepcopy
import math
import unittest
import torch
from margam.components.network_config import NETWORK_NAMES,network_options,validate_network
from margam.nn.factory import build_network
from margam.nn import FullyConnected
from margam.templates.poisson import PoissonConfig,build_pinn


class PointwiseNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_shapes_batch_independence_and_second_derivatives(self):
        for kind in NETWORK_NAMES:
            with self.subTest(network=kind):
                c=PoissonConfig(network_type=kind,layers=(4,4),physics='incompressible_2d')
                net=build_network(c).double()
                x=torch.tensor([[.21,.37],[.72,.63]],dtype=torch.float64,requires_grad=True)
                self.assertEqual(net(x).shape,(2,3))
                torch.testing.assert_close(net(x),torch.cat([net(row[None]) for row in x]),rtol=1e-9,atol=1e-9)
                self.assertTrue(torch.autograd.gradcheck(net,(x,),eps=1e-6,atol=1e-4,rtol=1e-3))
                self.assertTrue(torch.autograd.gradgradcheck(net,(x,),eps=1e-6,atol=2e-4,rtol=2e-3))
                first=torch.autograd.grad(net(x).sum(),x,create_graph=True)[0]
                second=torch.autograd.grad(first.sum(),x,create_graph=True)[0]
                self.assertTrue(torch.isfinite(second).all())

    def test_original_fcnn_is_bitwise_unchanged(self):
        c=PoissonConfig(shape='Interval',network_seed=23)
        torch.manual_seed(23);old=FullyConnected([1,16,16,1],'tanh','Xavier normal',input_bounds=c.bounds)
        new=build_network(c)
        self.assertEqual(set(old.state_dict()),set(new.state_dict()))
        for key,value in old.state_dict().items():torch.testing.assert_close(value,new.state_dict()[key],rtol=0,atol=0)
        x=torch.linspace(0,1,17)[:,None]
        torch.testing.assert_close(old(x),new(x),rtol=0,atol=0)

    def test_fourier_basis_and_physical_siren_derivative(self):
        from margam.nn.fourier import FourierFeatures,AngularFeatures
        from margam.nn.siren import SirenNetwork
        basis=FourierFeatures(2,'axis',[0.,1.]).double()
        expected=torch.tensor([[0.,0.,1.,0.],[0.,0.,0.,1.]],dtype=torch.float64)
        torch.testing.assert_close(basis.frequencies,expected)
        x=torch.tensor([[.2,.3]],dtype=torch.float64)
        phase=2*torch.pi*(x@expected)
        torch.testing.assert_close(basis(x),torch.cat((phase.sin(),phase.cos()),-1))
        full=FourierFeatures(2,'full',[0.,1.]);self.assertEqual(full.output_size,8)
        diagonal=FourierFeatures(2,'diagonal',[0.,1.]);self.assertEqual(diagonal.output_size,8)
        angular=AngularFeatures(2,3).double();self.assertEqual(angular(x).shape,(1,12))
        net=SirenNetwork(1,1,(1,),([0.],[2.]),dict(first_omega=2.,omega=30.)).double()
        with torch.no_grad():
            net.layers[0].weight.fill_(.25);net.layers[0].bias.zero_()
            net.layers[1].weight.fill_(2.);net.layers[1].bias.fill_(.1)
        x=torch.tensor([[.2],[.8]],dtype=torch.float64,requires_grad=True)
        first=torch.autograd.grad(net(x).sum(),x,create_graph=True)[0]
        second=torch.autograd.grad(first.sum(),x)[0]
        torch.testing.assert_close(net(x),2*torch.sin(.25*x)+.1)
        torch.testing.assert_close(second,-.125*torch.sin(.25*x))

    def test_highway_modified_and_dgm_gate_equations(self):
        def scalar(kind):return build_network(PoissonConfig(shape='Interval',network_type=kind,layers=(1,1))).double()
        x=torch.tensor([[.2],[.7]],dtype=torch.float64)
        net=scalar('highway_fourier')
        with torch.no_grad():
            for p in net.parameters():p.zero_()
            net.transform.bias.fill_(math.log(.25/.75));net.carry.bias.fill_(.4)
            for layer in net.trunk.layers:layer.bias.fill_(.5)
            net.output.weight.fill_(1.)
        torch.testing.assert_close(net(x),torch.full_like(x,.25*math.tanh(.5)+.75*.4))
        net=scalar('modified_fourier')
        with torch.no_grad():
            for p in net.parameters():p.zero_()
            net.encoder_u.bias.fill_(math.atanh(.2));net.encoder_v.bias.fill_(math.atanh(.4))
            net.trunk.layers[1].bias.fill_(math.atanh(.3));net.output.weight.fill_(1.)
        torch.testing.assert_close(net(x),torch.full_like(x,.26))
        net=scalar('dgm')
        with torch.no_grad():
            for p in net.parameters():p.zero_()
            net.start.weight.fill_(1.);net.output.weight.fill_(1.)
            cell=net.cells[0]
            for key,value in [('z',.2),('g',.3),('r',.4)]:cell[key][0].bias.fill_(math.atanh(value))
            cell['h'][0].weight.fill_(1.);cell['h'][1].weight.fill_(2.)
        expected=.7*torch.tanh(x+.8*torch.tanh(x))+.2*torch.tanh(x)
        torch.testing.assert_close(net(x),expected)

    def test_gabor_is_smooth_at_its_centres(self):
        from margam.nn.multiplicative_filter import GaborFilter
        layer=GaborFilter(2,1,2,dict(input_scale=1.,alpha=6.,beta=1.)).double()
        x=layer.center.detach().T.clone().requires_grad_(True)
        self.assertTrue(torch.autograd.gradgradcheck(layer,(x,),atol=1e-4))

    def test_checkpoint_exact_resume_and_setting_mismatch(self):
        for kind in NETWORK_NAMES:
            with self.subTest(network=kind):
                config=PoissonConfig(shape='Interval',network_type=kind,layers=(8,8),method='self_adaptive',lr=.0005)
                a=build_pinn(config);a.train(3,display_every=0);state=a.training_state()
                b=build_pinn(config);b.load_training_state(state)
                a.train(2,display_every=0);b.train(2,display_every=0)
                self.assertEqual(a.loss_history,b.loss_history)
                torch.testing.assert_close(a.predict(torch.tensor([[.3],[.7]])),b.predict(torch.tensor([[.3],[.7]])),rtol=0,atol=0)
                self.assertEqual(state['network_spec']['type'],kind)
        config=PoissonConfig(shape='Interval',network_type='fourier',network_options={'frequencies':[0.,1.]})
        a=build_pinn(config);a.train(2,display_every=0);state=a.training_state()
        config.network_options={'frequencies':[0.,2.]}
        with self.assertRaisesRegex(ValueError,'architecture/settings'):build_pinn(config).load_training_state(state)
        state.pop('network_spec')
        with self.assertRaisesRegex(ValueError,'requires a network settings'):a.load_training_state(state)
        legacy=build_pinn(PoissonConfig(shape='Interval'));state=legacy.training_state();state.pop('network_spec')
        build_pinn(PoissonConfig(shape='Interval')).load_training_state(state)

    def test_real_flow_and_hard_boundary_integration(self):
        for kind in NETWORK_NAMES:
            for method in ('baseline','self_adaptive'):
                with self.subTest(network=kind,method=method):
                    config=PoissonConfig(physics='incompressible_2d',method=method,network_type=kind,layers=(4,4),points=9,boundary_points=4,lr=.0001)
                    model=build_pinn(config);model.train(2,display_every=0)
                    values=model.predict(torch.tensor([[.2,.3],[.7,.8]]))
                    self.assertEqual(values.shape,(2,3));self.assertTrue(torch.isfinite(values).all())
        config=PoissonConfig(shape='Interval',method='hard_dirichlet',network_type='dgm',layers=(8,8))
        model=build_pinn(config);model.train(3,display_every=0)
        torch.testing.assert_close(model.predict(torch.tensor([[0.],[1.]])),torch.zeros(2,1),rtol=0,atol=0)
        state=model.training_state();self.assertEqual(state['network_spec']['type'],'dgm')
        build_pinn(config).load_training_state(state)
        config.network_options={'activation':'silu'}
        with self.assertRaisesRegex(ValueError,'architecture/settings'):build_pinn(config).load_training_state(state)

    def test_smooth_activations_encodings_and_skip_branches(self):
        cases=[]
        for kind in ('fourier','highway_fourier','modified_fourier'):
            for encoding in ('gaussian','full','diagonal'):
                cases.append((kind,{'encoding':encoding,'features':3,'frequencies':[0.,1.],'activation':'silu','skip_connections':True}))
        cases.append(('highway_fourier',{'transform_fourier':False,'project_fourier':True,'activation':'sin'}))
        cases.append(('multiscale_fourier',{'features':3,'activation':'sin','skip_connections':True}))
        cases.append(('dgm',{'activation':'sin'}))
        for kind,options in cases:
            with self.subTest(network=kind,options=options):
                net=build_network(PoissonConfig(network_type=kind,network_options=options,layers=(4,4,4,4,4))).double()
                x=torch.tensor([[.23,.41]],dtype=torch.float64,requires_grad=True)
                first=torch.autograd.grad(net(x).sum(),x,create_graph=True)[0]
                second=torch.autograd.grad(first.sum(),x,create_graph=True)[0]
                self.assertTrue(torch.isfinite(second).all())
                gradients=torch.autograd.grad(second.square().sum(),list(net.parameters()),allow_unused=True)
                self.assertTrue(all(g is None or torch.isfinite(g).all() for g in gradients))

    def test_options_reject_operators_and_invalid_shapes(self):
        for kind,options in [('fno',{}),('siren',{'omega':0.}),('fourier',{'encoding':'FFT'}),('multiscale_fourier',{'scales':[1.]*5}),('fourier',{'frequencies':[float('nan')]}),('gabor_filter',{'beta':-1.}),('fcnn',{'activation':'relu'})]:
            with self.subTest(kind=kind),self.assertRaises(ValueError):network_options(kind,options)
        with self.assertRaisesRegex(ValueError,'equal hidden'):validate_network('dgm',{},(8,16))
        with self.assertRaisesRegex(ValueError,'equal hidden'):validate_network('fourier',{'skip_connections':True},(8,16,8))
