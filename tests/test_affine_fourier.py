"""Notebook affine inputs, trainable sine maps and physical-coordinate derivatives."""
from copy import deepcopy
import math,unittest
import torch
from margam.components.affine_inputs import affine_map
from margam.templates.poisson import PoissonConfig,build_pinn
from margam.templates.scale_examples import scale_example_config
from margam.nn.factory import build_network


class AffineFourierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_notebook_maps_and_raw_coordinate_units(self):
        for key,point,expected,cycles in [
            ('scale_cavity_3200',[.2,.7],[.2,.7,-.8,-.3],1.),
            ('scale_allen_cahn',[.2,.7],[.7,.2],1.),
            ('scale_kdv',[.2,.7],[.7,.2,-.3],1.),
            ('scale_ks',[math.pi,.1],[.1,.5,-.3],2.)]:
            with self.subTest(key=key):
                c=scale_example_config(key);n=build_network(c).double()
                self.assertEqual(c.network_type,'affine_fourier');self.assertEqual(n.cycles,cycles);self.assertFalse(n.normalize)
                torch.testing.assert_close(n.affine_inputs(torch.tensor([point],dtype=torch.float64)),torch.tensor([expected],dtype=torch.float64),rtol=1e-7,atol=1e-7)
        matrix,offset=affine_map('t/50; (x-1)/4; (x-2)/4; (x-3)/4; y; y-1',('x','y','t'))
        row=torch.tensor([.5,.2,25.],dtype=torch.float64)@torch.tensor(matrix,dtype=torch.float64).T+torch.tensor(offset,dtype=torch.float64)
        torch.testing.assert_close(row,torch.tensor([.5,-.125,-.375,-.625,.2,-.8],dtype=torch.float64))
        for text in ('x*x','sin(x)','x/y','x/0','__import__("os")','x**2','x[0]','x;','1e309','x; '*17):
            with self.subTest(text=text),self.assertRaises(ValueError):affine_map(text,('x','y'))
        c=PoissonConfig(shape='Interval',network_type='affine_fourier',network_options={'input_transform':'t'})
        with self.assertRaisesRegex(ValueError,'Unknown affine coordinate'):c.validate()

    def test_trainable_projection_and_fourth_physical_derivatives(self):
        for normalized in (False,True):
            c=PoissonConfig(shape='Interval',x_min=2.,x_max=6.,network_type='affine_fourier',network_options=dict(input_transform='x-1',features=1,cycles=.25,normalize_inputs=normalized),layers=(4,4))
            n=build_network(c).double()
            with torch.no_grad():n.projection.weight.fill_(.7);n.projection.bias.fill_(.2)
            x=torch.tensor([[2.3],[4.1]],dtype=torch.float64,requires_grad=True)
            argument=(x-2)/4 if normalized else x
            phase=(2*math.pi*.25)*(.7*(argument-1)+.2);frequency=2*math.pi*.25*.7/(4 if normalized else 1)
            value=n.features(x);torch.testing.assert_close(value,phase.sin())
            for order in range(1,5):
                value=torch.autograd.grad(value.sum(),x,create_graph=True)[0]
                torch.testing.assert_close(value,frequency**order*torch.sin(phase+order*math.pi/2),rtol=1e-10,atol=1e-10)
        c=PoissonConfig(shape='Interval',network_type='affine_fourier',network_options={'input_transform':'x; x-1','features':4},points=8,layers=(4,4))
        m=build_pinn(c);initial=m.network.projection.weight.detach().clone();m.train(2,0)
        self.assertFalse(torch.equal(initial,m.network.projection.weight));state=m.training_state()
        restored=build_pinn(c);restored.load_training_state(state);m.train(1,0);restored.train(1,0);self.assertEqual(m.loss_history,restored.loss_history)
        c.network_options['input_transform']='x; x-2'
        with self.assertRaises(ValueError):build_pinn(c).load_training_state(state)
        # Historical fixed-Fourier Scale configs remain explicit and load unchanged.
        old=scale_example_config('scale_cavity_400');old.network_type='fourier';old.network_options=dict(encoding='gaussian',features=2,scale=1/(2*math.pi));old.layers=(4,4);old.points=8;old.boundary_points=2
        before=build_pinn(old);before.train(1,0);after=build_pinn(old);after.load_training_state(before.training_state())
        torch.testing.assert_close(before.network.features.frequencies,after.network.features.frequencies,rtol=0,atol=0)
