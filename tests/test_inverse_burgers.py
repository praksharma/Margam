"""Joint coefficient gradients, data-driven training and strict exact continuation."""
from copy import deepcopy
from dataclasses import asdict
import tempfile
from pathlib import Path
import math
import unittest
import numpy as np
import torch
from margam.templates.inverse_burgers import inverse_example_config
from margam.templates.poisson import build_pinn
from margam.components.inverse_config import read_observations
from margam.methods.inverse_burgers import burgers_residual


class InverseBurgersTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def config(self):
        c=inverse_example_config();c.device='cpu';c.layers=(8,8);c.iterations=4;return c

    def test_physical_gradients_and_objective(self):
        c=self.config();m=build_pinn(c)
        points=torch.tensor([[.2,.3],[-.4,.5]],requires_grad=True);u=points[:,0:1]**2+points[:,1:2]
        residual=burgers_residual(points,u,m.network)
        nu=m.network.log_lambda_2.exp()
        expected=1+m.network.lambda_1*u*2*points[:,0:1]-nu*2
        torch.testing.assert_close(residual,expected)
        g1,g2=torch.autograd.grad(residual.square().mean(),(m.network.lambda_1,m.network.log_lambda_2))
        torch.testing.assert_close(g1,(2*expected*u*2*points[:,0:1]).mean())
        torch.testing.assert_close(g2,(2*expected*(-2*nu)).mean())
        terms=m.compute_losses();assert set(terms)=={'pde','observations'}
        total=terms['pde']+terms['observations'];total.backward()
        assert m.network.lambda_1.grad is not None and m.network.log_lambda_2.grad is not None
        assert m.network.lambda_1.grad.abs()>0 and m.network.log_lambda_2.grad.abs()>0
        points,values=read_observations(c)
        assert len(points)==2000 and values.shape==(2000,1)
        np.testing.assert_allclose(m.collocation_point_sample.numpy(),points,rtol=1e-6,atol=1e-7)
        assert [condition.name for condition in m.problem.constraints]==['observations']
        assert len(m.problem.constraints[0].points)==len(points)
        m.train(2,0);assert len(m.inverse_history)==2 and m.inverse_history[-1]['lambda_2']>0
        assert m.estimated_parameters()['lambda_1']!=0 and m.estimated_parameters()['lambda_2']!=math.exp(-6.)

    def test_exact_resume_prediction_and_rejections(self):
        c=self.config();whole=build_pinn(c);whole.train(4,0)
        a=build_pinn(c);a.train(2,0);state=a.training_state();b=build_pinn(c);b.load_training_state(state);b.train(2,0)
        assert whole.loss_history==b.loss_history and whole.inverse_history==b.inverse_history
        for key,value in whole.network.state_dict().items():torch.testing.assert_close(value,b.network.state_dict()[key],rtol=0,atol=0)
        from margam.components.prediction import prediction_model
        predictor=prediction_model(c,state)
        torch.testing.assert_close(predictor.predict([[0.,.2]]),a.predict([[0.,.2]]),rtol=0,atol=0)
        assert predictor.estimated_parameters()==a.estimated_parameters()
        before=deepcopy(b.training_state());bad=deepcopy(state);bad['inverse']['history'][-1]['lambda_2']=-1
        with self.assertRaises(ValueError):b.load_training_state(bad)
        assert b.iter==before['iter'] and b.inverse_history==before['inverse']['history']
        bad=deepcopy(state);identifier=next(iter(bad['optimiser']['state']));bad['optimiser']['state'][identifier]['exp_avg']=torch.ones(3)
        with self.assertRaises(ValueError):b.load_training_state(bad)
        assert b.iter==before['iter'] and b.inverse_history==before['inverse']['history']
        for key,value in before['network'].items():torch.testing.assert_close(value,b.network.state_dict()[key],rtol=0,atol=0)
        altered=deepcopy(c);altered.boundary_weight=2
        with self.assertRaises(ValueError):build_pinn(altered).load_training_state(state)
        altered=deepcopy(c);altered.inverse_settings['lambda_1']=.1
        with self.assertRaises(ValueError):build_pinn(altered).load_training_state(state)
        with tempfile.TemporaryDirectory() as directory:
            rows=np.genfromtxt(c.interior_data['path'],delimiter=',',names=True)
            rows['u'][0]+=.01
            path=Path(directory)/'changed.csv';np.savetxt(path,np.column_stack([rows[name] for name in ('x','t','u')]),delimiter=',',header='x,t,u',comments='')
            altered=deepcopy(c);altered.interior_data['path']=str(path)
            with self.assertRaises(ValueError):build_pinn(altered).load_training_state(state)
        bad=deepcopy(c);bad.loss_formulation='integral'
        with self.assertRaises(ValueError):bad.validate()
        bad=deepcopy(c);bad.interior_data['values']=''
        with self.assertRaises(ValueError):bad.validate()
        bad=deepcopy(c);bad.shape='Rectangle'
        with self.assertRaises(ValueError):bad.validate()


    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_joint_parameters_and_exact_resume(self):
        c=self.config();c.device='cuda';a=build_pinn(c);a.train(4,0)
        b=build_pinn(c);b.train(2,0);state=b.training_state();d=build_pinn(c);d.load_training_state(state);d.train(2,0)
        assert a.inverse_history==d.inverse_history and a.loss_history==d.loss_history
        assert d.network.lambda_1.device.type=='cuda' and d.network.log_lambda_2.device.type=='cuda'
        for key,value in a.network.state_dict().items():torch.testing.assert_close(value,d.network.state_dict()[key],rtol=0,atol=0)
