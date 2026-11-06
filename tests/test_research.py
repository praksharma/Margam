"""External-code replacement points, mathematical equivalence and safe continuation."""
from copy import deepcopy
from functools import partial
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import torch
from margam.methods.pinn import PINN
from margam.research import Trainer, PINNProblem, PointConstraint, derivative


def network(dropout=False):
    return torch.nn.Sequential(torch.nn.Linear(1,8),torch.nn.Tanh(),
        *([torch.nn.Dropout(.2)] if dropout else []),torch.nn.Linear(8,1))


def problem():
    return PINNProblem(lambda x,u:derivative(u,x,0,2)+2,
        lambda:torch.linspace(0,1,18)[1:-1,None],
        [PointConstraint('boundary',torch.tensor([[0.],[1.]]))])


def robust(prediction,target):
    error=prediction-target
    return error.square().mean()+.01*error.pow(4).mean()


def build(factory=torch.optim.Adam,device='cpu',dropout=False):
    torch.manual_seed(17)
    return Trainer(problem(),network(dropout).double(),loss=robust,checkpoint_id='research-test-v1',
                   weights={'boundary':2}).compile(optimizer=factory,lr=.003,device=device)


class ResearchTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_default_matches_existing_engine_exactly(self):
        torch.manual_seed(7);a=network();b=deepcopy(a)
        baseline=PINN(problem(),a,weights={'boundary':3}).compile(lr=.002,device='cpu')
        research=Trainer(problem(),b,weights={'boundary':3}).compile(lr=.002,device='cpu')
        baseline.train(5,0);research.train(5,0)
        self.assertEqual(baseline.loss_history,research.loss_history)
        for x,y in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(x,y))
        self.assertIs(research.optimizer,research.optimiser)

    def test_raw_sampling_and_custom_loss_optimizer_match_pytorch(self):
        calls=[];points=torch.tensor([[.2],[.7]],dtype=torch.float64)
        def sampler():calls.append(1);return points
        p=PINNProblem(lambda x,u:u+x,sampler,[PointConstraint('data',points,lambda x:x.square())])
        torch.manual_seed(8);net=torch.nn.Linear(1,1).double();reference=deepcopy(net)
        factory=partial(torch.optim.SGD,momentum=.7)
        model=Trainer(p,net,loss={'pde':robust},weights={'data':2}).compile(optimizer=factory,lr=.01,device='cpu')
        opt=factory(reference.parameters(),lr=.01)
        terms=dict(pde=robust(reference(points)+points,torch.zeros_like(points)),data=(reference(points)-points.square()).square().mean())
        total=terms['pde']+2*terms['data'];total.backward();opt.step()
        model.train(1,0)
        self.assertAlmostEqual(model.loss_history[0]['total'],total.item(),places=14)
        for a,b in zip(net.parameters(),reference.parameters()):self.assertTrue(torch.equal(a,b))
        model.train(2,0);self.assertEqual(len(calls),1);self.assertFalse(points.requires_grad)
        self.assertEqual(model.predict(points).dtype,torch.float64)
        model.compile(optimizer=factory,device='cpu');self.assertEqual(len(calls),2)

    def test_custom_update_reuses_loop_and_preupdate_logging(self):
        calls=[]
        def clipped(trainer,terms,total):
            calls.append(trainer.iter);total.backward()
            torch.nn.utils.clip_grad_norm_(trainer.network.parameters(),.1)
            trainer.optimizer.step()
            trainer.weights['boundary']=trainer.weights.get('boundary',1.)*2
        torch.manual_seed(7)
        model=Trainer(problem(),network(),step=clipped,checkpoint_id='clip-v1').compile(device='cpu')
        before=model.objective()[1].item();model.train(3,0)
        self.assertEqual(calls,[0,1,2]);self.assertEqual(model.iter,3)
        self.assertEqual(model.loss_history[0]['total'],before)
        for index,record in enumerate(model.loss_history):
            self.assertAlmostEqual(record['weighted:boundary'],record['boundary']*2**index)
        with self.assertRaisesRegex(NotImplementedError,'Custom update'):model.training_state()

    def test_invalid_losses_and_nonfinite_gradient_do_not_step(self):
        for loss in (lambda p,t:1., lambda p,t:p.square(),lambda p,t:p.sum()*float('nan'),lambda p,t:p.sum().detach()):
            model=Trainer(problem(),network(),loss=loss).compile(device='cpu');before=deepcopy(model.network.state_dict())
            with self.assertRaises((ValueError,FloatingPointError)):model.train(1,0)
            self.assertEqual(model.iter,0)
            for key,value in before.items():self.assertTrue(torch.equal(value,model.network.state_dict()[key]))
        model=Trainer(problem(),network(),loss=lambda p,t:torch.sqrt((p*0).sum())).compile(device='cpu')
        before=deepcopy(model.network.state_dict())
        with self.assertRaisesRegex(FloatingPointError,'gradient'):model.train(1,0)
        for key,value in before.items():self.assertTrue(torch.equal(value,model.network.state_dict()[key]))
        with self.assertRaises(ValueError):Trainer(problem(),network(),loss={'typo':robust}).compile(device='cpu')
        with self.assertRaises(TypeError):Trainer(problem(),network()).compile(optimizer=lambda p,lr:None)
        with self.assertRaises(ValueError):Trainer(problem(),network()).compile(optimizer=torch.optim.LBFGS)
        with self.assertRaises(ValueError):Trainer(problem(),network(),loss=robust,loss_formulation='integral')

    def checkpoint_case(self,factory,device='cpu',dropout=False):
        full=build(factory,device,dropout);full.train(6,0)
        split=build(factory,device,dropout);split.train(3,0)
        with TemporaryDirectory() as directory:
            path=Path(directory)/'research.pt';split.save_checkpoint(path)
            self.assertEqual(torch.load(path,weights_only=True,map_location='cpu')['format'],'margam-research-training')
            restored=build(factory,device,dropout);restored.load_checkpoint(path);restored.train(3,0)
        self.assertEqual(full.loss_history,restored.loss_history)
        for key,value in full.network.state_dict().items():self.assertTrue(torch.equal(value,restored.network.state_dict()[key]))
        self.assertEqual(restored.iter,6)

    def test_exact_resume_adam_adamw_sgd_and_torch_rng(self):
        for factory in (torch.optim.Adam,partial(torch.optim.AdamW,weight_decay=.02),partial(torch.optim.SGD,momentum=.8)):
            with self.subTest(factory=factory):self.checkpoint_case(factory)
        self.checkpoint_case(torch.optim.Adam,dropout=True)

    def test_checkpoint_rejects_changed_objective_data_and_corruption(self):
        source=build();source.train(2,0);state=source.training_state()
        for change in ('id','weights','data','lr'):
            model=build()
            if change=='id':model.checkpoint_id='different-v1'
            elif change=='weights':model.weights['boundary']=3
            elif change=='data':model.collocation_point_sample[0,0]=.3
            else:model.optimizer.param_groups[0]['lr']=.1
            before=deepcopy(model.network.state_dict())
            with self.assertRaises(ValueError):model.load_training_state(state)
            self.assertEqual(model.iter,0)
            for key,value in before.items():self.assertTrue(torch.equal(value,model.network.state_dict()[key]))
        for change in ('moment','history','network'):
            bad=deepcopy(state)
            if change=='moment':next(iter(bad['optimizer']['state'].values()))['exp_avg']=torch.zeros(123)
            elif change=='history':bad['loss_history'][0]['total']=float('nan')
            else:next(iter(bad['network'].values())).fill_(float('inf'))
            model=build()
            with self.assertRaises(ValueError):model.load_training_state(bad)
            self.assertEqual(model.iter,0)
        model=build();before=deepcopy(model.network.state_dict());original=model.optimizer.load_state_dict;calls=[]
        def fail_once(value):
            calls.append(1)
            if len(calls)==1:raise RuntimeError('native optimizer rejection')
            return original(value)
        rng=torch.get_rng_state().clone()
        with patch.object(model.optimizer,'load_state_dict',side_effect=fail_once):
            with self.assertRaises(RuntimeError):model.load_training_state(state)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(model.iter,0)
        for key,value in before.items():self.assertTrue(torch.equal(value,model.network.state_dict()[key]))

    def test_custom_pinn_can_add_terms_without_reimplementing_training(self):
        class MyPINN(Trainer):
            def compute_losses(self):
                terms=super().compute_losses()
                terms['regularisation']=.01*sum(p.square().sum() for p in self.network.parameters())
                return terms
        model=MyPINN(problem(),network(),checkpoint_id='regularised-v1').compile(device='cpu')
        before=deepcopy(model.network.state_dict());model.train(2,0)
        self.assertIn('regularisation',model.loss_history[-1])
        self.assertTrue(any(not torch.equal(value,model.network.state_dict()[key]) for key,value in before.items()))
        state=model.training_state();restored=MyPINN(problem(),network(),checkpoint_id='regularised-v1').compile(device='cpu')
        restored.load_training_state(state);self.assertEqual(restored.iter,2)
        plain=Trainer(problem(),network(),checkpoint_id='regularised-v1').compile(device='cpu')
        with self.assertRaises(ValueError):plain.load_training_state(state)

    def test_checkpoint_limits_are_explicit_and_save_is_atomic(self):
        model=Trainer(problem(),network()).compile(device='cpu')
        with TemporaryDirectory() as directory:
            path=Path(directory)/'existing.pt';path.write_bytes(b'preserve me')
            with self.assertRaises(NotImplementedError):model.save_checkpoint(path)
            self.assertEqual(path.read_bytes(),b'preserve me')
        class CustomSGD(torch.optim.SGD):pass
        model=Trainer(problem(),network(),checkpoint_id='custom').compile(optimizer=CustomSGD,device='cpu')
        model.train(1,0)
        with self.assertRaises(NotImplementedError):model.training_state()

    def test_external_script_without_gui_or_registration(self):
        with TemporaryDirectory() as directory:
            script=Path(directory)/'my_experiment.py'
            script.write_text("""
import sys
import torch
from margam.research import Trainer, PINNProblem, PointConstraint, sample_box, derivative
net=torch.nn.Sequential(torch.nn.Linear(1,4),torch.nn.Tanh(),torch.nn.Linear(4,1))
problem=PINNProblem(lambda x,u:derivative(u,x,0,2)+2,
    lambda:sample_box([0],[1],8,method='stratified',seed=3),
    [PointConstraint('wall',torch.tensor([[0.],[1.]]))])
trainer=Trainer(problem,net,loss=torch.nn.SmoothL1Loss()).compile(optimizer=torch.optim.AdamW,device='cpu')
trainer.train(2,0)
assert trainer.iter==2 and len(trainer.loss_history)==2
assert not any(name.split('.')[0] in ('PySide6','mcp','uvicorn') for name in sys.modules)
assert not any(name=='margam.ui' or name.startswith('margam.ui.') for name in sys.modules)
""")
            result=subprocess.run([sys.executable,str(script)],cwd=directory,capture_output=True,text=True,timeout=30,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class ResearchCUDATests(unittest.TestCase):
    # Run a concrete CUDA checkpoint case; CPU contracts stay in the base class.
    checkpoint_case=ResearchTests.checkpoint_case
    def setUp(self):torch.set_num_threads(1)
    def test_cuda_custom_components_and_exact_resume(self):
        self.checkpoint_case(partial(torch.optim.AdamW,weight_decay=.02),device='cuda')
        class BufferedLoss(torch.nn.Module):
            def __init__(self):super().__init__();self.register_buffer('scale',torch.tensor(2.))
            def forward(self,prediction,target):return self.scale*(prediction-target).square().mean()
        loss=BufferedLoss();model=Trainer(problem(),network(),loss=loss).compile(device='cuda')
        model.train(1,0);self.assertTrue(loss.scale.is_cuda)
