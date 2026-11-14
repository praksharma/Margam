"""Whole-objective gradient equivalence, bounded forwards and exact continuation."""
from copy import deepcopy
from functools import partial
import unittest
from unittest.mock import patch
import torch
from margam.research import Trainer,PINNProblem,PointConstraint,IntegralConstraint,derivative,StepDecay,CosineDecay


def build(device='cpu',size=3,optimizer=torch.optim.Adam,schedule=None,dtype=torch.float64):
    torch.manual_seed(91)
    network=torch.nn.Sequential(torch.nn.Linear(1,6),torch.nn.Tanh(),torch.nn.Linear(6,2)).to(dtype=dtype)
    x=torch.linspace(.05,.95,7,dtype=dtype)[:,None]
    target=torch.cat((x*.2,x*.3),1)
    def residual(x,u):return torch.cat((derivative(u[:,0:1],x,0,2)+u[:,1:2],derivative(u[:,1:2],x,0)-x*u[:,0:1]),1)
    values=torch.linspace(0,1,5,dtype=dtype)[:,None]
    constraints=[PointConstraint('value',values,torch.cat((values,values*.3),1)),
        PointConstraint('slope',torch.tensor([[0.],[.4],[1.]],dtype=dtype),.2,lambda x,u:derivative(u[:,0:1],x,0))]
    problem=PINNProblem(residual,x,constraints,target)
    return Trainer(problem,network,weights={'pde':1.7,'value':2.3,'slope':.8},checkpoint_id='microbatch-vector-derivative-v1').compile(
        device=device,optimizer=optimizer,lr=.002,schedule=schedule,pointwise_microbatch=size)


class MicrobatchTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def parity(self,device,dtype=torch.float64):
        full=build(device,size=None,optimizer=torch.optim.SGD,dtype=dtype);full.train(1,0)
        for size in (1,3,100):
            with self.subTest(device=device,size=size):
                chunked=build(device,size=size,optimizer=torch.optim.SGD,dtype=dtype)
                seen=[]
                handle=chunked.network.register_forward_pre_hook(lambda m,args:seen.append(len(args[0])))
                chunked.train(1,0);handle.remove()
                self.assertLessEqual(max(seen),size)
                self.assertEqual(sum(seen),7+5+3)
                for key,loss in full.loss_history[0].items():self.assertAlmostEqual(chunked.loss_history[0][key],loss,places=12 if dtype==torch.float64 else 5)
                for a,b in zip(full.network.parameters(),chunked.network.parameters()):
                    torch.testing.assert_close(a.grad,b.grad,rtol=1e-10 if dtype==torch.float64 else 2e-5,atol=1e-12 if dtype==torch.float64 else 1e-6)
                    torch.testing.assert_close(a,b,rtol=1e-12 if dtype==torch.float64 else 1e-6,atol=1e-13 if dtype==torch.float64 else 1e-7)
                self.assertEqual(chunked.iter,1)

    def test_cpu_loss_parameter_gradients_vector_targets_and_uneven_sets(self):self.parity('cpu')

    def continuation(self,device,dtype=torch.float64):
        for optimizer in (torch.optim.Adam,torch.optim.AdamW,partial(torch.optim.SGD,momentum=.8)):
            for schedule in (None,StepDecay(2,.5),CosineDecay(3,.05,.8)):
                with self.subTest(device=device,optimizer=optimizer,schedule=schedule,dtype=dtype):
                    full=build(device,optimizer=optimizer,schedule=schedule,dtype=dtype);full.train(6,0)
                    part=build(device,optimizer=optimizer,schedule=schedule,dtype=dtype);part.train(2,0);state=part.training_state();snapshot=deepcopy(state)
                    self.assertEqual(state['version'],3)
                    resumed=build(device,optimizer=optimizer,schedule=schedule,dtype=dtype);resumed.load_training_state(state);resumed.train(4,0)
                    self.assertEqual(full.loss_history,resumed.loss_history)
                    for key,value in full.network.state_dict().items():torch.testing.assert_close(value,resumed.network.state_dict()[key],rtol=0,atol=0)
                    for key,value in snapshot['optimizer']['state'].items():
                        for name,tensor in value.items():torch.testing.assert_close(tensor,state['optimizer']['state'][key][name],rtol=0,atol=0)
                    import io
                    stream=io.BytesIO();torch.save(state,stream);stream.seek(0)
                    build(device,optimizer=optimizer,schedule=schedule,dtype=dtype).load_training_state(torch.load(stream,weights_only=True,map_location='cpu'))
        for schedule in (None,StepDecay(2)):
            old=build(device,size=None,schedule=schedule,dtype=dtype);old.train(2,0);state=old.training_state()
            self.assertEqual(state['version'],2 if schedule else 1)
            self.assertNotIn('pointwise_microbatch',state['signature'])
            build(device,size=None,schedule=schedule,dtype=dtype).load_training_state(state)

    def test_float32_cpu_gradients_and_resume(self):self.parity('cpu',torch.float32);self.continuation('cpu',torch.float32)

    def test_cpu_checkpoint_optimizer_and_schedule_continuation(self):self.continuation('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_gradients_and_exact_continuation(self):
        for dtype in (torch.float32,torch.float64):self.parity('cuda',dtype);self.continuation('cuda',dtype)

    def test_failed_late_chunk_and_gradient_never_commit_optimizer_or_schedule(self):
        model=build(schedule=CosineDecay(3));old=deepcopy(model.network.state_dict());original=model.problem.residual
        calls=[]
        def invalid(x,u):
            calls.append(len(x));value=original(x,u)
            return value if len(calls)==1 else value*float('nan')
        model.problem.residual=invalid
        with self.assertRaises(FloatingPointError):model.train(1,0)
        self.assertEqual(calls,[3,3]);self.assertEqual(model.iter,0);self.assertEqual(model.loss_history,[])
        self.assertEqual(model.optimizer.state_dict()['state'],{});self.assertEqual(model.optimizer.param_groups[0]['lr'],.002)
        self.assertTrue(all(p.grad is None for p in model.network.parameters()))
        for key,value in old.items():torch.testing.assert_close(value,model.network.state_dict()[key],rtol=0,atol=0)
        class Bad(torch.autograd.Function):
            @staticmethod
            def forward(ctx,x):return x.clone()
            @staticmethod
            def backward(ctx,g):return g*float('nan')
        model.problem.residual=lambda x,u:Bad.apply(original(x,u))
        with self.assertRaises(FloatingPointError):model.train(1,0)
        self.assertEqual(model.iter,0);self.assertEqual(model.optimizer.state_dict()['state'],{})
        self.assertTrue(all(p.grad is None for p in model.network.parameters()))
        model.problem.residual=original;model.train(1,0);self.assertEqual(model.iter,1)

    def test_changed_reduction_order_or_checkpoint_contract_rejected_before_mutation(self):
        source=build();source.train(2,0);state=source.training_state()
        for size in (None,1,4):
            target=build(size=size);before=deepcopy(target.network.state_dict());rng=torch.get_rng_state().clone()
            with self.assertRaises(ValueError):target.load_training_state(state)
            self.assertEqual(target.iter,0);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for key,value in before.items():torch.testing.assert_close(value,target.network.state_dict()[key],rtol=0,atol=0)
        for key,value in (('version',2),('contract','other')):
            bad=deepcopy(state)
            if key=='version':bad['version']=value
            else:bad['signature']['pointwise_microbatch']['contract']=value
            with self.assertRaises(ValueError):build().load_training_state(bad)
        with self.assertRaises(AttributeError):source.pointwise_microbatch=4

    def test_rejects_global_stochastic_custom_and_adaptive_compositions(self):
        base=build()
        for size in (True,0,-1,1.5):
            with self.assertRaises(ValueError):Trainer(base.problem,deepcopy(base.network)).compile(device='cpu',pointwise_microbatch=size)
        for loss,step,mode in ((lambda a,b:(a-b).square().mean(),None,'mean'),(None,lambda *a:None,'mean'),(None,None,'integral')):
            with self.assertRaises(ValueError):Trainer(base.problem,deepcopy(base.network),loss=loss,step=step,loss_formulation=mode).compile(device='cpu',pointwise_microbatch=3)
        for layer in (torch.nn.Dropout(.1),torch.nn.BatchNorm1d(2),torch.nn.RReLU()):
            net=torch.nn.Sequential(torch.nn.Linear(1,2),layer,torch.nn.Linear(2,2)).double()
            with self.assertRaises(ValueError):Trainer(base.problem,net).compile(device='cpu',pointwise_microbatch=3)
        integral=IntegralConstraint('total',torch.ones(2,1),1.,quadrature_weights=torch.ones(2),operator=lambda x,u:u[:,0:1])
        problem=deepcopy(base.problem);problem.constraints=[integral]
        with self.assertRaisesRegex(ValueError,'IntegralConstraint'):Trainer(problem,deepcopy(base.network)).compile(device='cpu',pointwise_microbatch=3)
        for changed in ('step','loss','sampler'):
            model=build()
            if changed=='step':model.step=lambda *args:None
            elif changed=='loss':model.loss=lambda a,b:(a-b).square().mean()
            else:model.sampling_controller=object()
            with self.assertRaises(ValueError):model.train(1,0)
            self.assertEqual(model.iter,0)

    def test_extra_or_missing_target_rows_cannot_be_hidden_by_slicing(self):
        for extra in (-1,1):
            for term in ('pde','boundary'):
                for size in (None,3):
                    with self.subTest(extra=extra,term=term,size=size):
                        model=build(size=size)
                        if term=='pde':model.collocation_point_labels=torch.zeros((7+extra,2),dtype=torch.float64)
                        else:
                            condition,points,_=model._constraints[0]
                            model._constraints[0]=(condition,points,torch.zeros((5+extra,2),dtype=torch.float64))
                        with self.assertRaises(ValueError):model.train(1,0)
                        self.assertEqual(model.iter,0);self.assertEqual(model.optimizer.state_dict()['state'],{})

    def test_cli_writes_fresh_report_checkpoint_and_preserves_existing_experiments(self):
        import hashlib,json,os,subprocess,sys
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from examples.research_microbatch import build as example_trainer
        root=Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as directory:
            output=Path(directory)/'trial'
            command=[sys.executable,'-m','examples.research_microbatch','--device','cpu','--points','19','--steps','4','--microbatch','3','--output',str(output)]
            result=subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=45,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads((output/'result.json').read_text());before=(output/'result.json').read_bytes()
            self.assertTrue(report['exact_resume']);self.assertEqual(report['checkpoint_version'],3)
            self.assertEqual(report['steps'],4);self.assertEqual(report['pointwise_microbatch'],3);self.assertIsNone(report['cuda_allocator'])
            for name,digest in report['source_sha256'].items():self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),digest)
            restored=example_trainer('cpu',19,4,3).load_checkpoint(output/'final-checkpoint.pt')
            self.assertEqual(restored.iter,4)
            duplicate=subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=30)
            self.assertEqual(duplicate.returncode,2);self.assertEqual(before,(output/'result.json').read_bytes())

    def test_constant_derivative_term_can_coexist_with_trainable_constraints(self):
        full=build(size=None);chunked=build()
        for model in (full,chunked):model.problem.residual=lambda x,u:x.new_zeros((len(x),2))
        full.train(1,0);chunked.train(1,0)
        self.assertAlmostEqual(full.loss_history[0]['pde'],chunked.loss_history[0]['pde'],places=14)
        for a,b in zip(full.network.parameters(),chunked.network.parameters()):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-13)


if __name__=='__main__':unittest.main()
