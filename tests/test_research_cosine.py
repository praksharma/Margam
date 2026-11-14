"""Finite cosine horizon, failure-safe timing and checkpoint/RNG continuation."""
from copy import deepcopy
from functools import partial
import math
import unittest
import torch
from margam.research import CosineDecay,StepDecay,Trainer,PINNProblem,PointConstraint


def groups(parameters,lr):
    values=list(parameters)
    return torch.optim.AdamW([dict(params=values[:2],lr=lr),dict(params=values[2:],lr=lr/3)],lr=lr)


def build(device='cpu',schedule=CosineDecay(5,.05,.8),optimizer=torch.optim.Adam):
    torch.manual_seed(37)
    if device=='cuda':torch.cuda.manual_seed_all(37)
    network=torch.nn.Sequential(torch.nn.Linear(1,5),torch.nn.Tanh(),torch.nn.Dropout(.15),torch.nn.Linear(5,1)).double()
    problem=PINNProblem(lambda x,u:u+x,torch.tensor([[.1],[.3],[.8]],dtype=torch.float64),
        [PointConstraint('endpoints',torch.tensor([[0.],[1.]],dtype=torch.float64))])
    return Trainer(problem,network,weights={'endpoints':2.},checkpoint_id='cosine-research-v1').compile(
        device=device,lr=.006,schedule=schedule,optimizer=optimizer)


class CosineTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_formula_horizon_power_and_multiple_groups(self):
        for power in (.8,1.,1.2):
            schedule=CosineDecay(4,.03,power)
            for step in range(9):
                expected=[base*(.03+.97*((1+math.cos(math.pi*min(step/4,1)))/2)**power) for base in (.006,.002)]
                self.assertEqual(schedule.rates((.006,.002),step),expected)
            self.assertEqual(schedule.rates((.006,.002),4),[.006*.03,.002*.03])
            self.assertEqual(schedule.rates((.006,.002),8),schedule.rates((.006,.002),4))
        model=build(schedule=CosineDecay(4,.03));model.train(2,0);model.train(5,0)
        self.assertEqual([row['learning_rate:0'] for row in model.loss_history],[model.schedule.rates((.006,),k)[0] for k in range(7)])
        self.assertEqual(model.iter,7);self.assertEqual(model.optimizer.param_groups[0]['lr'],.006*.03)
        model.compile(device='cpu',lr=.01,schedule=CosineDecay(2,.5));self.assertEqual(model.iter,0)
        model.train(1,0);self.assertEqual(model.loss_history[0]['learning_rate:0'],.01)
        self.assertEqual(model.optimizer.param_groups[0]['lr'],.0075)

    def test_matches_the_existing_scale_cavity_driver_schedule(self):
        from examples.scale_cavity_benchmark import protocol,learning_rate
        for reynolds in (3200,7500):
            settings=protocol(reynolds,20)
            schedule=CosineDecay(settings['schedule_steps'],settings['end_lr']/settings['lr'],settings['cosine_exponent'])
            for step in (0,1,100,25000,49999,50000,60000):
                self.assertAlmostEqual(schedule.rates((settings['lr'],),step)[0],learning_rate(step,settings),places=17)

    def test_standard_power_matches_independent_pytorch_cosine_before_horizon(self):
        model=build(schedule=CosineDecay(5,.05));reference=deepcopy(model.network)
        optimizer=torch.optim.Adam(reference.parameters(),lr=.006)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=5,eta_min=.006*.05)
        rng=torch.get_rng_state().clone()
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            x=model.collocation_point_sample
            loss=(reference(x)+x).square().mean()+2*reference(model._constraints[0][1]).square().mean()
            loss.backward();optimizer.step();scheduler.step()
        torch.set_rng_state(rng);model.train(5,0)
        for a,b in zip(reference.parameters(),model.network.parameters()):torch.testing.assert_close(a,b,rtol=1e-14,atol=1e-14)
        self.assertEqual(model.optimizer.param_groups[0]['lr'],scheduler.get_last_lr()[0])

    def continuation(self,device):
        for factory in (torch.optim.Adam,partial(torch.optim.SGD,momentum=.8),groups):
            for phase in (4,5,6):
                with self.subTest(device=device,optimizer=factory,phase=phase):
                    full=build(device,optimizer=factory);full.train(8,0)
                    part=build(device,optimizer=factory);part.train(phase,0);state=part.training_state();frozen=deepcopy(state)
                    self.assertEqual(state['version'],2);self.assertEqual(state['signature']['schedule']['kind'],'cosine_decay')
                    restored=build(device,optimizer=factory);restored.load_training_state(state);restored.train(8-phase,0)
                    self.assertEqual(full.loss_history,restored.loss_history)
                    for name,value in full.network.state_dict().items():torch.testing.assert_close(value,restored.network.state_dict()[name],rtol=0,atol=0)
                    for key,value in frozen['optimizer']['state'].items():
                        for name,tensor in value.items():torch.testing.assert_close(tensor,state['optimizer']['state'][key][name],rtol=0,atol=0)
                    self.assertEqual([g['lr'] for g in full.optimizer.param_groups],[g['lr'] for g in restored.optimizer.param_groups])
                    restored.load_training_state(state);self.assertEqual(restored.iter,phase)
                    import io
                    stream=io.BytesIO();torch.save(state,stream);stream.seek(0)
                    build(device,optimizer=factory).load_training_state(torch.load(stream,weights_only=True,map_location='cpu'))

    def test_cpu_exact_resume_rng_groups_and_floor(self):self.continuation('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_exact_resume_rng_groups_and_floor(self):self.continuation('cuda')

    def test_example_cli_records_resume_sources_and_rejects_conflicting_flags(self):
        import hashlib,json,os,subprocess,sys
        from pathlib import Path
        from tempfile import TemporaryDirectory
        root=Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as directory:
            command=[sys.executable,'-m','examples.research_custom','--cosine-decay','--device','cpu','--steps','8','--output',directory]
            result=subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=45,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads((Path(directory)/'result.json').read_text())
            self.assertTrue(report['exact_resume']);self.assertTrue(report['cosine_decay'])
            self.assertEqual(report['next_learning_rates'],[.003*.05])
            self.assertEqual(report['schedule'],dict(kind='cosine_decay',steps=8,minimum_ratio=.05,power=1.))
            for name,digest in report['source_sha256'].items():self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),digest)
            for flag in ('--custom-step','--step-decay'):
                invalid=subprocess.run(command+[flag],cwd=root,capture_output=True,text=True,timeout=30)
                self.assertEqual(invalid.returncode,2)

    def test_changed_schedule_history_and_optimizer_rates_fail_before_mutation(self):
        source=build();source.train(3,0);state=source.training_state()
        for schedule in (None,StepDecay(5),CosineDecay(6,.05,.8),CosineDecay(5,.06,.8),CosineDecay(5,.05,1.)):
            target=build(schedule=schedule)
            with self.assertRaises(ValueError):target.load_training_state(state)
            self.assertEqual(target.iter,0)
        for corrupt in ('rate','history','kind','base','floor','power','horizon'):
            bad=deepcopy(state)
            if corrupt=='rate':bad['optimizer']['param_groups'][0]['lr']=.006
            elif corrupt=='history':bad['loss_history'][1]['learning_rate:0']=.006
            else:
                key={'kind':'kind','base':'base_lrs','floor':'minimum_ratio','power':'power','horizon':'steps'}[corrupt]
                bad['signature']['schedule'][key]= {'kind':'step_decay','base':[.1],'floor':.9,'power':2.,'horizon':50}[corrupt]
            target=build();before=deepcopy(target.network.state_dict());rng=torch.get_rng_state().clone()
            with self.subTest(corrupt=corrupt),self.assertRaises(ValueError):target.load_training_state(bad)
            self.assertEqual(target.iter,0);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for name,value in before.items():torch.testing.assert_close(value,target.network.state_dict()[name],rtol=0,atol=0)

    def test_failures_rate_edits_and_invalid_configuration(self):
        for args in ((True,.1,1),(0,.1,1),(2,0,1),(2,True,1),(2,1.1,1),(2,float('nan'),1),(2,.1,0),(2,.1,True),(2,.1,float('inf'))):
            with self.subTest(args=args),self.assertRaises(ValueError):CosineDecay(*args)
        for completed in (-1,True,.5):
            with self.assertRaises(ValueError):CosineDecay(2).rates((.01,),completed)
        model=build();model.loss=lambda p,t:p.sum()*float('nan')
        with self.assertRaises(FloatingPointError):model.train(1,0)
        self.assertEqual(model.iter,0);self.assertEqual(model.optimizer.param_groups[0]['lr'],.006)
        # Finite forward objective, non-finite backward; no optimizer/schedule update.
        class BadGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx,x):return x.clone()
            @staticmethod
            def backward(ctx,gradient):return gradient*float('nan')
        model=build();before=deepcopy(model.network.state_dict());model.loss=lambda p,t:BadGradient.apply(p).square().mean()
        with self.assertRaises(FloatingPointError):model.train(1,0)
        self.assertEqual(model.iter,0);self.assertEqual(model.optimizer.param_groups[0]['lr'],.006)
        for name,value in before.items():torch.testing.assert_close(value,model.network.state_dict()[name],rtol=0,atol=0)
        model=build();model.optimizer.param_groups[0]['lr']=.4
        for action in (lambda:model.train(1,0),model.training_state):
            with self.assertRaisesRegex(ValueError,'outside Trainer'):action()
        model=build(schedule=CosineDecay(1,5e-324));before=deepcopy(model.network.state_dict())
        with self.assertRaisesRegex(ValueError,'underflow'):model.train(1,0)
        self.assertEqual(model.iter,0)
        for name,value in before.items():torch.testing.assert_close(value,model.network.state_dict()[name],rtol=0,atol=0)
        model=build();model.step=lambda *args:None
        with self.assertRaisesRegex(ValueError,'custom steps'):model.compile(device='cpu',schedule=CosineDecay(2))


if __name__=='__main__':unittest.main()
