"""Update timing, optimizer groups, safe failure and exact scheduled continuation."""
from copy import deepcopy
from functools import partial
import unittest
import torch
from margam.research import Trainer, PINNProblem, StepDecay


def build(device='cpu', schedule=StepDecay(2,.5), optimizer=torch.optim.AdamW):
    torch.manual_seed(71)
    net=torch.nn.Sequential(torch.nn.Linear(1,4),torch.nn.Tanh(),torch.nn.Dropout(.2),torch.nn.Linear(4,1)).double()
    problem=PINNProblem(lambda x,u:u+x,torch.tensor([[.2],[.6],[.9]],dtype=torch.float64),[])
    return Trainer(problem,net,checkpoint_id='scheduled-research-v1').compile(
        optimizer=optimizer,lr=.008,device=device,schedule=schedule)


def groups(parameters,lr):
    values=list(parameters)
    return torch.optim.AdamW([dict(params=values[:2],lr=lr),dict(params=values[2:],lr=lr/2)],lr=lr)


class ScheduleTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_timing_segmented_training_and_recompile(self):
        model=build();model.train(1,0);model.train(4,0)
        self.assertEqual([r['learning_rate:0'] for r in model.loss_history],[.008,.008,.004,.004,.002])
        self.assertEqual(model.optimizer.param_groups[0]['lr'],.002)
        model.train(0,0);self.assertEqual(model.iter,5)
        model.compile(lr=.01,device='cpu',schedule=StepDecay(1,.5))
        self.assertEqual(model.iter,0);self.assertEqual(model.optimizer.param_groups[0]['lr'],.01)
        model.train(1,0);self.assertEqual(model.optimizer.param_groups[0]['lr'],.005)
        self.assertEqual(model.loss_history[0]['learning_rate:0'],.01)

    def test_matches_manual_pytorch_updates(self):
        model=build();reference=deepcopy(model.network)
        optimizer=torch.optim.AdamW(reference.parameters(),lr=.008)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=.5)
        rng=torch.get_rng_state().clone()
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            x=model.collocation_point_sample.detach()
            (reference(x)+x).square().mean().backward();optimizer.step();scheduler.step()
        torch.set_rng_state(rng);model.train(5,0)
        for x,y in zip(reference.parameters(),model.network.parameters()):self.assertTrue(torch.equal(x,y))
        self.assertEqual(model.optimizer.param_groups[0]['lr'],scheduler.get_last_lr()[0])

    def continuation(self,device='cpu'):
        for factory in (torch.optim.Adam,partial(torch.optim.SGD,momentum=.8),groups):
            with self.subTest(optimizer=factory,device=device):
                full=build(device,optimizer=factory);full.train(7,0)
                split=build(device,optimizer=factory);split.train(3,0);state=split.training_state()
                self.assertEqual(state['version'],2)
                restored=build(device,optimizer=factory);restored.load_training_state(state);restored.train(4,0)
                self.assertEqual(full.loss_history,restored.loss_history)
                for key,value in full.network.state_dict().items():self.assertTrue(torch.equal(value,restored.network.state_dict()[key]))
                self.assertEqual([g['lr'] for g in full.optimizer.param_groups],[g['lr'] for g in restored.optimizer.param_groups])
                restored.load_training_state(state);self.assertEqual(restored.iter,3)
                # State is primitives/tensors and accepted by the restricted loader.
                import io
                stream=io.BytesIO();torch.save(state,stream);stream.seek(0)
                build(device,optimizer=factory).load_training_state(torch.load(stream,weights_only=True,map_location='cpu'))

    def test_exact_resume_cpu_rng_and_multiple_parameter_groups(self):self.continuation()

    def test_changed_schedule_and_corruption_rejected_without_mutation(self):
        source=build();source.train(3,0);state=source.training_state()
        for schedule in (None,StepDecay(3,.5),StepDecay(2,.9)):
            model=build(schedule=schedule)
            with self.assertRaises(ValueError):model.load_training_state(state)
            self.assertEqual(model.iter,0)
        for field in ('lr','history','version','base'):
            bad=deepcopy(state)
            if field=='lr':bad['optimizer']['param_groups'][0]['lr']=.008
            elif field=='history':bad['loss_history'][0]['learning_rate:0']=.4
            elif field=='version':bad['version']=1
            else:bad['signature']['schedule']['base_lrs']=[.1]
            model=build();before=deepcopy(model.network.state_dict());rng=torch.get_rng_state().clone()
            with self.assertRaises(ValueError):model.load_training_state(bad)
            self.assertEqual(model.iter,0);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for key,value in before.items():self.assertTrue(torch.equal(value,model.network.state_dict()[key]))
        model=build(schedule=None);model.train(1,0);legacy=model.training_state()
        self.assertEqual(legacy['version'],1);build(schedule=None).load_training_state(legacy)

    def test_failed_update_and_external_rate_edits_do_not_advance(self):
        model=build();model.loss=lambda p,t:p.sum()*float('nan')
        with self.assertRaises(FloatingPointError):model.train(1,0)
        self.assertEqual(model.iter,0);self.assertEqual(model.optimizer.param_groups[0]['lr'],.008)
        model=build();model.optimizer.param_groups[0]['lr']=.2
        for action in (lambda:model.train(1,0),model.training_state):
            with self.assertRaisesRegex(ValueError,'outside Trainer'):action()
        self.assertEqual(model.iter,0)
        model=build(schedule=StepDecay(1,5e-324));before=deepcopy(model.network.state_dict())
        with self.assertRaisesRegex(ValueError,'underflow'):model.train(1,0)
        self.assertEqual(model.iter,0)
        for key,value in before.items():self.assertTrue(torch.equal(value,model.network.state_dict()[key]))

    def test_configuration_guards(self):
        for every,gamma in ((True,.5),(0,.5),(1,0),(1,True),(1,1.1),(1,float('nan'))):
            with self.assertRaises(ValueError):StepDecay(every,gamma)
        with self.assertRaises(TypeError):build(schedule=lambda x:x)
        model=build();model.step=lambda *args:None
        with self.assertRaisesRegex(ValueError,'custom steps'):model.compile(schedule=StepDecay(2),device='cpu')
        with self.assertRaises(ValueError):build(optimizer=lambda p,lr:torch.optim.SGD(p,lr=0))
        class ConflictingTerm(Trainer):
            def compute_losses(self):
                terms=super().compute_losses();terms['learning_rate:0']=terms['pde'];return terms
        model=build()
        conflict=ConflictingTerm(model.problem,deepcopy(model.network)).compile(device='cpu',schedule=StepDecay(2))
        with self.assertRaisesRegex(ValueError,'learning_rate:'):conflict.train(1,0)

    def test_unscheduled_snapshots_can_be_reused_after_training(self):
        source=build(schedule=None);source.train(3,0);state=source.training_state()
        before=deepcopy(state['optimizer'])
        restored=build(schedule=None);restored.load_training_state(state);restored.train(2,0)
        for key,values in before['state'].items():
            for name,value in values.items():self.assertTrue(torch.equal(value,state['optimizer']['state'][key][name]))
        restored.load_training_state(state);self.assertEqual(restored.iter,3)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class ScheduleCUDATests(unittest.TestCase):
    continuation=ScheduleTests.continuation
    def setUp(self):torch.set_num_threads(1)
    # Only the actual continuation case is rerun on CUDA.
    def test_cuda_scheduled_resume(self):self.continuation('cuda')
