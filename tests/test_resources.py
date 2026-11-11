"""Resource arithmetic is independent of allocations; memory failures remain recoverable."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from margam.components.resources import architecture_size,resource_report,report_text,worker_failure
from margam.components.outputs import OutputRequest
from margam.templates.poisson import PoissonConfig


def memory(free=1024**3):
    return dict(host_available=free,host_total=4*1024**3,gpu_available=free,gpu_total=6*1024**3,dtype='torch.float32',scalar_bytes=4)


class ResourceTests(unittest.TestCase):
    def test_counts_match_actual_supported_architectures_without_using_factory(self):
        import torch
        from margam.nn.factory import build_network
        from margam.components.network_config import NETWORK_NAMES,OPTION_SCHEMAS
        from margam.templates.expert_examples import case_config
        for dimension in (1,2,3):
            for kind in NETWORK_NAMES:
                c=case_config('expert_torus') if dimension==3 else PoissonConfig(shape='Interval' if dimension==1 else 'Rectangle')
                c.network_type=kind;c.network_options={};c.layers=(4,4)
                with self.subTest(dimension=dimension,kind=kind):
                    result=architecture_size(c);model=build_network(c)
                    self.assertEqual(result['parameters'],sum(p.numel() for p in model.parameters()))
                    self.assertEqual(result['buffer_elements'],sum(b.numel() for b in model.buffers()))
        for kind in ('fcnn','fourier','modified_fourier'):
            c=PoissonConfig(network_type=kind,network_options=dict(weight_factorization=True),layers=(4,4))
            result=architecture_size(c);model=build_network(c)
            self.assertEqual(result['parameters'],sum(p.numel() for p in model.parameters()))
        for options in (dict(encoding='full',frequencies=[0,1,2]),dict(encoding='diagonal'),dict(encoding='gaussian',features=7)):
            c=case_config('expert_torus');c.network_type='fourier';c.network_options=options;c.layers=(4,4)
            model=build_network(c);result=architecture_size(c)
            self.assertEqual(result['parameters'],sum(p.numel() for p in model.parameters()))
            self.assertEqual(result['buffer_elements'],sum(b.numel() for b in model.buffers()))
        c=PoissonConfig(network_type='affine_fourier',network_options=dict(input_transform='x;y;x-1;y-2'),layers=(4,))
        result=architecture_size(c);model=build_network(c)
        self.assertEqual(result['parameters'],sum(p.numel() for p in model.parameters()))
        self.assertEqual(result['buffer_elements'],sum(b.numel() for b in model.buffers()))
        c=PoissonConfig(method='inverse_burgers',shape='SpaceTime1D',layers=(4,4))
        self.assertEqual(architecture_size(c)['parameters'],sum(p.numel() for p in build_network(c).parameters()))
        rng=torch.get_rng_state().clone()
        with patch('margam.nn.factory.build_network',side_effect=AssertionError('Resource arithmetic must not allocate a model')):architecture_size(c)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_combined_workload_bounds_advice_and_unknown_memory(self):
        c=PoissonConfig(layers=(32,32));c.adaptive_sampling=dict(method='rar',maximum=70000,candidates=1500,chunk=64)
        counts=dict(pde=1000,sets=dict(left=3,right=5))
        data=SimpleNamespace(points=[[0,0]]*7)
        requests=[OutputRequest('grid','solution',resolution=11),OutputRequest('cloud','export',point_data=data),OutputRequest('monitor','monitor')]
        report=resource_report(c,counts,requests,memory=memory(1024),monitor_limits=[500],output_names={'grid':'Grid preview'})
        self.assertEqual(report['maximum_pde_rows'],70000);self.assertEqual(report['candidate_rows'],1500);self.assertEqual(report['scoring_chunk'],64)
        self.assertEqual(report['output_rows'],128);self.assertEqual(report['named_set_rows'],8);self.assertEqual(report['monitor_record_budget'],500)
        self.assertTrue(report['warnings']);text=report_text(c,report)
        self.assertIn('Grid preview',text);self.assertIn('NOT total training memory',text);self.assertIn('NOT a peak estimate',text)
        empty=memory();empty['host_available']=None
        self.assertFalse(resource_report(c,counts,requests,memory=empty)['warnings'])
        self.assertTrue(resource_report(c,counts,requests,memory=memory(0))['warnings'])
        c.adaptive_sampling=dict(method='importance',candidates=20,chunk=256)
        self.assertEqual(resource_report(c,counts,requests,memory=memory())['maximum_pde_rows'],1000)
        self.assertEqual(resource_report(c,counts,requests,memory=memory())['scoring_chunk'],20)

    def test_large_full_fourier_expansion_is_flagged_before_allocating_features(self):
        from margam.templates.expert_examples import case_config
        c=case_config('expert_torus');c.device='cuda';c.network_type='fourier';c.layers=(256,)*6
        c.network_options=dict(encoding='full',frequencies=list(range(16)))
        report=resource_report(c,dict(pde=100000,sets={}),(),memory=memory(6*1024**3))
        self.assertEqual(report['expanded_inputs'],8195)
        self.assertEqual(report['encoded_input_scale_bytes'],100000*8195*4)
        self.assertTrue(any('encoded-input' in warning for warning in report['warnings']))

    def test_cuda_memory_snapshot_is_advisory_without_rng_or_tensor_allocation(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('CUDA required')
        from margam.components.resources import memory_snapshot
        cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone();allocated=torch.cuda.memory_allocated()
        data=memory_snapshot('cuda')
        self.assertIsNotNone(data['gpu_available']);self.assertGreater(data['gpu_total'],0)
        self.assertTrue(0<=data['gpu_available']<=data['gpu_total'])
        self.assertEqual(allocated,torch.cuda.memory_allocated())
        self.assertTrue(torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state()))

    def test_error_classification_preserves_ordinary_errors(self):
        import torch
        for error,stage,device,expected in [
            (MemoryError('allocation failed'),'model construction','cuda','host RAM'),
            (torch.OutOfMemoryError('allocation failed'),'training update','cuda','GPU memory'),
            (RuntimeError('CUDA out of memory. Tried to allocate'),'output evaluation','cuda','output resolution'),
            (RuntimeError("DefaultCPUAllocator: can't allocate memory"),'training update','cpu','host RAM'),
            (torch.OutOfMemoryError('allocation failed'),'checkpoint restore','cuda','Continuation needs')]:
            text=worker_failure(error,stage,device);self.assertIn(expected,text);self.assertIn(stage,text);self.assertIn('not changed automatically',text)
        self.assertEqual(worker_failure(ValueError('out of memory expression'),'training update','cpu'),'ValueError: out of memory expression')
        self.assertEqual(worker_failure(RuntimeError('bad tensor shape'),'training update','cuda'),'RuntimeError: bad tensor shape')

    def run_ui(self,code):
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=120,
            env=dict(os.environ,QT_QPA_PLATFORM='offscreen',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        self.assertNotIn('Traceback',result.stderr)

    def test_summary_warning_is_readonly_and_does_not_allocate_network(self):
        self.run_ui(r"""
from copy import deepcopy
from unittest.mock import patch
import torch
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
from margam.ui.model_document import capture_model
app=QApplication([]);w=Window(state_directory=False)
try:
 w.set_application_device('cpu');w.load_heat_template();w.change('iterations',2);w.change('layers',(4,))
 assert w.start_training()
 while w.worker is not None:app.processEvents()
 before=capture_model(w);index=w.undo_stack.index();result=w.result;model=w.last_run.model;rng=torch.get_rng_state().clone()
 profile=dict(host_available=1,host_total=4096,gpu_available=None,gpu_total=None,dtype='torch.float32',scalar_bytes=4)
 with patch('margam.components.resources.memory_snapshot',return_value=profile),patch('margam.nn.factory.build_network',side_effect=AssertionError('No model allocation')):
  dialog=w.show_run_summary();assert dialog.train.isEnabled()
  text=dialog.text.toPlainText();assert 'Memory and workload' in text and 'WARNING:' in text and 'not a fit prediction' in text
  assert capture_model(w)==before and w.undo_stack.index()==index and w.result is result and w.last_run.model is model
  assert torch.equal(rng,torch.get_rng_state());dialog.close()
finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()
""")

    def test_workers_report_oom_and_release_failed_state_then_gui_can_run_again(self):
        self.run_ui(r"""
from unittest.mock import patch
from types import SimpleNamespace
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
from margam.ui.training import TrainingWorker,EvaluationWorker
from margam.templates.poisson import PoissonConfig
app=QApplication([]);c=PoissonConfig(device='cpu',iterations=1)
for phase in ('construction','restore','train','outputs'):
 model=SimpleNamespace(iter=0,sampling_controller=None,device='cpu',loss_history=[],train=lambda *a,**k:None,load_training_state=lambda state:None)
 failures=[];completed=[]
 worker=TrainingWorker(c,checkpoint={} if phase=='restore' else None);worker.failed.connect(failures.append);worker.completed.connect(completed.append)
 if phase=='restore':model.load_training_state=lambda state:(_ for _ in ()).throw(MemoryError('simulated restore allocation'))
 if phase=='train':model.train=lambda *a,**k:(_ for _ in ()).throw(MemoryError('simulated training allocation'))
 if phase=='outputs':model.iter=1
 with patch('margam.templates.poisson.build_pinn',side_effect=MemoryError('simulated construction allocation') if phase=='construction' else None,return_value=model),patch('margam.components.outputs.evaluate_outputs',side_effect=MemoryError('simulated output allocation')):
  worker.run()
 assert len(failures)==1 and not completed and worker.model is None and worker.checkpoint is None
 expected={'construction':'model construction','restore':'checkpoint restore','train':'training update','outputs':'output evaluation'}[phase]
 assert expected in failures[0]
model=SimpleNamespace(iter=2,device='cpu');worker=EvaluationWorker(c,(),model=model);failures=[];worker.failed.connect(failures.append)
with patch('margam.components.outputs.evaluate_outputs',side_effect=MemoryError('simulated output allocation')):worker.run()
assert failures and 'output evaluation' in failures[0] and worker.model is None
w=Window(state_directory=False)
try:
 w.set_application_device('cpu');w.change('iterations',2);w.change('layers',(4,))
 with patch('margam.execution.transport.JobTransport.run',side_effect=MemoryError('simulated allocation')):
  assert w.start_training()
  while w.worker is not None:app.processEvents()
 assert w.result is None and 'Memory allocation failed during isolated numerical job' in w.log.toPlainText() and w.train_button.isEnabled()
 assert w.start_training()
 while w.worker is not None:app.processEvents()
 assert w.result is not None and w.last_run.model.iter==2
finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()
""")
