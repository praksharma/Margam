"""Retained adaptive point pages, boundary roles, safe Qt lifecycle and archives."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import unittest
import torch
from margam.ui.sample_points import snapshot_page


class SamplePointsTests(unittest.TestCase):
    def test_page_order_anchor_positions_and_correction_weights(self):
        points=torch.arange(428,dtype=torch.float64).reshape(214,2)
        before=points.clone()
        data=snapshot_page(points,None,initial_count=208,boundary_count=2,method='rar',steps=5,history=[],page=2)
        self.assertEqual([r['row'] for r in data['rows']],list(range(201,215)))
        self.assertEqual([r['kind'] for r in data['rows']][6:8],['Boundary/initial PDE anchor']*2)
        self.assertEqual([r['kind'] for r in data['rows']][8:],['RAR addition']*6)
        self.assertTrue(torch.equal(points,before));data['rows'][0]['coordinates'][0]=-1;self.assertTrue(torch.equal(points,before))
        weights=torch.linspace(.2,2.,208)[:,None];weights[-2:]=1
        data=snapshot_page(points[:208],weights,initial_count=208,boundary_count=2,method='importance',steps=5,history=[],page=99)
        self.assertEqual(data['page'],2);self.assertEqual(data['rows'][0]['correction'],weights[200].item())
        self.assertEqual(data['rows'][-1]['kind'],'Boundary/initial PDE anchor')
        self.assertEqual(data['rows'][0]['kind'],'Importance draw')
        warmup=snapshot_page(points[:208],None,initial_count=208,boundary_count=0,method='importance',steps=0,history=[])
        self.assertEqual(warmup['rows'][0]['kind'],'Initial interior')
        self.assertEqual(warmup['rows'][0]['correction'],1.)

    def test_invalid_shapes_counts_and_nonfinite_page_are_rejected(self):
        values=dict(points=torch.ones(2,1),weights=None,initial_count=2,boundary_count=0,method='rar',steps=1,history=[])
        for change in (dict(points=torch.ones(2,3)),dict(initial_count=3),dict(boundary_count=3),dict(method='other'),dict(page=0),dict(points=torch.tensor([[float('nan')],[1.]])),dict(weights=torch.zeros(2,1))):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):snapshot_page(**(values|change))

    def test_native_model_and_archive_inspection_is_readonly_and_worker_guarded(self):
        code=r"""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import torch
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
from margam.ui.model_document import capture_model
from examples.capture_sample_points import prepare
app=QApplication([]);w=Window(state_directory=False)
try:
 w.set_application_device('cpu')
 for kind in ('rar','importance'):
  node=prepare(w,kind);empty=w.show_sample_points(node);assert empty.snapshot is None
  assert w.start_training()
  while w.worker is not None:app.processEvents()
  model=w.last_run.model;state=deepcopy(model.training_state());result=w.result;before=capture_model(w);index=w.undo_stack.index()
  assert any(a.text()=='Inspect retained PDE points…' for a in w.create_block_menu(node).actions())
  with patch.object(model.problem,'residual',side_effect=AssertionError('No residual scoring')),patch.object(model,'predict',side_effect=AssertionError('No inference')),patch('margam.components.adaptive_sampling.sample_box',side_effect=AssertionError('No sampling')):
   dialog=w.show_sample_points(node);dialog.page.setValue(2);data=deepcopy(dialog.snapshot)
   assert data['boundary_count']==2 and data['count']==(214 if kind=='rar' else 208)
   assert data['rows'][6]['kind']=='Boundary/initial PDE anchor'
   for row in data['rows']:
    position=row['row']-1
    assert row['coordinates']==model.collocation_point_sample[position].cpu().tolist()
    expected=1. if model.sampling_controller.weights is None else model.sampling_controller.weights[position].item()
    assert row['correction']==expected
   assert w.result is result and w.last_run.model is model and model.loss_history==state['loss_history']
   assert capture_model(w)==before and w.undo_stack.index()==index
   w.set_busy(True)
   assert dialog.snapshot is None;dialog.refresh();assert dialog.snapshot is None
   w.set_busy(False);dialog.refresh();assert dialog.snapshot==data
  for name,value in state['network'].items():assert torch.equal(value,model.training_state()['network'][name])
  with TemporaryDirectory() as directory:
   archive=Path(directory)/'run.pt';assert w.save_current_run(archive);w.undo_stack.setClean();assert w.open_saved_run(archive)
   node=next(n for n in w.scene.nodes() if n.kind==kind)
   with patch('margam.components.adaptive_sampling.sample_box',side_effect=AssertionError('No resampling on archive inspection')):
    dialog=w.show_sample_points(node);dialog.page.setValue(2);assert dialog.snapshot==data
   assert w.rename_block(node,'Retained sample source') and dialog.snapshot==data
   w.change('source',3.)
   assert dialog.snapshot is None and dialog.table.rowCount()==0
   dialog.refresh();assert dialog.snapshot is None
   w.delete_block(node);assert node.identifier not in w.sample_point_dialogs
 # A stopped fresh run still retains its initial/adapted point set for inspection.
 node=prepare(w,'importance');w.change('iterations',100)
 assert w.start_training();w.cancel_training()
 while w.worker is not None:app.processEvents()
 assert w.result is not None and w.result['cancelled']
 dialog=w.show_sample_points(node);assert dialog.snapshot is not None
 assert dialog.snapshot['steps']==w.result['steps']
 assert w.show_sample_points(next(n for n in w.scene.nodes() if n.kind=='domain')) is None
finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=120,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        self.assertNotIn('Traceback',result.stderr)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_snapshot_copies_only_a_detached_page(self):
        points=torch.arange(1000.,device='cuda').reshape(500,2).requires_grad_()
        weights=torch.ones(500,1,device='cuda')
        result=snapshot_page(points,weights,initial_count=500,boundary_count=2,method='importance',steps=1,history=[],page=2)
        self.assertEqual(len(result['rows']),200);self.assertEqual(result['rows'][0]['coordinates'],[400.,401.])
        self.assertIsNone(points.grad);self.assertTrue(points.is_cuda and points.requires_grad)
