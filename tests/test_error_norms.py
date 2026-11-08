"""Exact discrete norm definitions, numerical range and read-only UI lifecycle."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np
from margam.components.error_norms import error_norms


class ErrorNormTests(unittest.TestCase):
    def test_scalar_norms_and_vector_magnitudes(self):
        reference=[[3.,0.,0.],[0.,4.,0.]];prediction=[[6.,4.,0.],[0.,4.,0.]]
        fields=['u','v','p']
        expected=dict(relative_l2=1.,relative_max=1.25,rms=math.sqrt(12.5),mean=2.5,max=5.,l2=5.)
        for norm,value in expected.items():
            result=error_norms(prediction,reference,fields=fields,selected=['u','v'],norm=norm)
            self.assertAlmostEqual(result['value'],value,places=13);self.assertEqual(result['status'],'defined')
            self.assertEqual(result['rows'],2);self.assertEqual(result['fields'],['u','v'])
        scalar=error_norms([2,4],[1,2],norm='rms');self.assertAlmostEqual(scalar['value'],math.sqrt(2.5))
        self.assertEqual(error_norms(prediction,reference,fields=fields,selected=['u'],norm='l2')['value'],3.)
        self.assertEqual(error_norms(prediction,reference,fields=fields,selected=['v'],norm='max')['value'],4.)

    def test_zero_reference_conventions_and_json(self):
        for policy in ('undefined','zero_if_exact'):
            result=error_norms([1,0],[0,0],zero_reference=policy)
            self.assertIsNone(result['value']);self.assertEqual(result['status'],'undefined_zero_reference')
            json.dumps(result,allow_nan=False)
        self.assertIsNone(error_norms([0,0],[0,0])['value'])
        exact=error_norms([0,0],[0,0],zero_reference='zero_if_exact')
        self.assertEqual(exact['value'],0.);self.assertEqual(exact['status'],'defined_zero_reference')
        self.assertEqual(error_norms([0,0],[0,0],norm='l2')['value'],0.)
        self.assertEqual(error_norms([1,2],[1,2])['value'],0.)

    def test_large_tiny_and_out_of_range_values(self):
        self.assertEqual(error_norms([1e308],[-1e308])['value'],2.)
        self.assertEqual(error_norms([1e308,1e308],[-1e308,-1e308],norm='relative_max')['value'],2.)
        overflow=error_norms([1e308],[-1e308],norm='l2')
        self.assertIsNone(overflow['value']);self.assertEqual(overflow['status'],'overflow')
        self.assertAlmostEqual(error_norms([4e-200],[2e-200])['value'],1.)
        self.assertEqual(error_norms([1e-200],[0.],norm='l2')['value'],1e-200)
        tiny=float(np.nextafter(0.,1.));maximum=float(np.finfo(float).max)
        underflow=error_norms([[tiny,maximum]],[[0.,maximum]])
        self.assertIsNone(underflow['value']);self.assertEqual(underflow['status'],'underflow')
        overflow=error_norms([maximum],[tiny]);self.assertEqual(overflow['status'],'overflow')
        self.assertFalse(overflow['reference_zero'])
        for result in (overflow,underflow):json.dumps(result,allow_nan=False)

    def test_shape_selection_and_values_are_explicit(self):
        for prediction,reference in (([],[]),([[1,2]],[1]),([float('nan')],[0]),([1j],[0]),(['1'],[0]),([True],[False]),(np.ma.array([1.],mask=[True]),[0])):
            with self.subTest(prediction=prediction):
                with self.assertRaises(ValueError):error_norms(prediction,reference)
        for options in (dict(fields=['u','u']),dict(fields=['u']),dict(selected=[]),dict(selected=['missing']),dict(selected=['u','u']),dict(norm='unknown'),dict(zero_reference='epsilon')):
            with self.assertRaises(ValueError):error_norms([[1,2]],[[2,3]],**options)
        prediction=np.array([[3.,4.],[0.,0.]]);original=prediction.copy()
        error_norms(prediction,np.zeros_like(prediction),norm='max')
        np.testing.assert_array_equal(prediction,original)

    def test_research_tensor_inputs_remain_detached(self):
        import torch
        from margam.research import error_norms as research_norms
        p=torch.tensor([2.,4.],requires_grad=True);r=torch.tensor([1.,2.])
        self.assertEqual(research_norms(p,r)['value'],1.);self.assertIsNone(p.grad)
        self.assertEqual(research_norms(p.to(torch.bfloat16),r.to(torch.bfloat16))['value'],1.)


class ErrorNormUITests(unittest.TestCase):
    def run_script(self,code):
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=120,
            env=dict(os.environ,QT_QPA_PLATFORM='offscreen',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr);self.assertNotIn('Traceback',result.stderr)

    def test_vector_selection_zero_policy_and_readonly_report(self):
        self.run_script(r'''
from copy import deepcopy
from unittest.mock import patch
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
from margam.ui.model_document import capture_model
app=QApplication([]);w=Window(state_directory=False);w.set_application_device('cpu');w.load_flow_template()
try:
 node=next(n for n in w.scene.nodes() if n.kind=='comparison');config=w.scene.compile_config()
 data=dict(points=[[0,0],[1,1]],values=[[6.,4.,0.],[0.,4.,0.]],reference=[[3.,0.,0.],[0.,4.,0.]],fields=['u','v','p'],config=config,steps=2,reference_name='<b>Explicit fixture</b>')
 w.result=dict(outputs={node.identifier:data},steps=2,cancelled=False,device='cpu',config=config)
 before=capture_model(w);index=w.undo_stack.index();result=w.result
 with patch('margam.components.evaluation_data.read_point_data',side_effect=AssertionError('No input reads')):
  dialog=w.show_error_norms(node)
  assert dialog.reports[0]['fields']==['u'] and dialog.reports[0]['value']==1.
  dialog.fields.item(1).setCheckState(Qt.CheckState.Checked)
  assert dialog.reports[0]['fields']==['u','v'] and dialog.reports[0]['value']==1. and len(dialog.reports)==3
  dialog.norm.setCurrentIndex(dialog.norm.findData('rms'));assert abs(dialog.reports[0]['value']-12.5**.5)<1e-12
  assert not dialog.zero.isEnabled()
  dialog.fields.item(0).setCheckState(Qt.CheckState.Unchecked);dialog.fields.item(1).setCheckState(Qt.CheckState.Unchecked)
  assert not dialog.reports and 'Select at least' in dialog.status.text()
  dialog.fields.item(2).setCheckState(Qt.CheckState.Checked);dialog.norm.setCurrentIndex(dialog.norm.findData('relative_l2'))
  assert dialog.reports[0]['value'] is None
  dialog.zero.setCurrentIndex(dialog.zero.findData('zero_if_exact'));assert dialog.reports[0]['value']==0.
 assert capture_model(w)==before and w.undo_stack.index()==index and w.result is result
 assert dialog.status.textFormat()==Qt.TextFormat.PlainText
 assert 'Error norms…' in [a.text() for a in w.create_block_menu(node).actions()]
 other=next(n for n in w.scene.nodes() if n.kind=='network');assert w.show_error_norms(other) is None
 assert 'Error norms…' not in [a.text() for a in w.create_block_menu(other).actions()]
 w.rename_block(node,'Velocity check');assert 'Velocity check' in dialog.windowTitle() and dialog.reports
 # Semantic edits invalidate report data immediately; Undo does not resurrect it.
 w.change('viscosity',.2);assert not dialog.reports and dialog.data is None
 w.undo_edit();assert not dialog.reports
 w.scene.clearSelection();node.setSelected(True);w.delete_selected();assert node.identifier not in w.error_norm_dialogs
finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()
''')

    def test_real_run_archive_refresh_and_reference_free_output(self):
        self.run_script(r'''
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
from margam.ui.model_document import capture_model
app=QApplication([]);w=Window(state_directory=False);w.set_application_device('cpu');w.load_heat_template();w.change('iterations',2);w.change('layers',(4,))
try:
 node=next(n for n in w.scene.nodes() if n.kind=='comparison');dialog=w.show_error_norms(node)
 assert not dialog.reports
 assert w.start_training()
 while w.worker is not None:app.processEvents()
 before=capture_model(w);index=w.undo_stack.index();result=w.result;run=w.last_run;history=deepcopy(run.model.loss_history)
 dialog.refresh();assert dialog.reports and dialog.reports[0]['rows']==len(result['outputs'][node.identifier]['points'])
 value=dialog.reports[0]['value'];assert capture_model(w)==before and w.undo_stack.index()==index and w.result is result and w.last_run is run and run.model.loss_history==history
 with TemporaryDirectory() as directory:
  archive=Path(directory)/'norms.pt';assert w.save_current_run(archive)
  w.undo_stack.setClean() # fixture explicitly accepts replacement; preserve real discard prompts
  assert w.open_saved_run(archive);assert not dialog.reports
  dialog.refresh();assert dialog.reports[0]['value']==value
 assert w.start_training();assert not dialog.reports
 while w.worker is not None:app.processEvents()
 assert not dialog.reports # new data requires explicit Refresh
 dialog.refresh();assert dialog.reports
finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()
''')
