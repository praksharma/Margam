"""Bounded design tables use block-owned settings without training or mutations."""
from copy import deepcopy
import json
import os
import subprocess
import sys
import time
import unittest
from unittest.mock import patch
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
from margam.ui.model_document import capture_model
from margam.studies import catalogue,values_for,preview


class StudyPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
    def setUp(self):self.w=Window(state_directory=False)
    def tearDown(self):self.w.undo_stack.setClean();self.w.close();self.app.clipboard().clear();self.app.processEvents()
    def axis(self,key,text,mode='candidates'):
        row=next(row for row in catalogue(capture_model(self.w)) if row['parameter']==key)
        return dict(node_id=row['node_id'],parameter=key,mode=mode,text=text)
    def test_cartesian_order_budgets_invalid_trials_and_no_mutation(self):
        document=capture_model(self.w);before=deepcopy(document);index=self.w.undo_stack.index()
        axes=[self.axis('iterations','[2,3]'),self.axis('layers','[[8],[16,16],[-1]]')]
        with patch('margam.templates.poisson.build_pinn',side_effect=AssertionError('No network allocation')),patch('margam.templates.poisson.sample_problem',side_effect=AssertionError('No sampling')):
            report=preview(document,axes,scene=self.w.scene)
        self.assertEqual((report['total'],report['valid'],report['invalid']),(6,4,2));self.assertEqual(report['total_updates'],10)
        self.assertEqual([row['design_point'] for row in report['trials']],['DP0001','DP0002','DP0003','DP0004','DP0005','DP0006'])
        self.assertEqual([row['choices'][0]['value'] for row in report['trials']],[2,2,2,3,3,3])
        self.assertTrue(report['trials'][2]['reason']);self.assertIsNone(report['trials'][2]['updates'])
        self.assertEqual(document,before);self.assertEqual(capture_model(self.w),before);self.assertEqual(self.w.undo_stack.index(),index)
        self.assertEqual(preview(document,axes,scene=self.w.scene),report)
    def test_parse_ranges_lists_limits_and_rejected_payloads(self):
        self.assertEqual(values_for('lr','range','[0.001,0.003,0.001]'),[.001,.002,.003])
        self.assertEqual(values_for('iterations','range','[5,1,-2]'),[5,3,1])
        self.assertEqual(values_for('iterations','range','[1,6,2]'),[1,3,5])
        self.assertEqual(values_for('network_seed','fixed','null'),[None])
        self.assertEqual(values_for('layers','fixed','[8,8]'),[[8,8]])
        for key,mode,text in [('iterations','fixed','true'),('iterations','candidates','[1,1]'),('iterations','range','[1,1000,1]'),('iterations','range','[1,5,0]'),('iterations','range','[1,5,-1]'),('lr','fixed','NaN'),('lr','fixed','1e400'),('lr','candidates','[1,1.0]'),('lr','range','['+'9'*400+ ',1,1]'),('layers','range','[1,5,1]'),('layers','candidates','[8,16]'),('seed','fixed','null'),('lr','fixed','"__import__(\"os\")"'),('iterations','candidates','[]')]:
            with self.subTest(text=text),self.assertRaises(ValueError):values_for(key,mode,text)
        doc=capture_model(self.w);a=self.axis('iterations','[1,2]')
        with self.assertRaises(ValueError):preview(doc,[a,a])
        with self.assertRaises(ValueError):preview(doc,[dict(a,node_id='deleted')])
        with self.assertRaises(ValueError):preview(doc,[self.axis('iterations','[1,64,1]','range'),self.axis('seed','[1,9,1]','range')])
    def test_dialog_preview_staleness_close_reopen_and_missing_owner(self):
        dialog=self.w.show_design_points();self.app.processEvents();dialog.add_axis()
        row=self.axis('iterations','[2,4]');combo=dialog.axes.cellWidget(0,0);combo.setCurrentIndex(combo.findData(json.dumps([row['node_id'],'iterations'])))
        dialog.axes.cellWidget(0,1).setCurrentIndex(1);dialog.axes.cellWidget(0,2).setText('[2,4]')
        before=capture_model(self.w);index=self.w.undo_stack.index();self.assertTrue(dialog.refresh());self.assertEqual(dialog.table.rowCount(),2)
        self.assertEqual(dialog.result['trials'][0]['choices'][0]['parameter'],'iterations')
        self.assertEqual(dialog.result['trials'][0]['choices'][0]['node_id'],row['node_id'])
        self.assertEqual(capture_model(self.w),before);self.assertEqual(self.w.undo_stack.index(),index);self.assertIsNone(self.w.worker)
        dialog.close();self.assertFalse(dialog.connected);self.w.change('lr',.002)
        self.assertIs(self.w.show_design_points(),dialog);self.assertTrue(dialog.connected);self.assertIsNone(dialog.result)
        self.w.change('iterations',5);self.app.processEvents();self.assertIsNone(dialog.result)
        self.assertTrue(dialog.refresh());self.w.scene.locked=True;self.assertFalse(dialog.refresh());self.w.scene.locked=False
        combo=dialog.axes.cellWidget(0,0);combo.addItem('Missing',json.dumps(['deleted','iterations']));combo.setCurrentIndex(combo.count()-1)
        self.assertFalse(dialog.refresh());self.assertIn('no longer exists',dialog.status.text())
        self.assertEqual(dialog.axes.rowCount(),1)
    def test_preview_preserves_trained_results_rng_and_rejects_ineffective_controls(self):
        import random
        import numpy as np
        import torch
        self.w.change('iterations',2);self.w.change('layers',(4,));self.w.set_application_device('cpu')
        self.assertTrue(self.w.start_training());deadline=time.monotonic()+30
        while self.w.worker is not None and time.monotonic()<deadline:self.app.processEvents();time.sleep(.005)
        self.assertIsNone(self.w.worker);self.assertIsNotNone(self.w.result)
        result,run=self.w.result,self.w.last_run;document=capture_model(self.w);index=self.w.undo_stack.index()
        py=random.getstate();numpy=np.random.get_state();rng=torch.get_rng_state().clone()
        dialog=self.w.show_design_points();self.app.processEvents();dialog.add_axis()
        self.assertTrue(dialog.refresh())
        self.assertIs(self.w.result,result);self.assertIs(self.w.last_run,run)
        self.assertEqual(capture_model(self.w),document);self.assertEqual(index,self.w.undo_stack.index())
        self.assertEqual(random.getstate(),py);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        after=np.random.get_state();self.assertEqual(numpy[0],after[0]);np.testing.assert_array_equal(numpy[1],after[1]);self.assertEqual(numpy[2:],after[2:])
        fake=deepcopy(document)
        method=next(node for node in fake['nodes'] if node['kind']=='method')
        method['kind']='dsgnar_method'
        self.assertFalse(any(item['node_id']==method['id'] and item['parameter']=='lr' for item in catalogue(fake)))
        method['settings']['parameters']['expert_settings']={'windows':2}
        self.assertFalse(any(item['node_id']==method['id'] and item['parameter']=='iterations' for item in catalogue(fake)))

    def test_architecture_compatibility_compact_recipe_and_no_torch_or_qt_in_planner_import(self):
        self.w.change_network_type('dgm')
        report=preview(capture_model(self.w),[self.axis('layers','[[8,8],[8,16]]')],scene=self.w.scene)
        self.assertEqual((report['valid'],report['invalid']),(1,1));self.assertIn('equal',report['trials'][1]['reason'])
        method=next(node for node in self.w.scene.nodes() if node.kind=='method')
        self.assertTrue(self.w.collapse_recipe(method))
        report=preview(capture_model(self.w),[self.axis('layers','[[8,8],[16,16]]')],scene=self.w.scene)
        self.assertEqual(report['valid'],2)
        code="import sys; import margam.studies; assert 'torch' not in sys.modules; assert 'PySide6' not in sys.modules"
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)
        document=capture_model(self.w);axis=self.axis('iterations','[1,2]')
        code="import sys,json; from margam.studies import preview; data=json.load(sys.stdin); assert preview(data['document'],data['axes'])['valid']==2; assert 'torch' not in sys.modules; assert 'PySide6' not in sys.modules"
        result=subprocess.run([sys.executable,'-c',code],input=json.dumps(dict(document=document,axes=[axis])),capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
