"""Manual coefficients, objective contributions and checkpoint identity."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import unittest
import torch
from margam.templates.poisson import PoissonConfig,build_pinn,BoundarySampling
from margam.templates.examples import example_config


class BoundaryWeightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_unit_defaults_manual_objective_and_gradients(self):
        c=PoissonConfig(shape='Interval',points=8,boundary_points=2,layers=(4,4),boundary_weight=6.)
        default=build_pinn(c);unit=deepcopy(c);unit.bc_weights=dict(left=1.,right=1.)
        equal=build_pinn(unit)
        self.assertEqual(default.weights,equal.weights)
        self.assertEqual(default.data_fingerprint(),equal.data_fingerprint())
        default.train(2,0);equal.train(2,0);self.assertEqual(default.loss_history,equal.loss_history)
        c.bc_weights=dict(left=2.,right=.25);model=build_pinn(c)
        terms=model.compute_losses()
        expected=terms['pde']+6.*terms['left']+.75*terms['right']
        from margam.components.losses import weighted_loss
        actual=weighted_loss(terms,model.weights)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        parameters=tuple(model.network.parameters())
        expected_grad=torch.autograd.grad(expected,parameters,retain_graph=True)
        actual_grad=torch.autograd.grad(actual,parameters)
        for a,b in zip(actual_grad,expected_grad):torch.testing.assert_close(a,b,rtol=0,atol=0)
        model.train(1,0);row=model.loss_history[0]
        self.assertAlmostEqual(row['weighted:left'],6.*row['left'],places=5)
        self.assertAlmostEqual(row['weighted:right'],.75*row['right'],places=5)
        for invalid in ({'left':0.},{'left':float('nan')},{'left':True},{'pde':1.},{'left':1e7}):
            c.bc_weights=invalid
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):c.validate()

    def check_methods(self,device):
        for method in ('baseline','self_adaptive','lra','expert','scale','gpinn','rba'):
            with self.subTest(method=method,device=device):
                c=example_config('expert_advection') if method=='expert' else PoissonConfig(shape='Interval',points=8,boundary_points=2,layers=(4,4))
                c.method=method;c.device=device;c.points=8;c.boundary_points=2;c.layers=(4,4)
                c.expert_settings.update(chunks=2,balance_every=1)
                c.boundary_sampling={side:BoundarySampling(2,c.boundary_recipe(side).method,c.boundary_recipe(side).seed) for side in c.sides}
                c.bc_weights={name:2. if index==0 else .5 for index,name in enumerate(c.constraint_names)}
                if method=='self_adaptive':c.sa_boundary=True;c.sa_initial=2.
                if method=='scale':c.scale_settings['fresh']=False
                full=build_pinn(c);full.train(4,0)
                part=build_pinn(c);part.train(2,0);resumed=build_pinn(c)
                resumed.load_training_state(part.training_state());resumed.train(2,0)
                self.assertEqual(full.loss_history,resumed.loss_history)
                for name,value in full.network.state_dict().items():torch.testing.assert_close(value,resumed.network.state_dict()[name],rtol=0,atol=0)
                row=full.loss_history[-1]
                for name in c.constraint_names:
                    self.assertGreaterEqual(row['weighted:'+name],0.)
                    multiplier=full.weights[name]
                    if method=='lra':multiplier*=full.lra_weights[name]
                    if method=='expert':multiplier*=full.coefficients['initial' if name=='initial' else 'boundary']
                    if method!='self_adaptive':self.assertAlmostEqual(row['weighted:'+name],row[name]*multiplier,delta=max(1e-6,row['weighted:'+name]*1e-6))
                if method=='self_adaptive':
                    row=full.loss_history[0]
                    for name in c.constraint_names:self.assertAlmostEqual(row['weighted:'+name],row[name]*full.weights[name]*4.,delta=max(1e-6,row['weighted:'+name]*1e-6))
                altered=deepcopy(c);altered.bc_weights[c.constraint_names[0]]=3.
                with self.assertRaises(ValueError):build_pinn(altered).load_training_state(part.training_state())

    def test_all_trainers_cpu(self):self.check_methods('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_all_trainers_cuda(self):self.check_methods('cuda')

    def test_ui_model_monitor_and_legacy(self):
        code=r'''
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from copy import deepcopy
from PySide6.QtWidgets import QApplication,QDoubleSpinBox
from margam.ui.app import Window
from margam.ui.model_document import capture_model,decode_document
app=QApplication([]);w=Window(state_directory=False)
def finish():
 deadline=time.monotonic()+30
 while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
 assert w.worker is None and w.result is not None,w.log.toPlainText()
try:
 assert w.load_training_data_example();w.set_application_device('cpu')
 bc=next(n for n in w.scene.nodes() if n.kind=='bc');w.show_block_properties(bc)
 control=w.property_scroll.findChild(QDoubleSpinBox,'boundary_coefficient_slope');assert control
 control.setValue(2.5);assert w.scene.compile_config().bc_weights=={'slope':2.5}
 w.undo_edit();assert w.scene.compile_config().bc_weights=={}
 w.redo_edit();assert w.scene.compile_config().bc_weights=={'slope':2.5}
 doc=capture_model(w);assert doc['version']==28;decode_document(doc)
 old=deepcopy(doc);old['version']=26;old['config'].pop('bc_weights')
 for n in old['nodes']:n['settings'].get('parameters',{}).pop('bc_weights',None)
 migrated,records,_=decode_document(old);assert migrated.bc_weights=={}
 assert next(n for n in records if n['kind']=='bc')['settings']['parameters']['bc_weights']=={}
 with TemporaryDirectory() as directory:
  path=Path(directory)/'model.margam.json';assert w.save_model(path) and w.load_model(path)
  assert w.scene.compile_config().bc_weights=={'slope':2.5}
  w.change('iterations',2);w.change('layers',(4,4))
  monitor=next(n for n in w.scene.nodes() if n.kind=='monitor')
  assert w.start_training();finish();w.open_block(monitor)
  view=w.output_views.entries[monitor.identifier][1];view.constraint.setCurrentText('slope')
  assert len(view.detail.series)==2
  assert view.detail.series[1][1]==[(step,row['weighted:slope']) for step,row in w.monitor_records[monitor.identifier]]
  archive=Path(directory)/'run.pt';assert w.save_current_run(archive)
  w.undo_stack.setClean();assert w.open_saved_run(archive) and w.resume_training(1);finish()
  assert w.result['steps']==3
  w.change_boundary_coefficient('slope',3.);assert w.result is None and w.loaded_checkpoint is None
finally:
 app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=75,
                              env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
