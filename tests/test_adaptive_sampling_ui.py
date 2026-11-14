"""Adaptive sampler blocks follow normal editing, worker and archive paths."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class AdaptiveSamplingUITests(unittest.TestCase):
    def test_blocks_controls_history_worker_archive_and_recipe(self):
        code=r'''import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from PySide6.QtWidgets import QApplication,QSpinBox,QDoubleSpinBox,QComboBox
from margam.ui.app import Window
from margam.ui.graph_model import parameters_for
from margam.ui.model_document import capture_model,decode_document
from margam.templates.poisson import build_pinn
app=QApplication([]);w=Window(state_directory=False);w.show();app.processEvents()
try:
 for title,kind in [('Residual refinement (RAR)','rar'),('Importance sampling','importance')]:
    w.load_example('interval_poisson')
    old=next(n for n in w.scene.nodes() if n.kind=='interior')
    w.delete_block(old)
    sampler=w.add_node(title)
    domain=next(n for n in w.scene.nodes() if n.kind=='domain')
    pde=next(n for n in w.scene.nodes() if n.kind=='predefined_pde')
    w.scene.connect_nodes(domain,sampler);w.scene.connect_nodes(sampler,pde)
    assert w.validate_model(),w.validation.text()
    w.scene.clearSelection();sampler.setSelected(True);app.processEvents()
    if kind=='importance':
        assert w.property_scroll.findChild(QDoubleSpinBox,'sampling_mixture') is not None
        assert w.property_scroll.findChild(QSpinBox,'sampling_batch') is None
    control=w.property_scroll.findChild(QSpinBox,'sampling_interval');assert control
    control.setValue(2);app.processEvents();assert w.scene.compile_config().adaptive_sampling['interval']==2
    w.undo_edit();assert w.scene.compile_config().adaptive_sampling['interval']==100
    w.redo_edit();assert w.scene.compile_config().adaptive_sampling['interval']==2
    if kind=='importance':
        pool=w.property_scroll.findChild(QComboBox,'sampling_pool');assert pool.currentData()=='refresh'
        pool.setCurrentIndex(pool.findData('fixed'));app.processEvents()
        assert w.scene.compile_config().adaptive_sampling['pool']=='fixed'
        score=w.property_scroll.findChild(QComboBox,'sampling_score');assert score.currentData()=='residual'
        score.setCurrentIndex(score.findData('field_gradient'));app.processEvents()
        assert w.scene.compile_config().adaptive_sampling['score']=='field_gradient'
        assert w.property_scroll.findChild(QComboBox,'sampling_field').currentData()==0
        w.undo_edit();assert 'score' not in w.scene.compile_config().adaptive_sampling
        w.redo_edit();assert w.scene.compile_config().adaptive_sampling['score']=='field_gradient'
    w.change('points',6);w.change('layers',(8,));w.change('iterations',7)
    settings=deepcopy(w.scene.compile_config().adaptive_sampling);settings.update(candidates=19,chunk=4)
    if kind=='rar':settings.update(batch=3,maximum=15)
    w.change('adaptive_sampling',settings)
    document=capture_model(w);decode_document(document)
    # Clipboard preserves independent sampling parameters and undo removes only the copy.
    w.scene.clearSelection();sampler.setSelected(True)
    assert w.copy_blocks() and w.paste_blocks()
    duplicate=next(n for n in w.scene.nodes() if n.kind==kind and n is not sampler)
    assert duplicate.settings==sampler.settings
    w.undo_edit();assert len([n for n in w.scene.nodes() if n.kind==kind])==1
    method=next(n for n in w.scene.nodes() if n.kind=='method')
    before=asdict(w.scene.compile_config());compact=w.collapse_recipe(method)
    assert asdict(w.scene.compile_config())==before
    w.expand_recipe(compact);assert asdict(w.scene.compile_config())==before
    with TemporaryDirectory() as directory:
        path=Path(directory)/'adaptive.margam.json'
        assert w.save_model(path) and w.load_model(path)
        assert asdict(w.scene.compile_config())==before
        assert w.start_training()
        def finish():
            deadline=time.monotonic()+20
            while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
            assert w.worker is None and w.result is not None,w.log.toPlainText()
        finish()
        assert not w.output_views.entries
        assert 'Sampling at update 2' in w.log.toPlainText()
        assert len(w.last_run.model.sampling_controller.history)==3
        if kind=='importance':
            assert 'field-gradient magnitude' in w.log.toPlainText()
            current=next(n for n in w.scene.nodes() if n.kind=='importance')
            dialog=w.show_sample_points(current)
            assert 'mean_gradient' in dialog.snapshot['last_event']
            assert 'field-gradient magnitude' in dialog.status.text()
        archive=Path(directory)/'adaptive.margam-run.pt'
        assert w.save_current_run(archive)
        w.undo_stack.setClean();assert w.open_saved_run(archive)
        assert w.resume_training(2);finish()
        continuous=build_pinn(w.config);continuous.train(9,display_every=0)
        assert w.last_run.model.loss_history==continuous.loss_history
        assert w.last_run.model.sampling_controller.history==continuous.sampling_controller.history
 # Coupled field choices and the time-dependent guard use the same inspector.
 w.load_flow_template()
 old=next(n for n in w.scene.nodes() if n.kind=='interior');w.delete_block(old)
 sampler=w.add_node('Importance sampling')
 domain=next(n for n in w.scene.nodes() if n.kind=='domain');pde=next(n for n in w.scene.nodes() if n.kind=='pde')
 w.scene.connect_nodes(domain,sampler);w.scene.connect_nodes(sampler,pde)
 w.scene.clearSelection();sampler.setSelected(True);app.processEvents()
 score=w.property_scroll.findChild(QComboBox,'sampling_score');score.setCurrentIndex(score.findData('field_gradient'));app.processEvents()
 field=w.property_scroll.findChild(QComboBox,'sampling_field');assert [field.itemText(i) for i in range(field.count())]==['u','v','p']
 field.setCurrentIndex(field.findData(2));app.processEvents()
 assert w.scene.compile_config().adaptive_sampling['field']==2 and w.validate_model()
 score=w.property_scroll.findChild(QComboBox,'sampling_score');score.setCurrentIndex(score.findData('residual'));app.processEvents()
 assert 'field' not in w.scene.compile_config().adaptive_sampling and 'score' not in w.scene.compile_config().adaptive_sampling
 w.change('shape','SpaceTime1D');app.processEvents()
 score=w.property_scroll.findChild(QComboBox,'sampling_score');assert not score.model().item(score.findData('field_gradient')).isEnabled()
 # Old saved instance parameters gain the fixed default without changing wiring.
 w.reset_template();legacy=capture_model(w);legacy['version']=12
 legacy['config'].pop('adaptive_sampling')
 for n in legacy['nodes']:
    if n['kind']=='interior':n['settings']['parameters'].pop('adaptive_sampling')
 c,nodes,wires=decode_document(legacy)
 assert c.adaptive_sampling=={} and next(n for n in nodes if n['kind']=='interior')['settings']['parameters']['adaptive_sampling']=={}
finally:
 app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],
            env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stderr)
