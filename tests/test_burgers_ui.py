"""Burgers example has explicit time/initial labels, optional tables and PVD export."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class BurgersUITests(unittest.TestCase):
    def test_example_controls_worker_table_archive_and_export(self):
        code=r'''import time
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import asdict
from PySide6.QtWidgets import QApplication,QComboBox,QLabel
from margam.ui.app import Window
from margam.ui.model_document import capture_model,decode_document
from margam.ui.output_views import VectorResultView
app=QApplication([]);w=Window(state_directory=False);w.show();app.processEvents()
try:
 assert w.example_actions['burgers_1d'] in w.examples_1d_menu.actions()
 w.load_example('burgers_1d');assert w.validate_model(),w.validation.text()
 assert not any(n.kind in ('reference','comparison') for n in w.scene.nodes())
 assert not w.output_views.entries
 domain=next(n for n in w.scene.nodes() if n.kind=='domain')
 w.scene.clearSelection();domain.setSelected(True);app.processEvents()
 assert domain.title=='1D space–time'
 assert 't start' in [label.text() for label in w.property_scroll.findChildren(QLabel)]
 boundary=next(n for n in w.scene.nodes() if n.kind=='boundary')
 w.scene.clearSelection();boundary.setSelected(True);app.processEvents()
 choice=w.property_scroll.findChild(QComboBox,'boundary_side')
 assert choice.count()==3 and choice.findText('initial (t start)')>=0
 assert choice.findData('top')==-1
 choice.setCurrentIndex(choice.findData('bottom'));app.processEvents()
 bc=next(n for n in w.scene.nodes() if n.kind=='bc');w.scene.clearSelection();bc.setSelected(True);app.processEvents()
 assert 'initial (t start) target expression' in [label.text() for label in w.property_scroll.findChildren(QLabel)]
 w.change('points',25);w.change('boundary_points',6);w.change('boundary_sampling',{});w.change('layers',(8,));w.change('iterations',5)
 for node in w.scene.nodes():
    if node.kind in ('solution','export','paraview'):node.settings['resolution']=5
 before=asdict(w.scene.compile_config());trainer=next(n for n in w.scene.nodes() if n.kind=='method')
 compact=w.collapse_recipe(trainer);assert asdict(w.scene.compile_config())==before
 w.expand_recipe(compact);assert asdict(w.scene.compile_config())==before
 document=capture_model(w);decode_document(document)
 with TemporaryDirectory() as directory:
    path=Path(directory)/'burgers.json';assert w.save_model(path) and w.load_model(path)
    assert asdict(w.scene.compile_config())==before
    assert w.start_training()
    def finish():
        deadline=time.monotonic()+20
        while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
        assert w.worker is None and w.result is not None,w.log.toPlainText()
    finish();assert not w.output_views.entries
    solution=next(n for n in w.scene.nodes() if n.kind=='solution');w.open_block(solution)
    view=w.output_views.entries[solution.identifier][1];assert isinstance(view,VectorResultView)
    assert [view.table.horizontalHeaderItem(i).text() for i in range(3)]==['x','t','u']
    export=next(n for n in w.scene.nodes() if n.kind=='paraview')
    path=Path(directory)/'burgers.pvd';assert w.export_block_data(export,path)
    assert path.exists()
    run=Path(directory)/'burgers.pt';assert w.save_current_run(run)
    w.undo_stack.setClean();assert w.open_saved_run(run)
    assert w.config.is_time_dependent and not w.output_views.entries
    assert w.resume_training(2);finish();assert w.result['steps']==7
finally:
 app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stderr)
