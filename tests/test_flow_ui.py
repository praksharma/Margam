"""Coupled flow stays explicit across UI, tables, CSV, models and checkpoints."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class FlowUITests(unittest.TestCase):
    def test_flow_template_outputs_and_persistence(self):
        code=r'''
import time
from tempfile import TemporaryDirectory
from pathlib import Path
from PySide6.QtWidgets import QApplication,QLineEdit
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from margam.ui.app import Window
from margam.ui.output_views import VectorResultView
from margam.components.prepared_data import read_points
app=QApplication([]);w=Window(state_directory=False);w.show();w.flow_action.trigger();app.processEvents()
assert w.config.field_names==('u','v','p') and len(w.scene.nodes())==12
assert w.validate_model() and not w.output_views.entries
w.change('iterations',100)
assert w.start_training()
deadline=time.monotonic()+25
while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
assert w.worker is None and w.result is not None,w.log.toPlainText()
comparison=next(n for n in w.scene.nodes() if n.kind=='comparison')
monitor=next(n for n in w.scene.nodes() if n.kind=='monitor')
export=next(n for n in w.scene.nodes() if n.kind=='export')
data=w.result['outputs'][comparison.identifier]
assert data['fields']==['u','v','p'] and set(data['rmse_by_field'])=={'u','v','p'}
assert 'pressure_gauge' in w.monitor_records[monitor.identifier][-1][1]
w.open_block(comparison);app.processEvents()
assert isinstance(w.output_views.entries[comparison.identifier][1],VectorResultView)
assert w.output_views.entries[comparison.identifier][1].table.columnCount()==11
w.open_block(monitor);app.processEvents()
assert len(w.output_views.entries[monitor.identifier][1].series)==4
with TemporaryDirectory() as directory:
    csv=Path(directory)/'flow.csv';assert w.export_block_data(export,csv)
    points,_,values=read_points({'path':str(csv),'coordinates':'x,y','normals':'','values':'u,v,p'},2,3)
    assert values.shape[1]==3 and len(points)==len(w.result['outputs'][export.identifier]['points'])
    from margam.ui.data_mapping import DataMappingDialog
    dialog=DataMappingDialog(2,{'path':str(csv),'coordinates':'x,y','normals':'','values':''},w,components=3,require_values=True)
    assert not dialog.check()
    dialog.fields['values'].setText('u,v,p');assert dialog.check();dialog.reject()
    archive=Path(directory)/'flow.margam-run.pt';assert w.save_current_run(archive)
    w.undo_stack.setClean();assert w.open_saved_run(archive)
    assert w.config.physics=='incompressible_2d' and not w.output_views.entries
    assert w.loaded_checkpoint is not None
    model=Path(directory)/'flow.margam.json';assert w.save_model(model) and w.load_model(model)
    assert w.scene.compile_config().field_names==('u','v','p') and w.validate_model()
    bc=next(n for n in w.scene.nodes() if n.kind=='bc');w.show_block_properties(bc)
    edit=w.property_scroll.widget().findChild(QLineEdit,'flow_left_u_target')
    edit.setText('y + 0');QTest.keyClick(edit,Qt.Key.Key_Return)
    assert w.config.flow_boundary['left']['u']['target']=='y + 0'
    reference=next(n for n in w.scene.nodes() if n.kind=='reference')
    w.change_output(reference,'reference_expression','y; 0; 0')
    assert w.validate_model()
    request=next(r for r in w.scene.output_requests(w.config) if r.kind=='comparison')
    values=request.reference.evaluate([[.2,.3]])
    assert abs(values[0][0]-.3)<1e-6 and values[0][1:]==[0.,0.]
w.undo_stack.setClean();w.close();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=40)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertNotIn('Traceback',result.stderr)
