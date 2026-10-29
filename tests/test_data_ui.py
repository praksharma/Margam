"""Mapped inputs and explicit reference expressions/data work through the editor."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class DataUITests(unittest.TestCase):
    def test_mapping_dialog_and_reference_sources(self):
        code=r'''
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
from margam.templates.poisson import sample_problem
app=QApplication([]);w=Window(state_directory=False);w.show();w.change('shape','Interval');w.change('iterations',25)
with TemporaryDirectory() as directory:
    path=Path(directory)/'pde.csv';path.write_text('position\n0.2\n0.5\n0.8\n')
    def map_dialog():
        dialog=app.activeModalWidget()
        dialog.fields['path'].setText(str(path));dialog.fields['coordinates'].setText('position')
        assert dialog.check() and '3 points' in dialog.preview.toPlainText()
        dialog.accept()
    QTimer.singleShot(0,map_dialog);w.edit_point_data()
    assert sample_problem(w.config)[0]==[[.2],[.5],[.8]]
    method=next(n for n in w.scene.nodes() if n.kind=='method')
    reference=w.add_node('Reference');comparison=w.add_node('Comparison')
    w.scene.connect_nodes(method,comparison);w.scene.connect_nodes(reference,comparison)
    w.change_output(reference,'reference_expression','x*(1-x)')
    with patch('margam.ui.app.make_poisson_reference',side_effect=AssertionError('Implicit preset')):
        assert w.start_training()
        deadline=time.monotonic()+20
        while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
    assert w.result is not None and w.result['outputs'][comparison.identifier]['reference_name']=='User reference expression'
    assert len(w.last_run.model.collocation_point_sample)==3+sum(len(points) for condition,points,target in w.last_run.model._constraints)
    state=w.last_run.model.training_state()
    path.write_text('position\n0.1\n0.5\n0.9\n')
    from margam.templates.poisson import build_pinn
    try:build_pinn(w.config).load_training_state(state)
    except ValueError as error:assert 'point data' in str(error)
    else:raise AssertionError('Changed external data accepted for resume')
    ref=Path(directory)/'ref.csv'
    ref.write_text('x,u\n'+''.join(f'{i/100},{i/100*(1-i/100)}\n' for i in range(101)))
    w.change_output(reference,'reference_expression','')
    w.change_output(reference,'reference_data',{'path':str(ref),'coordinates':'x','normals':'','values':'u'})
    request=w.scene.output_requests(w.config)[0]
    values=request.reference.evaluate([[i/100] for i in range(101)])
    assert abs(values[50]-.25)<1e-12
    model=Path(directory)/'mapped.margam.json'
    assert w.save_model(model) and w.load_model(model)
    assert w.config.interior_data['path']==str(path)
w.undo_stack.setClean();w.close();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
