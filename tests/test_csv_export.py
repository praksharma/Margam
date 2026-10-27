"""CSV is requested by a block and written only through an explicit export action."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class CSVExportTests(unittest.TestCase):
    def test_explicit_export(self):
        code=r'''
import time
from tempfile import TemporaryDirectory
from pathlib import Path
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
from margam.components.prepared_data import read_points
app=QApplication([]);w=Window(state_directory=False);w.show();w.change('shape','Interval');w.change('iterations',25)
method=next(n for n in w.scene.nodes() if n.kind=='method')
export=w.add_node('CSV Export');w.scene.connect_nodes(method,export)
assert not w.export_block_data(export) and not w.output_views.entries
assert w.start_training()
deadline=time.monotonic()+20
while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
assert w.result is not None and export.identifier in w.result['outputs'] and not w.output_views.entries
with TemporaryDirectory() as directory:
    path=Path(directory)/'prediction.csv'
    assert not path.exists() and w.export_block_data(export,path)
    points,_,values=read_points({'path':str(path),'coordinates':'x','normals':'','values':'u'},1)
    assert len(points)==101 and values[:,0].tolist()==w.result['outputs'][export.identifier]['values']
    model=Path(directory)/'export.margam.json'
    assert w.save_model(model) and w.load_model(model)
    assert any(n.kind=='export' for n in w.scene.nodes()) and not w.output_views.entries
w.undo_stack.setClean();w.close();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
