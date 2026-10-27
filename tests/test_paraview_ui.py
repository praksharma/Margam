"""ParaView export is an optional block, independent of training sample placement."""
import os,subprocess,sys,unittest

class ParaViewUITests(unittest.TestCase):
    def test_opt_in_random_samples_archive_and_explicit_export(self):
        code=r'''
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import xml.etree.ElementTree as ET
from PySide6.QtWidgets import QApplication,QMessageBox
from margam.ui.app import Window
from margam.ui.model_document import capture_model,decode_document
from margam.ui.run_document import load_run
app=QApplication([]);w=Window(state_directory=False);w.show();app.processEvents()
def warning(*args):raise AssertionError(str(args[1:]))
QMessageBox.warning=warning
assert not any(n.kind=='paraview' for n in w.scene.nodes())
w.change('shape','Interval');w.change('sampling','random');w.change('iterations',25)
trainer=next(n for n in w.scene.nodes() if n.kind=='method')
export=w.add_node('ParaView Export');assert not w.export_block_data(export)
w.scene.connect_nodes(trainer,export);w.change_output(export,'resolution',17)
w.scene.clearSelection();export.setSelected(True)
assert w.copy_blocks() and w.paste_blocks();w.undo_edit()
config=w.scene.compile_config();assert config.sampling=='random'
with TemporaryDirectory() as directory:
    directory=Path(directory);definition=directory/'model.margam.json'
    assert w.save_model(definition) and w.load_model(definition)
    export=next(n for n in w.scene.nodes() if n.kind=='paraview')
    assert w.start_training()
    deadline=time.monotonic()+30
    while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
    assert w.worker is None and w.result is not None,w.log.toPlainText()
    assert not w.output_views.entries and not list(directory.glob('*.vti'))
    assert w.result['outputs'][export.identifier]['points']==[[i/16] for i in range(17)]
    path=directory/'prediction.vti';assert w.export_block_data(export,path)
    image=ET.parse(path).find('ImageData');assert image.get('WholeExtent')=='0 16 0 0 0 0'
    assert image.find('FieldData') is None
    run=directory/'run.margam-run.pt';assert w.save_current_run(run)
    assert export.identifier in load_run(run)['result']['outputs']
    assert w.open_saved_run(run) and not w.output_views.entries
    export=next(n for n in w.scene.nodes() if n.kind=='paraview')
    with patch('margam.ui.app.QFileDialog.getSaveFileName',return_value=(str(directory/'dialog-output'),'')):
        menu=w.create_block_menu(export);next(a for a in menu.actions() if a.text()=='Export ParaView…').trigger()
    assert (directory/'dialog-output.vti').exists()
    # Old models remain readable; there were no ParaView blocks by default.
    w.undo_stack.setClean();w.reset_template();legacy=capture_model(w);legacy['version']=11;decode_document(legacy)
    w.load_flow_template();w.change('sampling','random');w.change('iterations',5)
    trainer=next(n for n in w.scene.nodes() if n.kind=='method')
    export=w.add_node('ParaView Export');w.scene.connect_nodes(trainer,export);w.change_output(export,'resolution',5)
    assert w.start_training()
    deadline=time.monotonic()+30
    while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
    assert w.worker is None and w.result is not None,w.log.toPlainText()
    assert not w.output_views.entries
    vector=directory/'flow.vti';assert w.export_block_data(export,vector)
    pd=ET.parse(vector).find('ImageData/Piece/PointData')
    assert pd.get('Vectors')=='velocity'
    assert [a.get('Name') for a in pd]==['u','v','p','velocity']

app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=80)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
