"""Desktop-wide execution preference and expandable inspector descriptions."""
import os
from pathlib import Path
import subprocess
import sys
import unittest
import torch


class ApplicationDeviceTests(unittest.TestCase):
    def run_ui(self,code,timeout=45):
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=timeout)
        self.assertEqual(result.returncode,0,result.stderr+'\n'+result.stdout)

    def test_preferences_override_model_and_descriptions(self):
        self.run_ui(r'''
from pathlib import Path
from tempfile import TemporaryDirectory
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication,QToolButton,QTextBrowser,QDialog
from margam.ui.app import Window
app=QApplication([])
with TemporaryDirectory() as directory:
 w=Window(state_directory=directory);w.show();app.processEvents()
 assert w.application_device=='auto'
 assert w.set_application_device('cpu')
 trainer=next(n for n in w.scene.nodes() if n.kind=='method')
 trainer.settings['parameters']['device']='cuda'
 assert w.validate_model() and w.config.device=='cpu'
 path=Path(directory)/'model.margam.json';assert w.save_model(path) and w.load_model(path)
 assert w.scene.compile_config().device=='cpu'
 assert w.undo_stack.isClean()
 w.show_block_properties(next(n for n in w.scene.nodes() if n.kind=='method'));app.processEvents()
 buttons=w.property_scroll.findChildren(QToolButton)
 assert any(b.objectName()=='block_info' for b in buttons)
 learning=next(b for b in buttons if b.objectName()=='setting_info_adam_learning_rate')
 QTest.mouseClick(learning,Qt.MouseButton.LeftButton);app.processEvents()
 dialog=next(d for d in w.findChildren(QDialog) if d.objectName()=='information_dialog')
 text=dialog.findChild(QTextBrowser).toPlainText()
 assert 'learning rate' in text.lower() or 'step size' in text.lower()
 assert 'Allowed range' in text and 'Block description' in text
 dialog.close();app.processEvents()
 network=next(n for n in w.scene.nodes() if n.kind=='network');w.show_block_properties(network);app.processEvents()
 architecture=w.property_scroll.findChild(QToolButton,'setting_info_architecture');assert architecture
 architecture.click();app.processEvents()
 choices=next(d for d in w.findChildren(QDialog) if d.objectName()=='information_dialog' and d.isVisible())
 assert 'Available choices' in choices.findChild(QTextBrowser).toPlainText() and 'SIREN' in choices.findChild(QTextBrowser).toPlainText()
 choices.close();app.processEvents()
 count=len(w.scene.nodes())
 w.library.tree.setCurrentItem(w.library.tree.topLevelItem(0).child(0))
 assert w.library.info_button.isEnabled()
 library_info=w.library.show_description();assert library_info is not None and len(w.scene.nodes())==count;library_info.close()
 info=w.show_device_information();assert 'inference' in info.findChild(QTextBrowser).toPlainText();info.close()
 w.undo_stack.setClean();w.close();app.processEvents()
 other=Window(state_directory=directory)
 assert other.application_device=='cpu' and other.scene.compile_config().device=='cpu'
 other.undo_stack.setClean();other.close()
app.clipboard().clear();app.processEvents()
''')

    def test_unavailable_gpu_and_running_lock(self):
        self.run_ui(r'''
from unittest.mock import patch
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
app=QApplication([])
with patch('torch.cuda.is_available',return_value=False):
 w=Window(state_directory=False)
 assert w.execution_device()=='cpu' and not w.device_actions['cuda'].isEnabled()
 assert not w.set_application_device('cuda') and w.application_device=='auto'
 w.set_busy(True);assert not w.device_menu.isEnabled() and not w.set_application_device('cpu')
 w.set_busy(False);assert w.set_application_device('cpu')
 w.undo_stack.setClean();w.close()
app.clipboard().clear();app.processEvents()
''')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_training_inference_and_resume_follow_selection(self):
        self.run_ui(r'''
import time,torch
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
app=QApplication([]);w=Window(state_directory=False)
def finish():
 deadline=time.monotonic()+20
 while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
 assert w.worker is None and w.result is not None,w.log.toPlainText()
assert w.application_device=='auto' and w.execution_device()=='cuda'
assert w.set_application_device('cpu')
w.change('iterations',2);w.change('layers',(4,4));assert w.start_training();finish()
assert str(w.last_run.model.device)=='cpu'
weights={name:value.detach().cpu().clone() for name,value in w.last_run.model.network.state_dict().items()}
result=w.result;assert w.set_application_device('cuda') and w.result is result
output=w.add_node('Solution Plot');output.settings['resolution']=5
w.scene.connect_nodes(next(n for n in w.scene.nodes() if n.kind=='method'),output)
assert w.evaluate_trained_outputs();finish()
assert str(w.last_run.model.device)=='cuda' and w.result['steps']==2
for name,value in weights.items():torch.testing.assert_close(value,w.last_run.model.network.state_dict()[name].cpu(),rtol=0,atol=0)
assert w.resume_training(1);finish()
assert str(w.last_run.model.device)=='cuda' and w.result['steps']==3
assert w.set_application_device('cpu') and w.evaluate_trained_outputs();finish()
assert str(w.last_run.model.device)=='cpu' and w.result['steps']==3
w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()
''',timeout=60)
