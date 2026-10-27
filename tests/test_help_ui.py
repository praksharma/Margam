"""F1 opens the local guide; absent builds/browser errors are explained safely."""
import os,subprocess,sys,unittest

class ManualHelpUITests(unittest.TestCase):
    def test_shortcut_local_url_and_missing_build(self):
        code=r'''
import sys
from unittest.mock import patch
from PySide6.QtWidgets import QApplication,QMessageBox
from margam.ui.app import Window
assert 'torch' not in sys.modules
app=QApplication([]);w=Window(state_directory=False)
assert w.manual_action.text()=='User manual' and w.manual_action.shortcut().toString()=='F1'
with patch('margam.ui.app.Path.is_file',return_value=True),patch('margam.ui.app.QDesktopServices.openUrl',return_value=True) as browser:
    assert w.open_manual();url=browser.call_args.args[0]
    assert url.isLocalFile() and url.toLocalFile().endswith('/build/manual/site/index.html')
    w.manual_action.trigger();assert browser.call_count==2
with patch('margam.ui.app.Path.is_file',return_value=False),patch.object(QMessageBox,'information') as dialog,patch('margam.ui.app.QDesktopServices.openUrl') as browser:
    assert not w.open_manual();browser.assert_not_called()
    assert 'examples.documentation build' in dialog.call_args.args[2]
with patch('margam.ui.app.Path.is_file',return_value=True),patch('margam.ui.app.QDesktopServices.openUrl',return_value=False),patch.object(QMessageBox,'warning') as warning:
    assert not w.open_manual();warning.assert_called_once()
assert 'torch' not in sys.modules
app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
