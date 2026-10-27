"""Expression editor changes participate in validation, undo and model storage."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class ExpressionUITests(unittest.TestCase):
    def test_expression_fields(self):
        code=r'''
from tempfile import TemporaryDirectory
from pathlib import Path
from PySide6.QtWidgets import QApplication,QLineEdit
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from margam.ui.app import Window
app=QApplication([]);w=Window();w.show();w.change('shape','Interval')
pde=next(n for n in w.scene.nodes() if n.kind=='pde');w.show_block_properties(pde)
edit=w.property_scroll.widget().findChild(QLineEdit,'pde_expression')
edit.setText('D(D(u,x),x) + u - x*x - 2');QTest.keyClick(edit,Qt.Key.Key_Return)
assert w.config.pde_expression=='D(D(u,x),x) + u - x*x - 2'
w.undo_edit();assert w.config.pde_expression==''
w.redo_edit();assert w.config.pde_expression
bc=next(n for n in w.scene.nodes() if n.kind=='bc')
w.change_bc('right','type','Custom');w.show_block_properties(bc)
for name,text in [('right_operator_expression','u + dn(u)'),('right_target_expression','3*x')]:
    edit=w.property_scroll.widget().findChild(QLineEdit,name)
    edit.setText(text);QTest.keyClick(edit,Qt.Key.Key_Return)
assert w.validate_model()
with TemporaryDirectory() as directory:
    path=Path(directory)/'custom.margam.json'
    assert w.save_model(path) and w.load_model(path)
    assert w.config.bc_targets['right']=='3*x' and w.config.bc_operators['right']=='u + dn(u)'
    assert w.validate_model()
w.undo_stack.setClean();w.close();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
