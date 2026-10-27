"""Per-side inspector edits affect only their point set and survive navigation."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class SamplingInspectorTests(unittest.TestCase):
    def test_independent_recipes_and_reset(self):
        code = r'''
from PySide6.QtWidgets import QApplication, QComboBox, QSpinBox
from margam.ui.app import Window
from margam.templates.poisson import sample_problem
app=QApplication([]);w=Window();w.show();app.processEvents()
node=next(n for n in w.scene.nodes() if n.kind=='boundary')
owner=next(n for n in w.scene.nodes() if n.kind=='domain')
w.show_block_properties(node)
original=sample_problem(w.config)
def control(cls,name):return w.property_scroll.widget().findChild(cls,name)
control(QComboBox,'boundary_method').setCurrentText('random')
control(QSpinBox,'boundary_count').setValue(31)
control(QSpinBox,'boundary_seed').setValue(123)
new=sample_problem(w.config)
assert original[0]==new[0] and len(new[1]['left'])==31
assert all(original[1][s]==new[1][s] for s in ('right','bottom','top'))
assert '79 points' in node.detail
control(QComboBox,'boundary_side').setCurrentText('right');app.processEvents()
assert control(QSpinBox,'boundary_count').value()==16
control(QSpinBox,'boundary_count').setValue(7)
control(QComboBox,'boundary_side').setCurrentText('left');app.processEvents()
assert control(QSpinBox,'boundary_count').value()==31
assert control(QSpinBox,'boundary_seed').value()==123
assert control(QComboBox,'boundary_method').currentText()=='random'
preview=w.open_rectangle_preview(owner,True);app.processEvents()
assert not preview.grab().isNull()
assert w.validate_model()
w.change('shape','Interval');w.show_block_properties(node)
assert control(QComboBox,'boundary_side') is None
assert sample_problem(w.config)[1]=={'left':[[0.]],'right':[[1.]]}
w.change('shape','Rectangle');w.show_block_properties(node)
assert control(QSpinBox,'boundary_count').value()==31
w.set_busy(True)
assert not control(QSpinBox,'boundary_count').isEnabled()
w.set_busy(False)
w.reset_template();app.processEvents()
assert not w.config.boundary_sampling
assert all(len(rows)==16 for rows in sample_problem(w.config)[1].values())
w.undo_stack.setClean();w.close();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],
            env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),
            capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
