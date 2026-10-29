"""Library browsing must preserve insertion and never insert category headings."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class BlockLibraryTests(unittest.TestCase):
    def test_groups_search_and_insertion(self):
        code = r'''
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window, KINDS
from margam.ui.block_library import BLOCK_GROUPS
app = QApplication([])
w = Window(state_directory=False); w.show(); app.processEvents()
library = w.library; tree = library.tree
assert sorted(title for titles in BLOCK_GROUPS.values() for title in titles) == sorted(KINDS)
assert tree.topLevelItemCount() == 8
assert BLOCK_GROUPS['PINN recipes']==('Baseline PINN','Self-adaptive PINN','Learning-rate annealing PINN','Expert PINN','Scale-PINN','Gradient-enhanced PINN')
assert BLOCK_GROUPS['Training']==('PINN method','Self-adaptive trainer','LRA trainer','Expert trainer','Scale trainer','gPINN trainer')
assert 'Optimisation / PINN' not in BLOCK_GROUPS
geometry = tree.topLevelItem(0)
before = len(w.scene.nodes())
QTest.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton, pos=tree.visualItemRect(geometry).center())
QTest.mouseDClick(tree.viewport(), Qt.MouseButton.LeftButton, pos=tree.visualItemRect(geometry).center())
assert not geometry.isExpanded()
assert len(w.scene.nodes()) == before
library.search.setText('monitor')
results = next(tree.topLevelItem(i) for i in range(tree.topLevelItemCount()) if tree.topLevelItem(i).text(0)=='Results')
assert geometry.isHidden() and not results.isHidden() and results.isExpanded()
assert [results.child(i).text(0) for i in range(results.childCount()) if not results.child(i).isHidden()] == ['Training Monitor']
monitor = results.child(1)
app.processEvents()
QTest.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton, pos=tree.visualItemRect(monitor).center())
QTest.mouseDClick(tree.viewport(), Qt.MouseButton.LeftButton, pos=tree.visualItemRect(monitor).center())
assert len(w.scene.nodes()) == before + 1
assert sum(n.kind == 'monitor' for n in w.scene.nodes()) == 1
tree.setCurrentItem(monitor); tree.setFocus()
QTest.keyClick(tree, Qt.Key.Key_Return)
assert sum(n.kind == 'monitor' for n in w.scene.nodes()) == 2
library.search.setText('SAMPLING')
sampling = tree.topLevelItem(1)
assert not sampling.isHidden() and all(not sampling.child(i).isHidden() for i in range(2))
library.search.setText('no matching component')
assert all(tree.topLevelItem(i).isHidden() for i in range(tree.topLevelItemCount()))
library.search.clear()
assert not geometry.isHidden() and not geometry.isExpanded()
assert all(not tree.topLevelItem(i).isHidden() for i in range(tree.topLevelItemCount()))
w.set_busy(True)
assert not library.search.isEnabled() and not tree.isEnabled()
w.set_busy(False)
assert library.search.isEnabled() and tree.isEnabled()
w.undo_stack.setClean();w.close(); app.processEvents()
'''
        result = subprocess.run([sys.executable, '-c', code],
            cwd=Path(__file__).resolve().parents[1],
            env=dict(os.environ, QT_QPA_PLATFORM='offscreen', PYTHONDONTWRITEBYTECODE='1'),
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
