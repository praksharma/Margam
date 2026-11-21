import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import unittest
from PySide6.QtCore import QPoint, QPointF
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window


def get_app():
    return QApplication.instance() or QApplication([])


class UIImprovementsTests(unittest.TestCase):
    def setUp(self):
        self.app = get_app()
        self.window = Window(state_directory=False)

    def tearDown(self):
        QApplication.clipboard().clear()
        self.window.undo_stack.setClean()
        self.window.close()
        self.app.processEvents()

    def test_compatible_port_highlighting_and_wire_drop_quick_insert(self):
        window = self.window
        method = next(n for n in window.scene.nodes() if n.kind == 'method')
        pde = next(n for n in window.scene.nodes() if n.kind == 'pde')
        sol = window.add_node('Solution Plot', QPointF(1000, 160))

        # Begin dragging a wire from method.output ('pinn_run')
        window.scene.begin_wire(method.output)
        self.assertIsNotNone(window.scene.pending)
        self.assertEqual(sol.inputs['input'].wire_candidate, 'compatible')
        self.assertEqual(pde.inputs['input'].wire_candidate, 'incompatible')

        # Dropping the wire near the source port (< 24 units) cancels without opening QuickInsert
        window.scene.finish_wire(method.output.scenePos() + QPointF(5, 0), QPoint(100, 100))
        self.assertIsNone(window.scene.pending)
        self.assertIsNone(sol.inputs['input'].wire_candidate)
        self.assertIsNone(window.quick_insert)

        # Dropping the wire on empty canvas space opens filtered QuickInsert
        window.scene.begin_wire(method.output)
        drop_pos = method.output.scenePos() + QPointF(150, 80)
        window.scene.finish_wire(drop_pos, QPoint(200, 200))
        self.assertIsNone(window.scene.pending)
        self.assertIsNone(sol.inputs['input'].wire_candidate)
        self.assertIsNotNone(window.quick_insert)
        self.assertIs(window.quick_insert.source_port, method.output)

        # Verify QuickInsert only shows blocks with an input port compatible with 'pinn_run'
        tree = window.quick_insert.library.tree
        visible_titles = []
        for g_idx in range(tree.topLevelItemCount()):
            group = tree.topLevelItem(g_idx)
            if group.isHidden():
                continue
            for c_idx in range(group.childCount()):
                item = group.child(c_idx)
                if not item.isHidden():
                    visible_titles.append(item.text(0))
        self.assertIn('Training Monitor', visible_titles)
        self.assertIn('Solution Plot', visible_titles)
        self.assertIn('Comparison', visible_titles)
        self.assertNotIn('Rectangle', visible_titles)
        self.assertNotIn('PDE residual', visible_titles)

        # Choosing 'Training Monitor' inserts and connects in a single Undo command
        before_nodes = len(window.scene.nodes())
        before_wires = len(window.scene.wires)
        window.quick_insert.choose('Training Monitor')
        self.assertEqual(len(window.scene.nodes()), before_nodes + 1)
        self.assertEqual(len(window.scene.wires()), before_wires + 1) if callable(getattr(window.scene, 'wires', None)) else self.assertEqual(len(window.scene.wires), before_wires + 1)
        monitor = next(n for n in window.scene.nodes() if n.kind == 'monitor')
        self.assertTrue(any(w.source is method.output and w.target is monitor.inputs['input'] for w in window.scene.wires))
        self.assertEqual(window.undo_stack.undoText(), 'Add and connect Training Monitor')

        # Single Undo removes both the wire and the newly inserted block
        window.undo_edit()
        self.assertEqual(len(window.scene.nodes()), before_nodes)
        self.assertEqual(len(window.scene.wires), before_wires)

        # Single Redo restores both the block and connection
        window.redo_edit()
        self.assertEqual(len(window.scene.nodes()), before_nodes + 1)
        self.assertEqual(len(window.scene.wires), before_wires + 1)


if __name__ == '__main__':
    unittest.main()
