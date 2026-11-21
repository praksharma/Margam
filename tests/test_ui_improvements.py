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

    def test_wire_branching_and_wire_context_menu_and_inspector_delete(self):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QGraphicsSceneMouseEvent, QPushButton
        window = self.window
        method = next(n for n in window.scene.nodes() if n.kind == 'method')
        monitor = window.add_node('Training Monitor', QPointF(1000, 0))
        sol = window.add_node('Solution Plot', QPointF(1000, 160))
        wire1 = window.scene.connect_nodes(method, monitor)

        # Simulate Ctrl+LeftButton press on existing wire1 to branch from wire1.source (method.output)
        press = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
        press.setButton(Qt.MouseButton.LeftButton)
        press.setModifiers(Qt.KeyboardModifier.ControlModifier)
        press.setScenePos((method.output.scenePos() + monitor.inputs['input'].scenePos()) / 2)
        wire1.mousePressEvent(press)
        self.assertIsNotNone(window.scene.pending)
        self.assertIs(window.scene.pending.source, method.output)

        # Release the branched wire on sol.inputs['input']
        release = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMouseRelease)
        release.setButton(Qt.MouseButton.LeftButton)
        release.setScenePos(sol.inputs['input'].scenePos())
        release.setScreenPos(QPoint(300, 300))
        wire1.mouseReleaseEvent(release)
        self.assertIsNone(window.scene.pending)
        wire2 = next((w for w in window.scene.wires if w.target is sol.inputs['input']), None)
        self.assertIsNotNone(wire2)
        self.assertIs(wire2.source, method.output)

        # Verify wire context menu actions (Select source, Select destination, Delete connection)
        menu = window.create_wire_menu(wire2)
        labels = [a.text() for a in menu.actions() if not a.isSeparator()]
        self.assertEqual(labels, ['Select source', 'Select destination', 'Delete connection'])
        next(a for a in menu.actions() if a.text() == 'Select source').trigger()
        self.assertTrue(method.isSelected())
        next(a for a in menu.actions() if a.text() == 'Select destination').trigger()
        self.assertTrue(sol.isSelected())
        menu.deleteLater()

        # Select wire2 and verify the Properties inspector has 'Delete connection' button
        window.scene.clearSelection()
        wire2.setSelected(True)
        self.app.processEvents()
        del_btn = window.property_scroll.widget().findChild(QPushButton, 'delete_connection_button')
        self.assertIsNotNone(del_btn)
        before_wires = len(window.scene.wires)
        del_btn.click()
        self.assertEqual(len(window.scene.wires), before_wires - 1)
        self.assertEqual(window.undo_stack.undoText(), 'Delete connection')
        window.undo_edit()
        self.assertEqual(len(window.scene.wires), before_wires)

    def test_training_monitor_hover_log_toggle_and_csv_export(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        window = self.window
        method = next(n for n in window.scene.nodes() if n.kind == 'method')
        monitor = window.add_node('Training Monitor', QPointF(1000, 0))
        window.scene.connect_nodes(method, monitor)
        window.monitor_records[monitor.identifier] = [
            (1, {'total': 1.0, 'pde': 0.8, 'left': 0.05, 'right': 0.05, 'bottom': 0.05, 'top': 0.05}),
            (2, {'total': 0.1, 'pde': 0.07, 'left': 0.01, 'right': 0.01, 'bottom': 0.005, 'top': 0.005}),
            (3, {'total': 0.01, 'pde': 0.006, 'left': 0.001, 'right': 0.001, 'bottom': 0.001, 'top': 0.001}),
        ]
        window.open_block(monitor)
        _, view = window.output_views.entries[monitor.identifier]
        self.assertTrue(view.log_scale.isChecked())
        self.assertTrue(view.detail.logarithmic)
        self.assertIn('log scale', view.detail.caption)

        # Toggle to linear scale
        view.log_scale.setChecked(False)
        self.assertFalse(view.detail.logarithmic)
        self.assertIn('linear scale', view.detail.caption)
        view.log_scale.setChecked(True)

        # Hover readout at step 2
        readout = view.detail.hover_at_x(2.1)
        self.assertIn('x=2', readout)
        self.assertIn('Total=0.1', readout)
        self.assertEqual(view.hover_label.text(), readout)
        view.detail.clear_hover()
        self.assertEqual(view.hover_label.text(), '')

        # Copy CSV to clipboard and Export CSV to file
        self.assertTrue(view.copy_csv())
        csv_text = QApplication.clipboard().text()
        self.assertIn('step,total,pde,left,right,bottom,top', csv_text)
        self.assertIn('2,0.1,0.07,0.01,0.01,0.005,0.005', csv_text)
        with TemporaryDirectory() as tmp:
            out_path = Path(tmp) / 'losses.csv'
            self.assertTrue(view.export_csv(out_path))
            self.assertEqual(out_path.read_text(encoding='utf-8'), csv_text)

    def test_vector_result_view_pagination_locate_max_error_and_copy_csv(self):
        from margam.ui.output_views import VectorResultView
        from margam.templates.poisson import PoissonConfig
        view = VectorResultView()
        config = PoissonConfig()
        points = [[float(i), 0.0] for i in range(450)]
        values = [float(i) for i in range(450)]
        reference = [float(i) for i in range(450)]
        # Make row index 325 (which lives on page 2, 0-based page=1, row_in_page=125) have max |error| = 99.0
        values[325] = 325.0 + 99.0
        view.show_result({
            'config': config,
            'fields': ['u'],
            'points': points,
            'values': values,
            'reference': reference,
            'steps': 50,
        })
        self.assertEqual(view.page, 0)
        self.assertEqual(view.table.rowCount(), 200)
        self.assertIn('Page 1 of 3 (rows 1–200 of 450)', view.page_label.text())
        self.assertFalse(view.prev_page_button.isEnabled())
        self.assertTrue(view.next_page_button.isEnabled())

        # Navigate pages
        view.next_page_button.click()
        self.assertEqual(view.page, 1)
        self.assertEqual(view.table.rowCount(), 200)
        self.assertIn('Page 2 of 3 (rows 201–400 of 450)', view.page_label.text())
        view.next_page_button.click()
        self.assertEqual(view.page, 2)
        self.assertEqual(view.table.rowCount(), 50)
        self.assertIn('Page 3 of 3 (rows 401–450 of 450)', view.page_label.text())

        # Locate max |error| jumps from page 2 back to page 1 (row_in_page 125, global index 325)
        located = view.locate_max_error()
        self.assertEqual(located, (325, 1, 125))
        self.assertEqual(view.page, 1)
        selected_rows = sorted({idx.row() for idx in view.table.selectedIndexes()})
        self.assertEqual(selected_rows, [125])

        # Copy selected row CSV to clipboard
        self.assertTrue(view.copy_rows_csv())
        copied = QApplication.clipboard().text().strip().splitlines()
        self.assertEqual(copied[0], 'x,y,u,ref u,error u')
        self.assertEqual(len(copied), 2)
        self.assertIn('325,0,424,325,99', copied[1])


if __name__ == '__main__':
    unittest.main()
