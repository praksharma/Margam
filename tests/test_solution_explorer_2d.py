"""Offscreen Qt tests for SolutionExplorerWindow and Viewport2D (Phase 2)."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from margam.ui.app import Window
from margam.ui.solution_explorer import SolutionExplorerWindow
from margam.ui.theme import configure_theme
from margam.visualization import DERIVED_SPEED, DERIVED_VORTICITY_2D, from_arrays


def get_app():
    return QApplication.instance() or QApplication([])


class SolutionExplorer2DTests(unittest.TestCase):
    def setUp(self):
        self.app = get_app()
        xs = np.linspace(0.0, 1.0, 21)
        ys = np.linspace(0.0, 1.0, 17)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        u = np.sin(np.pi * pts[:, 0]) * np.cos(np.pi * pts[:, 1])
        v = -np.cos(np.pi * pts[:, 0]) * np.sin(np.pi * pts[:, 1])
        p = pts[:, 0] ** 2 - pts[:, 1] ** 2
        ref_u = u + 0.02 * pts[:, 0]
        self.dataset = from_arrays(
            pts,
            {"u": u, "v": v, "p": p},
            coordinate_names=("x", "y"),
            references={"u": ref_u, "v": v, "p": p},
            title="Vortex 2D",
        )

    def tearDown(self):
        QApplication.clipboard().clear()
        self.app.processEvents()

    def test_2d_representations_vectors_comparison_and_exports(self):
        win = SolutionExplorerWindow(dataset=self.dataset)
        try:
            win.show()
            self.app.processEvents()

            # Check derived fields populated
            fields = [win.field_combo.itemText(i) for i in range(win.field_combo.count())]
            self.assertIn("u", fields)
            self.assertIn(DERIVED_SPEED, fields)
            self.assertIn(DERIVED_VORTICITY_2D, fields)

            # Cycle all 2D representations & vector overlays
            for idx in range(win.repr_2d_combo.count()):
                win.repr_2d_combo.setCurrentIndex(idx)
                self.app.processEvents()
                img = win.viewport_2d.render_image(scale=1)
                self.assertFalse(img.isNull())

            win.quiver_check.setChecked(True)
            win.streamlines_check.setChecked(True)
            win.points_overlay_check.setChecked(True)
            self.app.processEvents()
            self.assertGreater(len(win.viewport_2d._panels[0]["streamlines"]), 0)

            # Switch to 3-panel comparison mode
            win.view_mode_combo.setCurrentIndex(win.view_mode_combo.findData("compare"))
            self.app.processEvents()
            self.assertEqual(len(win.viewport_2d._panels), 3)

            # Locate max |error|
            pt = win.locate_max_error()
            self.assertIsNotNone(pt)
            self.assertAlmostEqual(pt[0], 1.0, places=5)
            self.assertIsNotNone(win.viewport_2d.highlight_point)

            # Line cut + probe CSV export + PNG snapshot
            win.apply_preset_cut("horizontal")
            self.assertIsNotNone(win._last_line_cut)
            self.assertGreater(len(win.probe_plot.series), 0)

            with TemporaryDirectory() as tmp:
                png = Path(tmp) / "field2d.png"
                csv_out = Path(tmp) / "probe.csv"
                self.assertTrue(win.export_snapshot(png))
                self.assertFalse(QImage(str(png)).isNull())
                self.assertTrue(win.export_probe_csv(csv_out))
                text = csv_out.read_text(encoding="utf-8")
                self.assertIn("s,x,y,value,reference", text)

            # Light/Dark theme switch
            configure_theme(self.app, "dark")
            self.app.processEvents()
            configure_theme(self.app, "light")
            self.app.processEvents()
        finally:
            win.close()
            self.app.processEvents()

    def test_launch_from_main_window_and_sync_active_run(self):
        main = Window(state_directory=False)
        try:
            sol = main.add_node("Solution Plot")
            main.result = {
                "outputs": {
                    sol.identifier: {
                        "config": main.config,
                        "points": self.dataset.points.tolist(),
                        "values": self.dataset.fields["u"].tolist(),
                        "reference": self.dataset.references["u"].tolist(),
                        "steps": 40,
                    }
                }
            }
            explorer = main.show_solution_explorer(preferred_id=sol.identifier)
            self.app.processEvents()
            self.assertIsNotNone(explorer.dataset)
            self.assertEqual(explorer.dataset.steps, 40)
            self.assertIn("Solution Explorer", explorer.windowTitle())
        finally:
            main.undo_stack.setClean()
            main.close()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
