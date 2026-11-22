"""Offscreen Qt and geometry tests for Viewport3D and 3D Solution Explorer modes (Phase 3)."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from margam.ui.solution_explorer import SolutionExplorerWindow
from margam.visualization import from_arrays
from margam.visualization.geometry_3d import marching_tetrahedra, normalize_bounds


def get_app():
    return QApplication.instance() or QApplication([])


class SolutionExplorer3DTests(unittest.TestCase):
    def setUp(self):
        self.app = get_app()

    def tearDown(self):
        QApplication.clipboard().clear()
        self.app.processEvents()

    def test_marching_tetrahedra_sphere_isosurface_accuracy(self):
        xs = np.linspace(-1.0, 1.0, 13)
        ys = np.linspace(-1.0, 1.0, 13)
        zs = np.linspace(-1.0, 1.0, 13)
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        vol = gx**2 + gy**2 + gz**2
        center, scale = normalize_bounds([(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)])
        # Radius^2 = 0.36 => radius = 0.6 in world == normalized coords
        mesh = marching_tetrahedra(xs, ys, zs, vol, 0.36, center, scale)
        self.assertGreater(len(mesh["faces"]), 50)
        radii = np.linalg.norm(mesh["verts"], axis=1)
        self.assertTrue(np.allclose(radii, 0.6, atol=0.06))

    def test_3d_grid_slices_isosurface_camera_and_snapshot(self):
        xs = np.linspace(0.0, 1.0, 9)
        ys = np.linspace(0.0, 1.0, 9)
        zs = np.linspace(0.0, 1.0, 9)
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        u = np.sin(np.pi * pts[:, 0]) * np.sin(np.pi * pts[:, 1]) * np.sin(np.pi * pts[:, 2])
        ds = from_arrays(pts, {"u": u}, coordinate_names=("x", "y", "z"), title="3D Poisson")

        win = SolutionExplorerWindow(dataset=ds)
        try:
            win.show()
            self.app.processEvents()
            # Auto mode for 3D dataset selects 3D viewport (stack index 2)
            self.assertEqual(win.view_stack.currentIndex(), 2)
            self.assertGreater(len(win.viewport_3d._meshes), 0)

            # Enable all 3 orthogonal slices
            for ax in ("x", "y", "z"):
                win.slice_3d_checks[ax].setChecked(True)
            win.mode_3d_combo.setCurrentIndex(win.mode_3d_combo.findData("slices_iso"))
            self.app.processEvents()
            self.assertGreaterEqual(len(win.viewport_3d._meshes), 4)

            # Camera presets & orthographic toggle
            for preset in ("xy", "xz", "yz", "iso"):
                win.viewport_3d.set_preset(preset)
            win.ortho_check.setChecked(True)
            self.assertTrue(win.viewport_3d.camera.orthographic)
            win.ortho_check.setChecked(False)

            with TemporaryDirectory() as tmp:
                png = Path(tmp) / "view3d.png"
                self.assertTrue(win.export_snapshot(png))
                img = QImage(str(png))
                self.assertFalse(img.isNull())
                self.assertGreater(img.width(), 200)
        finally:
            win.close()
            self.app.processEvents()

    def test_3d_sphere_cloud_thresholding_and_2d_heightfield_warp(self):
        # Unstructured 3D wake around a sphere obstacle
        rng = np.random.default_rng(7)
        pts = rng.uniform(-1.0, 1.0, size=(600, 3))
        r2 = np.sum(pts**2, axis=1)
        pts = pts[r2 >= 0.25**2]
        vort_mag = np.exp(-2.0 * (pts[:, 1] ** 2 + pts[:, 2] ** 2)) * (pts[:, 0] + 1.0)
        ds_cloud = from_arrays(
            pts,
            {"vorticity": vort_mag},
            coordinate_names=("x", "y", "z"),
            title="3D Sphere Wake",
            obstacle={"kind": "sphere", "center": (0.0, 0.0, 0.0), "radius": 0.25},
        )
        win = SolutionExplorerWindow(dataset=ds_cloud)
        try:
            win.show()
            self.app.processEvents()
            self.assertEqual(win.view_stack.currentIndex(), 2)
            self.assertEqual(win.viewport_3d._effective_mode(), "cloud")
            full_count = len(win.viewport_3d._cloud_pts)
            self.assertGreater(full_count, 50)

            # Threshold to top 50% values -> strictly fewer rendered cloud points
            win.thresh_min_slider.setValue(50)
            self.app.processEvents()
            self.assertLess(len(win.viewport_3d._cloud_pts), full_count)
            self.assertFalse(win.viewport_3d.render_image(scale=1).isNull())

            # Load a 2D dataset and view it as a 3D heightfield surface
            xs = np.linspace(0.0, 1.0, 15)
            ys = np.linspace(0.0, 1.0, 15)
            gx, gy = np.meshgrid(xs, ys, indexing="ij")
            p2 = np.column_stack([gx.ravel(), gy.ravel()])
            ds_2d = from_arrays(p2, {"u": np.sin(np.pi * p2[:, 0]) * np.sin(np.pi * p2[:, 1])}, coordinate_names=("x", "y"))
            win.set_dataset(ds_2d)
            win.view_mode_combo.setCurrentIndex(win.view_mode_combo.findData("3d"))
            win.wireframe_check.setChecked(True)
            self.app.processEvents()
            self.assertEqual(win.view_stack.currentIndex(), 2)
            self.assertEqual(win.viewport_3d._effective_mode(), "heightfield")
            self.assertFalse(win.viewport_3d.render_image(scale=1).isNull())
        finally:
            win.close()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
