"""Offscreen Qt tests for time playback, frame sequence export, and parameter-study exploration (Phase 4)."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from PySide6.QtWidgets import QApplication

from margam.ui.solution_explorer import SolutionExplorerWindow
from margam.visualization import from_arrays


def get_app():
    return QApplication.instance() or QApplication([])


class _FakeStudyCollection:
    """In-memory StudyCollection stand-in matching StudyCollection's exact contract."""

    def __init__(self, datasets_by_pair):
        self.root = Path("demo-study")
        self.axes_meta = [
            {"index": 0, "block": "PDE", "parameter": "source", "label": "PDE · source", "values": [1.0, 2.0, 4.0]},
            {"index": 1, "block": "PDE", "parameter": "viscosity", "label": "PDE · viscosity", "values": [0.1, 0.5]},
        ]
        self._pairs = []
        self._datasets = {}
        self.completed_indices = []
        idx = 0
        for s in self.axes_meta[0]["values"]:
            for nu in self.axes_meta[1]["values"]:
                self._pairs.append((s, nu))
                self._datasets[idx] = datasets_by_pair[(s, nu)]
                self.completed_indices.append(idx)
                idx += 1

    def trial_for_values(self, values):
        pair = (values[0], values[1])
        return self._pairs.index(pair) if pair in self._pairs else None

    def trial_choices(self, trial_index):
        s, nu = self._pairs[trial_index]
        return [
            {"block": "PDE", "parameter": "source", "value": s},
            {"block": "PDE", "parameter": "viscosity", "value": nu},
        ]

    def trial_status(self, trial_index):
        return "completed"

    def output_ids_for_trial(self, trial_index):
        return ["out-1"]

    def load_trial_dataset(self, trial_index, output_id=None):
        return self._datasets[trial_index]

    def parametric_response(self, axis_index, fixed_values, *, field_name, metric="max", probe_coord=None, output_id=None, time_index=None):
        xs, ys = [], []
        for val in self.axes_meta[axis_index]["values"]:
            combo = list(fixed_values)
            combo[axis_index] = val
            t_idx = self.trial_for_values(combo)
            ds = self.load_trial_dataset(t_idx)
            stats = ds.statistics(field_name, time_index=time_index)
            xs.append(val)
            ys.append(float(stats[metric]))
        return xs, ys


class SolutionExplorerStudyTests(unittest.TestCase):
    def setUp(self):
        self.app = get_app()

    def tearDown(self):
        QApplication.clipboard().clear()
        self.app.processEvents()

    def test_time_playback_trace_and_frame_export(self):
        xs = np.linspace(0.0, 1.0, 9)
        ys = np.linspace(0.0, 1.0, 9)
        ts = np.array([0.0, 0.25, 0.5, 1.0])
        pts_all, u_all = [], []
        for t in ts:
            gx, gy = np.meshgrid(xs, ys, indexing="ij")
            p = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, t)])
            pts_all.append(p)
            u_all.append((1.0 + t) * np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]))
        ds = from_arrays(np.vstack(pts_all), {"u": np.concatenate(u_all)}, coordinate_names=("x", "y", "t"), title="Transient 2D")

        win = SolutionExplorerWindow(dataset=ds)
        try:
            win.show()
            self.app.processEvents()
            self.assertFalse(win.time_bar.isHidden())
            self.assertEqual(win.time_slider.maximum(), 3)

            win.fps_combo.setCurrentIndex(2)  # 10 FPS -> 100ms
            self.assertEqual(win.play_timer.interval(), 100)

            win.set_time_index(2)
            self.assertIn("Frame 3/4", win.time_label.text())
            win._advance_time_frame()
            self.assertEqual(win.time_slider.value(), 3)
            win._advance_time_frame()
            self.assertEqual(win.time_slider.value(), 0)  # looped

            trace = win.plot_time_trace((0.5, 0.5))
            self.assertIsNotNone(trace)
            self.assertEqual(len(trace[0]), 4)
            self.assertTrue(np.allclose(trace[1], [1.0, 1.25, 1.5, 2.0], atol=1e-6))

            with TemporaryDirectory() as tmp:
                frames = win.export_frame_sequence(tmp, scale=1)
                self.assertEqual(len(frames), 4)
                for f in frames:
                    self.assertTrue(f.is_file())
        finally:
            win.close()
            self.app.processEvents()

    def test_discrete_parameter_study_scrubbing_diff_and_response(self):
        xs = np.linspace(0.0, 1.0, 9)
        ys = np.linspace(0.0, 1.0, 9)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        base_shape = np.sin(np.pi * pts[:, 0]) * np.sin(np.pi * pts[:, 1])
        datasets = {}
        for s in (1.0, 2.0, 4.0):
            for nu in (0.1, 0.5):
                datasets[(s, nu)] = from_arrays(
                    pts, {"u": (s / nu) * base_shape}, coordinate_names=("x", "y"), title=f"s={s},nu={nu}"
                )
        study = _FakeStudyCollection(datasets)

        win = SolutionExplorerWindow()
        try:
            win.show()
            win.set_study(study)
            self.app.processEvents()

            self.assertEqual(len(win._study_axis_combos), 2)
            # Initial DP0001: s=1.0, nu=0.1 -> max u = 10.0
            self.assertAlmostEqual(win.viewport_2d.effective_vmax, 10.0, places=4)

            # Lock range and switch source to 4.0 -> locked limits remain preserved
            win.lock_range_check.setChecked(True)
            win._study_axis_combos[0].setCurrentIndex(2)  # s=4.0
            self.app.processEvents()
            self.assertAlmostEqual(win.viewport_2d.effective_vmax, 10.0, places=4)

            # Unlock and verify new peak (s=4.0, nu=0.1 -> 40.0)
            win.lock_range_check.setChecked(False)
            self.app.processEvents()
            self.assertAlmostEqual(win.viewport_2d.effective_vmax, 40.0, places=4)

            # Enable Compare vs baseline (DP0001: 10.0) -> difference peak = 30.0
            win._study_diff_check.setChecked(True)
            self.app.processEvents()
            self.assertAlmostEqual(win.viewport_2d.effective_vmax, 30.0, places=4)
            win._study_diff_check.setChecked(False)

            # Plot parametric response along axis 0 (source: 1, 2, 4 -> max u: 10, 20, 40)
            xs_r, ys_r = win.plot_study_response()
            self.assertEqual(xs_r, [1.0, 2.0, 4.0])
            self.assertTrue(np.allclose(ys_r, [10.0, 20.0, 40.0], atol=1e-5))

            # Export parameter sweep frames
            with TemporaryDirectory() as tmp:
                frames = win.export_frame_sequence(tmp, scale=1)
                self.assertEqual(len(frames), 3)
        finally:
            win.close()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
