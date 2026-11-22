"""Unit tests for margam.visualization colormaps, SolutionDataset, and file loaders."""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np

from margam.visualization import (
    COLORMAPS,
    DERIVED_SPEED,
    DERIVED_VORTICITY_2D,
    DERIVED_VORTICITY_3D,
    SolutionDataset,
    colormap_lut,
    from_arrays,
    from_file,
    from_output_result,
    map_scalars_to_rgba,
)
from margam.templates.poisson import PoissonConfig


class VisualizationDataTests(unittest.TestCase):
    def test_colormaps_and_scalar_mapping(self):
        for name in COLORMAPS:
            lut = colormap_lut(name)
            self.assertEqual(lut.shape, (256, 3))
            self.assertEqual(lut.dtype, np.uint8)

        vals = np.array([[-2.0, 0.0], [1.0, np.nan]])
        rgba, vmin, vmax = map_scalars_to_rgba(vals, cmap="coolwarm", symmetric=True)
        self.assertEqual(rgba.shape, (2, 2, 4))
        self.assertAlmostEqual(vmin, -2.0)
        self.assertAlmostEqual(vmax, 2.0)
        self.assertEqual(int(rgba[1, 1, 3]), 0)  # NaN transparent
        self.assertEqual(int(rgba[0, 0, 3]), 255)

        log_vals = np.array([1e-4, 1e-2, 1.0])
        rgba_log, lo, hi = map_scalars_to_rgba(log_vals, cmap="viridis", log_scale=True)
        self.assertEqual(rgba_log.shape, (3, 4))
        self.assertAlmostEqual(lo, 1e-4, places=6)
        self.assertAlmostEqual(hi, 1.0, places=6)

    def test_2d_regular_coupled_derived_fields_and_line_probe(self):
        xs = np.linspace(0.0, 1.0, 11)
        ys = np.linspace(0.0, 1.0, 9)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        # Solid-body rotation: u = -y, v = x => speed = sqrt(x^2+y^2), vorticity = dv/dx - du/dy = 2
        u = -pts[:, 1]
        v = pts[:, 0]
        p = pts[:, 0] + pts[:, 1]
        ref_u = u + 0.05
        ds = from_arrays(pts, {"u": u, "v": v, "p": p}, coordinate_names=("x", "y"), references={"u": ref_u, "v": v, "p": p})

        self.assertEqual(ds.spatial_dim, 2)
        self.assertTrue(ds.is_regular_grid(0))
        self.assertIn(DERIVED_SPEED, ds.available_fields())
        self.assertIn(DERIVED_VORTICITY_2D, ds.available_fields())

        vort = ds.field_values(DERIVED_VORTICITY_2D)
        self.assertTrue(np.allclose(vort, 2.0, atol=1e-6))

        speed = ds.field_values(DERIVED_SPEED)
        self.assertTrue(np.allclose(speed, np.hypot(u, v), atol=1e-10))

        err = ds.field_values("u", mode="signed_error")
        self.assertTrue(np.allclose(err, -0.05, atol=1e-10))

        gx_out, gy_out, grid, labels = ds.to_2d_grid("u")
        self.assertEqual(labels, ("x", "y"))
        self.assertEqual(grid.shape, (9, 11))  # (ny, nx)
        self.assertTrue(np.allclose(grid[:, 0], -ys, atol=1e-10))

        cut = ds.sample_line("p", (0.0, 0.0), (1.0, 1.0), n_samples=21)
        self.assertEqual(len(cut["s"]), 21)
        self.assertAlmostEqual(float(cut["s"][-1]), np.sqrt(2.0), places=6)
        self.assertTrue(np.allclose(cut["values"], np.linspace(0.0, 2.0, 21), atol=1e-6))

        stats = ds.statistics("u")
        self.assertAlmostEqual(stats["rmse"], 0.05, places=6)
        self.assertAlmostEqual(stats["max_error"], 0.05, places=6)

    def test_unstructured_2d_obstacle_masking(self):
        rng = np.random.default_rng(42)
        pts = rng.uniform(0.0, 1.0, size=(300, 2))
        keep = (pts[:, 0] - 0.5) ** 2 + (pts[:, 1] - 0.5) ** 2 >= 0.2**2
        pts = pts[keep]
        u = pts[:, 0] ** 2 + pts[:, 1] ** 2
        ds = from_arrays(
            pts,
            {"u": u},
            coordinate_names=("x", "y"),
            obstacle={"kind": "cylinder", "center": (0.5, 0.5), "radius": 0.2},
        )
        self.assertFalse(ds.is_regular_grid(0))
        xs, ys, grid, _ = ds.to_2d_grid("u", resolution=31)
        self.assertEqual(grid.shape, (31, 31))
        # Center of cylinder (0.5, 0.5) must be NaN
        self.assertTrue(np.isnan(grid[15, 15]))

    def test_3d_grid_slicing_vorticity_and_time_trace(self):
        xs = np.linspace(0.0, 1.0, 6)
        ys = np.linspace(0.0, 1.0, 5)
        zs = np.linspace(0.0, 1.0, 4)
        ts = np.array([0.0, 0.5, 1.0])
        pts_list = []
        u_list, v_list, w_list = [], [], []
        for t in ts:
            gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
            p = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel(), np.full(gx.size, t)])
            pts_list.append(p)
            # Curl of (-y, x, 0) is (0, 0, 2) -> |w| = 2
            u_list.append(-p[:, 1] + t)
            v_list.append(p[:, 0])
            w_list.append(np.zeros(len(p)))
        ds = from_arrays(
            np.vstack(pts_list),
            {"u": np.concatenate(u_list), "v": np.concatenate(v_list), "w": np.concatenate(w_list)},
            coordinate_names=("x", "y", "z", "t"),
        )
        self.assertEqual(ds.spatial_dim, 3)
        self.assertTrue(ds.is_time_dependent)
        self.assertEqual(ds.num_frames, 3)
        self.assertIn(DERIVED_VORTICITY_3D, ds.available_fields())

        vort3 = ds.field_values(DERIVED_VORTICITY_3D, time_index=1)
        self.assertTrue(np.allclose(vort3, 2.0, atol=1e-6))

        s_xs, s_ys, slice_grid, labels = ds.to_2d_grid("u", time_index=2, slice_axis="z", slice_value=0.5)
        self.assertEqual(labels, ("x", "y"))
        self.assertEqual(slice_grid.shape, (5, 6))
        self.assertAlmostEqual(float(slice_grid[0, 0]), 1.0, places=6)  # y=0, t=1.0 -> u = 1.0

        times, trace_vals, _ = ds.time_trace((0.0, 0.0, 0.0), "u")
        self.assertTrue(np.allclose(times, [0.0, 0.5, 1.0]))
        self.assertTrue(np.allclose(trace_vals, [0.0, 0.5, 1.0]))

    def test_file_loaders_npz_csv_netcdf_and_output_result(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            # NPZ
            pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
            npz_path = root / "field3d.npz"
            np.savez(npz_path, points=pts, u=np.array([1.0, 2.0, 3.0]), ref_u=np.array([1.1, 2.1, 3.1]))
            ds_npz = from_file(npz_path)
            self.assertEqual(ds_npz.spatial_dim, 3)
            self.assertTrue(ds_npz.has_reference_for("u"))

            # CSV
            csv_path = root / "field2d.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["point_id", "x", "y", "u", "ref u"])
                w.writerow(["p0", 0.0, 0.0, 1.0, 1.0])
                w.writerow(["p1", 1.0, 0.0, 2.0, 1.5])
                w.writerow(["p2", 0.0, 1.0, 3.0, 2.5])
                w.writerow(["p3", 1.0, 1.0, 4.0, 3.5])
            ds_csv = from_file(csv_path)
            self.assertEqual(ds_csv.spatial_dim, 2)
            self.assertEqual(ds_csv.point_ids, ["p0", "p1", "p2", "p3"])
            self.assertTrue(ds_csv.has_reference_for("u"))

            # Desktop output dict
            cfg = PoissonConfig()
            out_ds = from_output_result(
                {
                    "config": cfg,
                    "points": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                    "values": [0.0, 1.0, 2.0, 3.0],
                    "reference": [0.0, 1.0, 2.0, 3.0],
                    "steps": 100,
                }
            )
            self.assertEqual(out_ds.spatial_dim, 2)
            self.assertEqual(out_ds.steps, 100)

            # Difference between datasets
            shifted = from_arrays(
                out_ds.points,
                {"u": out_ds.fields["u"] + 0.5},
                coordinate_names=out_ds.coordinate_names,
            )
            diff = shifted.with_difference(out_ds)
            self.assertTrue(np.allclose(diff.fields["u"], 0.5))


if __name__ == "__main__":
    unittest.main()
