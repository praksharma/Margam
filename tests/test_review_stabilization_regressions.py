"""Regression tests covering all findings from the post-PR-#53 stabilization audit."""
from copy import deepcopy
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import torch

from margam.components.paraview_export import export_vti
from margam.components.point_export import export_vtu
from margam.templates.poisson import PoissonConfig, build_pinn
from margam.visualization import SolutionDataset, from_arrays, from_file


class StabilizationRegressionTests(unittest.TestCase):
    def test_explorer_shutdown_waits_safely_during_slow_inference(self):
        code = r'''
import time
import numpy as np
from PySide6.QtWidgets import QApplication
from margam.visualization import from_arrays
from margam.ui.solution_explorer import SolutionExplorerWindow

app = QApplication([])
pts = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

def slow_evaluator(spatial_pts, *, time_value=None, params=None):
    # Exceeds the old 5-second wait(5000) timeout that triggered Qt abort.
    time.sleep(5.8)
    mu = float((params or {}).get("mu", 1.0))
    return {"u": np.full(len(spatial_pts), mu, dtype=np.float64)}

ds = from_arrays(
    pts,
    {"u": np.ones(4)},
    coordinate_names=("x", "y", "mu"),
    live_evaluator=slow_evaluator,
    param_specs=[{"name": "mu", "min": 0.5, "max": 2.0, "default": 1.0}],
)
win = SolutionExplorerWindow(dataset=ds)
win.show()
app.processEvents()

# Start slow background inference and immediately close the window.
win.evaluate_live_params({"mu": 1.75}, sync=False)
assert win.is_evaluating()
win.close()
app.processEvents()
# Window must hide promptly and defer full destruction while the worker finishes.
assert not win.isVisible()

deadline = time.monotonic() + 12.0
while win.is_evaluating() and time.monotonic() < deadline:
    app.processEvents()
    time.sleep(0.05)
assert not win.is_evaluating()
app.clipboard().clear()
app.processEvents()
'''
        result = subprocess.run(
            [sys.executable, '-c', code],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=30,
            env=dict(
                os.environ,
                QT_QPA_PLATFORM='offscreen',
                CUDA_VISIBLE_DEVICES='',
                PYTHONDONTWRITEBYTECODE='1',
            ),
        )
        self.assertEqual(result.returncode, 0, result.stdout + '\n' + result.stderr)
        self.assertNotIn('Destroyed while thread', result.stderr)
        self.assertNotIn('Traceback', result.stderr)

    def test_adaptive_sampling_never_places_points_inside_sphere_obstacle(self):
        torch.set_num_threads(1)
        for method, extra in (
            ('rar', dict(method='rar', interval=1, candidates=256, chunk=64, batch=24, maximum=128)),
            ('importance_refresh', dict(method='importance', interval=1, candidates=256, chunk=64, power=2.0, mixture=0.9, pool='refresh')),
            ('importance_fixed', dict(method='importance', interval=1, candidates=256, chunk=64, power=2.0, mixture=0.9, pool='fixed')),
        ):
            with self.subTest(adaptive=method):
                cfg = PoissonConfig(
                    shape='SphereObstacle3D',
                    sphere_x=0.5,
                    sphere_y=0.5,
                    sphere_z=0.5,
                    sphere_radius=0.35,
                    points=64,
                    boundary_points=8,
                    sampling='random',
                    adaptive_sampling=extra,
                    pde_points='interior',
                    layers=(8, 8),
                    iterations=3,
                    lr=0.01,
                    device='cpu',
                    pde_expression='laplace(u)',
                )
                model = build_pinn(cfg)
                model.train(3, display_every=0)
                pts = model.collocation_point_sample.detach().cpu().numpy()
                dist = np.sqrt((pts[:, 0] - 0.5) ** 2 + (pts[:, 1] - 0.5) ** 2 + (pts[:, 2] - 0.5) ** 2)
                inside_count = int(np.sum(dist <= 0.35))
                self.assertEqual(inside_count, 0, f'{inside_count} adaptive PDE points entered the sphere obstacle')

                # Adaptive checkpoint validation must also reject points injected inside the sphere.
                state = model.training_state()
                bad_state = deepcopy(state)
                bad_state['sampling']['points'][0] = torch.tensor([0.5, 0.5, 0.5], dtype=bad_state['sampling']['points'].dtype)
                fresh = build_pinn(cfg)
                with self.assertRaisesRegex(ValueError, 'solid obstacle'):
                    fresh.load_training_state(bad_state)

    def test_3d_paraview_velocity_export_distinguishes_3d_from_2d_vorticity(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)

            # 3D grid (2x2x2 = 8 points) with u,v,w and u,v,w,p
            pts_3d = [[float(x), float(y), float(z)] for z in (0.0, 1.0) for y in (0.0, 1.0) for x in (0.0, 1.0)]
            u = [1.0 + 0.1 * i for i in range(8)]
            v = [2.0 + 0.1 * i for i in range(8)]
            w = [3.0 + 0.1 * i for i in range(8)]
            p = [4.0 + 0.1 * i for i in range(8)]

            for fields, vals in (
                (['u', 'v', 'w'], [[a, b, c] for a, b, c in zip(u, v, w)]),
                (['u', 'v', 'w', 'p'], [[a, b, c, d] for a, b, c, d in zip(u, v, w, p)]),
            ):
                payload = dict(points=pts_3d, values=vals, fields=fields)
                vti = root / f'grid3d_{"_".join(fields)}.vti'
                vtu = root / f'cloud3d_{"_".join(fields)}.vtu'
                export_vti(vti, payload)
                export_vtu(vtu, payload)
                for path in (vti, vtu):
                    pd = ET.parse(path).getroot().find('.//PointData')
                    self.assertEqual(pd.get('Vectors'), 'velocity')
                    vel_node = next(el for el in pd.findall('DataArray') if el.get('Name') == 'velocity')
                    vel = np.fromstring(vel_node.text, sep=' ').reshape(-1, 3)
                    np.testing.assert_allclose(vel[:, 0], u)
                    np.testing.assert_allclose(vel[:, 1], v)
                    np.testing.assert_allclose(vel[:, 2], w)

            # 2D grid (2x2 = 4 points) with u,v,w (torus scalar vorticity w) must keep velocity=(u,v,0)
            pts_2d = [[float(x), float(y)] for y in (0.0, 1.0) for x in (0.0, 1.0)]
            vals_2d = [[1.0, 2.0, 9.0] for _ in range(4)]
            payload_2d = dict(points=pts_2d, values=vals_2d, fields=['u', 'v', 'w'])
            vti2 = root / 'torus2d.vti'
            vtu2 = root / 'torus2d.vtu'
            export_vti(vti2, payload_2d)
            export_vtu(vtu2, payload_2d)
            for path in (vti2, vtu2):
                pd = ET.parse(path).getroot().find('.//PointData')
                self.assertEqual(pd.get('Vectors'), 'velocity')
                vel_node = next(el for el in pd.findall('DataArray') if el.get('Name') == 'velocity')
                vel = np.fromstring(vel_node.text, sep=' ').reshape(-1, 3)
                np.testing.assert_allclose(vel[:, 2], np.zeros(4))

    def test_explorer_with_difference_enforces_exact_alignment_safeguards(self):
        pts_a = np.array([[1_000_000.0, 0.0], [1_000_001.0, 1.0]])
        pts_shifted = np.array([[1_000_000.5, 0.0], [1_000_001.5, 1.0]])
        ds_a = from_arrays(pts_a, {'u': np.array([1.0, 2.0])}, coordinate_names=('x', 'y'))
        ds_shifted = from_arrays(pts_shifted, {'u': np.array([1.0, 2.0])}, coordinate_names=('x', 'y'))
        with self.assertRaisesRegex(ValueError, 'Evaluation coordinates'):
            ds_a.with_difference(ds_shifted)

        # Different coordinate meanings (x, y) vs (x, t)
        ds_xt = from_arrays(pts_a, {'u': np.array([1.0, 2.0])}, coordinate_names=('x', 't'))
        with self.assertRaisesRegex(ValueError, 'Coordinate names'):
            ds_a.with_difference(ds_xt)

        # Reversed point IDs
        ds_id1 = SolutionDataset(pts_a, ('x', 'y'), {'u': np.array([1.0, 2.0])}, point_ids=['p0', 'p1'])
        ds_id2 = SolutionDataset(pts_a, ('x', 'y'), {'u': np.array([3.0, 5.0])}, point_ids=['p1', 'p0'])
        with self.assertRaisesRegex(ValueError, 'Point IDs'):
            ds_id1.with_difference(ds_id2)

        # Identical coordinates, names, IDs, and fields succeed cleanly
        ds_id3 = SolutionDataset(pts_a.copy(), ('x', 'y'), {'u': np.array([0.5, 0.5])}, point_ids=['p0', 'p1'])
        diff = ds_id1.with_difference(ds_id3)
        np.testing.assert_allclose(diff.fields['u'], [0.5, 1.5])

    def test_physical_time_frames_remain_strictly_disjoint_at_large_t(self):
        t0, t1 = 1_000_000.0, 1_000_000.05
        pts = np.array([
            [0.0, 0.0, t0],
            [1.0, 0.0, t0],
            [0.0, 1.0, t0],
            [0.0, 0.0, t1],
            [1.0, 0.0, t1],
            [0.0, 1.0, t1],
        ])
        vals = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        ds = from_arrays(pts, {'u': vals}, coordinate_names=('x', 'y', 't'))
        self.assertEqual(ds.num_frames, 2)
        f0 = ds._frame_indices[0]
        f1 = ds._frame_indices[1]
        self.assertEqual(len(f0), 3)
        self.assertEqual(len(f1), 3)
        self.assertEqual(set(f0.tolist()) & set(f1.tolist()), set())
        self.assertAlmostEqual(float(ds.times[0]), t0, places=6)
        self.assertAlmostEqual(float(ds.times[1]), t1, places=6)
        np.testing.assert_allclose(ds.field_values('u', time_index=0), [1.0, 1.0, 1.0])
        np.testing.assert_allclose(ds.field_values('u', time_index=1), [2.0, 2.0, 2.0])

    def test_jaxpi2_checkpoint_rejects_nonfinite_state_without_corrupting_trainer(self):
        torch.set_num_threads(1)
        from margam.research.jaxpi2 import (
            ArchConfig,
            Domain,
            Jaxpi2SphereTrainer,
            Jaxpi2TrainingConfig,
            OptimConfig,
            SpatialPools,
            SphereField,
            SphereSampler,
            create_arch,
        )

        domain = Domain()
        pools = SpatialPools.from_domain(domain, volume_count=32, boundary_count=16, seed=42)
        q = torch.tensor([[0.0, 1.0, 0.7, 0.6], [0.0, 2.0, -0.7, 0.8], [0.0, 3.0, 0.9, -0.7]])
        values = torch.tensor([[1.0, 0.1, 0.2, -0.1], [0.9, -0.2, 0.1, 0.2], [0.8, 0.2, -0.1, -0.2]])
        sampler = SphereSampler(
            domain, q, values, residual_count=8, boundary_count=2, initial_count=3, spatial_pools=pools
        )
        net = SphereField(
            create_arch(ArchConfig(arch_name='Mlp', input_dim=4, out_dim=4, num_layers=2, hidden_dim=8))
        )
        cfg = Jaxpi2TrainingConfig(
            reynolds=100.0,
            causal_enabled=True,
            chunks=2,
            optim=OptimConfig(optimizer='adam', learning_rate=1e-3, warmup_steps=0, schedule_free=False),
        )
        trainer = Jaxpi2SphereTrainer(net, sampler, cfg).compile(device='cpu')
        trainer.train(2, display_every=0)
        good_state = trainer.training_state()
        probe = torch.tensor([[0.1, 0.5, 0.5, 0.5]])
        before_pred = trainer.predict(probe).detach().clone()

        # Inject NaN into a model tensor
        bad_net_state = deepcopy(good_state)
        first_key = next(iter(bad_net_state['network']))
        bad_net_state['network'][first_key] = bad_net_state['network'][first_key].clone()
        bad_net_state['network'][first_key].view(-1)[0] = float('nan')
        with self.assertRaises(ValueError):
            trainer.load_training_state(bad_net_state)

        # Trainer predictions must remain finite and untouched after failed restore
        after_pred = trainer.predict(probe).detach().clone()
        self.assertTrue(torch.isfinite(after_pred).all().item())
        torch.testing.assert_close(before_pred, after_pred, rtol=0, atol=0)

        # Inject Inf into optimizer state tensor
        bad_opt_state = deepcopy(good_state)
        first_pid = next(iter(bad_opt_state['optimizer']['state']))
        bad_opt_state['optimizer']['state'][first_pid]['exp_avg'] = bad_opt_state['optimizer']['state'][first_pid]['exp_avg'].clone()
        bad_opt_state['optimizer']['state'][first_pid]['exp_avg'].view(-1)[0] = float('inf')
        with self.assertRaises(ValueError):
            trainer.load_training_state(bad_opt_state)

        # Inject NaN into loss weights dict
        bad_w_state = deepcopy(good_state)
        first_w = next(iter(bad_w_state['weights']))
        bad_w_state['weights'][first_w] = float('nan')
        with self.assertRaises(ValueError):
            trainer.load_training_state(bad_w_state)

        # Valid checkpoint still restores cleanly
        trainer.load_training_state(good_state)
        torch.testing.assert_close(trainer.predict(probe).detach(), before_pred, rtol=0, atol=0)

    def test_sphere_integral_loss_uses_exact_fluid_volume(self):
        cfg = PoissonConfig(
            shape='SphereObstacle3D',
            x_min=0.0, x_max=1.0,
            y_min=0.0, y_max=1.0,
            z_min=0.0, z_max=1.0,
            sphere_x=0.5, sphere_y=0.5, sphere_z=0.5, sphere_radius=0.4,
            points=32,
            boundary_points=8,
            sampling='random',
            loss_formulation='integral',
            pde_points='interior',
            layers=(8,),
            iterations=1,
            device='cpu',
            pde_expression='laplace(u)',
        )
        model = build_pinn(cfg)
        expected_fluid_vol = 1.0 - (4.0 / 3.0) * math.pi * (0.4 ** 3)
        self.assertAlmostEqual(model.problem.measure, expected_fluid_vol, places=12)

    def test_explorer_interpolation_works_without_scipy_installed(self):
        code = r'''
import sys
import numpy as np

class _BlockSciPy:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'scipy' or fullname.startswith('scipy.'):
            raise ImportError('Simulated missing optional scipy dependency')
        return None

sys.meta_path.insert(0, _BlockSciPy())
for name in list(sys.modules):
    if name == 'scipy' or name.startswith('scipy.'):
        sys.modules.pop(name, None)

from margam.visualization import DERIVED_VORTICITY_2D, from_arrays

# 1) Regular 2D grid line cut (previously used RegularGridInterpolator)
xs = np.linspace(0.0, 1.0, 9)
ys = np.linspace(0.0, 1.0, 7)
gx, gy = np.meshgrid(xs, ys, indexing='ij')
pts_reg = np.column_stack([gx.ravel(), gy.ravel()])
ds_reg = from_arrays(pts_reg, {'u': pts_reg[:, 0] + 2.0 * pts_reg[:, 1]}, coordinate_names=('x', 'y'))
cut = ds_reg.sample_line('u', (0.0, 0.0), (1.0, 1.0), n_samples=11)
assert np.allclose(cut['values'], np.linspace(0.0, 3.0, 11), atol=1e-10)

# 2) Unstructured 2D cloud + obstacle masking + unstructured vorticity
rng = np.random.default_rng(7)
pts = rng.uniform(0.0, 1.0, size=(120, 2))
keep = (pts[:, 0] - 0.5) ** 2 + (pts[:, 1] - 0.5) ** 2 >= 0.2 ** 2
pts = pts[keep]
ds_un = from_arrays(
    pts,
    {'u': -pts[:, 1], 'v': pts[:, 0]},
    coordinate_names=('x', 'y'),
    obstacle={'kind': 'cylinder', 'center': (0.5, 0.5), 'radius': 0.2},
)
_, _, grid, _ = ds_un.to_2d_grid('u', resolution=21)
assert grid.shape == (21, 21)
assert np.isnan(grid[10, 10])
assert np.any(np.isfinite(grid))
vort = ds_un.field_values(DERIVED_VORTICITY_2D)
assert len(vort) == len(pts) and np.all(np.isfinite(vort))

# 3) Unstructured 3D cloud slice
pts3 = rng.uniform(0.0, 1.0, size=(150, 3))
ds_3d = from_arrays(pts3, {'u': pts3[:, 0] + pts3[:, 1] + pts3[:, 2]}, coordinate_names=('x', 'y', 'z'))
_, _, sl, _ = ds_3d.to_2d_grid('u', slice_axis='z', slice_value=0.5, resolution=17)
assert sl.shape == (17, 17) and np.any(np.isfinite(sl))
'''
        result = subprocess.run(
            [sys.executable, '-c', code],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=60,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
        )
        self.assertEqual(result.returncode, 0, result.stdout + '\n' + result.stderr)
        self.assertNotIn('Traceback', result.stderr)

    def test_npz_loader_handles_square_cubic_and_paired_1d_archives(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)

            # Square 3x3 grid
            sq = root / 'square.npz'
            x3 = np.array([0.0, 0.5, 1.0])
            y3 = np.array([0.0, 0.5, 1.0])
            u33 = np.arange(9, dtype=np.float64).reshape(3, 3)
            np.savez(sq, x=x3, y=y3, u=u33)
            ds_sq = from_file(sq)
            self.assertEqual(ds_sq.points.shape, (9, 2))
            self.assertTrue(ds_sq.is_regular_grid(0))
            np.testing.assert_allclose(ds_sq.fields['u'], u33.ravel())

            # Cubic 3x3x3 grid
            cb = root / 'cube.npz'
            z3 = np.array([0.0, 0.5, 1.0])
            u333 = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
            np.savez(cb, x=x3, y=y3, z=z3, u=u333)
            ds_cb = from_file(cb)
            self.assertEqual(ds_cb.points.shape, (27, 3))
            self.assertTrue(ds_cb.is_regular_grid(0))
            np.testing.assert_allclose(ds_cb.fields['u'], u333.ravel())

            # Paired 1D point columns (3 points, not a 3x3 grid)
            paired = root / 'paired.npz'
            np.savez(paired, x=x3, y=y3, u=np.array([10.0, 20.0, 30.0]))
            ds_p = from_file(paired)
            self.assertEqual(ds_p.points.shape, (3, 2))
            np.testing.assert_allclose(ds_p.fields['u'], [10.0, 20.0, 30.0])


if __name__ == '__main__':
    unittest.main()
