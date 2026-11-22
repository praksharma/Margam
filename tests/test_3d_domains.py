"""Native 3D domains (Box3D, SpaceTime3D, SphereObstacle3D), 3D PDEs, exports, and 3D Explorer."""
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
import xml.etree.ElementTree as ET

import torch

from margam.components.expressions import Expression
from margam.components.output_export import export_data
from margam.components.outputs import OutputRequest, evaluate_outputs, evaluation_points
from margam.templates.examples import example_config
from margam.templates.poisson import PoissonConfig, build_pinn, sample_problem


class Native3DDomainTests(unittest.TestCase):
    def test_3d_expression_derivatives_laplace_and_normal(self):
        x = torch.tensor(
            [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]],
            dtype=torch.float64,
            requires_grad=True,
        )
        # u = x^2 + 2*y^2 + 3*z^2 -> D(u,z)=6z, laplace(u)=2+4+6=12
        u = (x[:, 0:1] ** 2) + 2.0 * (x[:, 1:2] ** 2) + 3.0 * (x[:, 2:3] ** 2)
        dz = Expression('D(u,z)', 3, coordinates=('x', 'y', 'z'))(x, u)
        torch.testing.assert_close(dz, 6.0 * x[:, 2:3], rtol=1e-6, atol=1e-6)

        lap = Expression('laplace(u)', 3, coordinates=('x', 'y', 'z'))(x, u)
        torch.testing.assert_close(lap, torch.full_like(u, 12.0), rtol=1e-6, atol=1e-6)

        normal = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
        dn_val = Expression('dn(u)', 3, coordinates=('x', 'y', 'z'))(x, u, normal=normal)
        expected_dn = torch.tensor([[6.0 * 0.4], [2.0 * 0.5]], dtype=torch.float64)
        torch.testing.assert_close(dn_val, expected_dn, rtol=1e-6, atol=1e-6)

        nz_expr = Expression('nz * z + nx * x', 3, target=True, coordinates=('x', 'y', 'z'))(
            x, normal=normal
        )
        expected_nz = torch.tensor([[0.4], [0.5]], dtype=torch.float64)
        torch.testing.assert_close(nz_expr, expected_nz, rtol=1e-6, atol=1e-6)

    def test_3d_domain_sampling_and_validation(self):
        box = PoissonConfig(shape='Box3D', points=27, boundary_points=9, sampling='grid')
        box.validate()
        self.assertTrue(box.is_3d)
        self.assertEqual(box.coordinate_names, ('x', 'y', 'z'))
        self.assertEqual(box.sides, ('left', 'right', 'bottom', 'top', 'front', 'back'))
        interior, sets = sample_problem(box)
        self.assertEqual(len(interior[0]), 3)
        self.assertEqual(set(sets), set(box.sides))
        self.assertTrue(all(abs(pt[2] - 0.0) < 1e-12 for pt in sets['front']))
        self.assertTrue(all(abs(pt[2] - 1.0) < 1e-12 for pt in sets['back']))

        st3 = PoissonConfig(
            shape='SpaceTime3D',
            points=16,
            boundary_points=8,
            sampling='random',
            t_min=0.0,
            t_max=0.5,
            pde_expression='D(u,t) - 0.1*laplace(u)',
        )
        st3.validate()
        self.assertTrue(st3.is_3d)
        self.assertTrue(st3.is_time_dependent)
        self.assertEqual(st3.coordinate_names, ('x', 'y', 'z', 't'))
        self.assertIn('initial', st3.sides)
        interior_st, sets_st = sample_problem(st3)
        self.assertEqual(len(interior_st[0]), 4)
        self.assertTrue(all(abs(pt[3] - 0.0) < 1e-12 for pt in sets_st['initial']))

        sphere = PoissonConfig(
            shape='SphereObstacle3D',
            points=40,
            boundary_points=12,
            sampling='random',
            sphere_x=0.5,
            sphere_y=0.5,
            sphere_z=0.5,
            sphere_radius=0.2,
        )
        sphere.validate()
        self.assertIn('sphere', sphere.sides)
        interior_sp, sets_sp = sample_problem(sphere)
        for pt in interior_sp:
            r = math.sqrt((pt[0] - 0.5) ** 2 + (pt[1] - 0.5) ** 2 + (pt[2] - 0.5) ** 2)
            self.assertGreaterEqual(r, 0.2 - 1e-9)
        for pt in sets_sp['sphere']:
            r = math.sqrt((pt[0] - 0.5) ** 2 + (pt[1] - 0.5) ** 2 + (pt[2] - 0.5) ** 2)
            self.assertAlmostEqual(r, 0.2, places=6)

        # Evaluation points exclude the solid sphere interior
        eval_pts = evaluation_points(sphere, 7)
        self.assertLess(len(eval_pts), 7 ** 3)
        for pt in eval_pts:
            r = math.sqrt((pt[0] - 0.5) ** 2 + (pt[1] - 0.5) ** 2 + (pt[2] - 0.5) ** 2)
            self.assertGreaterEqual(r, 0.2 - 1e-9)

        # Invalid sphere (extends past box) raises ValueError
        bad_sphere = PoissonConfig(
            shape='SphereObstacle3D', sphere_x=0.9, sphere_y=0.5, sphere_z=0.5, sphere_radius=0.2
        )
        with self.assertRaises(ValueError):
            bad_sphere.validate()

    def test_3d_examples_train_evaluate_and_export_paraview(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            # 1) Manufactured 3D Poisson with HardDirichletPINN
            cfg_box = example_config('poisson_3d')
            cfg_box.points = 27
            cfg_box.boundary_points = 9
            cfg_box.layers = (8, 8)
            cfg_box.iterations = 3
            cfg_box.device = 'cpu'
            model_box = build_pinn(cfg_box)
            model_box.train(cfg_box.iterations)
            outputs_box = evaluate_outputs(
                model_box,
                cfg_box,
                [OutputRequest('sol', 'solution', resolution=5)],
            )
            out_box = outputs_box['sol']
            self.assertEqual(len(out_box['points']), 125)
            self.assertEqual(len(out_box['values']), 125)

            vti_path = root / 'poisson_3d.vti'
            written = export_data(vti_path, dict(out_box, config=cfg_box), 'paraview')
            self.assertTrue(vti_path.exists())
            self.assertEqual(written, vti_path)
            tree = ET.parse(vti_path)
            image_data = tree.find('ImageData')
            self.assertIsNotNone(image_data)
            self.assertEqual(image_data.get('WholeExtent'), '0 4 0 4 0 4')

            # 2) 3D Laplace outside hot sphere
            cfg_sp = example_config('sphere_heat_3d')
            cfg_sp.points = 32
            cfg_sp.boundary_points = 8
            cfg_sp.layers = (8, 8)
            cfg_sp.iterations = 3
            cfg_sp.device = 'cpu'
            model_sp = build_pinn(cfg_sp)
            model_sp.train(cfg_sp.iterations)
            outputs_sp = evaluate_outputs(
                model_sp,
                cfg_sp,
                [OutputRequest('sol', 'solution', resolution=5)],
            )
            out_sp = outputs_sp['sol']
            self.assertLess(len(out_sp['points']), 125)
            self.assertEqual(len(out_sp['points']), len(out_sp['values']))

            vtu_path = root / 'sphere_heat_3d.vtu'
            written_sp = export_data(vtu_path, dict(out_sp, config=cfg_sp), 'paraview')
            self.assertTrue(vtu_path.exists())
            self.assertEqual(written_sp, vtu_path)
            tree_sp = ET.parse(vtu_path)
            self.assertEqual(tree_sp.getroot().get('type'), 'UnstructuredGrid')

    def test_3d_desktop_workflow_migration_and_explorer(self):
        code = r'''
import time
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
from margam.ui.model_document import CURRENT_VERSION, capture_model, decode_document

app = QApplication([])
w = Window(state_directory=False)
w.set_application_device('cpu')
w.show()
app.processEvents()

def wait_worker():
    deadline = time.monotonic() + 30
    while w.worker is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    assert w.worker is None and w.result is not None, w.log.toPlainText()

try:
    for key in ('poisson_3d', 'sphere_heat_3d'):
        assert w.load_example(key) and w.validate_model(), w.validation.text()
        assert w.example_actions[key] in w.examples_3d_menu.actions()
        assert w.config.is_3d

        doc = capture_model(w)
        assert doc['version'] == CURRENT_VERSION
        decode_document(doc)

        # Verify format-33 migration fills 3D defaults
        if key == 'poisson_3d':
            legacy = deepcopy(doc)
            legacy['version'] = 33
            legacy['config']['shape'] = 'Rectangle'
            for k in ('z_min', 'z_max', 'sphere_x', 'sphere_y', 'sphere_z', 'sphere_radius'):
                legacy['config'].pop(k, None)
            for node in legacy['nodes']:
                if node['kind'] == 'domain':
                    node['settings']['parameters']['shape'] = 'Rectangle'
                    for k in ('z_min', 'z_max', 'sphere_x', 'sphere_y', 'sphere_z', 'sphere_radius'):
                        node['settings']['parameters'].pop(k, None)
            migrated, _, _ = decode_document(legacy)
            assert migrated.z_min == 0.0 and migrated.z_max == 1.0 and migrated.sphere_radius == 0.2

        w.change('points', 27)
        w.change('boundary_points', 6)
        w.change('layers', (6, 6))
        w.change('iterations', 2)
        for node in w.scene.nodes():
            if node.kind in ('solution', 'comparison', 'export', 'paraview'):
                node.settings['resolution'] = 5

        assert w.start_training()
        wait_worker()
        assert w.result['steps'] == 2

        sol_node = next(n for n in w.scene.nodes() if n.kind == 'solution')
        explorer = w.show_solution_explorer(sol_node)
        assert explorer is not None
        assert explorer.dataset is not None
        assert explorer.dataset.spatial_dim == 3
        assert explorer.view_stack.currentIndex() == 2
        app.processEvents()
        explorer.close()

        with TemporaryDirectory() as directory:
            path = Path(directory) / f'{key}.margam.json'
            assert w.save_model(path)
            w.undo_stack.setClean()
            assert w.load_model(path)
            assert w.config.is_3d
finally:
    app.clipboard().clear()
    w.undo_stack.setClean()
    w.close()
    app.processEvents()
'''
        result = subprocess.run(
            [sys.executable, '-c', code],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=120,
            env=dict(
                os.environ,
                QT_QPA_PLATFORM='offscreen',
                CUDA_VISIBLE_DEVICES='',
                PYTHONDONTWRITEBYTECODE='1',
            ),
        )
        self.assertEqual(result.returncode, 0, result.stdout + '\n' + result.stderr)
        self.assertNotIn('Traceback', result.stderr)


if __name__ == '__main__':
    unittest.main()
