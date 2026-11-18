import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import xml.etree.ElementTree as ET
import numpy as np
import torch

from margam.components.evaluation_data import PointData, align_reference, read_point_data, validate_ids
from margam.components.paraview_export import export_pvd, export_vti
from margam.components.point_export import export_supplied, point_grid
from margam.components.prepared_data import reference_values
from margam.run_comparison import compare_predictions, read_predictions
from margam.study_document import capture as capture_study
from margam.templates.poisson import PoissonConfig
from margam.ui.model_document import decode_document
from margam.ui.project_workspace import ProjectWorkspace
from margam.ui.run_document import save_run


class AlignmentExportAndPathRegressionTests(unittest.TestCase):
    def test_2d_align_reference_and_reference_values_tolerate_sub_tolerance_grid_jitter(self):
        cfg = PoissonConfig(shape='Rectangle', x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0)
        eval_pts = [
            [0.1, 0.0],
            [0.1, 1.0],
            [0.9, 0.0],
            [0.9, 1.0],
        ]
        # Sub-tolerance jitter (1e-13 << 1e-9) reverses exact lexsort order on x=0.1 and x=0.9
        ref_pts = [
            [0.1 + 1e-13, 0.0],
            [0.1 - 1e-13, 1.0],
            [0.9 + 1e-13, 0.0],
            [0.9 - 1e-13, 1.0],
        ]
        expected_vals = [[10.0], [20.0], [30.0], [40.0]]
        ref_data = PointData(
            points=ref_pts,
            point_ids=None,
            values=expected_vals,
            provenance={'kind': 'supplied', 'source_name': 'ref.csv', 'sha256': '0' * 64},
        )
        aligned = align_reference(ref_data, eval_pts, None, cfg)
        np.testing.assert_allclose(aligned, np.array(expected_vals))

        # Also prevent silent row swap when y-spacing is tiny if points are still distinct,
        # and verify reference_values on CSV with sub-tolerance jitter + duplicate requested rejection
        with TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / 'ref.csv'
            lines = ['x,y,u']
            for (rx, ry), (val,) in zip(ref_pts, expected_vals):
                lines.append(f'{rx:.17g},{ry:.17g},{val:.17g}')
            csv_path.write_text('\n'.join(lines) + '\n')
            src = {'path': str(csv_path), 'coordinates': 'x,y', 'normals': '', 'values': 'u'}
            out = reference_values(src, eval_pts, components=1)
            self.assertEqual(out, [10.0, 20.0, 30.0, 40.0])

            # Duplicate requested points must be rejected
            dup_eval = [[0.1, 0.0], [0.1, 0.0], [0.9, 0.0], [0.9, 1.0]]
            with self.assertRaises(ValueError):
                reference_values(src, dup_eval, components=1)

            # Unambiguous point cloud where 1D axis projections have consecutive steps < 1e-9
            # but every 2D pair differs by >= 1.5e-9 > 1e-9 in at least one coordinate
            perm_pts = np.vstack((
                np.column_stack((
                    np.arange(5) * 0.75e-9,
                    np.array([0, 2, 4, 1, 3]) * 0.75e-9,
                )),
                [[1.0, 1.0]],
            ))
            perm_vals = [[float(i + 1)] for i in range(len(perm_pts))]
            perm_ref = PointData(
                points=perm_pts[::-1].tolist(),
                point_ids=None,
                values=perm_vals[::-1],
                provenance={'kind': 'supplied', 'source_name': 'perm.csv', 'sha256': '0' * 64},
            )
            aligned_perm = align_reference(perm_ref, perm_pts.tolist(), None, cfg)
            np.testing.assert_allclose(aligned_perm, np.array(perm_vals))

            perm_csv = Path(tmp) / 'perm.csv'
            perm_lines = ['x,y,u']
            for (rx, ry), (val,) in zip(perm_pts[::-1].tolist(), perm_vals[::-1]):
                perm_lines.append(f'{rx:.17g},{ry:.17g},{val:.17g}')
            perm_csv.write_text('\n'.join(perm_lines) + '\n')
            perm_src = {'path': str(perm_csv), 'coordinates': 'x,y', 'normals': '', 'values': 'u'}
            self.assertEqual(reference_values(perm_src, perm_pts.tolist(), components=1), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

            # Genuinely ambiguous points within 1e-9 in BOTH coordinates must still be rejected
            ambig_pts = np.vstack((
                [[0.0, 0.0], [0.5e-9, 0.5e-9]],
                [[1.0, 1.0]],
            ))
            ambig_ref = PointData(
                points=ambig_pts.tolist(),
                point_ids=None,
                values=[[1.0], [2.0], [3.0]],
                provenance={'kind': 'supplied', 'source_name': 'ambig.csv', 'sha256': '0' * 64},
            )
            with self.assertRaises(ValueError):
                align_reference(ambig_ref, ambig_pts.tolist(), None, cfg)

    def test_vti_grid_jitter_and_pvd_companion_cleanup(self):
        # 3x3 regular grid with 1e-14 coordinate representation noise along x=0.5
        pts = []
        vals = []
        for j, y in enumerate((0.0, 0.5, 1.0)):
            for i, x in enumerate((0.0, 0.5, 1.0)):
                jitter = (j - 1) * 1e-14 if i == 1 else 0.0
                pts.append([x + jitter, y])
                vals.append(float(x + 2.0 * y))
        with TemporaryDirectory() as tmp:
            vti_path = Path(tmp) / 'field.vti'
            export_vti(vti_path, {'points': pts, 'values': vals, 'fields': ['u']})
            self.assertTrue(vti_path.is_file())

            # Overwriting an existing PVD should clean up its superseded _frames_<uuid> folder
            pvd_path = Path(tmp) / 'series.pvd'
            frame = {'points': [[0.0], [1.0]], 'values': [1.0, 2.0], 'fields': ['u']}
            export_pvd(pvd_path, [(0.0, frame), (0.5, frame)])
            first_dirs = [p for p in Path(tmp).iterdir() if p.is_dir() and p.name.startswith('series_frames_')]
            self.assertEqual(len(first_dirs), 1)

            export_pvd(pvd_path, [(0.0, frame), (1.0, frame)])
            second_dirs = [p for p in Path(tmp).iterdir() if p.is_dir() and p.name.startswith('series_frames_')]
            self.assertEqual(len(second_dirs), 1)
            self.assertNotEqual(first_dirs[0], second_dirs[0])

    def test_point_export_and_evaluation_data_edge_cases(self):
        # Whitespace-only field name must be rejected by point_grid
        with self.assertRaises(ValueError):
            point_grid({'points': [[0.0, 0.0]], 'values': [[1.0]], 'fields': ['   ']})

        cfg = PoissonConfig(shape='SpaceTime1D', x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0)
        # 1-ULP timestamp difference must still detect duplicate IDs in the same physical frame
        with self.assertRaises(ValueError):
            validate_ids(['sensor_1', 'sensor_1'], [[0.2, 0.3], [0.4, 0.3 + 1e-16]], transient=True)

        with TemporaryDirectory() as tmp:
            # Empty transient supplied export must raise ValueError, not IndexError
            with self.assertRaises(ValueError):
                export_supplied(Path(tmp) / 'empty.pvd', {'config': cfg, 'points': [], 'values': [], 'fields': ['u']})

            # Mixed numeric and string IDs across frames must keep consistent PointData schema,
            # and 1-ULP timestamp noise must stay in one physical frame
            pvd_path = Path(tmp) / 'supplied.pvd'
            data = {
                'config': cfg,
                'points': [[0.1, 0.0], [0.9, 0.0 + 1e-16], [0.1, 0.5], [0.9, 0.5]],
                'values': [1.0, 2.0, 3.0, 4.0],
                'point_ids': ['1', '2', '1', 'sensor_B'],
                'fields': ['u'],
            }
            export_supplied(pvd_path, data)
            root = ET.parse(pvd_path).getroot()
            datasets = root.findall('.//DataSet')
            self.assertEqual(len(datasets), 2)
            array_names = []
            for ds in datasets:
                vtu = ET.parse(Path(tmp) / ds.attrib['file']).getroot()
                names = [el.attrib['Name'] for el in vtu.findall('.//PointData/DataArray')]
                array_names.append(names)
            self.assertEqual(array_names[0], array_names[1])

            # Leading/trailing whitespace in CSV ids column mapping must be stripped
            csv_path = Path(tmp) / 'pts.csv'
            csv_path.write_text('x,t,point_id\n0.2,0.1,A\n0.8,0.1,B\n')
            snap = read_point_data(
                {'path': str(csv_path), 'coordinates': ' x , t ', 'values': '', 'ids': ' point_id '},
                cfg,
            )
            self.assertEqual(snap.point_ids, ['A', 'B'])

    def test_compare_predictions_accepts_numpy_and_torch_containers(self):
        pts_np = np.array([[0.0], [1.0]])
        pts_th = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        ref_np = np.array([[0.0], [1.0]])
        a = {'coordinates': ['x'], 'config': {}, 'outputs': {'out': {'points': pts_np, 'values': np.array([0.0, 1.0]), 'reference': ref_np}}}
        b = {'coordinates': ['x'], 'config': {}, 'outputs': {'out': {'points': pts_th, 'values': torch.tensor([0.5, 1.5]), 'reference': ref_np.copy()}}}
        res = compare_predictions(a, 'out', b, 'out')
        self.assertEqual(res['rows'], 2)
        self.assertTrue(res['identical_reference_values'])

    def test_symlinked_ancestor_directories_and_dotdot_paths_work_while_leaf_symlinks_fail(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_dir = root / 'real_storage'
            real_dir.mkdir()
            sym_dir = root / 'symlink_mount'
            sym_dir.symlink_to(real_dir)

            # 1. Run archive inside symlinked ancestor directory
            cfg = PoissonConfig(shape='Interval', points=4, boundary_points=2, layers=(4,), iterations=1)
            model_doc = {
                'format': 'margam-model',
                'version': 33,
                'config': {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
                'nodes': [],
                'wires': [],
            }
            model_doc['config']['layers'] = list(cfg.layers)
            model_doc['config']['flow_residuals'] = list(cfg.flow_residuals)
            model_doc['config']['pressure_point'] = list(cfg.pressure_point)
            run_path = sym_dir / 'archive.margam-run.pt'
            save_run(
                real_dir / 'archive.margam-run.pt',
                {
                    'format': 'margam-run',
                    'version': 4,
                    'model': model_doc,
                    'result': {'outputs': {}, 'steps': 0, 'cancelled': False, 'device': 'cpu'},
                    'monitors': {},
                    'checkpoint': None,
                    'owned_files': {},
                },
            )
            loaded = read_predictions(run_path)
            self.assertEqual(loaded['steps'], 0)

            # Leaf symlink must still be rejected
            leaf_run_link = real_dir / 'leaf_link.pt'
            leaf_run_link.symlink_to(real_dir / 'archive.margam-run.pt')
            with self.assertRaisesRegex(ValueError, 'symlink'):
                read_predictions(leaf_run_link)

            # 2. Study capture(include_links=True) with linked CSV under symlinked ancestor and '..' segment
            csv_file = real_dir / 'linked.csv'
            csv_file.write_text('x\n0.25\n0.75\n')
            sub = sym_dir / 'nested'
            sub.mkdir()
            dotdot_via_symlink = str(sub / '..' / 'linked.csv')

            # Build minimal valid model document with interior_data linked path
            from margam.ui.app import QApplication, Window
            app = QApplication.instance() or QApplication([])
            w = Window(state_directory=False)
            try:
                w.change('shape', 'Interval')
                w.change('interior_data', {'path': dotdot_via_symlink, 'coordinates': 'x', 'normals': '', 'values': ''})
                from margam.ui.model_document import capture_model
                doc = capture_model(w)
                boundary_node = next(n for n in doc['nodes'] if n['kind'] == 'boundary')
                axes = [{'node_id': boundary_node['id'], 'parameter': 'boundary_points', 'mode': 'candidates', 'text': '[2, 4]'}]
                study, files = capture_study(
                    doc,
                    axes,
                    w.project_workspace,
                    execution={'device': 'cpu', 'wall_time_seconds': 0},
                    include_links=True,
                )
                self.assertEqual(len(files), 1)

                # Leaf symlink for linked CSV must still raise OSError
                csv_leaf_link = real_dir / 'alias.csv'
                csv_leaf_link.symlink_to(csv_file)
                w.change('interior_data', {'path': str(csv_leaf_link), 'coordinates': 'x', 'normals': '', 'values': ''})
                doc_bad = capture_model(w)
                with self.assertRaises(OSError):
                    capture_study(
                        doc_bad,
                        axes,
                        w.project_workspace,
                        execution={'device': 'cpu', 'wall_time_seconds': 0},
                        include_links=True,
                    )
            finally:
                app.clipboard().clear()
                w.undo_stack.setClean()
                w.close()
                app.processEvents()


if __name__ == '__main__':
    unittest.main()
