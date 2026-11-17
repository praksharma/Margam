from copy import deepcopy
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import margam
import margam.components
import margam.methods
import margam.nn
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QLabel
from margam.slurm import export_job, run_job
from margam.study_document import capture as capture_study, write as write_study
from margam.ui.app import Window
from margam.ui.model_document import capture_model, decode_document
from margam.ui.run_document import capture_run


class UIMCPAndAPIRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.w = Window(state_directory=False)
        self.w.set_application_device('cpu')

    def tearDown(self):
        self.app.clipboard().clear()
        self.w.undo_stack.setClean()
        self.w.close()
        self.app.processEvents()

    def _finish_worker(self, timeout=30.0):
        import time
        end = time.monotonic() + timeout
        while self.w.worker is not None and time.monotonic() < end:
            self.app.processEvents()
            time.sleep(0.005)
        self.assertIsNone(self.w.worker, self.w.log.toPlainText())

    def test_public_package_dir_and_lazy_exports(self):
        self.assertIn('PINN', dir(margam))
        self.assertIn('PINNProblem', dir(margam))
        self.assertIn('research', dir(margam))
        self.assertIn('PINNProblem', dir(margam.components))
        self.assertIn('weighted_loss', dir(margam.components))
        self.assertIn('DSGNARPINN', dir(margam.methods))
        self.assertIn('ExpertPINN', dir(margam.methods))
        self.assertIn('ScalePINN', dir(margam.methods))
        self.assertIn('FourierNetwork', dir(margam.nn))
        self.assertIn('build_network', dir(margam.nn))
        self.assertTrue(callable(margam.nn.build_network))
        self.assertIsNotNone(margam.methods.DSGNARPINN)

    def test_jorvik_readonly_inspection_lock_and_run_archive_cleanliness(self):
        import shutil
        with TemporaryDirectory() as tmp:
            model_path = Path(tmp) / 'geom.margam.json'
            self.assertTrue(self.w.save_model(model_path))
            companion = Path(tmp) / 'geom_files'

            # Add Jorvik geometry block, save model, remove companion folder, then reopen & inspect:
            # read-only inspection (show_block_properties) must NOT recreate geom_files on disk.
            j_node = self.w.add_node('Jorvik geometry')
            self.assertTrue(self.w.save_model(model_path))
            if companion.exists():
                shutil.rmtree(companion)
            self.assertFalse(companion.exists())

            w_inspect = Window(state_directory=False)
            try:
                self.assertTrue(w_inspect.load_model(model_path))
                loaded_j = next(n for n in w_inspect.scene.nodes() if n.kind == 'jorvik_geometry')
                w_inspect.show_block_properties(loaded_j)
                self.assertFalse(companion.exists())
            finally:
                w_inspect.undo_stack.setClean()
                w_inspect.close()

            # Remove jorvik node so standard interval model can train, then place editor.log in companion
            self.w.scene.clearSelection()
            j_node.setSelected(True)
            self.w.delete_selected()
            self.w.change('shape', 'Interval')
            self.w.change('points', 6)
            self.w.change('boundary_points', 2)
            self.w.change('layers', (4,))
            self.w.change('iterations', 2)
            sol = self.w.add_node('Solution Plot')
            sol.settings['resolution'] = 5
            method_node = next(n for n in self.w.scene.nodes() if n.kind == 'method')
            self.w.scene.connect_nodes(method_node, sol)
            self.assertTrue(self.w.save_model(model_path))
            (companion / 'geometry' / 'abc').mkdir(parents=True)
            (companion / 'geometry' / 'abc' / 'editor.log').write_text('/secret/local/path/trace.py\n')

            self.assertTrue(self.w.start_training())
            self._finish_worker()
            run_data = capture_run(self.w)
            self.assertEqual(run_data['owned_files'], {})

            # Simulate active Jorvik editor lock: context menu & evaluate_trained_outputs must be blocked
            self.w.project_workspace.busy = True
            self.w.set_busy(True)
            try:
                menu = self.w.create_block_menu(sol)
                eval_action = next(a for a in menu.actions() if a.text() == 'Evaluate requested outputs')
                self.assertFalse(eval_action.isEnabled())
                self.assertFalse(self.w.evaluate_trained_outputs())
            finally:
                self.w.project_workspace.busy = False
                self.w.set_busy(False)

    def test_load_model_resolves_owned_config_and_cavity_re_updates_on_viscosity_edit(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / 'owned_cfg.margam.json'
            self.w.change('shape', 'Interval')
            self.assertTrue(self.w.save_model(model_path))
            owned_rel = 'datasets/sample/pts.csv'
            csv_disk = Path(self.w.project_workspace.resolve('margam-owned:' + owned_rel))
            csv_disk.parent.mkdir(parents=True, exist_ok=True)
            csv_disk.write_text('x\n0.2\n0.8\n')
            self.w.change('interior_data', {'path': 'margam-owned:' + owned_rel, 'coordinates': 'x', 'normals': '', 'values': ''})
            self.assertTrue(self.w.save_model(model_path))

            w2 = Window(state_directory=False)
            try:
                self.assertTrue(w2.load_model(model_path))
                self.assertFalse(w2.config.interior_data['path'].startswith('margam-owned:'))
                self.assertEqual(Path(w2.config.interior_data['path']), csv_disk)
            finally:
                w2.undo_stack.setClean()
                w2.close()

        # Expert cavity Re label in open Properties inspector must update when viscosity changes
        self.w.undo_stack.setClean()
        self.w.load_example('expert_cavity')
        pde_node = next(n for n in self.w.scene.nodes() if n.kind == 'predefined_pde')
        self.w.show_block_properties(pde_node)
        self.w.change('viscosity', 0.01)
        labels = [lbl.text() for lbl in self.w.property_scroll.findChildren(QLabel)]
        self.assertTrue(any('100' in text and 'Re' in text for text in labels), labels)

    def test_decode_dsgnar_method_mcp_features_and_ui_lifecycle(self):
        # 1. Forged method inside dsgnar_method block must fail decode_document
        self.w.load_example('dsgnar_poisson_1d')
        for r_node in [n for n in self.w.scene.nodes() if n.kind == 'dsgnar_recipe']:
            self.w.expand_recipe(r_node)
        doc = capture_model(self.w)
        dsg_node = next(n for n in doc['nodes'] if n['kind'] == 'dsgnar_method')
        dsg_node['settings']['parameters']['method'] = 'baseline'
        with self.assertRaises(ValueError):
            decode_document(doc)

        # 2. MCP editable settings, Comparison & Monitor results, and presentation-edit stability
        self.w.undo_stack.setClean()
        self.w.load_example('expert_advection')
        self.w.expand_recipe(next(n for n in self.w.scene.nodes() if n.kind == 'expert_recipe')) if any(n.kind == 'expert_recipe' for n in self.w.scene.nodes()) else None
        exp_node = next(n for n in self.w.scene.nodes() if n.kind == 'expert_method')
        cmds = self.w.automation_server.bridge.commands
        sid = cmds.identity()['session_id']
        schema = cmds.execute('read_schema', {'session_id': sid})
        exp_schema = next(b for b in schema['blocks'] if b['id'] == exp_node.identifier)
        self.assertIn('expert_settings', exp_schema['editable'])

        # Switch to baseline interval with Comparison + Training Monitor and verify MCP read_results
        self.w.undo_stack.setClean()
        self.w.reset_template()
        self.w.change('shape', 'Interval')
        self.w.change('points', 6)
        self.w.change('boundary_points', 2)
        self.w.change('layers', (4,))
        self.w.change('iterations', 2)
        method_node = next(n for n in self.w.scene.nodes() if n.kind == 'method')
        mon = self.w.add_node('Training Monitor')
        ref = self.w.add_node('Reference')
        comp = self.w.add_node('Comparison')
        comp.settings['resolution'] = 5
        self.w.scene.connect_nodes(method_node, mon)
        self.w.scene.connect_nodes(method_node, comp, 'model')
        self.w.scene.connect_nodes(ref, comp, 'reference')

        rev = cmds.identity()['revision']
        started = cmds.execute('start_training', {'session_id': sid, 'expected_revision': rev})
        self._finish_worker()
        job_id = started['job_id']

        comp_res = cmds.execute('read_results', {'session_id': sid, 'job_id': job_id, 'node_id': comp.identifier})
        self.assertIsNotNone(comp_res.get('rmse'))
        self.assertIn('reference', comp_res['rows'][0])

        mon_res = cmds.execute('read_results', {'session_id': sid, 'job_id': job_id, 'node_id': mon.identifier})
        self.assertEqual(mon_res['total_rows'], 2)

        # Dragging a node on canvas must NOT invalidate MCP read_results
        method_node.setPos(method_node.pos().x() + 25.0, method_node.pos().y() + 10.0)
        comp_after_move = cmds.execute('read_results', {'session_id': sid, 'job_id': job_id, 'node_id': comp.identifier})
        self.assertEqual(comp_after_move['total_rows'], 5)

        # 3. DesignPoints preview must survive canvas block selection
        dp_dialog = self.w.show_design_points()
        dp_dialog.add_axis()
        self.assertTrue(dp_dialog.refresh())
        self.assertIsNotNone(dp_dialog.result)
        self.w.scene.clearSelection()
        method_node.setSelected(True)
        self.app.processEvents()
        self.assertIsNotNone(dp_dialog.result)
        dp_dialog.close()

        # 4. Cancelled Window.closeEvent must NOT wipe open RunComparison state
        rc_dialog = self.w.show_run_comparison()
        rc_dialog.runs[0] = {'dummy': True}
        self.w.change('source', 2.0)  # dirty model
        with patch.object(self.w, 'confirm_discard', return_value=False):
            ev = QCloseEvent()
            self.w.closeEvent(ev)
            self.assertFalse(ev.isAccepted())
            self.assertEqual(rc_dialog.runs[0], {'dummy': True})
        rc_dialog.close()

        # 5. Prepared2D clear_point_data prunes orphaned custom boundary keys from bc block
        self.w.undo_stack.setClean()
        self.w.load_training_data_example()
        bc_node = next(n for n in self.w.scene.nodes() if n.kind == 'bc')
        self.assertIn('slope', bc_node.settings['parameters']['bc_types'])
        self.w.clear_point_data(side='slope')
        self.assertNotIn('slope', bc_node.settings['parameters']['bc_types'])

        # 6. ExecutableGraph.validate_graph rejects invalid output wires on inverse_burgers graphs
        from margam.ui.app import Wire
        self.w.undo_stack.setClean()
        self.assertTrue(self.w.load_inverse_example())
        sol_node = next(n for n in self.w.scene.nodes() if n.kind == 'solution')
        ref_node = self.w.add_node('Reference')
        bad_wire = Wire(ref_node.output, sol_node.inputs['input'])
        self.w.scene.wires.append(bad_wire)
        try:
            with self.assertRaises(ValueError):
                self.w.scene.validate_graph()
        finally:
            self.w.scene.wires.remove(bad_wire)

    def test_slurm_resume_without_results_raises_clean_value_error(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.w.change('shape', 'Interval')
            doc = capture_model(self.w)
            b_node = next(n for n in doc['nodes'] if n['kind'] == 'boundary')
            axes = [{'node_id': b_node['id'], 'parameter': 'boundary_points', 'mode': 'candidates', 'text': '[2, 4]'}]
            study, files = capture_study(doc, axes, self.w.project_workspace, execution={'device': 'cpu', 'wall_time_seconds': 0})
            study_path = root / 's.margam-study'
            write_study(study_path, study, files)
            bundle = root / 'bundle'
            export_job(study_path, bundle, cpus=2, memory_mb=2048, minutes=10)
            with self.assertRaises(ValueError):
                run_job(bundle, resume=True)
            (bundle / 'results').mkdir(mode=0o700)
            (bundle / 'results' / 'summary.json').write_text('{}\n')
            with self.assertRaises(ValueError):
                run_job(bundle, resume=True)


if __name__ == '__main__':
    unittest.main()
