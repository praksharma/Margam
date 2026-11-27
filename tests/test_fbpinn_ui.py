"""Desktop UI, recipe round-trip, inspector, model v36 migration, and run-archive tests for FBPINN."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class FBPINNDesktopUITests(unittest.TestCase):
    def test_lazy_torch_and_complete_fbpinn_ui_workflow(self):
        code = r'''
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QSpinBox

import margam.ui.app as app_module
assert 'torch' not in sys.modules, 'Desktop import must keep torch lazy.'

from margam.ui.app import Window
from margam.ui.model_document import CURRENT_VERSION, capture_model, decode_document
from margam.ui.run_document import capture_run, save_run, load_run
from margam.ui.run_summary import execution_plan
from margam.ui.settings_search import entries
from margam.ui.validation_report import collect_issues
from margam.templates.examples import FBPINN_EXAMPLES

app = QApplication([])
w = Window(state_directory=False)

# 1. All FBPINN examples load, validate, and appear under FBPINN examples menu
assert len(w.fbpinn_examples_menu.actions()) == len(FBPINN_EXAMPLES)
for key in FBPINN_EXAMPLES:
    assert w.load_example(key), key
    cfg = w.scene.compile_config()
    cfg.validate()
    assert cfg.method == 'fbpinn', key
    issues = [i for i in collect_issues(w) if i.severity == 'Error']
    assert not issues, (key, issues)

# 2. Compact <-> expanded recipe equivalence, inspector widgets, settings search & undo/redo
assert w.load_example('fbpinn_multiscale_1d')
trainer = next(n for n in w.scene.nodes() if n.kind == 'fbpinn_method')
network = next(n for n in w.scene.nodes() if n.kind == 'network')
w.show_block_properties(trainer)
sx = w.property_scroll.findChild(QSpinBox, 'fbpinn_subdomains_x')
sy = w.property_scroll.findChild(QSpinBox, 'fbpinn_subdomains_y')
win = w.property_scroll.findChild(QComboBox, 'fbpinn_window')
sig = w.property_scroll.findChild(QDoubleSpinBox, 'fbpinn_window_sigma')
ml = w.property_scroll.findChild(QCheckBox, 'fbpinn_multilevel')
sched = w.property_scroll.findChild(QComboBox, 'fbpinn_schedule')
steps = w.property_scroll.findChild(QSpinBox, 'fbpinn_active_steps')
assert sx and sx.isEnabled() and sy and not sy.isEnabled()
assert win and sig and not sig.isEnabled()
assert ml and ml.isChecked() and sched and steps and not steps.isEnabled()

# Switching window to smooth_sigmoid enables sigma spinbox
w.change_fbpinn_setting('window', 'smooth_sigmoid', refresh=True)
sig2 = w.property_scroll.findChild(QDoubleSpinBox, 'fbpinn_window_sigma')
assert sig2 and sig2.isEnabled()
w.undo_stack.undo()
assert w.scene.compile_config().fbpinn_settings['window'] == 'cosine_sq'

# Settings search indexes FBPINN controls
keys = {item[1] for item in entries(w)}
assert 'fbpinn_subdomains_x' in keys and 'fbpinn_multilevel' in keys and 'fbpinn_schedule' in keys

# Collapse trainer + private network into compact fbpinn_recipe, then expand back
before_cfg = w.scene.compile_config()
w.collapse_recipe(trainer)
compact = next(n for n in w.scene.nodes() if n.kind == 'fbpinn_recipe')
assert w.scene.compile_config() == before_cfg
w.show_block_properties(compact)
assert w.property_scroll.findChild(QSpinBox, 'fbpinn_subdomains_x')
w.expand_recipe(compact)
assert any(n.kind == 'fbpinn_method' for n in w.scene.nodes())
assert any(n.kind == 'network' for n in w.scene.nodes())
assert w.scene.compile_config() == before_cfg

# 3. Run Summary includes FBPINN decomposition and multiplied parameter count
_, _, summary_text = execution_plan(w)
assert 'FBPINN decomposition:' in summary_text and 'fine subdomains' in summary_text

# 4. Version 35 model document migrates cleanly to v36
doc = capture_model(w)
assert doc['version'] == CURRENT_VERSION == 36
legacy = capture_model(w)
legacy['version'] = 35
legacy['config'].pop('fbpinn_settings')
# Swap method to baseline for a genuine legacy v35 document
legacy['config']['method'] = 'baseline'
legacy['nodes'] = []
legacy['wires'] = []
migrated_cfg, _, _ = decode_document(legacy)
assert migrated_cfg.fbpinn_settings['window'] == 'cosine_sq' and migrated_cfg.fbpinn_settings['subdomains_x'] == 4

# 5. Actual short isolated training, diagnostics dialog, run archive save/load, and resume
w.change('points', 32)
w.change('iterations', 6)
w.change('layers', (8,))
w.change_fbpinn_setting('subdomains_x', 4)
w.start_training()
while w.is_running():
    app.processEvents()
assert w.result is not None and w.result['steps'] == 6

with patch('margam.ui.app.QMessageBox.information') as info:
    w.show_fbpinn_summary()
    assert info.called

with TemporaryDirectory() as td:
    run_path = Path(td) / 'fbpinn.margam-run.pt'
    save_run(run_path, capture_run(w))
    loaded = load_run(run_path)
    assert loaded['checkpoint']['iter'] == 6 and 'fbpinn' in loaded['checkpoint']

QApplication.clipboard().clear()
w.undo_stack.setClean()
w.close()
app.processEvents()
'''
        result = subprocess.run(
            [sys.executable, '-c', code],
            cwd=Path(__file__).resolve().parents[1],
            env=dict(os.environ, QT_QPA_PLATFORM='offscreen', PYTHONDONTWRITEBYTECODE='1'),
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stderr + '\n' + result.stdout)


if __name__ == '__main__':
    unittest.main()
