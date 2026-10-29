"""Recipe equivalence, editable components, persistence, history and training."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class RecipeTests(unittest.TestCase):
    def test_compact_expanded_graph_and_lifecycle(self):
        code=r'''
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPointF
from margam.ui.app import Window
from margam.ui.model_document import capture_model,decode_document
from margam.templates.poisson import build_pinn
from margam.recipes import baseline
import torch
app=QApplication([]);w=Window(state_directory=False);w.show();app.processEvents()
w.load_heat_template()
root=next(n for n in w.scene.nodes() if n.kind=='method')
original=asdict(w.scene.compile_config());original_doc=capture_model(w)
outputs={n.identifier for n in w.scene.nodes() if n.kind in ('monitor','solution','comparison')}
compact=w.collapse_recipe(root)
assert compact and compact.identifier==root.identifier
assert asdict(w.scene.compile_config())==original
assert len(w.scene.nodes())==10 and len(w.scene.wires)==11
assert not w.output_views.entries
collapsed_doc=capture_model(w)
w.undo_edit();assert capture_model(w)==original_doc
w.redo_edit();assert capture_model(w)==collapsed_doc
# Compact parameters belong to the compact block, not an inspector cache.
w.change('iterations',25);w.change('lr',.003)
expected=asdict(w.scene.compile_config())
assert compact.settings['parameters']['iterations']==25
with TemporaryDirectory() as directory:
    path=Path(directory)/'recipe.margam.json'
    assert w.save_model(path) and w.load_model(path)
    assert asdict(w.scene.compile_config())==expected
    compact=next(n for n in w.scene.nodes() if n.kind=='recipe')
    w.scene.clearSelection();compact.setSelected(True)
    assert w.copy_blocks() and w.paste_blocks()
    pasted=next(n for n in w.scene.nodes() if n.kind=='recipe' and n is not compact)
    assert pasted.identifier!=compact.identifier and pasted.settings==compact.settings
    w.undo_edit()
# Double-click uses the same expansion operation; all output wires survive.
w.open_block(compact)
root=next(n for n in w.scene.nodes() if n.kind=='method')
assert asdict(w.scene.compile_config())==expected
assert outputs=={n.identifier for n in w.scene.nodes() if n.kind in ('monitor','solution','comparison')}
# Replace the actual network instance through its existing typed input.
old=next(n for n in w.scene.nodes() if n.kind=='network')
w.delete_block(old)
network=w.add_node('Neural network',QPointF(420,650))
w.change('layers',(12,12));w.change('network_seed',19)
w.scene.connect_nodes(network,root,'network')
expanded=w.scene.compile_config()
assert expanded.layers==(12,12) and expanded.network_seed==19
w.scene.clearSelection();root.setSelected(True)
compact=w.collapse_recipe(root)
assert asdict(w.scene.compile_config())==asdict(expanded)
# Actual numerical execution is identical for compact and expanded graphs.
torch.set_num_threads(1)
a=build_pinn(w.scene.compile_config());a.train(25,display_every=0)
root=w.expand_recipe(compact)
b=build_pinn(w.scene.compile_config());b.train(25,display_every=0)
points=torch.tensor([[.2,.3],[.6,.7]])
torch.testing.assert_close(a.predict(points),b.predict(points),rtol=0,atol=0)
compact=w.collapse_recipe(root)
assert w.start_training()
assert not w.expand_recipe(compact) and not w.insert_expanded_recipe()
deadline=time.monotonic()+30
while w.worker is not None and time.monotonic()<deadline:
    app.processEvents();time.sleep(.005)
assert w.worker is None and w.result is not None and w.result['steps']==25
assert not w.output_views.entries
# Shared network and specialised trainer remain expanded rather than losing data.
root=w.expand_recipe(compact);network=next(n for n in w.scene.nodes() if n.kind=='network')
other=w.add_node('PINN method');w.scene.connect_nodes(network,other,'network')
before=capture_model(w);index=w.undo_stack.index()
assert not w.collapse_recipe(root)
assert capture_model(w)==before and w.undo_stack.index()==index
w.delete_block(other)
w.scene.clearSelection();root.setSelected(True);w.change('method','hard_dirichlet')
assert not w.collapse_recipe(root)
# Scalar preset and coupled flow graphs also retain their physics contracts.
w.load_flow_template()
root=next(n for n in w.scene.nodes() if n.kind=='method')
flow=asdict(w.scene.compile_config())
compact=w.collapse_recipe(root)
assert asdict(w.scene.compile_config())==flow and w.validate_model()
w.load_example('advection_diffusion')
root=next(n for n in w.scene.nodes() if n.kind=='method')
physics=asdict(w.scene.compile_config())
compact=w.collapse_recipe(root)
assert asdict(w.scene.compile_config())==physics and w.validate_model()
# The file's fixed recipe identity cannot be changed through a forged method.
w.reset_template();compact=w.add_node(baseline.NAME)
data=capture_model(w)
record=next(n for n in data['nodes'] if n['kind']=='recipe')
record['settings']['parameters']['method']='hard_dirichlet'
try:decode_document(data);raise AssertionError('unsupported recipe accepted')
except ValueError:pass
w.undo_edit()
# Inserting expanded components is one undo step and comes from recipe defaults.
count=len(w.scene.nodes());index=w.undo_stack.index()
root=w.insert_expanded_recipe()
assert len(w.scene.nodes())==count+2 and w.undo_stack.index()==index+1
assert root.settings['parameters']['iterations']==baseline.DEFAULTS['iterations']
w.undo_edit();assert len(w.scene.nodes())==count
app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=70)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)


if __name__=='__main__':unittest.main()
