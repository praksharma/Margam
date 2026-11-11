"""Exact stored-time display selection; numerical data and output contracts stay intact."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class ResultTimesTests(unittest.TestCase):
    def run_ui(self,code):
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=120,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        self.assertNotIn('Traceback',result.stderr)

    def test_unsorted_coupled_times_ids_duplicates_and_global_metrics(self):
        self.run_ui(r"""
from copy import deepcopy
import math
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from margam.ui.output_views import VectorResultView
from margam.templates.expert_examples import case_config
from margam.templates.poisson import PoissonConfig
app=QApplication([]);view=VectorResultView();view.show()
try:
 c=case_config('expert_cylinder');close=math.nextafter(.1,1.)
 data=dict(config=c,fields=['u','v','p'],points=[[0.,0.,.5],[1.,0.,.1],[2.,0.,.5],[3.,0.,close]],
  values=[[1.,2.,3.],[4.,5.,6.],[7.,8.,9.],[10.,11.,12.]],reference=[[0.,0.,0.]]*4,
  point_ids=['node-B','node-A','node-B',2**60],steps=8,rmse_by_field={'u':9.,'v':8.,'p':7.})
 before=deepcopy(data);view.show_result(data);app.processEvents()
 assert view.time_controls.isVisible() and view.time.count()==4 and view.table.rowCount()==4
 assert [view.time.itemData(i) for i in range(4)]==[None,.1,close,.5]
 assert view.time.itemText(1)!=view.time.itemText(2)
 view.time.setCurrentIndex(view.time.findData(.5))
 assert view.table.rowCount()==2 and view.table.item(0,0).text()=='node-B'
 assert [view.table.verticalHeaderItem(i).text() for i in range(2)]==['1','3']
 assert view.table.item(1,4).text()=='7' and view.table.item(1,7).text()=='0' and view.table.item(1,10).text()=='7'
 assert 't = 0.5' in view.caption.text() and 'RMSE (all stored rows)' in view.caption.text()
 view.show_result(data);assert view.time.currentData()==.5
 view.time.setCurrentIndex(view.time.findData(close));assert view.table.rowCount()==1 and view.table.item(0,0).text()==str(2**60)
 view.time.setCurrentIndex(0);assert view.table.rowCount()==4 and data==before
 # A new result resets the ephemeral selector; no silent nearest-time match.
 other=deepcopy(data);view.show_result(other);assert view.time.currentIndex()==0
 view.clear();assert view._result_data is None and not view.time_controls.isVisible() and view.table.rowCount()==0
 steady=dict(config=PoissonConfig(),points=[[0.,0.],[1.,1.]],values=[2.,3.],steps=1)
 view.show_result(steady);assert not view.time_controls.isVisible() and view.table.rowCount()==2
finally:view.close();app.clipboard().clear();app.processEvents()
""")

    def test_large_time_list_is_lazy_and_truncated_tables_keep_source_rows(self):
        self.run_ui(r"""
from PySide6.QtWidgets import QApplication
from margam.ui.output_views import VectorResultView,StoredTimesModel
from margam.templates.burgers import burgers_config
app=QApplication([]);view=VectorResultView()
try:
 model=StoredTimesModel();model.replace(list(range(100000)))
 assert model.rowCount()==100001 and model.data(model.index(100000,0),256)==99999
 assert model.rowCount(model.index(1,0))==0
 c=burgers_config();data=dict(config=c,points=[[i/300,0.] for i in range(250)]+[[i/300,1.] for i in range(260)],values=list(range(510)),steps=3)
 view.show_result(data);assert view.table.rowCount()==200 and '200 of 510' in view.caption.text()
 view.time.setCurrentIndex(view.time.findData(1.));assert view.table.rowCount()==200
 assert '200 of 260' in view.caption.text() and view.table.verticalHeaderItem(0).text()=='251'
 assert view.table.item(0,2).text()=='250'
 # No training-update selector: one stored physical time still has one frame.
 one=dict(data,points=[[.2,.37]],values=[4.]);view.show_result(one)
 assert view.time.count()==2 and view.time.itemData(1)==.37
finally:view.close();app.clipboard().clear();app.processEvents()
""")

    def test_real_burgers_filter_archive_export_and_invalidation(self):
        self.run_ui(r"""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
from margam.ui.model_document import capture_model
app=QApplication([]);w=Window(state_directory=False);w.show()
try:
 w.set_application_device('cpu');w.load_example('burgers_1d');w.change('iterations',2);w.change('points',16);w.change('layers',(4,))
 for node in w.scene.nodes():
  if node.kind in ('solution','export','paraview'):w.change_output(node,'resolution',9)
 assert w.start_training()
 while w.worker is not None:app.processEvents()
 assert w.result is not None
 solution=next(n for n in w.scene.nodes() if n.kind=='solution');export=next(n for n in w.scene.nodes() if n.kind=='export');pv=next(n for n in w.scene.nodes() if n.kind=='paraview')
 w.open_block(solution);view=w.output_views.entries[solution.identifier][1]
 before=capture_model(w);result=w.result;model=w.last_run.model;state=deepcopy(model.training_state());index=w.undo_stack.index()
 assert view.time.count()==10 and view.table.rowCount()==81
 with TemporaryDirectory() as directory:
  directory=Path(directory);assert w.export_block_data(export,directory/'before.csv')
  with patch.object(model,'predict',side_effect=AssertionError('Display filter must not infer')):
   view.time.setCurrentIndex(view.time.findData(.75));app.processEvents()
   assert view.table.rowCount()==9 and all(view.table.item(i,1).text()=='0.75' for i in range(9))
   assert capture_model(w)==before and w.undo_stack.index()==index and w.result is result
   assert w.last_run.model is model and model.iter==2 and model.loss_history==state['loss_history']
   assert w.export_block_data(export,directory/'after.csv')
   assert (directory/'before.csv').read_bytes()==(directory/'after.csv').read_bytes()
   assert w.export_block_data(pv,directory/'all-times.pvd')
   import xml.etree.ElementTree as ET
   assert len(ET.parse(directory/'all-times.pvd').findall('.//DataSet'))==9
   archive=directory/'run.pt';assert w.save_current_run(archive)
  w.undo_stack.setClean();assert w.open_saved_run(archive)
  solution=next(n for n in w.scene.nodes() if n.kind=='solution');w.open_block(solution)
  view=w.output_views.entries[solution.identifier][1];assert view.time.currentIndex()==0 and view.table.rowCount()==81
  view.time.setCurrentIndex(view.time.findData(.75));assert view.table.rowCount()==9
  w.change('viscosity',.02);assert view._result_data is None and view.table.rowCount()==0 and not view.time.isEnabled()
finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()
""")
