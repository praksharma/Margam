"""NTK Jacobians, sequential IC transfer, piecewise output and safe resume."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import torch
from margam.methods.expert import ExpertPINN
from margam.templates.examples import example_config
from margam.templates.expert_examples import CASES
from margam.templates.poisson import build_pinn


def config(key='expert_burgers',device='cpu',windows=2):
    c=example_config(key);c.device=device;c.points=8;c.boundary_points=2;c.layers=(4,4);c.network_options['features']=2
    c.expert_settings.update(chunks=2,balance_every=2,balance_mode='ntk',ntk_points=1,windows=windows,window_steps=2)
    c.iterations=4
    return c


class ExpertExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_ntk_diagonal_matches_explicit_jacobian_at_zero_loss(self):
        theta=torch.nn.Parameter(torch.tensor([2.,-1.],dtype=torch.float64));x=torch.tensor([1.,2.,3.],dtype=torch.float64)
        values=torch.stack((theta[0]*x+theta[1],theta[0]*x*x),1)
        expected=sum(a*a+1+a**4 for a in (1.,2.,3.))
        self.assertEqual(ExpertPINN.ntk_trace(values,(theta,),3),expected)
        # A zero error has zero loss gradient, but its prediction NTK is positive.
        loss=((values-values.detach())**2).mean();self.assertEqual(torch.autograd.grad(loss,theta,retain_graph=True)[0].abs().sum(),0)
        self.assertGreater(ExpertPINN.ntk_trace(values,(theta,),3),0)
        index=torch.randperm(3,generator=torch.Generator().manual_seed(0))[0].item();a=index+1
        self.assertEqual(ExpertPINN.ntk_trace(values,(theta,),1),3*(a*a+1+a**4))

    def test_ntk_ratios_and_ema_use_traces_not_loss_gradients(self):
        model=build_pinn(config(windows=1));traces=model._ntk_traces();weights,event=model._balanced(model._groups(model.compute_losses()))
        self.assertEqual(traces,event['norms'])
        for name,trace in traces.items():
            target=min(sum(traces.values())/trace,model.options['weight_cap'])
            self.assertAlmostEqual(event['targets'][name],target)
            self.assertAlmostEqual(weights[name],.9+.1*target)

    def test_all_cases_train_ntk_and_transients_march(self):
        for key in ('expert_burgers',*CASES):
            with self.subTest(key=key):
                c=config(key,windows=1 if key in ('expert_stokes','expert_cavity') else 2)
                m=build_pinn(c);m.train(4,0)
                self.assertEqual(m.iter,4);self.assertTrue(all(torch.isfinite(torch.tensor(v)) for row in m.loss_history for v in row.values()))
                if c.is_time_dependent:self.assertTrue(m.outputs_ready);self.assertEqual(len(m.completed),1)

    def test_transfer_frozen_predictions_piecewise_and_optimizer_reset(self):
        c=config();m=build_pinn(c);m.train(2,0)
        points=torch.tensor([[-.3,.5],[.7,.5]])
        expected=m.predict(points).clone();first=deepcopy(m.active.network.state_dict());m.train(1,0)
        initial=next((condition,points,target) for condition,points,target in m.active._constraints if condition.name=='initial')
        torch.testing.assert_close(initial[2],m.teachers[0](initial[1]),rtol=0,atol=0)
        self.assertFalse(initial[2].requires_grad);self.assertEqual(m.active.iter,1)
        self.assertTrue(all(not v.requires_grad for v in m.teachers[0].network.parameters()))
        torch.testing.assert_close(m.predict(points),expected,rtol=0,atol=0)
        for name,value in first.items():torch.testing.assert_close(value,m.teachers[0].network.state_dict()[name],rtol=0,atol=0)
        self.assertEqual(m.active.optimiser.param_groups[0]['lr'],c.lr)
        self.assertEqual(m.active.network.input_lower[-1].item(),.5)

    def test_resume_before_at_and_after_window_boundary(self):
        for key in ('expert_burgers','expert_torus','expert_cylinder'):
            c=config(key);full=build_pinn(c);full.train(6,0)
            for phase in (0,1,2,3,4):
                with self.subTest(key=key,phase=phase):
                    partial=build_pinn(c);partial.train(phase,0);resumed=build_pinn(c);resumed.load_training_state(partial.training_state());resumed.train(6-phase,0)
                    self.assertEqual(full.loss_history,resumed.loss_history)
                    for name,value in full.network.state_dict().items():torch.testing.assert_close(value,resumed.network.state_dict()[name],rtol=0,atol=0)
                    points=torch.tensor([c.bounds[0],c.bounds[1]])
                    torch.testing.assert_close(full.predict(points),resumed.predict(points),rtol=0,atol=0)

    def test_gradient_balancing_and_piecewise_physical_time_export(self):
        import xml.etree.ElementTree as ET
        from margam.components.outputs import evaluate_outputs,OutputRequest
        from margam.components.paraview_export import export_spacetime_pvd
        c=config('expert_torus');c.expert_settings['balance_mode']='gradient'
        full=build_pinn(c);full.train(5,0);part=build_pinn(c);part.train(2,0)
        resumed=build_pinn(c);resumed.load_training_state(part.training_state());resumed.train(3,0)
        self.assertEqual(full.loss_history,resumed.loss_history)
        data=evaluate_outputs(full,c,[OutputRequest('series','paraview',3)])['series']
        with TemporaryDirectory() as directory:
            path=Path(directory)/'series.pvd';export_spacetime_pvd(path,data)
            frames=ET.parse(path).findall('Collection/DataSet')
            self.assertEqual([float(f.get('timestep')) for f in frames],[0.,5.,10.])
            first=ET.parse(Path(directory)/frames[0].get('file'))
            values=[float(v) for v in first.find(".//PointData/DataArray[@Name='w']").text.split()]
            expected=full.teachers[0].predict_all(torch.tensor(data['points'][:9]))[:,2].tolist()
            self.assertEqual(values,[row[2] for row in data['values'][:9]])
            torch.testing.assert_close(torch.tensor(values),torch.tensor(expected),rtol=1e-6,atol=1e-7)

    def test_legacy_gradient_checkpoint_defaults(self):
        c=config(windows=1);c.expert_settings.update(balance_mode='gradient',ntk_points=16,window_steps=1000)
        model=build_pinn(c);model.train(3,0);old=model.training_state()
        options=old['expert']['settings']['configuration']['expert_settings']
        for name in ('balance_mode','ntk_points','windows','window_steps'):options.pop(name)
        for name in ('t_min','t_max','cylinder_x','cylinder_y','cylinder_radius','fourth_order','startup_time'):old['expert']['settings']['configuration'].pop(name)
        restored=build_pinn(c);restored.load_training_state(old);restored.train(1,0);model.train(1,0)
        self.assertEqual(restored.loss_history,model.loss_history)

    def test_cylinder_transfers_velocity_and_preserves_global_ramp_time(self):
        c=config('expert_cylinder');m=build_pinn(c);m.train(3,0)
        initial=next(v for v in m.active._constraints if v[0].name=='initial');self.assertEqual(initial[2].shape[1],2)
        inlet=next(v for v in m.active._constraints if v[0].name=='inlet')
        x=torch.tensor([[0.,c.y_max/2,.5]])
        expected=c.bc_values['left']*(1-torch.exp(torch.tensor(-.5/c.startup_time)))
        self.assertAlmostEqual(inlet[0].target(x)[0,0].item(),expected.item(),places=6)

    def test_partial_windows_reject_future_and_corruption_is_atomic(self):
        c=config();m=build_pinn(c);m.train(1,0);self.assertFalse(m.outputs_ready)
        with self.assertRaises(ValueError):m.predict(torch.tensor([[0.,.9]]))
        state=m.training_state()
        for change in (lambda s:s['marching']['configuration']['expert_settings'].update(window_steps=7),lambda s:s.update(iter=5),lambda s:s['marching']['completed'].append({})):
            bad=deepcopy(state);change(bad)
            with self.assertRaises(ValueError):m.load_training_state(bad)
            self.assertEqual(m.iter,1)
        m.train(2,0);bad=m.training_state()
        parameter=next(iter(dict(m.teachers[0].network.named_parameters())))
        bad['marching']['completed'][0]['network'][parameter]+=0.01
        with self.assertRaises(ValueError):m.load_training_state(bad)
        self.assertEqual(m.iter,3)
        c=config('expert_cavity')
        with self.assertRaises(ValueError):build_pinn(c)

    def test_archive_restricted_loading_and_old_settings_migration(self):
        from margam.ui.run_document import save_run,load_run
        from margam.ui.model_document import decode_document
        c=config('expert_torus');m=build_pinn(c);m.train(3,0)
        data=dict(format='margam-run',version=2,model=dict(format='margam-model',version=18,config=asdict(c),nodes=[],wires=[]),result=dict(outputs={},steps=3,cancelled=False,device='cpu'),monitors={},checkpoint=m.training_state())
        with TemporaryDirectory() as directory:
            path=Path(directory)/'windows.pt';save_run(path,data);saved=load_run(path);n=build_pinn(c);n.load_training_state(saved['checkpoint']);n.train(1,0);m.train(1,0);self.assertEqual(n.loss_history,m.loss_history)
        old=asdict(example_config('expert_burgers'))
        for name in ('balance_mode','ntk_points','windows','window_steps'):old['expert_settings'].pop(name)
        loaded,_,_=decode_document(dict(format='margam-model',version=17,config=old,nodes=[],wires=[]))
        self.assertEqual(loaded.expert_settings['balance_mode'],'gradient');self.assertEqual(loaded.expert_settings['windows'],1)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_ntk_time_marching_archive(self):
        c=config('expert_cylinder',device='cuda');m=build_pinn(c);m.train(3,0);n=build_pinn(c);n.load_training_state(m.training_state());m.train(1,0);n.train(1,0)
        self.assertEqual(m.loss_history,n.loss_history);self.assertTrue(m.predict(torch.tensor([c.bounds[1]])).is_cuda)
        from margam.ui.run_document import save_run,load_run
        data=dict(format='margam-run',version=2,model=dict(format='margam-model',version=18,config=asdict(c),nodes=[],wires=[]),result=dict(outputs={},steps=4,cancelled=False,device='cuda'),monitors={},checkpoint=m.training_state())
        with TemporaryDirectory() as directory:
            path=Path(directory)/'cuda.pt';save_run(path,data);saved=load_run(path);reopened=build_pinn(c);reopened.load_training_state(saved['checkpoint'])
            self.assertEqual(reopened.loss_history,m.loss_history)


    def test_ui_controls_worker_archive_and_resume(self):
        import os,sys,subprocess
        code=r"""import time
from pathlib import Path
from tempfile import TemporaryDirectory
from PySide6.QtWidgets import QApplication,QComboBox,QSpinBox
from margam.ui.app import Window
from margam.ui.model_document import capture_model,decode_document
from margam.templates.poisson import build_pinn
app=QApplication([]);w=Window(state_directory=False);w.show();app.processEvents()
def finish():
 deadline=time.monotonic()+30
 while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
 assert w.worker is None and w.result is not None,w.log.toPlainText()
try:
 w.load_example('expert_burgers');w.change('device','cpu');w.change('points',8);w.change('boundary_points',2);w.change('layers',(4,4))
 options=dict(w.scene.compile_config().network_options);options['features']=2;w.change('network_options',options)
 trainer=next(n for n in w.scene.nodes() if n.kind=='expert_method');w.scene.clearSelection();trainer.setSelected(True);app.processEvents()
 mode=w.property_scroll.findChild(QComboBox,'expert_balance_mode');assert mode;mode.setCurrentIndex(mode.findData('ntk'));app.processEvents()
 w.property_scroll.findChild(QSpinBox,'expert_windows').setValue(2);app.processEvents()
 w.property_scroll.findChild(QSpinBox,'expert_window_steps').setValue(2);app.processEvents()
 w.change_expert_option('chunks',2);w.change_expert_option('ntk_points',1);w.change_expert_option('balance_every',2)
 assert w.scene.compile_config().iterations==4;decode_document(capture_model(w))
 w.change('iterations',1);assert w.start_training();finish();assert w.result['outputs']=={};assert w.expert_summary()['window']==1
 with TemporaryDirectory() as directory:
  path=Path(directory)/'partial.pt';assert w.save_current_run(path);w.undo_stack.setClean();assert w.open_saved_run(path)
  assert w.resume_training(3);finish()
 assert w.result['outputs'];assert w.expert_summary()['window']==2
 assert 'Time window 2/2' in w.log.toPlainText()
 with TemporaryDirectory() as directory:
  path=Path(directory)/'windows.pt';assert w.save_current_run(path)
  w.undo_stack.setClean();assert w.open_saved_run(path);assert w.expert_summary()['window']==2
  assert w.resume_training(2);finish();full=build_pinn(w.config);full.train(6,0)
  assert full.loss_history==w.last_run.model.loss_history
finally:
 app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=90)
        self.assertEqual(result.returncode,0,result.stderr+'\n'+result.stdout)
