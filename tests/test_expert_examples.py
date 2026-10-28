"""Concrete paper equations, geometry, continuation and physical-time output."""
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import xml.etree.ElementTree as ET
import torch
from margam.templates.expert_examples import CASES,case_config,residual,samples,inside_fluid,make_constraints
from margam.templates.poisson import build_pinn
from margam.components.outputs import evaluate_outputs,OutputRequest,evaluation_points
from margam.components.paraview_export import export_spacetime_pvd
from margam.components.point_export import export_vtu


def small(key,device='cpu'):
    c=case_config(key);c.device=device;c.points=16;c.boundary_points=3;c.layers=(8,8)
    c.network_options['features']=4;c.expert_settings.update(chunks=2,balance_every=2)
    return c


class ExpertExamplesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_manufactured_residuals(self):
        for key in CASES:
            c=small(key);x=torch.tensor([[.17,.23,.31],[.43,.37,.53]],dtype=torch.float64)[:,:len(c.bounds[0])].clone().requires_grad_(True)
            a=x[:,0:1];b=x[:,1:2];zero=a*0
            if key=='expert_allen_cahn':
                u=a*a+b;values=u;expected=1-2*c.viscosity+c.source*(u**3-u)
            elif key=='expert_advection':values=torch.sin(a-c.source*b);expected=torch.zeros_like(a)
            elif key=='expert_ks':
                values=torch.cos(a);expected=-c.source*torch.cos(a)*torch.sin(a)-c.viscosity*torch.cos(a)+c.fourth_order*torch.cos(a)
            elif key=='expert_stokes':
                values=torch.cat((b*(c.y_max-b),zero,-2*c.viscosity*a),1);expected=torch.zeros((len(x),3),dtype=x.dtype)
            elif key=='expert_torus':
                decay=torch.exp(-2*c.viscosity*x[:,2:3]);values=torch.cat((torch.sin(a)*torch.cos(b)*decay,-torch.cos(a)*torch.sin(b)*decay,2*torch.sin(a)*torch.sin(b)*decay),1);expected=torch.zeros((len(x),3),dtype=x.dtype)
            elif key=='expert_cylinder':
                values=torch.cat((x[:,2:3],zero,-a),1);expected=torch.zeros((len(x),3),dtype=x.dtype)
            else:values=torch.cat((b,zero,zero),1);expected=torch.zeros((len(x),3),dtype=x.dtype)
            with self.subTest(key=key):torch.testing.assert_close(residual(c,x,values),expected,atol=1e-12,rtol=1e-12)

    def test_sampling_geometry_and_temporal_strata(self):
        for key in CASES:
            c=small(key);points,sets=samples(c)
            with self.subTest(key=key):
                self.assertTrue(all(inside_fluid(c,row) for row in points));self.assertEqual(set(sets),set(c.constraint_names))
                self.assertNotEqual(points,samples(c,1)[0])
                if c.is_time_dependent:
                    lo,hi=c.bounds[0][-1],c.bounds[1][-1]
                    self.assertTrue(all(row[-1]==lo for row in sets['initial']))
                    self.assertTrue(all(lo<=row[-1]<=(lo+hi)/2 for row in points[:8]))
                    self.assertTrue(all((lo+hi)/2<=row[-1]<=hi for row in points[8:]))
                if c.has_obstacle:
                    for row in sets['cylinder']:self.assertAlmostEqual(math.hypot(row[0]-c.cylinder_x,row[1]-c.cylinder_y),c.cylinder_radius)
                if key=='expert_cavity':self.assertTrue(all(c.x_min<row[0]<c.x_max for row in sets['lid']))

    def test_periodic_operators_match_all_required_derivatives(self):
        for key in ('expert_allen_cahn','expert_advection','expert_ks','expert_torus'):
            c=small(key)
            class Analytic(torch.nn.Module):
                def forward(self,x):
                    y=torch.sin(2*math.pi*(x[:,0:1]-c.x_min)/(c.x_max-c.x_min))
                    if key=='expert_torus':y=y*torch.cos(x[:,1:2]);return torch.cat((y,2*y,3*y),1)
                    return y
            network=Analytic();_,sets=samples(c)
            for condition in make_constraints(c,network,sets):
                if condition.name.startswith('periodic'):
                    x=condition.points.double().requires_grad_(True)
                    torch.testing.assert_close(condition.operator(x,network(x)),torch.zeros_like(condition.operator(x,network(x))),atol=1e-5,rtol=0)

    def test_each_case_trains_and_resumes_exactly(self):
        for key in CASES:
            with self.subTest(key=key):
                c=small(key);full=build_pinn(c);full.train(4,display_every=0)
                partial=build_pinn(c);partial.train(2,display_every=0)
                resumed=build_pinn(c);resumed.load_training_state(partial.training_state());resumed.train(2,display_every=0)
                self.assertEqual(full.loss_history,resumed.loss_history)
                for name,value in full.network.state_dict().items():torch.testing.assert_close(value,resumed.network.state_dict()[name],rtol=0,atol=0)
                self.assertTrue(all(math.isfinite(v) for row in full.loss_history for v in row.values()))
                self.assertEqual(set(full.coefficients),{'pde','initial','boundary'} if c.is_time_dependent else {'pde','boundary'})

    def test_fluid_exports_and_actual_physical_time(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            for key in ('expert_stokes','expert_torus','expert_cylinder'):
                c=small(key);model=build_pinn(c);n=23 if c.has_obstacle else 3
                data=evaluate_outputs(model,c,[OutputRequest('out','paraview',n)])['out']
                self.assertEqual(len(data['values']),len(evaluation_points(c,n)))
                if c.has_obstacle:self.assertLess(len(data['points']),n**len(c.bounds[0]));self.assertTrue(all(inside_fluid(c,row) for row in data['points']))
                if c.is_time_dependent:
                    path=root/(key+'.pvd');export_spacetime_pvd(path,data);entries=ET.parse(path).findall('Collection/DataSet')
                    self.assertEqual(len(entries),n);self.assertEqual(float(entries[0].get('timestep')),c.bounds[0][-1]);self.assertEqual(float(entries[-1].get('timestep')),c.bounds[1][-1])
                    frame=root/entries[0].get('file');self.assertEqual(frame.suffix,'.vtu' if c.has_obstacle else '.vti')
                else:
                    frame=root/'stokes.vtu';export_vtu(frame,data)
                tree=ET.parse(frame)
                if c.has_obstacle:self.assertTrue(all(int(v)==1 for v in tree.find("UnstructuredGrid/Piece/Cells/DataArray[@Name='types']").text.split()))
                self.assertIsNotNone(tree.find(".//PointData/DataArray[@Name='velocity']"))

    def test_torus_zero_mean_removes_uniform_velocity_drift(self):
        c=small('expert_torus')
        class Drift(torch.nn.Module):
            def forward(self,x):return torch.cat((1+x[:,2:3],x[:,2:3]*0,x[:,2:3]*0),1)
        network=Drift();_,sets=samples(c)
        condition=next(v for v in make_constraints(c,network,sets) if v.name=='mean_velocity')
        x=condition.points.double().requires_grad_(True)
        value=condition.operator(x,network(x))
        torch.testing.assert_close(value[:,0:1],1+x[:,2:3]);self.assertTrue((value[:,0]>1).all())

    def test_startup_traction_and_no_assumed_reference(self):
        from margam.templates.poisson import make_poisson_reference
        c=small('expert_cylinder');_,sets=samples(c)
        conditions={v.name:v for v in make_constraints(c,lambda x:x,sets)}
        x=torch.tensor([[c.x_min,c.y_max/2,c.t_min],[c.x_min,c.y_max/2,c.t_min+c.startup_time]],dtype=torch.float64,requires_grad=True)
        target=conditions['inlet'].target(x)
        self.assertEqual(target[0,0].item(),0.)
        self.assertAlmostEqual(target[1,0].item(),c.bc_values['left']*(1-math.exp(-1)))
        # ν u_x - p, ν v_x, including the pressure sign.
        values=torch.cat((2*x[:,0:1],3*x[:,0:1],x[:,0:1]*0+5),1)
        expected=torch.tensor([[2*c.viscosity-5,3*c.viscosity]]*2,dtype=x.dtype)
        torch.testing.assert_close(conditions['outlet_traction'].operator(x,values),expected)
        self.assertNotIn('pressure_gauge',conditions)
        for key in CASES:
            with self.assertRaises(ValueError):make_poisson_reference(small(key))

    def test_reject_invalid_case_contracts(self):
        for change in (lambda c:setattr(c,'cylinder_radius',2),lambda c:c.bc_values.update(left=float('nan')),lambda c:setattr(c,'sampling','grid')):
            c=small('expert_cylinder');change(c)
            with self.assertRaises(ValueError):build_pinn(c)
        c=small('expert_torus')
        with self.assertRaises(ValueError):evaluation_points(c,47)

    def test_all_examples_in_ui_worker_archive_and_resume(self):
        import subprocess,sys,os
        code=r"""import time
from pathlib import Path
from tempfile import TemporaryDirectory
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
from margam.templates.expert_examples import CASES
from margam.ui.model_document import capture_model,decode_document
app=QApplication([]);w=Window(state_directory=False);w.show();app.processEvents()
def finish():
 deadline=time.monotonic()+30
 while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
 assert w.worker is None and w.result is not None,w.log.toPlainText()
try:
 with TemporaryDirectory() as directory:
  for key in CASES:
   assert w.example_actions[key] in w.expert_examples_menu.actions()
   w.load_example(key);assert w.validate_model(),w.validation.text()
   w.change('device','cpu');w.change('points',16);w.change('boundary_points',3);w.change('layers',(8,8));w.change('iterations',2)
   options=dict(w.scene.compile_config().network_options);options['features']=4;w.change('network_options',options)
   settings=dict(w.scene.compile_config().expert_settings);settings.update(chunks=2,balance_every=2);w.change('expert_settings',settings)
   decode_document(capture_model(w))
   trainer=next(n for n in w.scene.nodes() if n.kind=='expert_method');compact=w.collapse_recipe(trainer);assert compact
   assert w.expand_recipe(compact);decode_document(capture_model(w));assert w.start_training();finish()
   archive=Path(directory)/(key+'.pt');assert w.save_current_run(archive)
   w.undo_stack.setClean();assert w.open_saved_run(archive);assert w.resume_training(1);finish()
   assert w.last_run.model.iter==3
finally:
 app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120)
        self.assertEqual(result.returncode,0,result.stderr+'\n'+result.stdout)
