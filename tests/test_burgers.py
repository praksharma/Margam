"""Burgers space/time derivatives, initial data, sampling and physical-time export."""
from copy import deepcopy
from dataclasses import asdict
import csv,math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import xml.etree.ElementTree as ET
import torch
from margam.templates.burgers import burgers_config
from margam.templates.poisson import build_pinn,sample_problem
from margam.components.expressions import Expression
from margam.components.outputs import OutputRequest,evaluate_outputs
from margam.components.adaptive_sampling import sampling_defaults


class BurgersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def small(self):
        c=burgers_config();c.points=25;c.boundary_points=6;c.boundary_sampling={};c.layers=(8,8);c.iterations=7
        return c

    def test_original_problem_and_initial_set_without_final_condition(self):
        c=self.small();self.assertEqual(c.bounds,([-1.,0.],[1.,1.]))
        self.assertEqual(c.coordinate_names,('x','t'));self.assertAlmostEqual(c.viscosity,.01/math.pi)
        interior,boundary=sample_problem(c)
        self.assertEqual(set(boundary),{'left','right','bottom'})
        self.assertTrue(all(row[1]==0. for row in boundary['bottom']))
        self.assertTrue(all(row[0]==-1. for row in boundary['left']))
        self.assertTrue(all(row[0]==1. for row in boundary['right']))
        self.assertTrue(all(-1.<x<1. and 0.<t<1. for x,t in interior))
        m=build_pinn(c);conditions={v.name:v for v in m.problem.constraints}
        self.assertEqual(set(conditions),{'left','right','initial'})
        initial=conditions['initial'];points=initial.points
        torch.testing.assert_close(initial.target(points),-torch.sin(torch.pi*points[:,0:1]))
        self.assertEqual(conditions['left'].target,0.);self.assertEqual(conditions['right'].target,0.)

    def test_derivative_axes_nonlinearity_and_spatial_laplacian(self):
        m=build_pinn(self.small());points=torch.tensor([[.2,.3],[-.7,.8]],requires_grad=True)
        x,t=points[:,0:1],points[:,1:2];u=x*x+t*t
        expected=2*t+u*2*x-2*self.small().viscosity
        torch.testing.assert_close(m.problem.residual(points,u),expected)
        # This exact Burgers solution has nonzero u_tt, which must not enter u_xx.
        u=x/(1+t)
        torch.testing.assert_close(m.problem.residual(points,u),torch.zeros_like(x),atol=1e-7,rtol=0)
        torch.testing.assert_close(Expression('laplace(u)',2,coordinates=('x','t'))(points,u),torch.zeros_like(x))
        for text in ('D(u,t)','t'):
            with self.assertRaises(ValueError):Expression(text,2)
        with self.assertRaises(ValueError):Expression('D(u,y)',2,coordinates=('x','t'))

    def test_supported_methods_sampling_measures_and_exact_resume(self):
        for trainer in ('baseline','self_adaptive','lra'):
            for sampler in ('fixed','rar','importance'):
                if trainer=='self_adaptive' and sampler!='fixed':continue
                c=self.small();c.method=trainer
                if sampler!='fixed':
                    c.adaptive_sampling=sampling_defaults(sampler);c.adaptive_sampling.update(interval=2,candidates=32,chunk=8)
                    if sampler=='rar':c.adaptive_sampling.update(batch=3,maximum=40)
                if sampler=='importance':c.loss_formulation='integral'
                with self.subTest(trainer=trainer,sampler=sampler):
                    continuous=build_pinn(c);continuous.train(7,display_every=0)
                    partial=build_pinn(c);partial.train(3,display_every=0)
                    resumed=build_pinn(c);resumed.load_training_state(partial.training_state());resumed.train(4,display_every=0)
                    self.assertEqual(continuous.loss_history,resumed.loss_history)
                    self.assertEqual(resumed.loss_measures,dict(pde=2.,left=1.,right=1.,initial=2.))
                    self.assertTrue(all(math.isfinite(v) for row in resumed.loss_history for v in row.values()))

    def test_outputs_csv_and_playable_physical_times(self):
        from margam.components.export import export_csv
        from margam.components.paraview_export import export_vti,export_spacetime_pvd
        c=self.small();m=build_pinn(c);m.train(3,display_every=0)
        result=evaluate_outputs(m,c,[OutputRequest('csv','export',5),OutputRequest('vtk','paraview',5)])
        data=result['csv'];self.assertEqual(len(data['points']),25);self.assertIsNone(data['reference'])
        with TemporaryDirectory() as directory:
            path=Path(directory)/'burgers.csv';export_csv(path,data)
            with path.open() as stream:self.assertEqual(next(csv.reader(stream)),['x','t','u'])
            path=Path(directory)/'burgers.pvd';export_spacetime_pvd(path,result['vtk'])
            entries=ET.parse(path).findall('./Collection/DataSet')
            self.assertEqual([float(e.get('timestep')) for e in entries],[0.,.25,.5,.75,1.])
            for index,entry in enumerate(entries):
                tree=ET.parse(path.parent/entry.get('file'))
                image=tree.find('ImageData');self.assertEqual(image.get('WholeExtent'),'0 4 0 0 0 0')
                time=float(image.find('./FieldData/DataArray').text);self.assertEqual(time,index/4)
                values=[float(v) for v in tree.find('./ImageData/Piece/PointData/DataArray').text.split()]
                self.assertEqual(values,data['values'][5*index:5*(index+1)])
            with self.assertRaisesRegex(ValueError,'PVD'):export_vti(Path(directory)/'wrong.vti',data)
            previous=path.read_bytes();bad=deepcopy(data);bad['points'][1]=bad['points'][0]
            with self.assertRaises(ValueError):export_spacetime_pvd(path,bad)
            self.assertEqual(path.read_bytes(),previous)

    def test_validation_initial_compatibility_and_model_archive(self):
        from margam.ui.model_document import decode_document
        from margam.ui.run_document import save_run,load_run
        c=self.small()
        for field,value in [('viscosity',0.),('shape','Rectangle'),('method','hard_dirichlet'),('y_max',0.)]:
            bad=deepcopy(c);setattr(bad,field,value)
            with self.subTest(field=field),self.assertRaises(ValueError):build_pinn(bad)
        bad=deepcopy(c);bad.bc_targets['bottom']='1'
        with self.assertRaisesRegex(ValueError,'corner'):build_pinn(bad)
        bad=deepcopy(c);bad.bc_types['bottom']='Neumann'
        with self.assertRaisesRegex(ValueError,'initial'):build_pinn(bad)
        decoded,_,_=decode_document(dict(format='margam-model',version=15,config=asdict(c),nodes=[],wires=[]))
        self.assertEqual(decoded.coordinate_names,('x','t'))
        m=build_pinn(c);m.train(3,display_every=0)
        archive=dict(format='margam-run',version=2,model=dict(format='margam-model',version=15,config=asdict(c),nodes=[],wires=[]),
                     result=dict(outputs={},steps=3,cancelled=False,device='cpu'),monitors={},checkpoint=m.training_state())
        with TemporaryDirectory() as directory:
            p=Path(directory)/'burgers.pt';save_run(p,archive);loaded=load_run(p)
            resumed=build_pinn(c);resumed.load_training_state(loaded['checkpoint'])
            resumed.train(2,display_every=0);m.train(2,display_every=0)
            self.assertEqual(resumed.loss_history,m.loss_history)
