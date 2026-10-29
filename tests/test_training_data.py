"""Prepared triangle geometry, outward operators, data guards and exact restart."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil,unittest
import numpy as np
import torch
from margam.templates.prepared_triangle import prepared_triangle_config
from margam.templates.poisson import build_pinn,sample_problem
from margam.components.training_data import read_training_sets,valid_boundary_name
from margam.components.outputs import evaluation_points
from margam.ui.model_document import decode_document


class TrainingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_manufactured_residual_and_mixed_imported_operators(self):
        c=prepared_triangle_config();c.validate();m=build_pinn(c)
        self.assertEqual(c.sides,('base','upright','slope'))
        x=m.collocation_point_sample.detach().clone().requires_grad_();u=(x*x).sum(1,keepdim=True)
        torch.testing.assert_close(m.problem.residual(x,u),torch.zeros_like(u))
        for condition,points,target in m._constraints:
            x=points.detach().clone().requires_grad_();u=(x*x).sum(1,keepdim=True)
            actual=u if condition.operator is None else condition.operator(x,u)
            torch.testing.assert_close(actual,target.expand_as(actual),rtol=1e-6,atol=1e-6)
        interior,boundaries=sample_problem(c)
        self.assertEqual(len(interior),55);self.assertEqual(set(boundaries),set(c.sides))
        self.assertTrue(all(x+y<1 for x,y in interior))
        with self.assertRaisesRegex(ValueError,'Evaluation points'):evaluation_points(c,5)
        for preset in ('poisson','helmholtz','reaction_diffusion'):
            other=deepcopy(c);other.pde_preset=preset;other.source=1.;other.validate();build_pinn(other).train(1,0)

    def test_restart_and_normal_identity_for_supported_methods(self):
        for method in ('baseline','lra','self_adaptive'):
            with self.subTest(method=method):
                c=prepared_triangle_config();c.layers=(4,4);c.method=method
                full=build_pinn(c);full.train(4,0);part=build_pinn(c);part.train(2,0)
                resumed=build_pinn(c);resumed.load_training_state(part.training_state());resumed.train(2,0)
                self.assertEqual(full.loss_history,resumed.loss_history)
                for name,value in full.network.state_dict().items():torch.testing.assert_close(value,resumed.network.state_dict()[name],rtol=0,atol=0)
        with TemporaryDirectory() as directory:
            shutil.copytree('assets/training/triangle',directory,dirs_exist_ok=True)
            c=prepared_triangle_config(directory);m=build_pinn(c);m.train(1,0);state=m.training_state()
            p=Path(directory)/'upright.csv';p.write_text(p.read_text().replace(',-1,0',',1,0'))
            with self.assertRaisesRegex(ValueError,'point data'):build_pinn(c).load_training_state(state)

    def test_mapping_normals_bounds_and_unsupported_combinations(self):
        c=prepared_triangle_config();bad=deepcopy(c);bad.boundary_data['slope']['normals']=''
        with self.assertRaisesRegex(ValueError,'outward unit normals'):read_training_sets(bad)
        bad=deepcopy(c);bad.bc_targets['base']='x'
        with self.assertRaisesRegex(ValueError,'imported target'):read_training_sets(bad)
        for method in ('hard_dirichlet','scale','expert'):
            bad=deepcopy(c);bad.method=method
            with self.assertRaises(ValueError):bad.validate()
        bad=deepcopy(c);bad.loss_formulation='integral'
        with self.assertRaisesRegex(ValueError,'quadrature'):bad.validate()
        from margam.components.adaptive_sampling import sampling_defaults
        bad=deepcopy(c);bad.adaptive_sampling=sampling_defaults('rar')
        with self.assertRaisesRegex(ValueError,'candidate'):bad.validate()
        for name in ('pde','total','bad name','../wall','',1):self.assertFalse(valid_boundary_name(name))
        with TemporaryDirectory() as directory:
            p=Path(directory)/'bad.npz';np.savez(p,points=[[1.2,.1]])
            bad=deepcopy(c);bad.interior_data=dict(path=str(p),coordinates='points',normals='',values='')
            with self.assertRaisesRegex(ValueError,'outside'):read_training_sets(bad)
            np.savez(p,points=[[.2,.1]]);self.assertEqual(len(read_training_sets(bad)[0][0]),1)
        document=dict(format='margam-model',version=21,config=asdict(c),nodes=[],wires=[])
        restored,_,_=decode_document(document);self.assertEqual(restored.sides,c.sides)
        incomplete=deepcopy(document);incomplete['config']['shape']='Rectangle'
        draft,_,_=decode_document(incomplete) # Save a structurally valid draft during a mode change.
        with self.assertRaises(ValueError):draft.validate()
        bad=deepcopy(document);bad['config']['boundary_data']['../wall']=bad['config']['boundary_data'].pop('base')
        with self.assertRaises(ValueError):decode_document(bad)
