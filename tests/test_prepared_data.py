"""Explicit mappings, normal validation and reference point alignment."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from margam.components.prepared_data import read_points,reference_values
from margam.templates.poisson import PoissonConfig,build_pinn,sample_problem


class PreparedDataTests(unittest.TestCase):
    def test_csv_npz_mapping_and_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'points.csv';path.write_text('position,value\n0,1\n1,3\n')
            source={'path':str(path),'coordinates':'position','normals':'','values':'value'}
            points,_,values=read_points(source,1)
            self.assertEqual(points.tolist(),[[0],[1]])
            self.assertEqual(reference_values(source,[[1],[0]]),[3,1])
            with self.assertRaisesRegex(ValueError,'match'):reference_values(source,[[.2],[.8]])
            npz=Path(directory)/'points.npz';np.savez(npz,points=points,normals=[[-1],[1]],values=values)
            source={'path':str(npz),'coordinates':'points','normals':'normals','values':'values'}
            self.assertEqual(read_points(source,1)[1].tolist(),[[-1],[1]])
            config=PoissonConfig(shape='Interval',interior_data={**source,'normals':'','values':''})
            self.assertEqual(sample_problem(config)[0],[[0],[1]])
            config.boundary_data['left']={**source,'values':''}
            config.bc_types['left']='Neumann'
            model=build_pinn(config)
            bc=model.problem.constraints[0];x=torch.tensor([[0.],[1.]],requires_grad=True)
            torch.testing.assert_close(bc.operator(x,x*x),torch.tensor([[0.],[2.]]))
            bad=Path(directory)/'bad.npz';np.savez(bad,points=points,normals=[[0],[2]])
            with self.assertRaisesRegex(ValueError,'unit'):read_points({**source,'path':str(bad),'values':''},1)
