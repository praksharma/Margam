"""Exact supplied coordinates/IDs, reference alignment and lossless exports."""
from dataclasses import asdict
import csv,json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import xml.etree.ElementTree as ET
import numpy as np
import torch
from margam.components.evaluation_data import read_point_data,align_reference,validate_points
from margam.components.outputs import OutputRequest,evaluate_outputs
from margam.components.export import export_csv
from margam.components.point_export import export_supplied
from margam.templates.examples import example_config
from margam.ui.run_document import save_run,load_run


def source(path,coordinates='x',values='',ids='id'):return dict(path=str(path),coordinates=coordinates,values=values,ids=ids)


class EvaluationDataTests(unittest.TestCase):
    def test_csv_snapshot_order_text_ids_and_reference_alignment(self):
        c=example_config('interval_poisson')
        with TemporaryDirectory() as directory:
            path=Path(directory)/'points.csv';path.write_text('id,x,u\n003,0.75,0.1875\nnode-A,0.25,0.1875\n')
            points=read_point_data(source(path),c)
            path.write_text('id,x,u\nnode-A,0.25,0.1875\n003,0.75,0.1875\n');reference=read_point_data(source(path,values='u'),c,reference=True);path.unlink()
            class Exact:
                iter=7
                def predict(self,x):return x[:,0:1]*(1-x[:,0:1])
            data=evaluate_outputs(Exact(),c,[OutputRequest('out','comparison',3,None,points,reference)])['out']
            self.assertEqual(data['points'],[[.75],[.25]]);self.assertEqual(data['point_ids'],['003','node-A']);self.assertEqual(data['rmse'],0.);self.assertIsNone(data['size'])
            self.assertNotEqual(data['evaluation']['sha256'],data['reference_provenance']['sha256'])
            csv_path=Path(directory)/'out.csv';export_csv(csv_path,data)
            with csv_path.open() as stream:rows=list(csv.DictReader(stream))
            self.assertEqual([row['point_id'] for row in rows],['003','node-A']);self.assertEqual([row['source_row'] for row in rows],['0','1']);self.assertEqual([float(row['error_u']) for row in rows],[0.,0.])
            vtu=Path(directory)/'out.vtu';export_supplied(vtu,data);tree=ET.parse(vtu)
            codes=tree.find("UnstructuredGrid/FieldData/DataArray[@Name='point_ids_json_utf8']").text
            self.assertEqual(json.loads(bytes(map(int,codes.split())).decode()),['003','node-A'])
            self.assertIsNotNone(tree.find(".//PointData/DataArray[@Name='error_u']"))

    def test_npz_int64_ids_are_not_rounded_and_objects_are_rejected(self):
        c=example_config('interval_poisson')
        with TemporaryDirectory() as directory:
            path=Path(directory)/'nodes.npz';ids=np.array([2**62+123,17],dtype=np.int64);np.savez(path,points=[[.8],[.2]],ids=ids)
            data=read_point_data(source(path,coordinates='points',ids='ids'),c)
            self.assertEqual(data.point_ids,ids.tolist())
            snapshot=dict(points=data.points,values=[1.,2.],point_ids=data.point_ids,evaluation=data.provenance,config=c)
            out=Path(directory)/'nodes.vtu';export_supplied(out,snapshot)
            self.assertEqual([int(v) for v in ET.parse(out).find(".//PointData/DataArray[@Name='point_id']").text.split()],ids.tolist())
            np.savez(path,points=[[.8],[.2]],ids=np.array(['a','b'],dtype=object))
            with self.assertRaises(ValueError):read_point_data(source(path,coordinates='points',ids='ids'),c)

    def test_transient_ids_varying_points_and_physical_timestamps(self):
        c=example_config('expert_burgers')
        with TemporaryDirectory() as directory:
            path=Path(directory)/'time.csv';path.write_text('id,x,t,u\n10,0.5,1,0.5\n10,-0.2,0,-0.2\n11,0.1,0,0.1\n')
            points=read_point_data(source(path,'x,t'),c);ref=read_point_data(source(path,'x,t','u'),c,reference=True)
            class Field:
                iter=3
                def predict(self,x):return x[:,0:1]
            data=evaluate_outputs(Field(),c,[OutputRequest('out','paraview',3,None,points,ref)])['out']
            out=Path(directory)/'time.pvd';export_supplied(out,data);frames=ET.parse(out).findall('Collection/DataSet')
            self.assertEqual([float(v.get('timestep')) for v in frames],[0.,1.])
            first=ET.parse(out.parent/frames[0].get('file'));last=ET.parse(out.parent/frames[1].get('file'))
            self.assertEqual(first.find('UnstructuredGrid/Piece').get('NumberOfPoints'),'2');self.assertEqual(last.find('UnstructuredGrid/Piece').get('NumberOfPoints'),'1')
            self.assertEqual(first.find(".//PointData/DataArray[@Name='source_row']").text,'1 2')
            self.assertEqual(first.find('UnstructuredGrid/Piece/Points/DataArray').text.split()[1::3],['0','0'])

    def test_invalid_ids_bounds_reference_and_cylinder_solid(self):
        c=example_config('interval_poisson')
        with TemporaryDirectory() as directory:
            path=Path(directory)/'bad.csv'
            for text in ('id,x\na,0.2\na,0.3\n','id,x\na,nan\n','id,x\na,1.2\n'):
                path.write_text(text)
                with self.assertRaises(ValueError):read_point_data(source(path),c)
            path.write_text('id,x,u\na,0.3,1\n');ref=read_point_data(source(path,values='u'),c,reference=True)
            with self.assertRaises(ValueError):align_reference(ref,[[.2]],['a'],c)
            with self.assertRaises(ValueError):align_reference(ref,[[.3]],['b'],c)
        flow=example_config('expert_stokes')
        with self.assertRaises(ValueError):validate_points([[flow.cylinder_x,flow.cylinder_y]],flow)
        validate_points([[flow.cylinder_x+flow.cylinder_radius,flow.cylinder_y]],flow)
        periodic=example_config('expert_torus');validate_points(np.array([periodic.bounds[1]],dtype=np.float32),periodic)

    def test_supplied_archive_opens_without_source_and_rejects_corrupt_ids(self):
        from copy import deepcopy
        c=example_config('interval_poisson')
        with TemporaryDirectory() as directory:
            path=Path(directory)/'points.csv';path.write_text('id,x\na,0.7\nb,0.2\n');points=read_point_data(source(path),c)
            class Field:
                iter=2
                def predict(self,x):return x
            output=evaluate_outputs(Field(),c,[OutputRequest('out','solution',3,None,points)])['out'];output.pop('config')
            data=dict(format='margam-run',version=3,model=dict(format='margam-model',version=20,config=asdict(c),nodes=[dict(id='out',kind='solution',title='Result',position=[0,0],settings=dict(resolution=3,history_limit=100))],wires=[]),result=dict(outputs={'out':output},steps=2,cancelled=False,device='cpu'),monitors={},checkpoint=None)
            archive=Path(directory)/'run.pt';save_run(archive,data);path.unlink();loaded=load_run(archive);self.assertEqual(loaded['result']['outputs']['out']['point_ids'],['a','b'])
            bad=deepcopy(data);bad['result']['outputs']['out']['point_ids']=['a','a']
            with self.assertRaises(ValueError):save_run(archive,bad)
            self.assertEqual(load_run(archive)['result']['outputs']['out']['point_ids'],['a','b'])

    def test_reference_without_ids_and_ambiguous_coordinates(self):
        from margam.components.evaluation_data import PointData
        c=example_config('interval_poisson')
        ref=PointData([[.2],[.8]],None,[[2.],[8.]],{})
        self.assertEqual(align_reference(ref,[[.8],[.2]],None,c).tolist(),[[8.],[2.]])
        ref=PointData([[.2],[.2]],None,[[2.],[3.]],{})
        with self.assertRaisesRegex(ValueError,'Repeated coordinates'):align_reference(ref,[[.2],[.2]],None,c)
        ref=PointData([[.2],[.2]],['a','b'],[[2.],[3.]],{})
        self.assertEqual(align_reference(ref,[[.2],[.2]],['b','a'],c).tolist(),[[3.],[2.]])

    def test_comparison_exports_keep_errors_on_regular_grids_too(self):
        from margam.components.outputs import ReferenceSource
        from margam.components.paraview_export import export_vti,export_spacetime_pvd
        from margam.components.point_export import export_vtu
        for key in ('interval_poisson','expert_burgers','expert_stokes','expert_torus','expert_cylinder'):
            c=example_config(key)
            class Field:
                iter=2
                def predict(self,x):return x[:,0:1].repeat(1,len(c.field_names))
            reference=ReferenceSource('Known offset',lambda points:[[p[0]+1 for _ in c.field_names] for p in points])
            data=evaluate_outputs(Field(),c,[OutputRequest('export','paraview',3,reference)])['export']
            with self.subTest(case=key),TemporaryDirectory() as directory:
                path=Path(directory)/('out.pvd' if c.is_time_dependent else 'out.vtu' if c.has_obstacle else 'out.vti')
                (export_spacetime_pvd if c.is_time_dependent else export_vtu if c.has_obstacle else export_vti)(path,data)
                if c.is_time_dependent:path=path.parent/ET.parse(path).find('Collection/DataSet').get('file')
                tree=ET.parse(path)
                for field in c.field_names:
                    self.assertIsNotNone(tree.find(".//PointData/DataArray[@Name='ref_"+field+"']"))
                    errors=list(map(float,tree.find(".//PointData/DataArray[@Name='error_"+field+"']").text.split()))
                    self.assertTrue(np.allclose(errors,-1.,atol=1e-6))
