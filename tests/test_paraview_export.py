"""Regular-grid fields, actual physical times and failure-safe ParaView output."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
import numpy as np
from margam.components.paraview_export import export_vti,export_pvd


def snapshot(time=0.,flow=False):
    points=[[x,y] for y in (2.,3.) for x in (-1.,0.,1.)]
    values=[x+10*y+time for x,y in points]
    if flow:return dict(points=points,values=[[u,2*u,3*u] for u in values],fields=['u','v','p'])
    return dict(points=points,values=values)

class ParaViewExportTests(unittest.TestCase):
    def test_scalar_vector_grid_and_escaping(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'scalar.vti';export_vti(path,snapshot())
            root=ET.parse(path).getroot();image=root.find('ImageData')
            self.assertEqual(image.get('WholeExtent'),'0 2 0 1 0 0')
            self.assertEqual(image.get('Origin'),'-1.0 2.0 0.0')
            self.assertIsNone(image.find('FieldData')) # Steady has no fabricated time.
            self.assertEqual([float(v) for v in image.find('Piece/PointData/DataArray').text.split()],[19.,20.,21.,29.,30.,31.])
            export_vti(path,snapshot(flow=True));pd=ET.parse(path).find('ImageData/Piece/PointData')
            self.assertEqual(pd.get('Vectors'),'velocity')
            vector=next(a for a in pd if a.get('Name')=='velocity')
            self.assertEqual(vector.get('NumberOfComponents'),'3')
            self.assertEqual([float(v) for v in vector.text.split()][:6],[19.,38.,0.,20.,40.,0.])
            line={'points':[[2.],[3.],[4.]],'values':[1.,2.,3.],'fields':['temperature & concentration']}
            export_vti(path,line)
            self.assertEqual(ET.parse(path).find('ImageData').get('WholeExtent'),'0 2 0 0 0 0')
            self.assertEqual(ET.parse(path).find('ImageData/Piece/PointData/DataArray').get('Name'),'temperature & concentration')

    def test_physical_time_series_and_transactional_failure(self):
        with TemporaryDirectory() as directory:
            directory=Path(directory);path=directory/'moving & field.pvd'
            times=[0.,.125,.9];export_pvd(path,[(t,snapshot(t)) for t in times])
            entries=ET.parse(path).findall('Collection/DataSet')
            self.assertEqual([float(e.get('timestep')) for e in entries],times)
            for entry,t in zip(entries,times):
                relative=Path(entry.get('file'));self.assertFalse(relative.is_absolute())
                root=ET.parse(directory/relative)
                self.assertEqual(float(root.find('ImageData/FieldData/DataArray').text),t)
                self.assertAlmostEqual(float(root.find('ImageData/Piece/PointData/DataArray').text.split()[0]),19+t)
            before=path.read_bytes();files=set(directory.rglob('*'))
            bad=snapshot();bad['values'][0]=float('nan')
            for frames in ([],[(0,snapshot()),(0,snapshot())],[(1,snapshot()),(.5,snapshot())],[(True,snapshot())],[(float('nan'),snapshot())],[(0,snapshot()),(1,bad)],[(0,snapshot()),(1,snapshot(flow=True))]):
                with self.subTest(frames=frames),self.assertRaises(ValueError):export_pvd(path,frames)
                self.assertEqual(path.read_bytes(),before);self.assertEqual(set(directory.rglob('*')),files)
            from margam.components import paraview_export as writer
            original=writer._write
            def fail_index(destination,tree):
                if tree.getroot().get('type')=='Collection':raise OSError('simulated final index failure')
                return original(destination,tree)
            with patch.object(writer,'_write',side_effect=fail_index),self.assertRaises(OSError):export_pvd(path,[(0,snapshot())])
            self.assertEqual(path.read_bytes(),before);self.assertEqual(set(directory.rglob('*')),files)

    def test_reject_invalid_grid_before_replacing_output(self):
        cases=[]
        for change in (lambda d:d['points'].reverse(),lambda d:d['points'].pop(),lambda d:d['values'].pop(),lambda d:d['points'][0].__setitem__(0,float('nan')),lambda d:d['values'].__setitem__(0,float('inf')),lambda d:d.update(fields=['u','u']),lambda d:d.update(points=[[0.],[1.],[3.]],values=[1.,2.,3.])):
            d=deepcopy(snapshot());change(d);cases.append(d)
        with TemporaryDirectory() as directory:
            path=Path(directory)/'old.vti';path.write_text('keep me')
            for data in cases:
                with self.subTest(data=data),self.assertRaises(ValueError):export_vti(path,data)
                self.assertEqual(path.read_text(),'keep me')
            self.assertEqual(len(list(Path(directory).iterdir())),1)

    @unittest.skipUnless(importlib.util.find_spec('vtkmodules'),'Optional VTK reader not installed; runtime exporter needs no VTK')
    def test_real_vtk_reader_geometry_fields_vectors_and_times(self):
        from vtkmodules.vtkIOXML import vtkXMLImageDataReader
        from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline as Pipeline
        from vtkmodules.util.numpy_support import vtk_to_numpy
        with TemporaryDirectory() as directory:
            line=Path(directory)/'line.vti'
            export_vti(line,dict(points=[[2.],[3.],[4.]],values=[1.,2.,3.]))
            reader=vtkXMLImageDataReader();reader.SetFileName(str(line));reader.Update()
            self.assertEqual(reader.GetOutput().GetDimensions(),(3,1,1))
            np.testing.assert_array_equal(vtk_to_numpy(reader.GetOutput().GetPointData().GetArray('u')),[1.,2.,3.])
            self.assertFalse(reader.GetOutputInformation(0).Has(Pipeline.TIME_STEPS()))
            path=Path(directory)/'series.pvd';times=[0.,.125,.9]
            export_pvd(path,[(np.float64(t),snapshot(t,flow=True)) for t in times])
            for entry,t in zip(ET.parse(path).findall('Collection/DataSet'),times):
                reader=vtkXMLImageDataReader();reader.SetFileName(str(Path(directory)/entry.get('file')));reader.Update()
                self.assertEqual(reader.GetErrorCode(),0)
                result=reader.GetOutput();self.assertEqual(result.GetDimensions(),(3,2,1))
                self.assertEqual(result.GetOrigin(),(-1.,2.,0.));self.assertEqual(result.GetSpacing(),(1.,1.,1.))
                self.assertEqual(result.GetPoint(4),(0.,3.,0.))
                np.testing.assert_array_equal(vtk_to_numpy(result.GetPointData().GetArray('u')),[19+t,20+t,21+t,29+t,30+t,31+t])
                np.testing.assert_array_equal(vtk_to_numpy(result.GetPointData().GetVectors())[0],[19+t,2*(19+t),0.])
                self.assertEqual(reader.GetOutputInformation(0).Get(Pipeline.TIME_STEPS()),(t,))
