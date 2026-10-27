"""Versioned planar import preserves labels/normals and rejects implicit 3D projection."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from margam.components.geosampler_adapter import prepare_planar_export
from margam.components.prepared_data import read_points


class GeoSamplerAdapterTests(unittest.TestCase):
    def test_planar_schema_and_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.npz'
            metadata={'schema_version':2,'coordinate_units':'source units unspecified',
                      'datasets':[{'label':'wall','array':'wall','normals':'normals','entity_tags':'tags'}]}
            points=np.array([[0.,0.,2.],[0.,1.,2.]])
            np.savez(path,wall=points,normals=[[-1.,0.,0.],[-1.,0.,0.]],tags=[1,1],_gs_metadata=np.asarray(json.dumps(metadata)))
            output=Path(directory)/'prepared';report=prepare_planar_export(path,output,(0,1),['wall'])
            self.assertEqual(report['datasets'][0]['label'],'wall');self.assertEqual(report['plane_coordinate'],2.)
            source={key:report['datasets'][0][key] for key in ('coordinates','normals','values')};source['path']=str(output/'dataset-000.npz')
            coordinates,normals,_=read_points(source,2)
            self.assertEqual(coordinates.tolist(),[[0.,0.],[0.,1.]])
            self.assertEqual(normals.tolist(),[[-1.,0.],[-1.,0.]])
            with self.assertRaises(FileExistsError):prepare_planar_export(path,output)
            points[1,2]=3
            np.savez(path,wall=points,normals=[[-1.,0.,0.],[-1.,0.,0.]],tags=[1,1],_gs_metadata=np.asarray(json.dumps(metadata)))
            with self.assertRaisesRegex(ValueError,'plane'):prepare_planar_export(path,Path(directory)/'invalid')
