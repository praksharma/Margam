"""Data-only document validation and replacement-safe writes."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from margam.templates.poisson import PoissonConfig, BoundarySampling
from margam.ui.model_document import decode_document, read_model, write_model, MAX_BYTES


def draft():
    config=PoissonConfig()
    config.boundary_sampling['left']=BoundarySampling(23,'random',100)
    return {'format':'margam-model','version':1,'config':asdict(config),
            'nodes':[{'id':'geometry','kind':'domain','title':'Rectangle','position':[0,10],
                      'settings':{'resolution':41,'history_limit':10000}},
                     {'id':'sampler','kind':'interior','title':'Interior sampler','position':[250,10],
                      'settings':{'resolution':41,'history_limit':10000}}],
            'wires':[{'source':'geometry','target':'sampler','input':'input'}]}


class ModelDocumentTests(unittest.TestCase):
    def test_draft_round_trip_and_atomic_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'draft.margam.json'
            data=draft();data['config']['layers']=[]  # Incomplete drafts are saveable.
            write_model(path,data)
            config,nodes,wires=decode_document(read_model(path))
            self.assertEqual(config.layers,())
            self.assertEqual(config.boundary_recipe('left'),BoundarySampling(23,'random',100))
            self.assertEqual(nodes,data['nodes']);self.assertEqual(wires,data['wires'])
            original=path.read_bytes()
            with patch('margam.ui.model_document.os.replace',side_effect=OSError('failed replacement')):
                with self.assertRaises(OSError):write_model(path,draft())
            self.assertEqual(path.read_bytes(),original)
            self.assertEqual(list(Path(directory).iterdir()),[path])

    def test_reject_unsupported_and_malformed_documents(self):
        cases=[]
        def add(change):
            data=deepcopy(draft());change(data);cases.append(data)
        add(lambda d:d.update(version=99))
        add(lambda d:d['nodes'][0].update(kind='python_script'))
        add(lambda d:d['nodes'][1].update(id='geometry'))
        add(lambda d:d['nodes'][0].update(position=[float('nan'),0]))
        add(lambda d:d['wires'][0].update(source='missing'))
        add(lambda d:d['wires'][0].update(input='missing'))
        add(lambda d:d['wires'].append(dict(d['wires'][0])))
        add(lambda d:d['nodes'][1].update(kind='network'))
        add(lambda d:d['config'].update(sampling=['grid']))
        add(lambda d:d['config']['bc_types'].pop('left'))
        add(lambda d:d['config']['boundary_sampling']['left'].update(seed=-1))
        add(lambda d:d['config'].update(seed=True))
        add(lambda d:d['config'].update(points=-1))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.json'
            for data in cases:
                with self.subTest(data=data):
                    path.write_text(json.dumps(data))
                    with self.assertRaises(ValueError):read_model(path)
            for content in (b'{',b'\xff',b'{"version":1,"version":2}',b' '* (MAX_BYTES+1)):
                path.write_bytes(content)
                with self.assertRaises(ValueError):read_model(path)
