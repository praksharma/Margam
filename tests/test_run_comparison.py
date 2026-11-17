"""Exact-row numerical comparison and immutable CPU archive inspection."""
from copy import deepcopy
import math
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch
try:
    from . import test_study_cli as fixture
except ImportError:
    import test_study_cli as fixture
from margam.run_comparison import compare_predictions,read_predictions
from margam.execution.study_runner import run_study
from margam.ui.model_document import capture_model


def prediction(values,*,points=None,fields=None,reference=None,ids=None):
    output=dict(points=points or [[0.],[1.]],values=values,reference=reference)
    if fields is not None:output['fields']=fields
    if ids is not None:output['point_ids']=ids
    return dict(coordinates=['x'],config={},outputs={'out':output})


class PredictionMathTests(unittest.TestCase):
    def test_fieldwise_analytical_values_and_separate_reference_errors(self):
        a=prediction([[0.,2.],[2.,4.]],fields=['u','v'],reference=[[0.,2.],[2.,4.]])
        b=prediction([[1.,2.],[3.,6.]],fields=['u','v'],reference=[[1.,2.],[3.,6.]])
        value=compare_predictions(a,'out',b,'out');u,v=value['diagnostics']
        self.assertAlmostEqual(u['rms_difference']['value'],1.);self.assertAlmostEqual(u['max_difference']['value'],1.)
        self.assertAlmostEqual(u['relative_difference']['value'],math.sqrt(.5));self.assertAlmostEqual(v['rms_difference']['value'],math.sqrt(2.))
        self.assertEqual(u['reference_a']['value'],0.);self.assertEqual(u['reference_b']['value'],0.);self.assertFalse(value['identical_reference_values'])
        self.assertEqual(value['preview'][0],[0.,0.,1.,1.,2.,2.,0.])

    def test_rows_coordinates_fields_and_typed_ids_cannot_be_silently_reordered(self):
        a=prediction([0.,1.],ids=[1,'2'])
        for mutate in (lambda b:b.update(coordinates=['t']),lambda b:b['outputs']['out'].update(points=[[1.],[0.]]),
            lambda b:b['outputs']['out'].update(fields=['v']),lambda b:b['outputs']['out'].update(point_ids=['1','2']),
            lambda b:b['outputs']['out'].pop('point_ids'),lambda b:b['outputs']['out'].update(points=[[0.],[1.+1e-15]])):
            b=deepcopy(a);mutate(b)
            with self.assertRaises(ValueError):compare_predictions(a,'out',b,'out')
        a=prediction([0.,1.],points=[[0],[2**53]])
        b=prediction([0.,1.],points=[[0],[2**53+1]])
        with self.assertRaises(ValueError):compare_predictions(a,'out',b,'out')

    def test_zero_denominator_overflow_and_preview_cap_are_explicit(self):
        a=prediction([0.,0.]);b=prediction([1.,1.]);value=compare_predictions(a,'out',b,'out')
        self.assertIsNone(value['diagnostics'][0]['relative_difference']['value']);self.assertEqual(value['diagnostics'][0]['relative_difference']['status'],'undefined_zero_reference')
        a=prediction([-1e308,-1e308]);b=prediction([1e308,1e308]);value=compare_predictions(a,'out',b,'out')
        self.assertEqual(value['preview'][0][-1],'overflow');self.assertIsNone(value['diagnostics'][0]['max_difference']['value'])
        points=[[float(i)] for i in range(250)];a=prediction([0.]*250,points=points);b=prediction([0.]*249+[10.],points=points)
        value=compare_predictions(a,'out',b,'out');self.assertEqual(len(value['preview']),200);self.assertEqual(value['rows'],250)
        self.assertAlmostEqual(value['diagnostics'][0]['rms_difference']['value'],math.sqrt(.4))


class RunComparisonTests(unittest.TestCase):
    setUpClass=classmethod(fixture.StudyCLITests.setUpClass.__func__)
    def setUp(self):
        fixture.StudyCLITests.setUp(self);self.errors=[];self.hook=sys.excepthook;sys.excepthook=lambda *args:self.errors.append(args)
    def tearDown(self):
        try:fixture.StudyCLITests.tearDown(self);self.assertEqual(self.errors,[])
        finally:sys.excepthook=self.hook
    wait=fixture.StudyCLITests.wait
    study=fixture.StudyCLITests.study
    def archives(self,device='cpu'):
        self.w.load_burgers_reference_example()
        for key,value in [('points',8),('boundary_points',4),('boundary_sampling',{}),('layers',(4,))]:self.w.change(key,value)
        output=self.root/'runs';report=run_study(self.study('[1,2]'),output,device=device);self.assertEqual(report['counts'],{'completed':2})
        return [output/f'DP{index:04d}/artifacts/run.margam-run.pt' for index in (1,2)]
    def settle(self,dialog):
        deadline=time.monotonic()+45
        while dialog.worker is not None and time.monotonic()<deadline:self.app.processEvents();time.sleep(.005)
        self.assertIsNone(dialog.worker);self.app.processEvents()
    def test_cpu_reader_preserves_rng_and_files_without_network_restoration(self):
        import torch
        paths=self.archives();before=[path.read_bytes() for path in paths];rng=torch.get_rng_state().clone()
        with patch('margam.templates.poisson.build_pinn',side_effect=AssertionError('No network')),patch('margam.nn.factory.build_network',side_effect=AssertionError('No network')):
            a,b=[read_predictions(path) for path in paths]
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertNotIn('checkpoint',a);self.assertNotIn('owned_files',a)
        identifier=next(key for key,data in a['outputs'].items() if data['reference'] is not None);value=compare_predictions(a,identifier,b,identifier)
        self.assertEqual(value['rows'],2827);self.assertTrue(value['identical_reference_values']);self.assertIn('iterations',value['configuration_differences'])
        self.assertEqual(before,[path.read_bytes() for path in paths])
    @unittest.skipUnless(__import__('torch').cuda.is_available(),'CUDA required')
    def test_cuda_archives_are_compared_on_cpu_without_rng_or_device_changes(self):
        import torch
        paths=self.archives('cuda');cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone();device=self.w.application_device
        a,b=[read_predictions(path) for path in paths];identifier=next(iter(a['outputs']));value=compare_predictions(a,identifier,b,identifier)
        self.assertEqual(a['device'],'cuda');self.assertEqual(value['rows'],2827);self.assertTrue(torch.equal(cpu,torch.get_rng_state()));self.assertTrue(torch.equal(gpu,torch.cuda.get_rng_state()))
        self.assertEqual(self.w.application_device,device)
    def test_dialog_comparison_does_not_replace_current_model_and_close_cancels(self):
        paths=self.archives();model=capture_model(self.w);index=self.w.undo_stack.index();result=self.w.result
        dialog=self.w.show_run_comparison()
        for side,path in enumerate(paths):self.assertTrue(dialog.choose_file(side,path));self.settle(dialog)
        self.assertTrue(dialog.compare());self.assertEqual(dialog.preview.rowCount(),200);self.assertEqual(dialog.metrics.rowCount(),1)
        self.assertEqual(model,capture_model(self.w));self.assertEqual(index,self.w.undo_stack.index());self.assertIs(result,self.w.result)
        self.assertTrue(dialog.choose_file(1,self.root/'missing.pt'));self.settle(dialog);self.assertFalse(dialog.compare());self.assertIsNone(dialog.comparison)
        with patch('margam.ui.run_comparison.read_predictions',side_effect=lambda path,check: self.cancel_wait(check)):
            self.assertTrue(dialog.choose_file(1,paths[1]));dialog.close();self.assertIsNone(dialog.worker)
        self.assertIs(dialog,self.w.show_run_comparison());self.assertFalse(dialog.compare_button.isEnabled());dialog.close()
    def cancel_wait(self,check):
        while not check():time.sleep(.002)
        raise InterruptedError('cancelled')
    def test_corrupt_symlink_and_cancelled_reads_do_not_touch_archives(self):
        path=self.root/'bad.pt';path.write_bytes(b'invalid archive')
        with self.assertRaises(Exception):read_predictions(path)
        other=self.root/'link.pt';other.symlink_to(path)
        with self.assertRaisesRegex(ValueError,'symlink'):read_predictions(other)
        with self.assertRaises(InterruptedError):read_predictions(path,check=lambda:True)
        self.assertEqual(path.read_bytes(),b'invalid archive')


if __name__=='__main__':unittest.main()
