"""Exact-row multi-run comparisons against A; no implicit ranking or alignment."""
from copy import deepcopy
import unittest
from unittest.mock import patch
try:
    from . import test_run_comparison as fixture
except ImportError:
    import test_run_comparison as fixture
from margam.run_comparison import compare_predictions,compare_prediction_group,read_predictions


class PredictionGroupMathTests(unittest.TestCase):
    def test_three_vector_runs_use_a_denominator_and_separate_own_references(self):
        a=fixture.prediction([[0.,2.],[2.,4.]],fields=['u','v'],reference=[[0.,2.],[2.,4.]])
        b=fixture.prediction([[1.,2.],[3.,6.]],fields=['u','v'],reference=[[1.,2.],[3.,6.]])
        c=fixture.prediction([[2.,3.],[4.,5.]],fields=['u','v']);before=deepcopy([a,b,c])
        value=compare_prediction_group([(r,'out') for r in [a,b,c]])
        for index,run in enumerate([b,c]):
            pair=compare_predictions(a,'out',run,'out');self.assertEqual(value['comparisons'][index]['diagnostics'],pair['diagnostics'])
        self.assertEqual(value['headers'],['x','u · A','u · B','u · B−A','u · C','u · C−A','v · A','v · B','v · B−A','v · C','v · C−A'])
        self.assertEqual(value['preview'][0],[0.,0.,1.,1.,2.,2.,2.,2.,0.,3.,1.]);self.assertIsNone(value['comparisons'][1]['diagnostics'][0]['reference_b']);self.assertEqual(before,[a,b,c])

    def test_later_mismatch_rejects_whole_group_and_preserves_typed_duplicate_rows(self):
        a=fixture.prediction([1.,2.],points=[[0.],[0.]],ids=[1,'1']);b=deepcopy(a);c=deepcopy(a)
        value=compare_prediction_group([(a,'out'),(b,'out'),(c,'out')]);self.assertEqual([row[1] for row in value['preview']],[1,'1'])
        for mutate in (lambda r:r['outputs']['out'].update(point_ids=['1',1]),lambda r:r['outputs']['out'].update(points=[[0.],[1.]]),lambda r:r.update(coordinates=['t']),lambda r:r['outputs']['out'].update(fields=['v'])):
            bad=deepcopy(c);mutate(bad)
            with self.assertRaisesRegex(ValueError,'Run C versus A'):compare_prediction_group([(a,'out'),(b,'out'),(bad,'out')])
        a=fixture.prediction([0.,1.]);a['outputs']['out']['point_ids']=None
        self.assertEqual(compare_prediction_group([(a,'out')]*3)['headers'],['x','u · A','u · B','u · B−A','u · C','u · C−A'])

    def test_bounds_all_rows_metrics_undefined_overflow_and_cancel(self):
        a=fixture.prediction([0.]*250,points=[[float(i)] for i in range(250)]);b=fixture.prediction([0.]*249+[10.],points=a['outputs']['out']['points'])
        value=compare_prediction_group([(a,'out')]+[(b,'out')]*7);self.assertEqual(len(value['preview']),200);self.assertEqual(value['rows'],250);self.assertEqual(len(value['diagnostics']),7);self.assertEqual(value['diagnostics'][-1]['max_difference']['value'],10.);self.assertIsNone(value['diagnostics'][-1]['relative_difference']['value'])
        for n in (0,1,9):
            with self.assertRaises(ValueError):compare_prediction_group([(a,'out')]*n)
        with self.assertRaises(InterruptedError):compare_prediction_group([(a,'out')]*3,check=lambda:True)
        with self.assertRaises(InterruptedError):compare_prediction_group([(a,'out')]*3,check=iter([False,True]).__next__)
        a=fixture.prediction([-1e308,-1e308]);b=fixture.prediction([1e308,1e308]);self.assertEqual(compare_prediction_group([(a,'out'),(b,'out'),(b,'out')])['preview'][0][-1],'overflow')


class PredictionGroupUITests(unittest.TestCase):
    setUpClass=classmethod(fixture.RunComparisonTests.setUpClass.__func__)
    setUp=fixture.RunComparisonTests.setUp
    tearDown=fixture.RunComparisonTests.tearDown
    wait=fixture.RunComparisonTests.wait
    study=fixture.RunComparisonTests.study
    archives=fixture.RunComparisonTests.archives
    settle=fixture.RunComparisonTests.settle

    def loaded(self,device='cpu'):
        paths=self.archives(device);dialog=self.w.show_run_comparison();self.assertTrue(dialog.add_slot())
        for i,path in enumerate([paths[0],paths[1],paths[0]]):self.assertTrue(dialog.choose_file(i,path));self.settle(dialog)
        return dialog,paths
    def exercise(self,device):
        import torch
        from margam.ui.model_document import capture_model
        dialog,paths=self.loaded(device);model=capture_model(self.w);undo=self.w.undo_stack.index();cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None;files=[p.read_bytes() for p in paths]
        self.assertTrue(dialog.compare());self.assertIsNotNone(dialog.worker);self.assertFalse(dialog.add_slot());self.assertFalse(dialog.remove_slot());self.settle(dialog)
        self.assertIsNotNone(dialog.comparison,dialog.status.text());self.assertEqual(dialog.metrics.rowCount(),2);self.assertEqual(dialog.metrics.item(0,0).text(),'B');self.assertEqual(dialog.metrics.item(1,0).text(),'C');self.assertEqual(dialog.preview.rowCount(),200);self.assertIn('C ·',dialog.details.toPlainText())
        self.assertEqual(model,capture_model(self.w));self.assertEqual(undo,self.w.undo_stack.index());self.assertTrue(torch.equal(cpu,torch.get_rng_state()));self.assertEqual(files,[p.read_bytes() for p in paths])
        if gpu is not None:self.assertTrue(torch.equal(gpu,torch.cuda.get_rng_state()))
        self.assertTrue(dialog.remove_slot());self.assertIsNone(dialog.comparison);self.assertTrue(dialog.compare());self.assertEqual(dialog.metrics.rowCount(),1)
        self.w.undo_stack.setClean();self.w.close();self.assertTrue(all(r is None for r in dialog.runs));self.assertIsNone(dialog.comparison)
    def test_real_cpu_three_snapshots_async_comparison_preserves_state(self):self.exercise('cpu')
    @unittest.skipUnless(__import__('torch').cuda.is_available(),'CUDA required')
    def test_real_cuda_three_snapshots_stay_cpu_only_and_preserve_rng(self):self.exercise('cuda')

    def test_later_mismatch_failed_start_cancel_and_parent_close_release_snapshots(self):
        dialog,_=self.loaded();third=dialog.runs[2];identifier=dialog.choices[2].currentData();third['outputs'][identifier]['points'][0][0]+=.1
        self.assertTrue(dialog.compare());self.settle(dialog);self.assertIsNone(dialog.comparison);self.assertIn('coordinates',dialog.status.text())
        with patch('margam.ui.run_comparison.Reader.start',side_effect=RuntimeError('injected start failure')):self.assertFalse(dialog.compare())
        self.assertIsNone(dialog.worker);self.assertFalse(dialog.cancel.isEnabled())
        def slow(selections,check):
            import time
            while not check():time.sleep(.002)
            raise InterruptedError('cancelled')
        with patch('margam.ui.run_comparison.compare_prediction_group',side_effect=slow):
            self.assertTrue(dialog.compare());dialog.cancel_reading();self.settle(dialog);self.assertIsNone(dialog.comparison)
            self.assertTrue(dialog.compare());self.w.undo_stack.setClean();self.w.close();self.assertIsNone(dialog.worker);self.assertTrue(all(r is None for r in dialog.runs))
        self.app.processEvents()

    def test_slot_limits_and_finished_cancel_ignore_stale_completion(self):
        dialog=self.w.show_run_comparison()
        for _ in range(6):self.assertTrue(dialog.add_slot())
        self.assertFalse(dialog.add_slot());self.assertFalse(dialog.add_run.isEnabled())
        for _ in range(6):self.assertTrue(dialog.remove_slot())
        self.assertFalse(dialog.remove_slot());self.assertFalse(dialog.remove_run.isEnabled());self.app.processEvents()
        dialog.close();dialog,_=self.loaded()
        with patch('margam.ui.run_comparison.compare_prediction_group',return_value=object()):
            self.assertTrue(dialog.compare());dialog.worker.wait();dialog.cancel_reading();self.settle(dialog)
        self.assertIsNone(dialog.comparison);dialog.close();self.assertIsNone(dialog.worker);self.assertTrue(all(r is None for r in dialog.runs));self.app.processEvents()


if __name__=='__main__':unittest.main()
