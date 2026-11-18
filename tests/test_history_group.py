"""Bounded multi-run history overlays retain sparse records and reader ownership."""
from copy import deepcopy
import time
import unittest
from unittest.mock import patch
try:
    from . import test_history_comparison as fixture
except ImportError:
    import test_history_comparison as fixture
from margam.history_comparison import compare_history_group,compare_histories


class GroupHistoryMathTests(unittest.TestCase):
    def test_three_histories_gaps_missing_terms_zero_and_configuration_differences(self):
        a=fixture.history([(5,{'total':4.}),(6,{'total':0.}),(9,{'pde':1.})]);b=fixture.history([(1,{'total':3.}),(3,{'total':2.})]);c=fixture.history([(5,{'pde':8.}),(8,{'total':1.})]);c['config']={'layers':[8]}
        before=deepcopy([a,b,c]);value=compare_history_group([(r,'monitor') for r in [a,b,c]],'total')
        self.assertEqual(value['preview'],[(1,None,3.,None),(3,None,2.,None),(5,4.,None,None),(6,0.,None,None),(8,None,None,1.)])
        self.assertEqual(value['series'][0]['segments'],[[(5,4.)]]);self.assertEqual(value['series'][2]['missing'],1);self.assertEqual(value['configuration_differences'],['layers']);self.assertEqual(before,[a,b,c])
        pair=compare_histories(a,'monitor',b,'monitor','total');group=compare_history_group([(a,'monitor'),(b,'monitor')],'total')
        for key in pair:self.assertEqual(pair[key],group[key])
    def test_bounds_eight_curves_and_exact_preview_limit(self):
        run=fixture.history([(i,{'total':1./i}) for i in range(200,500)])
        value=compare_history_group([(run,'monitor')]*8,'total');self.assertEqual(len(value['series']),8);self.assertEqual(len(value['preview']),200);self.assertEqual(value['preview_total'],300);self.assertEqual(len(value['series'][7]['values']),300)
        for selections in [[],[(run,'monitor')],[(run,'monitor')]*9]:
            with self.assertRaises(ValueError):compare_history_group(selections,'total')
        with self.assertRaises(ValueError):compare_history_group([(run,'monitor')]*3,'missing')


class GroupHistoryUITests(unittest.TestCase):
    setUpClass=classmethod(fixture.HistoryUITests.setUpClass.__func__)
    setUp=fixture.HistoryUITests.setUp
    tearDown=fixture.HistoryUITests.tearDown
    wait=fixture.HistoryUITests.wait
    study=fixture.HistoryUITests.study
    archives=fixture.HistoryUITests.archives
    settle=fixture.HistoryUITests.settle

    def exercise(self,device):
        import torch
        from margam.ui.model_document import capture_model
        paths=self.archives(device);model=capture_model(self.w);undo=self.w.undo_stack.index();cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
        dialog=self.w.show_history_comparison();self.assertTrue(dialog.add_slot())
        for i,path in enumerate([paths[0],paths[1],paths[0]]):self.assertTrue(dialog.choose_file(i,path));self.settle(dialog)
        self.assertTrue(dialog.compare());self.assertEqual(dialog.metrics.rowCount(),3);self.assertEqual(dialog.preview.columnCount(),4);self.assertEqual(dialog.preview.horizontalHeaderItem(3).text(),'C · exact value');self.assertEqual(len(dialog.comparison['series']),3);self.assertIn('C ·',dialog.details.toPlainText());self.assertFalse(dialog.plot.grab().isNull())
        self.assertTrue(dialog.remove_slot());self.assertIsNone(dialog.comparison);self.assertTrue(dialog.compare());self.assertEqual(dialog.metrics.rowCount(),2)
        self.assertEqual(model,capture_model(self.w));self.assertEqual(undo,self.w.undo_stack.index());self.assertTrue(torch.equal(cpu,torch.get_rng_state()))
        if gpu is not None:self.assertTrue(torch.equal(gpu,torch.cuda.get_rng_state()))
        self.w.undo_stack.setClean();self.w.close();self.assertTrue(all(run is None for run in dialog.runs));self.assertIsNone(dialog.comparison)
        dialog.close()
    def test_real_cpu_three_snapshots_and_remove_preserve_state(self):self.exercise('cpu')
    @unittest.skipUnless(__import__('torch').cuda.is_available(),'CUDA required')
    def test_real_cuda_three_snapshots_and_remove_preserve_state(self):self.exercise('cuda')

    def test_slot_bounds_busy_cancel_failure_close_and_stale_callbacks(self):
        dialog=self.w.show_history_comparison()
        for _ in range(6):self.assertTrue(dialog.add_slot())
        self.assertFalse(dialog.add_slot());self.assertFalse(dialog.add_run.isEnabled());self.assertEqual(len(dialog.choices),8)
        def slow(path,check):
            while not check():time.sleep(.002)
            raise InterruptedError('cancelled')
        with patch('margam.ui.history_comparison.read_histories',side_effect=slow):
            self.assertTrue(dialog.choose_file(7,self.root/'missing.pt'));self.assertFalse(dialog.remove_slot());self.assertFalse(dialog.add_slot());dialog.cancel_reading();self.settle(dialog);self.assertIsNone(dialog.runs[7])
        self.assertTrue(dialog.choose_file(7,self.root/'missing.pt'));self.settle(dialog);self.assertFalse(dialog.compare())
        for _ in range(6):self.assertTrue(dialog.remove_slot())
        self.assertFalse(dialog.remove_slot());self.assertFalse(dialog.remove_run.isEnabled());self.app.processEvents()
        self.assertTrue(dialog.add_slot())
        with patch('margam.ui.history_comparison.read_histories',return_value=object()):
            self.assertTrue(dialog.choose_file(2,self.root/'dummy.pt'));dialog.worker.wait();dialog.close()
        self.assertEqual(dialog.runs,[None]*3);self.assertIsNone(dialog.worker);self.app.processEvents();self.assertIs(self.w.show_history_comparison(),dialog);self.assertFalse(dialog.compare())
        with patch('margam.ui.history_comparison.read_histories',side_effect=slow):
            self.assertTrue(dialog.choose_file(2,self.root/'dummy.pt'));self.w.undo_stack.setClean();self.w.close();self.assertIsNone(dialog.worker)

    def test_eight_curves_render_extreme_sparse_and_nonpositive_values(self):
        from margam.ui.history_comparison import HistoryPlot
        plot=HistoryPlot();plot.resize(850,300)
        for logarithmic in (True,False):
            runs=[fixture.history([(1,{'total':-1e308}),(3,{'total':5e-324}),(4,{'total':1e308})])]*8
            value=compare_history_group([(r,'monitor') for r in runs],'total',logarithmic=logarithmic);plot.set_comparison(value);self.assertFalse(plot.grab().isNull())
        plot.close()


if __name__=='__main__':unittest.main()
