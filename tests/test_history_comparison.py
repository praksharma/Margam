"""Recorded-history gaps, loss semantics and immutable CPU/CUDA archive inspection."""
from copy import deepcopy
import sys
import time
import unittest
from unittest.mock import patch
try:
    from . import test_run_comparison as fixture
except ImportError:
    import test_run_comparison as fixture
from margam.history_comparison import retained_curve,compare_histories,available_curves,curve_label,read_histories


def history(records):return dict(config={},monitors={'monitor':records})


class HistoryMathTests(unittest.TestCase):
    def test_missing_updates_terms_and_nonpositive_log_values_break_segments(self):
        records=[(5,{'total':4.}),(6,{'total':0.}),(7,{'pde':3.}),(8,{'total':2.}),(10,{'total':1.})]
        value=retained_curve(records,'total',True)
        self.assertEqual(value['segments'],[[(5,4.)],[(8,2.)],[(10,1.)]])
        self.assertEqual(value['recorded'],5);self.assertEqual(value['present'],4);self.assertEqual(value['missing'],1);self.assertEqual(value['hidden_nonpositive'],1)
        self.assertEqual((value['first_step'],value['last_step']),(5,10));self.assertIn((6,0.),value['values'])
        linear=retained_curve(records,'total',False);self.assertEqual(linear['segments'][0],[(5,4.),(6,0.)])
    def test_legacy_weighted_history_is_never_reconstructed_or_zero_padded(self):
        a=history([(20,{'left':2.}),(21,{'left':1.,'weighted:left':5.})]);b=history([(1,{'left':3.}),(2,{'left':2.})])
        value=compare_histories(a,'monitor',b,'monitor','weighted:left')
        self.assertEqual(value['a']['present'],1);self.assertEqual(value['b']['present'],0);self.assertEqual(value['b']['missing'],2)
        self.assertEqual(value['preview'],[(21,5.,None)]);self.assertEqual(value['a']['first_step'],20)
        self.assertIn('objective contribution',curve_label('weighted:left'));self.assertIn('raw',curve_label('left'))
    def test_preview_is_bounded_but_series_keep_every_record_and_real_indices(self):
        a=history([(index,{'total':1./index}) for index in range(500,800)]);b=history([(1,{'total':3.})])
        value=compare_histories(a,'monitor',b,'monitor','total')
        self.assertEqual(value['a']['segments'][0][0][0],500);self.assertEqual(len(value['a']['segments'][0]),300)
        self.assertEqual(len(value['preview']),200);self.assertEqual(value['preview_total'],301)
        with self.assertRaises(ValueError):compare_histories(a,'monitor',b,'monitor','not saved')
        with self.assertRaises(ValueError):retained_curve([(2,{'total':1.}),(1,{'total':2.})],'total',True)


class HistoryUITests(unittest.TestCase):
    setUpClass=classmethod(fixture.RunComparisonTests.setUpClass.__func__)
    setUp=fixture.RunComparisonTests.setUp
    tearDown=fixture.RunComparisonTests.tearDown
    wait=fixture.RunComparisonTests.wait
    study=fixture.RunComparisonTests.study
    archives=fixture.RunComparisonTests.archives
    settle=fixture.RunComparisonTests.settle

    def test_real_cpu_histories_overlay_without_model_or_rng_mutation(self):
        import torch
        from margam.ui.model_document import capture_model
        paths=self.archives();before=[path.read_bytes() for path in paths];model=capture_model(self.w);undo=self.w.undo_stack.index();rng=torch.get_rng_state().clone()
        with patch('margam.templates.poisson.build_pinn',side_effect=AssertionError('No network')):
            a,b=[read_histories(path) for path in paths]
        self.assertNotIn('checkpoint',a);self.assertNotIn('outputs',a);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        identifier=next(iter(a['monitors']));self.assertEqual(len(a['monitors'][identifier]),1);self.assertEqual(len(b['monitors'][identifier]),2)
        value=compare_histories(a,identifier,b,identifier,'total');self.assertEqual(value['a']['first_step'],1);self.assertEqual(value['b']['last_step'],2)
        dialog=self.w.show_history_comparison()
        for side,path in enumerate(paths):self.assertTrue(dialog.choose_file(side,path));self.settle(dialog)
        self.assertIn(identifier[:8],dialog.choices[0].currentText());self.assertTrue(dialog.compare());self.assertEqual(dialog.metrics.rowCount(),2);self.assertIsNotNone(dialog.plot.comparison)
        dialog.log_scale.setChecked(False);self.assertIsNone(dialog.comparison);self.assertTrue(dialog.compare())
        self.assertEqual(model,capture_model(self.w));self.assertEqual(undo,self.w.undo_stack.index());self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(before,[path.read_bytes() for path in paths])
        dialog.close();self.assertIs(dialog,self.w.show_history_comparison());self.assertFalse(dialog.compare());dialog.close()
    @unittest.skipUnless(__import__('torch').cuda.is_available(),'CUDA required')
    def test_cuda_history_archives_stay_cpu_only_and_rng_is_unchanged(self):
        import torch
        paths=self.archives('cuda');cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone()
        a,b=[read_histories(path) for path in paths];identifier=next(iter(a['monitors']));self.assertEqual(a['device'],'cuda')
        self.assertEqual(compare_histories(a,identifier,b,identifier,'pde')['b']['present'],2)
        self.assertTrue(torch.equal(cpu,torch.get_rng_state()));self.assertTrue(torch.equal(gpu,torch.cuda.get_rng_state()))
    def test_no_monitor_does_not_fall_back_to_checkpoint_and_failed_read_invalidates(self):
        from margam.ui.run_document import load_run_payload,save_run
        paths=self.archives();data,_=load_run_payload(paths[0]);data['monitors']={};path=self.root/'no-monitor.pt';save_run(path,data)
        self.assertTrue(data['checkpoint']['loss_history']);self.assertEqual(read_histories(path)['monitors'],{})
        dialog=self.w.show_history_comparison();dialog.choose_file(0,path);self.settle(dialog);self.assertFalse(dialog.compare());self.assertIn('No saved',dialog.status.text())
        dialog.choose_file(0,paths[0]);self.settle(dialog);dialog.choose_file(1,paths[1]);self.settle(dialog);self.assertTrue(dialog.compare())
        dialog.choose_file(1,self.root/'missing.pt');self.settle(dialog);self.assertIsNone(dialog.comparison);self.assertIsNone(dialog.plot.comparison);self.assertFalse(dialog.compare())
        dialog.close()
    def test_cancelled_finished_reader_and_parent_shutdown_ignore_stale_callbacks(self):
        dialog=self.w.show_history_comparison()
        with patch('margam.ui.history_comparison.read_histories',return_value=object()):
            self.assertTrue(dialog.choose_file(0,self.root/'snapshot.pt'));dialog.worker.wait()
            dialog.cancel_reading();self.settle(dialog);self.assertIsNone(dialog.runs[0])
        def slow(path,check):
            while not check():time.sleep(.002)
            raise InterruptedError('cancelled')
        with patch('margam.ui.history_comparison.read_histories',side_effect=slow):
            self.assertTrue(dialog.choose_file(0,self.root/'snapshot.pt'));dialog.close();self.assertIsNone(dialog.worker)
            self.w.show_history_comparison();self.assertTrue(dialog.choose_file(1,self.root/'snapshot.pt'))
            self.w.undo_stack.setClean();self.w.close();self.assertIsNone(dialog.worker)
        self.app.processEvents()

    def test_plot_handles_zero_extremes_and_sparse_singletons_without_qt_errors(self):
        from margam.ui.history_comparison import HistoryPlot
        plot=HistoryPlot();plot.resize(800,280)
        for logarithmic,rows in [(True,[(1,{'total':0.})]),(True,[(1,{'total':5e-324}),(3,{'total':1e308})]),(False,[(1,{'total':-1e308}),(2,{'total':1e308})])]:
            a=history(rows);value=compare_histories(a,'monitor',a,'monitor','total',logarithmic=logarithmic);plot.set_comparison(value);self.assertFalse(plot.grab().isNull())
        plot.close()


if __name__=='__main__':unittest.main()
