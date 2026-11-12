"""Paired comparison covers a complete continuation and restores global hooks."""
import unittest
from unittest.mock import patch
import torch
from examples.compare_dsgnar_sketch import paired


class SketchComparisonTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_production_and_reference_full_continuation(self):
        report=paired('wave','cpu',16,attempts=1,order=('candidate','dense'))
        self.assertTrue(report['full_checkpoint_parity']);self.assertEqual(report['candidate_mode'],'production-dispatch')
        self.assertEqual(report['order'],['candidate','dense'])
        self.assertGreater(report['branches']['candidate']['seconds_per_attempt'],0)

    def test_reference_hook_is_restored_when_candidate_fails(self):
        from margam.methods import dsgnar
        original=dsgnar.count_add
        with patch.object(dsgnar,'_count_add_touched',side_effect=MemoryError('injected')):
            with self.assertRaisesRegex(MemoryError,'injected'):paired('wave','cpu',16,attempts=1,order=('candidate','dense'))
        self.assertIs(dsgnar.count_add,original)


if __name__=='__main__':unittest.main()
