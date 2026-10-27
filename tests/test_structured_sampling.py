"""GeoSampler-derived exact-count designs stay reproducible in physical bounds."""
import unittest
from margam.components.sampling import sample_box
from margam.templates.poisson import PoissonConfig,BoundarySampling,sample_problem


class StructuredSamplingTests(unittest.TestCase):
    def test_exact_counts_bounds_and_seeds(self):
        for method in ('hammersley','stratified'):
            for count in (1,7,31):
                rows=sample_box([-2,4],[3,7],count,method,11)
                self.assertEqual(len(rows),count)
                self.assertTrue(all(-2<=x<3 and 4<=y<7 for x,y in rows))
                self.assertEqual(rows,sample_box([-2,4],[3,7],count,method,11))
                self.assertNotEqual(rows,sample_box([-2,4],[3,7],count,method,12))
                self.assertEqual(len({tuple(row) for row in rows}),count)
            config=PoissonConfig(sampling=method,points=31)
            config.boundary_sampling['left']=BoundarySampling(7,method,13)
            interior,boundaries=sample_problem(config)
            self.assertEqual(len(interior),31);self.assertEqual(len(boundaries['left']),7)
