"""Check the series against independent finite differences and truncation changes."""
import unittest
import numpy as np
from margam.templates.poisson import PoissonConfig, rectangle_reference, make_poisson_reference


class RectangleReferenceTests(unittest.TestCase):
    def test_boundary_symmetry_and_truncation(self):
        c=PoissonConfig()
        points=[[0,.3],[1,.3],[.4,0],[.4,1],[.25,.3],[.75,.3],[.25,.7],[.5,.5]]
        values=rectangle_reference(c,points)
        np.testing.assert_allclose(values[:4],0,atol=1e-12)
        self.assertAlmostEqual(values[4],values[5],12)
        self.assertAlmostEqual(values[4],values[6],12)
        self.assertGreater(values[-1],0)
        np.testing.assert_allclose(values,rectangle_reference(c,points,80),atol=3e-6)
        self.assertIn('truncated',make_poisson_reference(c).name)
        c.bc_types['top']='Neumann'
        with self.assertRaisesRegex(ValueError,'Dirichlet'):make_poisson_reference(c)

    def test_matches_independent_finite_difference_solution(self):
        c=PoissonConfig(x_min=-1,x_max=2,y_min=2,y_max=4,source=2.)
        c.bc_values=dict.fromkeys(c.sides,3.)
        nx,ny=31,21
        dx=(c.x_max-c.x_min)/(nx-1);dy=(c.y_max-c.y_min)/(ny-1)
        grid=np.full((ny,nx),3.)
        for _ in range(10000):
            new=grid.copy()
            new[1:-1,1:-1]=((grid[1:-1,2:]+grid[1:-1,:-2])/dx**2+(grid[2:,1:-1]+grid[:-2,1:-1])/dy**2+c.source)/(2/dx**2+2/dy**2)
            difference=np.max(np.abs(new-grid));grid=new
            if difference<1e-12:break
        points=[[x,y] for y in np.linspace(c.y_min,c.y_max,ny) for x in np.linspace(c.x_min,c.x_max,nx)]
        reference=np.array(rectangle_reference(c,points)).reshape(ny,nx)
        self.assertLess(np.max(np.abs(reference-grid)),.002)
