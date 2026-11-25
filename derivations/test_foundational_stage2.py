import math
import unittest

from derivations.foundational_stage2 import (
    run_checks, minmax_interval, periodic_circle_eigenvalue,
    static_poles, finite_minimum,
)


class FoundationalStage2Tests(unittest.TestCase):
    def test_all_scoped_checks(self):
        checks=run_checks()
        self.assertEqual(len(checks),6)
        self.assertTrue(all(c.passed for c in checks),checks)

    def test_periodic_circle_and_holonomy(self):
        L=2*math.pi
        self.assertAlmostEqual(periodic_circle_eigenvalue(0,L),0)
        self.assertAlmostEqual(periodic_circle_eigenvalue(1,L),1)
        self.assertAlmostEqual(periodic_circle_eigenvalue(0,L,.5),.25)
        self.assertEqual(minmax_interval(1,-.2,.4),(.8,1.4))
        with self.assertRaises(ValueError):
            minmax_interval(1,2,-1)
        with self.assertRaises(ValueError):
            periodic_circle_eigenvalue(1,0)

    def test_finite_minimum_opposite_residue(self):
        result=finite_minimum(.001,2.,2000.,1.)
        self.assertEqual(result["status"],"LOWER_OSCILLATORY_MINIMUM_NOT_UNSTABLE_BAND")
        self.assertGreater(result["q_min"],0)
        self.assertLess(result["q_min"],result["q_collision"])
        self.assertGreater(result["z_lower"],0)
        self.assertGreater(result["z_upper"],result["z_lower"])
        self.assertGreater(result["lower_derivative"],0)
        self.assertLess(result["upper_derivative"],0)
        self.assertEqual(finite_minimum(.001,0,2000,1)["status"],
                         "NO_INTERIOR_LOW_BRANCH_MINIMUM")

    def test_pole_collision_and_complex(self):
        # At q=0, 4*K*m2=1 yields exact repeated pole.
        collision=static_poles(0,1,0,1,.25)
        self.assertEqual(collision["status"],"COLLISION")
        # Allowed positive quadratic form does NOT guarantee real high-k poles.
        complex_pole=static_poles(2,1,0,1,0)
        self.assertEqual(complex_pole["status"],"COMPLEX_POLE")
        with self.assertRaises(ValueError):
            static_poles(0,1,3,1,0)


if __name__=="__main__":
    unittest.main()
