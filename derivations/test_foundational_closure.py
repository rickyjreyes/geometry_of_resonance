import math
import unittest

import numpy as np

from derivations.foundational_closure import eft_error_check, numerical_checks, uv_growth


class FoundationalCounterexamples(unittest.TestCase):
    def test_uv_growing_modes_against_companion_matrix(self):
        result = numerical_checks()
        self.assertAlmostEqual(result["growth_coefficient_limit"],
                               math.sqrt((math.sqrt(2)-1)/2), places=14)
        self.assertGreater(min(result["asymptotic_orders"][-3:]), 1.99)

    def test_reject_parameters_outside_theorem(self):
        for parameters in [(0, 0, 1), (1, 2, 1), (1, 3, 1),
                           (1, 0, -1), (1, float("nan"), 1)]:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    uv_growth(*parameters)

    def test_negative_mixed_coefficient_is_not_silently_excluded(self):
        self.assertGreater(uv_growth(1, -1.9, 1), 0)

    def test_linear_EFT_remainder_has_explicit_error_bound(self):
        data = eft_error_check()
        self.assertEqual(len(data["rows"]), 4)
        for row in data["rows"]:
            self.assertLessEqual(float(row["error"]), float(row["bound"]))

    def test_nonconstant_locked_curve_is_not_mean_spectral_identity(self):
        # Independent quadrature of the exact smooth periodic eigenmode.
        arc = 2*np.pi*np.arange(1024)/1024
        a = .5
        chi = np.cos(arc)+1j*a*np.sin(arc)
        chi_prime = -np.sin(arc)+1j*a*np.cos(arc)
        density = abs(chi)**2
        curvature = np.imag(np.conjugate(chi)*chi_prime)/density
        tangent = chi/abs(chi)
        mean = np.mean(density*curvature)/np.mean(density)
        eigenvalue = np.mean(abs(chi_prime)**2)/np.mean(density)
        self.assertAlmostEqual(mean, .8, places=13)
        self.assertAlmostEqual(eigenvalue, 1, places=13)
        self.assertLess(abs(np.mean(tangent)), 1e-14)
        self.assertGreater(curvature.min(), 0)


if __name__ == "__main__":
    unittest.main()
