"""Independent exact interpolation and negative controls for the certificate."""

from copy import deepcopy
from fractions import Fraction as F
import math
import unittest

import certify_gap as c


def factored_polynomial(b, gap=c.GAP):
    h = 1 - (c.MASS + b)*c.SIGMA_HI/6
    d = c.BETA*b/(40*c.PI_HI) - F(1, 50)
    denominator = 4*b*(F(30203, 40000) + c.BETA*b*b/(900*c.PI_LO*c.PI_LO))
    return c.MASS*h*d*d - gap*denominator


class ExactGapTests(unittest.TestCase):
    def test_hex_decoding_against_integer_fixtures_and_binary64(self):
        self.assertEqual(c.BETA, F(4608178181141995, 1125899906842624))
        self.assertEqual(c.MASS, F(1432536998431173, 562949953421312))
        self.assertEqual(c.BETA, F(float.fromhex(c.BETA_HEX)))
        self.assertEqual(c.MASS, F(float.fromhex(c.MASS_HEX)))

    def test_machin_angle_tangent_exactly_one(self):
        twice = 2*F(1, 5)/(1-F(1, 5)**2)
        four = 2*twice/(1-twice**2)
        tangent = (four-F(1, 239))/(1+four*F(1, 239))
        self.assertEqual(tangent, 1)
        lo, hi = c.machin_bounds()
        self.assertLess(c.PI_LO, lo)
        self.assertLess(hi, c.PI_HI)
        self.assertLess(hi-lo, F(1, 10**16))

    def test_independent_cubic_interpolation_recovers_bernstein(self):
        # Solve the four-value interpolation system, without polynomial
        # expansion, convolution, or the production basis transform.
        p0, p1, p2, p3 = [factored_polynomial(c.MASS*(1+t)/2)
                          for t in (F(0), F(1, 3), F(2, 3), F(1))]
        a = 27*p1 - 8*p0 - p3
        b = 27*p2 - p0 - 8*p3
        independent = [p0, (2*a-b)/18, (2*b-a)/18, p3]
        self.assertEqual(independent, c.positive_gap_coefficients())
        lower_bounds = [F(28, 100000), F(56, 100000), F(83, 100000), F(29, 10**7)]
        for coefficient, lower in zip(independent, lower_bounds):
            self.assertGreater(coefficient, lower)

    def test_polynomial_and_bernstein_evaluations_agree_exactly(self):
        coefficients = c.positive_gap_coefficients()
        for t in [F(0), F(1, 101), F(1, 3), F(1, 2), F(100, 101), F(1)]:
            b = c.MASS*(1+t)/2
            bernstein = sum(coefficients[i]*math.comb(3, i)*t**i*(1-t)**(3-i)
                            for i in range(4))
            self.assertEqual(bernstein, factored_polynomial(b))
            self.assertEqual(bernstein, c.evaluate(c.gap_polynomial(), b))

    def test_basis_transform_known_polynomials(self):
        self.assertEqual(c.bernstein_coefficients([F(1), F(-2), F(1)], 0, 1), [1, 0, 0])
        self.assertEqual(c.bernstein_coefficients([F(0), F(0), F(0), F(1)], 0, 1), [0, 0, 0, 1])
        with self.assertRaises(ValueError):
            c.bernstein_coefficients([F(1)], 1, 1)

    def test_upper_root_and_frozen_margin(self):
        self.assertGreater(c.SIGMA_HI*c.SIGMA_HI, c.RHO)
        self.assertGreater(9-c.RHO*c.MASS*c.MASS, 0)
        self.assertLess(9-c.RHO*(c.MASS*F(101, 100))**2, 0)

    def test_endpoint_extremely_small_component_no_underflow(self):
        mu = c.MASS/F(10**200)
        b = c.MASS-mu
        self.assertGreater(factored_polynomial(b), 0)
        self.assertGreater(c.GAP*mu, 0)
        self.assertEqual(c.evaluate(c.gap_polynomial(), b), factored_polynomial(b))

    def test_excessive_claim_rejected(self):
        with self.assertRaises(ValueError):
            c.positive_gap_coefficients(F(1, 300000))
        with self.assertRaises(ValueError):
            c.positive_gap_coefficients(F(0))

    def test_tampered_scalar_certificate_rejected(self):
        record = c.certificate()
        c.verify_record(record)
        for key in ("mass_hex", "gamma_hex"):
            modified = deepcopy(record)
            modified["parameters"][key] = "0x1.0p+1"
            with self.assertRaises(ValueError):
                c.verify_record(modified)
        modified = deepcopy(record)
        modified["bernstein_coefficients"][3] = "0"
        with self.assertRaises(ValueError):
            c.verify_record(modified)

    def test_overstated_proof_scope_rejected(self):
        modified = c.certificate()
        modified["scope"]["full_analytic_proof_formalized"] = True
        with self.assertRaises(ValueError):
            c.verify_record(modified)


if __name__ == "__main__":
    unittest.main()
