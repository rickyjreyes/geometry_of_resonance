"""Unit tests for derivations/s004_closure_verify.py (S004 Phases XXV-XXXII).

Each test re-runs one exact (sympy) or adversarial check against the corrected S004 density.
Run: python -m unittest -v derivations.test_s004_closure
"""
import unittest

from derivations import s004_closure_verify as v


class PrincipalSymbol(unittest.TestCase):
    def test_principal_symbol_1plus1(self):
        self.assertEqual(v.principal_symbol_check(1)["spatial_dims"], 1)

    def test_principal_symbol_1plus2(self):
        self.assertEqual(v.principal_symbol_check(2)["spatial_dims"], 2)

    def test_symbol_algebra(self):
        self.assertIn("boundary_perfect_square", v.symbol_algebra_checks())

    def test_ellipticity_has_no_counterexample(self):
        self.assertEqual(v.ellipticity_adversarial(samples=500)["counterexamples"], 0)

    def test_hyperbolicity_criterion_matches_roots(self):
        r = v.hyperbolicity_classification(samples=500)
        self.assertEqual(r["agreement"], r["samples"])


class Hadamard(unittest.TestCase):
    def test_exact_roots(self):
        self.assertEqual(len(v.hadamard_exact()), 2)

    def test_lower_order_terms_do_not_remove_growth(self):
        for row in v.hadamard_lower_order()["trials"]:
            self.assertLess(row["rel_err@1000"], 2e-2)

    def test_h4_replay(self):
        v.h4_replay()

    def test_kappa_to_zero_increasing(self):
        v.kappa_to_zero_table()


class Nodal(unittest.TestCase):
    def test_nodal_jets(self):
        self.assertIn("p1_at_node=2v", v.nodal_jet_checks())

    def test_unified_tangency(self):
        self.assertIn("2 kappa F", v.unified_tangency()["jerk_coefficient"])

    def test_brackets(self):
        self.assertEqual(v.nodal_brackets()["det"], 1)

    def test_crossing_series_unique(self):
        self.assertEqual(v.crossing_series(order=5)["a2"], "0")

    def test_collision_exponents(self):
        self.assertEqual(v.collision_indicial()["exponents"]["nontrivial"], ["3/5", "8/5"])
        self.assertIn("3/5", v.dominant_balance())


class NonlinearGhost(unittest.TestCase):
    def test_harmonic_forcing_coefficients(self):
        self.assertIn("2 sigma2", v.harmonic_forcing()["traveling_wave_second_harmonic"])

    def test_resonance_band(self):
        self.assertEqual(v.resonance_band_classification(samples=150)["out_of_band_with_root"], 0)

    def test_resonance_certificate(self):
        r = v.resonance_certificate()
        self.assertEqual(r["s*=B k*^4"], "663/25600")
        self.assertEqual(r["cancellation_surface"], "sigma2 = 0 iff V'''(rho0) = 4131/800")

    def test_failed_candidate_is_preserved(self):
        r = v.failed_original_candidate()
        self.assertTrue(r["label"].startswith("REFUTED"))
        self.assertGreater(r["min_resonance_function_numeric"], 0)


class Diagnostics(unittest.TestCase):
    def test_spatial_member(self):
        v.spatial_member_checks()

    def test_node_collision(self):
        self.assertEqual(v.node_collision_diagnostic()["predicted_exponent"], 0.6)


if __name__ == "__main__":
    unittest.main()
