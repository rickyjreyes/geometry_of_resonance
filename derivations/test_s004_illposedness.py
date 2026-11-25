"""Unit tests for derivations/s004_illposedness_verify.py (S004 Phase XXXIII).

Each test re-runs an exact (sympy) or numerical check of the variable-coefficient ill-posedness
construction against the corrected S004 density.
Run: python -m unittest -v derivations.test_s004_illposedness
"""
import unittest

from derivations import s004_illposedness_verify as v


class Linearization(unittest.TestCase):
    def test_structure_1plus1(self):
        r = v.linearization_structure(1)
        self.assertEqual((r["operator_order"], r["max_background_derivative"],
                          r["max_F_derivative"], r["max_V_derivative"]), (4, 4, 3, 2))

    def test_structure_1plus2(self):
        r = v.linearization_structure(2)
        self.assertEqual((r["operator_order"], r["max_F_derivative"], r["max_V_derivative"]), (4, 3, 2))

    def test_principal_part_exact_on_nonconstant_backgrounds(self):
        # raises AssertionError unless the order-4 coefficients are exactly 2F(rho0)(...)I2
        for spec in (v._bg_class_c(), v._bg_class_b(), v._bg_sampled()):
            _, _, rho00 = v._background_operator(spec)
            self.assertGreater(rho00, 0)


class Characteristics(unittest.TestCase):
    def test_hadamard_root_simple(self):
        self.assertEqual(len(v.hadamard_root_simplicity()["samples"]), 4)

    def test_exact_beam_1d(self):
        self.assertIn("sqrt(pi/N)", v.exact_beam_1d()["L2_norm_squared"])

    def test_exact_beam_phase_terminates_in_all_dimensions(self):
        r = v.exact_beam_phase((1, 2, 3))
        self.assertEqual(sorted(k for k in r if k.startswith("d=")), ["d=1", "d=2", "d=3"])
        self.assertLessEqual(r["sup |S(x)-1|/|x| on |x|<=1/4 (sampled)"], 9 / 4)

    def test_conjugation_identity(self):
        self.assertEqual(sorted(v.conjugation_expansion((1, 2))), ["1+1", "1+2"])

    def test_class_a_corotating_root_asymptotics(self):
        table = v.corotating_class_a()["table"]
        self.assertLess(table[-1]["|lam/k - w| (nearest root)"], 1e-3)
        errs = [row["|lam/k - w| (nearest root)"] for row in table]
        self.assertTrue(all(b < a for a, b in zip(errs, errs[1:])))

    def test_exponent_bookkeeping(self):
        self.assertTrue(all(all(g[0] != "-" for g in row["gaps"]) for row in v.exponent_bookkeeping()["table"]))


class EndToEnd(unittest.TestCase):
    def test_localized_beam_on_class_c_background(self):
        r = v.wkb_end_to_end(full=False)
        run = r["B1 class C: cos y + 0.3 i cos y sin(y+t) (nodes at y = +-pi/2)"]
        self.assertTrue(all(run["verdict"].values()))
        ratios = [row["ratio |u(T_N)|/(data+residual)"] for row in run["rows"]]
        self.assertGreater(ratios[-1], 100)

    def test_negative_control_wrong_root_gives_no_contradiction(self):
        Bexact, keys, _ = v._background_operator(v._bg_class_c())
        w = v._root(1.0, 1.0, 0.0)
        c = v._wkb_run(Bexact, keys, 1.0, 1.0, 0.0, 1.05 * w, Ns=(400, 800, 1600), exact_eikonal=False)
        ratios = [row["ratio |u(T_N)|/(data+residual)"] for row in c["rows"]]
        self.assertLess(ratios[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
