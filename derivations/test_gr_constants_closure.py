import unittest

from derivations.gr_constants_closure import run_checks


class DerivationClosureTests(unittest.TestCase):
    def test_symbolic_audits(self):
        results = run_checks()
        self.assertTrue(all(item.passed for item in results))


if __name__ == "__main__":
    unittest.main()
