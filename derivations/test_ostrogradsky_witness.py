import unittest
import sympy as sp
from derivations.ostrogradsky_witness import check

class HigherTimeWitness(unittest.TestCase):
    def test_unbounded_momentum(self):
        result=check()
        self.assertTrue(result["passed"])
        H=sp.sympify(result["hamiltonian"])
        self.assertEqual(sp.simplify(sp.diff(H,sp.Symbol("p1"))-sp.Symbol("Q2")),0)

if __name__=="__main__":
    unittest.main()
