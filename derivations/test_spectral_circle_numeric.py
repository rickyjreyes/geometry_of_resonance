import math
import unittest

from derivations.spectral_circle_numeric import gallery


class SpectralCircleNumericalTests(unittest.TestCase):
    def test_two_lengths_nonconstant_potential_and_connection(self):
        for L in (2*math.pi,7.):
            report=gallery(L=L,a=.17,v0=.4,v1=.2,N=12)
            self.assertTrue(report["finite_matrix_minmax_pass"])
            self.assertTrue(all(x<=y<=z for x,y,z in zip(
                report["lower_first_six"],report["eigenvalues_first_six"],
                report["upper_first_six"])))
    def test_invalid(self):
        with self.assertRaises(ValueError):
            gallery(L=0)
        with self.assertRaises(ValueError):
            gallery(N=1)


if __name__=="__main__":
    unittest.main()
