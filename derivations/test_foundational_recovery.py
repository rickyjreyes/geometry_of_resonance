"""Independent probes and failure boundaries for the Phase III propositions."""
from fractions import Fraction as Q
import math
import unittest

import numpy as np
from scipy.integrate import quad
import sympy as s

from derivations.foundational_recovery import (
    gaussian_certificate, gaussian_energy, independent_probes,
    pi_enclosure, poisson, symbolic_checks,
)


class RecoveryTests(unittest.TestCase):
    def test_independent_symbolic_reconstruction(self):
        results=symbolic_checks()
        self.assertEqual(len(results),38)
        self.assertTrue(all(results.values()))

    def test_uv_companion_roots_with_tachyonic_and_massless_controls(self):
        result=independent_probes()
        self.assertEqual(len(result['UV_companion_polynomial_checks']),9)
        for row in result['UV_companion_polynomial_checks']:
            self.assertGreater(row['c_uv'],0)
            e=[abs(x-row['c_uv']) for x in row['growth_over_k']]
            self.assertLess(e[-1],e[0])

    def test_B0_does_not_mean_uniformly_stable(self):
        for k in [100.,400.]:
            q=k*k
            pure=np.roots([1,0,1+2*q,0,q+q*q])
            critical=np.roots([1,0,1,0,q])  # K=T=1,G=-2
            self.assertLess(max(pure.real),1e-5)
            self.assertGreater(max(critical.real),.6*math.sqrt(k))

    def test_auxiliary_scalar_metric_never_loses_rank(self):
        for c in [.01,1.,100.]:
            for H in [np.diag([100.,-100.]),np.array([[0.,8.],[8.,-4.]])]:
                G=np.block([[H,c*np.eye(2)],[c*np.eye(2),np.zeros((2,2))]])
                ev=np.linalg.eigvalsh(G)
                self.assertEqual((sum(ev<0),sum(ev>0)),(2,2))

    def test_Hamiltonian_second_class_constraints(self):
        a,b,pa,pb=s.symbols('a b pa pb')
        self.assertEqual(poisson(pa-b,pb+a,[a,b],[pa,pb]),-2)
        self.assertEqual(s.Matrix([[0,-2],[2,0]]).det(),4)

    def test_fixed_mass_lower_bound_near_saturation(self):
        beta,gamma=4.,3.
        for rho in [0.,.1,3*beta/(4*gamma),10.,100.]:
            residual=-beta*rho*rho/2+gamma*rho**3/3+3*beta**2*rho/(16*gamma)
            self.assertGreaterEqual(residual,-1e-10)
        # Changing the defocusing sign fails the lower-bound argument.
        self.assertLess(-beta*100**2/2-gamma*100**3/3,-1e5)

    def test_M1_curve_and_separate_nonnegative_terms(self):
        a=.5
        rho=lambda t:math.cos(t)**2+a*a*math.sin(t)**2
        N=quad(rho,0,2*math.pi,epsabs=1e-12)[0]
        mean=2*math.pi*a/N
        amplitude=quad(lambda t:((a*a-1)*math.sin(t)*math.cos(t))**2/rho(t),0,2*math.pi,epsabs=1e-12)[0]/N
        var=quad(lambda t:rho(t)*(a/rho(t)-mean)**2,0,2*math.pi,epsabs=1e-12)[0]/N
        closure=[quad(lambda t:math.cos(t)/math.sqrt(rho(t)),0,2*math.pi)[0],
                 quad(lambda t:a*math.sin(t)/math.sqrt(rho(t)),0,2*math.pi)[0]]
        self.assertAlmostEqual(mean,.8,12)
        self.assertAlmostEqual(amplitude,.2,12)
        self.assertAlmostEqual(var,.16,12)
        self.assertLess(np.linalg.norm(closure),1e-12)

    def test_generated_mode_by_direct_complex_FFT(self):
        n=128;x=2*np.pi*np.arange(n)/n
        a=.7+.3j;b=-.2+.6j;beta=1.2;gamma=2.1
        u=a*np.exp(2j*x)+b*np.exp(-1j*x)
        nonlin=-beta*abs(u)**2*u+gamma*abs(u)**4*u
        coeff=np.fft.fft(nonlin)/n
        self.assertAlmostEqual(abs(coeff[8]-gamma*a**3*np.conj(b)**2),0,12)
        self.assertGreater(abs(coeff[8]),.01)
        one=np.exp(2j*x)
        single=np.fft.fft(-beta*abs(one)**2*one+gamma*abs(one)**4*one)/n
        single[2]=0
        self.assertLess(np.linalg.norm(single),1e-13)

    def test_exact_negative_trial_and_independent_Gauss_Hermite_integral(self):
        receipt=gaussian_certificate()
        lower,upper=map(Q,receipt['energy_interval_rational'])
        self.assertLess(upper,0)
        m=float.fromhex(receipt['mass_hex']);beta=gamma=float.fromhex(receipt['beta_gamma_hex'])
        a,b,R,k=4.,1.,4.,1.
        pts,weights=np.polynomial.hermite.hermgauss(80)
        X,Y=np.meshgrid(pts,pts,indexing='ij');W=weights[:,None]*weights[None,:]
        re=(X*X-1)/(a*a*R*R)+(Y*Y-1)/(b*b*R*R)
        im=-2*k*X/(a*R)
        kinetic=m/np.pi*np.sum(W*(re*re+im*im))
        quartic=m*m/(np.pi**2*a*b*R*R)*np.sum(W*np.exp(-X*X-Y*Y))
        sextic=m**3/(np.pi**3*a*a*b*b*R**4)*np.sum(W*np.exp(-2*(X*X+Y*Y)))
        self.assertAlmostEqual(kinetic-beta*quartic/2+gamma*sextic/3,float(upper),12)
        # The theorem is a trial bound, not an asserted stationary certificate.
        self.assertFalse(receipt['whole_space_minimizer_proved'])

    def test_all_mass_2D_trial_choice(self):
        for m in [.01,.1,1.,10.]:
            beta,gamma,k,b=2.,3.,1.,1.
            a=16*np.pi*k*k/(beta*m)
            negative=2*m*k*k/a**2-beta*m*m/(4*np.pi*a*b)
            positive=m*(.75/a**4+.5/(a*a*b*b)+.75/b**4)+gamma*m**3/(9*np.pi**2*a*a*b*b)
            R=2*math.sqrt(positive/(-negative))
            self.assertLess(float(gaussian_energy(m,beta,gamma,k,a,b,R,np.pi)),0)

    def test_pi_enclosure_is_strict_and_refines(self):
        l1,h1=pi_enclosure(10);l2,h2=pi_enclosure(32)
        self.assertLess(l1,l2);self.assertLess(h2,h1)
        self.assertLess(h2-l2,Q(1,10**40))


if __name__=='__main__':unittest.main()
