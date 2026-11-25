import math
from fractions import Fraction as F
from pathlib import Path
import sys
import unittest

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import quad

sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"audit"/"strict_binding"))
from strict_binding import BETA, D4_MASS, constants, energy, gaussian_moments, negative_trial, split_is_certified
from run_diagnostics import two_gaussians


class BindingChecks(unittest.TestCase):
    def test_resolvent_integral_independent_quadrature(self):
        for k in [.3,1.,2.7]:
            value,_ = quad(lambda r:r/((r*r-k*k)**2+k**4),0,np.inf,epsabs=1e-12)
            self.assertAlmostEqual(value/(2*math.pi),3/(16*k*k),places=10)

    def test_exact_frozen_mass_outside_sufficient_range(self):
        ms=constants()["m_star"]
        self.assertLess(ms,D4_MASS)
        self.assertLess(D4_MASS,4*ms/3)
        self.assertEqual(BETA,F(float.fromhex("0x1.05f1d178c75ebp+2")))

    def test_all_split_endpoints_inside_range(self):
        ms=constants()["m_star"]
        for m in [ms/100,ms/2,ms]:
            for fraction in [F(1,10**9),F(1,7),F(1,2),1-F(1,10**9)]:
                self.assertTrue(split_is_certified(m,fraction*m))

    def test_D4_partial_splits_and_uncovered_small_splits(self):
        ms=constants()["m_star"]
        low=2*D4_MASS-2*ms
        self.assertTrue(split_is_certified(D4_MASS,low))
        self.assertTrue(split_is_certified(D4_MASS,D4_MASS/2))
        self.assertFalse(split_is_certified(D4_MASS,low-F(1,10**10)))
        self.assertFalse(split_is_certified(D4_MASS,D4_MASS/1000))

    def test_invalid_hypotheses_rejected(self):
        for args in [(0,1,1),(1,0,1),(1,1,0),(-1,1,1)]:
            with self.assertRaises(ValueError): constants(*args)
        for p in [0,1,-1,2]:
            with self.assertRaises(ValueError): split_is_certified(1,p)

    def test_gaussian_moments_independent_integrals(self):
        # Direct Gaussian-Hermite integration of the field and its Laplacian.
        mass,a,b,q,k=1.7,2.3,1.4,.8,1.1
        nodes,weights=hermgauss(64)
        xx,yy=np.meshgrid(nodes,nodes,indexing="ij")
        x,y=a*xx,b*yy
        u=np.sqrt(mass/(math.pi*a*b))*np.exp(-(xx*xx+yy*yy)/2+1j*q*x)
        lap=((x*x/a**4+y*y/b**4)-(1/a**2+1/b**2)-q*q-2j*q*x/a**2)*u
        weight=a*b*weights[:,None]*weights[None,:]*np.exp(xx*xx+yy*yy)
        expected=gaussian_moments(mass,a,b,q,k)
        measured={"M":np.sum(weight*abs(u)**2),"D":np.sum(weight*abs(lap)**2),
                  "T":np.sum(weight*abs(lap+k*k*u)**2),
                  "Q":np.sum(weight*abs(u)**4),"S":np.sum(weight*abs(u)**6)}
        for key in measured:self.assertAlmostEqual(float(measured[key]),expected[key],places=10)

    def test_negative_trial_across_scales(self):
        for mass in [1e-4,.01,1.,float(D4_MASS),1e3]:
            t=negative_trial(mass)
            self.assertLess(t["coefficient_R_minus_2"],0)
            self.assertGreater(t["coefficient_R_minus_4"],0)
            a,b=t["a"]*math.sqrt(t["R_squared"]),math.sqrt(t["R_squared"])
            h=energy(gaussian_moments(mass,a,b))
            self.assertLess(h,0)
            self.assertAlmostEqual(h/t["energy_upper_bound"],1.,places=9)

    def test_negative_energy_does_not_force_amplitude_monotonicity(self):
        mm=gaussian_moments(150*math.pi,10,10)
        beta=float(BETA)
        self.assertLess(energy(mm),0)
        f=lambda t:energy({key:(t**{"Q":2,"S":3}.get(key,1))*v for key,v in mm.items()})/t
        derivative=(f(1+1e-5)-f(1-1e-5))/(2e-5)
        self.assertGreater(derivative,0)
        self.assertAlmostEqual(derivative, -beta*mm["Q"]/2+2*beta*mm["S"]/3,places=6)

    def test_density_dilation_can_raise_energy_per_mass(self):
        m=75*math.pi
        f=lambda t:energy(gaussian_moments(t*m,10*math.sqrt(t),10*math.sqrt(t),math.sqrt(.75/t)))/t
        mm=gaussian_moments(m,10,10,math.sqrt(.75))
        self.assertLess(f(1),0)
        derivative=(f(1+1e-5)-f(1-1e-5))/(2e-5)
        self.assertGreater(derivative,0)
        self.assertAlmostEqual(derivative,2*(mm["K"]-mm["D"]),places=6)

    def test_two_profiles_independent_quadrature(self):
        m1,m2,w,d,theta=.7,1.1,2.,3.,.8
        nodes,weights=hermgauss(96)
        scale=2*w
        xx,yy=np.meshgrid(nodes,nodes,indexing="ij")
        x,y=scale*xx,scale*yy
        def packet(m,c):
            u=np.sqrt(m/(math.pi*w*w))*np.exp(-((x-c)**2+y*y)/(2*w*w)+1j*x)
            au=(((x-c)**2+y*y)/w**4-2/w**2-2j*(x-c)/w**2)*u
            return u,au
        u,au=packet(m1,-d/2);v,av=packet(m2,d/2)
        u=u+np.exp(1j*theta)*v;au=au+np.exp(1j*theta)*av
        weight=scale*scale*weights[:,None]*weights[None,:]*np.exp(xx*xx+yy*yy)
        raw=float(np.sum(weight*abs(u)**2));factor=(m1+m2)/raw
        measured={"M":m1+m2,"T":factor*float(np.sum(weight*abs(au)**2)),
                  "Q":factor**2*float(np.sum(weight*abs(u)**4)),
                  "S":factor**3*float(np.sum(weight*abs(u)**6))}
        exact=two_gaussians(m1,m2,w,d,theta)
        self.assertAlmostEqual(raw,exact["raw_mass"],places=8)
        for key in measured:self.assertAlmostEqual(measured[key],exact["moments"][key],places=6)

    def test_far_separation_and_cancellation(self):
        for phase in [0.,.7,math.pi]:
            r=two_gaussians(1,1,2,40,phase)
            self.assertLess(abs(r["interaction_difference"]),1e-12)
        with self.assertRaises(ValueError):two_gaussians(1,1,2,0,math.pi)

    def test_uncovered_does_not_mean_refuted(self):
        # Preserve the logical distinction required by the analytic theorem.
        self.assertFalse(split_is_certified(10,F(1,2)))
        self.assertLess(negative_trial(10)["energy_upper_bound"],0)


if __name__ == "__main__":unittest.main()
