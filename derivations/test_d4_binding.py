import math
from fractions import Fraction as F
import unittest

import numpy as np
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss

from d4_binding import (BETA,GAMMA,MASS,RHO,Interval,adjust,energy,endpoint_bound,
                       frozen_gate,gaussian_moments,pi_interval,report,symbolic_checks,trial_upper)


class FrozenD4Tests(unittest.TestCase):
    def test_symbolic_independent_reconstruction(self):
        checks=symbolic_checks();self.assertEqual(len(checks),20)
        self.assertTrue(all(x['passed'] for x in checks.values()))
    def test_exact_threshold_and_adversarial_mass(self):
        self.assertTrue(frozen_gate());self.assertFalse(frozen_gate(MASS*F(101,100)))
        self.assertLess(MASS,(3/Interval(RHO).sqrt()).lo)
    def test_outward_interval_and_zero_rejection(self):
        v=Interval(2).sqrt();self.assertLessEqual(v.lo*v.lo,2);self.assertGreaterEqual(v.hi*v.hi,2)
        q=Interval(-3,-2)*Interval(-2,4);self.assertLessEqual(q.lo,-12);self.assertGreaterEqual(q.hi,6)
        with self.assertRaises(ZeroDivisionError):_ =1/Interval(-1,1)
    def test_pi_independent_machin(self):
        p=pi_interval();self.assertLess(p.hi-p.lo,F(1,10**55))
        self.assertTrue(F('3.14159265358979323846264338327950288419716939937510')<p.lo)
        self.assertLess(p.hi,F('3.14159265358979323846264338327950288419716939937511'))
    def test_fourier_constant_quadrature(self):
        for eta in [.1,1,10]:
            value,err=quad(lambda s:1/(s*s+eta),0,np.inf,epsabs=1e-11)
            self.assertAlmostEqual(value/(4*math.pi),1/(8*math.sqrt(eta)),places=11)
    def test_chirped_dilation_independent_reintegration(self):
        m,a,b,q,cx,cy=float(MASS),2.3,.7,.8,.4,-.2
        v=gaussian_moments(m,a,b,q,cx,cy);z,w=adjust(v);lam=math.sqrt(z)
        direct=gaussian_moments(m,a/lam,b/lam,q*lam,cx*z,cy*z)
        for key in w:self.assertAlmostEqual(w[key],direct[key],places=10)
        self.assertLessEqual(energy(w),energy(v)+1e-11)
    def test_uniform_ratio_and_virial(self):
        for mass in [.01,1,float(MASS),4]:
            z,w=adjust(gaussian_moments(mass,.4,3,2,.3,-.5))
            self.assertAlmostEqual(w['D']-w['K']-float(BETA)*w['Q']/4+float(GAMMA)*w['S']/3,0,places=10)
            self.assertLessEqual(w['S']/w['Q'],math.sqrt(float(RHO))*mass/4+1e-11)
    def test_endpoint_full_span_and_junction(self):
        ms=4*BETA/(GAMMA*(1+3*BETA**2/(16*GAMMA)))
        for mu in [MASS/F(10**j) for j in range(1,19)]+[2*MASS-2*ms,MASS/2]:
            self.assertGreater(endpoint_bound(mu).lo,0)
        with self.assertRaises(ValueError):endpoint_bound(0)
    def test_endpoint_slope_and_asymptotic(self):
        r=report();self.assertLess(F(r['enclosures']['left_slope_upper_enclosure']['exact'][1]),0)
        lead=float(F(r['enclosures']['endpoint_F_linear_coefficient']['exact'][0]))
        mu=MASS/F(10**12);self.assertAlmostEqual(float(endpoint_bound(mu).lo/mu)/lead,1,places=8)
    def test_gaussian_bound_direct_energy(self):
        m=float(MASS);beta=float(BETA);g=float(GAMMA)
        c2=m/50-beta*m*m/(40*math.pi);c4=30203*m/40000+g*m**3/(900*math.pi**2)
        R=math.sqrt(-2*c4/c2);v=gaussian_moments(m,10*R,R,1)
        self.assertAlmostEqual(energy(v),float(trial_upper(MASS).lo),places=12)
    def test_unadjusted_profile_shortcut_false(self):
        v=gaussian_moments(float(MASS),.5,.5)
        self.assertGreater(-float(BETA)*v['Q']/2+2*float(GAMMA)*v['S']/3,0)
    def test_two_profile_formula_independent_quadrature(self):
        # Independent coordinate quadrature checks the inherited trial evaluator.
        import importlib.util
        from pathlib import Path
        spec=importlib.util.spec_from_file_location('old_diag',Path(__file__).resolve().parents[1]/'audit/strict_binding/run_diagnostics.py')
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        m1,m2,w,d,theta=1.,float(MASS)-1,4.,8.,.7
        z,weights=leggauss(180);x=z*(d/2+10*w);y=z*10*w
        xx,yy=np.meshgrid(x,y,indexing='ij');ww=weights[:,None]*weights[None,:]*(d/2+10*w)*10*w
        f1=math.sqrt(m1/(math.pi*w*w))*np.exp(-((xx+d/2)**2+yy*yy)/(2*w*w)+1j*xx)
        f2=math.sqrt(m2/(math.pi*w*w))*np.exp(-((xx-d/2)**2+yy*yy)/(2*w*w)+1j*xx+1j*theta)
        field=f1+f2;raw=float(np.sum(ww*abs(field)**2));fac=(m1+m2)/raw
        af1=(((xx+d/2)**2+yy*yy)/w**4-2/w**2-2j*(xx+d/2)/w**2)*f1
        af2=(((xx-d/2)**2+yy*yy)/w**4-2/w**2-2j*(xx-d/2)/w**2)*f2
        value=fac*float(np.sum(ww*abs(af1+af2)**2))-float(BETA)/2*fac**2*float(np.sum(ww*abs(field)**4))+float(GAMMA)/3*fac**3*float(np.sum(ww*abs(field)**6))
        self.assertAlmostEqual(value,old.two_gaussians(m1,m2,w,d,theta)['energy'],places=11)


if __name__=='__main__':unittest.main()
