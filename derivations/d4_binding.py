#!/usr/bin/env python3
"""Phase V: independent identities and rationally enclosed frozen gates.

No Phase IV derivation is imported. Analytic inequalities live in the reports;
these identities and scalar enclosures corroborate, rather than replace, proof.
"""
import argparse
from decimal import Decimal, localcontext
from fractions import Fraction as F
import json
import math
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite import hermgauss
import sympy as sp

BETA_HEX='0x1.05f1d178c75ebp+2'
MASS_HEX='0x1.45b8a0bad2714p+1'
BETA=F(float.fromhex(BETA_HEX));GAMMA=BETA;MASS=F(float.fromhex(MASS_HEX))
RHO=1+3*BETA**2/(32*GAMMA)
SCALE=10**60


class Interval:
    """Outward-rounded rational interval, with no floating-point sign decisions."""
    def __init__(self,lo,hi=None):
        lo=F(lo);hi=lo if hi is None else F(hi)
        if lo>hi:raise ValueError('reversed interval')
        self.lo=F((lo*SCALE).__floor__(),SCALE)
        self.hi=F((hi*SCALE).__ceil__(),SCALE)
    @staticmethod
    def make(x):return x if isinstance(x,Interval) else Interval(x)
    def __add__(self,x):
        x=self.make(x);return Interval(self.lo+x.lo,self.hi+x.hi)
    __radd__=__add__
    def __neg__(self):return Interval(-self.hi,-self.lo)
    def __sub__(self,x):return self+-self.make(x)
    def __rsub__(self,x):return self.make(x)+-self
    def __mul__(self,x):
        x=self.make(x);v=[a*b for a in [self.lo,self.hi] for b in [x.lo,x.hi]]
        return Interval(min(v),max(v))
    __rmul__=__mul__
    def __truediv__(self,x):
        x=self.make(x)
        if x.lo<=0<=x.hi:raise ZeroDivisionError('interval contains zero')
        return self*Interval(1/x.hi,1/x.lo)
    def __rtruediv__(self,x):return self.make(x)/self
    def __pow__(self,n):
        if n<0:return 1/(self**(-n))
        out=Interval(1)
        for _ in range(n):out=out*self
        return out
    def sqrt(self):
        if self.lo<0:raise ValueError('negative square-root interval')
        low=math.isqrt(self.lo.numerator*SCALE*SCALE//self.lo.denominator)
        high=math.isqrt(self.hi.numerator*SCALE*SCALE//self.hi.denominator)+1
        return Interval(F(low,SCALE),F(high,SCALE))
    def record(self):
        with localcontext() as ctx:
            ctx.prec=42
            display=[str(Decimal(v.numerator)/Decimal(v.denominator)) for v in [self.lo,self.hi]]
        return {'exact':[str(self.lo),str(self.hi)],'decimal_display':display}


def pi_interval():
    # Alternating remainder, then Machin's exact identity pi=16 atan(1/5)-4 atan(1/239).
    def atan(q):
        x=F(1,q);s=sum(((-1)**j*x**(2*j+1)/F(2*j+1) for j in range(80)),F(0))
        nxt=x**161/F(161)
        return Interval(s,s+nxt)
    return 16*atan(5)-4*atan(239)


def trial_upper(s):
    s=Interval.make(s);pi=pi_interval()
    c2=s/50-BETA*s*s/(40*pi)
    c4=F(30203,40000)*s+GAMMA*s**3/(900*pi*pi)
    assert c2.hi<0 and c4.lo>0
    return -(c2*c2)/(4*c4)


def endpoint_bound(mu):
    mu=F(mu)
    if not 0<mu<=MASS/2:raise ValueError('half-interval requires 0<mu<=m/2')
    b=MASS-mu;mb=3/Interval(RHO).sqrt()
    return MASS*mu/(Interval(b)**2)*(1-(MASS+b)/(2*mb))*(-trial_upper(b))


def frozen_gate(mass=MASS):return 9-RHO*F(mass)**2>0


def gaussian_moments(m,a,b,q=1.0,cx=0.0,cy=0.0,k=1.0):
    """Complex chirped anisotropic Gaussian; exact polynomial GH moments."""
    if min(m,a,b,k)<=0:raise ValueError('positive mass, widths and k required')
    z,w=hermgauss(5);x,y=np.meshgrid(a*z,b*z,indexing='ij')
    weights=w[:,None]*w[None,:]/math.pi
    ax=-1/a**2+1j*cx;ay=-1/b**2+1j*cy
    lap=ax+ay+(ax*x+1j*q)**2+(ay*y)**2
    D=m*float(np.sum(weights*abs(lap)**2))
    K=m*(q*q+(abs(ax)**2*a*a+abs(ay)**2*b*b)/2)
    T=m*float(np.sum(weights*abs(lap+k*k)**2))
    return dict(M=m,D=D,K=K,T=T,Q=m*m/(2*math.pi*a*b),S=m**3/(3*math.pi**2*a*a*b*b))


def energy(v):return v['T']-float(BETA)*v['Q']/2+float(GAMMA)*v['S']/3


def adjust(v):
    beta=float(BETA);gamma=float(GAMMA)
    z=(2*v['K']+beta*v['Q']/2)/(2*(v['D']+gamma*v['S']/3))
    w=dict(M=v['M'],D=z*z*v['D'],K=z*v['K'],Q=z*v['Q'],S=z*z*v['S'])
    w['T']=w['D']-2*w['K']+w['M']
    return z,w


def symbolic_checks():
    D,K,M,Q,S,beta,gamma,k,r,t,z,eta=sp.symbols('D K M Q S beta gamma k r t z eta',positive=True)
    T=D-2*k*k*K+k**4*M;H=T-beta*Q/2+gamma*S/3
    P=D+gamma*S/3;B=2*k*k*K+beta*Q/2;V=D-k*k*K-beta*Q/4+gamma*S/3
    rho=k**4+3*beta**2/(32*gamma);sigma=sp.sqrt(rho);mb=3*beta/(gamma*sigma)
    c=3*beta/(8*gamma);checks={}
    def check(name,left,right):
        v=sp.simplify(left-right);checks[name]={'passed':v==0,'residual':str(v)}
    check('coercivity_square',-beta*r*r/2+gamma*r**3/3,gamma*r*(r-3*beta/(4*gamma))**2/3-3*beta**2*r/(16*gamma))
    check('amplitude_defect',(t*T-beta*t*t*Q/2+gamma*t**3*S/3)/t-H,(t-1)*(-beta*Q/2+gamma*(t+1)*S/3))
    check('amplitude_derivative',sp.diff(T-beta*t*Q/2+gamma*t*t*S/3,t).subs(t,1),-beta*Q/2+2*gamma*S/3)
    dil=z*z*D-2*k*k*z*K+k**4*M-beta*z*Q/2+gamma*z*z*S/3
    check('dilation_polynomial',dil,P*z*z-B*z+k**4*M)
    check('dilation_minimum',dil-dil.subs(z,B/(2*P)),P*(z-B/(2*P))**2)
    check('virial_identity',sp.diff(dil,z).subs(z,1),2*V)
    check('virial_square',D+T+2*gamma*(S-2*c*Q+c*c*M)/3-rho*M,2*V)
    s=sp.symbols('s',real=True)
    check('fourier_antiderivative',sp.diff(sp.atan(s/sp.sqrt(eta))/sp.sqrt(eta),s),1/(s*s+eta))
    check('fourier_integral_constant',(sp.pi/(2*sp.sqrt(eta)))/(4*sp.pi),1/(8*sp.sqrt(eta)))
    check('optimized_interpolation',((D+eta*M)/(8*sp.sqrt(eta))).subs(eta,D/M),sp.sqrt(M*D)/4)
    check('new_threshold',gamma*(sigma/4)*(2*mb)/3,beta/2)
    a,b=sp.symbols('a b',positive=True);h=1-(a+b)/(2*mb)
    check('two_mass_bracket',-beta/2+gamma*(b/a+1)*(sigma*a/4)/3,-beta*h/2)
    e=sp.symbols('e',real=True)
    check('slope_quotient',(b*(1+(b-a)*h/a)*e-a*e)/(b-a),(1+(b/a)*h)*e)
    check('endpoint_slope_factor',(1+(b/a)*h).subs(a,b),2-b/mb)
    mu=sp.symbols('mu',positive=True)
    check('weighted_binding',mu*e+(M-mu)*e,M*e)
    check('endpoint_F_leading',sp.limit((M*mu/(M-mu)**2*(1-(2*M-mu)/(2*mb)))/mu,mu,0),(1-M/mb)/M)
    c2,c4=sp.symbols('c2 c4',nonzero=True)
    check('gaussian_optimal_scale',(c2/z+c4/z**2).subs(z,-2*c4/c2),-c2*c2/(4*c4))
    check('old_threshold',3*beta/(4*gamma*(3*(k**4+3*beta**2/(16*gamma))/(16*k*k))),4*beta*k*k/(gamma*(k**4+3*beta**2/(16*gamma))))
    check('chemical_potential',T-beta*Q+gamma*S,H-beta*Q/2+2*gamma*S/3)
    check('density_dilation',D/t**2-2*k*k*K/t+k**4*M-beta*Q/2+gamma*S/3-H,(1-1/t)*(-(1+1/t)*D+2*k*k*K))
    return checks


def report():
    checks=symbolic_checks();assert len(checks)==20 and all(r['passed'] for r in checks.values())
    sigma=Interval(RHO).sqrt();mb=3/sigma;u=trial_upper(MASS);h0=1-MASS/mb
    ms=4*BETA/(GAMMA*(1+3*BETA**2/(16*GAMMA)))
    encl={'pi':pi_interval(),'new_m_B':mb,'mass_margin':mb-MASS,'h0':h0,'U_m':u,
          'left_slope_upper_enclosure':(2-MASS/mb)*u/MASS,
          'endpoint_F_linear_coefficient':h0*(-u)/MASS,'small_mass_quadratic_constant':BETA*sigma/8}
    gates={'D4_below_new_threshold':frozen_gate(),'D4_above_old_threshold':MASS>ms,
           'gaussian_negative_on_half_interval':BETA*MASS/2>F(4,5)*F(22,7),
           'negative_endpoint_slope':encl['left_slope_upper_enclosure'].hi<0,
           'positive_endpoint_coefficient':encl['endpoint_F_linear_coefficient'].lo>0,
           'central_interval_overlapped':0<2*MASS-2*ms<MASS/2}
    assert all(gates.values())
    return {'schema':'d4-binding-identities-v1','status':'PASS','primary':'D4_STRICT_BINDING_PROVED',
      'parameters':{'beta_hex':BETA_HEX,'gamma_hex':BETA_HEX,'mass_hex':MASS_HEX,'k':'1'},
      'exact':{'rho':str(RHO),'mass':str(MASS),'old_m_star':str(ms),'old_split_endpoints':[str(2*MASS-2*ms),str(2*ms-MASS)],'D4_gate_9_minus_rho_m_squared':str(9-RHO*MASS*MASS)},
      'enclosures':{n:v.record() for n,v in encl.items()},'whole_interval_gates':gates,
      'symbolic_checks':checks,'universal_all_mass_theorem':False,'external_mathematical_review':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report(),indent=2)+'\n')
    print('PASS: 20 independent symbolic identities; exact D4 gate; certified endpoint constants')
