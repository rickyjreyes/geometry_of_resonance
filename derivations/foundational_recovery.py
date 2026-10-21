#!/usr/bin/env python3
"""Phase III reconstructions, independent of foundational_closure.py.

Proofs and hypotheses are in docs/foundational_recovery. Symbolic identities,
finite-dimensional probes and exact rational enclosures are distinct outputs.
No PDE campaign, interval root campaign or mass fit is performed here.
"""
import argparse
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import sympy as s


def poisson(f, g, coordinates, momenta):
    return s.expand(sum(s.diff(f, x)*s.diff(g, p)-s.diff(f, p)*s.diff(g, x)
                        for x, p in zip(coordinates, momenta)))


def symbolic_checks():
    checks = {}
    def zero(name, expression):
        value = all(s.simplify(x) == 0 for x in expression) if isinstance(expression, s.MatrixBase) else s.simplify(expression) == 0
        checks[name] = bool(value)
        if not value:
            raise AssertionError((name, expression))

    # Direct Cartesian density, then its Legendre transform and Poisson algebra.
    k, g, th, f = s.symbols('k g th f', nonzero=True, real=True)
    x = s.Matrix(s.symbols('x1 x2', real=True))
    v = s.Matrix(s.symbols('v1 v2', real=True))
    a = s.Matrix(s.symbols('a1 a2', real=True))
    b = s.Matrix(s.symbols('b1 b2', real=True))
    lam = s.Matrix(s.symbols('l1 l2', real=True))
    p0 = s.Matrix(s.symbols('p01 p02', real=True))
    p1 = s.Matrix(s.symbols('p11 p12', real=True))
    pl = s.Matrix(s.symbols('pl1 pl2', real=True))
    grad2, pot = s.symbols('grad2 pot', real=True)
    density = v.dot(v)-grad2+pot-f*(k*(b-a).dot(b-a)+th*b.dot(b)+g*(b-a).dot(b))
    zero('S004_velocity_hessian', s.hessian(density, v)-2*s.eye(2))
    zero('S004_acceleration_hessian', s.hessian(density, a)+2*k*f*s.eye(2))
    a_inv = ((2*k+g)*b-p1/f)/(2*k)
    zero('S004_Legendre_inverse', s.Matrix([s.diff(density, ai) for ai in a]).subs(dict(zip(a, a_inv)))-p1)
    ham = (p0.dot(v)+p1.dot(a)-density).subs(dict(zip(a, a_inv)))
    expected = p0.dot(v)-v.dot(v)+grad2-pot-p1.dot(p1)/(4*k*f)+(2*k+g)*p1.dot(b)/(2*k)+(th-g*g/(4*k))*f*b.dot(b)
    zero('S004_canonical_energy', ham-expected)
    coords, moms = list(x)+list(v)+list(lam), list(p0)+list(p1)+list(pl)
    constraints = list(p0-lam)+list(pl)
    pb = s.Matrix(4, 4, lambda i,j: poisson(constraints[i], constraints[j], coords, moms))
    target = s.BlockMatrix([[s.zeros(2),-s.eye(2)],[s.eye(2),s.zeros(2)]]).as_explicit()
    zero('S004_primary_Poisson_matrix', pb-target)
    zero('S004_primary_matrix_determinant', pb.det()-1)
    # Preservation: dot(p_lambda)=-v+mu and dot(p_x-lambda)=-H_x-nu.
    mu = s.Matrix(s.symbols('mu1 mu2'))
    nu = s.Matrix(s.symbols('nu1 nu2'))
    h1 = expected.subs(dict(zip(p0, lam)))
    ht = h1+mu.dot(p0-lam)+nu.dot(pl)
    for i in range(2):
        zero(f'S004_multiplier_mu_{i}', poisson(pl[i],ht,coords,moms)-(mu[i]-v[i]))
        zero(f'S004_multiplier_nu_{i}', poisson(p0[i]-lam[i],ht,coords,moms)+nu[i])
    # x dependence suppressed in this last finite-jet check; functional H_x
    # changes only nu=-delta H_c/delta x, never the invertible multiplier block.
    j = s.Matrix([[0,-1],[1,0]])
    charge = p0.dot(j*x)+p1.dot(j*v)
    for i in range(2):
        zero(f'phase_generator_x_{i}', poisson(x[i],charge,coords,moms)-(j*x)[i])
        zero(f'phase_generator_v_{i}', poisson(v[i],charge,coords,moms)-(j*v)[i])

    # Independent Fourier substitution into the real quadratic EL equation.
    K,G,T,q,z,m2,r,V2 = s.symbols('K G T q z m2 r V2', real=True)
    ss = s.symbols('ss')
    D = z-q-m2-K*(z-q)**2+G*q*(z-q)-T*q*q
    time_poly = K*ss**4+(1+(2*K+G)*q)*ss**2+q+m2+(K+G+T)*q*q
    zero('quadratic_time_symbol', D.subs(z,-ss**2)+time_poly)
    discriminant = s.discriminant(D,z)
    zero('quadratic_discriminant', discriminant-(1-4*K*m2+2*G*q-(4*K*T-G**2)*q*q))
    for sign in [-1,1]:
        pole = q+(1+G*q+sign*s.sqrt(discriminant))/(2*K)
        zero(f'quadratic_pole_{sign}',D.subs(z,pole))
    aa=(2*K+G)/(2*K)
    principal=-K*(z-aa*q)**2-(4*K*T-G**2)*q*q/(4*K)
    zero('principal_completed_square',principal-(-K*(z-q)**2+G*q*(z-q)-T*q*q))
    zero('B0_pure_Lorentz_control',D.subs({G:0,T:0,m2:0})-(z-q)*(1-K*(z-q)))
    zero('B0_negative_mixed_boundary',D.subs({G:-2*K,T:K})-(-K*z*z+z-q-m2))
    zero('kappa_zero_spatial_symbol',D.subs({K:0,G:0})-(z-q-m2-T*q*q))
    eps,eta,xi = s.symbols('eps eta xi',real=True)
    rho=(r+eps*eta)**2+(eps*xi)**2
    zero('amplitude_potential_quadratic',s.expand(V2*(rho-r*r)**2/2).coeff(eps,2)-2*V2*r*r*eta*eta)

    # Exact auxiliary completion of the square, and Einstein-frame inertia.
    A,y,box=s.symbols('A y box',nonzero=True,real=True)
    zero('scalar_tensor_auxiliary_elimination',(y*box+y*y/(4*A)).subs(y,-2*A*box)+A*box*box)
    h11,h12,h22,c=s.symbols('h11 h12 h22 c',real=True,nonzero=True)
    H=s.Matrix([[h11,h12],[h12,h22]])
    metric=s.BlockMatrix([[H,c*s.eye(2)],[c*s.eye(2),s.zeros(2)]]).as_explicit()
    shear=s.BlockMatrix([[s.eye(2),s.zeros(2)],[-H/(2*c),s.eye(2)]]).as_explicit()
    off=s.BlockMatrix([[s.zeros(2),c*s.eye(2)],[c*s.eye(2),s.zeros(2)]]).as_explicit()
    zero('Einstein_scalar_metric_congruence',shear.T*metric*shear-off)
    zero('Einstein_scalar_metric_determinant',metric.det()-c**4)
    zero('Einstein_scalar_metric_inverse',metric.inv()-s.BlockMatrix([[s.zeros(2),s.eye(2)/c],[s.eye(2)/c,-H/c**2]]).as_explicit())

    # First-order Hamiltonian symplectic constraints: no fourth-time modes.
    ar,bi,pa,pb0=s.symbols('ar bi pa pb0')
    c1,c2=pa-bi,pb0+ar
    zero('Hamiltonian_constraint_bracket',poisson(c1,c2,[ar,bi],[pa,pb0])+2)
    density_r,beta,gamma=s.symbols('rho beta gamma',positive=True)
    W=-beta*density_r**2/2+gamma*density_r**3/3
    zero('sharp_fixed_mass_lower_bound',W-gamma*density_r*(density_r-3*beta/(4*gamma))**2/3+3*beta**2*density_r/(16*gamma))
    zero('sextic_retaining_lower_bound',W-gamma*density_r**3/6-gamma*density_r*(density_r-3*beta/(2*gamma))**2/6+3*beta**2*density_r/(8*gamma))
    zero('Hamiltonian_potential_variation',s.diff(W,density_r)-(-beta*density_r+gamma*density_r**2))
    # Laurent convolution, not trigonometric hard-coded harmonic identities.
    w,aa0,bb0=s.symbols('w aa0 bb0',nonzero=True)
    uu=aa0*w+bb0/w; ub=aa0/w+bb0*w  # real nonzero amplitudes suffice
    vertex=s.expand(-beta*uu**2*ub+gamma*uu**3*ub**2)
    zero('generated_fifth_harmonic',vertex.coeff(w,5)-gamma*aa0**3*bb0**2)

    # M1 independent polar energy, current and closed-curve antisymmetry.
    R,Rp,phip=s.symbols('R Rp phip',real=True)
    zero('polar_Dirichlet_identity',(Rp+s.I*R*phip)*(Rp-s.I*R*phip)-Rp**2-R**2*phip**2)
    t,ell=s.symbols('t ell',real=True)
    chi=s.cos(t)+s.I*ell*s.sin(t)
    zero('elliptic_mode_eigen_equation',s.diff(chi,t,2)+chi)
    zero('elliptic_mode_current',s.im(s.conjugate(chi)*s.diff(chi,t))-ell)
    zero('closed_curve_half_period',chi.subs(t,t+s.pi)+chi)
    mean=2*ell/(1+ell**2)
    zero('mass_counterexample_4_over_5',mean.subs(ell,s.Rational(1,2))-s.Rational(4,5))
    zero('mean_defect',1-mean**2-(ell**2-1)**2/(ell**2+1)**2)
    return checks


def pi_enclosure(n=32):
    """Alternating-series bounds and Machin's exact identity."""
    def atan_inv(x):
        value=sum((Q(-1)**j)/(Q(2*j+1)*Q(x)**(2*j+1)) for j in range(n))
        nxt=(Q(-1)**n)/(Q(2*n+1)*Q(x)**(2*n+1))
        return min(value,value+nxt),max(value,value+nxt)
    l5,h5=atan_inv(5);l239,h239=atan_inv(239)
    return 16*l5-4*h239,16*h5-4*l239


def gaussian_energy(m,beta,gamma,k,a,b,R,pi):
    """Exact analytic integral for the normalized 2D modulated Gaussian."""
    return (2*m*k*k/(a*a*R*R)
            +m*(Q(3,4)/a**4+Q(1,2)/(a*a*b*b)+Q(3,4)/b**4)/R**4
            -beta*m*m/(4*pi*a*b*R*R)
            +gamma*m**3/(9*pi*pi*a*a*b*b*R**4))


def gaussian_certificate():
    m=Q(float.fromhex('0x1.45b8a0bad2714p+1'))
    beta=gamma=Q(float.fromhex('0x1.05f1d178c75ebp+2'))
    a,b,R,k=map(Q,[4,1,4,1])
    lo,hi=pi_enclosure()
    linear=2*m*k*k/(a*a*R*R)+m*(Q(3,4)/a**4+Q(1,2)/(a*a*b*b)+Q(3,4)/b**4)/R**4
    lower=linear-beta*m*m/(4*lo*a*b*R*R)+gamma*m**3/(9*hi*hi*a*a*b*b*R**4)
    upper=linear-beta*m*m/(4*hi*a*b*R*R)+gamma*m**3/(9*lo*lo*a*a*b*b*R**4)
    assert lower <= upper < 0
    return {'classification':'VERIFIED_ANALYTIC_TRIAL_BOUND','mass_hex':float(m).hex(),
            'beta_gamma_hex':float(beta).hex(),'k_star':1,'a':4,'b':1,'R':4,
            'energy_interval_rational':[str(lower),str(upper)],
            'energy_interval_display':[float(lower),float(upper)],
            'pi_interval':[str(lo),str(hi)],'whole_space_minimizer_proved':False,
            'role':'Negative variational upper bound; excludes vanishing, not dichotomy.'}


def independent_probes():
    records=[]
    for K,G,T in [(1.,0.,1.),(.5,.3,.8),(2.,-1.,1.)]:
        a=(2*K+G)/(2*K);b=np.sqrt(4*K*T-G*G)/(2*K)
        c=np.sqrt((np.hypot(a,b)-a)/2)
        for mass2 in [-2.,0.,.1]:
            values=[]
            for kval in [100.,200.,400.]:
                q=kval*kval
                roots=np.roots([K,0.,1+(2*K+G)*q,0.,q+mass2+(K+G+T)*q*q])
                values.append(float(np.max(roots.real)/kval))
            assert abs(values[-1]-c)<1e-4
            records.append({'K':K,'G':G,'T':T,'m2':mass2,'c_uv':c,'growth_over_k':values})
    eig=[]
    for H in [np.array([[2.,0.],[0.,3.]]),np.array([[-7.,4.],[4.,2.]]),np.zeros((2,2))]:
        M=np.block([[H,2*np.eye(2)],[2*np.eye(2),np.zeros((2,2))]])
        e=np.linalg.eigvalsh(M)
        assert np.count_nonzero(e<0)==2 and np.count_nonzero(e>0)==2
        eig.append(e.tolist())
    return {'UV_companion_polynomial_checks':records,'scalar_tensor_inertia_examples':eig}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    checks=symbolic_checks()
    result={'schema':'wct-foundational-recovery-v1','status':'PASS','symbolic_checks':checks,
            'symbolic_check_count':len(checks),'gaussian_certificate':gaussian_certificate(),
            'independent_probes':independent_probes(),
            'versions':{'python':platform.python_version(),'sympy':s.__version__,'numpy':np.__version__},
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'Independent reconstruction in this workflow; not external review or formal proof checking.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(f'PASS: {len(checks)} symbolic identities; 27 UV polynomial probes; exact Gaussian enclosure; 3 inertia probes')


if __name__=='__main__':main()
