#!/usr/bin/env python3
"""Adversarial finite-dimensional trials, explicitly not bounds on both infima."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import quad
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "derivations"))
from strict_binding import BETA, D4_MASS, constants, energy, gaussian_moments, negative_trial


def gaussian_upper_bound(mass):
    """Three deterministic starts; returned value remains only a trial upper bound."""
    trial = negative_trial(mass)
    aa = trial["a"]*math.sqrt(trial["R_squared"])
    bb = math.sqrt(trial["R_squared"])

    def moments(logs):
        a,b = np.exp(logs)
        q2 = max(0.0, 1-(3/a**2+1/b**2)/2)
        return gaussian_moments(mass,a,b,math.sqrt(q2)),a,b,q2

    def objective(logs):
        return energy(moments(logs)[0])/mass

    starts = [np.log([aa,bb]), np.log([max(2,math.sqrt(mass)),max(2,math.sqrt(mass))]),
              np.log([max(8,2*math.sqrt(mass)),max(1,math.sqrt(mass)/2)])]
    attempts = []
    for start in starts:
        result = minimize(objective,start,method="Nelder-Mead",
                          bounds=[(-5,25),(-5,25)],
                          options={"maxiter":1200,"xatol":1e-10,"fatol":1e-13})
        mm,a,b,q2 = moments(result.x)
        attempts.append({"energy":energy(mm),"a":float(a),"b":float(b),
                         "q_squared":float(q2),"optimizer_success":bool(result.success),
                         "evaluations":int(result.nfev),"message":str(result.message)})
    best = min(attempts,key=lambda r:r["energy"])
    return {"mass":mass,"analytic_negative_trial":trial,
            "best_gaussian_upper_bound":best,"attempts":attempts}


def two_gaussians(m1,m2,width,separation,phase,k=1.0):
    """Exact Gaussian moments up to floating arithmetic; kinetic via degree-4 GH.

    Both packets have carrier exp(i*k*x). Relative phase is not discarded.
    The sum is normalized to precisely the original total mass convention.
    """
    w,d,theta = width,separation,phase
    overlap = math.sqrt(m1*m2)*math.exp(-d*d/(4*w*w))
    raw_mass = m1+m2+2*math.cos(theta)*overlap
    if raw_mass <= 1e-14*(m1+m2):
        raise ValueError("destructive cancellation leaves no normalizable field")
    nodes,weights = hermgauss(6)
    x,y = np.meshgrid(w*nodes,w*nodes,indexing="ij")
    z1 = ((x+d/2)**2+y*y)/w**4-2/w**2-2j*k*(x+d/2)/w**2
    z2 = ((x-d/2)**2+y*y)/w**4-2/w**2-2j*k*(x-d/2)/w**2
    inner = overlap*np.sum(weights[:,None]*weights[None,:]*z1.conjugate()*z2)/math.pi
    raw_t = (m1+m2)*(2/w**4+2*k*k/w**2)+2*(np.exp(1j*theta)*inner).real
    n1,n2 = math.sqrt(m1/(math.pi*w*w)),math.sqrt(m2/(math.pi*w*w))

    def density_power(p):
        value = 0.0
        for i in range(p+1):
            for j in range(p-i+1):
                ell = p-i-j
                n,h = 2*i+ell,2*j+ell
                weight = math.factorial(p)/(math.factorial(i)*math.factorial(j)*math.factorial(ell))
                weight *= (2*math.cos(theta))**ell
                integral = n1**n*n2**h*(2*math.pi*w*w/(n+h))
                integral *= math.exp(-n*h*d*d/(2*w*w*(n+h)))
                value += weight*integral
        return value

    factor = (m1+m2)/raw_mass
    result = {"M":m1+m2,"T":float(factor*raw_t),
              "Q":factor**2*density_power(2),"S":factor**3*density_power(3)}
    h = energy(result)
    separate = energy(gaussian_moments(m1,w,w,k,k))+energy(gaussian_moments(m2,w,w,k,k))
    return {"moments":result,"raw_mass":raw_mass,"normalization_squared":factor,
            "energy":h,"sum_of_single_trial_energies":separate,
            "interaction_difference":h-separate}


def run():
    beta = float(BETA)
    mstar = float(constants()["m_star"])
    target = float(D4_MASS)
    # Independent numerical radial integral for the Fourier estimate.
    integral,error = quad(lambda s:1/((s-1)**2+1),0,np.inf,epsabs=2e-12)
    embedding = integral/(4*math.pi)
    assert abs(embedding-3/16)<1e-12
    masses = [.01,.05,.1,.25,.5,1.,2.,mstar,target,4.,8.,16.,64.,256.]
    upper = [gaussian_upper_bound(m) for m in masses]
    assert all(row["analytic_negative_trial"]["energy_upper_bound"]<0 for row in upper)
    splits = []
    total_trial = gaussian_upper_bound(target)["best_gaussian_upper_bound"]["energy"]
    for fraction in [.01,.05,.1,.2,.25,.4,.5]:
        a = gaussian_upper_bound(fraction*target)["best_gaussian_upper_bound"]["energy"]
        b = gaussian_upper_bound((1-fraction)*target)["best_gaussian_upper_bound"]["energy"]
        splits.append({"fraction":fraction,"sum_of_trial_upper_bounds":a+b,
                       "total_trial_upper_bound":total_trial,"trial_gap":a+b-total_trial,
                       "certifies_infimum_comparison":False})
    overlaps = []
    for d in [0.,2.,4.,8.,16.,32.,64.]:
        for phase in [0.,math.pi/2,math.pi,3*math.pi/2]:
            if d==0 and abs(phase-math.pi)<1e-14:
                continue
            value = two_gaussians(target/2,target/2,4.,d,phase)
            overlaps.append({"separation":d,"relative_phase":phase,**value})
    far = [r for r in overlaps if r["separation"]==64.]
    assert max(abs(r["interaction_difference"]) for r in far)<1e-12
    # Explicit negative-energy profile where amplitude scaling has the wrong sign.
    w,rho = 10.,1.5
    mm = gaussian_moments(rho*math.pi*w*w,w,w)
    amplitude_slope = -beta*mm["Q"]/2+2*beta*mm["S"]/3
    assert energy(mm)<0 and amplitude_slope>0
    # Negative-energy counterexample to universal density-preserving dilation.
    dm = gaussian_moments(.75*math.pi*w*w,w,w,math.sqrt(.75))
    dilation_slope = 2*(dm["K"]-dm["D"])
    assert energy(dm)<0 and dilation_slope>0
    return {"schema":"wct-binding-adversarial-diagnostics-v1","status":"PASS",
            "interpretation":"Finite-dimensional diagnostics; no numerical infimum, stationary root, or binding certificate",
            "resolvent_quadrature":{"value":embedding,"target":3/16,"reported_integral_error":error},
            "gaussian_upper_bounds":upper,"D4_trial_split_comparisons":splits,
            "two_profile_interactions":overlaps,
            "amplitude_scaling_counterexample":{"mass":mm["M"],"energy":energy(mm),
               "derivative_of_H_sqrt_theta_over_theta":amplitude_slope,"moments":mm},
            "density_dilation_counterexample":{"mass":dm["M"],"energy":energy(dm),
               "derivative_of_H_u_x_over_sqrt_theta_over_theta":dilation_slope,"moments":dm},
            "full_functional_counterexample_found":False,
            "negative_unattained_infimum_example_for_frozen_coefficients":"UNRESOLVED"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print("PASS: resolvent quadrature; 14 mass trials x 3 starts; 7 split diagnostics; 27 two-profile interactions; 2 scaling counterexamples")


if __name__ == "__main__":
    main()
