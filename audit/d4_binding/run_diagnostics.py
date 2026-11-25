#!/usr/bin/env python3
"""Adversarial trials, never certified comparisons between unknown infima."""
import argparse
from fractions import Fraction as F
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'derivations'))
from d4_binding import BETA,GAMMA,MASS,RHO,Interval,adjust,energy,endpoint_bound,gaussian_moments


def run():
    beta=float(BETA);gamma=float(GAMMA);m=float(MASS);mb=3/math.sqrt(float(RHO))
    rng=np.random.default_rng(20261023);probes=[]
    for mass in [.01,.5,1,m,mb,4]:
        for _ in range(40):
            a,b=np.exp(rng.uniform(-2,3,2));q=float(rng.uniform(0,2));cx,cy=rng.uniform(-1,1,2)
            original=gaussian_moments(mass,float(a),float(b),q,float(cx),float(cy));z,w=adjust(original)
            residual=w['D']-w['K']-beta*w['Q']/4+gamma*w['S']/3
            assert abs(residual)<1e-9*max(1,w['D'])
            assert energy(w)<=energy(original)+1e-9*max(1,abs(energy(original)))
            assert w['S']/w['Q']<=math.sqrt(float(RHO))*mass/4+1e-10
            slope=-beta*w['Q']/2+2*gamma*w['S']/3
            if mass<=mb:assert slope<0
            probes.append({'mass':mass,'a':float(a),'b':float(b),'q':q,'cx':float(cx),'cy':float(cy),
               'dilation_squared':z,'original_energy':energy(original),'adjusted_energy':energy(w),
               'virial_residual':residual,'ratio':w['S']/w['Q'],'ratio_upper':math.sqrt(float(RHO))*mass/4,
               'adjusted_amplitude_slope':slope,'in_sufficient_range':mass<=mb})
    # Inherited exact Gaussian-pair evaluator; independently checked by coordinate quadrature in new tests.
    spec=importlib.util.spec_from_file_location('old_diag',ROOT/'audit/strict_binding/run_diagnostics.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    pairs=[]
    for f in [.5,.1,1e-6]:
        for d in [0,4,8,16,64]:
            for phase in [0,math.pi/2,math.pi,3*math.pi/2]:
                if f==.5 and d==0 and phase==math.pi:continue
                value=old.two_gaussians(m*f,m*(1-f),4,d,phase)
                pairs.append({'fraction':f,'separation':d,'relative_phase':phase,**value,'certifies_infimum_comparison':False})
    assert len(pairs)==59
    assert max(abs(r['interaction_difference']) for r in pairs if r['separation']==64)<2e-11
    ms=4*BETA/(GAMMA*(1+3*BETA**2/(16*GAMMA)))
    mus=[MASS/F(10**j) for j in range(1,19)]+[2*MASS-2*ms,MASS/2]
    endpoints=[{'mu_exact':str(mu),'mu_fraction':float(mu/MASS),'F':endpoint_bound(mu).record()} for mu in mus]
    assert all(F(r['F']['exact'][0])>0 for r in endpoints)
    quadratures=[]
    for eta in [.1,1,10]:
        v,err=quad(lambda s:1/(s*s+eta),0,np.inf,epsabs=1e-11)
        expected=1/(8*math.sqrt(eta));observed=v/(4*math.pi)
        assert abs(observed-expected)<1e-11
        quadratures.append({'eta':eta,'observed':observed,'expected':expected,'quad_error':err})
    C=3*beta**2/(16*gamma)
    def old_weight_bound(logeta):
        eta=math.exp(logeta);j=(math.pi/2+math.atan(1/math.sqrt(eta)))/(4*math.pi*math.sqrt(eta))
        return 3*beta/(4*gamma*j*(C+eta))
    opt=minimize_scalar(lambda x:-old_weight_bound(x),bounds=(-12,12),method='bounded')
    narrow=gaussian_moments(m,.5,.5)
    badslope=-beta*narrow['Q']/2+2*gamma*narrow['S']/3
    assert badslope>0
    return {'schema':'d4-binding-diagnostics-v1','status':'PASS','seed':20261023,
      'interpretation':'Analytic trial moments plus ordinary floating diagnostics; not computed continuum infima',
      'gaussian_dilation_probes':probes,'two_profile_trials':pairs,'endpoint_certified_scalar_probes':endpoints,
      'fourier_quadratures':quadratures,
      'old_weight_exploration':{'eta':math.exp(float(opt.x)),'sufficient_mass':old_weight_bound(float(opt.x)),'certified_global_optimum':False},
      'frozen_unadjusted_shortcut_counterexample':{'energy':energy(narrow),'amplitude_slope':badslope,'negative_energy_example':energy(narrow)<0},
      'whole_space_counterexample_found':False,'stationary_PDE_search':False,'PDE_propagation':False,
      'pair_evaluator':'Inherited evaluator, independently cross-checked by coordinate quadrature in test_d4_binding.py'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(run(),indent=2,allow_nan=False)+'\n')
    print('PASS: 240 complex dilation probes; 59 phased pairs; 20 exact endpoint probes; 3 Fourier quadratures')
