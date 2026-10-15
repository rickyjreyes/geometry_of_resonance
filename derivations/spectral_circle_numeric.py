"""Specified periodic-circle Galerkin test, not physical WCT spectrum.

H=(-i*d/ds+a)^2+v0+v1*cos(2*pi*s/L), periodic H^2 domain.
Fourier basis n=-N,...,N. Finite matrix enclosures here are *numerical
illustrations* of the exact continuum min-max FORM bound in the accompanying
paper, not certified PDE eigenvalue enclosures.
"""
import argparse
import json
import math

import numpy as np


def gallery(L=2*math.pi,a=.17,v0=.4,v1=.2,N=12):
    if not L>0 or N<2:
        raise ValueError("invalid test domain/cutoff")
    n=np.arange(-N,N+1,dtype=float)
    free=(2*math.pi*n/L+a)**2
    H=np.diag(free+v0)
    for i in range(len(n)-1):
        H[i,i+1]=H[i+1,i]=v1/2
    measured=np.linalg.eigvalsh(H)
    f=np.sort(free)
    lo=f+v0-abs(v1)
    hi=f+v0+abs(v1)
    if not (np.all(measured>=lo-1e-10) and np.all(measured<=hi+1e-10)):
        raise AssertionError("Galerkin violates form-order enclosure")
    return {"model":"specified circle + imposed scalar potential","L":L,"connection":a,
            "v0":v0,"v1":v1,"cutoff":N,"no_physical_mass_conversion":True,
            "eigenvalues_first_six":measured[:6].tolist(),
            "lower_first_six":lo[:6].tolist(),"upper_first_six":hi[:6].tolist(),
            "finite_matrix_minmax_pass":True,
            "warning":"Extrinsic curvature and torsion do not enter without an independently "
                      "chosen operator/potential/connection. Finite truncation is not a certified "
                      "continuum numerical error bound."}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--length",type=float,default=2*math.pi)
    args=p.parse_args()
    print(json.dumps(gallery(L=args.length),indent=2))
