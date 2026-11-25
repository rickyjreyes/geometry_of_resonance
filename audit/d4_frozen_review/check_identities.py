#!/usr/bin/env python3
"""Independent symbolic reconstruction, importing no prior derivation module."""

import argparse
import json
from pathlib import Path

import sympy as s


def identities():
    D, K, M, Q, S, beta, gamma, k, z, t, r, eta = s.symbols(
        "D K M Q S beta gamma k z t r eta", positive=True
    )
    T = D - 2*k**2*K + k**4*M
    H = T - beta*Q/2 + gamma*S/3
    P = D + gamma*S/3
    B = 2*k**2*K + beta*Q/2
    V = D - k**2*K - beta*Q/4 + gamma*S/3
    rho = k**4 + 3*beta**2/(32*gamma)
    h = s.symbols("h")
    dilation = z**2*D - 2*k**2*z*K + k**4*M - beta*z*Q/2 + gamma*z**2*S/3
    checks = {
        "coercivity_square": -beta*r**2/2+gamma*r**3/3
            - (gamma*r*(r-3*beta/(4*gamma))**2/3-3*beta**2*r/(16*gamma)),
        "dilation_minimum": dilation-(k**4*M-B**2/(4*P))-P*(z-B/(2*P))**2,
        "virial": s.diff(dilation, z).subs(z, 1)-2*V,
        "virial_square": D+T+2*gamma*(S-2*(3*beta/(8*gamma))*Q
            +(3*beta/(8*gamma))**2*M)/3-rho*M-2*V,
        "amplitude_defect": T-beta*t*Q/2+gamma*t**2*S/3-H
            -(t-1)*(-beta*Q/2+gamma*(t+1)*S/3),
        "uniform_sequence_defect": H-(t-1)*beta*h*(-2*H/beta)/2
            -(1+(t-1)*h)*H,
        "fourier_integral": s.integrate(1/(r**2+eta), (r, 0, s.oo))/(4*s.pi)
            -1/(8*s.sqrt(eta)),
        "optimized_fourier_constant": ((D+eta*M)/(8*s.sqrt(eta))).subs(eta, D/M)
            -s.sqrt(M*D)/4,
    }
    a, b = s.symbols("a b", positive=True)
    mB = 3*beta/(gamma*s.sqrt(rho))
    checks["two_mass_bracket"] = -beta/2+gamma*(b/a+1)*s.sqrt(rho)*a/12 \
        +beta*(1-(a+b)/(2*mB))/2
    # Derive the optimized Gaussian polynomial, then the rational-gap reduction.
    pi = s.symbols("pi", positive=True)
    c2 = b/50-beta*b*b/(40*pi)
    c4 = s.Rational(30203, 40000)*b+gamma*b**3/(900*pi*pi)
    d = beta*b/(40*pi)-s.Rational(1, 50)
    checks["trial_reduction"] = c2*c2/(4*c4)-b*d*d/(4*(s.Rational(30203, 40000)+gamma*b*b/(900*pi*pi)))
    report = {name: str(s.simplify(value)) for name, value in checks.items()}
    if any(value != "0" for value in report.values()):
        raise ValueError(report)
    return {"status": "PASS", "identities": report,
            "scope": "scalar identities only; analytic arguments are in PROOF.md"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = identities()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"PASS: {len(result['identities'])} independently reconstructed symbolic identities")
