#!/usr/bin/env python3
"""Phase IV identities and exact constants; no import of the Phase III proof.

These checks corroborate the analytic arguments in docs/strict_binding.
They do not numerically certify a constrained infimum or a stationary state.
"""
import argparse
from fractions import Fraction as F
import json
import math
from pathlib import Path

import sympy as sp

BETA_HEX = "0x1.05f1d178c75ebp+2"
MASS_HEX = "0x1.45b8a0bad2714p+1"
BETA = F(float.fromhex(BETA_HEX))
GAMMA = BETA
D4_MASS = F(float.fromhex(MASS_HEX))


def constants(beta=BETA, gamma=GAMMA, k=F(1)):
    beta, gamma, k = map(F, (beta, gamma, k))
    if min(beta, gamma, k) <= 0:
        raise ValueError("The restricted theorem requires beta,gamma,k > 0")
    c = 3 * beta**2 / (16 * gamma)
    j = 3 / (16 * k**2)
    kap = j * (k**4 + c)
    mass_star = 3 * beta / (4 * gamma * kap)
    return {"C": c, "J": j, "K_infinity": kap, "m_star": mass_star}


def split_is_certified(mass, part, beta=BETA, gamma=GAMMA, k=F(1)):
    mass, part = F(mass), F(part)
    if not 0 < part < mass:
        raise ValueError("A split requires 0 < part < mass")
    return mass + max(part, mass - part) <= 2 * constants(beta, gamma, k)["m_star"]


def gaussian_moments(mass, a, b, q=1.0, k=1.0):
    """Whole-space moments of sqrt(m/(pi*a*b))*Gaussian*exp(i*q*x)."""
    if min(mass, a, b) <= 0:
        raise ValueError("positive mass and widths required")
    grad = mass * (q*q + 1/(2*a*a) + 1/(2*b*b))
    bih = mass * (q**4 + q*q*(3/(a*a)+1/(b*b))
                  + 3/(4*a**4)+1/(2*a*a*b*b)+3/(4*b**4))
    # Avoid cancellation of the k^4 mass term for broad carrier packets.
    t = mass * ((q*q-k*k)**2 + (3*q*q-k*k)/(a*a)
                + (q*q-k*k)/(b*b) + 3/(4*a**4)
                + 1/(2*a*a*b*b)+3/(4*b**4))
    return {"M": mass, "K": grad, "D": bih, "T": t,
            "Q": mass**2/(2*math.pi*a*b),
            "S": mass**3/(3*math.pi**2*a*a*b*b)}


def energy(moments, beta=float(BETA), gamma=float(GAMMA)):
    return moments["T"] - beta*moments["Q"]/2 + gamma*moments["S"]/3


def negative_trial(mass, beta=float(BETA), gamma=float(GAMMA), k=1.0):
    """Explicit analytic negative trial, not an optimization certificate."""
    if min(mass, beta, gamma, k) <= 0:
        raise ValueError("positive parameters required")
    a = 1 + 16*math.pi*k*k/(beta*mass)
    b = 1.0
    c2 = 2*mass*k*k/a**2 - beta*mass**2/(4*math.pi*a*b)
    c4 = mass*(3/(4*a**4)+1/(2*a*a*b*b)+3/(4*b**4))
    c4 += gamma*mass**3/(9*math.pi**2*a*a*b*b)
    r2 = -2*c4/c2
    return {"a": a, "b": b, "R_squared": r2,
            "coefficient_R_minus_2": c2, "coefficient_R_minus_4": c4,
            "energy_upper_bound": -c2*c2/(4*c4)}


def symbolic_checks():
    t, D, K, M, Q, S, beta, gamma, k, r = sp.symbols(
        "t D K M Q S beta gamma k r", positive=True)
    H = D-2*k*k*K+k**4*M-beta*Q/2+gamma*S/3
    T = sp.symbols("T", nonnegative=True)
    checks = {}

    def check(name, lhs, rhs):
        residual = sp.simplify(lhs-rhs)
        checks[name] = {"passed": residual == 0, "residual": str(residual)}

    check("density_square_completion", -beta*r*r/2+gamma*r**3/3,
          gamma*r*(r-3*beta/(4*gamma))**2/3-3*beta**2*r/(16*gamma))
    amp = t*T-beta*t*t*Q/2+gamma*t**3*S/3
    check("amplitude_energy_per_mass_difference", amp/t-(T-beta*Q/2+gamma*S/3),
          (t-1)*(-beta*Q/2+gamma*(t+1)*S/3))
    check("amplitude_derivative", sp.diff(amp/t,t).subs(t,1), -beta*Q/2+2*gamma*S/3)
    dil = t**4*D-2*k*k*t*t*K+k**4*M-beta*t*t*Q/2+gamma*t**4*S/3
    check("mass_preserving_virial", sp.diff(dil,t).subs(t,1)/2,
          2*D-2*k*k*K-beta*Q/2+2*gamma*S/3)
    density_dil = D/t-2*k*k*K+t*k**4*M-beta*t*Q/2+gamma*t*S/3
    check("density_preserving_dilation_difference", density_dil/t-H,
          (1-1/t)*(-(1+1/t)*D+2*k*k*K))
    check("density_dilation_derivative", sp.diff(density_dil/t,t).subs(t,1),
          2*(k*k*K-D))
    c = 3*beta**2/(16*gamma)
    j = 3/(16*k*k)
    cap = j*(k**4+c)
    ms = 4*beta*k*k/(gamma*(k**4+c))
    check("threshold_definition", cap*ms, 3*beta/(4*gamma))
    check("threshold_two_mass_boundary", gamma*cap*(2*ms)/3, beta/2)
    check("resolvent_constant", (sp.pi/(k*k))*(sp.pi/2+sp.pi/4)/(2*sp.pi)**2, j)
    s = sp.symbols("s", real=True)
    check("resolvent_antiderivative", sp.diff(sp.atan((s-k*k)/(k*k))/(k*k),s),
          1/((s-k*k)**2+k**4))
    q, a, b = sp.symbols("q a b", positive=True)
    d_g = M*(q**4+q*q*(3/a**2+1/b**2)+3/(4*a**4)+1/(2*a*a*b*b)+3/(4*b**4))
    k_g = M*(q*q+1/(2*a*a)+1/(2*b*b))
    check("gaussian_carrier_kinetic", (d_g-2*k*k*k_g+k**4*M).subs(q,k),
          M*(2*k*k/a**2+3/(4*a**4)+1/(2*a*a*b*b)+3/(4*b**4)))
    c2 = 2*M*k*k/a**2-beta*M*M/(4*sp.pi*a*b)
    c4 = M*(3/(4*a**4)+1/(2*a*a*b*b)+3/(4*b**4))+gamma*M**3/(9*sp.pi**2*a*a*b*b)
    check("negative_trial_optimal_scale", (c2/t+c4/t**2).subs(t,-2*c4/c2), -c2**2/(4*c4))
    A, B = sp.symbols("A B", nonnegative=True)
    potential = lambda z: -beta*z*z/2+gamma*z**3/3
    check("quadrature_phase_overlap", potential(A+B)-potential(A)-potential(B),
          A*B*(-beta+gamma*(A+B)))
    check("chemical_potential_identity", T-beta*Q+gamma*S,
          (T-beta*Q/2+gamma*S/3)-beta*Q/2+2*gamma*S/3)
    return checks


def report():
    c = constants()
    ms = c["m_star"]
    low, high = 2*D4_MASS-2*ms, 2*ms-D4_MASS
    checks = symbolic_checks()
    assert all(x["passed"] for x in checks.values())
    # Exact rational comparisons; decimal displays are not theorem definitions.
    assert F("2.2631912293727") < ms < F("2.2631912293729")
    assert ms < D4_MASS < 4*ms/3
    assert 0 < low < D4_MASS/2 < high < D4_MASS
    return {"schema": "wct-strict-binding-identities-v1", "status": "PASS",
            "primary_classification": "STRICT_BINDING_RESTRICTED",
            "parameters": {"beta_hex": BETA_HEX, "gamma_hex": BETA_HEX,
                           "k": "1", "D4_mass_hex": MASS_HEX},
            "exact_constants": {key: str(v) for key,v in c.items()},
            "decimal_displays": {key: float(v) for key,v in c.items()},
            "D4_mass_exact": str(D4_MASS), "D4_mass_decimal": float(D4_MASS),
            "D4_certified_split_interval_exact": [str(low),str(high)],
            "D4_certified_split_interval_decimal": [float(low),float(high)],
            "D4_all_splits_proved": False, "all_positive_masses_proved": False,
            "symbolic_checks": checks,
            "proof_location": "docs/strict_binding/SUBADDITIVITY_PROOF_OR_COUNTEREXAMPLE.md"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(f"PASS: {len(result['symbolic_checks'])} identities; exact restricted threshold and split interval")


if __name__ == "__main__":
    main()
