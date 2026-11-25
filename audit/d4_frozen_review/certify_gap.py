#!/usr/bin/env python3
"""Exact scalar certificate for the frozen whole-space split deficit.

Standard library only. No inherited interval, symbolic, or diagnostic code is
imported. The analytic reduction to this cubic is proved in PROOF.md; this
program does not purport to formalize Sobolev or concentration compactness.
"""

import argparse
from fractions import Fraction as F
import json
from math import comb
from pathlib import Path


BETA_HEX = "0x1.05f1d178c75ebp+2"
MASS_HEX = "0x1.45b8a0bad2714p+1"
PI_LO = F(333, 106)
PI_HI = F(355, 113)
SIGMA_HI = F(1176311146708614, 10**15)
GAP = F(1, 400000)


def require(condition, message):
    """Fail even with python -O; no security or proof gate uses assert."""
    if not condition:
        raise ValueError(message)


def exact_hex(value):
    """Parse the pinned hexadecimal dyadic without binary floating point."""
    mantissa, exponent = value.lower().split("p")
    require(mantissa.startswith("0x"), "expected a positive hexadecimal dyadic")
    whole, fractional = mantissa[2:].split(".")
    integer = int(whole + fractional, 16)
    return F(integer, 16**len(fractional)) * F(2) ** int(exponent)


BETA = exact_hex(BETA_HEX)
MASS = exact_hex(MASS_HEX)
RHO = 1 + 3 * BETA / 32


def add(p, q):
    return [
        (p[i] if i < len(p) else F(0)) + (q[i] if i < len(q) else F(0))
        for i in range(max(len(p), len(q)))
    ]


def multiply(p, q):
    out = [F(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            out[i + j] += a * b
    return out


def evaluate(p, x):
    value = F(0)
    for coefficient in reversed(p):
        value = value * x + coefficient
    return value


def bernstein_coefficients(p, left, right):
    """Coefficients in B_{i,n}(t), x=left+(right-left)t, t in [0,1]."""
    require(left < right, "positive-length interval required")
    n = len(p) - 1
    power = [
        sum(
            (p[i] * comb(i, j) * left ** (i - j) * (right - left) ** j
             for i in range(j, n + 1)), F(0)
        )
        for j in range(n + 1)
    ]
    return [
        sum((power[j] * F(comb(i, j), comb(n, j)) for j in range(i + 1)), F(0))
        for i in range(n + 1)
    ]


def atan_bounds(q, terms=12):
    require(q > 1 and terms > 0 and terms % 2 == 0, "even positive series length required")
    x = F(1, q)
    lo = sum(((-1)**j * x ** (2*j+1) / (2*j+1) for j in range(terms)), F(0))
    return lo, lo + x ** (2*terms+1) / (2*terms+1)


def machin_bounds():
    a_lo, a_hi = atan_bounds(5)
    b_lo, b_hi = atan_bounds(239)
    return 16*a_lo - 4*b_hi, 16*a_hi - 4*b_lo


def gap_polynomial(gap=GAP):
    """P(b) from PROOF.md section 6, in ascending powers of b."""
    h = [1 - MASS*SIGMA_HI/6, -SIGMA_HI/6]
    d = [-F(1, 50), BETA/(40*PI_HI)]
    numerator = [MASS*v for v in multiply(h, multiply(d, d))]
    return add(numerator, [F(0), -4*gap*F(30203, 40000), F(0),
                           -4*gap*BETA/(900*PI_LO**2)])


def positive_gap_coefficients(gap=GAP):
    require(gap > 0, "positive gap constant required")
    coefficients = bernstein_coefficients(gap_polynomial(gap), MASS/2, MASS)
    require(all(x > 0 for x in coefficients), "whole-interval positivity not certified")
    return coefficients


def certificate():
    pi_lo, pi_hi = machin_bounds()
    require(PI_LO < pi_lo < pi_hi < PI_HI, "Machin enclosure does not prove the pi bounds")
    require(SIGMA_HI > 0 and SIGMA_HI**2 > RHO, "square-root upper bound invalid")
    h_min = 1 - MASS*SIGMA_HI/3
    d_min = BETA*MASS/(80*PI_HI) - F(1, 50)
    require(h_min > 0 and d_min > 0, "positive comparison factors required")
    require(9 - RHO*MASS**2 > 0, "frozen mass is outside the sufficient range")
    coefficients = positive_gap_coefficients()
    return {
        "schema": "d4-frozen-uniform-gap-v1",
        "status": "PASS",
        "claim": "I(mu)+I(m-mu)-I(m) > min(mu,m-mu)/400000 for 0<mu<m",
        "arithmetic": "exact rational; no floating-point sign decisions",
        "parameters": {"beta_hex": BETA_HEX, "gamma_hex": BETA_HEX,
                       "mass_hex": MASS_HEX, "k": "1", "beta": str(BETA), "mass": str(MASS)},
        "rho": str(RHO),
        "frozen_threshold_gate": str(9 - RHO*MASS**2),
        "pi_coarse_enclosure": [str(PI_LO), str(PI_HI)],
        "pi_machin_enclosure": [str(pi_lo), str(pi_hi)],
        "sigma_upper": str(SIGMA_HI),
        "sigma_square_margin": str(SIGMA_HI**2 - RHO),
        "h_min": str(h_min),
        "d_min": str(d_min),
        "gap_constant": str(GAP),
        "component_mass_interval": [str(MASS/2), str(MASS)],
        "monomial_coefficients": list(map(str, gap_polynomial())),
        "bernstein_coefficients": list(map(str, coefficients)),
        "bernstein_minimum": str(min(coefficients)),
        "scope": {"continuum_scalar_interval_certified": True,
                  "analytic_reduction_in": "docs/d4_frozen_review/PROOF.md",
                  "full_analytic_proof_formalized": False,
                  "numerical_infima_used": False,
                  "universal_all_mass_binding_claimed": False},
    }


def verify_record(record):
    require(record == certificate(), "certificate differs from independent exact reconstruction")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--output", type=Path)
    choice.add_argument("--check", type=Path)
    args = parser.parse_args()
    if args.check:
        verify_record(json.loads(args.check.read_text()))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(certificate(), indent=2) + "\n")
    print("PASS: exact frozen mass, pi and square-root gates; four positive Bernstein coefficients")
    print("Certified reduction: split deficit > min(mu,m-mu)/400000, for every interior split")


if __name__ == "__main__":
    main()
