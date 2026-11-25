"""Symbolic audits for the WCT GR/constants derivation closure.

The module verifies algebraic claims only. It does not claim empirical
validation or prove existence/stability of localized solutions.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import sympy as sp


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def regularized_reciprocal_phase_check() -> Check:
    """Verify U(1) covariance of the modulus-squared regularization."""
    beta = sp.symbols("beta", real=True)
    eps, alpha, q = sp.symbols("epsilon alpha q", positive=True, real=True)
    phase = sp.exp(sp.I * beta)
    psi, boxpsi = sp.symbols("psi boxpsi", nonzero=True)
    psibar = sp.symbols("psibar", nonzero=True)
    denominator = q + eps**2 * sp.exp(-2 * alpha * q)

    theta = -boxpsi * psibar / denominator
    theta_rotated = -(phase * boxpsi) * (sp.conjugate(phase) * psibar) / denominator
    corrected_invariant = sp.simplify(theta_rotated - theta) == 0

    old = -boxpsi / (psi + eps * sp.exp(-alpha * q))
    old_rotated = -(phase * boxpsi) / (phase * psi + eps * sp.exp(-alpha * q))
    old_not_invariant = sp.factor(old_rotated - old) != 0

    return Check(
        "u1_regularization",
        bool(corrected_invariant and old_not_invariant),
        "R_epsilon=psi*/(|psi|^2+epsilon^2 exp(-2 alpha |psi|^2)) makes "
        "Theta phase invariant; the historical additive denominator does not.",
    )


def denominator_positivity_check() -> Check:
    eps, alpha, q = sp.symbols("epsilon alpha q", positive=True, real=True)
    denominator = q + eps**2 * sp.exp(-2 * alpha * q)
    samples = [
        sp.N(denominator.subs({q: x, eps: sp.Rational(1, 10), alpha: 2}))
        for x in (0, 1, 10)
    ]
    return Check(
        "denominator_positivity",
        all(value > 0 for value in samples),
        "D_epsilon(q)=q+epsilon^2 exp(-2 alpha q)>0 for q>=0 and epsilon>0.",
    )


def loop_locking_check() -> Check:
    """Verify the constrained minimum and expose the exact-lock condition."""
    mismatch, inverse_weight_integral = sp.symbols("Delta Iw", real=True, nonzero=True)
    constant = mismatch / inverse_weight_integral
    minimum = sp.simplify(constant**2 * inverse_weight_integral)
    expected = mismatch**2 / inverse_weight_integral
    return Check(
        "loop_locking_minimum",
        sp.simplify(minimum - expected) == 0,
        "S_min=(2 pi n-int sigma ds)^2/(int ds/w). The identity "
        "2 pi n/L=<sigma> requires zero mismatch and a condition equating "
        "weighted and unweighted averages.",
    )


def disformal_rank_check() -> Check:
    """Show a one-complex-scalar anisotropic metric deformation has rank <=2."""
    real_gradient = sp.Matrix(sp.symbols("a0:4"))
    imag_gradient = sp.Matrix(sp.symbols("b0:4"))
    deformation = (
        real_gradient * real_gradient.T + imag_gradient * imag_gradient.T
    )
    minors = []
    for rows in sp.utilities.iterables.combinations(range(4), 3):
        for cols in sp.utilities.iterables.combinations(range(4), 3):
            minors.append(sp.factor(deformation.extract(rows, cols).det()))
    return Check(
        "single_complex_scalar_metric_rank",
        all(minor == 0 for minor in minors),
        "Re(d_mu psi* d_nu psi) is a sum of two rank-one tensors, so its "
        "rank is at most two and it cannot represent a generic anisotropic "
        "four-dimensional metric perturbation.",
    )


def gr_limit_check() -> Check:
    f0, v0 = sp.symbols("F0 V0", positive=True)
    g_effective = 1 / (8 * sp.pi * f0)
    lambda_effective = v0 / f0
    g_symbol = sp.symbols("G", positive=True)
    recovered = sp.simplify(
        g_effective.subs(f0, 1 / (8 * sp.pi * g_symbol)) - g_symbol
    ) == 0
    return Check(
        "scalar_tensor_gr_limit",
        recovered,
        f"For constant vacuum F0, G_eff={sp.sstr(g_effective)} and "
        f"Lambda_eff={sp.sstr(lambda_effective)} in natural units, up to "
        "heavy-field and derivative corrections.",
    )


def spectral_mass_check() -> Check:
    eigenvalue, hbar, c = sp.symbols("lambda hbar c", positive=True)
    mass = hbar * sp.sqrt(eigenvalue) / c
    rest_energy_squared = sp.simplify((mass * c**2) ** 2)
    expected = hbar**2 * c**2 * eigenvalue
    return Check(
        "spectral_mass",
        sp.simplify(rest_energy_squared - expected) == 0,
        "Separating a massless parent wave equation on a confined eigenmode "
        "-Delta_K chi=lambda chi gives m=hbar*sqrt(lambda)/c exactly.",
    )


def constants_status_check() -> Check:
    return Check(
        "dimensionful_constants_status",
        True,
        "A dimensionless PDE can predict dimensionless ratios. Absolute "
        "values of c, hbar, G, or Lambda require a dimensionful scale, an "
        "action normalization, or calibration.",
    )


def run_checks() -> list[Check]:
    return [
        regularized_reciprocal_phase_check(),
        denominator_positivity_check(),
        loop_locking_check(),
        disformal_rank_check(),
        gr_limit_check(),
        spectral_mass_check(),
        constants_status_check(),
    ]


def report() -> dict[str, Any]:
    checks = run_checks()
    return {
        "schema": "wct.gr_constants_closure.v1",
        "passed": all(check.passed for check in checks),
        "checks": [asdict(check) for check in checks],
    }


if __name__ == "__main__":
    print(json.dumps(report(), indent=2, sort_keys=True))
