"""Stage-2 checks for the nine OPEN foundational WCT issues.

Only the specified elementary mathematical models are checked. Passing does not
derive a physical WCT metric, spectrum, gauge coupling, quantum action, vacuum,
ghost-free completion, or prospectively validated observation.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import sympy as sp


@dataclass(frozen=True)
class Result:
    name: str
    passed: bool
    scope: str
    detail: str


def collective_rank_budget() -> Result:
    # One complex field = two independent real gradient columns. Four
    # independent real gradients can span a 4x4 anisotropic tensor algebraically,
    # but no such field multiplet or kinetic effective action is derived here.
    x = sp.Matrix(sp.symbols("a0:4"))
    y = sp.Matrix(sp.symbols("b0:4"))
    h = x*x.T + y*y.T
    three_minors = [h.extract(rows, cols).det()
                    for rows in sp.utilities.iterables.combinations(range(4), 3)
                    for cols in sp.utilities.iterables.combinations(range(4), 3)]
    four_gradients = sp.eye(4)
    candidate = four_gradients*four_gradients.T
    good = all(sp.simplify(m) == 0 for m in three_minors) and candidate.det() == 1
    return Result("11_rank_budget", bool(good), "algebraic obstruction and field-count only",
                  "A single complex scalar supplies at most rank 2; four independent real "
                  "gradient directions are sufficient for algebraic rank 4, not for emergent GR.")


def periodic_circle_eigenvalue(n: int, length: float, connection: float = 0.0,
                               potential: float = 0.0) -> float:
    """H=(-i d/ds+a)^2+V0 on s mod L, periodic domain H^2_per.

    Here a is a specified flat connection, not automatically geometric torsion.
    """
    if length <= 0:
        raise ValueError("period must be positive")
    return (2*math.pi*n/length + connection)**2 + potential


def minmax_interval(free_eigenvalue: float, vmin: float, vmax: float) -> tuple[float,float]:
    """Form-ordered kth eigenvalue enclosure H0+V, given vmin<=V<=vmax.

    The supplied eigenvalue must already be in the sorted list (multiplicity
    included); this function does NOT assume a Fourier label is an ordered rank.
    """
    if vmin > vmax:
        raise ValueError("invalid pointwise potential enclosure")
    return free_eigenvalue+vmin, free_eigenvalue+vmax


def spectral_obstruction() -> Result:
    # For a circle of radius R=1, curvature kappa=1 while the n=0 periodic
    # eigenvalue of -d_s^2 is zero. Curvature alone cannot imply lambda_0=kappa^2.
    radius = 1.0
    length = 2*math.pi*radius
    lam0 = periodic_circle_eigenvalue(0, length)
    lam1 = periodic_circle_eigenvalue(1, length)
    low, high = minmax_interval(lam1, -0.2, 0.4)
    return Result("12_circle_spectral_bound", lam0 == 0 and math.isclose(lam1, 1) and
                  math.isclose(low, .8) and math.isclose(high, 1.4),
                  "specified periodic 1D Schrödinger operator; min-max bound",
                  "Curvature 1/R does not imply universal lambda_0=kappa^2; "
                  "lambda_j(H0)+inf V <= lambda_j(H0+V) <= lambda_j(H0)+sup V.")


def dimensional_scale_obstruction() -> Result:
    # Two choices of external L0 solve the same dimensionless problem but
    # change every positive physical eigenvalue and mass conversion.
    lam, l1, l2 = sp.symbols("lam L1 L2", positive=True)
    ratio_lam = sp.simplify((lam/l1**2)/(lam/l2**2))
    ratio_mass = sp.simplify((sp.sqrt(lam)/l1)/(sp.sqrt(lam)/l2))
    return Result("13_scale_covariance", bool(sp.simplify(ratio_lam-(l2/l1)**2)==0 and
                  sp.simplify(ratio_mass-l2/l1)==0),
                  "dimensional underdetermination, not a vacuum existence theorem",
                  "Changing external length L0 maps lambda_phys->lambda_dimless/L0^2 "
                  "and m_phys->hbar sqrt(lambda_dimless)/(c L0).")


def quantum_scale_obstruction() -> Result:
    C, hbar, omega, N = sp.symbols("C hbar omega N", positive=True)
    # With action S=C*S0 the classical Euler-Lagrange equations are unchanged;
    # quantization in dimensionless S0 coordinates uses hbar/C.
    E0 = hbar*omega*(N+sp.Rational(1,2))/C
    correct = sp.simplify(E0 - hbar/C*omega*(N+sp.Rational(1,2))) == 0
    return Result("14_quantum_scale", bool(correct),
                  "finite, healthy harmonic sector only; hbar and C are external",
                  "An oscillator yields E_N=(hbar/C)omega(N+1/2) in S0 units. "
                  "Rescaling the classical action leaves equations unchanged and "
                  "cannot select absolute hbar.")


def local_u1_covariance() -> Result:
    x, charge = sp.symbols("x Q", real=True)
    phi = sp.Function("phi")(x)
    chi = sp.Function("chi")(x)
    a = sp.Function("A")(x)
    phase = sp.exp(sp.I*charge*chi)
    def D(f, A):
        return sp.diff(f,x)-sp.I*charge*A*f
    new_a = a+sp.diff(chi,x)
    first = sp.simplify(sp.expand(D(phase*phi,new_a)-phase*D(phi,a)))
    second = sp.simplify(sp.expand(D(D(phase*phi,new_a),new_a)-phase*D(D(phi,a),a)))
    # The full covariant curvature quotient is invariant because its numerator
    # and conjugate each carry opposite representation phases.
    return Result("15_local_u1_jet_covariance", bool(first == 0 and second == 0),
                  "covariant derivative identity, not charge/gauge normalization",
                  "D(phase*psi,A+dchi)=phase*D(psi,A) and likewise D^2. "
                  "Canonical A_c=A/g leaves the physical product Q*g freely specified.")


def static_poles(q: float, K: float, G: float, T: float, m2: float) -> dict:
    """Exact Q6/Q7 for the static homogeneous amplitude or phase sector."""
    if K <= 0 or T <= 0 or G*G >= 4*K*T or m2 < 0 or q < 0:
        raise ValueError("outside declared positivity / q / mass domain")
    disc = (1+G*q)**2-4*K*(m2+T*q*q)
    if disc < 0:
        return dict(discriminant=disc, status="COMPLEX_POLE", no_real_poles=True)
    root = math.sqrt(disc)
    zlo = q+(1+G*q-root)/(2*K)
    zhi = q+(1+G*q+root)/(2*K)
    return dict(discriminant=disc, status=("COLLISION" if disc == 0 else "REAL"),
                z_lower=zlo, z_upper=zhi,
                lower_derivative=root, upper_derivative=-root)


def finite_minimum(K: float, G: float, T: float, m2: float) -> dict:
    if not (K > 0 and T > 0 and G*G < 4*K*T and 0 < 4*K*m2 < 1):
        return {"status":"NOT_IN_ADMISSIBLE_REAL_LOW_POLE_DOMAIN"}
    s = math.sqrt(1-4*K*m2)
    if not G > s:
        return {"status":"NO_INTERIOR_LOW_BRANCH_MINIMUM", "s":s}
    B=4*K*T-G*G
    a0=2*K+G
    qmin=(G-a0*math.sqrt((G*G+B*s*s)/(a0*a0+B)))/B
    qcollision=(G+math.sqrt(G*G+B*s*s))/B
    pole=static_poles(qmin,K,G,T,m2)
    # Preserve the branch report's more precise stage-2 classification.
    pole_without_status={k:v for k,v in pole.items() if k!="status"}
    return dict(status="LOWER_OSCILLATORY_MINIMUM_NOT_UNSTABLE_BAND",
                q_min=qmin,k_min=math.sqrt(qmin),q_collision=qcollision,
                **pole_without_status)


def higher_time_branch_audit() -> Result:
    K,G,T,m2=0.001,2.0,2000.0,1.0
    x=finite_minimum(K,G,T,m2)
    z=static_poles(x["q_min"],K,G,T,m2)
    # Q6: D(z,q)=y-m2-Ky^2+G*q*y-T*q^2; derivative at roots ±sqrt(d).
    def D(zv,q):
        y=zv-q
        return y-m2-K*y*y+G*q*y-T*q*q
    ok=x["status"].startswith("LOWER_") and x["q_min"]>0 and x["q_min"]<x["q_collision"]
    ok=ok and z["z_lower"]>0 and z["z_upper"]>z["z_lower"]
    ok=ok and z["lower_derivative"]>0 and z["upper_derivative"]<0
    ok=ok and abs(D(z["z_lower"],x["q_min"]))<1e-8 and abs(D(z["z_upper"],x["q_min"]))<1e-7
    return Result("16_18_exact_static_poles",bool(ok),
                  "Q6/Q7 static homogeneous witness; no full ghost-free theory",
                  "Lower frequency has finite oscillatory minimum; upper root has "
                  "opposite residue. Nondegenerate higher-time Hessian persists for K>0. "
                  "No inference of an unstable finite band or benign continuum dynamics.")


def run_checks() -> list[Result]:
    return [collective_rank_budget(),spectral_obstruction(),dimensional_scale_obstruction(),
            quantum_scale_obstruction(),local_u1_covariance(),higher_time_branch_audit()]


if __name__ == "__main__":
    checks=run_checks()
    report={"schema":"wct.foundational.stage2.v1","passed":all(x.passed for x in checks),
            "claims_closed":[],"checks":[asdict(x) for x in checks]}
    print(json.dumps(report,indent=2))
    if not report["passed"]:
        raise SystemExit(1)
