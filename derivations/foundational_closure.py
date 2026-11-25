"""Reproducible S004 checks; analytic claims and scope are in docs/foundational.

No PDE propagation, fit, particle spectrum or new action is implemented here.
The continuum obstruction is proved in the accompanying note, not inferred
from this finite sample. Run with --output to retain a fresh receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as s


def symbolic_checks():
    checks = []

    def zero(name, expression):
        value = s.simplify(s.expand(expression))
        if value != 0:
            raise AssertionError((name, value))
        checks.append(name)

    # Differentiate the original real-field density, not its quoted EL equation.
    t, z, u = s.symbols("t z u", real=True)
    x, y = s.Function("x")(t, z), s.Function("y")(t, z)
    f, pot = s.Function("F"), s.Function("V")
    kap, theta, gam = s.symbols("kappa theta gamma", real=True)
    rho = x*x + y*y
    fields = (x, y)
    box = lambda q: -s.diff(q, t, 2) + s.diff(q, z, 2)
    lap = lambda q: s.diff(q, z, 2)
    curvature = sum(kap*box(q)**2 + theta*lap(q)**2
                    + gam*box(q)*lap(q) for q in fields)
    original = sum(-s.diff(q, t)**2+s.diff(q, z)**2 for q in fields)
    original += -pot(rho)+f(rho)*curvature
    for index, q in enumerate(fields):
        direct = (s.diff(original, q)
                  - s.diff(s.diff(original, s.diff(q, t)), t)
                  - s.diff(s.diff(original, s.diff(q, z)), z)
                  + s.diff(s.diff(original, s.diff(q, t, 2)), t, 2)
                  + s.diff(s.diff(original, s.diff(q, z, 2)), z, 2))
        composed = (-box(q)-s.diff(pot(u), u).subs(u, rho)*q
                    + s.diff(f(u), u).subs(u, rho)*q*curvature
                    + box(f(rho)*(kap*box(q)+gam*lap(q)/2))
                    + lap(f(rho)*(theta*lap(q)+gam*box(q)/2)))
        zero(f"direct_real_EL_component_{index}", direct-2*composed)

    # Independent real Cartesian acceleration/velocity jets and Legendre map.
    a = s.Matrix(s.symbols("a0 a1", real=True))
    b = s.Matrix(s.symbols("b0 b1", real=True))
    v = s.Matrix(s.symbols("v0 v1", real=True))
    p = s.Matrix(s.symbols("p0 p1", real=True))
    r = s.Matrix(s.symbols("r0 r1", real=True))
    F, grad, V = s.symbols("F grad V", real=True)
    A, C = 2*kap+gam, kap+theta+gam
    L = v.dot(v)-grad+V-F*(kap*a.dot(a)-A*a.dot(b)+C*b.dot(b))
    momenta = s.Matrix([s.diff(L, ai) for ai in a])
    acceleration = A*b/(2*kap)-r/(2*kap*F)
    substitutions = dict(zip(a, acceleration))
    for i in range(2):
        zero(f"Legendre_inverse_{i}", momenta[i].subs(substitutions)-r[i])
    H = (p.dot(v)+r.dot(a)-L).subs(substitutions)
    expected = (p.dot(v)-v.dot(v)+grad-V-r.dot(r)/(4*kap*F)
                + A*r.dot(b)/(2*kap)
                + (theta-gam**2/(4*kap))*F*b.dot(b))
    zero("full_Hamiltonian_Legendre_transform", H-expected)
    for i in range(2):
        for j in range(2):
            zero(f"acceleration_Hessian_{i}{j}",
                 s.diff(L, a[i], a[j])+2*kap*F*int(i == j))
            zero(f"ordinary_velocity_Hessian_{i}{j}",
                 s.diff(L, v[i], v[j])-2*int(i == j))

    # Fixed-charge descent curve: a direct identity for arbitrary canonical jets.
    X = s.Matrix(s.symbols("x0 x1", real=True))
    J = s.Matrix([[0, -1], [1, 0]])
    eta, h, tau = s.symbols("eta h tau", real=True)
    w = eta*X
    c = -(r.dot(J*w))/X.dot(X)*(J*X)
    vs, ps = v+h*w, p+h*(c+tau*w)
    charge = lambda pp, vv: pp.dot(J*X)+r.dot(J*vv)
    zero("exact_charge_preserving_curve", charge(ps, vs)-charge(p, v))
    difference = s.expand(ps.dot(vs)-vs.dot(vs)-p.dot(v)+v.dot(v))
    zero("fixed_charge_negative_quadratic", difference.coeff(h, 2)-(tau-1)*w.dot(w))

    # General adjoint: missing coefficient derivatives are a real restriction.
    hh, ff = s.Function("h")(z), s.Function("b")(z)
    zero("variable_projector_adjoint_correction",
         s.diff(hh*ff, z, 2)-hh*s.diff(ff, z, 2)
         -2*s.diff(hh, z)*s.diff(ff, z)-s.diff(hh, z, 2)*ff)

    # Direct constant-background amplitude/phase quadratic density and symbol.
    k, omega, amp, eps = s.symbols("k omega amp eps", real=True)
    rx, rv, rt, vt, rs, vsj, rtt, vtt, rss, vss = s.symbols(
        "rx rv rt vt rs vsj rtt vtt rss vss", real=True)
    f0, f1, f2, V1, V2 = s.symbols("f0 f1 f2 V1 V2", real=True)
    du = 2*amp*eps*rx+eps**2*(rx*rx+rv*rv)
    H2 = kap*((-rtt+rss)**2+(-vtt+vss)**2)+theta*(rss*rss+vss*vss)
    H2 += gam*((-rtt+rss)*rss+(-vtt+vss)*vss)
    expansion = (eps**2*(-rt*rt-vt*vt+rs*rs+vsj*vsj)
                 -V1*du-V2*du**2/2+(f0+f1*du+f2*du**2/2)*eps**2*H2)
    quadratic = -s.expand(expansion).coeff(eps, 2).subs(V1, 0)
    jets = [[rx, rt, rs, rtt, rss], [rv, vt, vsj, vtt, vss]]
    fourier = [1, -s.I*omega, s.I*k, -omega**2, -k**2]
    for i in range(2):
        block = sum(s.conjugate(fourier[a])*s.diff(quadratic, jets[i][a], jets[i][b])
                    *fourier[b] for a in range(5) for b in range(5))/2
        mass2 = -2*amp**2*V2 if i == 0 else 0
        yy = omega**2-k**2
        target = yy-mass2-f0*kap*yy**2+f0*gam*k**2*yy-f0*theta*k**4
        zero(f"direct_quadratic_symbol_{i}", block-target)

    K, G, T, q, m2, zz = s.symbols("K G T q m2 zz", real=True)
    D = (zz-q)-m2-K*(zz-q)**2+G*q*(zz-q)-T*q*q
    discriminant = (1+G*q)**2-4*K*(m2+T*q*q)
    lo = q+(1+G*q-s.sqrt(discriminant))/(2*K)
    hi = q+(1+G*q+s.sqrt(discriminant))/(2*K)
    zero("pole_factorization", D+K*(zz-lo)*(zz-hi))
    zero("low_residue", s.diff(D, zz).subs(zz, lo)-s.sqrt(discriminant))
    zero("high_residue", s.diff(D, zz).subs(zz, hi)+s.sqrt(discriminant))
    B = 4*K*T-G**2
    zero("UV_discriminant", discriminant-((1-4*K*m2)+2*G*q-B*q*q))
    velocity2 = s.symbols("velocity2", real=True)
    principal = s.Poly(s.expand(D.subs(zz, velocity2*q)), q).coeff_monomial(q**2)
    zero("principal_symbol", principal-(-K*(velocity2-1)**2+G*(velocity2-1)-T))
    a = (2*K+G)/(2*K)
    b2 = B/(4*K*K)
    zero("principal_complex_speed_pair", principal+K*((velocity2-a)**2+b2))

    # The proposed band cutoff is not a nonlinear invariant subspace.
    phase = s.symbols("phase", real=True)
    zero("cubic_generates_third_harmonic", s.expand_trig(s.cos(3*phase))
         -(4*s.cos(phase)**3-3*s.cos(phase)))
    R, Rp, phip = s.symbols("R Rp phip", real=True)
    differentiated_mode = Rp+s.I*R*phip  # e^(-i phi) d(R e^(i phi))/ds
    zero("polar_mode_Dirichlet_energy",
         differentiated_mode*s.conjugate(differentiated_mode)-Rp**2-R**2*phip**2)
    arc = s.symbols("arc", real=True)
    aspect = s.symbols("aspect", positive=True)
    mode = s.cos(arc)+s.I*aspect*s.sin(arc)
    zero("locked_ellipse_mode_eigenvalue", -s.diff(mode, arc, 2)-mode)
    zero("locked_ellipse_phase_flux", s.im(s.conjugate(mode)*s.diff(mode, arc))-aspect)
    zero("locked_ellipse_tangent_antisymmetry", mode.subs(arc, arc+s.pi)+mode)
    norm = s.integrate(s.expand_complex(mode*s.conjugate(mode)), (arc, 0, 2*s.pi))
    zero("locked_ellipse_mean_gap", 1-(2*s.pi*aspect/norm)-(aspect-1)**2/(1+aspect**2))

    # Exponent pairs (field amplitude A, length L), with tau=c0*t.
    dims = {"ordinary": (2, -2), "potential": (2, -2),
            "F_kappa_Q_squared": tuple(sum(v) for v in zip((-2, 0), (2, 2), (2, -4))),
            "F_theta_P_squared": tuple(sum(v) for v in zip((-2, 0), (2, 2), (2, -4)))}
    if any(value != (2, -2) for value in dims.values()):
        raise AssertionError(dims)
    checks.append("action_term_dimensions")
    return {"count": len(checks), "checks": checks, "action_dimensions": dims}


def uv_growth(K, G, T):
    if not all(math.isfinite(x) for x in (K, G, T)):
        raise ValueError("finite coefficients required")
    if K <= 0 or T <= 0 or 4*K*T-G*G <= 0:
        raise ValueError("UV theorem requires K,T>0 and strict curvature-form positivity")
    a = (2*K+G)/(2*K)
    b = math.sqrt(4*K*T-G*G)/(2*K)
    # Rationalized form avoids cancellation for small b and positive a.
    c = b/math.sqrt(2*(math.hypot(a, b)+a))
    return c


def numerical_checks():
    K, G, T, m2 = 1.0, 0.0, 1.0, 0.0  # analytic phase witness, no fit
    limit = uv_growth(K, G, T)
    rows = []
    for k in (1, 2, 4, 8, 16, 32, 64, 128):
        q = float(k*k)
        # Independent companion roots of the time-domain ODE in s/k.
        coefficients = [1, 0, 1/(K*q)+2+G/K, 0,
                        m2/(K*q*q)+1/(K*q)+1+G/K+T/K]
        roots = np.roots(coefficients)
        observed = float(max(roots.real))
        disc = complex((1+G*q)**2-4*K*(m2+T*q*q))
        zhi = q+(1+G*q+np.sqrt(disc))/(2*K)
        exact = float(abs(np.sqrt(complex(zhi)).imag))/k
        error = abs(observed-exact)
        if error > 2e-12 or observed <= 0:
            raise AssertionError((k, observed, exact))
        rows.append({"k": k, "growth_over_k": observed,
                     "companion_vs_pole_error": error,
                     "asymptotic_error": abs(observed-limit),
                     "log10_amplification_at_tau_1": observed*k/math.log(10)})
    if rows[-1]["asymptotic_error"] > 2e-5:
        raise AssertionError("UV asymptotic witness failed")
    orders = [math.log2(rows[i]["asymptotic_error"]/rows[i+1]["asymptotic_error"])
              for i in range(len(rows)-1)]
    # Boundary control: pure Lorentzian K term has B=0; it still has a ghost,
    # but its phase poles q and q+1/K are real. Do not extend the UV theorem.
    for q in (0, 1, 100):
        roots = np.roots([-1, 1+2*q, -q-q*q])
        if not np.allclose(sorted(roots), [q, q+1], atol=1e-10, rtol=1e-10):
            raise AssertionError("Lorentzian limiting control")
    return {"witness": {"K": K, "G": G, "T": T, "m2": m2},
            "growth_coefficient_limit": limit, "rows": rows,
            "asymptotic_orders": orders, "lorentzian_boundary_control": "PASS",
            "interpretation": "exact linear-mode diagnostic, not PDE simulation or nonlinear lifetime"}


def eft_error_check():
    with mp.workdps(70):
        K, G, T, m2 = mp.mpf(".001"), mp.mpf(2), mp.mpf(2000), mp.mpf(1)
        ss = mp.sqrt(1-4*K*m2)
        C0 = 2*m2/(1+ss)
        C2 = 1-G*C0/ss
        C4 = T/ss+G*G*m2/ss**3
        B = 4*K*T-G*G
        qmax = mp.mpf(".0005")
        disc = lambda q: ss**2+2*G*q-B*q*q
        dmin = min(disc(0), disc(qmax))  # concave quadratic
        if dmin <= 0:
            raise AssertionError("EFT window crosses collision")
        coefficient = ((B*ss**2+G*G)*max(abs(G), abs(G-B*qmax))
                       /(4*K*dmin**mp.mpf("2.5")))
        rows = []
        for q in [qmax/8, qmax/4, qmax/2, qmax]:
            exact = q+2*(m2+T*q*q)/(1+G*q+mp.sqrt(disc(q)))
            err = abs(exact-(C0+C2*q+C4*q*q))
            bound = coefficient*q**3
            if err > bound:
                raise AssertionError("quadratic EFT Taylor bound failed")
            rows.append({"q": str(q), "error": str(err), "bound": str(bound)})
        return {"scope": "one real linear amplitude branch only",
                "dmin": str(dmin), "rows": rows}


def run():
    return {"schema": "wct-foundational-verification-v1", "status": "PASS",
            "symbolic": symbolic_checks(), "linear_mode_checks": numerical_checks(),
            "EFT_remainder": eft_error_check(),
            "versions": {"python": platform.python_version(), "sympy": s.__version__,
                         "numpy": np.__version__, "mpmath": mp.__version__},
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "PDE_propagations": 0, "empirical_fit_inputs": [],
            "physical_mass_predictions": [], "nonlinear_EFT_closure": "NOT_ESTABLISHED"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite an existing evidence receipt")
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"status": result["status"], "symbolic_checks": result["symbolic"]["count"],
                      "output": str(args.output), "PDE_propagations": 0}))
