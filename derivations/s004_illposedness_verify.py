"""S004 Phase XXXIII: variable-coefficient Sobolev ill-posedness of the linearized S004 equations.

Independent checks for docs/s004_closure/VARIABLE_COEFFICIENT_ILLPOSEDNESS.md. Every check re-derives
its input from the corrected S004 density (WCT_CORRECTED_VARIATIONAL_CLOSURE_DRAFT.md section 2)
through ``sympy.calculus.euler.euler_equations``. Nothing is fitted.

I1  structure of the linearization about an arbitrary background (orders, coefficient dependence)
I2  Hadamard root: factorization of the frozen symbol, simplicity, Re w > 0 (exact)
I3  exact localized beam of the frozen principal part in 1+1D and its closed-form L2 norm (exact)
I4  exact Gaussian-beam phase in 1+d (d = 1, 2, 3): eikonal solved exactly, phase bounds
I5  conjugation identity e^{-N phi} P(d)(e^{N phi} a) = N^4 P(grad phi) a + N^3 T_P a + O(N^2) (exact)
I6  class A: co-rotating homogeneous background -> constant coefficients; root asymptotics
I7  end-to-end: localized WKB approximate solutions of the EXACT linearization about a class-C
    background (node elsewhere); true residual, growth, data, contradiction ratio (numerical)
I8  exponent bookkeeping for the contradiction (exact rational arithmetic)

Exact (sympy) checks raise AssertionError on failure. I6 and I7 are numerical and are labelled so.

Run:  python derivations/s004_illposedness_verify.py --output NEW/illposedness_verification.json
Test: python -m unittest -v derivations.test_s004_illposedness
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as s
from scipy.signal import convolve2d
from sympy.calculus.euler import euler_equations

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "WCT_CORRECTED_VARIATIONAL_CLOSURE_DRAFT.md"


def _zero(expr) -> bool:
    return s.simplify(s.expand(expr)) == 0


def _s004_equations(coords, F, V, kap, th, gam):
    """Euler-Lagrange equations of the corrected S004 density (original orientation) in real
    components psi = x1 + i x2 on flat space-time with signature (-,+,...,+)."""
    t, *xs = coords
    x1, x2 = s.Function("x1")(*coords), s.Function("x2")(*coords)
    rho = x1**2 + x2**2

    def lap(q):
        return sum(s.diff(q, y, 2) for y in xs)

    def box(q):
        return -s.diff(q, t, 2) + lap(q)

    curvature = sum(kap * box(q)**2 + th * lap(q)**2 + gam * box(q) * lap(q) for q in (x1, x2))
    kinetic = sum(-s.diff(q, t)**2 + sum(s.diff(q, y)**2 for y in xs) for q in (x1, x2))
    lagrangian = kinetic - V(rho) + F(rho) * curvature
    return euler_equations(lagrangian, [x1, x2], list(coords)), (x1, x2)


# ---------------------------------------------------------------------------
# I1  structure of the linearized operator about an ARBITRARY smooth background
# ---------------------------------------------------------------------------
def linearization_structure(spatial_dims: int = 1) -> dict:
    t = s.Symbol("t", real=True)
    coords = (t, *s.symbols(f"y1:{spatial_dims + 1}", real=True))
    kap, th, gam = s.symbols("kappa theta gamma", real=True)
    Ff, Vf = s.Function("F"), s.Function("V")
    eqs, (x1, x2) = _s004_equations(coords, Ff, Vf, kap, th, gam)
    X1, X2 = s.Function("X1")(*coords), s.Function("X2")(*coords)
    e1, e2 = s.Function("e1")(*coords), s.Function("e2")(*coords)
    ep = s.Symbol("ep")
    sub = {x1: X1 + ep * e1, x2: X2 + ep * e2}
    max_e, max_bg, max_F, max_V = 0, 0, 0, 0
    budget_ok = True
    for eq in eqs:
        lin = s.expand(s.diff((eq.lhs - eq.rhs).subs(sub), ep).subs(ep, 0).doit())
        if lin.has(s.I):
            raise AssertionError("imaginary unit in a real linearized coefficient")
        for term in s.Add.make_args(lin):
            e_order, bg_order = None, 0
            for d in term.atoms(s.Derivative):
                if d.expr in (e1, e2):
                    e_order = d.derivative_count
                elif d.expr in (X1, X2):
                    bg_order = max(bg_order, d.derivative_count)
                elif isinstance(d.expr, s.Function) and d.expr.func == Ff:
                    max_F = max(max_F, d.derivative_count)
                elif isinstance(d.expr, s.Function) and d.expr.func == Vf:
                    max_V = max(max_V, d.derivative_count)
            for sb in term.atoms(s.Subs):
                if isinstance(sb.expr, s.Derivative):
                    fn = sb.expr.expr.func
                    if fn == Ff:
                        max_F = max(max_F, sb.expr.derivative_count)
                    if fn == Vf:
                        max_V = max(max_V, sb.expr.derivative_count)
            if e_order is None:
                if not (term.has(e1) or term.has(e2)):
                    raise AssertionError(("term without perturbation", term))
                e_order = 0
            # the TOTAL number of background derivatives in the coefficient of d^alpha e is
            # at most 4 - |alpha| (Lemma 1: every term of E carries 4, 2 or 0 derivatives in all)
            bg_total = 0
            for base, power in term.as_powers_dict().items():
                if isinstance(base, s.Derivative) and base.expr in (X1, X2):
                    bg_total += int(power) * base.derivative_count
            if bg_total > 4 - e_order:
                budget_ok = False
            max_e, max_bg = max(max_e, e_order), max(max_bg, bg_order)
    if not budget_ok:
        raise AssertionError("a coefficient of d^alpha e uses more than 4-|alpha| background derivatives")
    if (max_e, max_bg, max_F, max_V) != (4, 4, 3, 2):
        raise AssertionError(("unexpected structure", max_e, max_bg, max_F, max_V))
    return {"spatial_dims": spatial_dims,
            "operator_order": int(max_e), "max_background_derivative": int(max_bg),
            "max_F_derivative": int(max_F), "max_V_derivative": int(max_V),
            "coefficient_of_d^alpha_has_total_background_derivative_order_at_most_4-|alpha|": True,
            "real_coefficients": True}


# ---------------------------------------------------------------------------
# I2  Hadamard root: factorization, simplicity, growth constant
# ---------------------------------------------------------------------------
def hadamard_root_simplicity() -> dict:
    kap, th = s.symbols("kappa theta", positive=True)
    gam, lam, Z = s.symbols("gamma lambda Z")
    disc = s.Symbol("Delta", positive=True)   # Delta = sqrt(4 kappa theta - gamma^2) > 0
    th_of = (disc**2 + gam**2) / (4 * kap)
    mu = -(2 * kap + gam) / (2 * kap) + s.I * disc / (2 * kap)
    mub = -(2 * kap + gam) / (2 * kap) - s.I * disc / (2 * kap)
    # P in the Laplace variable: d_t -> lambda, d_x -> zeta', Z = zeta'.zeta' (= -|xi|^2 if zeta'=i xi)
    P = kap * (Z - lam**2)**2 + gam * (Z - lam**2) * Z + th_of * Z**2
    if not _zero(P - kap * (lam**2 + mu * Z) * (lam**2 + mub * Z)):
        raise AssertionError("factorization P = kappa (lam^2 + mu Z)(lam^2 + conj(mu) Z)")
    # simple root: lam^2 = -mu Z;  with Z = -|xi|^2 = -1: lam = w, w^2 = mu
    w = s.Symbol("w")
    dP = s.diff(P, lam).subs(Z, -1)
    dP_at_root = s.expand(dP).subs(lam**3, w * mu).subs(lam, w)
    target = 2 * w * s.I * disc        # = 4 i kappa w Im(mu)
    if not _zero(s.expand(dP_at_root - target)):
        raise AssertionError(("dP/dlam at the Hadamard root", s.simplify(dP_at_root)))
    if not _zero(4 * kap * (disc / (2 * kap)) - 2 * disc):
        raise AssertionError("4 kappa Im(mu) = 2 Delta")
    # Re sqrt(mu) > 0 because Im mu != 0:  (Re w)^2 = (|mu| + Re mu)/2 > 0
    a, b = s.symbols("a b", real=True)
    if not _zero(s.expand((a + s.I * b)**2) - (a**2 - b**2 + 2 * s.I * a * b)):
        raise AssertionError("square")
    samples = []
    for (k_, t_, g_) in [(1, 1, 0), (1, 4, 0), (s.Rational(1, 3), 2, -1), (2, 1, s.Rational(5, 2))]:
        if not (g_**2 < 4 * k_ * t_):
            continue
        muv = complex(s.N(-(2 * k_ + g_) / (2 * k_) + s.I * s.sqrt(4 * k_ * t_ - g_**2) / (2 * k_), 30))
        wv = complex(np.sqrt(muv))
        pv = k_ * (muv + 1)**2 + g_ * (muv + 1) + t_
        samples.append({"kappa": str(k_), "theta": str(t_), "gamma": str(g_),
                        "w": [wv.real, wv.imag], "|p(mu)|": abs(complex(pv))})
        if not (wv.real > 0 and abs(complex(pv)) < 1e-12):
            raise AssertionError(("root sample", k_, t_, g_))
    return {"factorization": "P = kappa (lam^2 + mu Z)(lam^2 + conj(mu) Z)",
            "dP/dlam at lam=w|xi|": "2 i w |xi|^3 sqrt(4 kappa theta - gamma^2) != 0 (simple root)",
            "samples": samples}


# ---------------------------------------------------------------------------
# I3  exact localized beam of the frozen principal part (1+1D)
# ---------------------------------------------------------------------------
def exact_beam_1d() -> dict:
    t, y = s.symbols("t y", real=True)
    N = s.Symbol("N", positive=True)
    kap, gam, w = s.symbols("kappa gamma w")
    mu = w**2
    th = -kap * (mu + 1)**2 - gam * (mu + 1)      # imposes p(mu) = 0 exactly
    phi = s.I * y + w * t - (y - s.I * w * t)**2 / 2
    if not _zero(s.diff(phi, t) + s.I * w * s.diff(phi, y)):
        raise AssertionError("phi_t = -i w phi_y")
    u = s.exp(N * phi)

    def box(q):
        return -s.diff(q, t, 2) + s.diff(q, y, 2)

    def lap(q):
        return s.diff(q, y, 2)

    Pu = kap * box(box(u)) + gam * box(lap(u)) + th * lap(lap(u))
    if not _zero(s.simplify(Pu / u)):
        raise AssertionError("e^{N phi} is not an exact solution of the frozen principal part")
    # closed-form modulus: 2 N Re phi = -N (y + t b)^2 + 2 N t a + N t^2 a^2  for w = a + i b
    a, b = s.symbols("a b", real=True)
    phir = phi.subs(w, a + s.I * b)
    re2 = s.expand(2 * N * s.re(s.expand(phir)))
    target = -N * (y + t * b)**2 + 2 * N * t * a + N * t**2 * a**2
    if not _zero(re2 - s.expand(target)):
        raise AssertionError(("Re phi completed square", s.simplify(re2 - target)))
    yy = s.Symbol("yy", real=True)
    norm2 = s.integrate(s.exp(-N * yy**2), (yy, -s.oo, s.oo)) * s.exp(2 * N * t * a + N * t**2 * a**2)
    if not _zero(norm2 - s.sqrt(s.pi / N) * s.exp(2 * N * t * a + N * t**2 * a**2)):
        raise AssertionError("closed-form L2 norm")
    return {"phase": "phi = i y + w t - (y - i w t)^2/2, phi_t = -i w phi_y",
            "exact_solution": "P(d) e^{N phi} = 0 identically when kappa(mu+1)^2+gamma(mu+1)+theta = 0",
            "L2_norm_squared": "sqrt(pi/N) exp(2 N t Re w + N t^2 (Re w)^2)"}


# ---------------------------------------------------------------------------
# I4  exact Gaussian-beam phase in 1+d dimensions (d = 1, 2, 3)
# ---------------------------------------------------------------------------
def exact_beam_phase(dims=(1, 2, 3)) -> dict:
    """phi = i e.x - |x|^2/2 + w t S(x) + mu t^2/2, S = sqrt(1 + 2 i e.x - |x|^2), w^2 = mu.
    Claim: grad_x phi . grad_x phi = -(S + w t)^2, so phi_t = w (S + w t) = H(grad_x phi) and
    P(grad phi) = kappa (phi_t^2 + mu Z)(phi_t^2 + conj(mu) Z) = 0 identically: the eikonal equation
    of the localized construction is solved EXACTLY (no truncation in t)."""
    out = {}
    t, w, mub = s.symbols("t w mubar")
    kap, gam = s.symbols("kappa gamma")
    for d in dims:
        xs = s.symbols(f"x1:{d + 1}", real=True)
        q = 1 + 2 * s.I * xs[0] - sum(x**2 for x in xs)
        S = s.sqrt(q)
        phi = s.I * xs[0] - sum(x**2 for x in xs) / 2 + w * t * S + w**2 * t**2 / 2
        grad = [s.diff(phi, x) for x in xs]
        Z = sum(g**2 for g in grad)
        if not _zero(s.simplify(Z + (S + w * t)**2)):
            raise AssertionError(("grad.grad = -(S + w t)^2", d))
        pt = s.diff(phi, t)
        if not _zero(pt - w * (S + w * t)):
            raise AssertionError(("phi_t = w (S + w t)", d))
        # P(grad phi) with mu = w^2 and an arbitrary second factor (it vanishes through the first)
        first = s.simplify(pt**2 + w**2 * Z)
        if not _zero(first):
            raise AssertionError(("phi_t^2 + mu Z = 0", d))
        # normalizations used in the proof
        at0 = {x: 0 for x in xs}
        if not (_zero(S.subs(at0) - 1) and _zero(pt.subs(at0).subs(t, 0) - w)):
            raise AssertionError(("S(0) = 1, phi_t(0,0) = w", d))
        if not _zero(s.re(phi.subs(t, 0)) + sum(x**2 for x in xs) / 2):
            raise AssertionError(("Re phi(0,x) = -|x|^2/2", d))
        out[f"d={d}"] = "grad.grad = -(S+wt)^2; phi_t = w(S+wt) = H(grad phi); P(grad phi) = 0 exactly"
    # |S(x) - 1| <= |z - 1| <= 2|x| + |x|^2 <= (9/4)|x| for |x| <= 1/4 (principal branch, Re sqrt > 0)
    rng = np.random.default_rng(4)
    worst = 0.0
    for _ in range(20000):
        d = int(rng.integers(1, 4))
        x = rng.normal(size=d)
        x *= rng.uniform(0, 0.25) / np.linalg.norm(x)
        z = 1 + 2j * x[0] - float(x @ x)
        worst = max(worst, abs(np.sqrt(z) - 1) / max(np.linalg.norm(x), 1e-300))
    if worst > 9 / 4:
        raise AssertionError(("|S - 1| <= (9/4)|x|", worst))
    out["sup |S(x)-1|/|x| on |x|<=1/4 (sampled)"] = worst
    out["bound used"] = "|S(x) - 1| <= (9/4)|x|, hence |Re phi(t,x) - Re phi(t,0) + |x|^2/2| <= (9/4)|w| t |x|"
    return out


# ---------------------------------------------------------------------------
# I5  conjugation identity (exact, generic phase and amplitude)
# ---------------------------------------------------------------------------
def conjugation_expansion(dims=(1, 2)) -> dict:
    """e^{-N phi} P(d)(e^{N phi} a) = N^4 P(grad phi) a
       + N^3 [sum_j P^(j)(grad phi) d_j a + (1/2) sum_jk P^(jk)(grad phi) phi_jk a] + O(N^2),
    for GENERIC phi and a, in 1+1 and 1+2 dimensions (exact)."""
    out = {}
    N = s.Symbol("N")
    kap, th, gam = s.symbols("kappa theta gamma")
    for d in dims:
        X = s.symbols(f"t x1:{d + 1}")
        phi, a = s.Function("phi")(*X), s.Function("a")(*X)
        z = s.symbols(f"z0:{d + 1}")
        Z = sum(zz**2 for zz in z[1:])
        Psym = kap * (Z - z[0]**2)**2 + gam * (Z - z[0]**2) * Z + th * Z**2
        u = s.exp(N * phi) * a

        def lap(q):
            return sum(s.diff(q, x, 2) for x in X[1:])

        def box(q):
            return -s.diff(q, X[0], 2) + lap(q)

        conj = s.expand(s.powsimp(s.expand((kap * box(box(u)) + gam * box(lap(u)) + th * lap(lap(u)))
                                           * s.exp(-N * phi))))
        poly = s.Poly(conj, N)
        grad = {z[i]: s.diff(phi, X[i]) for i in range(d + 1)}
        c4 = Psym.subs(grad) * a
        c3 = sum(s.diff(Psym, z[j]).subs(grad) * s.diff(a, X[j]) for j in range(d + 1))
        c3 += s.Rational(1, 2) * sum(s.diff(Psym, z[j], z[k]).subs(grad) * s.diff(phi, X[j], X[k])
                                     for j in range(d + 1) for k in range(d + 1)) * a
        if poly.degree() != 4:
            raise AssertionError(("degree in N", d))
        if s.expand(poly.coeff_monomial(N**4) - c4) != 0:
            raise AssertionError(("N^4 coefficient", d))
        if s.expand(poly.coeff_monomial(N**3) - c3) != 0:
            raise AssertionError(("N^3 coefficient (transport operator)", d))
        out[f"1+{d}"] = "N^4: P(grad phi) a; N^3: sum_j P^(j)(grad phi) d_j a + (1/2) sum_jk P^(jk)(grad phi) phi_jk a"
    return out


# ---------------------------------------------------------------------------
# I6  class A: co-rotating homogeneous background (numerical root asymptotics)
# ---------------------------------------------------------------------------
def corotating_class_a(omega: float = 0.7, rho0: float = 0.8) -> dict:
    t, y, u = s.symbols("t y u", real=True)
    kap, th, gam = s.Integer(1), s.Integer(2), s.Rational(1, 2)
    om, r0 = s.Rational(str(omega)), s.Rational(str(rho0))
    # Only F and its first three derivatives, and the first two derivatives of V, enter the
    # linearization (I1), always evaluated at rho0, which is constant on this background. Replacing
    # F and V by their Taylor polynomials at rho0 is therefore exact.
    f0, f1c, f2c, f3c, v1c, v2c = s.symbols("f0 f1c f2c f3c v1c v2c", real=True)
    F = s.Lambda(u, f0 + f1c * (u - r0) + f2c * (u - r0)**2 / 2 + f3c * (u - r0)**3 / 6)
    V = s.Lambda(u, v1c * (u - r0) + v2c * (u - r0)**2 / 2)
    eqs, (x1, x2) = _s004_equations((t, y), F, V, kap, th, gam)
    # concrete corrected weight (eps = 1, alpha = 3/10) and a generic potential, NOT on-shell
    uu = s.Symbol("uu", positive=True)
    Fc = uu / (uu + s.exp(-2 * s.Rational(3, 10) * uu))**2
    Vc = -(uu - 1)**2 / 2 + uu**3 / 5
    values = {f0: Fc, f1c: s.diff(Fc, uu), f2c: s.diff(Fc, uu, 2), f3c: s.diff(Fc, uu, 3),
              v1c: s.diff(Vc, uu), v2c: s.diff(Vc, uu, 2)}
    values = {k_: s.Float(s.N(v_.subs(uu, r0), 30), 30) for k_, v_ in values.items()}
    A = s.sqrt(r0)
    c, sn = s.cos(om * t), s.sin(om * t)
    X1, X2 = A * c, A * sn
    e1, e2 = s.Function("e1")(t, y), s.Function("e2")(t, y)
    f1, f2 = s.Function("f1")(t, y), s.Function("f2")(t, y)
    ep = s.Symbol("ep")
    rows = []
    for eq in eqs:
        lin = s.diff((eq.lhs - eq.rhs).subs({x1: x1 + ep * e1, x2: x2 + ep * e2}), ep).subs(ep, 0)
        lin = lin.subs({x1: X1, x2: X2}).doit()
        # co-rotating perturbation e = R(om t) f
        lin = lin.subs({e1: c * f1 - sn * f2, e2: sn * f1 + c * f2}).doit()
        rows.append(lin)
    # rotate the equations back: R(om t)^T (lin)
    rot = [s.expand(c * rows[0] + sn * rows[1]), s.expand(-sn * rows[0] + c * rows[1])]
    lam, k = s.symbols("lam k")
    M = s.zeros(2, 2)
    for i in range(2):
        expr = rot[i]
        for j, f in enumerate((f1, f2)):
            ent = 0
            for a_ in range(5):
                for b_ in range(5 - a_):
                    if a_ + b_ == 0:
                        continue
                    d = s.diff(f, *([t] * a_ + [y] * b_))
                    cf = expr.coeff(d)
                    ent += cf * lam**a_ * (s.I * k)**b_
                    expr = expr - cf * d
            M[i, j] = ent
            M[i, j] += s.expand(expr).coeff(f)
    # constant coefficients: no t or y left
    tvals = [0, s.Rational(1, 3), s.Rational(7, 5)]
    for i in range(2):
        for j in range(2):
            vals = [complex(s.N(M[i, j].subs(values).subs({t: tv, lam: 1, k: 2}), 20)) for tv in tvals]
            spread = max(abs(p_ - q_) for p_ in vals for q_ in vals)
            if spread > 1e-12 * max(1.0, max(abs(p_) for p_ in vals)):
                raise AssertionError(("co-rotating coefficients are not constant", i, j, spread))
    Mc = M.subs(values).subs(t, 0)
    det = s.expand(Mc[0, 0] * Mc[1, 1] - Mc[0, 1] * Mc[1, 0])
    muv = complex(s.N(-(2 * kap + gam) / (2 * kap) + s.I * s.sqrt(4 * kap * th - gam**2) / (2 * kap)))
    wv = np.sqrt(muv)
    table = []
    for kv in (10.0, 100.0, 1000.0, 10000.0):
        coeffs = [complex(s.N(cf, 30)) for cf in s.Poly(det.subs(k, kv), lam).all_coeffs()]
        roots = np.roots(coeffs)
        top = roots[np.argmax(roots.real)]
        table.append({"k": kv, "max Re lam / k": float(top.real / kv), "Re w": float(wv.real),
                      "|lam/k - w| (nearest root)": float(np.min(np.abs(roots / kv - wv)))})
    if not (table[-1]["|lam/k - w| (nearest root)"] < 1e-3 and table[-1]["max Re lam / k"] > 0.9 * wv.real):
        raise AssertionError(("class A root asymptotics", table))
    return {"background": f"psi0 = sqrt({rho0}) exp(i {omega} t) (not on-shell for the chosen V)",
            "co-rotating coefficients constant": True, "w": [wv.real, wv.imag], "table": table}


# ---------------------------------------------------------------------------
# I7  end-to-end localized WKB on the EXACT linearization about nonconstant backgrounds
# ---------------------------------------------------------------------------
# The Euler-Lagrange linearization is computed once with generic F, V (as in I1); the values
# F^(k)(rho0), V^(k)(rho0) (k <= 3, resp. 2) are then placeholders f_k, v_k. This is exact, because
# after `doit()` F and V occur only through these values (I1), and it is ~40x faster than expanding
# with the explicit weight. Each background is substituted exactly; its fourth-order coefficients
# are checked SYMBOLICALLY against 2 f0 (kappa, -(2kappa+gamma), kappa+theta+gamma) I2.
_GENERIC_LIN_CACHE: dict = {}


def _generic_linearization(kap, th, gam):
    key = (kap, th, gam)
    if key in _GENERIC_LIN_CACHE:
        return _GENERIC_LIN_CACHE[key]
    t, y = s.symbols("t y", real=True)
    Ff, Vf = s.Function("F"), s.Function("V")
    eqs, (x1, x2) = _s004_equations((t, y), Ff, Vf, kap, th, gam)
    X1g, X2g = s.Function("X1")(t, y), s.Function("X2")(t, y)
    e1, e2 = s.Function("e1")(t, y), s.Function("e2")(t, y)
    ep = s.Symbol("ep")
    fk, vk = s.symbols("f0:4"), s.symbols("v0:3")

    def placeholders(expr):
        rep = {}
        for sb in expr.atoms(s.Subs):
            if isinstance(sb.expr, s.Derivative) and isinstance(sb.expr.expr, s.Function):
                n = sb.expr.derivative_count
                rep[sb] = fk[n] if sb.expr.expr.func == Ff else vk[n]
        for d in expr.atoms(s.Derivative):
            if isinstance(d.expr, s.Function) and d.expr.func in (Ff, Vf):
                n = d.derivative_count
                rep[d] = fk[n] if d.expr.func == Ff else vk[n]
        expr = expr.xreplace(rep)
        rep2 = {f: (fk[0] if f.func == Ff else vk[0]) for f in expr.atoms(s.Function) if f.func in (Ff, Vf)}
        return expr.xreplace(rep2)

    lins = []
    for eq in eqs:
        lin = s.diff((eq.lhs - eq.rhs).subs({x1: X1g + ep * e1, x2: X2g + ep * e2}), ep).subs(ep, 0).doit()
        lin = placeholders(s.expand(lin))
        if lin.has(Ff) or lin.has(Vf):
            raise AssertionError("F or V left after placeholder substitution")
        lins.append(lin)
    out = ((t, y), lins, (X1g, X2g), (e1, e2), fk, vk)
    _GENERIC_LIN_CACHE[key] = out
    return out


def _background_operator(spec):
    """Exact linearized operator of the S004 system about the background in `spec`.
    Returns the symbolic coefficients and a numerical evaluator of B_alpha = A_alpha / (2 F(rho0))."""
    kap, th, gam = spec["kappa"], spec["theta"], spec["gamma"]
    (t, y), lins, (X1g, X2g), (e1, e2), fk, vk = _generic_linearization(kap, th, gam)
    X1, X2 = spec["X1"](t, y), spec["X2"](t, y)
    coeffs = {}
    for i, lin in enumerate(lins):
        ex = s.expand(lin.subs({X1g: X1, X2g: X2}).doit())
        rest = ex
        for k, e in enumerate((e1, e2)):
            for a_ in range(5):
                for b_ in range(5 - a_):
                    if a_ + b_ == 0:
                        continue
                    d = s.diff(e, *([t] * a_ + [y] * b_))
                    coeffs[(i, k, a_, b_)] = ex.coeff(d)
                    rest = rest - coeffs[(i, k, a_, b_)] * d
        rest = s.expand(rest)
        for k, e in enumerate((e1, e2)):
            coeffs[(i, k, 0, 0)] = rest.coeff(e)
    # exact principal part on this background
    prin = {(4, 0): kap, (3, 1): 0, (2, 2): -(2 * kap + gam), (1, 3): 0, (0, 4): kap + th + gam}
    for (a_, b_), val in prin.items():
        for i in range(2):
            for k in range(2):
                target = 2 * fk[0] * val if i == k else 0
                if not _zero(coeffs[(i, k, a_, b_)] - target):
                    raise AssertionError(("principal coefficient", spec["name"], a_, b_, i, k))
    u = s.Symbol("u")
    Fexpr, Vexpr = spec["F"](u), spec["V"](u)
    Fd = [s.lambdify(u, s.diff(Fexpr, u, n), "numpy") for n in range(4)]
    Vd = [s.lambdify(u, s.diff(Vexpr, u, n), "numpy") for n in range(3)]
    rho = s.lambdify((t, y), X1**2 + X2**2, "numpy")
    keys = [(a_, b_) for a_ in range(4) for b_ in range(4 - a_)]
    flat = [(i, k, a_, b_) for (a_, b_) in keys for i in range(2) for k in range(2)]
    fn = s.lambdify((t, y, *fk, *vk), [coeffs[q] / (2 * fk[0]) for q in flat], modules="numpy", cse=True)

    def Bexact(tt, yy):
        shape = np.broadcast(tt, yy).shape
        r = np.broadcast_to(rho(tt, yy), shape)
        vals = fn(tt, yy, *[f(r) for f in Fd], *[g(r) for g in Vd])
        out = {}
        for n, (i, k, a_, b_) in enumerate(flat):
            out.setdefault((a_, b_), np.zeros((2, 2) + shape, complex))
            out[(a_, b_)][i, k] = np.broadcast_to(vals[n], shape)
        return out

    rho00 = float(rho(0.0, 0.0))
    if not rho00 > 0:
        raise AssertionError(("beam centre is not a regular point", spec["name"]))
    return Bexact, keys, rho00


def _bg_class_c():
    return {"name": "B1 class C: cos y + 0.3 i cos y sin(y+t) (nodes at y = +-pi/2)",
            "kappa": s.Integer(1), "theta": s.Integer(1), "gamma": s.Integer(0),
            "X1": lambda t, y: s.cos(y), "X2": lambda t, y: s.Rational(3, 10) * s.cos(y) * s.sin(y + t),
            "F": lambda u: u / (u + s.exp(-2 * s.Rational(3, 10) * u))**2,
            "V": lambda u: -(u - 1)**2 / 2}


def _bg_class_b():
    return {"name": "B2 class B (nodeless): 1 + 0.4 sin(y - 0.7t) + 0.5 i cos(2y + t)",
            "kappa": s.Integer(1), "theta": s.Integer(4), "gamma": s.Integer(-1),
            "X1": lambda t, y: 1 + s.Rational(2, 5) * s.sin(y - s.Rational(7, 10) * t),
            "X2": lambda t, y: s.Rational(1, 2) * s.cos(2 * y + t),
            "F": lambda u: u / (u + s.Rational(16, 25) * s.exp(-u))**2,     # eps = 4/5, alpha = 1/2
            "V": lambda u: u**2 / 2 - u**3 / 3}


def _bg_sampled(seed: int = 20261120):
    """Independently sampled smooth background and admissible triple (seeded, exact rationals)."""
    rng = np.random.default_rng(seed)

    def rat(lo, hi, den=40):
        return s.Rational(int(rng.integers(int(lo * den), int(hi * den) + 1)), den)

    kap, th = rat(0.5, 2.0), rat(0.5, 2.0)
    bound = 2 * s.sqrt(kap * th)
    gam = s.nsimplify(round(float(bound) * float(rng.uniform(-0.85, 0.85)), 2), rational=True)
    if not gam**2 < 4 * kap * th:
        raise AssertionError("sampled triple not admissible")
    amps = [rat(-0.15, 0.15) for _ in range(6)]
    freqs = [rat(-1.0, 1.0, 10) for _ in range(6)]
    eps2, alp = rat(0.3, 1.5), rat(0.1, 0.8)
    c3 = rat(-0.5, 0.5)

    def X1(t, y):
        return 1 + amps[0] * s.sin(y + freqs[0] * t) + amps[1] * s.cos(2 * y - freqs[1] * t) + amps[2] * s.sin(3 * y + freqs[2] * t + 1)

    def X2(t, y):
        return amps[3] * s.cos(y - freqs[3] * t) + amps[4] * s.sin(2 * y + freqs[4] * t) + amps[5] * t * s.cos(y)

    return {"name": f"B3 sampled (seed {seed}): kappa={kap}, theta={th}, gamma={gam}",
            "kappa": kap, "theta": th, "gamma": gam, "X1": X1, "X2": X2,
            "F": lambda u: u / (u + eps2 * s.exp(-2 * alp * u))**2,
            "V": lambda u: u**2 / 2 + c3 * u**3}


def _wkb_run(Bexact, keys, kapP, thP, gamP, w, Kt=12, J=4, Kg=10.0, Dy=64, RS=0.25,
             Ns=(400, 800, 1600, 3200), exact_eikonal=True, transport=True, grid=4001, half=0.4,
             tsamples=7):
    """Localized beam v_N = e^{N phi} A_N (cutoff identically 1 on |y| <= half; the beam is
    exp(-N half^2/2)-small there, reported as edge/peak) for the operator
    P(d) I2 + sum_{|alpha|<=3} B_alpha d^alpha, with P from (kapP, thP, gamP) and root w."""
    Dt = Kt + 6
    rt, ry, Mt, My = 0.4, 0.6, 64, 128
    tt = rt * np.exp(2j * np.pi * np.arange(Mt) / Mt)[:, None]
    yy = ry * np.exp(2j * np.pi * np.arange(My) / My)[None, :]
    Bc = Bexact(tt, yy)
    scale = (rt**np.arange(Dt))[:, None] * (ry**np.arange(Dy))[None, :]
    Bser = {ab: np.fft.fft2(v, axes=(-2, -1))[..., :Dt, :Dy] / (Mt * My) / scale * (RS**np.arange(Dy))
            for ab, v in Bc.items()}
    jet_err = 0.0
    for (tv, yv) in [(0.1, 0.1), (0.2, -0.15), (0.05, 0.25)]:
        ex = Bexact(np.array(tv), np.array(yv))
        for ab in keys:
            approx = np.einsum("ikjl,j,l->ik", Bser[ab], tv**np.arange(Dt), (yv / RS)**np.arange(Dy))
            jet_err = max(jet_err, float(np.max(np.abs(approx - ex[ab])) / max(1.0, np.max(np.abs(ex[ab])))))

    def pmul(p, q):
        return convolve2d(p, q)[:Dt, :Dy]

    def ymul(p, q):
        return np.convolve(p, q)[:Dy]

    def dtp(p):
        out = np.zeros_like(p)
        out[..., :-1, :] = p[..., 1:, :] * np.arange(1, Dt)[:, None]
        return out

    def dyp(p):
        out = np.zeros_like(p)
        out[..., :-1] = p[..., 1:] * np.arange(1, Dy) / RS
        return out

    def smul(p, v):
        return np.array([pmul(p, v[i]) for i in range(v.shape[0])])

    def mvmul(Mx, v):
        return np.array([sum(pmul(Mx[i, k], v[k]) for k in range(2)) for i in range(2)])

    # phase phi = i y + w t - (y - i w t)^2/2: phi_t = w + i w y + w^2 t, phi_y = i - y + i w t
    phit = np.zeros((Dt, Dy), complex)
    phit[0, 0], phit[0, 1], phit[1, 0] = w, 1j * w * RS, w**2
    phiy = np.zeros((Dt, Dy), complex)
    phiy[0, 0], phiy[0, 1], phiy[1, 0] = 1j, -RS, 1j * w

    def add(o1, o2, cst=1.0):
        out = dict(o1)
        for p, S in o2.items():
            out[p] = out.get(p, 0) + cst * S
        return out

    def Dt_(o):
        out = {}
        for p, S in o.items():
            out[p] = out.get(p, 0) + dtp(S)
            out[p + 1] = out.get(p + 1, 0) + smul(phit, S)
        return out

    def Dy_(o):
        out = {}
        for p, S in o.items():
            out[p] = out.get(p, 0) + dyp(S)
            out[p + 1] = out.get(p + 1, 0) + smul(phiy, S)
        return out

    def Q(o):
        return add(Dy_(Dy_(o)), Dt_(Dt_(o)), -1.0)

    def Pop(o):
        qo, d2 = Q(o), Dy_(Dy_(o))
        return add(add({p: kapP * S for p, S in Q(qo).items()}, {p: gamP * S for p, S in Q(d2).items()}),
                   {p: thP * S for p, S in Dy_(Dy_(d2)).items()})

    def derivs(o):
        out = {}
        for a_ in range(4):
            cur = o
            for _ in range(a_):
                cur = Dt_(cur)
            for b_ in range(4 - a_):
                out[(a_, b_)] = cur
                cur = Dy_(cur)
        return out

    def Mop(o):
        tot = Pop(o)
        for ab, Do in derivs(o).items():
            tot = add(tot, {p: mvmul(Bser[ab], S) for p, S in Do.items()})
        return tot

    z0, z1 = s.symbols("z0 z1")
    Psym = kapP * (-z0**2 + z1**2)**2 + gamP * (-z0**2 + z1**2) * z1**2 + thP * z1**4

    def polyval(expr):
        out = np.zeros((Dt, Dy), complex)
        for (i, j), cf in s.Poly(s.expand(expr), z0, z1).terms():
            term = np.zeros((Dt, Dy), complex)
            term[0, 0] = complex(cf)
            for _ in range(i):
                term = pmul(phit, term)
            for _ in range(j):
                term = pmul(phiy, term)
            out += term
        return out

    V0, V1 = polyval(s.diff(Psym, z0)), polyval(s.diff(Psym, z1))
    cpr = 0.5 * (polyval(s.diff(Psym, z0, 2)) * w**2 + 2 * polyval(s.diff(Psym, z0, z1)) * (1j * w)
                 - polyval(s.diff(Psym, z1, 2)))
    Cm = np.zeros((2, 2, Dt, Dy), complex)
    for i in range(2):
        Cm[i, i] = cpr
    for a_ in range(4):
        mon = np.zeros((Dt, Dy), complex)
        mon[0, 0] = 1
        for _ in range(a_):
            mon = pmul(phit, mon)
        for _ in range(3 - a_):
            mon = pmul(phiy, mon)
        for i in range(2):
            for k in range(2):
                Cm[i, k] += pmul(mon, Bser[(a_, 3 - a_)][i, k])
    rng = np.random.default_rng(1)
    atest = rng.normal(size=(2, Dt, Dy)) + 1j * rng.normal(size=(2, Dt, Dy))
    atest *= (0.5**np.arange(Dt))[:, None] * (0.5**np.arange(Dy))[None, :]
    Mt3 = Mop({0: atest})
    Ta = smul(V0, dtp(atest)) + smul(V1, dyp(atest)) + mvmul(Cm, atest)
    valid = (slice(None), slice(0, Dt - 6), slice(0, Dy - 6))
    transport_consistency = float(np.max(np.abs((Ta - Mt3[3])[valid])))
    eikonal_coeff = float(np.max(np.abs(Mt3[4][valid])))

    def ydiv(num, den):
        q = np.zeros(Dy, complex)
        for kk in range(Dy):
            q[kk] = (num[kk] - np.dot(den[1:kk + 1], q[kk - 1::-1][:kk])) / den[0]
        return q

    def solve_T(h, a0):
        amp = np.zeros((2, Dt, Dy), complex)
        amp[:, 0, :] = a0
        for j in range(Kt):
            acc = h[:, j, :].copy()
            for i in range(j):
                acc -= np.array([ymul(V0[j - i], (i + 1) * amp[c, i + 1]) for c in range(2)])
            for i in range(j + 1):
                day = dyp(amp[:, i, :])
                acc -= np.array([ymul(V1[j - i], day[c]) for c in range(2)])
                acc -= np.array([sum(ymul(Cm[c, k, j - i], amp[k, i]) for k in range(2)) for c in range(2)])
            amp[:, j + 1, :] = np.array([ydiv(acc[c], (j + 1) * V0[0]) for c in range(2)])
        return amp

    e0 = np.zeros((2, Dy), complex)
    e0[0, 0] = 1.0
    amps = []
    if transport:
        for m in range(J + 1):
            h = np.zeros((2, Dt, Dy), complex)
            for k in range(m):
                Mk = Mop({0: amps[k]})
                p = 3 - m + k
                if p in Mk:
                    h -= Mk[p]
            amps.append(solve_T(h, e0 if m == 0 else np.zeros((2, Dy), complex)))
        full = {}
        for k, amp in enumerate(amps):
            full = add(full, {p - k: S for p, S in Mop({0: amp}).items()})
        chk = (slice(None), slice(0, Kt), slice(0, max(4, Dy - (J + 1) * (Kt + 4))))
        hierarchy = {f"N^{p}": float(np.max(np.abs(full[p][chk]))) for p in range(3 - J, 4)}
    else:
        amp = np.zeros((2, Dt, Dy), complex)
        amp[:, 0, :] = e0
        amps = [amp]
        hierarchy = {"note": "transport hierarchy deliberately not solved (negative control)"}

    def peval(S, tv, yg):
        return np.einsum("...jl,j,ln->...n", S, tv**np.arange(Dt), (yg[None, :] / RS)**np.arange(Dy)[:, None])

    yg = np.linspace(-half, half, grid)
    dyq = yg[1] - yg[0]

    def phase(tv):
        return 1j * yg + w * tv - (yg - 1j * w * tv)**2 / 2

    obj = {-k: amp for k, amp in enumerate(amps)}
    D = derivs(obj)
    Pd = Pop(obj)

    def l2(N, tv, Fv):
        return float(np.sqrt(np.sum(np.abs(np.exp(N * phase(tv)))**2 * np.sum(np.abs(Fv)**2, axis=0)) * dyq))

    rows = []
    for N in Ns:
        TN = Kg * math.log(N) / N
        res, rnd = [], []
        for tv in np.linspace(0.0, TN, tsamples):
            B = Bexact(np.array(tv), yg)
            r = sum(N**p * peval(S, tv, yg) for p, S in Pd.items() if p <= 3)
            for ab, Do in D.items():
                for p, S in Do.items():
                    r = r + N**p * np.einsum("ikn,kn->in", B[ab], peval(S, tv, yg))
            n4 = N**4 * peval(Pd[4], tv, yg) if 4 in Pd else 0 * r
            if exact_eikonal:
                rnd.append(l2(N, tv, n4))       # analytically zero: pure rounding, reported separately
            else:
                r = r + n4                       # analytically nonzero (wrong root): part of the residual
            res.append(l2(N, tv, r))
        sol = l2(N, TN, sum(N**(-k) * peval(amp, TN, yg) for k, amp in enumerate(amps)))
        data, cur = 0.0, obj
        for j in range(4):
            data += l2(N, 0.0, sum(N**p * peval(S, 0.0, yg) for p, S in cur.items()))
            cur = Dt_(cur)
        gauss = (math.pi / N)**0.25 * math.exp(N * TN * w.real + N * TN**2 * w.real**2 / 2)
        ph = np.abs(np.exp(N * phase(TN)))
        row = {"N": N, "T_N": TN, "|u(T_N)|_L2": sol, "sum_j |d_t^j u(0)|_L2": data,
               "sup_t |L u|_L2": max(res), "residual/|u(T_N)|": max(res) / sol,
               "|u(T_N)|/Gaussian_prediction": sol / gauss,
               "ratio |u(T_N)|/(data+residual)": sol / (data + max(res)),
               "edge/peak": float(max(ph[0], ph[-1]) / np.max(ph))}
        if exact_eikonal:
            row["rounding N^4 term / |u(T_N)| (excluded: analytically zero)"] = max(rnd) / sol
        if transport:
            # float64 floor of the retained terms: the hierarchy jets that vanish analytically,
            # evaluated at the beam scale (t <= T_N, |y| ~ N^{-1/2}) and multiplied by N^p
            floor = 0.0
            for p in range(3 - J, 4):
                cj = np.abs(full[p][:, :Kt, :4])
                wts = (TN**np.arange(Kt))[:, None] * ((N**-0.5 / RS)**np.arange(4))[None, :]
                floor = max(floor, float(N)**p * float(np.max(np.sum(cj * wts, axis=(1, 2)))))
            row["rounding floor of retained terms (estimate)"] = floor
        rows.append(row)
    logN = np.log([r_["N"] for r_ in rows])
    if len(rows) > 1:
        slope_res = float(np.polyfit(logN, np.log([r_["residual/|u(T_N)|"] for r_ in rows]), 1)[0])
        slope_data = float(np.polyfit(logN, np.log([r_["sum_j |d_t^j u(0)|_L2"] for r_ in rows]), 1)[0])
    else:                                   # single-N refinement runs: no slope
        slope_res = slope_data = float("nan")
    return {"coefficient_jet_error": jet_err, "transport_consistency": transport_consistency,
            "eikonal_N4_coefficient": eikonal_coeff, "hierarchy_max_coefficients": hierarchy,
            "rows": rows, "slope(residual/|u|) vs log N": slope_res, "slope(data) vs log N": slope_data,
            "parameters": {"K'": Kt, "J": J, "T_N": f"{Kg} log N / N", "jet degree Dy": Dy,
                           "grid": grid, "window": half, "w": [w.real, w.imag]}}


def _root(kap, th, gam):
    mu = complex(-(2 * kap + gam) / (2 * kap), math.sqrt(4 * kap * th - gam**2) / (2 * kap))
    return complex(np.sqrt(mu))


def _main_verdict(run, J):
    rows = run["rows"]
    ratios = [r_["ratio |u(T_N)|/(data+residual)"] for r_ in rows]
    return {
        "jet_error_ok": run["coefficient_jet_error"] < 1e-10,
        "transport_consistency_ok": run["transport_consistency"] < 1e-10,
        "eikonal_exact_ok": run["eikonal_N4_coefficient"] < 1e-10,
        "hierarchy_ok": max(run["hierarchy_max_coefficients"].values()) < 1e-5,
        # |u(T_N)| = Gaussian prediction x |A_N(T_N, 0)| with |A_N - e| <= C (t + 1/N)  (Lemma 6)
        "growth_matches_gaussian_within_T_N": all(abs(r_["|u(T_N)|/Gaussian_prediction"] - 1) <= r_["T_N"]
                                                  for r_ in rows)
        and abs(rows[-1]["|u(T_N)|/Gaussian_prediction"] - 1) < 1e-2,
        "residual_decays": run["slope(residual/|u|) vs log N"] < -0.5,
        "ratio_increasing": all(b_ > a_ for a_, b_ in zip(ratios, ratios[1:])),
        "ratio_exceeds_100_at_largest_N": ratios[-1] > 100,
        # P(grad phi) = 0 is an exact identity (I3, I4, Lean beam_eikonal); its float64 value is
        # machine-precision noise amplified by N^4 and is excluded from the analytic residual.
        "N4_term_is_rounding_scale": all(
            r_["rounding N^4 term / |u(T_N)| (excluded: analytically zero)"] < 1e-13 * r_["N"]**4 for r_ in rows),
    }


def wkb_end_to_end(full: bool = True) -> dict:
    """End-to-end check on three independently specified backgrounds, a refinement study in
    (J, jet degree, quadrature grid, N), and four negative controls."""
    out = {"label": "NUMERICAL (exact operator, Taylor-jet amplitudes, true residual with exact coefficients)"}
    specs = [_bg_class_c(), _bg_class_b(), _bg_sampled()] if full else [_bg_class_c()]
    Ns = (400, 800, 1600, 3200) if full else (400, 800, 1600)
    ops = {}
    for spec in specs:
        Bexact, keys, rho00 = _background_operator(spec)
        ops[spec["name"]] = (Bexact, keys, spec)
        kap, th, gam = (float(spec[k]) for k in ("kappa", "theta", "gamma"))
        w = _root(kap, th, gam)
        run = _wkb_run(Bexact, keys, kap, th, gam, w, Ns=Ns)
        run["verdict"] = _main_verdict(run, 4)
        run["rho0(0,0)"] = rho00
        run["principal part"] = "exact symbolic check passed: 2F(rho0)(kappa, -(2kappa+gamma), kappa+theta+gamma) I2"
        if not all(run["verdict"].values()):
            raise AssertionError(("end-to-end WKB", spec["name"], run["verdict"], run["rows"]))
        out[spec["name"]] = run
    if not full:
        return out
    # refinement on B1: analytic error N^{2-J} responds to J; jet degree and grid do not matter
    Bexact, keys, spec = ops[specs[0]["name"]]
    kap, th, gam = 1.0, 1.0, 0.0
    w = _root(kap, th, gam)
    refine = {}
    for J in (2, 3):
        r_ = _wkb_run(Bexact, keys, kap, th, gam, w, J=J, Ns=(800, 1600, 3200))
        refine[f"J={J}: slope(residual/|u|)"] = r_["slope(residual/|u|) vs log N"]
    base = _wkb_run(Bexact, keys, kap, th, gam, w, Ns=(1600,))
    alt_dy = _wkb_run(Bexact, keys, kap, th, gam, w, Dy=48, Ns=(1600,))
    alt_grid = _wkb_run(Bexact, keys, kap, th, gam, w, grid=8001, Ns=(1600,))
    b0 = base["rows"][0]
    refine["jet degree 64 -> 48: rel change of residual"] = abs(alt_dy["rows"][0]["sup_t |L u|_L2"] / b0["sup_t |L u|_L2"] - 1)
    refine["grid 4001 -> 8001: rel change of |u(T_N)|"] = abs(alt_grid["rows"][0]["|u(T_N)|_L2"] / b0["|u(T_N)|_L2"] - 1)
    refine["grid 4001 -> 8001: rel change of residual"] = abs(alt_grid["rows"][0]["sup_t |L u|_L2"] / b0["sup_t |L u|_L2"] - 1)
    refine["J=4 (main run): slope(residual/|u|)"] = out[specs[0]["name"]]["slope(residual/|u|) vs log N"]
    refine["predicted slopes"] = "2 - J (the truncated WKB orders), approached from above as N grows"
    if not (refine["J=2: slope(residual/|u|)"] > refine["J=3: slope(residual/|u|)"] >
            refine["J=4 (main run): slope(residual/|u|)"]):
        raise AssertionError(("residual does not respond to J as predicted", refine))
    if not (refine["jet degree 64 -> 48: rel change of residual"] < 1e-3 and
            refine["grid 4001 -> 8001: rel change of |u(T_N)|"] < 1e-8):
        raise AssertionError(("refinement", refine))
    out["refinement (B1)"] = refine
    # negative controls on B1: each must FAIL to produce a contradiction
    controls = {}
    Nc = (400, 800, 1600, 3200)
    c1 = _wkb_run(Bexact, keys, kap, th, gam, 1.05 * w, Ns=Nc, exact_eikonal=False)
    c2 = _wkb_run(Bexact, keys, kap, th, gam, -w.conjugate(), Ns=Nc)
    c3 = _wkb_run(Bexact, keys, kap, th, gam, w, Ns=Nc, transport=False, J=0)
    hk, ht, hg = 1.0, 1.0, 3.0                       # strictly hyperbolic, NOT admissible
    muh = (-(2 * hk + hg) + math.sqrt(hg**2 - 4 * hk * ht)) / (2 * hk)
    wh = complex(0.0, math.sqrt(-muh))                 # purely imaginary root: no growth
    c4 = _wkb_run(Bexact, keys, hk, ht, hg, wh, Ns=Nc)
    for name, c in [("C1 wrong root 1.05 w (P(grad phi) != 0)", c1),
                    ("C2 decaying root -conj(w)", c2),
                    ("C3 transport hierarchy not solved", c3),
                    ("C4 strictly hyperbolic principal part (kappa,theta,gamma)=(1,1,3)", c4)]:
        ratios = [r_["ratio |u(T_N)|/(data+residual)"] for r_ in c["rows"]]
        c["ratios"] = ratios
        c["contradiction_produced"] = bool(ratios[-1] > 100 and all(b_ > a_ for a_, b_ in zip(ratios, ratios[1:])))
        if c["contradiction_produced"]:
            raise AssertionError(("negative control produced a contradiction", name, ratios))
        controls[name] = {"ratios": ratios, "contradiction_produced": False,
                          "slope(residual/|u|)": c["slope(residual/|u|) vs log N"],
                          "|u(T_N)|/|u(0)|-scale": [r_["|u(T_N)|_L2"] for r_ in c["rows"]]}
    out["negative controls (B1 lower-order terms)"] = controls
    return out


# ---------------------------------------------------------------------------
# I8  exponent bookkeeping (exact rationals)
# ---------------------------------------------------------------------------
def exponent_bookkeeping() -> dict:
    """For Sobolev indices (s, sigma) choose K, K', J; verify the three exponent gaps are positive.
    ratio >= c * min(N^{K Re w - 3 - s - sigma}, N^{K' - 4 - s - sigma}/(K log N)^{K'}, N^{J - 2 - s - sigma})."""
    rew = Fraction(455, 1000)            # lower bound for Re w = 0.45508... at kappa = theta = 1, gamma = 0
    if not math.sqrt((math.sqrt(2) - 1) / 2) > float(rew):
        raise AssertionError("Re w lower bound")
    table = []
    for s_, sig in itertools.product([Fraction(0), Fraction(1, 2), Fraction(3), Fraction(17, 2)],
                                     [Fraction(0), Fraction(1), Fraction(5, 2)]):
        K = math.floor((3 + s_ + sig) / rew) + 1
        Kp = math.floor(4 + s_ + sig) + 1
        J = math.floor(2 + s_ + sig) + 1
        gaps = (K * rew - 3 - s_ - sig, Kp - 4 - s_ - sig, J - 2 - s_ - sig)
        if min(gaps) <= 0:
            raise AssertionError(("bookkeeping", s_, sig, gaps))
        table.append({"s": str(s_), "sigma": str(sig), "K": K, "K'": Kp, "J": J,
                      "gaps": [str(g) for g in gaps]})
    return {"rule": "K > (3+s+sigma)/Re w, K' > 4+s+sigma, J > 2+s+sigma", "table": table}


def run(full: bool = True) -> dict:
    receipt = {
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "sympy": s.__version__,
        "numpy": np.__version__,
        "I1_linearization_structure": [linearization_structure(1), linearization_structure(2)],
        "I2_hadamard_root": hadamard_root_simplicity(),
        "I3_exact_beam_1d": exact_beam_1d(),
        "I4_exact_beam_phase": exact_beam_phase(),
        "I5_conjugation": conjugation_expansion(),
        "I6_class_A_corotating": corotating_class_a(),
        "I8_bookkeeping": exponent_bookkeeping(),
    }
    if full:
        receipt["I7_end_to_end_wkb"] = wkb_end_to_end(full=True)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quick", action="store_true", help="skip the end-to-end numerical check I7")
    args = parser.parse_args()
    receipt = run(full=not args.quick)
    text = json.dumps(receipt, indent=2, sort_keys=True, default=str)
    if args.output:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
