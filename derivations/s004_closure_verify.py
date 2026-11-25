"""S004 foundational closure: independent symbolic and numerical verification.

Every check below re-derives its input from the corrected S004 density in
WCT_CORRECTED_VARIATIONAL_CLOSURE_DRAFT.md section 2. Nothing is fitted. The
analytic claims and their scope are in docs/s004_closure/. Exact (sympy)
checks raise AssertionError on failure; numerical checks are diagnostics or
adversarial searches and are labelled as such in the receipt.

Run:  python derivations/s004_closure_verify.py --output NEW/verification.json
Test: python -m unittest -v derivations.test_s004_closure
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import random
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as s
from scipy.integrate import solve_ivp
from sympy.calculus.euler import euler_equations

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "WCT_CORRECTED_VARIATIONAL_CLOSURE_DRAFT.md"


def _zero(expr) -> bool:
    return s.simplify(s.expand(expr)) == 0


# ---------------------------------------------------------------------------
# C1  principal symbol of the linearized operator (brute force, sympy EL)
# ---------------------------------------------------------------------------
def principal_symbol_check(spatial_dims: int) -> dict:
    t = s.Symbol("t", real=True)
    xs = s.symbols(f"y1:{spatial_dims + 1}", real=True)
    kap, th, gam = s.symbols("kappa theta gamma", real=True)
    Ff, Vf = s.Function("F"), s.Function("V")
    coords = (t, *xs)
    x1, x2 = s.Function("x1")(*coords), s.Function("x2")(*coords)
    rho = x1**2 + x2**2

    def lap(q):
        return sum(s.diff(q, y, 2) for y in xs)

    def box(q):
        return -s.diff(q, t, 2) + lap(q)

    curvature = sum(kap * box(q)**2 + th * lap(q)**2 + gam * box(q) * lap(q) for q in (x1, x2))
    kinetic = sum(-s.diff(q, t)**2 + sum(s.diff(q, y)**2 for y in xs) for q in (x1, x2))
    L_orig = kinetic - Vf(rho) + Ff(rho) * curvature
    eqs = euler_equations(L_orig, [x1, x2], list(coords))
    X1, X2 = s.Function("X1")(*coords), s.Function("X2")(*coords)
    e1, e2 = s.Function("e1")(*coords), s.Function("e2")(*coords)
    eps = s.Symbol("eps")
    sub = {x1: X1 + eps * e1, x2: X2 + eps * e2}
    T = s.Symbol("T")
    Ys = s.symbols(f"Y1:{spatial_dims + 1}")
    R = X1**2 + X2**2
    for i, eq in enumerate(eqs):
        lin = s.expand(s.diff((eq.lhs - eq.rhs).subs(sub), eps).subs(eps, 0).doit())
        for j, e in enumerate((e1, e2)):
            poly = 0
            for exps in itertools.product(range(5), repeat=spatial_dims + 1):
                if sum(exps) != 4:
                    continue
                d = s.diff(e, *[v for v, n in zip(coords, exps) for _ in range(n)])
                mono = T**exps[0]
                for Yv, n in zip(Ys, exps[1:]):
                    mono *= Yv**n
                poly += s.simplify(lin.coeff(d)) * mono
            Yq = sum(Yv**2 for Yv in Ys)
            Xq = T**2 - Yq
            expected = 2 * Ff(R) * (kap * Xq**2 - gam * Xq * Yq + th * Yq**2) if i == j else 0
            if not _zero(poly - expected):
                raise AssertionError(("principal symbol", spatial_dims, i, j, s.factor(poly - expected)))
    return {"spatial_dims": spatial_dims,
            "result": "fourth-order part of linearization = 2F(rho0)(k box^2 + g box lap + t lap^2) I2"}


# ---------------------------------------------------------------------------
# C2  exact symbol algebra used by the Lean modules
# ---------------------------------------------------------------------------
def symbol_algebra_checks() -> list[str]:
    k, th, g, X, Y, sig, sv, Q, P = s.symbols("kappa theta gamma X Y sigma s Q P", real=True)
    form = k * X**2 - g * X * Y + th * Y**2
    dens = k * Q**2 + th * P**2 + g * Q * P
    A, C = 2 * k + g, k + th + g
    checks = []

    def zero(name, e):
        if not _zero(e):
            raise AssertionError((name, s.factor(e)))
        checks.append(name)

    zero("symbolForm=curvatureDensity(X,-Y)", form - dens.subs({Q: X, P: -Y}))
    zero("completed_square", 4 * k * form - ((2 * k * X - g * Y)**2 + (4 * k * th - g**2) * Y**2))
    zero("homogeneous_factorization", form.subs(X, sig - Y) - (k * sig**2 - A * sig * Y + C * Y**2))
    zero("reduced_quadratic", form.subs(X, sig - Y) - Y**2 * (k * (sig / Y)**2 - A * (sig / Y) + C))
    zero("reduced_discriminant", A**2 - 4 * k * C - (g**2 - 4 * k * th))
    zero("reduced_completed_square",
         (2 * k * sv - A)**2 - (4 * k * (k * sv**2 - A * sv + C) + g**2 - 4 * k * th))
    thb = g**2 / (4 * k)
    zero("boundary_perfect_square", form.subs(th, thb) - k * (X - g / (2 * k) * Y)**2)
    zero("pd_witness", form.subs({X: g, Y: 2 * k}) - k * (4 * k * th - g**2))
    zero("hyperbolic_indefinite_witness", dens.subs({Q: g, P: -2 * k}) - k * (4 * k * th - g**2))
    return checks


# ---------------------------------------------------------------------------
# C3/C4  ellipticity and hyperbolicity classification: exact and adversarial
# ---------------------------------------------------------------------------
def _frac(rng, lo, hi, den=97):
    return Fraction(rng.randint(int(lo * den), int(hi * den)), den)


def ellipticity_adversarial(samples: int = 4000, seed: int = 20261116) -> dict:
    """Search for a counterexample to: admissible => symbol > 0 off the origin.
    Exact rational arithmetic; any nonpositive value would refute the theorem."""
    rng = random.Random(seed)
    worst = None
    tested = 0
    while tested < samples:
        k = _frac(rng, 0.001, 5)
        th = _frac(rng, 0.001, 5)
        g = _frac(rng, -6, 6)
        if k <= 0 or th <= 0 or g * g >= 4 * k * th:
            continue
        tau = _frac(rng, -20, 20, 1009)
        xi = [_frac(rng, -20, 20, 1009) for _ in range(3)]
        Yv = sum(c * c for c in xi)
        if tau == 0 and Yv == 0:
            continue
        Xv = tau * tau - Yv
        val = k * Xv * Xv - g * Xv * Yv + th * Yv * Yv
        if val <= 0:
            raise AssertionError(("ellipticity counterexample", k, th, g, tau, xi))
        norm4 = (tau * tau + Yv) ** 2
        ratio = val / norm4
        worst = ratio if worst is None or ratio < worst else worst
        tested += 1
    return {"exact_rational_samples": tested, "counterexamples": 0,
            "min_symbol_over_|(tau,xi)|^4": float(worst)}


def hyperbolicity_classification(samples: int = 3000, seed: int = 7) -> dict:
    """Compare the closed-form criterion with numerical roots of the full characteristic
    polynomial in tau (degree 4) at |xi| = 1."""
    rng = np.random.default_rng(seed)
    agree = 0
    counts = {"elliptic": 0, "strictly_hyperbolic": 0, "other": 0}
    for _ in range(samples):
        k = rng.uniform(0.05, 3.0)
        th, g = rng.uniform(-3, 3), rng.uniform(-6, 6)
        A, C = 2 * k + g, k + th + g
        # tau^4 coefficient k; polynomial in tau: k(t^2-1)^2 - g(t^2-1) + th
        coeffs = [k, 0.0, -(2 * k + g), 0.0, k + g + th]
        roots = np.roots(coeffs)
        real = np.abs(roots.imag) < 1e-9
        all_real_distinct = bool(np.all(real)) and len(np.unique(np.round(roots.real, 7))) == 4
        none_real = not np.any(real)
        crit_strict = (g * g > 4 * k * th) and A > 0 and C > 0
        disc = A * A - 4 * k * C
        crit_elliptic = (disc < 0) or (A < 0 and C > 0) or (C > 0 and A < 0)
        # elliptic iff no root s >= 0 of k s^2 - A s + C
        if disc < 0:
            crit_elliptic = True
        else:
            sr = [(A + math.sqrt(disc)) / (2 * k), (A - math.sqrt(disc)) / (2 * k)]
            crit_elliptic = all(v < 0 for v in sr)
        if crit_strict == all_real_distinct and crit_elliptic == none_real:
            agree += 1
        counts["strictly_hyperbolic" if crit_strict else ("elliptic" if crit_elliptic else "other")] += 1
        if g * g < 4 * k * th and th > 0 and not none_real:
            raise AssertionError(("admissible but real characteristic", k, th, g))
        if crit_strict and g * g < 4 * k * th:
            raise AssertionError(("strictly hyperbolic yet curvature definite", k, th, g))
    if agree != samples:
        raise AssertionError(("criterion disagrees with numerical roots", agree, samples))
    return {"samples": samples, "agreement": agree, "class_counts": counts}


# ---------------------------------------------------------------------------
# C5/C6  Hadamard roots: exact, with arbitrary lower-order terms, and H4 replay
# ---------------------------------------------------------------------------
def c_uv(k: float, th: float, g: float) -> float:
    a = (2 * k + g) / (2 * k)
    b = math.sqrt(4 * k * th - g * g) / (2 * k)
    return math.sqrt((math.hypot(a, b) - a) / 2)


def hadamard_exact() -> list[str]:
    k, th, g = s.symbols("kappa theta gamma", positive=True)
    r = s.Symbol("r", positive=True)  # r = sqrt(4 k th - g^2)
    mu = -(2 * k + g) / (2 * k) + s.I * r / (2 * k)
    expr = s.expand(k * (mu + 1)**2 + g * (mu + 1) + th)
    expr = s.simplify(expr.subs(r**2, 4 * k * th - g**2))
    if s.simplify(expr) != 0:
        raise AssertionError(("Hadamard ratio root", expr))
    lam, Yv = s.symbols("lambda Y")
    frozen = k * (lam**2 + Yv)**2 + g * (lam**2 + Yv) * Yv + th * Yv**2
    if not _zero(frozen.subs(lam**2, Yv * mu) - Yv**2 * (k * (mu + 1)**2 + g * (mu + 1) + th)):
        raise AssertionError("frozen symbol factorization")
    w = s.Symbol("w")
    return ["hadamardRatio_root", "frozen_symbol_factorization"]


def hadamard_lower_order(seed: int = 3) -> dict:
    """Adversarial: frozen 2x2 symbol F*p(lambda,k) I2 + random constant lower-order terms of order
    <= 3 in (lambda, i k). max Re(lambda)/k must converge to c_uv; lower-order terms cannot remove
    the Hadamard growth (Rouche)."""
    rng = np.random.default_rng(seed)
    results = []
    for trial in range(6):
        # Python floats: numpy-scalar * poly1d silently returns a coefficient array
        k, th = (float(v) for v in rng.uniform(0.2, 2.0, 2))
        g = float(rng.uniform(-0.9, 0.9)) * 2 * math.sqrt(k * th)
        F = float(rng.uniform(0.05, 2.0))
        target = c_uv(k, th, g)
        # random lower-order coefficients c[a][b][i][j] for lambda^a (ik)^b, a+b<=3
        lower = {(a, b): rng.normal(0, 3, (2, 2)) for a in range(4) for b in range(4) if a + b <= 3}
        row = {"kappa": k, "theta": th, "gamma": g, "F": F, "c_uv": target}
        for K in (10.0, 100.0, 1000.0):
            # det of 2x2 polynomial matrix in lambda -> degree 8 polynomial; build via numpy poly
            def entry(i, j):
                p = np.poly1d([0.0])
                if i == j:
                    # F[k(l^2+K^2)^2 + g(l^2+K^2)K^2 + th K^4]
                    l2 = np.poly1d([1.0, 0.0, K * K])
                    p = p + F * (k * l2 * l2 + g * K * K * l2 + th * K**4)
                for (a, b), c in lower.items():
                    coef = complex(c[i, j]) * (1j * K)**b
                    p = p + np.poly1d([coef] + [0.0] * a)
                return p
            det = entry(0, 0) * entry(1, 1) - entry(0, 1) * entry(1, 0)
            roots = np.roots(det.coeffs)
            row[f"maxRe/k@k={int(K)}"] = float(np.max(roots.real) / K)
        err = abs(row["maxRe/k@k=1000"] - target) / target
        if err > 2e-2:
            raise AssertionError(("lower order terms changed asymptotic growth", row))
        row["rel_err@1000"] = err
        results.append(row)
    return {"trials": results}


def h4_replay() -> dict:
    """Replay the repository H4 polynomial D(z,q) with mass and compare growth with c_uv."""
    K, G, T, m2 = 1.0, 0.0, 1.0, 0.0
    out = []
    for kk in (4.0, 16.0, 64.0, 256.0):
        q = kk * kk
        # D(z,q) = (z-q) - m2 - K(z-q)^2 + G q (z-q) - T q^2 ; z = omega^2 ; growth s = -i omega
        coeffs_z = [-K, 1 + 2 * K * q + G * q, -(q + m2 + K * q * q + G * q * q + T * q * q)]
        zs = np.roots(coeffs_z)
        growth = max(abs(np.sqrt(complex(z)).imag) for z in zs)
        out.append({"k": kk, "growth/k": growth / kk})
    target = c_uv(1.0, 1.0, 0.0)
    if abs(out[-1]["growth/k"] - target) > 1e-3:
        raise AssertionError(("H4 replay", out, target))
    return {"c_uv(K=T=1,G=0)": target, "table": out}


def kappa_to_zero_table() -> dict:
    th = 1.0
    rows = [{"kappa": k, "c_uv(kappa,theta=1,gamma=0)": c_uv(k, th, 0.0)} for k in (1.0, 0.1, 0.01, 1e-4, 1e-6)]
    vals = [r["c_uv(kappa,theta=1,gamma=0)"] for r in rows]
    if not all(a < b for a, b in zip(vals, vals[1:])):
        raise AssertionError(("c_uv not increasing as kappa -> 0", rows))
    # fixed angle phi: gamma = 2 sqrt(kappa theta) cos(phi)
    ang = []
    for phi in (0.3, 1.0, 2.0):
        ang.append({"phi": phi, "values": [c_uv(k, th, 2 * math.sqrt(k * th) * math.cos(phi)) for k in (1e-2, 1e-4, 1e-6)]})
    for a in ang:
        if not a["values"][0] < a["values"][1] < a["values"][2]:
            raise AssertionError(("fixed-angle c_uv not increasing", a))
    return {"gamma=0 ray": rows, "fixed_angle_rays": ang}


# ---------------------------------------------------------------------------
# C8-C11  nodal jets, F'(0), formal crossing series, dominant balance, brackets
# ---------------------------------------------------------------------------
def nodal_jet_checks() -> list[str]:
    t = s.Symbol("t", real=True)
    kap, Acoef = s.symbols("kappa A", real=True)
    Ff, Vf = s.Function("F"), s.Function("V")
    u = s.Symbol("u")
    names = []
    v0 = s.Matrix(s.symbols("v0_1 v0_2", real=True))
    a0 = s.Matrix(s.symbols("a0_1 a0_2", real=True))
    j0 = s.Matrix(s.symbols("j0_1 j0_2", real=True))
    s0 = s.Matrix(s.symbols("s0_1 s0_2", real=True))
    c0 = s.Matrix(s.symbols("c0_1 c0_2", real=True))
    X = v0 * t + a0 * t**2 / 2 + j0 * t**3 / 6 + s0 * t**4 / 24 + c0 * t**5 / 120
    rho = (X.T * X)[0]
    Fr = Ff(rho)
    vel, acc = X.diff(t), X.diff(t, 2)
    Fp = s.diff(Ff(u), u).subs(u, rho)
    Vp = s.diff(Vf(u), u).subs(u, rho)
    node = {Ff(0): 0}
    EL = kap * (Fr * acc).diff(t, 2) + acc - Vp * X + kap * Fp * (acc.T * acc)[0] * X
    Fp0 = s.Subs(s.Derivative(Ff(u), u), u, 0).doit()
    vv = (v0.T * v0)[0]
    el0 = (EL.subs(t, 0).subs(node)).applyfunc(s.simplify)
    if not all(_zero(el0[i] - (1 + 2 * kap * Fp0 * vv) * a0[i]) for i in range(2)):
        raise AssertionError(("EL at node", el0))
    names.append("EL_at_node=(1+2kF'(0)|v|^2)a")
    el1 = (EL.diff(t).subs(t, 0).subs(node).subs({a0[0]: 0, a0[1]: 0})).applyfunc(s.simplify)
    Vp0 = s.Subs(s.Derivative(Vf(u), u), u, 0).doit()
    if not all(_zero(el1[i] - ((1 + 6 * kap * Fp0 * vv) * j0[i] - Vp0 * v0[i])) for i in range(2)):
        raise AssertionError(("dEL at node", el1))
    names.append("dEL_at_node=(1+6kF'(0)|v|^2)j-V'(0)v")
    b0 = s.Matrix(s.symbols("b0_1 b0_2")); b1 = s.Matrix(s.symbols("b1_1 b1_2"))
    bt = b0 + b1 * t
    p2 = -Fr * (2 * kap * acc - Acoef * bt)
    p1 = 2 * vel - p2.diff(t)
    if not all(_zero(p2.subs(t, 0).subs(node)[i]) for i in range(2)):
        raise AssertionError("p2 at node")
    if not all(_zero((p1.subs(t, 0) - 2 * v0).subs(node)[i]) for i in range(2)):
        raise AssertionError("p1 at node")
    names += ["p2_at_node=0", "p1_at_node=2v"]
    eps, alpha = s.symbols("epsilon alpha", positive=True)
    Fs = u / (u + eps**2 * s.exp(-2 * alpha * u))**2
    if not _zero(s.diff(Fs, u).subs(u, 0) - 1 / eps**4):
        raise AssertionError("F'(0)")
    names.append("F'(0)=1/eps^4")
    return names


def crossing_series(order: int = 6) -> dict:
    """Real homogeneous sector: substitute x = v0 t + sum a_n t^n into
    kappa (F(x^2) x'')'' + x'' - x V'(x^2) + kappa x F'(x^2) x''^2 = 0 and solve one order at a
    time with truncated arithmetic. F(u) = c1 u + c2 u^2 + c3 u^3 + ... is the Taylor series of
    the S004 weight (c1 = F'(0) = 1/eps^4 is checked exactly in nodal_jet_checks). Each a_n
    (n >= 2) enters its first equation linearly with the coefficient
    n(n-1)(1 + kappa v0^2 n(n-1) c1), which never vanishes for kappa >= 0, c1 > 0, so the formal
    crossing solution is unique given (t0, v0)."""
    t, v0, kap = s.symbols("t v0 kappa", real=True)
    c1, c2, c3 = s.symbols("c1 c2 c3", real=True)
    # Taylor data of the unspecified potential derivative: V'(u) = d1 + d2 u + d3 u^2 / 2 + ...
    d1, d2, d3 = s.symbols("d1 d2 d3", real=True)
    Fc = [0, c1, c2, c3]

    def trunc(expr, N):
        expr = s.expand(expr)
        return sum(expr.coeff(t, i) * t**i for i in range(N + 1))

    def series_in(x2, coeffs, N):
        out, power = 0, 1
        for c in coeffs:
            out += c * power
            power = trunc(power * x2, N)
        return trunc(out, N)

    known = v0 * t
    solved = {}
    for n in range(2, order + 1):
        an = s.Symbol(f"a{n}")
        x = known + an * t**n
        N = n
        x2 = trunc(x**2, N)
        Fx = series_in(x2, Fc, N)
        Fpx = series_in(x2, [(j + 1) * Fc[j + 1] for j in range(len(Fc) - 1)], N)
        Vpx = series_in(x2, [d1, d2, d3 / 2], N)
        xdd = s.diff(x, t, 2)
        E = (kap * s.diff(trunc(Fx * xdd, N), t, 2) + xdd - trunc(x * Vpx, N)
             + kap * trunc(x * Fpx * trunc(xdd**2, N), N))
        E = s.expand(E)
        for i in range(n - 2):
            if s.cancel(E.coeff(t, i)) != 0:
                raise AssertionError(("lower order not solved", n, i))
        cn = s.expand(E.coeff(t, n - 2))
        lin = s.Poly(cn, an)
        if lin.degree() != 1:
            raise AssertionError(("nonlinear in leading unknown", n))
        lc = s.cancel(lin.coeffs()[0])
        if s.cancel(lc - n * (n - 1) * (1 + kap * v0**2 * n * (n - 1) * c1)) != 0:
            raise AssertionError(("leading coefficient", n, lc))
        val = s.cancel(-lin.coeffs()[1] / lin.coeffs()[0]) if len(lin.coeffs()) > 1 else s.Integer(0)
        solved[n] = val
        known = known + val * t**n
    if solved[2] != 0:
        raise AssertionError(("a2 must vanish", solved[2]))
    if s.cancel(solved[3] - d1 * v0 / (6 * (1 + 6 * kap * v0**2 * c1))) != 0:
        raise AssertionError(("a3", solved[3]))
    return {"order": order, "a2": "0", "a3 (d1 = V'(0), c1 = F'(0))": str(s.factor(solved[3])),
            "leading_coefficient_a_n": "n(n-1)(1 + kappa v0^2 n(n-1) c1) != 0 for kappa >= 0, c1 > 0"}


def dominant_balance() -> str:
    tau, p = s.symbols("tau p", positive=True)
    x = tau**p
    expr = s.diff(x**2 * s.diff(x, tau, 2), tau, 2) + x * s.diff(x, tau, 2)**2
    target = p * (p - 1)**2 * (10 * p - 6) * tau**(3 * p - 4)
    if not _zero(s.powsimp(expr - target, force=True)):
        raise AssertionError(("dominant balance", s.factor(expr)))
    return "x=tau^p: (x^2 x'')'' + x x''^2 = p(p-1)^2(10p-6) tau^(3p-4); singular exponent p = 3/5"


def nodal_brackets() -> dict:
    q, v, p1, p2 = s.symbols("q v p1 p2")
    coords, momenta = (q, v), (p1, p2)

    def pb(f, g):
        return sum(s.diff(f, c) * s.diff(g, m) - s.diff(f, m) * s.diff(g, c) for c, m in zip(coords, momenta))

    cons = [q, v, p2, p1 - 2 * v]
    M = s.Matrix(4, 4, lambda i, j: pb(cons[i], cons[j]))
    expected = s.Matrix([[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 2], [-1, 0, -2, 0]])
    if M != expected or M.det() != 1:
        raise AssertionError(("nodal bracket matrix", M))
    trans = [q, p2, p1 - 2 * v]
    Mt = s.Matrix(3, 3, lambda i, j: pb(trans[i], trans[j]))
    if Mt.det() != 0 or Mt.rank() != 2:
        raise AssertionError(("transient bracket", Mt))
    return {"invariant_set_bracket": str(M.tolist()), "det": 1, "transient_rank": 2, "transient_size": 3}


# ---------------------------------------------------------------------------
# C12  nonlinear second-harmonic forcing of the ghost branch
# ---------------------------------------------------------------------------
def harmonic_forcing() -> dict:
    t, z = s.symbols("t z", real=True)
    kap, th, gam, A = s.symbols("kappa theta gamma A", positive=True)
    f0, f1, f2, f3 = s.symbols("f0 f1 f2 f3", real=True)
    V0, v1, v2, v3, v4 = s.symbols("V0 v1 v2 v3 v4", real=True)

    def Fpoly(u):
        w = u - A**2
        return f0 + f1 * w + f2 * w**2 / 2 + f3 * w**3 / 6

    def Vpoly(u):
        w = u - A**2
        return V0 + v1 * w + v2 * w**2 / 2 + v3 * w**3 / 6 + v4 * w**4 / 24

    x = s.Function("x")(t, z)
    box = -s.diff(x, t, 2) + s.diff(x, z, 2)
    lap = s.diff(x, z, 2)
    Lphys = s.diff(x, t)**2 - s.diff(x, z)**2 + Vpoly(x**2) - Fpoly(x**2) * (kap * box**2 + th * lap**2 + gam * box * lap)
    E = euler_equations(Lphys, [x], [t, z])[0]
    E = E.lhs - E.rhs
    d = s.Symbol("delta")
    uf = s.Function("u")(t, z)
    Ed = s.expand(E.subs(x, A + d * uf).doit())
    E0, E1, E2 = (s.expand(Ed.coeff(d, n)) for n in range(3))
    if not _zero(E0 - 2 * A * v1):
        raise AssertionError(("background", E0))
    k, w = s.symbols("k omega", positive=True)
    mode = s.exp(s.I * k * z - s.I * w * t)
    lin = s.expand(s.simplify(E1.subs(v1, 0).subs(uf, mode).doit() / mode))
    Xs, Ys = w**2 - k**2, k**2
    form = kap * Xs**2 - gam * Xs * Ys + th * Ys**2
    if not _zero(lin - 2 * (Xs + 2 * A**2 * v2 - f0 * form)):
        raise AssertionError(("linear symbol", lin))
    Pp, Qq = s.symbols("P Q")
    u1 = (Pp + 1 / Pp) * (Qq + 1 / Qq) / 4

    def deriv(expr, nt, nz):
        out = 0
        for term in s.Add.make_args(s.expand(expr)):
            pw = term.as_powers_dict()
            out += term * (s.I * k * pw.get(Pp, 0))**nz * (s.I * w * pw.get(Qq, 0))**nt
        return out

    reps = {}
    for der in E2.atoms(s.Derivative):
        nt = sum(c for v_, c in der.variable_count if v_ == t)
        nz = sum(c for v_, c in der.variable_count if v_ == z)
        reps[der] = deriv(u1, nt, nz)
    src = s.expand(E2.subs(v1, 0).subs(reps).subs(uf, u1))
    c2 = s.expand(2 * src.coeff(Pp, 2))
    sig0 = s.expand(c2.coeff(Qq, 0))
    sig2 = s.expand(2 * c2.coeff(Qq, 2))
    for n in (1, -1, 3, -3):
        if not _zero(c2.coeff(Qq, n)):
            raise AssertionError(("odd time harmonic", n))
    sig2_raw = A * (2 * A**2 * v3 - 9 * f1 * gam * k**4 + 9 * f1 * gam * k**2 * w**2 - 9 * f1 * k**4 * kap
                    - 9 * f1 * k**4 * th + 18 * f1 * k**2 * kap * w**2 - 9 * f1 * kap * w**4 + 3 * v2) / 2
    sig0_raw = A * (2 * A**2 * v3 - 9 * f1 * gam * k**4 + 5 * f1 * gam * k**2 * w**2 - 9 * f1 * k**4 * kap
                    - 9 * f1 * k**4 * th + 10 * f1 * k**2 * kap * w**2 - f1 * kap * w**4 + 3 * v2) / 2
    if not (_zero(sig2 - sig2_raw) and _zero(sig0 - sig0_raw)):
        raise AssertionError(("sigma polynomials differ from Lean definitions", sig0, sig2))
    if not _zero(sig2 - A / 2 * (3 * v2 + 2 * A**2 * v3 - 9 * f1 * form)):
        raise AssertionError("sigma2 factorization")
    # traveling ordinary wave cos(kz - wt): its cos(2kz - 2wt) source must equal 2 sigma2
    u_tw = (Pp / Qq + Qq / Pp) / 2
    reps_tw = {}
    for der in E2.atoms(s.Derivative):
        nt = sum(c for v_, c in der.variable_count if v_ == t)
        nz = sum(c for v_, c in der.variable_count if v_ == z)
        reps_tw[der] = deriv(u_tw, nt, nz)
    src_tw = s.expand(E2.subs(v1, 0).subs(reps_tw).subs(uf, u_tw))
    c_tw = s.expand(2 * src_tw.coeff(Pp, 2).coeff(Qq, -2))
    if not _zero(c_tw - 2 * sig2_raw):
        raise AssertionError(("traveling-wave coupling", c_tw))
    return {"sigma0": str(sig0_raw), "sigma2": str(sig2_raw),
            "sigma2_factor": "A/2 (3 v2 + 2 A^2 v3 - 9 f1 symbolForm(w^2-k^2, k^2))",
            "traveling_wave_second_harmonic": "= 2 sigma2 (no standing/traveling cancellation)"}


def s004_weight_values(rho0: Fraction, eps2: Fraction):
    """F and F' of the actual corrected weight at alpha = 0 (exact rationals)."""
    D = rho0 + eps2
    return rho0 / D**2, (eps2 - rho0) / D**3


def _poles(K, G, T, m2):
    """Exact pole formulas of the repository Q6/Q7 amplitude dispersion (R18.inversePropagator)."""
    def disc(q):
        return 1 - 4 * K * m2 + 2 * G * q - (4 * K * T - G**2) * q**2

    def zm(q):
        return q + (1 + G * q - s.sqrt(disc(q))) / (2 * K)

    def zp(q):
        return q + (1 + G * q + s.sqrt(disc(q))) / (2 * K)
    return disc, zm, zp


def resonance_band_classification(samples: int = 400, seed: int = 5) -> dict:
    """gamma = 0: z_+(4q) - 4 z_-(q) = (sqrt D(4q) + 4 sqrt D(q) - 3)/(2K). With d = 1 - 4 K m^2 the
    unique candidate is B q^2 = d - ((3+5d)/8)^2, valid iff 9/25 <= d <= 3/5. Exact identities plus an
    adversarial numerical scan of the resonance function off and on the band."""
    d = s.Symbol("d", positive=True)
    sstar = d - ((3 + 5 * d) / 8)**2
    if not (_zero(d - sstar - ((3 + 5 * d) / 8)**2) and _zero(d - 16 * sstar - ((3 - 5 * d) / 2)**2)):
        raise AssertionError("band identities")
    if not _zero(sstar - (-(25 * d - 9) * (d - 1) / 64)):
        raise AssertionError("band factorization")
    Kq, Tq, m2q, qv = s.symbols("K T m2 q", positive=True)
    disc, zm, zp = _poles(Kq, 0, Tq, m2q)
    if not _zero(2 * Kq * (zp(4 * qv) - 4 * zm(qv)) - (s.sqrt(disc(4 * qv)) + 4 * s.sqrt(disc(qv)) - 3)):
        raise AssertionError("resonance gap formula")
    rng = np.random.default_rng(seed)
    inside_hits, outside_hits = 0, 0
    for _ in range(samples):
        K, T = rng.uniform(0.02, 2.0), rng.uniform(0.02, 2.0)
        dval = rng.uniform(0.05, 0.99)
        m2 = (1 - dval) / (4 * K)
        B = 4 * K * T
        qmax = math.sqrt(dval / (16 * B))  # (2k)^2 = 4q must stay in the window: 16 B q^2 <= d
        qs = np.linspace(0, qmax, 4001)
        g = np.sqrt(np.maximum(dval - 16 * B * qs**2, 0)) + 4 * np.sqrt(dval - B * qs**2) - 3
        has_root = bool(np.any(g[:-1] * g[1:] <= 0))
        in_band = 9 / 25 <= dval <= 3 / 5
        if in_band != has_root and min(abs(dval - 9 / 25), abs(dval - 3 / 5)) > 1e-3:
            raise AssertionError(("band classification violated", K, T, dval))
        inside_hits += in_band and has_root
        outside_hits += (not in_band) and has_root
    return {"closed_form": "B q*^2 = d - ((3+5d)/8)^2, sqrt D(q*) = (3+5d)/8, sqrt D(4q*) = (3-5d)/2",
            "band": "9/25 <= d <= 3/5 (d = 1 - 4 K m^2)", "random_samples": samples,
            "in_band_with_root": int(inside_hits), "out_of_band_with_root": int(outside_hits)}


def resonance_certificate() -> dict:
    """Exact closed-form certificate (no interval arithmetic needed) with the actual S004 weight
    F(u) = u/(u + eps^2 exp(-2 alpha u))^2 at alpha = 0, eps = 1, rho0 = 1/2."""
    R = s.Rational
    rho0 = R(1, 2)
    uu = s.Symbol("u")
    Fw = uu / (uu + s.exp(-2 * 0 * uu))**2
    f0, f1 = Fw.subs(uu, rho0), s.diff(Fw, uu).subs(uu, rho0)
    if (f0, f1) != (R(2, 9), R(4, 27)):
        raise AssertionError(("weight values", f0, f1))
    kap, th, gam = R(1, 2), R(1), R(0)
    v2 = R(-459, 400)
    K, G, T, m2 = f0 * kap, f0 * gam, f0 * th, -2 * rho0 * v2
    dval = 1 - 4 * K * m2
    B = 4 * K * T - G**2
    if dval != R(49, 100) or B != R(8, 81):
        raise AssertionError(("d, B", dval, B))
    sst = dval - ((3 + 5 * dval) / 8)**2
    k4 = sst / B
    disc, zm, zp = _poles(K, G, T, m2)
    q = s.sqrt(k4)
    D1, D2 = disc(q), disc(4 * q)
    if (s.nsimplify(D1), s.nsimplify(D2)) != (R(109, 160)**2, R(11, 40)**2):
        raise AssertionError(("discriminants", D1, D2))
    if not _zero(zp(4 * q) - 4 * zm(q)):
        raise AssertionError("resonance")
    kstar = s.N(k4**R(1, 4), 30)
    if not (R(1, 100) <= kstar <= R(37, 50)):
        raise AssertionError(("window", kstar))
    if not (zm(4 * q) < zp(4 * q)):
        raise AssertionError("branch order")
    X = s.nsimplify(zm(q) - q)
    p = s.nsimplify(kap * X**2 + th * k4)
    if X != R(459, 320) or p != R(4131, 3200) or f0 * p != X - m2:
        raise AssertionError(("on-shell data", X, p))
    A = s.sqrt(rho0)
    sigma2 = s.nsimplify(A / 2 * (3 * v2 - 9 * f1 * p))
    if not _zero(sigma2 + 4131 * A / 1600):
        raise AssertionError(("sigma2", sigma2))
    v3c = s.solve(s.Eq(3 * v2 + 2 * rho0 * s.Symbol("v3") - 9 * f1 * p, 0))[0]
    if v3c != R(4131, 800):
        raise AssertionError(("cancellation surface", v3c))
    w2 = zm(q)
    sigma0 = A * (3 * v2 - f1 * (kap * w2**2 - 10 * kap * q * w2 + 9 * (kap + th) * q**2)) / 2
    D3 = s.nsimplify(disc(9 * q))
    if D3 != R(-41159, 25600):
        raise AssertionError(("third harmonic discriminant", D3))
    z3 = 9 * q + (1 + s.I * s.sqrt(-D3)) / (2 * K)
    growth3 = s.im(s.sqrt(s.N(z3, 40)))
    return {"parameters": {"alpha": 0, "eps": 1, "rho0": "1/2", "kappa": "1/2", "theta": 1, "gamma": 0,
                           "V'(rho0)": 0, "V''(rho0)": str(v2), "V'''(rho0)": 0,
                           "F(rho0)": str(f0), "F'(rho0)": str(f1), "K": str(K), "T": str(T), "m^2": str(m2)},
            "delta^2": str(dval), "B": str(B), "s*=B k*^4": str(sst), "k*^4": str(k4),
            "k*": str(kstar), "D(k*^2)": "(109/160)^2", "D(4k*^2)": "(11/40)^2",
            "resonance": "z_+(4k*^2) = 4 z_-(k*^2) exactly; resonant 2k* frequency is the upper (ghost) pole",
            "omega*^2 - k*^2": str(X), "symbolForm": str(p),
            "sigma2(V'''=0)": "-4131 sqrt(1/2)/1600", "sigma0(V'''=0)": str(s.N(sigma0, 15)),
            "cancellation_surface": "sigma2 = 0 iff V'''(rho0) = 4131/800",
            "third_harmonic": {"D(9k*^2)": str(D3), "Im omega(3k*)": str(s.N(growth3, 12)),
                               "status": "outside real window: complex frequencies (Hadamard band)"}}


def failed_original_candidate() -> dict:
    """Preserved failure: the earlier certificate used V''(rho0) = -1/2 and bracket k in [1/100, 1].
    It is refuted, not merely mis-bracketed: d = 7/9 > 3/5 so no resonance exists anywhere."""
    R = s.Rational
    K, T = R(1, 9), R(2, 9)
    m2 = -2 * R(1, 2) * R(-1, 2)
    dval = 1 - 4 * K * m2
    B = 4 * K * T
    disc_at_k1 = dval - B * 4**2  # q = (2k)^2 = 4 at k = 1
    qs = np.linspace(0, math.sqrt(float(dval) / (16 * float(B))), 20001)
    g = np.sqrt(np.maximum(float(dval) - 16 * float(B) * qs**2, 0)) + 4 * np.sqrt(float(dval) - float(B) * qs**2) - 3
    return {"label": "REFUTED (preserved failed certificate)", "V''(rho0)": "-1/2", "d": str(dval),
            "D((2k)^2) at k=1": str(disc_at_k1), "reason_bracket": "k = 1 puts 2k outside the real window",
            "reason_global": "d = 7/9 > 3/5: resonance function strictly positive on the whole window",
            "min_resonance_function_numeric": float(np.min(g))}


def unified_tangency() -> dict:
    """Skinner-Rusk unified formalism at one spatial point: coefficient of the jerk in
    p1 = dL/dv - d/dt dL/da is 2 kappa F I (= -d^2L/da^2), and at a nodal instant p1 = 2 v."""
    t = s.Symbol("t", real=True)
    kap, th, gam = s.symbols("kappa theta gamma", real=True)
    Ff, Vf = s.Function("F"), s.Function("V")
    q = s.Matrix([s.Function("q1")(t), s.Function("q2")(t)])
    b = s.Matrix([s.Function("b1")(t), s.Function("b2")(t)])
    v, a = q.diff(t), q.diff(t, 2)
    A, C = 2 * kap + gam, kap + th + gam
    rho = (q.T * q)[0]
    asym, vsym = s.Matrix(s.symbols("A1 A2")), s.Matrix(s.symbols("W1 W2"))
    back = {asym[0]: a[0], asym[1]: a[1], vsym[0]: v[0], vsym[1]: v[1]}
    L = (vsym.T * vsym)[0] + Vf(rho) - Ff(rho) * (kap * (asym.T * asym)[0] - A * (asym.T * b)[0] + C * (b.T * b)[0])
    dLda = s.Matrix([s.diff(L, asym[i]) for i in range(2)]).subs(back)
    dLdv = s.Matrix([s.diff(L, vsym[i]) for i in range(2)]).subs(back)
    p1 = dLdv - dLda.diff(t)
    jerk = q.diff(t, 3)
    Jac = s.Matrix(2, 2, lambda i, k: s.diff(p1[i], jerk[k]))
    if not all(_zero(Jac[i, k] - 2 * kap * Ff(rho) * int(i == k)) for i in range(2) for k in range(2)):
        raise AssertionError(("jerk coefficient", Jac))
    v0, a0, j0 = (s.Matrix(s.symbols(f"{n}1 {n}2")) for n in ("v0", "a0", "j0"))
    b0, b1 = s.Matrix(s.symbols("b01 b02")), s.Matrix(s.symbols("b11 b12"))
    path = {q[i]: v0[i] * t + a0[i] * t**2 / 2 + j0[i] * t**3 / 6 for i in range(2)}
    path.update({b[i]: b0[i] + b1[i] * t for i in range(2)})
    p1node = p1.subs(path).doit().subs(t, 0).subs(Ff(0), 0)
    if not all(_zero(p1node[i] - 2 * v0[i]) for i in range(2)):
        raise AssertionError(("p1 at node", p1node))
    return {"jerk_coefficient": "2 kappa F(rho) I2 (vanishes exactly on the nodal stratum)",
            "nodal_value": "p1 = 2 v for every a, jerk, b, bdot"}


def collision_indicial() -> dict:
    """Linearize the leading nodal balance (x^2 x'')'' + x x''^2 = 0 at x0 = C tau^(3/5)."""
    tau, mu, C = s.symbols("tau mu C", positive=True)
    p = s.Rational(3, 5)
    x0 = C * tau**p
    eta = tau**(p + mu)

    def d2(f):
        return s.diff(f, tau, 2)
    lin = d2(2 * x0 * d2(x0) * eta + x0**2 * d2(eta)) + eta * d2(x0)**2 + 2 * x0 * d2(x0) * d2(eta)
    P = s.expand(s.simplify(lin / (C**2 * tau**(3 * p + mu - 4))))
    target = mu * (mu + 1) * (5 * mu - 8) * (5 * mu - 3) / 25
    if not _zero(P - target):
        raise AssertionError(("indicial polynomial", s.factor(P)))
    ratio = s.simplify(d2(x0) / d2(x0**2 * d2(x0)))
    return {"indicial_polynomial": "mu (mu+1) (5 mu - 8) (5 mu - 3) / 25",
            "exponents": {"time_translation": -1, "amplitude_scaling": 0, "nontrivial": ["3/5", "8/5"]},
            "neglected_ordinary_terms_relative_size": str(ratio),
            "status": "FORMAL: 4-parameter (open) family of self-similar collisions at leading order"}


# ---------------------------------------------------------------------------
# C13  near-node dynamics (diagnostic) and spatial member check
# ---------------------------------------------------------------------------
def node_collision_diagnostic(seed: int = 11) -> dict:
    kap, m2 = 1.0, 2.0

    def F(u):
        return u / (u + 1.0)**2

    def Fp(u):
        return (1.0 - u) / (u + 1.0)**3

    def rhs(t, zz):
        x, v, p1, p2 = zz
        uu = x * x
        f = F(uu)
        fx = 2 * x * Fp(uu)
        return [v, -p2 / (2 * kap * f), -p2**2 * fx / (4 * kap * f**2) - m2 * x, 2 * v - p1]

    def H(zz):
        x, v, p1, p2 = zz
        return p1 * v - p2**2 / (4 * kap * F(x * x)) - v * v + m2 / 2 * x * x

    rng = np.random.default_rng(seed)
    rows = []
    for trial in range(5):
        x0, v0 = 1.0, rng.uniform(-0.3, 0.3)
        a0, j0 = rng.uniform(-2, 2), rng.uniform(-1, 1)
        uu = x0 * x0
        f = F(uu)
        fdot = Fp(uu) * 2 * x0 * v0
        z0 = [x0, v0, 2 * v0 + 2 * kap * (fdot * a0 + f * j0), -2 * kap * f * a0]
        sol = solve_ivp(rhs, (0, 30), z0, method="DOP853", rtol=1e-12, atol=1e-14, max_step=0.002)
        ts, xs = sol.t, sol.y[0]
        tstar = ts[-1]
        sel = [i for i in range(len(ts)) if 1e-6 < tstar - ts[i] < 1e-3 and abs(xs[i]) > 0]
        if sol.status != -1 or len(sel) < 10:
            rows.append({"trial": trial, "status": int(sol.status), "note": "no collision within horizon"})
            continue
        lt = np.log(tstar - ts[sel])
        lx = np.log(np.abs(xs[sel]))
        slope = float(np.polyfit(lt, lx, 1)[0])
        drift = float(abs(H(sol.y[:, sel[-1]]) - H(z0)) / max(1.0, abs(H(z0))))
        rows.append({"trial": trial, "t_star": float(tstar), "fitted_exponent": slope,
                     "relative_energy_drift_near_collision": drift})
    exps = [r["fitted_exponent"] for r in rows if "fitted_exponent" in r]
    if not exps or max(abs(e - 0.6) for e in exps) > 0.05:
        raise AssertionError(("collision exponent", rows))
    return {"label": "NUMERICALLY SUPPORTED (diagnostic)", "predicted_exponent": 0.6, "trials": rows}


def spatial_member_checks() -> list[str]:
    lam, k, th, F, m2 = s.symbols("lambda k theta F m2", positive=True)
    # kappa = gamma = 0 member, physical orientation: lambda^2 = -(k^2 + m2 + theta F k^4)
    roots = s.solve(s.Eq(lam**2 + k**2 + m2 + th * F * k**4, 0), lam)
    if not all(s.re(r) == 0 for r in roots):
        raise AssertionError(("spatial member growth", roots))
    return ["spatial_member_purely_oscillatory_frozen_symbol"]


def run() -> dict:
    return {
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "sympy": s.__version__, "numpy": np.__version__},
        "C1_principal_symbol": [principal_symbol_check(1), principal_symbol_check(2)],
        "C2_symbol_algebra": symbol_algebra_checks(),
        "C3_ellipticity_adversarial": ellipticity_adversarial(),
        "C4_hyperbolicity_classification": hyperbolicity_classification(),
        "C5_hadamard_exact": hadamard_exact(),
        "C5_hadamard_lower_order_adversarial": hadamard_lower_order(),
        "C6_H4_replay": h4_replay(),
        "C7_kappa_to_zero": kappa_to_zero_table(),
        "C8_nodal_jets": nodal_jet_checks(),
        "C9_crossing_series": crossing_series(),
        "C10_dominant_balance": dominant_balance(),
        "C11_nodal_brackets": nodal_brackets(),
        "C12_harmonic_forcing": harmonic_forcing(),
        "C12_resonance_band_classification": resonance_band_classification(),
        "C12_resonance_certificate": resonance_certificate(),
        "C12_failed_original_candidate": failed_original_candidate(),
        "C13_node_collision_diagnostic": node_collision_diagnostic(),
        "C13_collision_indicial": collision_indicial(),
        "C14_spatial_member": spatial_member_checks(),
        "C15_unified_tangency": unified_tangency(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = run()
    text = json.dumps(receipt, indent=2, sort_keys=True, default=str)
    if args.output:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
