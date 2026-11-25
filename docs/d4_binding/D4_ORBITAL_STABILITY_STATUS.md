# Euler–Lagrange regularity and orbital stability

Date: 2026-10-23.

| Frozen D4 conclusion | Status | Scope |
|---|---|---|
| Whole-space minimizer existence | PROVED | Full complex H² class |
| Precompactness modulo translations | PROVED | Every minimizing sequence |
| Euler–Lagrange regularity | PROVED | Smooth profiles tending to zero at infinity |
| Orbital H² stability | PROVED | Entire minimizing set under the unchanged flow |

## O1. Equation and multiplier, after existence

The derivative of M at a nonzero minimizer is nonzero. Real variations of
both H and M carry the same factor 2, so stationarity of H-omega M gives

    (Delta+k²)²u-beta|u|²u+gamma|u|⁴u=omega u, omega real.

There is no extra factor 2 multiplying omega. The corresponding solution is
exp(-i omega t)u. No frequency is assigned from a periodic certificate.

The multiplier of A²+1 is positive and comparable to (1+|xi|²)². Thus

    u=(A²+1)^(-1)[(omega+1)u+beta|u|²u-gamma|u|⁴u].

H² is an algebra, so the right side lies in H⁶. Iteration gives all higher
Sobolev regularities and smoothness. Uniform continuity and L² imply decay
to zero at spatial infinity, without an asserted rate.

Minimizing the scalar dilation polynomial also gives (V), without requiring
a spatial moment. V2 and the equation tested against u yield

    omega*m=I(m)-beta Q/2+2gamma S/3
       <=I(m)-(beta/2)(1-m/m_B)Q<I(m)<0

at m=m_D4<m_B. Its value and agreement with the periodic frequency are open.

## O2. Global H² evolution

A² has a real symbol; exp(-it A²) is unitary on H². The cubic–quintic
polynomial is locally Lipschitz H²->H² by the algebra property. Duhamel
contraction gives unique local mild solutions, H² continuous dependence
and an H² blow-up alternative. This holds for arbitrary complex data.

For smooth solutions, self-adjoint integration by parts conserves M and H.
Smooth initial approximations persist locally; the tame product bound
controls higher norms on intervals with bounded H². Local H² continuous
dependence and continuity of M,H pass conservation to H² mild solutions.
Equivalently one may use Hamiltonian Fourier regularizations with projected
nonlinearities and then pass to the limit. No numerical scheme is a premise.

Conservation and coercivity imply for every time of existence

    T(u(t))<=H(u0)+C M(u0),
    ||Delta u(t)||<=sqrt(H(u0)+C M(u0))+k²sqrt(M(u0)).

The finite time-independent bound excludes blow-up in both directions.
The global H² flow, continuity and both conservation laws are therefore
established for the actual unchanged equation.

## O3. Stability of the full minimizing set

For each epsilon>0 there is delta>0 such that

    dist_H²(u0,G_m)<delta
       implies sup_(t in R)dist_H²(u(t),G_m)<epsilon.

Otherwise take initial data approaching G_m and times t_n at distance at
least epsilon. G_m is uniformly H²-bounded. Uniform continuity of M,H on
bounded sets makes the initial masses tend to m and energies to I(m).
Conservation gives the same limits at t_n; coercivity bounds these fields
in H². Normalize to mass m by factors tending to one. Both H² and energy
changes tend to zero, yielding a minimizing sequence. Compactness gives
strong H² convergence after translations to G_m, a contradiction since G_m
is translation invariant. Mass-changing complex perturbations are included.

This is full-set orbital stability, not uniqueness, stability of each
stationary branch, an isolated orbit, attraction, or spectral nondegeneracy.
The periodic basin certificate and foundational S004 obstruction remain
separate. No physical spectrum is exported.
