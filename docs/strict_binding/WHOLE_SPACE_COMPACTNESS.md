# Whole-space compactness, minimizers, and the dynamical conclusion

Date: 2026-10-22. The functional, positive coefficients, full complex H²(R²)
space, and m_* are those of
[the subadditivity proof](SUBADDITIVITY_PROOF_OR_COUNTEREXAMPLE.md).

**Established:** for every 0<m<=m_*, every minimizing sequence is precompact
in H² modulo translations, the whole-space minimizer set is nonempty, its
members are smooth stationary profiles, and the full minimizer set is
orbitally stable under the unchanged Hamiltonian flow.

**Not established at m_D4:** the all-splits inequality, whole-space minimizer
existence, or whole-space orbital stability. The new theorem does not identify
a minimizer with the periodic D4 root.

## C1. Compactness from the actual variational hypotheses

Let m>0. Assume strict binding I(m)<I(a)+I(m-a) for every 0<a<m. The preceding
paper supplies finite I, continuity, H² coercivity of near-minimizers, and
I(m)<0. We give the full argument, rather than treating negativity as binding.

Take M(u_n)=m and H(u_n)->I(m). Coercivity bounds u_n in H². If

    sup_y integral_(B(y,1)) |u_n|² -> 0,

the local vanishing lemma implies ||u_n||_4->0. One can prove this by a
bounded-overlap unit-cube cover and the local two-dimensional H¹/L²
interpolation inequality, summing with the uniform H¹ bound. The uniform
H²-to-L-infinity bound then gives ||u_n||_6^6<=||u_n||_infinity²||u_n||_4^4->0.
Since T>=0, liminf H(u_n)>=0, contradicting I(m)<0. Vanishing is excluded.

Consequently, after a subsequence and translations y_n, there is a nonzero
u with u_n(.+y_n) weakly convergent in H², strongly convergent locally in
every finite L^p, and convergent almost everywhere. The nonzero local L² mass
is retained by Rellich compactness. Let a=M(u), so 0<a<=m, and put
v_n=u_n(.+y_n)-u. Hilbert splitting and the Brezis–Lieb lemma give

    M(v_n)=m-a+o(1),
    T(u_n)=T(u)+T(v_n)+o(1),
    Q(u_n)=Q(u)+Q(v_n)+o(1),
    S(u_n)=S(u)+S(v_n)+o(1),
    H(u_n)=H(u)+H(v_n)+o(1).

The two nonlinear powers are split separately; their opposite coefficients
do not invalidate either identity. The quadratic identity follows from
weak convergence Au_n to Au in L². These arguments are valid for complex
fields. Higher-frequency components with vanishing L² mass do not require
a separate nonlinear critical profile: H² boundedness controls L-infinity,
so their L⁴ and L⁶ norms vanish, and T is nonnegative.

If a<m, normalize v_n to mass m-a. Its amplitude factor tends to one, and
its H² norm is bounded, so the change in H is o(1). Therefore

    I(m) >= H(u)+I(m-a) >= I(a)+I(m-a),

a contradiction. Thus a=m. Now v_n->0 in L², and the bounded sup norm gives
Q(v_n),S(v_n)->0. Since H(u)>=I(m), the splitting gives T(v_n)->0 and
H(u)=I(m). Finally

    ||Delta v_n|| <= ||A v_n||+k²||v_n|| -> 0

and interpolation yields strong H² convergence. This proves the assertion
for every minimizing sequence after subsequence and translations, not only
for a specially chosen trial family. Global phase requires no further loss
of compactness because U(1) is compact; the minimizer set includes all phases.

S2 verifies all of these strict inequalities for 0<m<=m_*, including m=m_*.
This discharges PR #34's B4 conditional premise precisely in that range.

## C2. What profile decomposition says outside that range

The preceding extraction can be repeated on v_n if it has nonvanishing local
mass. Translations diverge between distinct profiles, masses and T decouple,
and the fourth and sixth powers decouple. This is the relevant translation
profile argument in this subcritical H² embedding setting. A positive-mass
remainder can instead spread and have nonnegative limiting energy.

Weak subadditivity makes the resulting lower-bound chain consistent with
equality. It does not turn that chain into a contradiction. In a minimizing
sequence, a nonzero extracted profile with mass a<m and an equality chain
would itself attain I(a), while the remainder approaches I(m-a). Knowing
that such profiles exist at some smaller masses does not establish strict
binding between them. This is the unresolved step for uncovered splits.

There is also a useful converse. **If I(m)=I(a)+I(m-a) for one 0<a<m, there
exists a dichotomizing minimizing sequence at mass m.** Choose compactly
supported smooth near-minimizers at the two exact masses with errors tending
to zero, and translate their supports apart by distances tending to infinity
and exceeding both support diameters. Their sums have mass exactly m and
energy tending to I(m). No choice of a single translation makes them tight:
any fixed ball eventually misses at least min(a,m-a) mass. They cannot have
a strongly L², hence strongly H², convergent translated subsequence.

Thus, in this setting with I(m)<0, all-split strict binding is equivalent to
precompactness of **all** minimizing sequences modulo translations. An
equality would not by itself prove that no minimizer exists: attainment can
coexist with other noncompact minimizing sequences. Nor does this argument
prove instability of every stationary solution.

## C3. Euler–Lagrange equation and regularity

For 0<m<=m_*, denote

    G_m={u in H²(R²;C): M(u)=m, H(u)=I(m)}.

It is nonempty and compact modulo translations. Since the real derivative
of M does not vanish at a nonzero u, the Lagrange multiplier theorem gives
a real omega with

    (Delta+k²)² u-beta|u|²u+gamma|u|⁴u=omega u

in H^(-2). There is no frequency fixed in advance. The operator A²+1 has a
strictly positive Fourier multiplier comparable to (1+|xi|²)². Write

    u=(A²+1)^(-1)[(omega+1)u+beta|u|²u-gamma|u|⁴u].

H²(R²) is a Banach algebra; the right-hand side initially belongs to H⁶.
Iteration gives H^s for every finite s and smoothness. H² also gives uniform
continuity; together with L² this implies u(x)->0 as |x|->infinity. No
particular exponential decay rate, positivity, radial symmetry, uniqueness,
nondegeneracy, or individual-orbit spectral theorem is asserted.

Testing the equation against u gives

    omega m = I(m)-beta Q/2+2gamma S/3.

F3 and H(u)=I(m)<0 imply S/Q<K_inf m<=3beta/(4gamma). Hence
omega m<I(m)<0 throughout the proved interval, including its endpoint.
These are finite-energy, spatially localized mathematical stationary states.
Their L² norm is not identified as a physical inertial mass.

## C4. Global evolution and full-set orbital stability

The evolution is exactly

    i u_t=A²u-beta|u|²u+gamma|u|⁴u.

For completeness, its required hypotheses are verified here. The group
exp(-it A²) is unitary on H². The polynomial nonlinearity is locally Lipschitz
H²->H² by the algebra property. The Duhamel contraction gives a unique
local H² mild solution, continuous dependence, and an H² blow-up alternative.
Smooth spectral approximations conserve M and H. Their local H² convergence
and continuity of the energy pass conservation to the mild solution.
The bound

    ||Au(t)||² <= H(u(0))+C M(u(0)),
    ||Delta u(t)|| <= ||Au(t)||+k²||u(0)||

excludes the blow-up alternative in both time directions. This reconstructs
the global H² result used in PR #34; it is a standard conservation argument.
The new contribution here is discharging the variational compactness premise
in a specific mass range, not a new general local existence method.

**Orbital stability theorem.** For each fixed 0<m<=m_* and epsilon>0, there
is delta>0 such that

    dist_H²(u0,G_m)<delta
      => sup_(t in R) dist_H²(u(t),G_m)<epsilon.

The perturbation is an arbitrary complex H² field; it need not have exactly
mass m. To prove this, suppose data approach G_m but at times t_n the distance
is bounded below by a fixed epsilon. All elements of G_m have uniformly
bounded H² norms by coercivity. Hence initial masses tend to m and energies
to I(m), and conservation gives the same limits at t_n. The evolved fields
remain uniformly H² bounded. Normalize their masses to m by factors tending
to one; this changes H and H² distance by o(1). They form a minimizing
sequence. C1 supplies strong H² convergence modulo translations to G_m,
contradicting the distance lower bound. This proof also works at m=m_*;
it does not require binding at neighboring masses.

The statement is stability of the entire minimizing set. It does not identify
one phase/translation orbit, prove attraction, or give a numerical basin.

## Domain and historical boundary

The periodic D4 root/coercivity/orbital-basin evidence remains an independent
computer-assisted theorem on the length-42 torus. Its exact coefficients,
norm, perturbation class, and rational replay are retained. No box-enlargement
limit, uniform tightness of that periodic branch, or whole-space identity of
the D4 state is proved. At its mass, S2 excludes only the central splits.

The analytic results use standard concentration compactness and Brezis–Lieb
splitting, with each needed estimate stated above. The direct proof avoids
importing a pure-power or positive-dispersion minimizer theorem into the
competing cubic–quintic functional. The source ledger records the supporting
literature and exact inherited source blobs.
