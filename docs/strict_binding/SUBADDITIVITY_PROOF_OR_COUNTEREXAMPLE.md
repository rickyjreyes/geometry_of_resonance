# Phase IV: subadditivity proof and its exact boundary

Date: 2026-10-22, America/Los_Angeles. Base: geometry PR #34,
`dff2463e4dae6c1ee4fe0c4ebc6125bf90c9f49e`.
**Primary classification: STRICT_BINDING_RESTRICTED.**

All propositions below concern the full complex space H²(R²; C). No radial,
real, Fourier-band, momentum, or square-symmetry constraint is imposed.
“Proved” denotes the analytic argument supplied here, not external peer review
or a proof-assistant certificate.

## F1. Reconstructing the unchanged functional

Write k=k_*>0 and beta,gamma>0. With Lebesgue measure dx, set

    A = Delta+k²,
    M(u) = integral |u|²,
    T(u) = ||Au||_2²,
    Q(u) = integral |u|⁴,  S(u) = integral |u|⁶,
    H(u) = T(u) - beta Q(u)/2 + gamma S(u)/3,
    I(m) = inf {H(u): u in H²(R²; C), M(u)=m},   I(0)=0.

This is I_2 in PR #34. The source `wct-pde/src/wct_pde/projection_free.py`
at `051b6eca10c46b8e038993d60cabb20715194a20` uses the same coefficients,
cell-area approximation to dx, and squared L² norm. Its generic configuration
defaults are not the D4 certificate parameters. The latter are exactly

    k=1,
    beta=gamma=float.fromhex('0x1.05f1d178c75ebp+2'),
    m_D4=float.fromhex('0x1.45b8a0bad2714p+1').

The hexadecimal strings denote exact dyadic rationals, not decimal fits.
The physical integral mass is m_D4=2.54469689485712...; it is not the mean
mass m_D4/42² used in intermediate periodic certificate estimates. There is
no 1/2 in T, and the k⁴M term has not been subtracted. The whole-space domain
is explicitly R²; no assertion that the length-42 torus tends to this problem
is used. The quantities called mass here are norms, not particle rest masses.

H²(R²) embeds continuously in every finite L^p, and in L-infinity; consequently
H is a continuous real C¹ polynomial functional. It is translation and global
phase invariant. Define D=||Delta u||² and K=||grad u||². Then

    T = D-2k²K+k⁴M >= 0,    K <= sqrt(M D).

The negative gradient coefficient does not make T negative. It does obstruct
some familiar dilation arguments.

For theta>0, amplitude scaling and the two spatial dilations give

    M(sqrt(theta)u)=theta M,
    H(sqrt(theta)u)=theta T-beta theta²Q/2+gamma theta³S/3;

    u_lambda(x)=lambda u(lambda x): M(u_lambda)=M,
    H(u_lambda)=lambda⁴D-2k²lambda²K+k⁴M
               -beta lambda²Q/2+gamma lambda⁴S/3;

    v_theta(x)=u(x/sqrt(theta)): M(v_theta)=theta M,
    H(v_theta)/theta=theta^(-2)D-2k²theta^(-1)K+k⁴M
                    -beta Q/2+gamma S/3.

These are identities for the unchanged coefficients, not parameter symmetries
of the equation. Spatial dilation also dilates a carrier's wave number.

## F2. Coercivity, negative infima, continuity, and weak subadditivity

Put C=3 beta²/(16 gamma). The pointwise square completion is

    -beta r²/2+gamma r³/3
      = (gamma/3) r(r-3beta/(4gamma))²-Cr,   r>=0.

Thus H>=T-CM. At fixed mass, bounded energy above implies bounded T and
bounded H², since ||Delta u|| <= ||Au||+k²||u||. In particular I(m)>-infinity.
Neither T alone nor the unconstrained H is asserted to control H².

Here is a fresh reconstruction of the negative-infimum argument. For a,b,R>0,

    f_ab=(pi a b)^(-1/2) exp[-x²/(2a²)-y²/(2b²)],
    u_R=sqrt(m) R^(-1) f_ab(x/R,y/R) exp(i k x).

It belongs to the full admissible space and has mass exactly m. Direct
integration, independently checked by quadrature, gives

    H(u_R)=c2/R²+c4/R⁴,
    c2=2mk²/a²-beta m²/(4pi a b),
    c4=m[3/(4a⁴)+1/(2a²b²)+3/(4b⁴)]
       +gamma m³/(9pi²a²b²)>0.

Taking a/b>8pi k²/(beta m) makes c2<0; R²=-2c4/c2 gives
H=-c2²/(4c4)<0. This verifies every hypothesis needed for **I(m)<0 for all
m>0**: two dimensions, beta>0, gamma>0, k>0, complex anisotropic profiles,
and the declared unconstrained spatial domain. No numerical optimizer is
needed. The computations use, for example, b=1 and a=1+16pi k²/(beta m).

**Continuity.** I is locally Lipschitz on (0,infinity). Fix a compact positive
mass interval. Its near-minimizers can be chosen with H<=1; the preceding
bound makes their H², T, Q and S uniformly bounded. Rescaling a near-minimizer
of mass a by sqrt(b/a) changes H by at most L|b-a| on this interval, by the
amplitude identity. Taking the infimum, then interchanging a and b, proves
the claim. At zero, -Cm<=I(m)<0 proves continuity with I(0)=0. This proof does
not presume attainment.

**Weak subadditivity.** For every a,b>0,

    I(a+b) <= I(a)+I(b).

Approximate near-minimizers in H² by smooth compactly supported functions and
renormalize their masses. Continuity of H justifies both steps. Translate
one sufficiently far that supports are disjoint. All local terms, including
the squared differential operator A, then add exactly. Take the two errors
to zero. This proves the weak inequality, not its strict version.

Since every positive-mass infimum is negative, weak subadditivity also gives
I(a+b)<I(a). **Strict decrease of I itself is not strict decrease of I(m)/m.**

## F3. An explicit Fourier estimate for this operator

**Lemma.** For every u in H²(R²; C),

    ||u||_infinity² <= J [T(u)+k⁴ M(u)],   J=3/(16k²).

**Proof.** Use u(x)=(2pi)^(-2) integral exp(i x.xi) uhat(xi) dxi and the
matching Parseval convention. Weighted Cauchy–Schwarz gives the right side
with constant

    (2pi)^(-2) integral_R² 1/[(|xi|²-k²)²+k⁴] dxi
      = (1/(4pi)) integral_0^infinity 1/[(s-k²)²+k⁴] ds
      = (1/(4pi k²)) [pi/2+pi/4] = 3/(16k²).

The weight is positive and comparable to (1+|xi|²)², so the calculation
applies first to Schwartz functions and then by H² density. It is valid for
complex u and does not use a diamagnetic or rearrangement inequality. QED.

Define

    K_inf = J(k⁴+C),
    m_* = 3beta/(4gamma K_inf)
        = 4beta k²/[gamma(k⁴+C)].

For any negative-energy field with M=a,

    S <= ||u||_infinity² Q <= [K_inf a+J H(u)] Q.

This is the estimate used on minimizing sequences. Retaining the negative
H term will also handle a boundary equality in the next theorem.

## S1. Strict decrease of energy per mass on an explicit two-mass range

**Theorem.** If 0<a<b and a+b<=2m_*, then

    I(b)/b < I(a)/a.

**Proof without assuming a minimizer.** Let u_n have mass a and H(u_n)->I(a)<0.
For sufficiently large n, H(u_n)<=I(a)/2<0. Consequently

    Q(u_n) >= -2H(u_n)/beta >= -I(a)/beta = q0>0,
    S(u_n)/Q(u_n) <= K_inf a+J I(a)/2.

The latter upper bound is nonnegative whenever such u_n exist, since it also
bounds their squared sup norm. Let theta=b/a>1. The exact amplitude identity is

    H(sqrt(theta)u_n)/theta-H(u_n)
      = (theta-1)[-beta Q(u_n)/2+gamma(theta+1)S(u_n)/3].

Set

    delta = beta/2 - gamma(theta+1)[K_inf a+J I(a)/2]/3.

The assumed a+b<=2m_* gives

    delta >= -gamma(theta+1)J I(a)/6 > 0.

It follows that

    I(b)/theta <= H(u_n)-(theta-1)delta q0.

Taking n to infinity proves a strict, sequence-uniform gap. This is not the
invalid inference that a strict inequality for each trial automatically
remains strict after taking an infimum. QED.

The special case 0<a<b<=m_* proves strict monotonicity of I(m)/m throughout
(0,m_*], including its upper endpoint.

## S2. Restricted strict binding, including a certified D4 split interval

**Theorem.** For m>0 and 0<mu<m, strict binding holds whenever

    m + max(mu,m-mu) <= 2m_*:
    I(m) < I(mu)+I(m-mu).

**Proof.** Write a=min(mu,m-mu), b=max(mu,m-mu). S1 applied to b<m gives
I(m)/m<I(b)/b. If a<b, S1 also applies to a,b because a+b=m<2m_*, so
I(a)/a>I(b)/b. If a=b the two ratios are equal. In either case

    I(a)+I(b) >= m I(b)/b > I(m).

No equality of shapes, real-valuedness, or minimizer existence was used. QED.

**Corollary.** For every 0<m<=m_* and every 0<mu<m, strict binding holds.
For m_*<m<=4m_*/3, S2 also certifies the central closed split interval

    2m-2m_* <= mu <= 2m_*-m.

This latter conclusion is a statement about particular splits; it is not
enough to establish compactness at that total mass.

At the frozen D4 coefficients,

    C = 13824534543425985/18014398509481984,
    J = 3/16,
    m_* = 72057594037927936/31838933052907969
        = 2.2631912293727647... .

The exact m_D4 lies strictly between m_* and 4m_*/3. The certified central
interval has endpoints **2m_D4-2m_*** and **2m_*-m_D4**, approximately

    0.5630113309687108 <= mu <= 1.9816855638884092.

These displayed decimals do not define the interval; the rational expressions
do. The two complementary small-component intervals remain OPEN. In particular
the actual D4 mass has **not** passed the all-splits gate. m_* is a sufficient
bound from this estimate, not a demonstrated critical mass or failure onset.

## Three approaches and why they do not yet prove the universal assertion

1. **Amplitude scaling:** S1–S2 succeed in the stated range because F3 controls
   the repulsive term uniformly along minimizing sequences. Beyond that range,
   the displayed estimate can lose its sign. An explicit negative-energy
   Gaussian has positive amplitude derivative, so merely invoking H<0 fails.
   This is a counterexample to the shortcut, not to the inequality for I.
2. **Concentration compactness/profile extraction:** weak H² limits, quadratic
   Hilbert splitting and separate L⁴/L⁶ Brezis–Lieb splittings isolate any
   nonzero profile. Strict binding excludes a positive-mass remainder. Without
   strict binding the calculation allows equality and cannot eliminate a
   second distant profile. The full proof and converse obstruction appear in
   [WHOLE_SPACE_COMPACTNESS.md](WHOLE_SPACE_COMPACTNESS.md).
3. **Direct profile comparison:** disjoint profiles prove weak subadditivity.
   Overlap can lower or raise the local nonlinear energy: for real profiles
   placed in quadrature phase, the nonlinear defect is
   |u|²|v|²[-beta+gamma(|u|²+|v|²)]. Its sign is not uniform. Complex profiles
   additionally have quadratic and norm cross terms requiring normalization.
   Translations/relative phases in the recorded diagnostics give both signs.
   No comparison uniform over unknown minimizing profiles is obtained.

The alternative density-preserving dilation also has no universal sign:

    H(v_theta)/theta-H(u)
      = (1-1/theta)[-(1+1/theta)D+2k²K].

The negative-gradient term prevents importing the positive-dispersion proof.
For a sufficiently regular mass-constrained critical point, its mass-preserving
virial identity gives amplitude slope 2(k²K-D), which has no established
universal negative sign. No unproved Pohozaev regularity is used in S1–S2.

## Literature boundary

The concentration-compactness and minimizing-set stability method follows
[Lions (1984)](https://numdam.org/item/AIHPC_1984__1_2_109_0/) and
[Cazenave–Lions (1982), DOI 10.1007/BF01403504](https://doi.org/10.1007/BF01403504).
The actual estimates and hypotheses for this functional are supplied here.

[Luo–Zhang (2022)](https://www.aimspress.com/aimspress-data/era/2022/8/PDF/era-30-08-146.pdf)
is relevant general mixed-dispersion context, but its negative-dispersion
Theorem 1.7 cannot be imported: its F6 requires a nonnegative nonlinear
primitive with a growth inequality. Our focusing primitive
beta*s⁴/4-gamma*s⁶/6 changes sign. Its F4 also requires subcritical decay of
f(s)/s⁵ in dimension two; here the limit is -gamma, not zero. Lemma 3.4's
positive-dispersion dilation is likewise inapplicable. No all-mass conclusion
is attributed to that paper. S1–S2 are a direct sufficient estimate, with no
claim of priority over the literature.

The unproved universal assertion is neither assumed nor refuted. The next
precise gap at the frozen D4 mass is the pair of asymmetric split intervals,
including the limit in which one component's norm tends to zero.
