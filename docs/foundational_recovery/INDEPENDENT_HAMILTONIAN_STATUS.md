# Route B: independent Hamiltonian confinement

Date: 2026-10-20. **INDEPENDENT_EFFECTIVE_MODEL_SUPPORTED**, meaning a
mathematically defined independent effective model, not a derived effective
theory of S004 or a fundamental physical theory. Its original implementation
is `wct-pde/src/wct_pde/projection_free.py`, verified at evidence revision
`051b6eca10c46b8e038993d60cabb20715194a20`. That branch is open PR #26,
stacked on #17; default main is `ecc2a446cc8fbf2f104bc49aa0a9fdf5a9739876`.
The coefficient conventions and mathematical equation agree with the source.

## B1. Action, complete scalar constraints and conserved quantities

Let beta,gamma>0, k_*>0, A=Delta+k_*^2, with either periodic boundary conditions
on a fixed flat torus or whole space R^d. The independent equation is

    i u_t=A^2 u-beta|u|^2u+gamma|u|^4u.

In the declared units its action, Hamiltonian and norm are

    S=integral dt [ integral i(bar u u_t-u bar u_t)/2-H(u)],
    H=||Au||_2^2-beta||u||_4^4/2+gamma||u||_6^6/3,
    M=||u||_2^2.

Variation in bar u gives i u_t=delta H/delta bar u. Write u=a+i b; the
symplectic density is b a_t-a b_t. Momenta obey p_a-b=0, p_b+a=0. Their
Poisson matrix is [[0,-2],[2,0]] times the spatial delta. Preservation fixes
both multipliers, with no secondary constraint or first-class constraint.
The reduced bracket is {a(x),b(y)}_D=delta(x-y)/2. There is one canonical
pair per point (two real initial fields). Amplitude and phase are physical
variables wherever u!=0; global U(1) is not a gauge removal of phase.
The acceleration Hessian is zero; the first-order velocity Hessian is also
zero, with the invertible symplectic constraint matrix providing evolution.

H and M are conserved. For the generator delta u=i u, Noether charge in this
symplectic convention is -M; using M generates the opposite phase rotation.
This sign convention has no effect on conservation or fixed-M minimization.
Translation symmetry gives the conserved momentum integral Im(bar u grad u)
up to the same declared generator convention. The equation is time reversible
under u(t)->bar u(-t). It is not a dissipative relaxation equation.

For u=e^(-i omega t)phi the stationary equation is

    (Delta+k_*^2)^2 phi-beta|phi|^2phi+gamma|phi|^4phi=omega phi.

For a real stationary phi the full real/imaginary Hessians of H-omega M are

    L_+=A^2-omega-3beta phi^2+5gamma phi^4,
    L_-=A^2-omega-beta phi^2+gamma phi^4.

The real linearized flow is (a_t,b_t)=(L_- b,-L_+ a). Both sectors matter.
At zero the exact frequency is omega(k)=(|k|^2-k_*^2)^2, real nonnegative,
and the propagator is unitary in each H^s. At smooth bounded backgrounds the
high-order principal part remains the skew Hamiltonian multiple of Delta^2;
bounded local polynomial linearizations cannot produce S004's unbounded
exp(c|k|t) UV mechanism in H^2. This does not rule out finite-frequency
instabilities of other backgrounds or prove stability of every stationary wave.

There are no higher-time Ostrogradsky degrees here. H has a lower bound at
fixed M, not necessarily one global lower bound as M varies. This is a
classical mathematical health statement; it does not establish relativistic
covariance, quantum positivity, finite signal speed or a physical clock.

## B2. Global H^2 evolution in d=1,2,3

**Proposition.** For the above constants, any u0 in H^2(R^d) or H^2(T_L^d),
1<=d<=3, has a unique global mild solution in C(R;H^2), continuously dependent
on its initial data on bounded time intervals. It conserves H and M. Smooth
data persist as long as the usual higher-regularity estimates apply; the
global statement needed here is in H^2.

**Proof.** H^2 is a Banach algebra for these dimensions. The cubic–quintic
nonlinearity is locally Lipschitz H^2->H^2. Since exp(-it A^2) is unitary on
H^2, contraction of the Duhamel map on C([-T,T];H^2) gives local existence,
uniqueness, continuous dependence and an H^2 blow-up alternative. Smooth
spectral regularizations conserve mass and energy by differentiation and
self-adjoint integration by parts. Their convergence in H^2 on the local
interval, continuity of the polynomial energy on H^2, and iteration give
conservation for the mild solution. On R^d use smooth spectral cutoffs and
dense smooth data; on tori Fourier Galerkin truncations suffice.

For r>=0 the exact pointwise identity is

    -beta r^2/2+gamma r^3/3
      =gamma r(r-3beta/(4gamma))^2/3-3beta^2 r/(16gamma).

Consequently, for all times of existence,

    ||Au||_2^2 <= H(u0)+3beta^2 M(u0)/(16gamma),
    ||Delta u||_2^2 <= 2||Au||_2^2+2k_*^4 M(u0).

Fourier interpolation controls the remaining gradient norm. These bounds
exclude finite-time H^2 blow-up in either direction. An optional stronger
coercive inequality retains gamma||u||_6^6/6 at the cost of the larger mass
constant 3beta^2/(8gamma). QED.

This extends the argument already present in the periodic D4 appendix to
the stated whole-space/dimensional setting; the method is standard, not a
claim of a new general well-posedness technique. d>=4 is excluded from this
algebra proof. gamma=0 or gamma<0 requires other estimates. This theorem
does not give scattering, a bound state, or its orbital stability.

## B3. Whole-space vanishing threshold and a genuine negative trial bound

Define I_d(m)=inf{H(u):u in H^2(R^d), M(u)=m}, m>0. The convention includes
the k_*^4 M term in ||Au||^2; subtracting that constant changes numerical
thresholds and must be stated. In this convention the vanishing threshold
is **zero**, not the negative threshold of the other closure functional.

**Proposition B3a.** Any H^2-bounded sequence of mass m that vanishes locally
in the concentration-compactness sense

    sup_y integral_(B(y,1)) |u_n|^2 -> 0

has liminf H(u_n)>=0, for 1<=d<=3. Conversely there are vanishing fixed-mass
sequences with H->0, so the threshold is sharp.

**Proof.** The local-mass vanishing lemma, using a uniformly locally finite
unit-cube cover, local Sobolev interpolation and the bounded H^2 norm, implies
||u_n||_p->0 for 2<p<infinity in d<=3. In particular the quartic and sextic
terms vanish; the remaining squared A norm is nonnegative. For sharpness
choose real normalized f in Schwartz space and |k0|=k_*, and set
u_R=sqrt(m) R^(-d/2) f(x/R) exp(i k0.x). Then

    ||Au_R||^2=m R^-4 ||Delta f||^2
                     +4m R^-2 ||k0.grad f||^2,
    ||u_R||_4^4=m^2 R^-d ||f||_4^4,
    ||u_R||_6^6=m^3 R^-2d ||f||_6^6.

All terms tend to zero and local mass vanishes. This establishes a threshold
for vanishing, not for every possible split into distant bound components.

**Proposition B3b (negative trial energy at every mass in two dimensions).**
For any beta,gamma,k_*>0 and m>0, I_2(m)<0.

**Proof.** Let

    f_ab(x,y)=(pi a b)^(-1/2) exp[-x^2/(2a^2)-y^2/(2b^2)],
    u_R=sqrt(m) R^-1 f_ab(x/R,y/R) exp(i k_* x).

Direct Gaussian integration gives M=m and

    H(u_R)=R^-2 [2m k_*^2/a^2-beta m^2/(4pi a b)]
      +R^-4 [m(3/(4a^4)+1/(2a^2b^2)+3/(4b^4))
                                     +gamma m^3/(9pi^2 a^2b^2)].

Choose a/b>8pi k_*^2/(beta m). The R^-2 coefficient is negative and the
R^-4 coefficient positive. A sufficiently large finite R gives H<0. This
uses the already supplied focusing cubic term; no equation or coefficient
was changed. The trial profile is complex and anisotropic: it is not claimed
to be real, D4-symmetric, stationary, dynamically generated, or optimal.

At the exact D4 coefficient and mass dyadics, k_*=1, a=4,b=1,R=4, an
exact-rational Machin-pi enclosure gives

    H(u_R) in (-0.005094, -0.005093).

The unrounded rational endpoints are in the executable receipt. This is an
analytic trial calculation with certified scalar arithmetic, not a new PDE
simulation or a continuum stationary certificate. An independent Gaussian
quadrature checks the formula. The width parameters specify a trial function;
they do not fit observed masses or select a physical confinement scale.

**Consequence.** B2's coercivity bounds all minimizing sequences. B3a/B3b
therefore exclude vanishing for I_2(m), including the pinned D4 mass. This
closes one whole-space variational prerequisite. **Dichotomy remains open.**
A negative upper bound is insufficient to prove

    I_2(m)<I_2(mu)+I_2(m-mu) for every 0<mu<m.

No pair of numerical upper bounds can certify that strict inequality against
the two unknown infima. The defocusing sextic means the usual pure-power
amplitude scaling does not automatically give strict subadditivity.

## B4. Conditional compactness/stability route, with the missing hypothesis explicit

Suppose 1<=d<=3, beta,gamma>0, I_d(m)<0, and the strict splitting inequality
I_d(m)<I_d(mu)+I_d(m-mu) holds for **every** 0<mu<m. Then minimizing sequences
are precompact in H^2 modulo translations, the minimizer set is nonempty, and
that full set is orbitally stable under B2's flow.

Proof: coercivity gives H^2 boundedness. Concentration compactness for |u_n|^2
has vanishing, dichotomy or tightness after translation. B3a excludes the
first. In the second, smooth separated cutoffs with first/second derivatives
tending to zero split mass and the local polynomial terms; the squared A
form splits up to o(1), with a nonnegative escaped remainder. Renormalizing
the two masses costs o(1) because H is polynomial on bounded H^2 sets. The
result is I_d(m)>=I_d(mu)+I_d(m-mu), a contradiction. Tightness gives strong
L^2 after translation by local Rellich compactness, then strong L^4 and L^6
by H^2 boundedness/interpolation. Lower semicontinuity and minimality give
convergence of ||Au_n||_2, hence strong Au_n and, with strong L^2, strong H^2.
The limiting Euler–Lagrange equation is B1 with a real mass multiplier omega;
elliptic bootstrapping yields a smooth state. This proves finite energy and
localization in the H^2/L^2 sense, not uniqueness or a particular decay rate.

For stability, assume data approach the minimizer set but later solutions
stay a positive H^2 distance away. Conservation and B2's bound make those
later solutions a minimizing sequence after an amplitude normalization
tending to one if the masses differ slightly. The compactness just proved
contradicts that distance. Phase invariance is included in the minimizer set.
This is stability of the full minimizer set, not necessarily of one isolated
orbit, nor a separate spectral nondegeneracy theorem. The splitting premise
has **not** been verified for this cubic–quintic model at the target mass.

This is the standard concentration-compactness route specialized to this
energy. See [Lions, 1984](https://numdam.org/item/AIHPC_1984__1_2_109_0/) and
[Bonheure et al., 1710.09775v2](https://arxiv.org/abs/1710.09775v2).
[Fernandez et al., 2010.01448v3](https://arxiv.org/abs/2010.01448v3) treats
competing-dispersion pure-power variational problems. Their nonlinearity is
different; their minimizer thresholds are not imported into this saturating
cubic–quintic equation. B3 supplies its own trial proof.

## B5. D4 audit and the domain-enlargement boundary

The accepted certificates concern T_42^2, k_*=1,
beta=gamma=`0x1.05f1d178c75ebp+2`, mass=`0x1.45b8a0bad2714p+1`.
The exact root frequency lies in the joint radius-1e-8 Wiener/frequency ball
about `-0x1.712bc3b876ea0p-2`; it is not defined as the center.

| Proof component | Audited object and conclusion | Phase III execution |
|---|---|---|
| Root and gauge | Real D4 Wiener product ball, phase/reflection center fixed, mass border | Proof and exact receipt read; root interval campaign retained |
| Residual and inverse | Full degree-five nonaliasing convolution; 232-dimensional bordered inverse with defect <1 | Input hashes verified; no new matrix inversion claimed |
| Infinite complement | Monotone diagonal beyond K=20 and all low/high convolution interactions | Analytic bounds inspected; retained outward constants |
| Root-to-saved-array link | Outward exponential DFT and Nyquist splitting | Exact candidate/receipt pins verified; DFT campaign not rerun |
| Full coercivity | All 8 real/imaginary parity sectors plus infinite Schur complements | Proof/receipt read; interval matrices not recomputed |
| Exact-root transfer | Potential/projector errors, exact phase and translation kernels | Transfer argument inspected |
| Nonlinear orbital theorem | Mass-penalized Lyapunov function, full complex perturbations, modulation, global H^2 continuation | Proof inspected; B2 independently reconstructs continuation |
| Explicit basin | Rational Gram, norm, slice and energy gates | Existing exact-rational verifier rerun against exact blobs |

All five rational basin gates pass. The retained theorem is

    dist_H2(u0,O_phi)<4.2e-12
      => sup_(t in R) dist_H2(u(t),O_phi)<4.2e-9,
    ||v||_H2=||(1-Delta)v||_L2(dx),
    O_phi={exp(i theta)phi(.-a)}.

It permits mass-changing, complex and symmetry-breaking small data. It is
orbital stability, not attraction or arbitrary-amplitude stability. D4 is
square symmetry in two dimensions. The old appendix/PR body saying the
quantitative basin is open is superseded by the later #27/#28 evidence;
the old documents remain unchanged. #24,#25,#27 are closed; #21–23 remain open.
No new existence theorem is attributed to this verifier replay.

Uniform fixed-M/H bounds are available on expanding tori from B2. They do
not by themselves give tightness, a common stationary branch, or strong
whole-space convergence. Periodically tiling the same root multiplies its
mass with volume. Rescaling x=L y instead changes the operator to
(L^-2 Delta_y+k_*^2)^2; a simultaneous mass-preserving amplitude scaling also
changes the cubic/quintic coefficients. Resetting those coefficients to their
old numerical values changes the physical problem.

A justified continuation needs a fixed coefficient/unit law, uniform mass
and energy, controlled frequency, and tightness after translations, or a
minimizer family with the splitting and box-to-whole-space energy limits
proved. Uniform symmetry-reduced coercivity and modulation estimates would
be needed to transfer this particular root's stability. The fixed-box D4
certificate supplies none of these box-uniform estimates. B4 can instead
establish some whole-space minimizer if its missing hypothesis is proved;
it would not automatically identify that state with D4.
