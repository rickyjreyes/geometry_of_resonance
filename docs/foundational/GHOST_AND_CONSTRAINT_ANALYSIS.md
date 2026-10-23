# S004 physical-health audit: constraints, ghost and ultraviolet obstruction

Date: 2026-10-20. Primary verdict: **`FOUNDATIONAL_GHOST_OBSTRUCTION_CONFIRMED`**.
This concerns the literal corrected S004 complex-scalar action with fixed flat
metric and inertial foliation, epsilon>0, kappa>0, theta>0 and
gamma²<4 kappa theta. A dynamical metric/foliation/gauge completion has not been
specified by S004; its constraints cannot be presumed. A different completion
requires its own analysis. The primary portfolio verdict is
`FOUNDATIONAL_OBSTRUCTION_CONFIRMED` for this proposed foundational formulation.

## Definitions and independently reconstructed variation

Use tau=c0 t, signature (-,+,...,+), rho=|psi|²,
D=rho+epsilon² exp(-2 alpha rho), F=rho/D²,
Q=Box psi, P=Delta psi, and
H=kappa |Q|²+theta |P|²+gamma Re(Q conjugate(P)).
The source action is L_orig=partial conjugate(psi) partial psi -V(rho)+FH.
As in the source quadratic audit, L_phys=-L_orig gives the ordinary mode a
positive kinetic term. Changing the overall sign cannot change the relative
ghost. All canonical formulas below use this one orientation.

Compactly supported variations, periodic data, or sufficiently decaying fields
and variations with no boundary flux are required. Otherwise the higher-order
boundary form in [Q2](../cross_paper/QUADRATIC_ACTION_AND_DISPERSION.md) must be
retained. Fixed field values alone do not remove its normal-derivative term.
Writing A_Q=F(kappa Q+gamma P/2), A_P=F(theta P+gamma Q/2), the bulk equation is

\[
-\Box\psi-V'\psi+F'\psi H+\Box A_Q+\Delta A_P=0. \tag{H1}
\]

Direct real-Cartesian differentiation of the original density in one time and
one spatial coordinate checks both real equations, including all derivatives
of F. In any spatial dimension the same integrations by parts give H1 with
Delta the full Laplacian. For curved/nonparallel h, the final operator is
**Delta_h^dagger A_P = nabla_nu nabla_mu(h^{mu nu} A_P)**, not in general
Delta_h A_P. The order shown is the direct formal adjoint; symmetry of h and
dummy-index relabeling give the equivalent reversed-index expression. For a
one-dimensional coefficient h(x), the difference is 2h' A_P'+h'' A_P.
The older live draft's displayed curved-space equation therefore requires
parallel/self-adjoint h or this correction. This restriction is irrelevant to
the fixed flat proofs below but matters to a covariant generalization.

The [complete arbitrary-background quadratic coefficient Q1 and homogeneous
matrix Q4–Q7](../cross_paper/QUADRATIC_ACTION_AND_DISPERSION.md) retain their
scope. They include both Cartesian components, rotating-background mixing,
F', F'', and the full fourth-time-order determinant. They were rerun using
their independent real-jet Hessian construction. The new script also obtains
the static amplitude and phase kernels directly from the density.

## Theorem H2: complete constraint count on the regular fixed-background sector

Let x=(Re psi, Im psi), v=x_tau, a=x_tautau, b=Delta x,
A=2 kappa+gamma, C=kappa+theta+gamma. On a smooth patch with rho bounded away
from zero, F>0 and

\[
L_{phys}=|v|^2-|\nabla x|^2+V
-F(\kappa |a|^2-Aa\cdot b+C|b|^2).
\]

The ordinary-velocity Hessian at fixed accelerations is 2I; the acceleration
Hessian is -2 kappa F I, with determinant 4 kappa² F²>0. The first is not a
positivity test for the complete higher-order system.

**Claim.** There are four real configuration degrees of freedom per spatial
point (eight real phase-space variables), including two extra higher-time
degrees. No primary or secondary scalar constraint removes them.

**Proof.** Introduce independent v and lambda with
L1=L_phys(x,v,dot v,Delta x)+lambda dot (dot x-v). The canonical momenta obey
p_x-lambda=0 and p_lambda=0. Their Poisson matrix has the invertible blocks
{p_x-lambda,p_lambda}=-I and its transpose; these four scalar constraints are
second class. The acceleration Hessian makes p_v=-F(2 kappa dot v-Ab)
invertible. Preservation of the four primary constraints determines their
multipliers and creates no secondary constraint on x,v,p_x,p_v. Eliminating
lambda,p_lambda leaves the ordinary canonical symplectic form on these eight
variables. Equivalently, six original first-order coordinates give 12 phase
variables minus four second-class constraints, hence four degrees. There is
no first-class constraint or local gauge symmetry in this defined sector. QED.

Global U(1) is a conserved symmetry, not a gauge identification. Fixing its
charge is one global restriction, not a local removal of the extra modes.
At psi=0, F=0 changes differential order and the regular Legendre map fails;
this isolated vacuum does not establish a constant-rank constraint eliminating
the extra modes near nonzero fields. At kappa=0 a new analysis is required;
it is outside strict curvature-form positivity.

## Canonical Hamiltonian and actual phase charge

Put p1=partial L/partial a=-F(2 kappa a-Ab),
p0=2v-dot p1, B_perp=theta-gamma²/(4 kappa)>0, J(x1,x2)=(-x2,x1).
The independently reconstructed exact Legendre transform is

\[
H_{phys}=\int\left[p_0\cdot v-|v|^2+|\nabla x|^2-V
-\frac{|p_1|^2}{4\kappa F}
+\frac{A}{2\kappa}p_1\cdot b+B_\perp F|b|^2\right],\quad
Q_{phys}=\int(p_0\cdot Jx+p_1\cdot Jv). \tag{H3}
\]

Energy and charge are conserved for sufficiently smooth solutions with no
boundary flux, whenever the integrals exist. This formal conservation law is
not an initial-value existence theorem. The local formula does not divide by
F at nodes. Whole-space background energy needs a specified vacuum subtraction.

The [merged eigen Phase II theorem, C13–C14](https://github.com/rickyjreyes/eigen/blob/957cd8f382ba72f3764941c372222d7ebb58ed7e/docs/phase_ii/COVARIANT_DILATION_THEOREM.md)
already proves the full fixed-charge obstruction; it is recovered here, not
claimed as a new result. For a compact cutoff eta in a regular patch set
w=eta x, c=-(p1 dot Jw)Jx/rho, v_s=v+sw,
p0_s=p0+s(c+t w), holding x,p1 fixed. Both Q and N=integral|x|² are unchanged
exactly, while

\[
H(s)-H(0)=s\ell_t+s^2(t-1)\int|w|^2.
\]

A local minimum would require ell_t=0 for this two-sided variation; t<1 then
contradicts minimality. If ell_t is nonzero, one sign of s descends already.
The linear p0 dot v term also gives arbitrarily negative energy at fixed charge
on appropriate regular data. This is an energetic theorem in any spatial
dimension, not a proof that every standing wave is absent or dynamically unstable.

For flat constant coefficients the local Noether current can be written

\[
j^\mu_{orig}=2\Re\{i[(\partial^\mu\bar\psi
-\partial_\nu B^{\mu\nu})\psi+B^{\mu\nu}\partial_\nu\psi]\},\quad
B^{\mu\nu}=\bar A_Q g^{\mu\nu}+\bar A_P h^{\mu\nu};
\qquad j^\mu_{phys}=-j^\mu_{orig}.
\]

Its time component integrates to H3's Q. Its divergence is zero on H1.
The derivation is the boundary coefficient of delta psi=i eta psi in the
second-derivative variational formula, including the conjugate terms. Replacing
eta by eta(x) is not a symmetry: derivatives of eta appear in Box(e^{i eta}psi).
A gauge connection and its equations would be new field content. A conserved
phase charge is not automatically the L² norm used in the other models.

## Theorem H4: high-frequency failure of Sobolev continuous evolution

**Hypotheses.** Fix a nonzero constant on-shell background psi0, V'(rho0)=0,
with finite V''(rho0), on a flat torus or R^d. On R^d these are perturbations
of a homogeneous background, not a finite-energy self-bound background.
Set K=F(rho0) kappa>0, G=F(rho0) gamma,
T=F(rho0) theta>0, B=4KT-G²>0. For amplitude set m²=-2rho0 V'';
for phase m²=0. No positive-mass assumption is needed for the ultraviolet limit.

**Claim.** The full linearized Cauchy problem has exponentially growing
Fourier modes with growth c_uv |k|+O(|k|^-1), c_uv>0. At every fixed tau>0
its solution map on smooth initial data has no bounded extension in any
fixed Sobolev four-jet topology; no finite loss of derivatives repairs this.

**Proof.** With z=omega², q=|k|², direct quadratic variation yields

\[
\mathcal D(z,q)=(z-q)-m^2-K(z-q)^2+Gq(z-q)-Tq^2,
\]
\[
z_\pm=q+\frac{1+Gq\pm\sqrt{1-4Km^2+2Gq-Bq^2}}{2K}.
\]

The discriminant tends to -Bq². Hence

\[
\frac{z_\pm}{q}\longrightarrow a\pm ib,\quad
a=\frac{2K+G}{2K},\quad b=\frac{\sqrt B}{2K}>0,
\qquad c_{uv}=\sqrt{\frac{\sqrt{a^2+b^2}-a}{2}}>0. \tag{H4}
\]

Taking square roots of z gives one growing time exponent s=-i omega with
Re s=c_uv |k|+O(|k|^-1). This is stronger than merely finding complex poles.
On a torus choose integer modes k_n going to infinity and their exact initial
jets (1,s_n,s_n²,s_n³)e^{ik_n x}. Normalize in, for example,
X_r=H^{r+3} x H^{r+2} x H^{r+1} x H^r, then multiply by
exp(-Re(s_n) tau0). Their initial norms tend to zero, while the norm at any
fixed tau0>0 is one. Passing to real solutions by conjugate mode pairs does
not change the exponential conclusion. Every finite change of Sobolev indices
introduces polynomial weights, which cannot bound this exponential.
On R^d, use smooth Fourier packets in small neighborhoods of k_n, choosing
the smooth growing root there; the same uniform lower bound on Re s holds.
Thus the linear maps on finite-mode/packet data are unbounded. QED.

This excludes standard Sobolev well-posedness for this **linearized homogeneous
sector**, and also a nonlinear solution theory with a bounded Frechet derivative
there equal to its linearization. It is not by itself a nonlinear norm-inflation
proof, a statement about every inhomogeneous state, a measured lifetime, or a
theorem against all WCT model families. Analytic-data spaces with shrinking
analytic radius, explicit cutoffs, or modified principal terms would require
different hypotheses and do not establish the requested healthy full theory.

**Checks and failure boundaries.** K=T=1,G=0,m²=0 gives
c_uv=sqrt((sqrt(2)-1)/2). The script solves the independent time-domain
companion polynomial at eight increasing k and compares with the exact poles.
The error to the asymptotic growth coefficient has order k^-2. No PDE time
simulation is substituted for the proof. For theta=gamma=0, B=0 and the phase
poles are q and q+1/K: the UV theorem does not apply although opposite residues
and the Ostrogradsky obstruction persist. At psi0=0 or kappa=0 its hypotheses
also fail. Periodic cubic–quintic Hamiltonian evolution has a different symbol.

Where the two poles are real and distinct,
partial_z D(z_minus)=+sqrt(discriminant) and
partial_z D(z_plus)=-sqrt(discriminant). There is a ghost in each real sector;
collision does not remove it by a constraint. These signs are independent of
the potential's high derivatives and of the overall action orientation.

## What survives as an effective calculation

The [quadratic low-branch reduction](../cross_paper/CORRECTED_ACTION_TO_EFT_KERNEL.md)
is valid on a declared spectral window with discriminant uniformly positive,
frequency below z_plus, and a uniformly bounded branch-normalizing multiplier.
Its one-real-component Taylor remainder is explicit and replayed here.
This is a controlled **quadratic** reduction, not a nonlinear or quantum EFT
completion. The phase field remains gapless. Cubic products can generate a
third spatial harmonic (cos³ kx=(3 cos kx+cos 3kx)/4); the unspecified V already
allows such interactions. Therefore an initially band-limited state is not
automatically an invariant sector of the full action. Projection changes the
equation; discarding the high pole needs an error theorem over a stated time.
No selected cutoff, nonlinear charge-preserving reduction, UV completion, or
healthy quantization has been established. The primary verdict cannot be
`CONSISTENT_RESTRICTED_EFT_ONLY` on the basis of a quadratic calculation alone.

The separate scalar-tensor action in DERIVATION_CLOSURE.md is not S004. On a
fixed metric, its nonzero beta_Theta higher-time coefficient is likewise
nondegenerate; setting theta=gamma=0 removes H4's strict-B premise, not the
relative ghost. If g itself is varied, metric-scalar mixing and gravitational
constraints must be analyzed for that supplied full action. No result here
counts unspecified metric/foliation constraints as absent or as a cure.

## Reproduction and relation to prior literature

Run `python derivations/foundational_closure.py --output NEW/verification.json`
and `python -m unittest -v derivations.test_foundational_closure`. The script
refuses to overwrite receipts. [Verification](../../audit/foundational/verification.json)
records exact symbolic identities, numerical controls, versions and source hash.
H2 and H4 are new deductions in this workstream; H1, the general quadratic
expansion, pole signs and the fixed-charge energetic obstruction were already
established in the pinned sources and have been independently reconstructed.

R. P. Woodard, *The Theorem of Ostrogradsky*,
[arXiv:1506.02210v2](https://arxiv.org/abs/1506.02210v2), is background for the
canonical construction. The explicit S004 constraint count, Hamiltonian and
Fourier sequence above supply this audit's proofs; a general citation alone
would not verify their hypotheses. None of these proofs is Lean-kernel checked.
