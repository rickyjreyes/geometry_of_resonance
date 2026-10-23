# Localization and stability: precise surviving scope

Date: 2026-10-20. Primary Phase 3 verdict for a self-bound state of the proposed
foundational dynamics: **`EXISTENCE_OPEN`**. Its healthy physical interpretation
is already obstructed in Phase 1. The energetic non-minimum theorem does not
prove nonexistence of all stationary saddles for an unspecified potential.

| Model/question | Exact conclusion | What cannot be inferred |
|---|---|---|
| Additive eigen scalar positive energy, regular D>0 branch, fixed N>0 | `DILATION_OR_COMPACTNESS_OBSTRUCTION`: no local constrained minimum in d=1–4; infimum zero unattained; regular dilation-admissible critical points excluded | All weak/singular extensions, source confinement, periodic states or lifetimes |
| Corrected complex positive M2 energy | Same strict spreading minimum obstruction on complex H²(R^d), d=1–4; differentiable-dilation critical-point exclusion | S004's full conserved Hamiltonian or arbitrary V |
| Literal S004 regular canonical phase space | No local fixed-Q energy minimum, regardless of V and d; homogeneous linear UV ill-posedness under H4 hypotheses | Nonexistence or dynamical instability of every stationary saddle |
| Separate closure functional | Fixed-N boundedness/coercivity and periodic attainment; finite radius along fixed trial shape | Whole-space compactness or uniqueness from negative energy alone |
| Separate projection-free D4 application | `CONDITIONAL_BOUNDARY_CONFINEMENT_ONLY` as a physical confinement interpretation; validated mathematical existence and local orbital H² stability on the specified torus | Self-selected physical boundary, whole-space particle, large-data attraction, or 4D dimensional selection |

## Positive corrected spatial functional: decisive theorem already merged

On complex H²(R^d), let T=||grad psi||², B(x)=|Delta psi|²,
E_F=c1 T+c2 integral F(rho)B with c1,c2,epsilon>0, alpha>=0.
For psi_lambda=lambda^(d/2)psi(lambda x), N is fixed and

\[
E_F(\psi_\lambda)=c_1\lambda^2T+c_2\lambda^4\int F(\lambda^d\rho)B,
\]
\[
4F(s)+dsF'(s)=F(s)\left[4-d+
\frac{2d\epsilon^2e^{-2\alpha s}(1+2\alpha s)}{s+\epsilon^2e^{-2\alpha s}}\right].
\]

For 1<=d<=4 the bracket is strictly positive for finite s>0; T>0 for any
nonzero whole-space L² field. Therefore the derivative with respect to log
lambda is positive. F and sF' are bounded, so differentiation is dominated by
an L¹ multiple of B without an L-infinity assumption, including d=4.
Strong continuity of dilation in H² provides arbitrarily close descent paths.
Spreading lambda->0 gives zero infimum, never attained at positive N.
Excluding stationary critical points additionally requires Frechet variation
and a differentiable dilation curve in its variation space, e.g. suitably
smooth rapidly decaying fields. This is a proof of the stated no-minimum
claim, not a substitution of H² embedding for the d=4 endpoint analysis.

Adding a static potential U=-V contributes
d integral(rho U'-U) to the fixed-N virial. If this is nonnegative the same
static obstruction holds; a linear mass term contributes zero. No such sign
is assumed for every unspecified U. No attractive term was added in this audit.
Whole-space admissibility also requires integrability of U or a specified
vacuum subtraction, rather than assigning finite energy to a constant vacuum.

## Finite-band scale is not spatial compactness

For the separate closure energy, Fourier interpolation gives
F_delta >= (a4/2)||Delta psi||² - a2² N/(2a4). The coefficient a4>0 controls
the H² norm at fixed N; its negative gradient term permits scale competition.
Yet broad packets R^(-d/2)f(x/R)exp(ik0 x) lose local mass while approaching

\[
e_{van}(N)=-\frac{a_2^2N}{4(a_4+\kappa_r/\delta^2)}<0.
\]

Thus negativity and a minimum along a fixed-shape dilation orbit do not prove
a whole-space minimizer. A binding inequality below the relevant escape
threshold and a strict-subadditivity argument excluding splitting remain
separate obligations. Similarly, a constant-coefficient finite-band linear
operator on R^d has no L² eigenfunction supported on a single measure-zero
Fourier shell. A preferred carrier wavelength alone is not a bound envelope.

## Current D4 result: affirmative, independently scoped

The model is
i u_t=(Delta+1)²u-beta|u|²u+gamma|u|⁴u on R²/(42Z)²,
with beta=gamma=`0x1.05f1d178c75ebp+2`, mass=`0x1.45b8a0bad2714p+1`.
The exact root is within the joint real-D4 Wiener/frequency ball of radius
1e-8 around the pinned candidate and frequency center
`-0x1.712bc3b876ea0p-2`. The root is unique in that gauge-fixed ball; no global
uniqueness, mass interval or whole-space uniqueness is asserted.

The accepted #24 certificate controls the residual, bordered inverse, every
low/high interaction, infinite tail and saved-array linkage. #25 proves
coercivity after the correct symmetry/mass penalties, all eight real/imaginary
parity sectors and infinite complements, modulation, conservation and global
H² continuation. For arbitrary complex small perturbations—including mass
changes and broken D4 symmetry—it gives local-in-data, all-time orbital H²
stability of the exact root. It is not attraction.

The subsequent #27 theorem supplies the explicit physical-norm statement

\[
\operatorname{dist}_{H^2}(u_0,\mathcal O_\phi)<4.2\times10^{-12}
\Longrightarrow
\sup_{t\in\mathbb R}\operatorname{dist}_{H^2}(u(t),\mathcal O_\phi)
<4.2\times10^{-9}.
\]

Here ||v||H²=||(1-Delta)v||L²(dx) and
O_phi={e^(i theta)phi(.-a)}. Mean-norm radii are smaller by 42. This follows
from the quantitative Gram/modulation bound, energy upper constant <27,
exact-slice lower coefficient 0.00082085097379 and the first-exit argument.
The [new rational replay](../../audit/foundational/d4_basin_replay.json) passes,
including all input pins; the root/coercivity premises and their original
interval certificates are retained. They were audited, not recomputed here.
This is computer-assisted mathematical evidence, not external peer review or
Lean formalization. #21–23 concern different numerical quantities and remain open.

## Conditional dimensional research specification

No parameter scan was launched. Before comparing d=2,3,4, select the same
well-posed model family, units, physical coefficient law, domain/box-limit
protocol and matched initial-data rule. Define a confinement norm, loss event,
right-censoring rule, perturbation class and time interval before simulations;
separate radiation, dissipation, box effects and numerical errors. Validate
temporal and spatial convergence and only then seek a lifetime theorem or
bounded empirical lifetime comparison. DS-01's open nonlinear refinement
cannot serve as that validation.

H²(R^d) embeds in L-infinity for d<4. Failure of that endpoint estimate at
d=4 is not proof of blow-up, universal short lifetime, or unique physical
dimensional selection. H4's S004 UV problem occurs already in d=1 and does
not distinguish d=3 from d=4. The separate 2D D4 theorem is also not such a
comparison. Whether a justified common model yields metastable 4D states and
longer-lived 3D states remains a well-defined conditional research question.
