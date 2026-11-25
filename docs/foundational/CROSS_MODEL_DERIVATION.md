# Cross-model derivation and non-equivalence

Date: 2026-10-20. Model-level verdict: **`MODEL_FAMILY_NOT_YET_UNIFIED`**.
The full-chain verdict is stronger because S004 has a demonstrated physical
obstruction. See the [source matrix](FOUNDATIONAL_ACTION_AUDIT.md) for pinned
definitions and [health audit](GHOST_AND_CONSTRAINT_ANALYSIS.md) for proofs.

| Proposed connection | Classification | What is actually justified | Missing premise or counterexample |
|---|---|---|---|
| Corrected S004 density → fourth-order EL and full quadratic action | `EXACT_DERIVATION` | Full complex variation, boundary terms and all homogeneous perturbation blocks | Curved h requires its adjoint; no physical-health implication |
| S004 → one real lower amplitude quadratic sector | `CONTROLLED_ASYMPTOTIC_REDUCTION` | Positive-residue frequency window; analytic field multiplier; explicit O(q³) spatial remainder | One component only; cutoff, initial preparation, nonlinear error and physical normalization not fixed |
| That sector + independently supplied damping → linear PFF form | `CONDITIONAL_MODEL_CONNECTION` | Overdamped slow exponent and coefficient map with quantitative linear error | Damping/bath is an additional premise; stable witness is least-damped, not growing; nonlinear beta not derived |
| Entire closed S004 ↔ autonomous dissipative PFF by invertible same-time field change | `MATHEMATICAL_NON_EQUIVALENCE` | Linearized real coordinate similarity preserves eigenvalues; a reversible oscillatory pair cannot become purely decaying relaxation | Noninvertible coarse-graining/radiative baths are outside this obstruction and remain unproved routes |
| S004 → full complex projection-free cubic–quintic Hamiltonian | `OPEN` | A one-component quartic dispersion resemblance is available | Gapless phase, charge map, nonlinear coefficients, common field/clock/units, prepared data and finite-time error theorem absent |
| S004 finite-k shape ↔ Hamiltonian finite-band operator, without those maps | `PHENOMENOLOGICAL_ANALOGY_ONLY` | Similar polynomial shape | Similarity of spectra alone is insufficient; full S004 has extra time branches |
| Projection-free Hamiltonian → its stationary equation | `EXACT_DERIVATION` | psi=e^(-i omega t)phi gives [(Delta+kstar²)²-beta|phi|²+gamma|phi|⁴]phi=omega phi | Domain/parameters still must match the target application |
| S004 static jets → corrected positive spatial energy plus U=-V | `EXACT_DERIVATION` | E_static=||grad psi||²+(kappa+theta+gamma) integral F|Delta psi|²+integral U | Static restriction is not complete conserved H; fixed N is not actual Q |
| S004 static restriction ↔ eigen's additive real energy | `MATHEMATICAL_NON_EQUIVALENCE` | Denominators and field derivatives differ even on real nonzero fields | Scalar theorem does not transfer by naming; corrected spatial theorem is separately proved in eigen #16 |
| S004 → closure functional / projected gradient flow | `OPEN` | Separate exact variational and scaling results exist | Negative gradient sign, nonzero zero-field weight, constraint and dissipation lack a nonlinear parent map |
| Fixed periodic D4 theorem → whole-space physical particle | `OPEN` | A specific mathematical periodic state exists and is orbitally stable | Domain removal, compactness, parent dynamics and observable map not derived |
| S004 or Hamiltonian rail → DS-01 | `PHENOMENOLOGICAL_ANALOGY_ONLY` | Shared finite-band motifs | Different additive curvature, entropy term and nonconserved N; full model connection absent |

## Explicit controlled part, with its limitations

For a nonzero static background define K=F kappa, G=F gamma, T=F theta,
m²=-2rho0 V''>0 and s=sqrt(1-4Km²)>0. The exact lower pole has

\[
z_-(q)=C_0+C_2q+C_4q^2+R_3(q),\quad
C_0=\frac{1-s}{2K},\quad C_2=1-\frac{GC_0}{s},\quad
C_4=\frac{T}{s}+\frac{G^2m^2}{s^3}.
\]

Let B=4KT-G²>0, d(q)=s²+2Gq-Bq², N_B=Bs²+G². On
0<=q<=qmax with d>=dmin>0,

\[
|R_3(q)|\le \frac{N_B\sup|G-Bq|}{4K d_{min}^{5/2}}q^3.
\]

The action factorizes D=K(z_plus-z)(z-z_minus). On a selected low frequency
window, xi=[K(z_plus-z)]^(-1/2)chi transforms the **quadratic** action to
chi*(z-z_minus)chi/2. This Fourier multiplier is nonlocal before expansion,
fails at the high pole, and is not a global transformation of all initial data.
The explicit remainder test uses the source's rational witness, not a new fit.

If C2<0, completing the square gives
qstar=-C2/(2C4) and Delta0=C0-C2²/(4C4).
Restoring time produces c_EFT²=c0² C4, m0²=c0² Delta0. Here c_EFT has units
length²/time, not light speed. The phase sector has C0=0,C2=1,C4=T and is not
a second identical real copy of this amplitude sector.

Supply chi_tt+eta chi_t+c0² K_sp chi=0 with eta>0 and
c0²|K_sp|/eta²<<1. After the fast transient,

\[
\lambda_{slow}=-c_0^2K_{sp}/\eta+O(c_0^4K_{sp}^2/\eta^3),\qquad
r=-\Gamma C_0,\ a=-\Gamma C_2,\ b=\Gamma C_4,\ \Gamma=c_0^2/\eta.
\]

This is a conditional linear asymptotic statement, not a nonlinear PFF error
theorem. For the stable source witness Delta0>0 and the maximal growth rate
is -Gamma Delta0<0. A growth band requires independently justified onset/pump
conditions. V''' and V'''' change nonlinear vertices without changing the
quadratic dispersion, so no unique cubic/quintic Hamiltonian or PFF coefficient
can be inferred from C0,C2,C4. Derivative nonlinearities and induced exchanges
also occur; the gapless phase cannot simply be eliminated as a massive field.

## Charge and stationary equation prevent an accidental identification

The canonical Q in the health audit is not N. For a standing wave of S004,
eigen #16 derives, in the physical orientation,

\[
Q_{phys}=-2\omega N+2(2\kappa+\gamma)\omega\int F\Re(\bar\phi\Delta\phi)
+4\kappa\omega^3\int F|\phi|^2.
\]

The sign corresponds to psi=e^(-i omega tau)phi and generator delta psi=i psi.
The last two terms depend on the profile. By contrast, N is exactly the
Hamiltonian rail's phase charge up to its declared normalization. An envelope
reduction may relate them approximately, but a coefficient match alone proves
neither that relation nor conservation of the discarded sector.

Minimal next acceptance theorem: specify a physically justified, nonlinear
admissible sector, an explicit map of fields/time/domain/Q/H, independently
fixed coefficients and cutoff, and a norm/time-interval error bound that
controls excitation of the excluded high modes. The new UV theorem makes
that error control essential. No additional couplings, attraction or imposed
confinement scale were inserted to meet it.
