# Geometric mass: exact identities and the physical bridge

Date: 2026-10-20. Primary Phase 4 verdict for S004's proposed physical chain:
**`MASS_BRIDGE_OBSTRUCTED`**. A separate, specified separated-field model gives
`CONDITIONAL_EFFECTIVE_MASS_DERIVED`; an independently selected WCT state,
operator, physical scale and inertial response have not followed from S004.

## The exact separated-field theorem and its premises

Let the parent equation be Box_(4+d) Psi=0 on a product geometry M4 x K with
the specified separable metric and boundary conditions. Let -Delta_K be a
self-adjoint confinement operator and chi_n a normalizable eigenfunction
with nonnegative eigenvalue lambda_n. Then Psi=phi_n chi_n implies
(Box_4-lambda_n)phi_n=0. The dispersion is
omega²=c²(|k|²+lambda_n). With the additional physical identifications
E=hbar omega and p=hbar k,

\[
E^2=c^2|p|^2+\hbar^2c^2\lambda_n,
\qquad m_n=\frac{\hbar}{c}\sqrt{\lambda_n}.
\]

This is a correct conditional effective mass. It assumes the product space,
operator, eigenmode, boundary conditions and frequency-energy normalization;
it does not derive them from the corrected quotient action. Interpreting it
as a quantized excitation also requires a healthy normalized quantum sector.
For a classical field solution, the conserved energy depends on its amplitude
and action normalization, so its total rest energy is not fixed by lambda
alone. Multiplying a classical action by a positive constant leaves its EL
equation unchanged while multiplying its Noether energy.

For the relativistic dispersion above, group velocity is c²p/E, and near
p=0, E=mc²+|p|²/(2m)+... . If an applied force satisfies dp/dt=F_ext, this
gives the usual small-velocity inertial response. Neither that response nor a
postulated acceleration phase lag has been derived for a localized S004 state.
In particular, a preferred-foliation quartic low-mode dispersion is not this
Lorentz mass shell. Giving its frequency gap the name mass does not supply
the missing energy-momentum theorem.

## Theorem M1: a sharp obstruction to replacing spectral RMS by a mean

This theorem uses a **declared example operator**, not a claimed derived WCT
operator. Let H=-d²/ds² on a circle of length L with periodic domain H²_per.
Let chi=R exp(i phi) be a smooth, nonvanishing eigenmode, R>0, H chi=lambda chi,
with periodic chi and consistent phase winding. Put N=integral R²>0,
w=R²/N, and assume the additional pointwise lock
phi'=sigma=sqrt(kappa²+tau²)>=0. Then

\[
\boxed{\lambda-\langle\sigma\rangle_w^2
=\frac{\int |R'|^2}{N}+\operatorname{Var}_w(\sigma)\ge0.} \tag{M1}
\]

Consequently sqrt(lambda)=<sigma>_w holds **if and only if** R is constant
and sigma is constant (almost everywhere; everywhere under these hypotheses).

**Proof.** Periodic integration by parts gives
lambda N=integral|chi'|². Differentiating the polar form yields
|chi'|²=|R'|²+R²(phi')². Divide by N, impose the lock, and subtract
<sigma>². The residual is exactly the two nonnegative terms in M1. Since w>0,
both vanish precisely when R'=0 and sigma is constant. The converse follows
by substitution, with periodic winding supplying sigma L=2pi n. QED.

Thus even an assumed lock would ordinarily relate the Rayleigh energy to a
weighted square plus an amplitude-gradient term. Replacing its square root
with a weighted mean is an additional equality condition, not an algebraic
identity. A potential or connection changes the Rayleigh formula; arbitrary
weights do not inherit M1. Nodes require a separate polar-domain treatment.
This theorem does not infer kappa or tau from a field, select a loop, or claim
that all geometric operators have this form.

**Explicit locked counterexample to the general average identity.** In
dimensionless arclength s on [0,2pi], take
chi(s)=cos(s)+i a sin(s), a>0 and a!=1. It is a nonvanishing eigenmode of H
with lambda=1. Here R²=cos²(s)+a²sin²(s), phi'=a/R²>0,
N=pi(1+a²), and <phi'>_w=2a/(1+a²)<1. This can even be realized as a smooth
closed planar curve with unit tangent chi/|chi|: the tangent reverses under
s→s+pi, so its integral over the period is zero. Its curvature is phi'>0
and torsion is zero. Thus the pointwise lock holds, yet the proposed weighted
mean is smaller than sqrt(lambda). For a=1/2 the two values are 4/5 and 1.
Restoring a reference length scales both by its inverse and preserves the
failure. This counterexample concerns the declared free arclength operator;
it does not identify that operator with a derived physical WCT confiner.

For the circle R_geom=L/(2pi), the free spectrum is n²/R_geom². A constant
amplitude single-winding mode obeys sqrt(lambda_1)=1/R_geom=kappa, tau=0.
It realizes M1's equality under its explicit assumptions. The same circle has
a constant n=0 eigenmode with lambda_0=0 despite kappa>0: it is a direct
counterexample to an unconditional geometry-to-all-modes mass claim. Higher
windings have |n|/R_geom and require their own mode assignment. Boundary twists
change eigenvalues without changing the curve's local curvature.

The earlier bounded-potential/holonomy spectrum theorem in
[stage 2](../FOUNDATIONAL_STAGE2_2026-10-14.md) remains valid and more general
as an operator bound; neither theorem independently selects the physical
operator. Reproduce the polar Dirichlet-energy identity in the foundational
script and the preexisting exact circle/holonomy tests. The proof, rather than
a residual-only calculation, establishes M1's equality conditions.

## Seven required inputs: current status

| Input | Current evidence | Remaining physical obligation |
|---|---|---|
| Confinement operator | Several precisely declared but different operators | Derive the relevant operator from a healthy parent and chosen background |
| Domain / boundary | Fixed-circle and torus examples are well-defined | Select a physical domain or prove whole-space binding |
| Mode assignment | Fourier labels known for specified operators | Target-blind particle correspondence, not nearest measured mass |
| Length scale | lambda_phys=lambda_hat/L0² | Select L0 independently; doubling L0 halves inferred mass |
| Action normalization | Classical EL insensitive to an overall constant | Supply physical energy/charge/quantum normalization |
| Conserved rest energy | S004 H is explicit but unbounded; other models have separate H | A finite-energy healthy localized solution and identified rest frame |
| Inertial response | Conditional KG mass shell above | Derive energy-momentum and force response for the selected physical state |

No electron, muon, tau or other measured mass was used to fill any of these
inputs. The v18 status remains `EMPIRICAL_FIT_ONLY`; neither its in-sample fit
nor its 11-by-31 local sensitivity supplies this missing physical bridge.
