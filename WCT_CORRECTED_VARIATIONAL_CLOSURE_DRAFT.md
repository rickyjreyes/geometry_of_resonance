# Corrected Variational Closure of the WCT Curvature Action

**Status:** derivation draft; exact for the corrected real complex-field action defined below.  
**Purpose:** replace schematic variational symbols by the complete fourth-order Euler–Lagrange equation and expose the remaining modeling choice required for finite-\(k\) selection.

## 1. Fields, geometry, and regularization

Let \(\psi:M\to\mathbb C\) and treat \(\psi\) and \(\bar\psi\) as independent variables in the variational calculation. Define

\[
u:=\bar\psi\psi,
\qquad
D(u):=u+\varepsilon^2e^{-2\alpha u},
\qquad \varepsilon>0,
\]

and the corrected complex-safe reciprocal and curvature operators

\[
R_\varepsilon(\psi):=\frac{\bar\psi}{D(u)},
\qquad
\Theta_0[\psi]:=-(\Box\psi)R_\varepsilon(\psi),
\qquad
\Theta_s[\psi]:=-(\Delta_h\psi)R_\varepsilon(\psi).
\]

Here \(\Box=\nabla_\mu\nabla^\mu\). The spatial operator is written covariantly relative to a chosen unit timelike field \(n^\mu\):

\[
h^{\mu\nu}:=g^{\mu\nu}+n^\mu n^\nu,
\qquad
\Delta_h\psi:=h^{\mu\nu}\nabla_\mu\nabla_\nu\psi.
\]

In a fixed inertial laboratory frame, \(\Delta_h\) reduces to the ordinary spatial Laplacian \(\Delta\). The use of \(n^\mu\) makes explicit that a separate spatial-curvature term selects a preferred foliation unless \(n^\mu\) is itself dynamical.

Because

\[
D(u)>0 \quad (\varepsilon>0),
\]

the quotient is nonsingular for every finite complex \(\psi\).

## 2. Corrected real action

The literal historical squares of complex quantities are not generally real. The real complex-field completion is

\[
S[\psi,\bar\psi]
=
\int_M \sqrt{|g|}\,\mathcal L\,d^{n+1}x,
\]

with

\[
\boxed{
\mathcal L
=
\nabla_\mu\bar\psi\,\nabla^\mu\psi
-V(u)
+\kappa|\Theta_0|^2
+\theta|\Theta_s|^2
+\gamma\,\operatorname{Re}(\Theta_0\overline{\Theta_s})
}
\]

or equivalently

\[
\mathcal L
=
\nabla_\mu\bar\psi\,\nabla^\mu\psi
-V(u)
+F(u)H,
\]

where

\[
F(u):=\frac{u}{D(u)^2},
\]

\[
Q:=\Box\psi,
\qquad
P:=\Delta_h\psi,
\]

and

\[
H
:=
\kappa Q\bar Q
+\theta P\bar P
+\frac{\gamma}{2}(Q\bar P+P\bar Q).
\]

For the curvature quadratic form to be positive definite pointwise, require

\[
\kappa>0,
\qquad
\theta>0,
\qquad
\gamma^2<4\kappa\theta.
\]

## 3. Derivative of the nonlinear weight

Let

\[
q(u):=\varepsilon^2e^{-2\alpha u}.
\]

Then

\[
D=u+q,
\qquad
D'=1-2\alpha q.
\]

Since \(F=uD^{-2}\),

\[
F'(u)
=
\frac{D-2uD'}{D^3}
=
\boxed{
\frac{q-u+4\alpha uq}{(u+q)^3}
}.
\]

The field variation is

\[
\delta_{\bar\psi}F
=F'(u)\psi\,\delta\bar\psi.
\]

## 4. Complete variation with respect to \(\bar\psi\)

For compactly supported variations, sufficiently rapid decay, periodic data, or boundary data that set both \(\delta\bar\psi\) and its normal derivative to zero, the kinetic and potential terms give

\[
\delta_{\bar\psi}S_{\rm kin+pot}
=
\int\sqrt{|g|}
\left[-\Box\psi-V'(u)\psi\right]
\delta\bar\psi.
\]

For the higher-derivative sector,

\[
\delta_{\bar\psi}(FH)
=
F'(u)\psi H\,\delta\bar\psi
+F\left(\kappa Q+\frac\gamma2P\right)\delta\bar Q
+F\left(\theta P+\frac\gamma2Q\right)\delta\bar P,
\]

with

\[
\delta\bar Q=\Box\delta\bar\psi,
\qquad
\delta\bar P=\Delta_h\delta\bar\psi.
\]

Integrating both second-derivative terms by parts twice yields

\[
\delta_{\bar\psi}S
=
\int_M\sqrt{|g|}\,\mathcal E_\psi\,\delta\bar\psi,
\]

where

\[
\boxed{
\begin{aligned}
\mathcal E_\psi
={}&-\Box\psi-V'(u)\psi+F'(u)\psi H\\
&+\Box\!\left[F(u)\left(\kappa Q+\frac\gamma2P\right)\right]\\
&+\Delta_h\!\left[F(u)\left(\theta P+\frac\gamma2Q\right)\right].
\end{aligned}
}
\]

Therefore the exact field equation is

\[
\boxed{
-\Box\psi-V'(u)\psi+F'(u)\psi H
+\Box\!\left[F\left(\kappa\Box\psi+\frac\gamma2\Delta_h\psi\right)\right]
+\Delta_h\!\left[F\left(\theta\Delta_h\psi+\frac\gamma2\Box\psi\right)\right]
=0.
}
\]

Equivalently, multiplying by \(-1\),

\[
\boxed{
\Box\psi+V'(u)\psi-F'(u)\psi H
-\Box\!\left[F\left(\kappa\Box\psi+\frac\gamma2\Delta_h\psi\right)\right]
-\Delta_h\!\left[F\left(\theta\Delta_h\psi+\frac\gamma2\Box\psi\right)\right]
=0.
}
\]

The equation from variation with respect to \(\psi\) is its complex conjugate.

## 5. Differential order

The terms

\[
\Box(F\Box\psi),
\qquad
\Delta_h(F\Delta_h\psi),
\qquad
\Box(F\Delta_h\psi),
\qquad
\Delta_h(F\Box\psi)
\]

are fourth order in derivatives of \(\psi\). If \(Q=\Box\psi\) is retained literally, the equation is generically fourth order in time as well as space. It is not a second-order-in-time equation unless one of the following is supplied:

1. a degenerate higher-derivative structure;
2. auxiliary fields plus constraints;
3. a purely spatial curvature energy in the dynamical action;
4. an effective low-frequency reduction that removes the extra branch.

This issue is structural and cannot be removed by notation.

## 6. Vacuum linearization: decisive consequence

Near \(\psi=0\),

\[
D(u)=\varepsilon^2+O(u),
\qquad
F(u)=\frac{u}{\varepsilon^4}+O(u^2).
\]

Since \(Q\) and \(P\) are linear in \(\psi\),

\[
FH=O(|\psi|^4).
\]

Consequently the curvature sector contributes no linear fourth-order term about the zero vacuum. The linearization there is only

\[
\boxed{
\Box\psi+V'(0)\psi=0
}
\]

up to the sign convention of the metric and potential.

Therefore the corrected quotient-squared action does **not** by itself produce the linear finite-band dispersion

\[
\sigma(k)=r+ak^2-bk^4
\]

about \(\psi=0\).

A finite-\(k\) selector can still arise in one of three explicitly distinct ways:

### A. Linearization about a nonzero background

For \(\psi=\psi_0+\eta\) with \(|\psi_0|>0\), \(F(|\psi_0|^2)>0\), and the fourth-order terms enter the linearized operator for \(\eta\).

### B. Separate Swift–Hohenberg rail

Retain an explicitly effective pattern-selection dynamics such as

\[
\partial_tA
=\mu A-g|A|^2A-b(\Delta+k_\star^2)^2A,
\]

and do not claim that this rail is the zero-vacuum linearization of the corrected curvature action until a reduction theorem is supplied.

### C. Revised regularized higher-derivative weight

Use a positive weight \(G(u)\) with \(G(0)>0\), for example

\[
\mathcal L_{\rm hd}=G(u)H,
\]

rather than identifying the entire higher-derivative sector with \(|\Theta|^2\). This preserves a linear fourth-order term but defines a different canonical action and must be justified independently.

## 7. Boundary data required by the variation

Because the action depends on second derivatives, a well-posed variational principle requires cancellation of boundary terms. Sufficient choices include:

- compactly supported variations;
- periodic boundary conditions;
- decay of fields and variations at spatial infinity;
- clamped-type data fixing \(\psi\) and its normal derivative;
- explicit higher-derivative boundary counterterms with natural boundary conditions.

A paper claiming exact closure must select one of these rather than silently dropping the boundary contribution.

## 8. Canonical status resulting from this derivation

This calculation closes the formal bulk variation for the corrected real complex-field action. It does **not** yet establish:

- global existence or uniqueness;
- absence of Ostrogradsky-type extra modes;
- positivity of the Hamiltonian in Lorentzian signature;
- existence of a localized stationary solution;
- spectral or orbital stability;
- the mass-curvature law;
- equivalence between this action and the Swift–Hohenberg rail.

The immediate unresolved canonical choice is now precise:

\[
\boxed{
\text{Is finite-}k\text{ selection generated around a nonzero background, imposed as an effective rail, or built into a revised }G(0)>0\text{ action?}
}
\]

## 9. Recommended next derivation

Choose a constant background \(\psi_0=A_0e^{-i\omega_0t}\), expand

\[
\psi=\psi_0+\eta,
\]

retain all terms quadratic in \(\eta\), and derive the exact Fourier-space dispersion matrix. That calculation will determine whether the corrected action possesses a real finite unstable band while satisfying

\[
\gamma^2<4\kappa\theta.
\]

It is the next direct bridge between the finished Lagrangian and the claimed WCT finite-band mechanism.
