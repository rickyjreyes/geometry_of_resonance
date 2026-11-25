# Spectral observable mapping

**Result: `SPECTRAL_MEASUREMENT_MAP_NOT_DERIVED`.** WCT's complex field has a meaningful phase where it is nonzero. Neither the curvature diagnostic nor that spatial phase has been shown to be the phase of an empirical residual as a function of log mass, atomic wavenumber or another measured variable.

## The missing forward model

A physical prediction needs a specified map such as

`y(ell)=B(ell)+integral R(ell,z;C) O[psi](z;C) dz`,

including the source/population measure, observable O, physical map into x, reference x0, detector response R and background B. It then needs a justified oscillatory representation and estimator. No inspected foundational source derives that complete map for the six empirical systems. This task uses their coordinate conventions only; no fitted empirical frequency enters any derivation or symbolic check.

Dimensions already distinguish the objects: spatial phase gradient q has units inverse length; curvature Theta has units inverse length squared; an angular frequency in `ell=ln(x/x0)` is dimensionless (radians per unit log ratio). A length-to-observable relation and any physical unit calibration cannot be suppressed.

For a node-free field psi=A exp(i phi),

`-Delta psi/psi=|grad phi|^2-Delta A/A-i[Delta phi+2 grad ln(A)·grad phi]`.

The corrected complex-safe curvature multiplies this expression by `A^2/D(A^2)`. Thus Theta is not phi. The exact polar dynamics for the specified M4 PDE are recorded in the dynamics inventory; they live in space and time, not automatically in a log-observable coordinate.

## Proposition O1: one field does not determine an unspecified measurement

The same plane wave can give constant intensity `O1=|psi|^2`, fringes through the interferometric observable `O2=|psi+psi_ref|^2`, or higher harmonics through nonlinear response. These are physically different, explicitly defined measurements compatible with the same field equation. A phase-sensitive measurement needs its reference. The PDE alone therefore cannot choose which observed frequency, if any, is measured.

Similarly, the exact conditional loop mass ladder `M_n=hbar |n|/(cR)` has adjacent ratios `(n+1)/n`, not a fixed geometric ratio. Integer spatial winding by itself does not create a log-periodic mass spectrum. An exponential spectral recurrence would need a separate operator/measurement theorem.

## When can fitted and mean phase frequency agree?

Suppose the **observable** phase is exactly affine, `phi_O(ell)=k0 ell+phi0`, the fitted amplitude/background model is correct, the retained support identifies frequency without aliasing, and either detector response is known or it is translation-invariant with nonzero transfer at k0. In the noiseless identifiable model, fitting returns k0, and the endpoint mean gradient also equals k0. Physical winding still additionally requires a real endpoint closure law.

For nonaffine phase, equality is not a general theorem. A small-error bound can be calculated only after fixing the estimator, weights, support, nuisance projection and conditioning. For weighted least squares, let `s=partial_k y_model`, let s_perp be its projection orthogonal to nuisance scores, and perturb the response by delta_y. Then the local first-order shift is

`delta k=<s_perp,delta_y>/||s_perp||^2`,

with `|delta k|<=||delta_y||/||s_perp||` at first order. The remainder requires derivative bounds and a nonsingular local Hessian. Endpoint phase differences are a different functional, so neither this bound nor its uncertainty can be obtained from endpoint closure alone.

## New explicit counterexample O2: exact closure with a shifted fitted frequency

On the mathematical test interval ell in [0,1], take

`phi_epsilon(ell)=2pi ell+epsilon h(ell)`,

`h(ell)=ell(1-ell)(ell-1/2)`, `y_epsilon=cos(phi_epsilon)`.

Because h(0)=h(1)=0, `bar k=Delta phi/L=2pi` exactly for every epsilon: the endpoint winding remains one. Fit `a cos(k ell+phi0)` by uniform continuous least squares, with amplitude and phase free and known zero background. Near epsilon=0, the score for k is `-ell sin(2pi ell)`. Removing the amplitude/phase nuisance span gives

`s_perp=(1/2-ell)sin(2pi ell)-cos(2pi ell)/(4pi)`.

The nondegenerate normal equations and the implicit-function theorem give

`k_fit=2pi+epsilon C_fit+O(epsilon^2)`,

`C_fit=(16pi^4+60pi^2-225)/(160pi^4-360pi^2)=0.160044729544955... !=0`.

The symbolic script derives this coefficient by exact integration. For pi^2>9 the numerator and denominator are strictly positive. This is an explicit local estimator counterexample even with perfect coverage, no noise, no background uncertainty and no detector smearing. It is not a simulated WCT solution or an empirical fit; it disproves the proposed identification from phase closure alone.

## What each complication requires

| Effect | Consequence / required treatment |
|---|---|
| Nonstationary phase | Use the actual estimator response; O2 disproves universal equality to endpoint mean. |
| Finite domain | Window transform broadens spectral responses; identifiability and uncertainty depend on support. |
| Windowing | A crop changes the weighting of a fixed field, not necessarily its physical dynamics. |
| Disconnected support | Relative/gap phases and aliases require a declared matching law. |
| Mode mixing | Several modes or harmonics can move a fitted single peak without changing any topological degree. |
| Detector response | Constant convolution preserves a pure frequency when transfer is nonzero; nonstationary response, boundaries and acceptance can mix/bias modes. |
| Physical projection | Field intensity, current, transition rates and interference have different phase dependence. O must be derived. |
| Coordinate transformation | k rescales reciprocally to the log-coordinate stretch; this is a representation change. |
| Background subtraction | A misspecified nuisance family can absorb or create a fitted oscillation; projection and calibration must be included. |

## Conditional covariance lemmas

These verify algebra only. Since the physical W and winding law failed identification, no empirical transfer is calculated.

**T1 — translation.** For a stationary phase k ell+phi0, translating ell by c changes its intercept and leaves k and L unchanged. Transporting an entire nonstationary phase field and both endpoints together also preserves its phase increment and mean gradient. Sliding endpoints through a fixed nonstationary field may change the mean; that is a crop operation, not dynamical width covariance.

**T2 — width.** If two independently physical, oriented domains have L1,L2>0 and the same closed phase increment 2pi n, then `bar k2=bar k1 L1/L2`. The same-n premise is conditional; it is not supplied by the inspected evolution equations for different experiments.

**T3 — winding/boundary change.** Under integer closure and n1!=0, `bar k2=bar k1(L1/L2)(n2/n1)`. If n1=0 use `bar k_i=2pi n_i/L_i` directly. With twists/holonomy, replace 2pi n_i by the complete physical phase increment. The dynamics must predict these increments; choosing them after a result makes no prediction.

**T4 — coordinate power.** Define the dimensionally correct transformation `x'/x0'=(x/x0)^p`. For p>0, `ell'=p ell`, `k'=k/p`, `L'=pL`, and k'L'=kL. For p<0 the orientation reverses: ordinary positive width is |p|L and the signed degree changes if endpoints are reordered. p=0 is noninvertible. This is not a second observation. A hypothetical root-mass gain sqrt(6) implies mass gain 6, with corresponding reciprocal log-frequency conversion.

**T5 — conditional uncertainty.** If source k1,L1,L2 are identified, `k_pred=k1 L1/L2` has delta-method variance `g^T Sigma g`, `g=(L1/L2,k1/L2,-k1 L1/L2^2)`. Transfer variance also includes covariance with the target estimator. Discrete branch uncertainty must be treated as such. There is no numerical uncertainty to propagate for the present null prediction.

The missing theorem is a forward observable map plus a controlled estimator correspondence. More precise fitted peaks cannot establish either by themselves.
