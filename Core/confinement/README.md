# Corrected WCT 3D confinement numerical audit

This directory contains nondimensional numerical experiments testing whether the corrected WCT curvature sector can support a localized fixed-mass state after adding a positive UV rail and focusing interaction

\[
E[\psi]=\int_{\mathbb R^3}\left(|\nabla\psi|^2+\left[\delta+F(|\psi|^2)\right]|\Delta\psi|^2-\frac g2|\psi|^4\right)\,d^3x,
\]

with

\[
F(u)=\frac{u}{\left(u+\varepsilon^2e^{-2\alpha u}\right)^2}.
\]

These are numerical variational tests. They do **not** establish empirical validity of WCT and do **not** replace a continuum existence or stability proof.

## Reference parameter set

- spatial dimension: 3
- fixed mass: `M = 1`
- `epsilon = 0.5`
- `alpha = 1`
- `delta = 0.1`
- representative localized branch: `g = 120`

## Main numerical results

Direct minimization of

\[
g_c=2\inf_{\|\psi\|_2^2=M}\frac{P[\psi]}{\|\psi\|_4^4}
\]

gives

\[
g_c^{\rm num}\approx105.098413.
\]

At `g = 120`, the polished radial candidate has approximately

- energy: `-0.112018558`
- RMS radius: `1.893`
- fixed mass: `1`
- projected constrained-gradient residual: `~1e-6`
- Pohozaev relative residual: `~2.2e-4` at `N=160`, decreasing to about `9.4e-5` at `N=240`.

The negative energy matters because mass-preserving spreading tends toward energy zero. The tested localized branch therefore cannot lower its energy simply by dispersing to infinity.

## UV-collapse control

With `delta = 0`, normalized narrow probes drive the discrete energy rapidly downward as their width shrinks, and the computed critical coupling trends toward zero with refinement. With `delta = 0.1`, the same probes become strongly positive at small width. This numerically supports the role of the positive `delta |Delta psi|^2` term as the UV-stabilizing rail.

## Radial and nonspherical stability

The constrained radial Hessian is positive on the tested mass-preserving radial tangent space.

For nonspherical perturbations

\[
\psi=\phi(r)+\eta v(r)Y_{\ell m}(\theta,\varphi),
\]

the shape sectors `l = 2..20` all have strictly positive smallest constrained Hessian eigenvalues in the tested discretization. The small negative `l = 1` value converges toward zero under refinement and its eigenvector converges to `dphi/dr` with greater than 99.999% overlap on fine grids, identifying it as the discretized translational Goldstone mode rather than a shape instability.

Large axisymmetric nonlinear deformations were also tested in a truncated `m = 0`, `l <= 6` spherical-harmonic basis. All tested starts substantially relaxed toward the spherical branch. At the packaged production stopping criteria, the strongest start with 80% initial quadrupole mass finished with about `1.6e-3` nonspherical mass; easier deformations finished at roughly `1e-5` to `1e-4` scale.

## Files

- `wct_radial_confinement_test.py` — radial fixed-mass solver and Pohozaev diagnostic.
- `wct_radial_confinement_suite.py` — reruns the critical-coupling calculation, polished `g=120` stationary state, and UV-control probes. Archived result files also record separate grid/domain and sensitivity sweeps performed with the same radial discretization.
- `wct_nonspherical_stability.py` — spherical-harmonic constrained second-variation audit.
- `wct_nonlinear_nonspherical.py` — nonlinear axisymmetric deformation/relaxation test through `l=6`.
- `results/*.json` — machine-readable summaries.
- `results/*.csv` — threshold, convergence, stability, and deformation tables.

## Dependencies

Python 3 with `numpy`, `pandas`, `scipy`, and `torch`.

## Current limitation

The strongest remaining numerical attack is the full complex-field sector: phase perturbations and mixed amplitude-phase perturbations. The present nonspherical tests cover real spatial perturbations and axisymmetric nonlinear deformations, not full 3D orbital stability of the complex WCT evolution.
