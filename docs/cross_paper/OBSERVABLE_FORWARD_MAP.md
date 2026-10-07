# Observable and estimator map

## A concrete conditional response prediction now available

Prepare the homogeneous on-shell amplitude background of Q6, declare its parameters before observation, and couple a weak calibrated source J so the **original amplitude equation** is 𝒟(ω,k)r=J. This equation defines the normalization of J (equivalently the corresponding linear source term in the quadratic action). PFF's local intensity u=|ψ|² gives δu=2A₀r+O(r²). Therefore

\[
\boxed{\chi_{uJ}(\omega,k)=\frac{2A_0}{\mathcal D((\omega/c_0)^2,k^2)}.}\tag{O1}
\]

Specify retarded preparation and the low-sector cutoff to select the observable response. With weak damping, an experimental transfer-function peak approaches ω₋(k)=c₀√z₋(k); without damping an ideal pole is a distribution, not a finite peak height. The peak follows F2 only if the drive and detector have nonzero coupling there and smooth frequency response, with other modes separated. Low-pole residues depend on k through √disc. The canonical EFT field χ is not the measured r and that transformation must be retained.

This is an analytic **conditional linear-response forward map**, new to the audit. It predicts a dispersion minimum for an independently prepared, parameter-frozen state. It is not a prediction of particle masses or a fitted log-periodic cosine. No empirical frequency enters O1.

## Full log-observable map and required data

The general map requested by the audit is

\[
y(\ell)=B(\ell;\zeta)+\int R(\ell,z;C)\,\mathcal O[\psi_C](z)\,dz+\epsilon_y,
\qquad\ell=\log(x/x_{\rm ref}),\quad x=X[\psi_C].\tag{O2}
\]

Here C declares the action parameters, physical source J, initial/boundary preparation, response calibration and selection function. ζ denotes background parameters. For the concrete response experiment, O=|ψ|², x is a calibrated positive drive frequency or momentum, R is the measured instrumental convolution, and B is a independently specified baseline intensity/response. The estimator is the peak location of the transfer function J→δu, with resolution corrections. A spatial carrier minimum does not force periodicity in log x.

For a conditional DSI experiment instead, all of the following must be shown from its preparation and response:

1. The field observable has a nonzero fundamental Fourier coefficient A₁ in ℓ at ν fixed by D1, on the independently obtained domain.
2. The coordinate map has the form required by D2. A nonlinear coordinate warp can change the local frequency.
3. If R is translation invariant in ℓ, its transfer function R_hat(ν) is nonzero. A response annihilating the fundamental can make a harmonic dominant.
4. If O is nonlinear in the field, its harmonic content is calculated. For example a squared real cosine has a constant and a 2ν harmonic, whereas the intensity of a pure complex phase wave is constant.
5. B, windowing, resolution, nuisance fitting and the chosen estimator are fixed independently. The estimator consistently targets ν rather than whichever harmonic or background feature best fits a restricted interval.

Only under these conditions does the response preserve the internal ν as the fitted frequency. This is the controlled conditional use of the existing P03 theorem; it respects the earlier counterexample. No R,B,X,window or source has been selected to reproduce an external dataset.

## 2–3–6 operator test on the actual state spaces

The geometric map in P05/P06 is T:R→R³ with T†T=6 and singular value √6. On the derived homogeneous fluctuation space the natural operators are the translation-invariant amplitude/phase dispersion operators; on a bound branch they would be the corresponding Hessian and symmetry generators. None has been shown to implement this geometric T as a physical dilation.

The suggested D I=I T is not well-typed with one common embedding when T has different input/output spaces. A legitimate formulation needs I_in:R→H and I_out:R³→H and D I_in=I_out T, or an independently defined return map making an endomorphism. TT† has a nonzero eigenvalue 6 and two zeros; this exact finite-dimensional fact supplies no physical embeddings, invariant state, length ratio or generator. A mathematical dilation on all-space L² can be defined, but the dimensionful S004 couplings and its chosen background do not single out λ=6 or √6.

Status: O1 `CONDITIONAL` with an exactly derived susceptibility; O2/DSI estimator equality `CONDITIONAL`; physical 2–3–6 intertwiner `NOT_DERIVED` (the untyped proposed equation is `OBSTRUCTED` as written). Gap class `CROSS_PAPER_REDUCTION_REQUIRED`. A detector/preparation coupling is a physical input absent from the bare action, not something algebra can fix. `SPECTRAL_MEASUREMENT_MAP_NOT_DERIVED` remains for the original physical prediction chain, while this separate response protocol is now explicit.
