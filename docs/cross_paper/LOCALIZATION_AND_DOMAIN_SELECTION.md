# Finite wavelength, localization, and physical domain

## Strongest existing chain recovered

The revised closure paper P09 is substantially stronger than an argument from a Fourier ring alone. Its §4 Eq. (8) adopts, in three spatial dimensions,

\[
\mathcal F_{\delta_r}[\psi]=a_4\|\Delta\psi\|_2^2-a_2\|\nabla\psi\|_2^2
+\tfrac\beta2\|\psi\|_4^4+\kappa_r J_{\delta_r}[\psi],\qquad
J_{\delta_r}=\int\frac{|\Delta\psi|^2}{|\psi|^2+\delta_r^2},
\quad\|\psi\|_2^2=M_N.
\tag{L1}
\]

All five coefficients are positive. They are **not** S004's identically named couplings. If [ψ]=A and [F_energy]=E, then [a₄]=EL/A², [a₂]=E/(A²L), [β]=E/(A⁴L³), [κ_r]=EL, [δ_r]=A and [M_N]=A²L³. These units make every term consistent. The evolution needs a separate mobility/time normalization.

P09 Theorems 5.2–5.3 prove fixed-mass boundedness below/coercivity by Fourier interpolation. Its §7 derives the exact fourth-order quotient variation, and §8 derives exact mass conservation and Lyapunov descent for a projected gradient flow. Its §10 proves a minimizer on a compact periodic domain, explicitly distinguishing that cavity result from free-space confinement. These are **already in the corpus**, not missing physical ideas.

For ψ_R=R⁻³ᐟ²ψ(x/R), with R a dimensionless dilation ratio, G=||∇ψ||²,H=||Δψ||²,Q=||ψ||₄⁴,

\[
\mathcal F[\psi_R]=a_4HR^{-4}-a_2GR^{-2}+\tfrac\beta2QR^{-3}
+\kappa_rR^{-4}\int\frac{|\Delta\psi|^2}{R^{-3}|\psi|^2+\delta_r^2}.\tag{L2}
\]

P09 §6 proves collapse costs +∞, the large-R limit approaches zero from below, and every fixed nontrivial shape orbit has a finite minimizing radius. It correctly leaves compactness of general shape-changing minimizing sequences open in §11. This is a real variational scale result within L1; it is not yet a selected free-space field or a unique hard boundary.

## New explicit minimization within the existing closure functional

For a bounded fixed shape in the regime e_R=||ψ||∞²/(δ_r²R³)≪1, L2 becomes

\[
E(R)=A R^{-4}-B R^{-2}+C R^{-3}+O(\kappa_r\delta_r^{-4}\|\psi\|_\infty^2HR^{-7}),
\]

where A=(a₄+κ_r/δ_r²)H, B=a₂G, C=βQ/2. The stated bound follows directly from |(δ_r²+x)⁻¹−δ_r⁻²|≤xδ_r⁻⁴. Its derivative has the corresponding controlled O(R⁻⁸) remainder for fixed bounded shape. The leading orbit minimum is

\[
\boxed{R_{\rm orb}=\frac{3C+\sqrt{9C^2+32AB}}{4B},\qquad
E''(R_{\rm orb})=\frac{4BR_{\rm orb}-3C}{R_{\rm orb}^5}>0.}\tag{L3}
\]

Only use this approximation if e_R≪1 at the resulting radius; an arbitrary root outside that regime has no controlled status. Standard nondegenerate-root perturbation bounds its correction by the derivative remainder divided by this positive Hessian. Physical radius additionally depends on the chosen shape's reference length. This new formula completes a latent **fixed-shape** minimization; it does not complete the global existence theorem.

Another useful correction: P09 Eq. (11) concerns the first two terms of L1. The **full weak-field** quadratic symbol also receives κ_rJ: (a₄+κ_r/δ_r²)q²−a₂q. Therefore its weak-field center is a₂/[2(a₄+κ_r/δ_r²)]. For the declared real-gradient flow the corresponding leading coefficients are a_PFF=2a₂ and b_PFF=2(a₄+κ_r/δ_r²), with a global multiplier λ[ψ] supplying the mass-preserving scalar term. It is not a freely prescribed constant r at fixed mass. Nonlinear quotient terms add derivative-dependent cubic interactions.

## The remaining variational bridge is precise

The static S004 energy has +||∇ψ||², U=−V and the weight F=u/(u+ε²e⁻²αu)². L1 deliberately has a negative gradient term and a different quotient. In particular F(0)=0 whereas (u+δ_r²)⁻¹ is nonzero at u=0. Thus L1 is not the literal static restriction of S004. E1–E4 match its quadratic finite-band form in a canonically normalized amplitude sector, but do not match the full nonlinear functional, the fixed L² constraint, or its relaxation flow. S004's U(1) charge, with higher-derivative contributions, is not automatically ||ψ||². Establishing the charge/mass identification and nonlinear reduction is `CROSS_PAPER_REDUCTION_REQUIRED`.

There is also a more stringent binding threshold than e(M_N)<0. On broad modulated packets ψ_R=R⁻³ᐟ²f(x/R)e^{ik₀·x}, the quartic term vanishes while J tends to δ_r⁻²k₀⁴M_N. Minimizing over k₀ gives the vanishing-sequence threshold

\[
e_{\rm van}(M_N)=-\frac{a_2^2M_N}{4(a_4+\kappa_r/\delta_r^2)}.\tag{L4}
\]

The packets have vanishing local mass and bounded energy while retaining a carrier wavelength. Thus negativity alone does not exclude vanishing; a binding argument must compare with this threshold and exclude splitting, for example by suitable strict subadditivity. L4 is a new deduction from L1, not a proof that minimizers are impossible. P09's remaining compactness theorem and P14's revised fixed-point framework already recognize the distinction between existence, selection, stability, and continuum survival.

## What the new homogeneous finite-k result alone proves

On all of Rᵈ, a constant-coefficient polynomial in −Δ has continuous Fourier spectrum. An L² eigenfunction at one isolated level would have Fourier support on a measure-zero level surface, hence vanish. A finite-k minimum alone does not yield an isolated localized eigenstate. A localized wave packet may be prepared, but its envelope size is not selected by the carrier wavelength. This is an exact obstruction within the **linear homogeneous** class, not a nonlinear no-go theorem.

The older Compact Invariant Dynamics paper P15 Appendix A assumes annular Fourier support, bounded energy and nontrivial curvature-weighted mass, then claims no vanishing or dichotomy. Those hypotheses as written are insufficient. Choose band-limited Schwartz f with Fourier support strictly inside the annulus and take small amplitude to respect its denominator bound. With a constant positive weight σ, f(x−ne₁)+f(x+ne₁) splits while keeping those bounds. Alternatively R⁻ᵈᐟ²f(x/R)e^{ik₀·x} vanishes locally while remaining in an annulus for large R. The stated theorem gives σ=√(κ²+τ²) but no constraint forcing these geometric functions to track each field in a way that excludes this constant-weight countermodel. An intended additional field–geometry relation must be supplied and tested. In any case, the printed inference from local boundedness of σ to vanishing total weighted mass is invalid. Additional binding hypotheses are necessary. This does not undermine P09's correctly qualified newer orbit theorem.

## Entropy and interface route

Geometry P01 §28 pp. 87–88 writes the normalized Shannon functional S=−η∫p log p, p=|ψ|²/N, and a penalty in the Lagrangian; §40.3 p.129 describes entropy maximization resisting concentration. These imply different energy signs unless conventions are settled. The actual Wirtinger derivative is −ηψ/N(log p−⟨log p⟩); the real gradient is twice that. Log density requires a reference density; this changes constants, not dilation derivatives.

If the physical energy **penalizes** Shannon spread by +τS₀, then for fixed N, S₀[ψ_R]=S₀[ψ]+d log R. A controlled fixed-shape model

\[
E(R)=A/R^2+B/R^p+\tau d\log R+E_0,\quad A,B,\tau>0
\]

has a unique root τd=2A/R²+pB/Rᵖ and E″=4A/R⁴+p²B/R^{p+2}>0. For S004 at small amplitude with denominator dominated by ε², p=d+4 and the coefficient error is controlled by sup(u/ε²+|α|u)≪1. This is a new conditional balance using an existing entropy ingredient. If the energy is −τS₀, all displayed forces favor spreading instead. The sign must be selected physically, not to obtain a desired radius.

In the simpler zero-curvature log model, E=||∇ψ||²+τS₀ at fixed N has the stationary Gaussian ψ∝exp(−r²/(2a²)), a²=N/τ, after a Lagrange multiplier absorbs constants. A local logarithmic potential can encode that model at fixed N; nonzero S004 curvature and changes of N alter the equation. This example proves compatibility of an entropy-assisted radius with existing ingredients, not a stable solution of the full corrected theory. Neither a diffuse Gaussian nor a tension-zone description fixes two operational endpoints of an observable domain.

P17's electron-soliton Appendix B openly uses the surrogate F_geo(η)=η−λlog(1+η), κ_EM=κ₀η/(1+η): η*=λ−1 and α=[κ₀(λ−1)/λ]²/(4π). This selects a dimensionless shape given free coefficients, not an absolute radius from S004. Its Appendix C additionally supplies core matching, normalization and a sector. These are useful candidate premises, not parameter-free closure.

Route D status: `CONDITIONAL` with existing variational mechanisms; the nonlinear S004 match and global binding remain open. No missing entropy concept is asserted. No PDE simulation was run.
