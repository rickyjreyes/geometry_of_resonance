# Sector energy and winding selection

## What can be reconstructed exactly before a branch is solved

For a static cylindrical cross-section ψ=f_n(r)e^{inφ}, define C=κ+θ+γ and F as in S004. The energy per unit longitudinal length in the declared static sector is

\[
\boxed{E_n[f_n]=2\pi\int_0^\infty r\,dr\left[(f_n')^2+\frac{n^2f_n^2}{r^2}
-V(f_n^2)+C F(f_n^2)\left(f_n''+\frac{f_n'}r-\frac{n^2f_n}{r^2}\right)^2\right].}\tag{W1}
\]

Subtract the same vacuum density when necessary. Smooth cores require the appropriate f_n∼r^{|n|} behavior or another declared regular core; asymptotic conditions must make the energy meaningful. For nonzero constant amplitude at infinity a global ungauged vortex has a logarithmically divergent gradient energy per length, so a finite energy claim requires a finite loop, finite relative domain, screening, or different asymptotics. None is supplied by a carrier wavelength.

W1 is a new exact assembly of **all terms actually present in S004 for this static ansatz**. It is not the general higher-time Hamiltonian and contains no independently invented locking, entropy, electromagnetic, or holonomy energy. A rotating or time-dependent sector requires its complete higher-derivative energy and charge. To define E_n physically one must solve the constrained sector variational problem and assess its full Hessian. No profiles f_n, admissible finite-energy three-dimensional loop, or frozen preparation have yet been obtained from the new homogeneous amplitude result.

The complete sector selector is therefore not calculated. Even if n↦inf E_n existed, decay barriers and preparation can prevent an actual trajectory reaching the global minimum. For an example potential U=−V≥0 with a zero-energy uniform vacuum, n=0 vacuum is available without an imposed topological/charge constraint. This example does not establish a universal ordering for all V or all constrained sectors.

## Existing locking component and its exact minimum

P08/P22, §3 Eq. (3.1), already gives on a supplied loop Γ of length L_s

\[
E_{\rm lock}=\oint w(s)[\phi'(s)-\sigma(s)]^2ds,\qquad\oint\phi' ds=2\pi n,
\quad w>0.
\]

Let S_σ=∮σ ds and I_w=∮w⁻¹ds. Variation gives w(φ′−σ)=constant, hence

\[
\phi'=\sigma+\frac{2\pi n-S_\sigma}{I_w w},\qquad
E_{\rm lock,min}(n)=\frac{(2\pi n-S_\sigma)^2}{I_w}.\tag{W2}
\]

The nearest-integer rule minimizes this **component** for fixed loop and weight. At half-integer ties it is degenerate. Both the geometry and w may depend on n when solving a physical branch, and W1 is not shown to reduce to W2 plus specified remaining terms. The rest-energy identity E_rest=ℏc k_eff and weighted curvature estimator already in P22 are conditional physical identifications; their constants do not independently fix a S004 loop.

## Topological changes already present in the corpus

PFF's bridge dynamics (§3 and §6) explicitly permits unwinding when an amplitude reaches zero. P14's revised §8 distinguishes dynamical stability from topological protection. Thus phase slips/zero crossings are **already in the corpus**. A transition rate, barrier, drive, or initial-state distribution selecting one n is still required for a dynamical prediction. Imposing n=1 is preparation, not an outcome of W1.

The Coulomb paper P20 §§5–7 obtains θ=C/r outside a prescribed core from an approximate phase-gradient energy. Its inverse-square field and 1/R tail energy follow under those premises. But its step from ∮∇θ·dl=2πn and ∮∇θ·dS=4πC to C=n/2 is invalid: circulation and spherical flux are different integrals. For θ=C/r every closed-path circulation in the punctured three-dimensional exterior is zero while the flux is nonzero and C arbitrary (up to orientation sign). Thus that equation cannot fix a charge/sector normalization. A Maxwell connection or another physical map would have to be derived separately.

Status: W1 `DERIVED_BY_CROSS_PAPER_COMPOSITION` for a specified static ansatz; W2 and zero-crossing permission `DERIVED_IN_EXISTING_PAPER`; complete sector ordering and physical n_star `NOT_DERIVED`. Gap class `CROSS_PAPER_REDUCTION_REQUIRED`, with state preparation an independent physical input. Falsifier: a proposed full E_n omits terms of W1, adds energies without a common action, or promotes W2's fixed-geometry rule to a universal sector selector.
