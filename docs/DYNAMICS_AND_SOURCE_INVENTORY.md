# Dynamics and source inventory

Audit date: 2026-10-07 UTC. This is a new foundational audit, not an amendment of earlier results. Primary base: `geometry_of_resonance@a0048ef751a8876a9494f6de24b952ebc1db7213`. See [SOURCE_LEDGER.json](../audit/physical_domain/SOURCE_LEDGER.json) for immutable repository revisions, exact file hashes, manuscript identities, extraction coverage and limitations.

## Source precedence and scope

There is no single equation shared by every WCT asset. A result about one model is not transferred to another without a reduction theorem. This audit uses the corrected repository actions and the September 10 filament revision where they explicitly correct historical formulas; the October 6 Koide manuscript is authoritative for the active-domain proposal. A later upload date is not treated as a manuscript revision date.

| Source | Inspected version | Role and limits |
|---|---|---|
| The Geometry of Resonance | PDF internally dated March 22, 2026; governing-equation and curvature sections, not all 313 pages | Historical equations and claims; superseded formulas identified explicitly. The May 2025 TeX in `wave_confinement` is older still. |
| Corrected variational closure | `geometry_of_resonance`, pinned August 21 head, S004 | Complete formal variation of a declared real complex-field action; itself marked a derivation draft. |
| GR/constants derivation closure | Same head, S003 | Corrected U(1)-invariant curvature, conditional mode-to-mass separation, and stated scale-selection obligations. |
| Phase–Flux Field | PDF internally dated November 29, 2025, original September 8, 2025, P02 | Observable axioms, chosen gradient rail and shell rules. No automatic equivalence between their clocks or conserved quantities. |
| Scale–winding bijection | `bijection_UPDATED.tex`, internally revised August 20, 2026, P03 | Explicitly a fixed-domain algebraic correspondence; it does not require a WCT PDE. |
| Filament reduction | Revised September 10, 2026 TeX, P04 | Corrected complex log identity, vortex regularity, holonomy and fixed-sector locking; full dynamical stability remains open. |
| 2–3–6 Koide geometry | October 6 revised TeX, P05 | Exact finite-dimensional geometry; physical scale action and active-domain selection explicitly open. |
| WCT predicts Koide | September 10 revised thin-shell PDF, P06 | Conditional toroidal reduction, finite-thickness corrections and a still-required eigenmode correspondence. |
| Emergence of Effective Mass | PDF internally dated October 27, 2025, P07 | Guided-loop and geometric mass interpretation; supplied geometry is not intrinsically selected geometry. |
| Rest Energy from Density-Weighted Loop Curvature | PDF internally dated June 3, 2026, original November 11, 2025, P08 | Weighted locking variational problem at fixed curve and fixed integer; phase-only Hessian is not the full field Hessian. |

The Library manuscript reads are extracted text, not byte-identical original PDFs. Their exact extracted windows and hashes are recorded. Neither missing search hits nor this finite source inventory proves that no future or uninspected WCT extension can close the problem. The underdetermination result below concerns the explicitly specified models and premises reviewed here.

## Distinct mathematical models

Use spatial position `r`, physical time `t`, complex field phase `phi`, and curvature diagnostic `Theta`. The log-observable coordinate is `ell=ln(x/x0)`; it is not a spatial coordinate until a physical map is supplied.

### M0: historical additive curvature

`Theta_legacy = -Delta psi / (psi + epsilon exp(-alpha |psi|^2))`.

This is an operator definition, not a complete evolution equation. For epsilon,alpha>0 its denominator vanishes at exactly one negative-real field value, `psi=-sqrt(W0(2 alpha epsilon^2)/(2 alpha))`. The proof is monotonicity of `y exp(alpha y^2)` on y>0. It is also not invariant under arbitrary global phase rotations. These corrections already appear in P04 and S003; this audit does not claim them as new discoveries. Any admissible legacy solution must control the denominator away from that locus.

The historical manuscript P01 writes `(partial_t^2-c^2 Delta+c^2 W_psi)psi=0`, with an unregularized `W_psi=-Delta psi/psi`, and separately writes the mixed flow `psi_t=i(Delta psi-delta V/delta bar(psi))-gamma delta C/delta bar(psi)`. These are different dynamical formulations, not consequences of the curvature definition alone. The latter's asserted Lyapunov descent requires an actual variational and compatibility calculation. In general, for `z_t=J grad H-gamma grad C`, `dL/dt=grad L·J grad H-gamma grad L·grad C`; this is not automatically a negative square. The legacy literal square of a complex curvature is not generally real. A result for an exact real gradient flow cannot be imported into this mixed flow.

### M1: corrected real complex-field action

Let `rho=|psi|^2`, `D=rho+epsilon^2 exp(-2 alpha rho)>0`, `R_epsilon=bar(psi)/D`. The corrected operators are `Theta_0=-(Box psi) R_epsilon` and `Theta_s=-(Delta_h psi) R_epsilon`. They are invariant under constant phase rotations, and have inverse-length-squared units in the spatial case. They are not phase angles.

S004 defines `F(rho)=rho/D^2`, `Q=Box psi`, `P=Delta_h psi`, and

`H2=kappa Q bar(Q)+theta P bar(P)+(gamma/2)(Q bar(P)+P bar(Q))`.

Its real Lagrangian is `grad(bar psi)·grad(psi)-V(rho)+F H2`. On a flat fixed background with self-adjoint derivative operators and boundary terms cancelled, the Euler–Lagrange equation is

`-Box psi - V'(rho) psi + F'(rho) psi H2 + Box[F(kappa Q+gamma P/2)] + Delta_h[F(theta P+gamma Q/2)] = 0`.

Here `F'=(epsilon^2 exp(-2 alpha rho)-rho+4 alpha rho epsilon^2 exp(-2 alpha rho))/D^3`. With variable metric/foliation coefficients, formal adjoints and their coefficient derivatives must be included; the flat self-adjoint calculation is the verified scope here.

The curvature form is positive in its two operator arguments if kappa,theta>0 and gamma^2<4 kappa theta. This is not a proof of positive Lorentzian Hamiltonian or absence of extra time-derivative modes. The equation is generically fourth order in time when Box is retained. At psi=0, F=O(|psi|^2), so the curvature energy is quartic in field amplitude and supplies no linear fourth-order selector. V, the physical scale, admissible boundary data and an acceptable dynamical branch must still be fixed.

### M2: phase-invariant spatial companion energy

P04 separately defines

`E_delta=integral (|grad psi|^2 + |Delta psi|^2/(|psi|^2+delta^2)) dr`, delta>0.

This differs from both M0 and the squared corrected curvature of M1. A variational energy alone does not select a physical clock: gradient flow, Hamiltonian flow and a stationary constrained problem are different completions. The normal profile, norm, shell width and topology in the thin-shell reduction are additional assumptions. The positive free-space energy has a spreading minimizing sequence; see the domain report.

### M3: PFF/Swift–Hohenberg gradient rail

The complex PFF envelope uses `A=sqrt(u) exp(i phi)` and

`A_t=(r-a Delta-b Delta^2)A-beta |A|^2 A`.

Its spatial Fourier growth is `sigma(q)=r+a q^2-b q^4`, with `q_star=sqrt(a/(2b))` for a,b>0. This is an independently parameterized spatial scale. It is neither an active log-observable width nor a geometric dilation eigenvalue.

With the complex Wirtinger convention and energy `E=integral[-r|A|^2-a|grad A|^2+b|Delta A|^2+(beta/2)|A|^4]`, `dE/dt=-2 integral |A_t|^2`; a real-field energy with half quadratic coefficients gives the corresponding factor-one convention. The sign is controlled under boundary conditions cancelling every fourth-order boundary term, not under an unspecified second-order boundary label alone.

PFF also states physical continuity `u_t+div S=0` and `S=u grad phi` (a physical-units interpretation needs a mobility coefficient with units length^2/time). These cannot generally be the same isolated physical-time dynamics as the gradient rail with u=|A|^2: for a homogeneous field, S=0 but `u_t=2u(r-beta u)`. The exact witness u=1,r=2,beta=1 gives u_t=2. A reservoir, a different energy observable, or an auxiliary relaxation clock must be supplied. This is a new compatibility check, not a claim that such a completion is impossible.

`wct-pde`'s baseline implements the real scalar counterpart. A real A has no nontrivial complex U(1) phase winding. Its scope cannot be silently upgraded to the complex PFF system.

### M4: projection-free Hamiltonian rail

The actual maintained code S011 implements

`i psi_t=(Delta+k_star^2)^2 psi-beta |psi|^2 psi+gamma |psi|^4 psi`, beta,gamma>0,

with `H=integral[|(Delta+k_star^2)psi|^2-(beta/2)|psi|^4+(gamma/3)|psi|^6]` and `M=integral|psi|^2`. For smooth periodic/decaying solutions, H and M are conserved. There is no legacy curvature denominator in this equation. Its unit coefficients reflect a chosen nondimensionalization, not an independently calibrated physical clock or length.

On a node-free chart, write psi=A exp(i phi), let `Z=exp(-i phi)(Delta+k_star^2)^2(A exp(i phi))`. Then

`A_t=Im Z`, `phi_t=-Re Z/A+beta A^2-gamma A^4`.

This is a meaningful derived field phase evolution. It does not define a spectral phase as a function of dimuon mass or atomic wavenumber. Its mass current is

`j=-2 Im[bar(psi) grad Delta psi-(grad bar(psi)) Delta psi+2 k_star^2 bar(psi) grad psi]`,

so `rho_t+div j=0`; the fourth-order current generally differs from `rho grad phi`.

### M5: mass-projected selector

The older dissipative complex selector and its FEM reproduction use explicit norm projection. It is a constrained model. Its stationary/persistent output does not prove formation under M4 or M1. The repository documentation already makes this distinction; this audit preserves it.

## Dimensions and boundary data

For spatial dimension d, psi is a scalar or complex field on a declared spatial domain Omega with a time interval; a simulation's periodic torus is one such domain, not an inferred material interface. M1 is a spacetime action with an explicit preferred spatial foliation unless its timelike field is itself dynamical. Fourth-order spatial models need a complete pair of boundary conditions (or periodic/decay conditions); specifying only a generic Dirichlet label is insufficient. The companion-energy boundary calculation is in the domain report.

| Quantity | Units before nondimensionalization |
|---|---|
| Field phase phi | Dimensionless angle |
| Spatial phase gradient q | Length^-1 |
| Spatial curvature Theta | Length^-2 |
| PFF r,a,b,beta | Time^-1, length^2/time, length^4/time, 1/(field-amplitude^2 time) |
| epsilon and alpha in the corrected reciprocal | Field amplitude and inverse field-amplitude squared |
| Log-observable ell and its angular frequency | Dimensionless log ratio and radians per unit log ratio |

M4's normalized equation requires a coefficient with units length^4/time on the fourth-order term when dimensions are restored; its cubic/quintic coefficients must be restored consistently. The same letters reused in another model do not establish the same physical constants. A dimensional calibration or controlled reduction is required before comparing their spatial scales or clocks.

## Existing regularity and stability results

- The corrected denominator is strictly positive for finite complex fields. S016 contains an actual Lean proof of this local algebraic fact.
- For the polynomial Hamiltonian M4 in d=1,2,3 on a fixed torus or whole space, S044 supplies the H2 algebra/Duhamel argument for local well-posedness and the conserved-energy bound preventing finite-time H2 blow-up. Specifically, `||(Delta+k_star^2)psi||_2^2 <= H+3 beta^2 M/(16 gamma)` and `||Delta psi||_2^2 <= 2||(Delta+k_star^2)psi||_2^2+2 k_star^4 M`. With finite-energy data and standard conservation by smooth approximation, this gives global H2 well-posedness for that model. It does not establish localization or a selected physical state, and it is not a theorem for M1.
- H2 embeds into L-infinity for d<4 under the standard domain hypotheses. This controls amplitudes, not automatically every quotient, full curvature, global PDE trajectory, or all possible higher-dimensional mechanisms. In S014 the analytic embedding is a contract field; the downstream arithmetic theorem is not a kernel formalization of Sobolev analysis.
- On a fixed torus in d<=3, the positive-quintic M4 energy has a coercive fixed-mass bound. The direct method gives a minimizing set using compact embeddings; this does not identify that minimizer with a computed branch or prove intrinsic localization. Stability of a minimizing set additionally needs well-posed flow and conservation in the relevant function space.
- On whole space, binding and compactness modulo translations are separate requirements. A stationary virial identity supplies a necessary balance, not uniqueness or stability.
- The locking phase Hessian is positive modulo constant phase at fixed geometry, fixed positive weight and fixed winding. It does not test amplitude, shape, topology-changing or full PDE perturbations.
- Formal scale–winding round trips and locking identities accept their domains, winding and integrated quantities as inputs. They do not derive these inputs.

## Preserved numerical support

The additional `wct-pde` branch `d5335dc0fab77e14b66feb7e1a943c2c31eeaa19` was inspected because it contains later closure evidence absent from main. S044 reports accurate fixed-periodic-domain stationary roots, but the strict suite retains formation, domain, recovery and other failures. Its numerical existence witness is not a full closure verdict. The topology parent verdict remains `NO_VALIDATED_TOPOLOGICAL_SPECTRAL_SELF_BINDING` (S048). The inspected D4 refinement record S045–S046 is explicitly pending because original reference files were unavailable in that workspace; it cannot certify later Windows-only calculations.

FEM independently assembles the declared constrained/periodic equations and tracks projection provenance; no inspected FEM artifact supplies an intrinsic boundary plus winding plus observable map. Julia confirms properties of encoded finite-band symbols and their parameter non-identifiability. PERIODIC already proves a spreading obstruction for its positive legacy scalar energy, and its source-coupled extension adds a specified binding sector. `eigen` and `mass` define useful source-state gates; their validators are not a proof that a physical candidate passed those gates.

No original solver was executed. No continuum or all-time stability is inferred from finite-time output, and no imposed box or initial winding is described as an emergent physical prediction.
