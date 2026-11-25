# Cross-paper derivation ledger

Audit date: 2026-10-07. New branch: `agent/wct-cross-paper-derivation`, based on `a84a4f0745d8e440961797ab8da809ae8598a836`. All prior files are preserved. This ledger distinguishes existing equations from new deductions and independently supplied physical premises. Exact source versions, retrieval windows and hashes are in `audit/cross_paper/SOURCE_LEDGER.json`.

The search covered the requested themes throughout the indexed corpus, resolved original paper filenames through the research catalog, and read the relevant primary equations. It retains 48 pinned repository files and 22 identified paper sources. It does not claim access to every unpublished/unindexed document. Geometry was expanded to full extracted-text coverage; the current closure, Hessian and fixed-point revisions were read, not inferred from older audit summaries.

## Existing equations and their declared role

| Source | Section/equation | Equation or construction | Role and provenance |
|---|---|---|---|
| S004 | Corrected Variational Closure §§1–4 | D=u+ε²e⁻²αu, F=u/D²; complete fourth-order EL | Fundamental candidate action; permits unspecified V; source of Q1–Q7 |
| P01 | Geometry §28; §40.3; Appendix Y.1 Eq504 | Normalized entropy, tension zone, (Δ+k_star²)² EFT | Mixed heuristic/effective ingredients; the EFT coefficient map was not supplied |
| P02 | PFF §§2.3–7, Appendix R | Gradient rail, σ=r+ak²−bk⁴, O_Theta, phase slips, SH/CGL | Rail and band derived within chosen model; radial potential introduced kinematically |
| P03 | Bijection/scale–winding theorem and stated operator burden | nu Delta ell=2pi n; lambda=exp(Delta ell/n) | Conditional theorem; physical confined log-mode premise explicit |
| P04 | Filament localization corrected construction | Madelung/Cole–Hopf, logarithmic and filament relations | Geometric identities and conditional field construction; not a S004 bound-state theorem |
| P05/P06 | Koide/2–3–6 map and thin-shell geometry | Tdagger T=6; torus/shape relations | Exact geometric algebra plus separate physical assumptions |
| P07 | Effective mass from curvature/dispersion | Gap and geometric effective-mass identifications | Requires same branch, kinetic normalization and physical interpretation |
| P08/P22 | Rest Energy §§1–3 | Erest=hbar c keff; weighted lock; sampled estimator | Loop and weight supplied; conditional physical map with exact locking minimization |
| P09 | Closure §§4–11, Eqs8,19,24,29,35–37 | Fixed-mass functional, scaling, exact EL, projected flow | Current revision proves orbit scale/cavity existence and explicitly leaves R3 binding open |
| P10 | Revised Hessian §§3–7 | Delta(c2 h Delta)−div(R grad)+W_eff | Exact scalar Hessian; includes Delta V1/2 and withdraws universal topology/ellipticity shortcut |
| P11 | Earlier regime/topology arguments | Local coefficient and trial-width claims | Historical candidate; P10 corrected scope controls interpretation |
| P12 | Resonant Cavity §§1.2–1.4,2.1–2.5 | Fourth-order dispersion/cutoff; chosen SH growth; cavity projection | Different real scalar action and damped simulations; no S004 damping map |
| P13 | Fourier Cymatics canonical and surrogate sections | Legacy quotient action; adopted parabolic gradient rail | Explicit surrogate, numerical band/entropy narrative; no automatic real-time action reduction |
| P14 | Revised Fixed-Point §§2–12 | Discrete selector/Jacobian, spectral radius, continuum tests | Conditional discrete stability; self-bound continuum existence not claimed |
| P15 | Compact Invariant Dynamics Appendix A/B | Annular-support compactness claim; radial core closure | Counterexamples expose insufficient compactness hypotheses; supplied core boundaries remain |
| P16 | Earlier Logarithmic Curvature Flow | Logarithmic identities and filament mass proposal | Read alongside corrected P04; no transfer of older incorrect coefficient conclusions |
| P17 | Soliton Appendices B/C | Fgeo=eta−lambda log(1+eta); kappaEM=kappa0 eta/(1+eta) | Explicit surrogate shape selection with free parameters and separate normalization |
| P18 | Constants paper full extracted text; calibration sections | Proposed constants, simulation scales and geometric normalizations | No independently fixed S004 coefficient map found; readable version identified |
| P19 | Black-hole bridge §5 Eqs14–16 | Auxiliary higher-time field with indefinite kinetic matrix | Existing theory-health gate supports explicit low-energy interpretation |
| P20 | Coulomb phase §§5–7 Eqs35,44–46 | theta=C/r, inverse-square field, attempted C=n/2 | Phase-tail scaling conditional; circulation/flux identification fails |
| P21 | Schrodinger completion §§1–3 | k²/2+beta k⁴ Hamiltonian operator | Legacy principal-term construction; positive k² gives no finite-k minimum by itself |

## Route comparison

| Route | Constructive result |
|---|---|
| A — PFF bridge | Conditional coefficient match; damping/pumping and nonlinear/domain closure not inherited |
| B — EFT bridge | Controlled single-amplitude finite-k reduction completed; global localized branch incomplete |
| C — Radial spectral bridge | Exact corrected fourth-order radial Hessian derived; factorization/potential/background spectrum conditional |
| D — Entropy-curvature and variational closure | Existing finite shape-orbit scale recovered; explicit approximate root derived; binding and matching incomplete |
| E — Geometric dilation bridge | Exact geometric singular value retained; physical intertwiner absent |

## Bridge records

Each record gives all thirteen requested research fields. “New” means a deduction made in this audit from the identified ingredients, never retroactively attributed to the source. A conditional theorem can be exact while its physical premises remain unresolved. `NOT_DERIVED` is scoped to the inspected equations and specified reduction, not a universal impossibility claim.

### B01 — S004 → general_Hessian

**Status:** `DERIVED_BY_CROSS_PAPER_COMPOSITION`. **Gap:** `CROSS_PAPER_REDUCTION_REQUIRED`. **Prior/new:** `NEW_DEDUCTION_FROM_EXISTING_INGREDIENTS`.

- **Source equations/papers:** S004 §§1-4; Q1.
- **Assumptions:** On-shell background, smooth weight and perturbation jets, fixed flat foliation.
- **Units and fields:** [psi]=A; [L]=A^2/L^2. eta; u1=2Re(psi0bar eta); u2=|eta|^2.
- **Transformation:** Taylor-expand F H and V including all derivative terms.
- **Control/error:** Perturbation amplitude; Quadratic coefficient exact; O(amplitude^3) remainder.
- **Free parameters:** potential, background, epsilon, alpha, kappa, theta, gamma, foliation.
- **Interpretation:** Complete amplitude/phase quadratic action.
- **Falsifier:** Direct jet differentiation disagrees with Q1.
- **Detailed derivation:** `QUADRATIC_ACTION_AND_DISPERSION.md`.

### B02 — general_Hessian → plane_wave_matrix

**Status:** `DERIVED_BY_CROSS_PAPER_COMPOSITION`. **Gap:** `CROSS_PAPER_REDUCTION_REQUIRED`. **Prior/new:** `NEW_DEDUCTION_FROM_EXISTING_INGREDIENTS`.

- **Source equations/papers:** S004 §4; Q3-Q5.
- **Assumptions:** Homogeneous plane carrier; background condition Q3.
- **Units and fields:** [omega,k,Omega,Kbg]=L^-1. xi=(r,v), v=A0 phase; matrices Q,P,J.
- **Transformation:** Co-rotate, Fourier transform, retain off-diagonal terms and determinant.
- **Control/error:** None; Exact within declared assumptions; no asymptotic error.
- **Free parameters:** carrier, potential, background, kappa, theta, gamma.
- **Interpretation:** All homogeneous fluctuation branches.
- **Falsifier:** Four matrix entries or det fail independent real-jet expansion.
- **Detailed derivation:** `QUADRATIC_ACTION_AND_DISPERSION.md`.

### B03 — plane_wave_matrix → all_static_poles

**Status:** `DERIVED_BY_CROSS_PAPER_COMPOSITION`. **Gap:** `CROSS_PAPER_REDUCTION_REQUIRED`. **Prior/new:** `NEW_DEDUCTION_FROM_EXISTING_INGREDIENTS`.

- **Source equations/papers:** S004; Q6-Q7.
- **Assumptions:** Static carrier; Vprime=0; K>0.
- **Units and fields:** [z,q,m^2]=L^-2; [K,G,T]=L^2. D=y-m^2-Ky^2+Gqy-Tq^2.
- **Transformation:** Solve quadratic in y and evaluate pole residues.
- **Control/error:** None; Exact within declared assumptions; no asymptotic error.
- **Free parameters:** background, potential, kappa, theta, gamma.
- **Interpretation:** Positive-residue lower and negative-residue upper modes in both sectors.
- **Falsifier:** Pole substitution or residues disagree.
- **Detailed derivation:** `QUADRATIC_ACTION_AND_DISPERSION.md`.

### B04 — all_static_poles → Geometry_EFT_amplitude

**Status:** `CONTROLLED_EFFECTIVE_REDUCTION`. **Gap:** `CROSS_PAPER_REDUCTION_REQUIRED`. **Prior/new:** `NEW_DEDUCTION_FROM_EXISTING_INGREDIENTS`.

- **Source equations/papers:** S004; Q7; P01 Appendix Y.1 Eq504; E1-E4.
- **Assumptions:** Positive discriminant margin; spectral support below ghost; C2<0.
- **Units and fields:** [C0]=L^-2,[C2]=1,[C4]=L^2; [cEFT]=L^2/T. xi=sqrt(2)r=Z^-1/2 chi; Z=K(zplus-z).
- **Transformation:** Analytic branch normalization and spatial Taylor expansion; E3 coefficient map.
- **Control/error:** delta<<1; q=delta M^2 Q; O(q^3) explicit E4 bound; O(delta^2 M^2) on bounded Q.
- **Free parameters:** background, potential, epsilon, alpha, kappa, theta, gamma, cutoff, time_scale.
- **Interpretation:** One real EFT amplitude component; no full-complex degeneracy claim.
- **Falsifier:** Direct action normalization or coefficients fail; window reaches collision.
- **Detailed derivation:** `CORRECTED_ACTION_TO_EFT_KERNEL.md`.

### B05 — all_static_poles → finite_wavenumber

**Status:** `PARAMETER_DEPENDENT`. **Gap:** `CROSS_PAPER_REDUCTION_REQUIRED`. **Prior/new:** `NEW_DEDUCTION_FROM_EXISTING_INGREDIENTS`.

- **Source equations/papers:** Q7; E1; F1-F2.
- **Assumptions:** 0<4Km^2<1; G C0>s; G^2<4KT.
- **Units and fields:** [q_min]=L^-2. q=|k|^2; amplitude lower pole.
- **Transformation:** Strict convexity and exact positive stationary root.
- **Control/error:** None; Exact within declared assumptions; no asymptotic error.
- **Free parameters:** background, potential, epsilon, alpha, kappa, theta, gamma.
- **Interpretation:** Genuine parameter-dependent minimum, not a fitted scale.
- **Falsifier:** F1 fails or proposed stationary root violates the original unsquared equation.
- **Detailed derivation:** `FINITE_SCALE_SELECTION.md`.

### B06 — S004 → ghost_free_fundamental_theory

**Status:** `OBSTRUCTED`. **Gap:** `NEW_PHYSICS_REQUIRED`. **Prior/new:** `EXISTING_HEALTH_ISSUE_WITH_NEW_S004_POLE_CALCULATION`.

- **Source equations/papers:** Q6-Q7; P19 §5 Eqs14-16.
- **Assumptions:** Literal nondegenerate higher-time action at nonzero background.
- **Units and fields:** Pole residue sign dimensionless. Both low and high poles.
- **Transformation:** Retain upper negative-residue pole; nondegenerate time Hessian.
- **Control/error:** None; Exact within declared assumptions; no asymptotic error.
- **Free parameters:** kappa, background, cutoff.
- **Interpretation:** Full fundamental health not established; a low-energy interpretation is separate.
- **Falsifier:** A constraint/degeneracy eliminating the extra mode is actually supplied.
- **Detailed derivation:** `QUADRATIC_ACTION_AND_DISPERSION.md`.

### B07 — Geometry_EFT_amplitude → damped_EFT

**Status:** `CONDITIONAL`. **Gap:** `NEW_PHYSICS_REQUIRED`. **Prior/new:** `NEW_DEDUCTION_FROM_EXISTING_INGREDIENTS`.

- **Source equations/papers:** P1; P02 §4.
- **Assumptions:** Physical damping/preparation specified independently in canonical sector.
- **Units and fields:** [eta]=T^-1. chi_tt+eta chi_t+c0^2 Ksp chi=0.
- **Transformation:** Adopt the physical damping premise already phenomenologically used by gradient rails.
- **Control/error:** None; Exact within declared assumptions; no asymptotic error.
- **Free parameters:** damping, observable, cutoff.
- **Interpretation:** Defines the additional premise needed for real-time relaxation.
- **Falsifier:** No bath/coarse-graining supports the prescribed friction.
- **Detailed derivation:** `CORRECTED_ACTION_TO_PFF_REDUCTION.md`.

### B08 — damped_EFT → PFF_coefficients

**Status:** `CONTROLLED_EFFECTIVE_REDUCTION`. **Gap:** `CROSS_PAPER_REDUCTION_REQUIRED`. **Prior/new:** `NEW_DEDUCTION_FROM_EXISTING_INGREDIENTS`.

- **Source equations/papers:** P1-P2; P02 Eqs8-10.
- **Assumptions:** Overdamped slow preparation; t>>eta^-1; low-energy cutoff.
- **Units and fields:** [r]=T^-1,[a]=L^2/T,[b]=L^4/T. Gamma=c0^2/eta; r=-Gamma C0,a=-Gamma C2,b=Gamma C4.
- **Transformation:** Expand both damped exponents and select the prepared slow sector.
- **Control/error:** c0^2 |Ksp|/eta^2<<1; O(c0^4 Ksp^2/eta^3) in exponent plus EFT error.
- **Free parameters:** damping, background, potential, kappa, theta, gamma.
- **Interpretation:** Same finite-band center; stable witness remains decaying.
- **Falsifier:** Eigenvalue expansion fails or sigma_max is wrongly asserted positive.
- **Detailed derivation:** `CORRECTED_ACTION_TO_PFF_REDUCTION.md`.

### B09 — PFF_coefficients → PFF_band_and_saturation

**Status:** `DERIVED_IN_EXISTING_PAPER`. **Gap:** `ALREADY_IN_CORPUS`. **Prior/new:** `EXISTING_PAPER`.

- **Source equations/papers:** P02 §§4-5 Eqs8-10.
- **Assumptions:** Declared gradient rail; b,beta>0; onset requires mu>0.
- **Units and fields:** k_star L^-1; sigma T^-1. A complex PFF envelope.
- **Transformation:** Complete square sigma=mu-b(q-qstar)^2; variational descent.
- **Control/error:** None; Exact within declared assumptions; no asymptotic error.
- **Free parameters:** pff_nonlinearity, damping.
- **Interpretation:** Analytic finite-band onset and saturation within PFF.
- **Falsifier:** Chosen r yields no positive growth or derivative is not gradient flow.
- **Detailed derivation:** `CORRECTED_ACTION_TO_PFF_REDUCTION.md`.

### B10 — PFF_band_and_saturation → SH_CGL_envelope

**Status:** `PERTURBATIVE`. **Gap:** `ALREADY_IN_CORPUS`. **Prior/new:** `EXISTING_ROUTE_WITH_NEW_SCOPED_COEFFICIENT_CHECK`.

- **Source equations/papers:** P02 §7 and Appendix R; P12 §§1.4,2.1-2.5; P13 surrogate section.
- **Assumptions:** One-dimensional single complex carrier and weak detuning; real field changes cubic factor.
- **Units and fields:** X=eps^1/2 x, T=eps t. A=eps^1/2 B exp(i kstar x).
- **Transformation:** Multiscale expansion gives 4b kstar^2 B_XX and cubic saturation.
- **Control/error:** eps<<1; Leading normal form; relative O(sqrt(eps)) formal correction, no PDE theorem.
- **Free parameters:** pff_nonlinearity.
- **Interpretation:** Recovers existing envelope route with scaling scope explicit.
- **Falsifier:** Transverse critical modes or omitted harmonics contribute at same order.
- **Detailed derivation:** `CORRECTED_ACTION_TO_PFF_REDUCTION.md`.

### B11 — finite_wavenumber → linear_homogeneous_localized_eigenstate

**Status:** `OBSTRUCTED`. **Gap:** `CROSS_PAPER_REDUCTION_REQUIRED`. **Prior/new:** `NEW_DEDUCTION_FROM_EXISTING_INGREDIENTS`.

- **Source equations/papers:** Q7; E2; L localization Fourier-support argument.
- **Assumptions:** Whole-space constant coefficients; isolated spectral level; L2 state.
- **Units and fields:** Spatial eigenvalue L^-2. Fourier support on polynomial level set.
- **Transformation:** A measure-zero level surface supports no nonzero L2 Fourier function.
- **Control/error:** None; Exact within declared assumptions; no asymptotic error.
- **Free parameters:** None in this identity.
- **Interpretation:** Linear carrier selection alone does not localize.
- **Falsifier:** An inhomogeneity, nonlinearity, or different spectral class is introduced.
- **Detailed derivation:** `LOCALIZATION_AND_DOMAIN_SELECTION.md`.

### B12 — closure_functional → fixed_shape_radius

**Status:** `DERIVED_IN_EXISTING_PAPER`. **Gap:** `ALREADY_IN_CORPUS`. **Prior/new:** `EXISTING_PAPER`.

- **Source equations/papers:** P09 Eq8 and Eqs15-22; Corollary6.4.
- **Assumptions:** All closure coefficients positive; fixed nonzero H2 shape and L2 mass; R3.
- **Units and fields:** R dimensionless dilation; energy E. psi_R=R^-3/2 psi(x/R).
- **Transformation:** Exact dilation formula; collapse to +infinity and spreading to 0 from below.
- **Control/error:** None; Exact within declared assumptions; no asymptotic error.
- **Free parameters:** closure_coefficients, mass_constraint, shape.
- **Interpretation:** Existing rigorous finite orbit scale.
- **Falsifier:** Either limiting sign or fixed-shape continuity fails.
- **Detailed derivation:** `LOCALIZATION_AND_DOMAIN_SELECTION.md`.

### B13 — fixed_shape_radius → explicit_orbit_radius

**Status:** `PERTURBATIVE`. **Gap:** `CROSS_PAPER_REDUCTION_REQUIRED`. **Prior/new:** `NEW_DEDUCTION_FROM_EXISTING_INGREDIENTS`.

- **Source equations/papers:** P09 Eq19; L2-L3.
- **Assumptions:** Bounded fixed shape; small density in quotient at derived root.
- **Units and fields:** R dimensionless; A,B,C energies. A=(a4+kappa_r/delta_r^2)H,B=a2G,C=beta Q/2.
- **Transformation:** Minimize A/R^4-B/R^2+C/R^3 exactly at retained order.
- **Control/error:** ||psi||inf^2/(delta_r^2 R^3)<<1; Energy remainder <=kappa_r delta_r^-4 ||psi||inf^2 H R^-7; nondegenerate derivative correction.
- **Free parameters:** closure_coefficients, mass_constraint, shape.
- **Interpretation:** New explicit radius and positive orbit Hessian.
- **Falsifier:** Density parameter not small or second derivative is nonpositive.
- **Detailed derivation:** `LOCALIZATION_AND_DOMAIN_SELECTION.md`.

### B14 — S004 → closure_functional

**Status:** `NOT_DERIVED`. **Gap:** `CROSS_PAPER_REDUCTION_REQUIRED`. **Prior/new:** `NEW_DEDUCTION_FROM_EXISTING_INGREDIENTS`.

- **Source equations/papers:** S004 §2; P09 Eq8; E5.
- **Assumptions:** A common nonlinear reduction and inherited constraint would be required.
- **Units and fields:** Weights F and (u+delta_r^2)^-1 both A^-2 but different limits. S004 low amplitude sector versus full complex closure field.
- **Transformation:** Quadratic matching is available; nonlinear vertices, charge constraint and functional are not matched.
- **Control/error:** None; Exact within declared assumptions; no asymptotic error.
- **Free parameters:** potential, closure_coefficients, mass_constraint, background.
- **Interpretation:** Existing confinement functional is a candidate nonlinear completion, not literal static S004.
- **Falsifier:** A controlled nonlinear matching derives all terms and the constraint.
- **Detailed derivation:** `LOCALIZATION_AND_DOMAIN_SELECTION.md`.

### B15 — closure_functional → free_space_bound_minimizer

**Status:** `NOT_DERIVED`. **Gap:** `CROSS_PAPER_REDUCTION_REQUIRED`. **Prior/new:** `NEW_DEDUCTION_FROM_EXISTING_INGREDIENTS`.

- **Source equations/papers:** P09 §11 Eqs35-37; L4; P14 §§11-12.
- **Assumptions:** Need exclusion of vanishing and dichotomy for general minimizing sequences.
- **Units and fields:** Energy threshold E. e(M_N)=inf F at fixed mass.
- **Transformation:** Compare against delocalized-carrier threshold and establish compactness/strict binding.
- **Control/error:** None; Exact within declared assumptions; no asymptotic error.
- **Free parameters:** closure_coefficients, mass_constraint.
- **Interpretation:** Mathematical binding problem remains even within selected closure model.
- **Falsifier:** A minimizing sequence splits or vanishes, or a valid binding/compactness proof closes it.
- **Detailed derivation:** `LOCALIZATION_AND_DOMAIN_SELECTION.md`.

### B16 — entropy_penalty → entropy_orbit_radius

**Status:** `CONDITIONAL`. **Gap:** `CROSS_PAPER_REDUCTION_REQUIRED`. **Prior/new:** `NEW_DEDUCTION_FROM_EXISTING_INGREDIENTS`.

- **Source equations/papers:** P01 §28 pp87-88; §40.3 p129; L entropy calculation.
- **Assumptions:** Choose positive entropy penalty at fixed mass; controlled fixed shape.
- **Units and fields:** S dimensionless; tau E; R dimensionless. p=|psi|^2/N; S0=-int p log(p/pref).
- **Transformation:** Dilation gives S0+d log R; minimize positive inverse powers plus tau d log R.
- **Control/error:** sup(u/epsilon^2+|alpha|u)<<1 for S004 curvature scaling; Exact dilation entropy; curvature power approximation controlled by small amplitude.
- **Free parameters:** entropy, shape, mass_constraint, potential.
- **Interpretation:** Existing entropy ingredient can select a conditional radius.
- **Falsifier:** Physical energy uses opposite sign, or approximation fails.
- **Detailed derivation:** `LOCALIZATION_AND_DOMAIN_SELECTION.md`.

### B17 — S004 → entropy_penalty

**Status:** `CONDITIONAL`. **Gap:** `CROSS_PAPER_REDUCTION_REQUIRED`. **Prior/new:** `NEW_DEDUCTION_FROM_EXISTING_INGREDIENTS`.

- **Source equations/papers:** S004 unspecified V; P01 §§28,40.3.
- **Assumptions:** Declare energy sign, normalization and fixed-mass or nonlocal entropy functional.
- **Units and fields:** [tau]=E; normalized density uses reference scale. V logarithmic choice or separate normalized entropy.
- **Transformation:** Specify an existing candidate physical term; it is not forced by general V.
- **Control/error:** None; Exact within declared assumptions; no asymptotic error.
- **Free parameters:** potential, entropy, mass_constraint.
- **Interpretation:** No claim that entropy mechanism is absent; its adoption remains a physical choice.
- **Falsifier:** Different legitimate V or entropy sign changes the balance.
- **Detailed derivation:** `LOCALIZATION_AND_DOMAIN_SELECTION.md`.

### B18 — S004 → corrected_radial_Hessian

**Status:** `DERIVED_BY_CROSS_PAPER_COMPOSITION`. **Gap:** `CROSS_PAPER_REDUCTION_REQUIRED`. **Prior/new:** `NEW_DEDUCTION_FROM_EXISTING_INGREDIENTS`.

- **Source equations/papers:** S004 §§2-4; P10 §§3-7 Hessian-general; R1-R4.
- **Assumptions:** Static real on-shell radial background; boundary terms vanish only under declared BC.
- **Units and fields:** [a4]=L^2,[R]=1,[W]=L^-2. a(r),u=a^2,b=Delta a; perturbations r,v.
- **Transformation:** Expand actual S004 weight and integrate B1 r Delta r including Delta B1/2.
- **Control/error:** None; Exact within declared assumptions; no asymptotic error.
- **Free parameters:** radial_background, potential, kappa, theta, gamma.
- **Interpretation:** Actual fourth-order radial operator; background still to be solved.
- **Falsifier:** Direct variation misses/mis-signs coefficient or potential derivative.
- **Detailed derivation:** `RADIAL_OPERATOR_AND_SHELL_SPECTRUM.md`.

### B19 — corrected_radial_Hessian → PFF_second_order_operator

**Status:** `CONDITIONAL`. **Gap:** `CROSS_PAPER_REDUCTION_REQUIRED`. **Prior/new:** `NEW_DEDUCTION_FROM_EXISTING_INGREDIENTS`.

- **Source equations/papers:** P02 §§2.3,6 Eq13; R5.
- **Assumptions:** Constant a4 and exact R,W matching identities, or another controlled factorization.
- **Units and fields:** [VTheta]=L^-2. OTheta=-Delta+VTheta.
- **Transformation:** Expand square: R=2a4 VTheta,W=a4(VTheta^2-Delta VTheta)+gap.
- **Control/error:** None; Exact within declared assumptions; no asymptotic error.
- **Free parameters:** radial_background.
- **Interpretation:** A checkable factorization criterion, not a guessed curvature potential.
- **Falsifier:** Either coefficient identity fails.
- **Detailed derivation:** `RADIAL_OPERATOR_AND_SHELL_SPECTRUM.md`.

### B20 — corrected_radial_Hessian → discrete_shell_spectrum

**Status:** `NOT_DERIVED`. **Gap:** `CROSS_PAPER_REDUCTION_REQUIRED`. **Prior/new:** `NEW_DEDUCTION_FROM_EXISTING_INGREDIENTS`.

- **Source equations/papers:** R4; P09 §11; P02 §6.
- **Assumptions:** Self-consistent bound background and self-adjoint physical domain needed.
- **Units and fields:** [Lambda]=L^-2 for second-order; Hessian units declared. Radial eigenspaces.
- **Transformation:** Free space is continuous; a derived well may have bound states; a supplied disk imports R.
- **Control/error:** None; Exact within declared assumptions; no asymptotic error.
- **Free parameters:** radial_background, shape.
- **Interpretation:** No intrinsic cavity/shell spectrum yet selected.
- **Falsifier:** Claimed level moves with arbitrary external boundary.
- **Detailed derivation:** `RADIAL_OPERATOR_AND_SHELL_SPECTRUM.md`.

### B21 — PFF_second_order_operator → conditional_radial_quantization

**Status:** `CONDITIONAL`. **Gap:** `CROSS_PAPER_REDUCTION_REQUIRED`. **Prior/new:** `NEW_DEDUCTION_FROM_EXISTING_INGREDIENTS`.

- **Source equations/papers:** P02 Eq13; R6-R7.
- **Assumptions:** Two simple turning points and WKB matching; declared self-adjoint domain.
- **Units and fields:** integral k_r dr dimensionless. f=w/sqrt(r); effective centrifugal term (m_ang^2-1/4)/r^2.
- **Transformation:** Liouville transform and radial oscillation action.
- **Control/error:** |k_r prime|/k_r^2 away from turning points; Liouville exact; WKB asymptotic with endpoint-dependent phase correction.
- **Free parameters:** radial_background.
- **Interpretation:** Correct conditional shell relation; not raw Lambda on RHS of phase winding.
- **Falsifier:** Turning-point assumptions fail or units/action integer are conflated.
- **Detailed derivation:** `RADIAL_OPERATOR_AND_SHELL_SPECTRUM.md`.

### B22 — discrete_shell_spectrum → physical_domain

**Status:** `NOT_DERIVED`. **Gap:** `CROSS_PAPER_REDUCTION_REQUIRED`. **Prior/new:** `NEW_DEDUCTION_FROM_EXISTING_INGREDIENTS`.

- **Source equations/papers:** R4-R7; D2.
- **Assumptions:** Need selected state and operational edge definition.
- **Units and fields:** Endpoints carry same physical units. x_minus,x_plus from X[psi].
- **Transformation:** A spectrum labels states but does not alone fix chosen domain edges.
- **Control/error:** None; Exact within declared assumptions; no asymptotic error.
- **Free parameters:** radial_background, observable, sector_preparation.
- **Interpretation:** Domain selection remains distinct from allowed eigenvalues.
- **Falsifier:** Domain is fixed by arbitrary threshold or numerical box.
- **Detailed derivation:** `ACTIVE_LOG_DOMAIN_MAP.md`.

### B23 — S004 → static_sector_energy

**Status:** `DERIVED_BY_CROSS_PAPER_COMPOSITION`. **Gap:** `CROSS_PAPER_REDUCTION_REQUIRED`. **Prior/new:** `NEW_DEDUCTION_FROM_EXISTING_INGREDIENTS`.

- **Source equations/papers:** S004 §2; W1.
- **Assumptions:** Static cylindrical ansatz; regular core; finite or renormalized energy domain.
- **Units and fields:** Energy per longitudinal length. psi=f_n(r) exp(i n phi).
- **Transformation:** Insert exact polar Laplacian into all S004 static terms.
- **Control/error:** None; Exact within declared assumptions; no asymptotic error.
- **Free parameters:** potential, radial_background, sector_preparation, kappa, theta, gamma.
- **Interpretation:** Full known static sector functional, not solved profiles or general time Hamiltonian.
- **Falsifier:** An actual S004 term is omitted or extra-model energies silently added.
- **Detailed derivation:** `SECTOR_ENERGY_AND_WINDING.md`.

### B24 — given_loop_geometry → locking_energy_minimum

**Status:** `DERIVED_IN_EXISTING_PAPER`. **Gap:** `ALREADY_IN_CORPUS`. **Prior/new:** `EXISTING_PAPER`.

- **Source equations/papers:** P22 §3 Eq3.1; P08; W2.
- **Assumptions:** Positive weight, fixed loop and fixed winding.
- **Units and fields:** phase mismatch dimensionless; Iw units reciprocal weight times L. sigma=sqrt(curvature^2+torsion^2), phi prime.
- **Transformation:** Weighted constrained minimization yields squared mismatch/Iw.
- **Control/error:** None; Exact within declared assumptions; no asymptotic error.
- **Free parameters:** loop_geometry, sector_preparation.
- **Interpretation:** Nearest integer minimizes one fixed-geometry component.
- **Falsifier:** Geometry/weight depend on n or other energy terms change ordering.
- **Detailed derivation:** `SECTOR_ENERGY_AND_WINDING.md`.

### B25 — static_sector_energy → selected_winding

**Status:** `NOT_DERIVED`. **Gap:** `CROSS_PAPER_REDUCTION_REQUIRED`. **Prior/new:** `NEW_DEDUCTION_FROM_EXISTING_INGREDIENTS`.

- **Source equations/papers:** W1-W2; P14 §11.
- **Assumptions:** Physical branch, complete constraint and preparation needed.
- **Units and fields:** n integer. n_star=argmin inf E_n where admissible.
- **Transformation:** No solved full sector energies; fixed-loop nearest integer is insufficient.
- **Control/error:** None; Exact within declared assumptions; no asymptotic error.
- **Free parameters:** sector_preparation, loop_geometry, potential, radial_background.
- **Interpretation:** Allowed topology differs from actual selected sector.
- **Falsifier:** A competing sector/zero-charge vacuum or barrier invalidates proposed selection.
- **Detailed derivation:** `SECTOR_ENERGY_AND_WINDING.md`.

### B26 — PFF_field → phase_slip_permission

**Status:** `DERIVED_IN_EXISTING_PAPER`. **Gap:** `ALREADY_IN_CORPUS`. **Prior/new:** `EXISTING_PAPER`.

- **Source equations/papers:** P02 §§3,6; P14 §8.
- **Assumptions:** Complex field nonzero on loop except during slip.
- **Units and fields:** n dimensionless integer. Local amplitude zero crossing.
- **Transformation:** Continuous nonzero evolution preserves winding; zeros allow change.
- **Control/error:** None; Exact within declared assumptions; no asymptotic error.
- **Free parameters:** sector_preparation.
- **Interpretation:** Mechanism exists; no rates/barriers selected.
- **Falsifier:** A claimed winding change occurs without a zero or changed domain.
- **Detailed derivation:** `SECTOR_ENERGY_AND_WINDING.md`.

### B27 — physical_domain → active_log_window

**Status:** `NOT_DERIVED`. **Gap:** `CROSS_PAPER_REDUCTION_REQUIRED`. **Prior/new:** `NEW_DEDUCTION_FROM_EXISTING_INGREDIENTS`.

- **Source equations/papers:** P03 premises; D1-D2.
- **Assumptions:** Independent positive measurable X and endpoints.
- **Units and fields:** Delta ell dimensionless. Delta ell=log(xplus/xminus).
- **Transformation:** Test candidate mass, curvature, frequency and radius maps.
- **Control/error:** None; Exact within declared assumptions; no asymptotic error.
- **Free parameters:** observable, radial_background, loop_geometry.
- **Interpretation:** No physical endpoints inferred from a cutoff or carrier minimum.
- **Falsifier:** Endpoint definition changes with threshold/box or depends on fitted frequency.
- **Detailed derivation:** `ACTIVE_LOG_DOMAIN_MAP.md`.

### B28 — confined_log_eigenmode → scale_winding_DSI

**Status:** `DERIVED_IN_EXISTING_PAPER`. **Gap:** `ALREADY_IN_CORPUS`. **Prior/new:** `EXISTING_PAPER`.

- **Source equations/papers:** P03 scale-winding theorem; D1.
- **Assumptions:** Independently generated stable log mode, fixed domain, nonzero n and orientation.
- **Units and fields:** nu,Delta ell dimensionless. nu Delta ell=2pi n; lambda=exp(Delta ell/n).
- **Transformation:** Apply existing recurrence theorem once premises hold.
- **Control/error:** None; Exact within declared assumptions; no asymptotic error.
- **Free parameters:** sector_preparation, observable.
- **Interpretation:** Conditional downstream theorem already established.
- **Falsifier:** No log eigenmode or measurement-domain matching exists.
- **Detailed derivation:** `ACTIVE_LOG_DOMAIN_MAP.md`.

### B29 — spatial_phase → log_phase_frequency

**Status:** `CONDITIONAL`. **Gap:** `CROSS_PAPER_REDUCTION_REQUIRED`. **Prior/new:** `NEW_DEDUCTION_FROM_EXISTING_INGREDIENTS`.

- **Source equations/papers:** D2; P03.
- **Assumptions:** Positive differentiable monotone X(s) on branch.
- **Units and fields:** spatial k L^-1; nu dimensionless. nu=(d phase/ds)/(d log X/ds).
- **Transformation:** Chain rule; constant carrier becomes exponential phase for log radius.
- **Control/error:** None; Exact within declared assumptions; no asymptotic error.
- **Free parameters:** observable, radial_background.
- **Interpretation:** Explicit necessary coordinate bridge.
- **Falsifier:** Nu varies over interval or X is not single-valued.
- **Detailed derivation:** `ACTIVE_LOG_DOMAIN_MAP.md`.

### B30 — prepared_amplitude_branch → linear_response

**Status:** `CONDITIONAL`. **Gap:** `CROSS_PAPER_REDUCTION_REQUIRED`. **Prior/new:** `NEW_DEDUCTION_FROM_EXISTING_INGREDIENTS`.

- **Source equations/papers:** Q6-Q7; O1; P02 intensity definition.
- **Assumptions:** Weak calibrated source, declared retarded state/cutoff and smooth detector response.
- **Units and fields:** D L^-2; chi_uJ has units from J=D r. delta u=2A0 r, D r=J.
- **Transformation:** Invert derived kernel; retain residues in physical field.
- **Control/error:** Source amplitude; Linear response O(r), omitted intensity O(r^2).
- **Free parameters:** observable, background, cutoff, damping.
- **Interpretation:** Concrete conditional dispersion-response prediction.
- **Falsifier:** Detector null, different harmonic coupling or invalid preparation.
- **Detailed derivation:** `OBSERVABLE_FORWARD_MAP.md`.

### B31 — scale_winding_DSI → estimator_frequency

**Status:** `CONDITIONAL`. **Gap:** `CROSS_PAPER_REDUCTION_REQUIRED`. **Prior/new:** `NEW_DEDUCTION_FROM_EXISTING_INGREDIENTS`.

- **Source equations/papers:** P03; O2.
- **Assumptions:** Nonzero fundamental, declared response and coordinate map, fixed nuisance/window/estimator.
- **Units and fields:** nu dimensionless. y=B+int R O[psi].
- **Transformation:** Compute harmonic content and detector transfer before frequency fitting.
- **Control/error:** None; Exact within declared assumptions; no asymptotic error.
- **Free parameters:** observable.
- **Interpretation:** Internal recurrence need not equal fitted cosine frequency.
- **Falsifier:** Response removes fundamental, nonlinear O doubles it, or coordinate warp changes it.
- **Detailed derivation:** `OBSERVABLE_FORWARD_MAP.md`.

### B32 — geometric_T → singular_value_sqrt6

**Status:** `DERIVED_IN_EXISTING_PAPER`. **Gap:** `ALREADY_IN_CORPUS`. **Prior/new:** `EXISTING_PAPER`.

- **Source equations/papers:** P05 geometric map; P06; Prior geometric audit.
- **Assumptions:** Exact Euclidean maps and stated inner products.
- **Units and fields:** Dimensionless. T:R->R3; Tdagger T=6.
- **Transformation:** Compute exact Gram map.
- **Control/error:** None; Exact within declared assumptions; no asymptotic error.
- **Free parameters:** None in this identity.
- **Interpretation:** Preserved mathematical 2-3-6 fact.
- **Falsifier:** Gram calculation fails under stated definition.
- **Detailed derivation:** `OBSERVABLE_FORWARD_MAP.md`.

### B33 — geometric_T → physical_dilation

**Status:** `NOT_DERIVED`. **Gap:** `CROSS_PAPER_REDUCTION_REQUIRED`. **Prior/new:** `NEW_DEDUCTION_FROM_EXISTING_INGREDIENTS`.

- **Source equations/papers:** P05/P06; O2 operator discussion.
- **Assumptions:** Need physical invariant state space and typed embeddings.
- **Units and fields:** Dimensionless dilation factor; operator units must match. D I_in=I_out T.
- **Transformation:** Test translation/radial operators; none supplies required embeddings or dilation.
- **Control/error:** None; Exact within declared assumptions; no asymptotic error.
- **Free parameters:** embeddings, radial_background.
- **Interpretation:** No physical 6 or sqrt6 scale selected.
- **Falsifier:** An independently derived intertwiner and state action establishes it.
- **Detailed derivation:** `OBSERVABLE_FORWARD_MAP.md`.

### B34 — single_embedding_equation → typed_intertwiner

**Status:** `OBSTRUCTED`. **Gap:** `CROSS_PAPER_REDUCTION_REQUIRED`. **Prior/new:** `NEW_DEDUCTION_FROM_EXISTING_INGREDIENTS`.

- **Source equations/papers:** User candidate D I=I T; P05 T:R->R3.
- **Assumptions:** One I cannot have two incompatible domains.
- **Units and fields:** Operator domain typing. Separate I_in and I_out needed.
- **Transformation:** Check domains and codomains before spectral interpretation.
- **Control/error:** None; Exact within declared assumptions; no asymptotic error.
- **Free parameters:** embeddings.
- **Interpretation:** Corrects the proposed equation without ruling out a properly typed completion.
- **Falsifier:** A common endomorphism/return map is explicitly derived.
- **Detailed derivation:** `OBSERVABLE_FORWARD_MAP.md`.

### B35 — S004 → nonlinear_vertices

**Status:** `DERIVED_BY_CROSS_PAPER_COMPOSITION`. **Gap:** `CROSS_PAPER_REDUCTION_REQUIRED`. **Prior/new:** `NEW_DEDUCTION_FROM_EXISTING_INGREDIENTS`.

- **Source equations/papers:** S004 §2; E5.
- **Assumptions:** Static homogeneous background; sufficient derivatives of V,F.
- **Units and fields:** Lagrangian A^2 L^-2. u1=2A0 r,u2=r^2+v^2,H2 quadratic jets.
- **Transformation:** Expand cubic and quartic potential/derivative vertices.
- **Control/error:** Perturbation amplitude; Exact cubic/quartic coefficients; O(amplitude^5).
- **Free parameters:** potential, background, epsilon, alpha, kappa, theta, gamma.
- **Interpretation:** Nonlinear saturation remains independent of quadratic pole data.
- **Falsifier:** Direct expansion fails or Vthird/Vfourth are silently fixed.
- **Detailed derivation:** `CORRECTED_ACTION_TO_EFT_KERNEL.md`.

### B36 — closure_functional → closure_weak_PFF_symbol

**Status:** `DERIVED_BY_CROSS_PAPER_COMPOSITION`. **Gap:** `CROSS_PAPER_REDUCTION_REQUIRED`. **Prior/new:** `NEW_DEDUCTION_FROM_EXISTING_INGREDIENTS`.

- **Source equations/papers:** P09 Eqs8,24,29; L weak-field correction.
- **Assumptions:** Weak amplitude compared with delta_r; declared projected real-gradient flow.
- **Units and fields:** q L^-2; coefficients in L1 units. qden=|psi|^2+delta_r^2; multiplier lambda[psi].
- **Transformation:** Retain quadratic kappa_r/delta_r^2 contribution; a=2a2,b=2(a4+kappa_r/delta_r^2).
- **Control/error:** |psi|^2/delta_r^2<<1; Leading quadratic flow; derivative nonlinearities at cubic order.
- **Free parameters:** closure_coefficients, mass_constraint.
- **Interpretation:** Refines first-two-term center; multiplier is constrained, not free pumping.
- **Falsifier:** Quotient contribution is omitted or lambda treated as unconstrained constant.
- **Detailed derivation:** `LOCALIZATION_AND_DOMAIN_SELECTION.md`.

### B37 — closure_functional → vanishing_threshold

**Status:** `DERIVED_BY_CROSS_PAPER_COMPOSITION`. **Gap:** `CROSS_PAPER_REDUCTION_REQUIRED`. **Prior/new:** `NEW_DEDUCTION_FROM_EXISTING_INGREDIENTS`.

- **Source equations/papers:** P09 Eq8; L4.
- **Assumptions:** Fixed mass, broad modulated Schwartz packets, R to infinity.
- **Units and fields:** Threshold energy E. psi_R=R^-3/2 f(x/R) exp(i k0.x).
- **Transformation:** Take packet limit and minimize full dilute quadratic symbol.
- **Control/error:** R^-1 -> 0; Exact limiting threshold for this sequence, not proof of global infimum.
- **Free parameters:** closure_coefficients, mass_constraint.
- **Interpretation:** Negative energy alone does not establish binding.
- **Falsifier:** Computed packet limit differs or a comparison ignores kappa_r term.
- **Detailed derivation:** `LOCALIZATION_AND_DOMAIN_SELECTION.md`.

### B38 — Coulomb_phase_tail → topological_charge_coefficient

**Status:** `OBSTRUCTED`. **Gap:** `CROSS_PAPER_REDUCTION_REQUIRED`. **Prior/new:** `NEW_DEDUCTION_FROM_EXISTING_INGREDIENTS`.

- **Source equations/papers:** P20 §§5-7 Eqs35,44-46.
- **Assumptions:** Spherical theta=C/r outside core.
- **Units and fields:** Circulation dimensionless; flux has length units for dimensionless theta. C arbitrary; theta single valued on punctured 3D exterior.
- **Transformation:** Closed-path circulation is zero while spherical flux is nonzero.
- **Control/error:** None; Exact within declared assumptions; no asymptotic error.
- **Free parameters:** loop_geometry.
- **Interpretation:** Flux and winding cannot be equated to derive C=n/2.
- **Falsifier:** An independent gauge/topological relation supplies the missing equality.
- **Detailed derivation:** `SECTOR_ENERGY_AND_WINDING.md`.

## Precedence and limits

The corrected action is S004. P09 is an independently declared fixed-mass energy; P02 is an independently declared dissipative rail. P10 supplies an exact Hessian method that can be applied to S004 without identifying its old denominator with F. The successful S004-to-EFT result is a controlled single-real-amplitude reduction. PFF growth, global binding, winding selection, log endpoints and measurement equality are separate dependencies. Their absence is never used to erase the finite-scale result.

No empirical frequency, collider search, ATLAS holdout, or expensive PDE simulation was used. The rational example in the verifier demonstrates nonempty parameter space only. The new records preserve the previous physical-domain and measurement labels with their original scope.
