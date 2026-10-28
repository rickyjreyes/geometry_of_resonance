# Canonical Geometry #18: manuscript-to-Lean claim map and closure audit

Date: 2026-10-28. Target: corrected real S004 action, not the historical schematic master equation or the independent spatial EFT. The maintained original sources are `WCT_CORRECTED_VARIATIONAL_CLOSURE_DRAFT.md` (S004), `docs/cross_paper/QUADRATIC_ACTION_AND_DISPERSION.md` (Q1–Q7), `docs/cross_paper/FINITE_SCALE_SELECTION.md` (F1–F2), `docs/cross_paper/CORRECTED_ACTION_TO_EFT_KERNEL.md` (E1–E5), and `docs/foundational/GHOST_AND_CONSTRAINT_ANALYSIS.md` (H1–H4). Those historical inputs are **not edited or silently replaced**. All signs refer to the physical ordinary-mode orientation S_phys=-S_orig.

## Parameters and background gates

In a fixed flat inertial foliation, `u=|psi|²`, `D=u+eps² exp(-2alpha u)`, `F=u/D²`, `Q=Box psi`, `P=Delta psi`; `H=kappa|Q|²+theta|P|²+gamma Re(Q conj(P))`.
For the static on-shell homogeneous nonzero background, require `u0>0,eps>0,V'(u0)=0`; use `K=F(u0)kappa>0, G=F(u0)gamma, T=F(u0)theta>0` and `B=4KT-G²>0`. No sign restriction on G follows from strict curvature positivity. Amplitude `massSq=-2u0V''(u0)` is additionally nonnegative **only in theorems which say so**. Phase has massSq=0. Other rotating and inhomogeneous backgrounds have different Hessians and are not silently covered by the static scalar block.

## Exact claim map

| Claim / equation | Source | Lean declaration / validation | Verification scope |
| --- | --- | --- | --- |
| Corrected action's field variation H1 | S004 §4, H1 | `derivations/foundational_closure.py`, prior source audit | Analytical + symbolic; full nonlinear field variation not Lean-formalized |
| Quadratic expansion at arbitrary on-shell background | Q1 | `audit/cross_paper/verify_derivations.py` | Exact symbolic; not Lean-functional analysis |
| Homogeneous carrier full complex 2x2 matrix | Q4 | `derivations/canonical_16_18_verify.py` `rotating_Q4_Q5_ij` | Exact symbolic; not a Lean complex-matrix Fourier transform |
| Rotating determinant | Q5 | `rotating_Q5_det` | Exact symbolic; arbitrary carrier Q4 kept separate |
| Static real-component inverse propagator | Q6 | `R18.inversePropagator`, `inverse_propagator_completion` | Lean exact real polynomial |
| Discriminant and curvature gap | Q7 | `R18.curvatureGap`, `poleDiscriminant`, `discriminant_expansion` | Lean exact |
| Both pole formulas are on shell | Q7 | `R18.lower_pole_on_shell`, `upper_pole_on_shell` | Lean if K≠0 and d≥0 |
| Opposite slopes/residues | Q7 | `R18` pole derivative theorems | Lean on distinct poles; sign is not full Hamiltonian positivity |
| No real z when d<0 | Q6–Q7 | `R18.no_real_frequency_root` | Lean; continuum Sobolev ill-posedness uses H4 |
| Exact full real-pole interval | Q7 | `R18Classification.real_pole_iff_vertex_interval`, `distinct_real_pole_iff_strict_vertex` | Lean for B>0, every G/massSq |
| Double pole boundary | Q7 | `collision_iff_vertex_boundary`, `vertex_zero_collision` | Lean, including N=0 degeneracy |
| Vertex N<0 has no real poles anywhere | Q7 | `everywhere_complex_if_vertex_negative` | Lean |
| Exact high-q complex criterion | H4, Q7 | `uv_complex_beyond_upper_collision`, `uv_complex_beyond_vertex` | Lean algebra; high-frequency PDE growth rate/H4 is analytic |
| Positive curvature implies positive static stiffness | Q6, H3 | `R18Canonical.spatial_coefficient_pos` | Lean, `K>0,B>0` |
| No q>0 zero-frequency static bifurcation | Q6 | `R18Canonical.no_static_finite_q_onset`, `phase_no_finite_q_onset` | Lean, `massSq>=0` |
| Positive squared-frequency poles in real sector (G>=0) | Q7 | `R18Canonical.real_poles_not_tachyonic` | Lean |
| Positive squared-frequency poles for **every G** satisfying strict B>0 | Q7 | `R18Admissibility.real_poles_non_tachyonic_all_G` | Lean under `K>0,T>0,B>0,massSq>=0,q>0,d>=0` |
| Exact finite-q minimizer formula | F2 | `R18ExactMinimum.centerRoot`, `candidateQ`, `candidateQ_discriminant` | Lean B>0,N>=0 |
| Candidate stationarity and q*>0 | F1–F2 | `candidateQ_stationary`, `candidateQ_positive` | Lean with explicit positive-q inequality |
| Strict minimum and uniqueness | F2 | `exact_candidate_is_minimum_in_window`, `exact_candidate_unique_in_window` | Lean on real-pole positive-tangent window |
| Global on q>=0 intersect real-pole window | F2 | `R18Admissibility.exact_minimum_global_on_nonnegative_real_window`, `exact_minimum_unique_on_nonnegative_real_window` | Lean with `A>=0`, positive tangent at q=0 |
| Phase gapless pole at q=0 | Q7 | `phase_pole_gapless_at_zero` | Lean |
| Amplitude/phase scalar kernels differ at q=z=0 when m²≠0 | Q6 | `static_amplitude_phase_kernel_inequivalent` | Lean; rules out naive two-copy full-complex EFT matching |
| S004 and competing-gradient *static* symbols cannot coincide as polynomials for a>0 | E2–E3, independent action | `no_identical_static_spatial_symbol` | Lean under unchanged field/time/spatial normalization; does not forbid nonlocal sector-dependent reduction |
| Independently postulated competing-gradient rail has q*=a/(2b), positive modes | #16 PR #44 | `R16Completion.competing_* ` | Lean **different action**, no S004 equivalence |
| Full Ostrogradsky extra modes | H2 | `R16GhostObstruction`, `R16Completion`, `R16Energy` (#44) | Finite Hessian/affine witnesses Lean; complete field constraint analysis remains analytic |
| Low-pole quadratic spectral normal form | E1–E4 | `CORRECTED_ACTION_TO_EFT_KERNEL.md`, `verify_derivations.py` | Analytical Fourier-multiplier EFT *single real mode* on spectral window; not full WCT |
| Cubic/quartic vertex freedom | E5 | symbolic `cubic_shift_preserves_jet_{0,1,2}`, `cubic_shift_third_derivative` | Exact symbolic: V''' free at fixed quadratic poles |
| Nontrivial localized and spectrally/orbitally stable solitons | H1–H4, #16 | **OPEN** | No verified healthy original-action nonlinear states |
| Physical wavelength independent of parameter calibration | F1–F2, E4 | **NOT ESTABLISHED** | q* depends on `u0,kappa,gamma,theta,V'',eps,alpha,c0` |
| S004-to-PFF nonlinear controlled reduction | E5 and H4 | **OPEN** | No state/charge/energy map, mode-elimination error, or UV remedy supplied |

Note: theorem names and their exact hypotheses are the authoritative statements; symbolic checks validate identities but do **not** upgrade continuum PDE assertions into Lean theorems. The source's F1 `G C0>s` and F2 parameter inequality are given as analytical conditions. A Lean theorem requiring its concrete positive-q/tangent algebra is a **conditional** theorem; the globally selected background and normalized physical length remain unchosen.

## Physical admissibility by sector

1. **Zero background:** `F(0)=0`; no linear curvature selector at all in the unchanged S004 action.
2. **Static nonzero homogeneous:** on strict `K,T>0,B>0`, phase and amplitude have separate Q6 blocks. For `massSq>=0,q>0`, the zero-frequency kernel is strictly negative and every *real-pole* squared frequency is positive even if `G<0`. The positive-residue amplitude minimum in F1–F2 is a **conservative oscillation-frequency minimum**. There is no real finite-q unstable growth band in that region. Pole collisions and UV complex frequencies coexist with the opposite-residue ghost; a complete global theory remains unhealthy. When `massSq<0`, the static no-onset theorem does not apply and potential tachyonic instabilities must be assessed separately, but the parent Ostrogradsky/UV obstructions do not disappear.
3. **Rotating homogeneous:** Q4/Q5 full matrix contains amplitude-phase mixing and quartic z polynomial; no static-sector pole or minimum classification may be imported without recomputing this determinant at the chosen `Omega,u0`. Highest acceleration Hessian remains nonzero for `F kappa>0`.
4. **Spatially inhomogeneous stationary:** Fourier frequencies are generally coupled by a non-diagonal kernel; an assumed scalar `M(omega,k)` is unjustified.
5. **Degenerate boundary models:** `kappa=0`, `F=0`, `B=0`, or a singular field redefinition change the hypotheses/constraint rank. These are **not** continuous ghost-removal operations at fixed nonzero `F kappa`. For `kappa=gamma=0,theta>0` the spatial boundary is stable under positive spatial coefficients but has monotone `z=m²+q+Tq²`, without finite-q selection.
6. **Independent spatial competing-gradient action:** `L_alt=|psi_t|²-mu|psi|²+a|grad psi|²-b|Delta psi|²-(lambda/2)|psi|⁴` has `omega²=mu-aq+bq²`, unique `q*=a/(2b)` for `a,b>0`, and bounded positive quadratic Hamiltonian for `4bmu>a²`. This reverses the S004 static gradient sign. It is a distinct phenomenological model and still does not establish nontrivial localized static equilibrium in the defocusing gap-positive case.
7. **Nonlinear bridge:** even when E1–E4 match one positive-residue *amplitude* component, the *phase* remains gapless, the high pole remains, and E5's V''' and V'''' may be varied at fixed quadratic poles. Thus no single universal nonlinear coefficient, full-complex symplectic map or positive global Hamiltonian can be inferred from the dispersion identities alone.

## The exact missing-lemma contract for a physical S004-to-effective bridge

A genuine replacement for the conditional quadratic EFT must exhibit, simultaneously: (i) explicit admissible parent and target field spaces, conserved charge and energy maps, (ii) a finite, parameter-fixed background and chosen preparation data, (iii) a justified UV cutoff or genuinely constrained ghost-elimination operation **preserving the original action's equations on the matched sector**, (iv) the exact nonlocal normal form and controlled nonlinear vertex/exchange estimates, (v) a finite-time error bound in a norm stable under nonlinear Fourier leakage, and (vi) a Lyapunov/spectral/orbital stability theorem for the resulting localized states. The quadratic Fourier multiplier `Z^{-1/2}` in E1–E4 requires `Z>0` on its restricted spectral interval; it diverges or changes branch at the excluded high pole. These are **open theorem obligations**, not axioms.

**Milestone A:** scoped static S004 algebra/dispersion classification and conditional F2 uniqueness is kernel-checked when latest PR43 Lean CI passes. **Milestone B:** healthy wavelength selecting mechanism remains open. **Milestone C:** foundational nonlinear localized WCT bridge remains open. Keep issue #18 OPEN until original physical criteria pass, even if A is completed.


## Subsequent exact-parameter strengthening

`R18_F1ToF2.lean` formalizes the manuscript's original **F1** inequality itself, not merely a conditional tangent at an unexplained stationary point. Let `A=2K+G`, `B=4KT-G²>0`, `d0=1-4Km²>0`, `s=sqrt(d0)>0`, `N=G²+B*d0`, `c=sqrt(N/(A²+B))`. Assume `K>0`, `A>=0` and `A*s<G` (equivalent to F1's `G*C0>s` because `C0=(1-s)/(2K)`). The exact algebra `(A²+B)c²=G²+Bs²` proves `A*c<G`, hence `qStar=(G-A*c)/B>0`. The same identity shows `(A²+B)c-A G>0` and thus the supporting tangent at `q=0` is positive. With `A>=0` it remains positive on **every q>=0**. Therefore the exact untruncated amplitude squared frequency has a global minimum at `qStar` on its full nonnegative real-pole interval. The prior strict tangent-identity lemma makes the minimizer unique. The existence is **parameter-dependent** and does not remove the ghost or produce unstable growth.

`R18_Boundaries.lean` adds an exact sign classification for the F2 candidate: `qStar>0` iff `A*c<G`, and `qStar<=0` iff `G<=A*c`, whenever `B>0`. It also treats the `B=0` discriminant as `d0+2Gq`, the `K=0` inverse propagator as a linear polynomial in the squared frequency, and the `K=G=T=0` zero-background propagator as `z-q-m²`. In the purely spatial boundary `K=G=0,T>=0`, `m²+q+Tq²` increases strictly on `q>=0`, hence it cannot select a nonzero q. **None of these boundary identities says that a singular degeneration is a healthy completion of finite positive-K S004.**

`R18_Admissibility.lean` adds an exact static action-vs-spatial-rail polynomial inequivalence witness. For the original S004 static stiffness `m²+q+(K+G+T)q²`, the odd part between `q=±1` equals `2`. For the independent competing-gradient rail `mu-aq+bq²`, it equals `-2a`. If `a>0`, these cannot be the **same polynomial under identity of the field and fixed time normalization**. A restricted nonlocal frequency-dependent EFT normal form is not excluded by this argument; it must carry its sector, observables, energy, charge and controlled error terms. On the original static background the phase pole at `q=0` is exactly gapless, whereas an amplitude with `m²≠0` has a different inverse propagator there, preventing a naive two-identical-real-copy identification.


## Strongest strict-S004 onset obstruction (all real amplitude masses)

`R18_Onset.lean` closes an important case left conditional in the previous `m²>=0` no-onset theorem. For any `K>0`, `B=4KT-G²>0`, any `G` and **any real m²**, suppose a positive `q>0` is a zero-frequency root of the *unchanged* original static S004 inverse propagator:
```text
D(0,q)=-m²-q-(K+G+T)q²=0.
```
Strict curvature positivity gives `K+G+T=((2K+G)²+B)/(4K)>0`; therefore `m²=-q-(K+G+T)q²<0`. But `m²<0` implies at `q=0` that `d(0)=1-4Km²>1`, so
```text
z_minus(0)=(1-sqrt(1-4Km²))/(2K) < 0.
```
The homogeneous `q=0` squared frequency is **already tachyonic** when any `q>0` static zero-frequency crossing occurs. Thus no **primary finite-q static instability onset** exists in this strict static homogeneous sector, including the negative-mass cases. The no-static-crossing theorem for `m²>=0` and this new negative-mass implication together exhaust the static mass-sign cases. This does **not** classify every rotating-carrier, inhomogeneous, constrained, or alternative-action instability and is not a universal no-go theorem.

Lean: `finite_q_zero_requires_negative_mass`, `negative_amplitude_mass_tachyonic_at_q_zero`, `no_primary_finite_q_static_onset`. The theorem is a mathematical static-sector ordering result; real-pole collisions, high-frequency ill-posedness and relative negative residues remain additional independent obstructions.
