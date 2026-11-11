# Task 1: Ghost-free WCT dynamical completion. Classification and verdict

Issues: #16 (health and stability), #18 (finite wavelength), #45 (S004 → effective bridge).
Status: **scoped mathematical verdict. Issues remain open.** No healthy *WCT* completion is
claimed.

## 1. Required properties

| # | Property |
|---|----------|
| P1 | Hamiltonian bounded below on the declared state space (or on each conserved-charge shell) |
| P2 | No unwanted higher-time-derivative (Ostrogradsky) degrees of freedom |
| P3 | Genuine finite-wavelength selection |
| P4 | Nonlinear finite-energy localized states |
| P5 | A stable evolution that preserves them |
| P6 | Physical relation to WCT/S004 |

## 2. New kernel-checked theorems (this PR)

All of these build under `lean/` with Lean `v4.23.0` and Mathlib `v4.23.0`. They contain no
`sorry`, `admit` or local axiom. `#print axioms` lists only `propext`, `Classical.choice` and
`Quot.sound`.

### `ReleaseProofs/R16_OstrogradskyClass.lean`: class-wide obstruction (`n` fields)

These theorems cover Lagrangians `L = ½⟪a, W a⟫ + ⟪b(q,v), a⟫ + c(q,v)` with symmetric `W` and
**arbitrary** `b`, `c`.

| Theorem | Statement |
|---|---|
| `quadratic_lagrangian_expansion` | Exact expansion in `a`: momentum `p₂ = W a + b`, Hessian `W` |
| `ostroForm_unbounded_below/above` | Any `H = ⟪p₁, v⟫ + G(q,v,p₂)` is unbounded both ways when `v ≠ 0` |
| `bddBelow_forces_v_zero` | **Necessary degeneracy condition:** bounded below on a `p₁`-shift-invariant set forces `v = 0` there |
| `legendre_regular_iff_bijective` | `p₂` can be solved for every `(q, v)` exactly when `W` is bijective |
| `surjective_iff_det_ne_zero` | …exactly when `det W ≠ 0` (finite dimension) |
| `regular_hamiltonian_unbounded` | For invertible `W`, the canonical Hamiltonian is unbounded both ways |
| `primary_constraint_iff`, `primary_count` | Singular `W`: constraints `⟪p₂ - b, e⟫ = 0` for `e ∈ ker W`, which is `n - rank W` of them |
| `reduced_value_fibre_independent` | The reduced Hamiltonian on the primary surface is well defined |
| `primary_surface_unbounded` | A singular `W` *alone* does not remove the ghost; you also need secondary constraints that involve `p₁` |
| `s004_regular`, `s004_symm` | The fixed-flat S004 Hessian `(-2Fκ)·id` is in the regular (ghost) class |

### `ReleaseProofs/R16_PaisUhlenbeck.lean`: exact S004 mode ghost

| Theorem | Statement |
|---|---|
| `pu_hamiltonian_eq`, `pu_energy_rate` | Ostrogradsky `H` on the 4-jet, with `dH/dt = ẋ·(EL)` |
| `pu_normal_mode_identity` | `2(z₁-z₂)H = K[(ẏ₂²+z₂y₂²) - (ẏ₁²+z₁y₁²)]`, exact on the whole phase space |
| `pu_mode_equations` | `y₁`, `y₂` oscillate at `z₁`, `z₂` on solutions |
| `pu_ghost_unbounded` | `K ≠ 0` and `z₁ ≠ z₂` make `H` unbounded both ways |
| `pu_real_positive_poles` | `K, A, B > 0` and `A² > 4KB` give real, positive, distinct `ω²`. **Stable frequencies do not imply bounded energy** |
| `s004_mode_polynomial` | `R18.inversePropagator = -Kz² + A_q z - B_q`, exactly |
| `s004_vieta`, `s004_mode_ghost`, `s004_real_window_ghost` | In the real-pole window, the S004 mode Hamiltonian is unbounded both ways, for any nonzero normalization |

### `ReleaseProofs/R16_FirstOrderCandidate.lean`: independent replacement model

| Theorem | Statement |
|---|---|
| `candidate_lagrangian_affine` | Mode Lagrangian is affine in velocity: zero velocity Hessian, no acceleration |
| `candidate_partials`, `candidate_euler_lagrange_iff` | Exact partials; the Euler–Lagrange system is the first-order flow `ẏ = 2wx`, `ẋ = -2wy` |
| `candidate_energy_conserved` | That flow conserves the mode energy |
| `candidate_symbol_nonneg/zero_iff/unique_min` | `ω(q) = (k²-q)² ≥ 0`, with a unique zero at `q = k²` |
| `candidate_mode_energy_nonneg` | Every linear mode has energy `≥ 0` |
| `candidate_density_lower`, `candidate_density_sharp` | `-(β/2)r² + (γ/3)r³ ≥ -(3β²/16γ) r`, and the bound is sharp |
| `candidate_vs_s004_dof` | S004 has two distinct pole branches on the real window; the candidate has one |

## 3. Classification of tested model classes

| Class | P1 | P2 | P3 | P4 | P5 | P6 | Evidence |
|---|---|---|---|---|---|---|---|
| **Corrected S004** (fourth order in time, `Fκ ≠ 0`) | **Fails** | **Fails** | Static frequency minimum only (#18) | Not established | Not established | Is WCT | `s004_regular` + `regular_hamiltonian_unbounded`; `s004_real_window_ghost` |
| S004 under invertible point transformations | Fails | Fails | — | — | — | — | `R16Completion.invertible_point_change_cannot_degenerate` |
| S004 with auxiliary field (mixed kinetic term) | Fails | Fails | — | — | — | — | `R16_AuxiliaryEquivalence.auxiliary_hamiltonian_unbounded_*` |
| Any quadratic-in-`a` Lagrangian with `det W ≠ 0` | **Fails** | **Fails** | — | — | — | — | `regular_hamiltonian_unbounded` |
| Singular `W` without `p₁`-dependent secondary constraints | Fails | Fails | — | — | — | — | `primary_surface_unbounded` |
| Second-order competing-gradient field (`R16Completion`) | Linear modes `≥ 0`; defocusing nonlinearity | Yes (no `a`) | `q* = a/2b` frequency minimum | Defocusing: no nonzero single-mode equilibrium | Open | **Independent** | `competing_mode_energy_nonneg`, `finite_fourier_modes_coercive`, `defocusing_mode_no_nonzero_stationary` |
| **Fourth-order cubic–quintic Schrödinger** `i uₜ = (Δ+k²)²u - β|u|²u + γ|u|⁴u` on `H²(ℝ²)` | Yes on mass shells: `H ≥ -(3β²/16γ)M` | Yes (first order) | Yes: linear symbol has unique zero at `\|ξ\| = \|k\|` | Yes for `0 < m ≤ m_B`* | Yes: orbital stability of the minimizing set* | **Independent; no derivation from S004** | This PR (structure) + `ReyesMath.OrbitalStability.main`* |

\* Proved in `rickyjreyes/math` (`lean/MathProofs/OrbitalStability.lean`, `main` and
`global_flow`), not in this repository. That chain is audited under Task 2. The table does not
claim that this build re-checks it.

## 4. Verdict

1. **Rigorous obstruction (S004 and its regular class).** The corrected canonical S004 action has
   an invertible acceleration Hessian. Every Lagrangian in that class has an Ostrogradsky
   Hamiltonian that is unbounded above and below off `q̇ = 0`, whatever the lower-order terms.
   For the S004 Fourier mode specifically, the two real poles carry energies of opposite sign
   (`s004_real_window_ghost`). This holds even where both squared frequencies are positive. A
   ghost-free sector of S004 would need secondary constraints involving `p₁`
   (`bddBelow_forces_v_zero`). On the nonzero-field regular S004 patch, the acceleration Hessian has no
   intrinsic primary acceleration-momentum constraints. At the singular nodal
   locus, by contrast, there are two primary acceleration-momentum conditions.
   These primary conditions alone do not bound the canonical Hamiltonian if
   nonzero nodal velocities are admitted. Secondary constraints and their
   dynamical preservation must still be established before classifying any
   healthy exceptional sector.
2. **Mathematically viable independent model.** The first-order fourth-order cubic–quintic
   Schrödinger flow meets P1–P5 as mathematics. P1–P3 are structural facts proved here. P4–P5
   rest on the math-repository chain, subject to the Task 2 audit. That chain is restricted to
   `0 < m ≤ m_B`. It proves stability of the *set* of minimizers, not of individual profiles,
   and nothing about uniqueness or radial symmetry.
3. **P6 is not established.** No map from S004 (or any WCT action) to this model is constructed.
   Per #45 that needs a specified parent sector, an explicit state/energy map, derived
   coefficients, control of the excluded ghost/UV poles, and a quantitative error bound. None of
   that exists. **So no healthy WCT completion is claimed.** The model is a replacement, not a
   completion.

## 5. Remaining obligations

* A full field-theoretic Dirac analysis for any *proposed* degenerate S004 modification. (The
  necessary condition is proved here. Sufficiency is open.)
* A derivation, or a no-go result, for a reduction from a healthy parent to the first-order
  candidate, including nonlinear coefficients and error bounds (#45).
* Orbital stability for the second-order competing-gradient class (Grillakis–Shatah–Strauss
  type). This is open, and is not inherited from the Schrödinger chain.
* Importing or mirroring the math-repo chain here would need a cross-repository Lean dependency.
  It is not done, and is not claimed.

## November 2026 exact nodal-primary addendum (issue #53)

The Lean branch `proofs/s004-regular-nodal-dichotomy-20261109` now imports `ReleaseProofs.S004NodalEnergyObstruction`. Its kernel-verified declarations `nodal_primary_surface_exact`, `nodal_primary_surface_shift`, `nodal_primary_surface_not_bddBelow`, and `nodal_primary_surface_not_bddAbove` identify the corrected S004 nodal primary Legendre image at `q=0` and prove canonical energy unboundedness on that primary surface whenever it admits `v≠0`. These results do not establish secondary Dirac constraint propagation, consistent nodal evolution, or a nonlinear healthy-sector no-go. The corresponding strict original-model axiom audit covers 15 roots. Keep issue #53 open pending those additional obligations.
