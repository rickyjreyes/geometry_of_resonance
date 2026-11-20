# S004 foundational closure: constraint algorithm, characteristics, nonlinear ghost coupling

Issues: [#53](https://github.com/rickyjreyes/geometry_of_resonance/issues/53) (primary),
[#16](https://github.com/rickyjreyes/geometry_of_resonance/issues/16) (secondary).
Lean phases XXV–XXXII (modules listed in §8). Symbolic suite `derivations/s004_closure_verify.py`
(checks C1–C15), receipt `audit/s004_closure/verification.json`.

Evidence labels: **LEAN** (kernel-checked, pinned Lean 4.23.0 / Mathlib v4.23.0, strict axiom
audit), **ANALYTIC** (complete proof, not formalized), **NUMERIC** (reproducible numerics, no
proof), **CONDITIONAL** (needs a stated unproved or cited hypothesis), **REFUTED**, **OPEN**,
**FORMAL** (exact formal/asymptotic calculation without an existence theorem).

## Verdicts

| Model | Verdict |
|---|---|
| Original corrected S004 (fixed background, admissible region) | **`S004_SCOPED_NO_GO_PROVED`** |
| Spatial member `κ = γ = 0` (the only ghost-free nonnegative-curvature member of the family) | **`S004_FOUNDATIONAL_RECOVERY_NOT_ESTABLISHED`** |

The original action has no nontrivial healthy sector in any of the classes in §7. The decisive new
obstruction is not the Ostrogradsky ghost alone: inside the source admissibility region the
S004 operator is **elliptic in spacetime** at every point where the field is nonzero. The time
evolution is then Hadamard ill-posed, and the only well-posed state is the vacuum.

## 1. Exact action and conventions

Source: `WCT_CORRECTED_VARIATIONAL_CLOSURE_DRAFT.md` §§1–7.

* Field `ψ : M → ℂ`, real components `x = (Re ψ, Im ψ)`, `ρ = |ψ|²`.
* Signature `(−,+,…,+)`. This is forced by the source's projector `h^{μν} = g^{μν} + n^μ n^ν`.
  So `□ = −∂ₜ² + Δ`, `Δ_h = Δ` in the inertial frame.
* Weight `F(u) = u/D(u)²`, `D(u) = u + ε² e^{−2αu}`, `ε ≠ 0`.
  `F ≥ 0`, `F(u) = 0 ⟺ u = 0`, `F'(0) = 1/ε⁴`, `F(u) ~ 1/u` as `u → ∞`.
* Density `L_orig = ∇_μψ̄∇^μψ − V(ρ) + F(ρ)[κ|□ψ|² + θ|Δψ|² + γ Re(□ψ · conj Δψ)]`.
* Source admissibility: `κ > 0, θ > 0, γ² < 4κθ`. The potential `V` is unspecified by the source.
* In jets: `v = ẋ`, `a = ẍ`, `b = Δx`, `A = 2κ+γ`, `C = κ+θ+γ`. Physical orientation:
  `L_phys = −L_orig = |v|² − |∇x|² + V − F(κ|a|² − A a·b + C|b|²)`.
  The overall sign changes no equation, constraint or relative ghost sign.
* Boundary terms: compactly supported, periodic or decaying variations (source §7). Otherwise the
  higher-order boundary form must be kept.
* Ostrogradsky momenta: `p₂ = ∂L/∂a = F(Ab − 2κa)` and `p₁ = ∂L/∂v − ∂ₜp₂ = 2v − ∂ₜp₂`
  (S004 has no mixed `∂ₜ∂ᵢ` jets).
  * In Lean (`R16Class.Phase`), `p1` is conjugate to `q` and `p2` to `v`.
  * In `docs/foundational/GHOST_AND_CONSTRAINT_ANALYSIS.md` the same objects are named `p₀, p₁`.
  * In `L_orig` orientation all momenta change sign, so the nodal constraint reads `p₁ = −2v`.
* Model separation:
  * Every theorem here concerns the fixed-background action.
  * The principal-symbol statements hold pointwise on any fixed Lorentzian background with any unit
    timelike `n` (they are tensorial). The dynamics statements use the flat inertial background.
  * Dynamical-metric completions are not covered, except the one scalar–tensor candidate already
    analyzed (two scalar ghost directions; see `docs/foundational_recovery/`).
  * Spatial-only models and effective rails (Swift–Hohenberg/PFF, cubic–quintic Schrödinger) are
    separate models.
* Strata of the configuration space:
  * regular: `ρ > 0`, Hessian rank 2;
  * nodal: `ρ = 0`, Hessian rank 0;
  * parameter boundaries: `κ = 0` (degenerate family member), `γ² = 4κθ` (perfect-square symbol),
    and `ε = 0` (excluded).

## 2. Nonlinear constraint analysis

### 2.1 Why literal Dirac fails and what replaces it

Dirac's algorithm presupposes a Hessian of constant rank on an open set. The S004 acceleration
Hessian is `W = ∂²L/∂a² = −2κF(ρ)·I₂` (LEAN: `R16Class.s004_regular`, `S004NodalPrimary`). It has
rank 2 on the open regular set and rank 0 on the nodal set, which has empty interior.

The rank-independent framework is the Skinner–Rusk unified formalism on
`W = {(q, v, a, p₁, p₂)}`, with energy `E = p₁·v + p₂·a − L`:

* primary constraint: `p₂ = ∂L/∂a`;
* dynamics: `q̇ = v`, `v̇ = a`, `ṗ₂ = ∂L/∂v − p₁`, `ṗ₁ = ∂L/∂q`.

Its solution curves are exactly the C⁴ solutions of the Euler–Lagrange equation. The Ostrogradsky
Hamiltonian `p₁·v − |p₂|²/(4κF) + …` (H3) does **not** extend across the nodal stratum as a
Hamiltonian vector field: `v̇ = −p₂/(2κF)` is undefined at `F = 0`. The unified formalism is the
justified extension, and it agrees with the Euler–Lagrange equations by construction. The
"nodal energy" used in Phases XX–XXVI is this energy function `E` restricted to the node. It is
the conserved Noether energy, not a Hamiltonian that generates nodal dynamics.

### 2.2 The tangency condition

Preserving `p₂ = ∂L/∂a` gives
`p₁ = 2v − Ḟ(Ab − 2κa) − F(Aḃ − 2κȧ)`.

The jerk `ȧ` enters only through `2κF·ȧ` (ANALYTIC: check C15, exact sympy).

| Stratum | Tangency condition | Classification | Evidence |
|---|---|---|---|
| Regular (`κF ≠ 0`) | uniquely solvable for the jerk, for every `(v,a,b,ḃ,p₁)` | multiplier; **no secondary constraint** | LEAN `tangency_regular_unique_jerk` |
| Nodal (`F = Ḟ = 0`) | independent of the jerk; equivalent to **`p₁ = 2v`** | **genuine secondary constraint on the nodal stratum** | LEAN `tangency_nodal_iff` |
| Type change | jerk-independent ⟺ `κF = 0` | rank change = constraint-type change | LEAN `tangency_jerk_independent_iff` |
| Along any path at a nodal instant | `F = 0` and `Ḟ = 0` (Fermat: `F ≥ 0` with a zero there) | `p₁ = 2v` holds with the actual weight | LEAN `nodal_path_tangency_iff`, `nodal_weight_hasDerivAt_zero` |

**Answer on `p₁ = 2v`.** It is a genuine secondary constraint of the stratified algorithm on the
nodal stratum.

* It is not a constraint on any open region of phase space.
* It is not gauge-dependent: S004 has no local gauge symmetry, and U(1) is global.
* It is not a special-background identity: it holds for every differentiable path at every nodal
  instant, independently of `a`, `ȧ`, `b`, `ḃ`.
* The coefficient `2` is the normalization of `|ψₜ|²`.

The earlier report was correct as a jet identity. It was wrong to call it an open-region Dirac
constraint, and it overstated the "nodal energy bound". The corrected docstrings are in
`S004NodalDiracClosure`.

### 2.3 Higher conditions and sector closure

Impose the nodal stratum as a sector, `q ≡ 0`. Each step is forced by the previous one:

1. `q = 0` gives `v = 0` (tertiary).
2. `v = 0` gives `a = 0`.
3. The jerk is then a free multiplier, fixed to `0` by preservation of `a = 0`.
4. Then `p₂ = 0` and `p₁ = 2v = 0`.

The four per-component constraints `(q, v, p₂, χ = p₁ − 2v)` have the bracket matrix
`∇C·J·∇Cᵀ`, with determinant `1` (LEAN `nodalBracketMatrix_eq`, `nodalBracketMatrix_det`). All
four are second class. They cut phase space to the single vacuum point (LEAN
`nodal_invariant_constraint_set`), leaving **0 physical degrees of freedom**. Every permanently
nodal path has canonical lift `(0,0,0,0)` (LEAN `permanently_nodal_lift_is_origin`).

The free-`p₁` holonomic fibre of Phases XXII–XXIII is dynamically spurious. Its points with
`p₁ ≠ 0` are the lift of no differentiable path (LEAN `holonomic_fibre_unattained`).

The transient nodal set `{q = 0, p₂ = 0, χ = 0}` has a singular `3×3` bracket of rank 2 (LEAN
`transientBracket_det_zero`). It is not a phase space, and it is left immediately when `v ≠ 0`.

**Node crossings in the homogeneous sector** (ANALYTIC: C8 exact; LEAN algebraic consequence
`nodal_crossing_acceleration_zero`, `correctedWeight_hasDerivAt_zero`):

* The Euler–Lagrange equation at a nodal instant reads `(1 + 2κF'(0)|v|²)a = 0`, so `a = 0`.
* At the next order, `ȧ = V'(0)v/(1 + 6κ|v|²/ε⁴)`.
* The formal crossing series is unique given `(t₀, v₀)`. Each `aₙ` enters linearly with
  coefficient `n(n−1)(1 + κv₀²n(n−1)/ε⁴) ≠ 0` (C9). Regular crossings therefore form a
  codimension-2 family in the real homogeneous sector.
* At a regular nodal instant the U(1) charge vanishes (LEAN `nodal_instant_charge_vanishes`).
  Since the charge is conserved, **charged homogeneous solutions never cross the node regularly**
  (ANALYTIC: Noether).

### 2.4 Degree-of-freedom ledger

| Sector | Constraints | Class | Physical DOF (real, per point) | Evidence |
|---|---|---|---|---|
| Regular, nodeless field | auxiliary pair only (H2) | second class | 4 (2 ordinary + 2 Ostrogradsky) | ANALYTIC (H2); LEAN finite-dim |
| Configuration whose zero set has empty interior (field theory, e.g. vortices) | nodal jet identities at nodal points only | — | 4 on the open dense regular set; the ghost lives there | ANALYTIC |
| Invariant nodal sector | `q, v, p₂, p₁−2v` | all second class (det 1) | 0 (vacuum point) | LEAN |
| Spatial member `κ=γ=0` | none | — | 2 | ANALYTIC |

## 3. Invariance and non-invariance of proposed healthy sectors

| Proposed sector | Result | Evidence |
|---|---|---|
| Nodal primary surface / holonomic fibre | invariant only at the vacuum | LEAN |
| Fixed U(1) charge fibre | energy unbounded above and below whenever radial velocity `≠ 0` | LEAN (merged, `S004ChargeNoGo`) |
| Any sector defined without `p₁` (e.g. by any constraint the regular stratum generates — there are none) | bounded-below energy ⟹ every classical trajectory in it is **static** | LEAN `shift_invariant_bounded_sector_is_static` |
| Restricted positive-energy linear Pais–Uhlenbeck sector at `k` | **not invariant**: at second order the ghost coordinate of mode `2k` is forced by `S/K` whenever `(σ₀,σ₂) ≠ 0` | ANALYTIC (exact second-order expansion C12) + LEAN `ghost_mode_forced`, `sigma2_neg` |
| Nonlinear continuation of the ordinary mode (Lyapunov standing-wave family) | **obstructed** at every certified resonance with `σ₂ ≠ 0` (§4) | ANALYTIC (Poincaré–Lindstedt) + LEAN `certified_no_periodic_second_order` |
| Exceptional invariant families: homogeneous, real (`Im ψ = 0`), single plane waves `Ae^{i(kz−ωt)}`, U(1) orbits | invariant (by symmetry), but either contain the Ostrogradsky ghost in their own phase space or are finite families whose transverse perturbations are Hadamard-unstable (§7) | ANALYTIC |

Scope of "not invariant" for the actual flow:

* The statement is exact for the formal second-order Taylor coefficient.
* For the true nonlinear flow it needs a solution map that is `C²` in the amplitude δ. This holds
  in the Cauchy–Kovalevskaya analytic class on short times. CONDITIONAL beyond that class.

## 4. Resonance: classification and certificate

Amplitude dispersion (repository Q6/Q7, `R18.inversePropagator`):

`D(z,q) = (z−q) − m² − K(z−q)² + Gq(z−q) − Tq²`,

with `K = F(ρ₀)κ`, `G = F(ρ₀)γ`, `T = F(ρ₀)θ`, `m² = −2ρ₀V''(ρ₀)`.

Branch assignment:

* lower pole `z₋` is ordinary (positive Pais–Uhlenbeck energy);
* upper pole `z₊` is the ghost (negative energy, LEAN `certified_upper_mode_negative_energy`
  via the existing `R16PU.pu_single_mode_values`).

The ghost resonance condition is `(2ω₋(k))² = z₊(4k²)`.

* **Exact reduction** (LEAN `resonanceGap_formula`): the `q` and `Gq` terms cancel exactly, giving
  `2K[z₊(4q) − 4z₋(q)] = √D(4q) + 4√D(q) − 3`.
* **Classification for `γ = 0`** (LEAN `resonance_exists_of_band`, `resonance_band_of_resonance`):
  * A resonance with `k` and `2k` inside the real window exists **iff** `9/25 ≤ δ² ≤ 3/5`,
    where `δ² = 1 − 4Km²`.
  * It occurs at `B k⁴ = δ² − ((3+5δ²)/8)²`, `B = 4KT`.
  * At that point `√D(k²) = (3+5δ²)/8` and `√D(4k²) = (3−5δ²)/2`.
  * No interval arithmetic is needed. The band edges correspond to `k* = 0` and to `2k*` at the
    window edge.
* **Certified instance** (LEAN `certified_*`; exact sympy C12). Parameters:
  * actual weight with `ε = 1`, `α = 0`, at `ρ₀ = 1/2`, so `F = 2/9` and `F' = 4/27`;
  * `κ = 1/2`, `θ = 1`, `γ = 0` (admissible);
  * `V'(ρ₀) = 0` and `V''(ρ₀) = −459/400`, i.e. `δ = 7/10`. A potential realizing this is
    `V = −(459/800)(ρ − 1/2)²`, a symmetry-breaking hat whose potential energy `−V` is minimal on
    `|ψ|² = 1/2`.

  Certified values:
  * `k*⁴ = 53703/204800`, so `k* ≈ 0.71559 ∈ [1/100, 37/50]`;
  * `D(k*²) = (109/160)²` and `D(4k*²) = (11/40)²`: both strictly inside the window, with
    distinct branches;
  * the resonance is exact;
  * `ω*² − k*² = 459/320`;
  * `σ₂ = −4131·√(1/2)/1600 ≠ 0` for `V'''(ρ₀) = 0`.
* **Cancellation surfaces**:
  * At the certified resonance, `σ₂ = 0` iff `V'''(ρ₀) = 4131/800` (LEAN
    `certified_sigma2_cancellation_iff`). There `σ₀ ≈ 0.585 ≠ 0`, but that forcing is static and
    non-resonant.
  * Standing and traveling waves do not cancel: the traveling-wave coefficient is exactly `2σ₂`
    (C12).
* **Preserved failure — REFUTED.** The earlier candidate used `V''(ρ₀) = −1/2` with
  `k ∈ [1/100, 1]`.
  * `k = 1` puts `2k` outside the window: `D(4) = −65/81`.
  * More strongly, `δ² = 7/9 > 3/5`, so no resonance exists anywhere: LEAN
    `original_candidate_no_resonance`, and the numerical minimum of the resonance function is
    0.416 (C12).
  * The revised window `δ = 7/10`, `k ∈ [1/100, 37/50]` was a candidate. It is now certified by
    the closed form above.

## 5. Ghost-mode forcing: what is and is not implied

Second-order expansion around the certified background (real sector, exact):
`x = A + δ cos(kz)cos(ωt) + δ²x₂ + O(δ³)`.

The `cos 2kz` part of the `δ²` source is `σ₀ + σ₂ cos 2ωt`, with:

* `σ₂ = (A/2)(3V'' + 2ρ₀V''' − 9F'(ρ₀)·symbolForm(ω²−k², k²))` (C12; LEAN `sigma2_factor`);
* `σ₂ < 0` at every real-window mode when `V'' < 0`, `V''' ≤ 0`, `F'(ρ₀) ≥ 0` (LEAN `sigma2_neg`).

The `2k` mode obeys `ÿ₊ + z₊y₊ = (σ₀ + σ₂cos2ωt)/(2K)` in its ghost coordinate (LEAN
`ghost_mode_forced`).

| Statement | Status |
|---|---|
| Nonzero nonlinear coupling into the ghost branch | **ANALYTIC** (exact coefficient) + **LEAN** (sign, certified value) |
| Bounded off-resonant excitation, `\|y\| ≤ 2\|c\|/\|z₊ − Ω²\|` uniformly in time | **LEAN** `offresonant_forced_response` (exact for the second-order coefficient) |
| Exact resonant forcing at the certified `k*` | **LEAN** |
| Secular growth `y₊ ∋ σ₂ t sin(Ωt)/(4KΩ)` (rate `\|σ₂\|/(4KΩ) ≈ 1.47` at `k*`) within second-order perturbation theory | **LEAN** `resonant_secular_solution`, `secular_amplitude_unbounded` |
| Non-existence of the ordinary-mode Lyapunov family through the resonance | **ANALYTIC** (+ LEAN core) |
| Explosive / runaway instability of the full nonlinear S004 evolution | **OPEN** — not claimed |

**Time-scale caveat (CONDITIONAL).** At the certified point the third harmonic `3k*` lies
outside the real window (LEAN `certified_third_harmonic_elliptic`: `D(9k*²) = −41159/25600`). It
has complex frequency with growth rate `Im ω(3k*) ≈ 0.905` (C12). If it is generated at order
`δ³` with a nonzero coefficient (generic; that coefficient was not computed), it overtakes the
resonant secular term at `t ~ 2 ln(1/δ)/0.905`. That is long before the secular time `~1/δ`.

So in the admissible region the controlled perturbative regime ends through spacetime
ellipticity, not through resonant ghost pumping. Resonance is a structural obstruction to
healthy invariant families; it is not the dominant dynamical instability.

## 6. Nodal singularity status

Exact algebraic results (FORMAL):

* The leading nodal balance of the real homogeneous equation, valid for `|ẍ| ≫ |x|`, is
  `(x²ẍ)¨ + xẍ² = 0`.
* On `x = τ^p` it equals `p(p−1)²(10p−6)τ^{3p−4}` (C10). The unique nontrivial singular exponent
  is `p = 3/5`.
* Perturbations of `x₀ = Cτ^{3/5}` have indicial polynomial `μ(μ+1)(5μ−8)(5μ−3)/25` (C13).
  * The trivial roots are `−1` (time translation) and `0` (amplitude).
  * The nontrivial roots `3/5` and `8/5` are positive, so at the formal level the collision is
    a 4-parameter, open family.
* The neglected ordinary terms are relatively `O(τ^{4/5})`.

Numerics (NUMERIC, homogeneous real sector only):

* Five random initial data all collide.
* Fitted exponents are 0.595–0.604.
* Relative energy drift near collision is `≤ 6·10⁻⁵`.
* The velocity diverges like `τ^{−2/5}`.

Not proved (OPEN):

* existence of solutions with this asymptotics (needs a Fuchsian fixed-point argument);
* the initial-data class;
* any statement for inhomogeneous data or the full PDE.

This does not change the sector classification. It is recorded and not pursued further.

## 7. Regular/nodal classification and the characteristic cone

**Principal symbol** (ANALYTIC: C1 brute-force linearization of `sympy.euler_equations` in 1+1
and 1+2 dimensions, about an arbitrary background). The symbol is
`2F(ρ₀)(κX² − γXY + θY²)·I₂`, with `X = τ² − |ξ|²` and `Y = |ξ|²`.

| Statement | Status |
|---|---|
| Symbol positive definite ⟺ `κ>0, θ>0, γ²<4κθ` (source admissibility) | LEAN `symbolForm_pos_definite_iff` |
| Admissible ⟹ spacetime-elliptic at every regular point; no real characteristic | LEAN `admissible_elliptic_covector`, `admissible_no_real_characteristic` |
| Strict hyperbolicity ⟺ `γ² > 4κθ`, `2κ+γ > 0`, `κ+θ+γ > 0` (outside admissibility) | LEAN `two_positive_roots_iff` |
| PSD curvature with a real characteristic only on `γ² = 4κθ` (perfect square `κ(X − γY/2κ)²`) | LEAN |
| Exact frozen roots `λ = n·w`, `w² = μ`, `Re λ = n·c`, `c > 0` independent of `F(ρ₀)` | LEAN `admissible_hadamard_growth` |
| `exp(bx)/(1+x)^N → ∞`: no finite Sobolev loss absorbs Hadamard growth | LEAN |
| Lower-order constant terms do not remove growth (`max Re λ/k → c_uv`) | NUMERIC adversarial (C5); ANALYTIC by Rouché |
| Linearization about any homogeneous static or co-rotating background: Cauchy problem not well-posed in any Sobolev pair | ANALYTIC (self-contained: constant coefficients) |
| Linearization about any smooth background with a regular point: not well-posed (Lax 1957; Mizohata 1961); nonlinear solution map not `C¹` there | CONDITIONAL (cited Lax–Mizohata theorem; hypotheses verified: distinct non-real roots) |
| No localized state can pass Sobolev spectral or orbital stability tests | CONDITIONAL on the same citation (ANALYTIC for homogeneous states) |
| `κ → 0⁺` on the `γ = 0` ray: Hadamard constant unbounded, so no Sobolev-controlled singular limit | LEAN `hadamard_constant_unbounded_as_kappa_to_zero` |

**Coefficient family** (LEAN, `S004CoefficientFamily`):

* Every admissible member is ghost-bearing, elliptic and not strictly hyperbolic.
* Ghost freedom with nonnegative curvature forces `κ = γ = 0`, which is outside admissibility.
* Strict hyperbolicity forces an indefinite curvature form.
* PSD members with real characteristics keep the ghost when `κ ≠ 0`.

## 8. Lean theorems (new in this branch)

* Pinned Lean `v4.23.0`, Mathlib `v4.23.0` (manifest rev `37df177`).
* `lake build` succeeds.
* The strict audit `lean/S004RegularAxiomAudit.lean` with `lean/check_s004_regular_axioms.py`
  covers **100 roots**. All depend only on `propext`, `Classical.choice`, `Quot.sound`.
  * The checker was negative-tested: it fails on a truncated log and on `sorryAx`.
* No `sorry`, `admit`, `axiom` or `native_decide`.

| Module | Main theorems | Imports |
|---|---|---|
| `S004CharacteristicCone` | `symbolForm_pos_definite_iff`, `admissible_elliptic_covector`, `two_positive_roots_iff`, `boundary_perfect_square`, `hadamardRatio_root`, `admissible_hadamard_growth`, `hadamard_constant_unbounded_as_kappa_to_zero`, `exp_beats_every_polynomial_weight` | Mathlib |
| `S004NodalDiracClosure` | `nodal_weight_hasDerivAt_zero`, `nodal_instant_momenta`, `nodalBracketMatrix_det`, `nodal_invariant_constraint_set`, `permanently_nodal_lift_is_origin`, `holonomic_fibre_unattained`, `correctedWeight_hasDerivAt_zero`, `nodal_crossing_acceleration_zero` | `R16_GhostObstruction`, `S004NodalHolonomic`, `S004NodalShiftClassification` |
| `S004StratifiedConstraint` | `tangency_regular_unique_jerk`, `tangency_nodal_iff`, `tangency_jerk_independent_iff`, `nodal_path_tangency_iff`, `shift_invariant_bounded_sector_is_static`, `nodal_instant_charge_vanishes` | `S004NodalDiracClosure`, `S004ChargeFibre` |
| `S004CoefficientFamily` | `accel_hessian_degenerate_iff`, `admissible_member_obstructed`, `ghostfree_psd_members`, `hyperbolic_region_curvature_indefinite`, `psd_hyperbolic_members_keep_ghost`, `spatial_member_single_branch`, `spatial_member_no_finite_k_minimum` | `R16_EnergyAndConstraints`, `S004CharacteristicCone`, `R18_FiniteScale` |
| `S004HarmonicGhostForcing` | `sigma2_factor`, `sigma2_neg`, `ghost_mode_forced`, `offresonant_forced_response`, `resonant_secular_solution`, `secular_amplitude_unbounded` | `R16_PaisUhlenbeck`, `S004CharacteristicCone` |
| `S004ResonanceCertificate` | `resonanceGap_formula`, `resonance_exists_of_band`, `resonance_band_of_resonance`, `certified_resonance`, `certified_window`, `certified_resonant_branch`, `certified_upper_mode_negative_energy`, `certified_sigma2`, `certified_sigma2_cancellation_iff`, `certified_no_periodic_second_order`, `certified_third_harmonic_elliptic`, `original_candidate_no_resonance` | `R18_FiniteScale`, `R16_PaisUhlenbeck`, `S004HarmonicGhostForcing` |
| `S004ScopedNoGo` | `s004_scoped_no_go` | regular, charge, characteristic, nodal, family modules |
| `S004Issue53Interface` | `s004_issue53_interface` (eight-item #53 interface) | `S004ScopedNoGo`, `S004StratifiedConstraint`, `S004ResonanceCertificate` |

**Source-to-theorem correspondence.**

* The polynomials `linearSymbolRaw`, `sigma0Raw`, `sigma2Raw` are pasted verbatim from the sympy
  expansion of the source density. The suite asserts their equality (C12).
* `linearSymbol_eq_inversePropagator` ties them to the repository's `R18.inversePropagator`.
* The jerk coefficient `2κF·I₂` and the principal symbol are checked against the source density
  (C15, C1).
* A passing Lean build certifies the encoded statements, not this correspondence. The
  correspondence rests on the exact symbolic checks.

## 9. Certificates and preserved failures

`python derivations/s004_closure_verify.py --output NEW.json` runs C1–C15, and
`python -m unittest -v derivations.test_s004_closure` runs 20 tests. CI workflow:
`.github/workflows/s004-closure.yml`. It recomputes the suite with pinned dependencies and requires
the exact fields to reproduce the committed receipt.

Preserved failures:

1. **REFUTED:** the original resonance certificate (`V''(ρ₀) = −1/2`, `k ∈ [1/100, 1]`); see §4.
2. **Repaired check:** the first crossing-series check expanded a degree-9 polynomial with nine
   symbolic unknowns. It exhausted memory and killed the worker. It now solves order by order with
   truncation; the mathematics is unchanged.
3. **Repaired check:** the lower-order Hadamard test first failed because `numpy.float64 * poly1d`
   returns a coefficient array. With Python floats it passes, with relative error at `k = 1000`
   below 1.7%, converging like `1/k`.
4. **Repaired audit:** an intermediate commit emptied the axiom checker, which then passed
   vacuously. It was restored in `175b13f` and negative-tested.
5. **Corrected framing:** `p₁ = 2v` was previously described as an open-region secondary
   constraint, and the nodal energy values as a lower bound. See §2.2.

## 10. Commit, CI, PR

See the pull request for the exact head SHA and hosted CI results. Local verification commands:

```bash
cd lean && lake build && lake env lean S004RegularAxiomAudit.lean > /tmp/ax.log && python3 check_s004_regular_axioms.py /tmp/ax.log
```

```bash
python derivations/s004_closure_verify.py --output /tmp/s004.json && python -m unittest -v derivations.test_s004_closure
```

## 11. Completion matrices

### #53

| Obligation / criterion | Status |
|---|---|
| Freeze action, boundary terms, canonical variables; separate Lorentzian and spatial reductions | DONE (§1) |
| Connect R16 Ostrogradsky / PU / bracket / constraint modules to a common statement | DONE: `s004_scoped_no_go`, `s004_issue53_interface` reuse `R16Class`, `R16PU`, `R16Energy`, `R18` |
| Negative-energy obstruction for its actual hypothesis class | DONE: regular patch, every charge fibre, every `p₁`-shift-invariant sector (static), stratified constraints (LEAN) |
| Healthy sector: constraints, dynamics, invariance, positivity | No healthy sector exists in the scoped classes. Nodal sector is the vacuum (LEAN). Linear PU sectors are non-invariant (ANALYTIC). Lyapunov continuation is obstructed at certified resonances. |
| Rule out importing independent wavelength-selection dynamics without a controlled reduction | DONE: spatial member and rails are replacements. Singular limit: LEAN `hadamard_constant_unbounded_as_kappa_to_zero`, `spatial_member_no_finite_k_minimum`. |
| No `sorry` or unreviewed axioms; dependency and hypothesis report; pinned build | DONE locally (100 roots). Hosted CI: see PR. |
| Definitive scoped no-go or identified unresolved assumptions | DONE: scoped no-go. Unresolved assumptions listed in §12. |
| Reconcile with R16 and avoid duplication | DONE: reuses `R18.lowerPole`, `R18.upperPole`, `upper_pole_on_shell`, `R16PU.pu_mode_equations`, `R16PU.pu_single_mode_values`, `R16Class.bddBelow_forces_v_zero` |

**Recommendation.** The obstruction question of #53 is answered by a scoped no-go. The remaining
non-Lean ingredients are the cited Lax–Mizohata step (for inhomogeneous backgrounds) and the
infinite-dimensional H2 count. Keep #53 open until review and green hosted CI. Closing it is the
owner's decision.

### #16

| Criterion | Original S004 | Spatial member (replacement) |
|---|---|---|
| DOF counted explicitly | DONE: 4 (regular), 0 (nodal sector) | DONE: 2 |
| Ghost / extra-mode conditions derived | DONE: ghost ⟺ `κF ≠ 0`; stratified | none |
| Bounded physical Hamiltonian or equivalent constrained formulation | REFUTED in the scoped classes | bounded below if `−V ≥ −c` (ANALYTIC); **not equivalent** to S004 |
| Linear dispersion stable in the intended region | REFUTED: elliptic at high `k` in the admissible region | single positive branch (LEAN); no finite-`k` minimum (LEAN) |
| A localized solution passes spectral/orbital stability | impossible in Sobolev sense (CONDITIONAL in general, ANALYTIC for homogeneous states) | OPEN: no confined state constructed; nonlinear well-posedness with nodes OPEN |
| Stable parameter domain documented | none with `κ > 0` | `κ = γ = 0`, `θ ≥ 0` (outside source admissibility) |

#16 remains **OPEN**. Its completion criteria are not met. The spatial member gives up:

* the time-curvature term and the covariant `Θ₀`;
* two of four degrees of freedom and the second branch;
* the finite-`k` static frequency minimum of #18.

There is no controlled correspondence with S004.

## 12. The single most important remaining obstruction

**Missing theorem: variable-coefficient ill-posedness of the S004 linearization.** For every
smooth solution `ψ₀` and every point with `ψ₀ ≠ 0`, the Cauchy problem for the linearization of
the corrected S004 equation is not well-posed in any pair of Sobolev spaces near that point.

* Today this rests on the classical Lax–Mizohata theorem. The hypotheses are verified (principal
  symbol `F(ρ₀)p(τ,ξ)·I₂` with four distinct non-real roots), but the theorem is cited, not proved
  or formalized here.
* A self-contained proof — WKB construction with complex phase, or a formalization — would make
  the no-go unconditional for every localized state.
* **How far the reduction goes.**
  * The background enters the principal symbol only through the scalar factor `F(ρ₀) > 0`. So the
    characteristic variety `p(τ,ξ) = 0` is the same at every regular point.
  * Hence the eikonal equation is solved globally by linear phases `φ = ξ₀·x − τ̂ t` with
    `p(τ̂, ξ₀) = 0` and `Im τ̂ > 0` (LEAN `admissible_hadamard_growth` gives these roots).
  * The first transport operator is then the constant complex vector field `∇_ζ p(τ̂, ξ₀)·∇`,
    which is locally solvable.
  * The step not yet written out is **localization**. A spatial cutoff creates commutator terms
    exactly where the WKB solution is already exponentially large. The Lax–Mizohata argument
    controls this with a complex phase concentrated at the point.
  * A self-contained version of that localization step, for the operator
    `F(ρ₀)p(∂)·I₂ + (order ≤ 3)`, is the precise missing lemma.

On the recovery side, the corresponding open structural question is **whether any completion
outside the `(κ, θ, γ)` family can change the type of the principal symbol while keeping S004's
predictions**. The candidates are a dynamical foliation `n^μ`, a dynamical metric beyond the
analyzed candidate, or a nonlocal curvature functional. Inside the family this is impossible (§7).
