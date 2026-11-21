# S004 foundational closure: constraint algorithm, characteristics, nonlinear ghost coupling

Issues: [#53](https://github.com/rickyjreyes/geometry_of_resonance/issues/53) (primary),
[#16](https://github.com/rickyjreyes/geometry_of_resonance/issues/16) (secondary).

* Lean phases XXV–XXXIII; modules are listed in §8.
* Symbolic suite `derivations/s004_closure_verify.py` (checks C1–C15), receipt
  `audit/s004_closure/verification.json`.
* Phase XXXIII: `derivations/s004_illposedness_verify.py` (I1–I8), receipt
  `audit/s004_closure/illposedness_verification.json`, full proof in
  [`VARIABLE_COEFFICIENT_ILLPOSEDNESS.md`](VARIABLE_COEFFICIENT_ILLPOSEDNESS.md). Summary in §13.

Evidence labels:

* **LEAN** = LEAN VERIFIED: kernel-checked, pinned Lean 4.23.0 / Mathlib v4.23.0, strict axiom
  audit.
* **ANALYTIC** = ANALYTICALLY PROVED: complete written proof, not formalized.
* **NUMERIC** = NUMERICALLY VERIFIED: reproducible numerics, no proof.
* **CONDITIONAL**: needs a stated unproved or cited hypothesis.
* **REFUTED**, **OPEN**.
* **FORMAL**: an exact formal or asymptotic calculation without an existence theorem.

## Verdicts

| Model | Verdict |
|---|---|
| Original corrected S004 (fixed background, admissible region) | **`S004_SCOPED_NO_GO_PROVED`** |
| Spatial member `κ = γ = 0` (the only ghost-free nonnegative-curvature member of the family) | **`S004_FOUNDATIONAL_RECOVERY_NOT_ESTABLISHED`** |

The original action has no nontrivial healthy sector in any of the classes in §7. The decisive
obstruction is not the Ostrogradsky ghost alone. Inside the source admissibility region the S004
operator is **elliptic in spacetime** at every point where the field is nonzero.

Phase XXXIII (§13) proves, without citing an external theorem, that the linearized Cauchy problem
is then Sobolev ill-posed:

* near **every regular point of every smooth background**;
* in every pair of Sobolev spaces, on every time interval.

Only backgrounds that vanish identically on the region considered escape the theorem. There the
linearization is the second-order wave operator.

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
| Lower-order constant terms do not remove growth (`max Re λ/k → c_uv`) | NUMERIC adversarial (C5); ANALYTIC by Rouché (Phase XXXIII Theorem 3) |
| Linearization about any homogeneous static or co-rotating background, on- or off-shell: exact growing modes at fixed time; Cauchy problem not well-posed in any Sobolev pair | ANALYTIC (Phase XXXIII Theorem 3, constant coefficients); NUMERIC (I6) |
| Linearization about **any smooth background** (solution or not) **near any regular point**: the local Cauchy estimate fails for every Sobolev pair, interval, ball and constant. No bounded solution operator with sources, no `C₀`-semigroup (static or co-rotating states), no evolution system satisfying (S1′). | **ANALYTIC — self-contained** (Phase XXXIII Theorems 1, 2, Corollary 2; §13). Lax–Mizohata is no longer cited. NUMERIC end to end (I7, three backgrounds, four negative controls). LEAN for the algebraic and 1D-integral ingredients. |
| Nonlinear solution map Fréchet-differentiable at a smooth solution with a regular point | impossible: ANALYTIC (Phase XXXIII Theorem 4) |
| Localized states with a nonzero core: no Sobolev `C₀` linearization; no `C¹` flow nearby. Hence spectral and GSS/Lyapunov orbital-stability methods are inapplicable. | ANALYTIC (Theorems 2, 4). Orbital stability with merely continuous flows: OPEN (§13). |
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
  covers **136 roots**: the 100 roots of Phases XVIII–XXXII and 36 Phase XXXIII roots. All depend
  only on `propext`, `Classical.choice`, `Quot.sound`.
* The checker enumerates every root by name from the audit file and requires each to be reported
  exactly once. It was negative-tested and fails on each of the following:
  * a truncated log;
  * an injected `sorryAx`;
  * a root missing from the log;
  * a root missing from the audit file;
  * a Lean error in the log;
  * a duplicated report.
* No `sorry`, `admit`, `axiom` or `native_decide`.

| Module | Main theorems | Imports |
|---|---|---|
| `S004CharacteristicCone` | `symbolForm_pos_definite_iff`, `admissible_elliptic_covector`, `two_positive_roots_iff`, `boundary_perfect_square`, `hadamardRatio_root`, `admissible_hadamard_growth`, `hadamard_constant_unbounded_as_kappa_to_zero`, `exp_beats_every_polynomial_weight` | Mathlib |
| `S004NodalDiracClosure` | `nodal_weight_hasDerivAt_zero`, `nodal_instant_momenta`, `nodalBracketMatrix_det`, `nodal_invariant_constraint_set`, `permanently_nodal_lift_is_origin`, `holonomic_fibre_unattained`, `correctedWeight_hasDerivAt_zero`, `nodal_crossing_acceleration_zero` | `R16_GhostObstruction`, `S004NodalHolonomic`, `S004NodalShiftClassification` |
| `S004StratifiedConstraint` | `tangency_regular_unique_jerk`, `tangency_nodal_iff`, `tangency_jerk_independent_iff`, `nodal_path_tangency_iff`, `shift_invariant_bounded_sector_is_static`, `nodal_instant_charge_vanishes` | `S004NodalDiracClosure`, `S004ChargeFibre` |
| `S004CoefficientFamily` | `accel_hessian_degenerate_iff`, `admissible_member_obstructed`, `ghostfree_psd_members`, `hyperbolic_region_curvature_indefinite`, `psd_hyperbolic_members_keep_ghost`, `spatial_member_single_branch`, `spatial_member_no_finite_k_minimum` | `R16_EnergyAndConstraints`, `S004CharacteristicCone`, `R18_FiniteScale` |
| `S004HarmonicGhostForcing` | `sigma2_factor`, `sigma2_neg`, `ghost_mode_forced`, `offresonant_forced_response`, `resonant_secular_solution`, `secular_amplitude_unbounded` | `R16_PaisUhlenbeck`, `S004CharacteristicCone` |
| `S004ResonanceCertificate` | `resonanceGap_formula`, `resonance_exists_of_band`, `resonance_band_of_resonance`, `certified_resonance`, `certified_window`, `certified_resonant_branch`, `certified_upper_mode_negative_energy`, `certified_sigma2`, `certified_sigma2_cancellation_iff`, `certified_no_periodic_second_order`, `certified_third_harmonic_elliptic`, `original_candidate_no_resonance` | `R18_FiniteScale`, `R16_PaisUhlenbeck`, `S004HarmonicGhostForcing` |
| `S004ScopedNoGo` | `s004_scoped_no_go`, `s004_scoped_no_go_localization` (Phase XXXIII) | regular, charge, characteristic, nodal, family, localized modules |
| `S004Issue53Interface` | `s004_issue53_interface` (eight-item #53 interface), `s004_issue53_interface_with_localization` (items 1–8 plus localization items 9–10) | `S004ScopedNoGo`, `S004StratifiedConstraint`, `S004ResonanceCertificate`, `S004LocalizedHadamard` |
| `S004CharacteristicCone` (Phase XXXIII additions) | `laplaceSymbol_factor`, `hadamardRatio_add_conj`, `hadamardRatio_mul_conj`, `laplaceSymbol_hasDerivAt`, `laplaceSymbolDeriv_at_hadamard_root`, `laplaceSymbolDeriv_homogeneous`, `laplaceSymbol_at_hadamard_root`, `hadamard_root_simple`, `exists_growing_simple_root` | Mathlib |
| `S004LocalizedHadamard` (new, Phase XXXIII) | `beam_eikonal`, `beam_transport`, `laplaceSymbol_first_order_factor`, `beam_first_order_annihilated`, `beam_sq_integrable`, `beam_norm_sq_integral`, `beam_growth_on_log_scale`, `log_scale_exponential`, `gaussian_completing_square`, `annulus_phase_bound`, `ratio_tendsto_atTop`, `no_uniform_estimate_of_ratio`, `localized_estimate_fails`, `parameter_choice`, `localized_hadamard_beam` | `S004CharacteristicCone` |

The Phase XXXIII Lean theorems are **ingredients** of the paper proof:

* the algebra of the characteristic roots;
* the exact 1+1 beam and its closed-form `L²` growth;
* the phase inequalities;
* the exponent bookkeeping that turns PDE bounds into the failure of an estimate.

Sobolev spaces, the variable-coefficient transport hierarchy and the `d`-dimensional norm
estimates are **not** formalized. No Lean theorem here is a formal proof of the PDE ill-posedness
theorem, which is ANALYTIC.

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
6. **Phase XXXIII, failed candidate construction:** an FFT-differentiated amplitude recursion on a
   periodic grid gave transport inconsistency `6×10⁴` and hierarchy residuals `10⁴¹`. The cause was
   Gibbs errors from non-periodic phase-derivative factors. It was replaced by exact Taylor-jet
   algebra.
7. **Phase XXXIII, repaired check:** the hierarchy lost precision to cancellation in unscaled jets
   (`0.15` at level `N⁰`). Scaled jets `Y = y/0.25` restore `~10⁻⁹`.
8. **Phase XXXIII, repaired check:** the class-A co-rotating check with the explicit weight ran out
   of memory. It now uses the exact cubic Taylor polynomial of `F` at `ρ₀`. This is exact, because
   only `F…F‴` enter the linearization. It runs in 4 s.
9. **Phase XXXIII, corrected acceptance criteria:**
   * "rounding negligible" was false at large `N`: the analytically zero `N⁴` term has float64 size
     `~10⁻¹⁶N⁴`. It is now excluded with justification and reported separately.
   * A fixed `2%` growth tolerance was replaced by the theoretically justified `|A_N(T_N) − e| ≤ T_N`.

## 10. Commit, CI, PR

See the pull request for the exact head SHA and hosted CI results. Local verification commands:

```bash
cd lean && lake build && lake env lean S004RegularAxiomAudit.lean > /tmp/ax.log && python3 check_s004_regular_axioms.py /tmp/ax.log
```

```bash
python derivations/s004_closure_verify.py --output /tmp/s004.json && python -m unittest -v derivations.test_s004_closure
```

```bash
python derivations/s004_illposedness_verify.py --output /tmp/s004-illposed.json && python -m unittest -v derivations.test_s004_illposedness
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
| No `sorry` or unreviewed axioms; dependency and hypothesis report; pinned build | DONE locally (136 enumerated roots). Hosted CI: see PR. |
| Definitive scoped no-go or identified unresolved assumptions | DONE: scoped no-go. Unresolved assumptions listed in §12. |
| Reconcile with R16 and avoid duplication | DONE: reuses `R18.lowerPole`, `R18.upperPole`, `upper_pole_on_shell`, `R16PU.pu_mode_equations`, `R16PU.pu_single_mode_values`, `R16Class.bddBelow_forces_v_zero` |

**Recommendation (updated, Phase XXXIII).** The obstruction question of #53 is answered by a
scoped no-go. The former external ingredient, the Lax–Mizohata step for inhomogeneous backgrounds,
is now replaced by a self-contained proof (ANALYTIC, §13). Its algebraic and 1D ingredients are
LEAN, and it is NUMERIC end to end.

The remaining non-Lean ingredients are:

* the Sobolev/transport part of that proof, which is ANALYTIC and not formalized;
* the infinite-dimensional H2 count.

**Scientific closure recommendation:** #53 can be closed as answered, once the owner has reviewed
the Phase XXXIII proof and hosted CI is green on the final head. Closing it is the owner's decision.
No issue is closed by this branch.

### #16

| Criterion | Original S004 | Spatial member (replacement) |
|---|---|---|
| DOF counted explicitly | DONE: 4 (regular), 0 (nodal sector) | DONE: 2 |
| Ghost / extra-mode conditions derived | DONE: ghost ⟺ `κF ≠ 0`; stratified | none |
| Bounded physical Hamiltonian or equivalent constrained formulation | REFUTED in the scoped classes | bounded below if `−V ≥ −c` (ANALYTIC); **not equivalent** to S004 |
| Linear dispersion stable in the intended region | REFUTED: elliptic at high `k` in the admissible region | single positive branch (LEAN); no finite-`k` minimum (LEAN) |
| A localized solution passes spectral/orbital stability | Excluded in every framework that needs a Sobolev `C₀` linearization or a `C¹` flow, around every state with a nonzero core (ANALYTIC: Phase XXXIII Theorems 1, 2, 4). Orbital stability with merely continuous flows: OPEN. | OPEN: no confined state constructed; nonlinear well-posedness with nodes OPEN |
| Stable parameter domain documented | none with `κ > 0` | `κ = γ = 0`, `θ ≥ 0` (outside source admissibility) |

#16 remains **OPEN**. Its completion criteria are not met. The spatial member gives up:

* the time-curvature term and the covariant `Θ₀`;
* two of four degrees of freedom and the second branch;
* the finite-`k` static frequency minimum of #18.

There is no controlled correspondence with S004.

## 12. Remaining obstructions after Phase XXXIII

**Resolved in Phase XXXIII.** The earlier "missing theorem", variable-coefficient ill-posedness of
the S004 linearization near every regular point of every smooth background, is now proved by a
self-contained localized Gaussian-beam argument (§13). The proof does not cite Lax–Mizohata. The
localization step that the previous report identified as the precise missing lemma is supplied by
three ingredients:

* an **exact** complex phase;
* explicit Gaussian concentration and annulus bounds;
* a finite WKB hierarchy with an explicit residual.

**What remains open** (none of it is needed for the scoped no-go):

1. **Formalization.** The Sobolev and transport-hierarchy part of the proof is ANALYTIC, not LEAN.
2. **Nonlinear instability without differentiability.** Norm inflation, Hölder discontinuity of a
   merely continuous flow, blow-up: OPEN and not claimed. Orbital stability with merely continuous
   flows is not refuted.
3. **Homogeneous problem from a single initial time only.** For time-dependent backgrounds, if
   solvability is assumed only from `t₀` (not from every initial time), the homogeneous no-go is
   not covered. This is a non-standard notion of well-posedness.
4. **Gevrey classes.** OPEN.
5. **Recovery side.** The open structural question is **whether any completion outside the
   `(κ, θ, γ)` family can change the type of the principal symbol while keeping S004's
   predictions**. The candidates are:
   * a dynamical foliation `n^μ`;
   * a dynamical metric beyond the analyzed candidate;
   * a nonlocal curvature functional.

   Inside the family this is impossible (§7).

## 13. Phase XXXIII: arbitrary-background Sobolev ill-posedness

The full proof is in [`VARIABLE_COEFFICIENT_ILLPOSEDNESS.md`](VARIABLE_COEFFICIENT_ILLPOSEDNESS.md),
with section numbers referenced below.

### 13.1 Final theorem

**Theorem (Corollary 1.1 of Theorem 1).** Hypotheses:

* `κ > 0, θ > 0, γ² < 4κθ`, `ε ≠ 0`, `α ∈ ℝ`, `V ∈ C^∞`;
* `ψ₀` any smooth complex background on an open set of flat `ℝ^{1+d}`, `d ≥ 1`, with the inertial
  foliation. It need not solve the field equations; nodes elsewhere are allowed;
* `(t₀, x₀)` a point with `ψ₀(t₀, x₀) ≠ 0`.

Let `L_{ψ₀}` be the linearization of the real Euler–Lagrange system. Then for every `s, σ ∈ ℝ`, every
`T > 0`, every `ρ > 0` (small enough to stay in the regular region) and every `C`, the local Cauchy
estimate fails:

```
‖v(t₀+T′)‖_{H^{−σ}} ≤ C ( Σ_{j≤3} ‖∂ₜ^j v(t₀)‖_{H^s} + sup_{[t₀, t₀+T′]} ‖L_{ψ₀} v‖_{H^s} )
for all real v ∈ C^∞ supported in [t₀, t₀+T] × B_ρ(x₀) and all T′ ∈ (0, T].
```

Quantitatively, for every `M` there are real `v_N` supported in `[t₀, t₀ + T_N] × B_ρ(x₀)` with
`T_N = K log N / N` whose left-hand side exceeds `N^M` times the right-hand side.

Consequences, all ANALYTIC:

* no bounded solution operator with sources that reproduces smooth solutions (Corollary 2);
* no `C₀`-semigroup for the linearization about static or co-rotating states (Theorem 2(a));
* no evolution system satisfying the backward equation on smooth data (Theorem 2(b));
* class A: exact growing Fourier modes at fixed time (Theorem 3);
* no forced nonlinear solution map Fréchet-differentiable at a smooth solution with a regular point
  (Theorem 4).

### 13.2 Proof structure

All steps are proved in the proof document. No existence theorem, energy estimate, microlocal
result or Lax–Mizohata citation is used.

1. **Principal symbol** (Lemma 1, all `d`).
   * The Euler–Lagrange operator is
     `E = □[F(κ□ψ + γΔψ/2)] + Δ[F(θΔψ + γ□ψ/2)] + F′Hψ − □ψ − V′ψ`.
   * Its linearization is `2F(ρ₀)P(∂)I₂ + Σ_{|α|≤3}A_α∂^α`.
   * The coefficient of `∂^α` has total background derivative order `≤ 4−|α|` and involves at most
     `F‴` and `V″`.
   * After division by `2F(ρ₀) > 0` the principal part `P(∂)I₂` is constant and independent of the
     background.
2. **Admissibility and roots** (Lemma 2).
   * Admissible ⟺ elliptic.
   * `P = κ(λ²+μZ)(λ²+μ̄Z)`.
   * Four distinct simple roots `±w|ξ|, ±w̄|ξ|`, with `Re w > 0` and
     `∂_λP = 2iw|ξ|³√(4κθ−γ²) ≠ 0`.
3. **Growing modes.**
   * Frozen part: `λ = w|ξ|` exactly.
   * Constant coefficients with lower-order terms: Rouché (Theorem 3).
   * Variable coefficients: localized beams growing like `e^{N t Re w}` at frequency `N`.
4. **Exact Gaussian-beam phase** (Lemma 3). `φ = iê·x − |x|²/2 + w t S(x) + μt²/2`, with
   `S = (1 + 2iê·x − |x|²)^{1/2}`.
   * `∇ₓφ·∇ₓφ = −(S+wt)²` and `∂ₜφ = w(S+wt)`, so `P(∇φ) ≡ 0` in every dimension.
   * The `t`-series terminates because the initial complex phase is spherical about the complex
     point `iê` and `H` is isotropic and homogeneous of degree one.
5. **Lower-order terms** (Lemmas 4–6).
   * Conjugation: `e^{−Nφ}L̃e^{Nφ} = Σ_{k=1}^4 N^{4−k}Q_k`, with
     `Q₁ = V·∇ + 𝒞` and `V₀ = 2i√(4κθ−γ²)w(S+wt)³ ≠ 0`.
   * The arbitrary variable coefficients `B_α = A_α/(2F(ρ₀))` enter only `𝒞` and `Q_k`, `k ≥ 2`.
   * Transport equations are solved by finite `t`-recursions, with Taylor remainders `t^{K′}r`.
6. **Residual** (Lemma 6). `e^{−Nφ}L̃(e^{Nφ}A_N) = R_N` with `|∂_x^βR_N| ≤ C(N³t^{K′} + N^{2−J})`.
   The cutoff commutator is smaller than the beam by `e^{−Nρ²/8}N^{C₁ρK}`.
7. **Norms** (Lemma 7, §5.8).
   * Data `≤ CN^{3+s−d/4}`.
   * Source `≤ CN^{s−d/4}N^{K Re w}(N³T_N^{K′} + N^{2−J})` plus super-polynomially small terms.
   * `‖v_N(T_N)‖_{H^{−σ}} ≥ cN^{K Re w−σ−d/4}`, by duality and the Gaussian mass bounds
     `G_N ≍ N^{−d/2}e^{2N Re φ(t,0)}`.
8. **Contradiction** (§5.9). With `K > (3+s+σ+M)/Re w`, `K′ > 3+s+σ+M` and `J > 2+s+σ+M`, the
   ratio exceeds `N^M`.

### 13.3 Assumptions and scope

* **In scope:**
  * the corrected S004 density with the corrected weight `F`;
  * any smooth `V`;
  * admissible coefficients;
  * a flat fixed background metric with the inertial foliation;
  * two real components;
  * `C^∞` backgrounds (finitely many derivatives suffice for fixed `(s, σ)`).
* **Not examined:**
  * curved or dynamical metrics;
  * a dynamical foliation;
  * nonlocal or auxiliary-field completions;
  * non-admissible parameters;
  * analytic classes, where Cauchy–Kovalevskaya gives local solutions;
  * Gevrey classes.

### 13.4 Verification (actual results)

**Symbolic checks** (`s004_illposedness_verify.py`):

* I1: linearization structure in 1+1 and 1+2, generic background: order 4, total background order
  `≤ 4−|α|`, `F‴`, `V″`, real coefficients.
* I2: factorization and simple root.
* I3: `P(∂)e^{Nφ} = 0` exactly in 1+1, and the closed-form `L²` norm.
* I4: the exact beam phase for `d = 1, 2, 3`.
* I5: the conjugation identity for a generic phase and amplitude in 1+1 and 1+2.
* I8: exponent bookkeeping.

**I6 (class A).** Co-rotating, off-shell background. Co-rotating coefficients are constant, and
`|λ/k − w| = 0.70/k` for `k = 10…10⁴`.

**I7 (end to end).** On B1 (class C), B2 (class B) and B3 (sampled), using the exact
Euler–Lagrange linearization with the principal part checked symbolically on each, Taylor-jet
amplitudes and the true residual with exact coefficients:

* coefficient jets `≤ 2×10⁻¹⁵`;
* transport consistency `≤ 1.2×10⁻¹³`;
* hierarchy jets `≤ 6.2×10⁻⁶`;
* growth within `T_N` of the Gaussian prediction;
* residual decaying;
* contradiction ratio increasing on every background:
  * B1: `146 → 363 → 1164 → 3774`;
  * B2: `0.004 → 1.2 → 386 → 2738`;
  * B3: `25 → 460 → 1688 → 38720`;

  for `N = 400 → 3200`.

Refinement on B1:

* the residual slope is `+0.09`, `−0.80`, `−1.58` for `J = 2, 3, 4`, approaching `2 − J` from above;
* jet degree `64 → 48` leaves the residual unchanged;
* quadrature `4001 → 8001` points changes the residual by `5×10⁻⁶` relative.

Rounding is separated from analytic error:

* the analytically zero `N⁴` term (float64 `~10⁻¹⁶N⁴`) is excluded and reported separately;
* an upper estimate of the rounding floor of the retained terms is reported per row.

Negative controls (wrong root, decaying root, no transport, hyperbolic principal part) all fail to
produce a contradiction: ratios fall to `3×10⁻¹⁴`, `3×10⁻²⁷`, `1×10⁻⁹` and `2×10⁻¹¹` at `N = 3200`.
Exact per-background numbers are in the receipt.

**Lean.**

* `lake build` passes.
* 136 enumerated audit roots, all on `propext`, `Classical.choice`, `Quot.sound`.
* The checker fails on each of: a missing root, an extra or duplicated report, `sorryAx`, a Lean
  error.

### 13.5 Phase XXXIII acceptance matrix

The criteria are quoted verbatim from the Phase XXXIII instructions.

| Criterion | Verdict | Evidence / missing obligation |
|---|---|---|
| **A. Full closure:** "A self-contained proof establishes variable-coefficient local Sobolev ill-posedness for the stated arbitrary smooth regular-background class." | **MET — ANALYTICALLY PROVED** | Theorem 1 and Corollary 1.1. Arbitrary smooth backgrounds; any regular point; any `s, σ, T, ρ, C`; no external theorem. Supported by LEAN ingredients (§8) and NUMERIC end-to-end checks (I7). Not LEAN VERIFIED as a PDE theorem. |
| **B. Strong restricted closure:** "A complete proof establishes the result for a clearly defined nonconstant-background class, with the remaining generalization reduced to one explicit lemma." | **MET (subsumed by A)** | Class B (nodeless nonconstant) is a special case of Theorem 1. No generalization lemma remains for the stated class. |
| **C. Precise obstruction:** "The proposed theorem or proof strategy is refuted or blocked by a specific mathematical issue, with counterexamples and the strongest valid result retained." | **NOT TRIGGERED** | The strategy was not refuted. The failed numerical candidates (FFT harness, unscaled jets, class-A OOM) were harness defects, not mathematical obstructions; they are preserved (§9). |

Supporting labels:

| Item | Label |
|---|---|
| Exact principal symbol and coefficient structure | ANALYTICALLY PROVED + symbolic (I1, C1, I7) |
| Admissibility ⟺ ellipticity; simple growing roots | LEAN VERIFIED |
| Exact beam phase, all `d` | ANALYTICALLY PROVED + symbolic (I4); LEAN VERIFIED for `d = 1` |
| Residual and norm estimates; contradiction | ANALYTICALLY PROVED; exponent step LEAN VERIFIED |
| End-to-end on three backgrounds, refinement, negative controls | NUMERICALLY VERIFIED |
| Nonlinear: no differentiable forced map / no `C¹` flows with (S1′) | ANALYTICALLY PROVED |
| Norm inflation, blow-up, orbital instability without differentiability | OPEN (not claimed) |

### 13.6 Consequences for #53 and #16

The following distinctions are kept separate:

1. **Ill-posedness of the original corrected S004 evolution.** ANALYTICALLY PROVED near every regular
   point of every smooth background, on a flat fixed background. The original action cannot provide
   a Sobolev well-posed (finite-loss), `C¹`, or `C₀`-semigroup evolutionary foundation near any
   nonzero field configuration.
2. **Scoped ghost obstruction and no-go** (§§2–7): unchanged, `S004_SCOPED_NO_GO_PROVED`. The
   ill-posedness theorem strengthens it. The two are logically independent: one is about the
   Hamiltonian's unboundedness, the other about the Cauchy problem.
3. **Potential ghost-free replacement** (spatial member `κ = γ = 0`): it is not affected by Theorem 1
   (its principal part is spatial-elliptic and first order in time). It is **not equivalent** to
   S004, and its well-posedness and stability are not established here.
4. **Whether a replacement supports stable confined states**: OPEN. No confined state of the
   replacement has been constructed or shown stable.

**#53:** the obstruction question is answered. The scoped no-go is now unconditional on the
Lax–Mizohata citation. Recommendation: close after owner review and green hosted CI on the final
head (owner's decision).

**#16:** remains **OPEN**. Its healthy-completion criteria are not met. Required for closure:

* a ghost-free action physically equivalent to S004 in a controlled limit;
* well-posedness of its Cauchy problem;
* existence and stability of a confined state.

This result does **not** show that Wave Confinement Theory as a whole is refuted. It shows that the
analyzed corrected S004 action is obstructed.

### 13.7 Remaining mathematical obligations

See §12: formalization, nonlinear instability without differentiability, the single-initial-time
homogeneous problem, Gevrey classes, and completions outside the family.
