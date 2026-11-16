import Mathlib

/-!
# S004 Phase XXV: characteristic cone, spacetime ellipticity and Hadamard roots

**Source.** `WCT_CORRECTED_VARIATIONAL_CLOSURE_DRAFT.md` §2 fixes the corrected S004 density

  `L = ∇ψ̄·∇ψ − V(|ψ|²) + F(|ψ|²) H`,  `H = κ|□ψ|² + θ|Δψ|² + γ Re(□ψ · conj Δψ)`,

signature `(−,+,…,+)`, `□ = −∂ₜ² + Δ`, and the source admissibility condition
`κ > 0, θ > 0, γ² < 4κθ` (pointwise positive definiteness of `H`).

**Independent symbolic input** (`derivations/s004_closure_verify.py`, check family `C1`):
brute-force linearization of the `sympy.euler_equations` output in 1+1 and 1+2 dimensions
shows that the fourth-order part of the linearized S004 operator about an ARBITRARY smooth
background `ψ₀` is exactly `2 F(|ψ₀|²) (κ □² + γ □Δ + θ Δ²) · I₂`. Its principal symbol at a
real covector `(τ, ξ)` is therefore `2 F(ρ₀) · symbolForm κ θ γ (τ² − |ξ|²) |ξ|²`, where
`symbolForm κ θ γ X Y = κX² − γXY + θY²`. Nothing else contributes at fourth order.

This module proves the exact real and complex algebra of that symbol:

* `symbolForm_pos_definite_iff`: the symbol form is positive definite on `ℝ² \ 0`
  **iff** the source admissibility condition holds (it is the curvature density with `P ↦ −P`).
* `admissible_elliptic_covector`: hence, under admissibility, at every regular point
  (`F > 0`, i.e. `ψ₀ ≠ 0`) the principal symbol is strictly positive at every real nonzero
  covector `(τ, ξ)`: the S004 operator is **elliptic in spacetime**. It has no real
  characteristic direction at all.
* `real_characteristic_iff`, `two_positive_roots_iff`: real characteristics exist iff the reduced
  quadratic `κs² − (2κ+γ)s + (κ+θ+γ)` has a root `s ≥ 0`; strict hyperbolicity (two distinct
  positive roots) holds iff `γ² > 4κθ`, `2κ+γ > 0`, `κ+θ+γ > 0`. Positivity of the curvature
  form and strict hyperbolicity are therefore mutually exclusive.
* `psd_with_real_characteristic_is_boundary`, `boundary_perfect_square`: a positive
  semidefinite curvature form admits real characteristics only on the degenerate boundary
  `γ² = 4κθ`, where the form is the perfect square `κ (X − (γ/2κ) Y)²`.
* `hadamardRatio_root`, `hadamard_sqrt_re_ne_zero`, `frozen_symbol_hadamard_root`: under
  admissibility, the frozen principal part has the exact plane-wave roots `λ = |ξ|·w`,
  `w² = μ`, with `Re w ≠ 0`. One sign gives growth `exp(c_uv |ξ| t)`,
  `c_uv² = (‖μ‖ + Re μ)/2 > 0`, independent of the background value `F(ρ₀) > 0`.
* `hadamard_constant_unbounded_as_kappa_to_zero`: on the `γ = 0` admissible ray the Hadamard
  constant is unbounded as `κ → 0⁺`.
* `exp_beats_every_polynomial_weight`: `exp(b x)/(1+x)^N → ∞` for `b > 0`, the quantitative
  reason no finite loss of Sobolev derivatives can absorb Hadamard growth.

**Scope.** These are exact finite-dimensional algebraic statements about the principal symbol
and about explicit frozen-coefficient plane-wave roots. The passage from a non-real
characteristic root at one point to failure of Sobolev well-posedness of the
variable-coefficient and nonlinear Cauchy problem is the classical Lax–Mizohata theorem; it is
NOT formalized here and is recorded as an analytical (cited) step in
`docs/s004_closure/CHARACTERISTIC_CONE_THEOREM.md`.
-/

namespace ReleaseProofs.S004CharacteristicCone

open Filter Topology

/-- `κX² − γXY + θY²`: principal-symbol form of the S004 curvature operator, with
`X = τ² − |ξ|²` and `Y = |ξ|²`. -/
def symbolForm (κ θ γ X Y : ℝ) : ℝ := κ * X ^ 2 - γ * X * Y + θ * Y ^ 2

/-- Per-real-component S004 curvature density `κQ² + θP² + γQP` with `Q = □x`, `P = Δx`. -/
def curvatureDensity (κ θ γ Q P : ℝ) : ℝ := κ * Q ^ 2 + θ * P ^ 2 + γ * Q * P

/-- Frozen principal symbol per real component (the common factor `2` is dropped) at the
real covector `(τ, ξ)`, `Y = |ξ|²`, background weight `F = F(|ψ₀|²)`. -/
def principalSymbol (F κ θ γ τ Y : ℝ) : ℝ := F * symbolForm κ θ γ (τ ^ 2 - Y) Y

/-- Reduced characteristic quadratic in `s = τ²/|ξ|²`. -/
def reducedQuadratic (κ θ γ s : ℝ) : ℝ := κ * s ^ 2 - (2 * κ + γ) * s + (κ + θ + γ)

/-- Source admissibility: pointwise positive definiteness of the curvature form. -/
def Admissible (κ θ γ : ℝ) : Prop := 0 < κ ∧ 0 < θ ∧ γ ^ 2 < 4 * κ * θ

/-- The principal-symbol form is the curvature density evaluated at `(Q, P) = (X, −Y)`. -/
theorem symbolForm_eq_curvatureDensity (κ θ γ X Y : ℝ) :
    symbolForm κ θ γ X Y = curvatureDensity κ θ γ X (-Y) := by
  unfold symbolForm curvatureDensity
  ring

theorem symbolForm_completed_square (κ θ γ X Y : ℝ) :
    4 * κ * symbolForm κ θ γ X Y =
      (2 * κ * X - γ * Y) ^ 2 + (4 * κ * θ - γ ^ 2) * Y ^ 2 := by
  unfold symbolForm
  ring

/-- Under admissibility the symbol form is strictly positive away from the origin. -/
theorem symbolForm_pos_of_admissible {κ θ γ : ℝ} (h : Admissible κ θ γ) {X Y : ℝ}
    (hXY : X ≠ 0 ∨ Y ≠ 0) : 0 < symbolForm κ θ γ X Y := by
  obtain ⟨hk, -, hg⟩ := h
  have hgap : 0 < 4 * κ * θ - γ ^ 2 := by linarith
  have key := symbolForm_completed_square κ θ γ X Y
  have h4 : 0 < 4 * κ * symbolForm κ θ γ X Y := by
    rcases eq_or_ne Y 0 with hY | hY
    · have hX : X ≠ 0 := by
        rcases hXY with hX | hY'
        · exact hX
        · exact absurd hY hY'
      have hkX : 2 * κ * X ≠ 0 := mul_ne_zero (mul_ne_zero two_ne_zero hk.ne') hX
      have hsq : 0 < (2 * κ * X) ^ 2 := sq_pos_iff.mpr hkX
      rw [key, hY]
      simpa using hsq
    · have h1 : 0 < (4 * κ * θ - γ ^ 2) * Y ^ 2 := mul_pos hgap (sq_pos_iff.mpr hY)
      rw [key]
      nlinarith [sq_nonneg (2 * κ * X - γ * Y)]
  have h4k : 0 < 4 * κ := by linarith
  by_contra hle
  push_neg at hle
  nlinarith

/-- **Exact positivity criterion.** The S004 principal-symbol form is positive definite iff the
source admissibility condition `κ > 0, θ > 0, γ² < 4κθ` holds. -/
theorem symbolForm_pos_definite_iff (κ θ γ : ℝ) :
    (∀ X Y : ℝ, (X ≠ 0 ∨ Y ≠ 0) → 0 < symbolForm κ θ γ X Y) ↔ Admissible κ θ γ := by
  constructor
  · intro h
    have hk : 0 < κ := by simpa [symbolForm] using h 1 0 (Or.inl one_ne_zero)
    have ht : 0 < θ := by simpa [symbolForm] using h 0 1 (Or.inr one_ne_zero)
    have hg := h γ (2 * κ) (Or.inr (by positivity))
    have hval : symbolForm κ θ γ γ (2 * κ) = κ * (4 * κ * θ - γ ^ 2) := by
      unfold symbolForm
      ring
    rw [hval] at hg
    refine ⟨hk, ht, ?_⟩
    by_contra hle
    push_neg at hle
    nlinarith
  · intro h X Y hXY
    exact symbolForm_pos_of_admissible h hXY

/-- **Spacetime ellipticity at every regular point.** Under admissibility and `F(ρ₀) > 0`
(i.e. `ψ₀ ≠ 0`), the frozen principal symbol is strictly positive at every real pair
`(τ, Y) ≠ (0, 0)`; for a covector take `Y = |ξ|²`. -/
theorem admissible_symbol_elliptic {F κ θ γ : ℝ} (hF : 0 < F) (h : Admissible κ θ γ)
    {τ Y : ℝ} (hne : τ ≠ 0 ∨ Y ≠ 0) :
    0 < principalSymbol F κ θ γ τ Y := by
  unfold principalSymbol
  refine mul_pos hF (symbolForm_pos_of_admissible h ?_)
  rcases eq_or_ne Y 0 with hY0 | hY0
  · left
    rcases hne with hτ | hY'
    · rw [hY0, sub_zero]
      exact pow_ne_zero 2 hτ
    · exact absurd hY0 hY'
  · right
    exact hY0

/-- Covector form of ellipticity on any real inner-product space of spatial dimensions. -/
theorem admissible_elliptic_covector {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    {F κ θ γ : ℝ} (hF : 0 < F) (h : Admissible κ θ γ) (τ : ℝ) (ξ : E)
    (hne : τ ≠ 0 ∨ ξ ≠ 0) :
    0 < principalSymbol F κ θ γ τ (‖ξ‖ ^ 2) := by
  apply admissible_symbol_elliptic hF h
  rcases hne with hτ | hξ
  · exact Or.inl hτ
  · exact Or.inr (pow_ne_zero 2 (norm_ne_zero_iff.mpr hξ))

/-- No real characteristic covector exists at a regular point of an admissible S004 theory. -/
theorem admissible_no_real_characteristic {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] {F κ θ γ : ℝ} (hF : 0 < F) (h : Admissible κ θ γ)
    (τ : ℝ) (ξ : E) (hne : τ ≠ 0 ∨ ξ ≠ 0) :
    principalSymbol F κ θ γ τ (‖ξ‖ ^ 2) ≠ 0 :=
  (admissible_elliptic_covector hF h τ ξ hne).ne'

/-- Any real characteristic at a regular point excludes admissibility. -/
theorem real_characteristic_not_admissible {F κ θ γ τ Y : ℝ} (hF : F ≠ 0) (hY : 0 < Y)
    (hchar : principalSymbol F κ θ γ τ Y = 0) : ¬ Admissible κ θ γ := by
  intro h
  unfold principalSymbol at hchar
  have hs : symbolForm κ θ γ (τ ^ 2 - Y) Y = 0 := by
    rcases mul_eq_zero.mp hchar with h0 | h0
    · exact absurd h0 hF
    · exact h0
  have hpos := symbolForm_pos_of_admissible h (X := τ ^ 2 - Y) (Or.inr hY.ne')
  linarith

/-- Homogeneous factorization of the symbol in `σ = τ²`. -/
theorem symbol_homogeneous (κ θ γ σ Y : ℝ) :
    symbolForm κ θ γ (σ - Y) Y = κ * σ ^ 2 - (2 * κ + γ) * σ * Y + (κ + θ + γ) * Y ^ 2 := by
  unfold symbolForm
  ring

theorem symbol_reduced (κ θ γ σ Y : ℝ) (hY : Y ≠ 0) :
    symbolForm κ θ γ (σ - Y) Y = Y ^ 2 * reducedQuadratic κ θ γ (σ / Y) := by
  unfold symbolForm reducedQuadratic
  field_simp
  ring

/-- Discriminant of the reduced quadratic equals the curvature-form discriminant. -/
theorem reduced_discriminant (κ θ γ : ℝ) :
    (2 * κ + γ) ^ 2 - 4 * κ * (κ + θ + γ) = γ ^ 2 - 4 * κ * θ := by
  ring

theorem reduced_completed_square (κ θ γ s : ℝ) :
    (2 * κ * s - (2 * κ + γ)) ^ 2 =
      4 * κ * reducedQuadratic κ θ γ s + (γ ^ 2 - 4 * κ * θ) := by
  unfold reducedQuadratic
  ring

/-- A real root of the reduced quadratic forces a nonnegative curvature discriminant. -/
theorem reduced_root_discriminant_nonneg {κ θ γ s : ℝ}
    (hs : reducedQuadratic κ θ γ s = 0) : 0 ≤ γ ^ 2 - 4 * κ * θ := by
  have key := reduced_completed_square κ θ γ s
  rw [hs, mul_zero, zero_add] at key
  rw [← key]
  exact sq_nonneg _

/-- Real characteristics at a regular point are exactly nonnegative roots of the reduced
quadratic. -/
theorem real_characteristic_iff (F κ θ γ Y : ℝ) (hF : F ≠ 0) (hY : 0 < Y) :
    (∃ τ : ℝ, principalSymbol F κ θ γ τ Y = 0) ↔
      ∃ s : ℝ, 0 ≤ s ∧ reducedQuadratic κ θ γ s = 0 := by
  constructor
  · rintro ⟨τ, hτ⟩
    refine ⟨τ ^ 2 / Y, div_nonneg (sq_nonneg τ) hY.le, ?_⟩
    unfold principalSymbol at hτ
    rw [symbol_reduced κ θ γ (τ ^ 2) Y hY.ne'] at hτ
    rcases mul_eq_zero.mp hτ with h0 | h0
    · exact absurd h0 hF
    · rcases mul_eq_zero.mp h0 with h1 | h1
      · exact absurd h1 (pow_ne_zero 2 hY.ne')
      · exact h1
  · rintro ⟨s, hs0, hs⟩
    refine ⟨Real.sqrt (s * Y), ?_⟩
    unfold principalSymbol
    have hsq : Real.sqrt (s * Y) ^ 2 = s * Y := Real.sq_sqrt (mul_nonneg hs0 hY.le)
    rw [symbol_reduced κ θ γ _ Y hY.ne', hsq]
    have : s * Y / Y = s := by field_simp
    rw [this, hs]
    ring

/-- **Strict hyperbolicity criterion** (`κ > 0`): two distinct positive characteristic ratios
exist iff `γ² > 4κθ`, `2κ + γ > 0` and `κ + θ + γ > 0`. -/
theorem two_positive_roots_iff {κ θ γ : ℝ} (hk : 0 < κ) :
    (∃ s₁ s₂ : ℝ, 0 < s₁ ∧ s₁ < s₂ ∧ reducedQuadratic κ θ γ s₁ = 0 ∧
        reducedQuadratic κ θ γ s₂ = 0) ↔
      (4 * κ * θ < γ ^ 2 ∧ 0 < 2 * κ + γ ∧ 0 < κ + θ + γ) := by
  constructor
  · rintro ⟨s₁, s₂, h1pos, hlt, h1, h2⟩
    unfold reducedQuadratic at h1 h2
    have hfac : (s₁ - s₂) * (κ * (s₁ + s₂) - (2 * κ + γ)) = 0 := by
      linear_combination h1 - h2
    have hA : 2 * κ + γ = κ * (s₁ + s₂) := by
      rcases mul_eq_zero.mp hfac with h0 | h0
      · exact absurd (sub_eq_zero.mp h0) hlt.ne
      · linarith
    have hC : κ + θ + γ = κ * s₁ * s₂ := by
      linear_combination h1 + s₁ * hA
    have h2pos : 0 < s₂ := lt_trans h1pos hlt
    refine ⟨?_, ?_, ?_⟩
    · have hd : γ ^ 2 - 4 * κ * θ = κ ^ 2 * (s₁ - s₂) ^ 2 := by
        have := reduced_discriminant κ θ γ
        rw [hA, hC] at this
        linear_combination -this
      have : 0 < κ ^ 2 * (s₁ - s₂) ^ 2 :=
        mul_pos (pow_pos hk 2) (sq_pos_iff.mpr (sub_ne_zero.mpr hlt.ne))
      linarith
    · rw [hA]
      positivity
    · rw [hC]
      positivity
  · rintro ⟨hD, hA, hC⟩
    set D := γ ^ 2 - 4 * κ * θ with hDdef
    have hDpos : 0 < D := by rw [hDdef]; linarith
    set r := Real.sqrt D with hr
    have hr2 : r ^ 2 = D := Real.sq_sqrt hDpos.le
    have hrpos : 0 < r := Real.sqrt_pos.mpr hDpos
    have hrA : r < 2 * κ + γ := by
      have h4 : D < (2 * κ + γ) ^ 2 := by
        have := reduced_discriminant κ θ γ
        nlinarith [mul_pos hk hC]
      nlinarith
    refine ⟨((2 * κ + γ) - r) / (2 * κ), ((2 * κ + γ) + r) / (2 * κ),
      div_pos (by linarith) (by linarith), ?_, ?_, ?_⟩
    · exact div_lt_div_of_pos_right (by linarith) (by linarith)
    · unfold reducedQuadratic
      field_simp
      have := reduced_discriminant κ θ γ
      nlinarith [hr2]
    · unfold reducedQuadratic
      field_simp
      have := reduced_discriminant κ θ γ
      nlinarith [hr2]

/-- Admissibility excludes strict hyperbolicity: there are never two distinct positive
characteristic ratios inside the source region. -/
theorem admissible_not_strictly_hyperbolic {κ θ γ : ℝ} (h : Admissible κ θ γ) :
    ¬ ∃ s₁ s₂ : ℝ, 0 < s₁ ∧ s₁ < s₂ ∧ reducedQuadratic κ θ γ s₁ = 0 ∧
        reducedQuadratic κ θ γ s₂ = 0 := by
  intro hex
  have := ((two_positive_roots_iff h.1).mp hex).1
  linarith [h.2.2]

/-- A positive semidefinite curvature discriminant with any real characteristic ratio lies on
the degenerate boundary `γ² = 4κθ`. -/
theorem psd_with_real_characteristic_is_boundary {κ θ γ s : ℝ}
    (hpsd : γ ^ 2 ≤ 4 * κ * θ) (hs : reducedQuadratic κ θ γ s = 0) :
    γ ^ 2 = 4 * κ * θ := by
  have := reduced_root_discriminant_nonneg hs
  linarith

/-- On the boundary the symbol is a perfect square: one double wave cone. -/
theorem boundary_perfect_square {κ θ γ : ℝ} (hk : κ ≠ 0) (hb : γ ^ 2 = 4 * κ * θ)
    (X Y : ℝ) :
    symbolForm κ θ γ X Y = κ * (X - γ / (2 * κ) * Y) ^ 2 := by
  unfold symbolForm
  have hθ : θ = γ ^ 2 / (4 * κ) := by field_simp; linarith
  rw [hθ]
  field_simp
  ring

/-! ## Exact Hadamard plane-wave roots of the frozen principal part -/

/-- Characteristic ratio `μ = λ²/|ξ|²` of the frozen principal part
`κ(λ²+Y)² + γ(λ²+Y)Y + θY²` (from `□ ↦ −(λ²+Y)`, `Δ ↦ −Y` on `exp(λt + iξ·x)`). -/
noncomputable def hadamardRatio (κ θ γ : ℝ) : ℂ :=
  ⟨-(2 * κ + γ) / (2 * κ), Real.sqrt (4 * κ * θ - γ ^ 2) / (2 * κ)⟩

/-- `μ` solves `κ(μ+1)² + γ(μ+1) + θ = 0` exactly. -/
theorem hadamardRatio_root {κ θ γ : ℝ} (hk : κ ≠ 0) (hD : 0 ≤ 4 * κ * θ - γ ^ 2) :
    (κ : ℂ) * (hadamardRatio κ θ γ + 1) ^ 2 + (γ : ℂ) * (hadamardRatio κ θ γ + 1) + θ = 0 := by
  have hs : Real.sqrt (4 * κ * θ - γ ^ 2) ^ 2 = 4 * κ * θ - γ ^ 2 := Real.sq_sqrt hD
  unfold hadamardRatio
  generalize Real.sqrt (4 * κ * θ - γ ^ 2) = r at hs ⊢
  apply Complex.ext
  · simp only [pow_two, Complex.add_re, Complex.mul_re, Complex.ofReal_re,
      Complex.ofReal_im, Complex.add_im, Complex.mul_im, Complex.one_re, Complex.one_im,
      Complex.zero_re]
    field_simp
    ring_nf
    linear_combination (-κ) * hs
  · simp only [pow_two, Complex.add_re, Complex.mul_re, Complex.ofReal_re,
      Complex.ofReal_im, Complex.add_im, Complex.mul_im, Complex.one_re, Complex.one_im,
      Complex.zero_im]
    field_simp
    ring

/-- Frozen principal part of the S004 operator on `exp(λ t + i ξ·x)`, `Y = |ξ|²`. -/
def frozenSymbolLambda (κ θ γ : ℝ) (lam : ℂ) (Y : ℝ) : ℂ :=
  (κ : ℂ) * (lam ^ 2 + Y) ^ 2 + (γ : ℂ) * (lam ^ 2 + Y) * Y + (θ : ℂ) * (Y : ℂ) ^ 2

/-- Every `λ` with `λ² = |ξ|² μ` is an exact characteristic root of the frozen principal part. -/
theorem frozen_symbol_hadamard_root {κ θ γ : ℝ} (hk : κ ≠ 0) (hD : 0 ≤ 4 * κ * θ - γ ^ 2)
    (Y : ℝ) (lam : ℂ) (hlam : lam ^ 2 = (Y : ℂ) * hadamardRatio κ θ γ) :
    frozenSymbolLambda κ θ γ lam Y = 0 := by
  have hroot := hadamardRatio_root hk hD
  unfold frozenSymbolLambda
  rw [hlam]
  have : (κ : ℂ) * ((Y : ℂ) * hadamardRatio κ θ γ + Y) ^ 2 +
      (γ : ℂ) * ((Y : ℂ) * hadamardRatio κ θ γ + Y) * Y + (θ : ℂ) * (Y : ℂ) ^ 2 =
      (Y : ℂ) ^ 2 * ((κ : ℂ) * (hadamardRatio κ θ γ + 1) ^ 2 +
        (γ : ℂ) * (hadamardRatio κ θ γ + 1) + θ) := by ring
  rw [this, hroot, mul_zero]

/-- Real part of a complex square root: `(Re w)² = (‖μ‖ + Re μ)/2` when `w² = μ`. -/
theorem sqrt_re_sq {μ w : ℂ} (hw : w ^ 2 = μ) : w.re ^ 2 = (‖μ‖ + μ.re) / 2 := by
  have hnorm : ‖μ‖ = ‖w‖ ^ 2 := by rw [← hw, norm_pow]
  have hre : μ.re = w.re ^ 2 - w.im ^ 2 := by
    rw [← hw]
    simp [pow_two, Complex.mul_re]
  have hsq : ‖w‖ ^ 2 = w.re ^ 2 + w.im ^ 2 := by
    rw [Complex.sq_norm, Complex.normSq_apply]
    ring
  rw [hnorm, hre, hsq]
  ring

/-- A complex square root of a non-real number has nonzero real part. -/
theorem hadamard_sqrt_re_ne_zero {μ w : ℂ} (hw : w ^ 2 = μ) (him : μ.im ≠ 0) : w.re ≠ 0 := by
  have hlt : |μ.re| < ‖μ‖ := Complex.abs_re_lt_norm.mpr him
  have hpos : 0 < (‖μ‖ + μ.re) / 2 := by
    have := neg_abs_le μ.re
    linarith
  rw [← sqrt_re_sq hw] at hpos
  intro h0
  rw [h0] at hpos
  norm_num at hpos

/-- Under admissibility the Hadamard ratio is non-real. -/
theorem hadamardRatio_im_ne_zero {κ θ γ : ℝ} (h : Admissible κ θ γ) :
    (hadamardRatio κ θ γ).im ≠ 0 := by
  obtain ⟨hk, -, hg⟩ := h
  have hD : 0 < 4 * κ * θ - γ ^ 2 := by linarith
  simp only [hadamardRatio]
  exact div_ne_zero (Real.sqrt_pos.mpr hD).ne' (by positivity)

/-- **Exact Hadamard growth.** Under admissibility, for every spatial frequency `n > 0` there is
an exact root `λ` of the frozen principal part (any background weight `F > 0` factors out) with
`Re λ = n · c`, `c > 0` independent of `n`. Hence `exp(λ t + i n x)` grows like `exp(c n t)`. -/
theorem admissible_hadamard_growth {κ θ γ : ℝ} (h : Admissible κ θ γ) :
    ∃ c : ℝ, 0 < c ∧ ∀ n : ℝ, 0 < n → ∃ lam : ℂ,
      frozenSymbolLambda κ θ γ lam (n ^ 2) = 0 ∧ lam.re = n * c := by
  have hk : κ ≠ 0 := h.1.ne'
  have hD : 0 ≤ 4 * κ * θ - γ ^ 2 := by linarith [h.2.2]
  obtain ⟨w, hw⟩ := IsAlgClosed.exists_pow_nat_eq (hadamardRatio κ θ γ) two_pos
  have hre := hadamard_sqrt_re_ne_zero hw (hadamardRatio_im_ne_zero h)
  -- choose the sign of the square root with positive real part
  obtain ⟨w', hw', hwpos⟩ : ∃ w' : ℂ, w' ^ 2 = hadamardRatio κ θ γ ∧ 0 < w'.re := by
    rcases lt_or_gt_of_ne hre with hneg | hposw
    · exact ⟨-w, by rw [neg_sq]; exact hw, by simp; linarith⟩
    · exact ⟨w, hw, hposw⟩
  refine ⟨w'.re, hwpos, fun n hn => ⟨(n : ℂ) * w', ?_, ?_⟩⟩
  · apply frozen_symbol_hadamard_root hk hD
    rw [mul_pow, hw']
    push_cast
    ring
  · simp

/-! ## Non-uniformity of the `κ → 0⁺` limit on the `γ = 0` admissible ray -/

/-- On the admissible ray `γ = 0`, the squared Hadamard constant `(‖μ‖ + Re μ)/2` exceeds every
bound for all sufficiently small `κ > 0`. -/
theorem hadamard_constant_unbounded_as_kappa_to_zero {θ : ℝ} (hθ : 0 < θ) (M : ℝ) :
    ∃ κ₀ : ℝ, 0 < κ₀ ∧ ∀ κ : ℝ, 0 < κ → κ < κ₀ →
      M < (‖hadamardRatio κ θ 0‖ + (hadamardRatio κ θ 0).re) / 2 := by
  set B : ℝ := max M 0 with hB
  have hB0 : 0 ≤ B := le_max_right _ _
  refine ⟨θ / (2 * B + 2) ^ 2, by positivity, fun κ hk hκ => ?_⟩
  have hre : (hadamardRatio κ θ 0).re = -1 := by
    simp only [hadamardRatio, add_zero]
    field_simp
  have him : (hadamardRatio κ θ 0).im = Real.sqrt (4 * κ * θ) / (2 * κ) := by
    simp [hadamardRatio]
  -- `|Im μ| ≤ ‖μ‖`
  have himle : |(hadamardRatio κ θ 0).im| ≤ ‖hadamardRatio κ θ 0‖ := Complex.abs_im_le_norm _
  -- `√(4κθ)/(2κ) > 2B + 2` for small κ
  have hsmall : 2 * B + 2 < Real.sqrt (4 * κ * θ) / (2 * κ) := by
    have h1 : κ * (2 * B + 2) ^ 2 < θ := by
      have hpos : 0 < (2 * B + 2) ^ 2 := by positivity
      have := (lt_div_iff₀ hpos).mp hκ
      linarith
    rw [lt_div_iff₀ (by positivity)]
    have hsq : (2 * B + 2) * (2 * κ) < Real.sqrt (4 * κ * θ) := by
      apply Real.lt_sqrt_of_sq_lt
      nlinarith
    linarith
  have himpos : 0 < Real.sqrt (4 * κ * θ) / (2 * κ) := by
    have := hsmall
    linarith
  rw [him, abs_of_pos himpos] at himle
  rw [hre]
  have : M ≤ B := le_max_left _ _
  linarith

/-! ## Exponential growth defeats every polynomial (Sobolev) weight -/

/-- For `b > 0` and every `N`, `exp(b x)/(1+x)^N → ∞`. Applied with `b = c t`, `x = |ξ|`, this is
why the Hadamard modes violate every estimate `‖u(t)‖_{H^{s'}} ≤ C ‖u(0)‖_{H^s}`. -/
theorem exp_beats_every_polynomial_weight {b : ℝ} (hb : 0 < b) (N : ℕ) :
    Tendsto (fun x : ℝ => Real.exp (b * x) / (1 + x) ^ N) atTop atTop := by
  have hbase : Tendsto (fun x : ℝ => Real.exp (b * x) / x ^ (N : ℝ)) atTop atTop :=
    tendsto_exp_mul_div_rpow_atTop (N : ℝ) b hb
  have hscaled : Tendsto (fun x : ℝ => (1 / 2 ^ N : ℝ) * (Real.exp (b * x) / x ^ (N : ℝ)))
      atTop atTop := Tendsto.const_mul_atTop (by positivity) hbase
  refine tendsto_atTop_mono' atTop ?_ hscaled
  filter_upwards [eventually_ge_atTop (1 : ℝ)] with x hx
  have hx0 : 0 < x := by linarith
  have hrpow : x ^ (N : ℝ) = x ^ N := Real.rpow_natCast x N
  rw [hrpow]
  have h1x : (1 + x) ^ N ≤ (2 * x) ^ N := pow_le_pow_left₀ (by linarith) (by linarith) N
  have hpos1 : 0 < (1 + x) ^ N := by positivity
  have hposx : 0 < x ^ N := by positivity
  have he : 0 < Real.exp (b * x) := Real.exp_pos _
  rw [mul_pow] at h1x
  have hrw : 1 / 2 ^ N * (Real.exp (b * x) / x ^ N) = Real.exp (b * x) / (2 ^ N * x ^ N) := by
    field_simp
  rw [hrw]
  exact div_le_div_of_nonneg_left he.le hpos1 h1x

end ReleaseProofs.S004CharacteristicCone
