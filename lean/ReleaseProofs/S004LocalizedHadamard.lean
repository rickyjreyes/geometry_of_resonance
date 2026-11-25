import Mathlib
import ReleaseProofs.S004CharacteristicCone

/-!
# S004 Phase XXXIII: kernel-checked ingredients of the localized Hadamard construction

The variable-coefficient Sobolev ill-posedness theorem for the linearized corrected S004 equations
(`docs/s004_closure/VARIABLE_COEFFICIENT_ILLPOSEDNESS.md`) is proved there on paper. Its proof is a
Gaussian-beam construction (the mechanism of Lax 1957 and Mizohata 1961, written out directly for
this operator; the theorems are not cited). This module kernel-checks the finite-dimensional
and one-dimensional-integral ingredients the proof uses. It does not formalize Sobolev spaces,
the variable-coefficient transport hierarchy, or the passage to the PDE estimate.

0. **Factorization and simplicity of the Hadamard root** (in `S004CharacteristicCone`:
   `laplaceSymbol_factor`, `laplaceSymbolDeriv_at_hadamard_root`, `hadamard_root_simple`). With
   `∂ₜ ↦ λ` and `Z = ζ'·ζ'`, the frozen symbol is `κ(λ² + μZ)(λ² + μ̄Z)`. At the root `λ = w`,
   `w² = μ`, `Z = −1`, its `λ`-derivative is `2 w √(4κθ − γ²) i ≠ 0`. This is the hypothesis that
   makes the transport equation of the WKB construction solvable (`∂_{ζ₀} P ≠ 0`).
1. **The exact localized beam of the frozen principal part in `1+1` dimensions**
   (`beamPhase`, `beam_eikonal`, `beam_first_order_annihilated`). The complex phase
   `φ = i y + w t − (y − i w t)²/2` satisfies `φₜ + i w φ_y = 0` and the eikonal equation exactly.
   Since `P(ζ) = κ(ζ₀ + iwζ₁)(ζ₀ − iwζ₁)(ζ₀² + μ̄ζ₁²)`, `exp(Nφ)` is annihilated by the first
   factor; the resulting identity `P(∂) exp(Nφ) = 0` is checked symbolically (I3 of
   `derivations/s004_illposedness_verify.py`).
2. **Exact `L²` growth of the beam** (`beam_norm_sq_integral`, `beam_growth_on_log_scale`):
   `∫ |exp(Nφ(t,y))|² dy = √(π/N) · exp(2Nt Re w + N t² (Re w)²)`. On the time scale
   `t = K log N / N` the squared norm grows by at least `N^{2K Re w}`.
3. **Phase inequalities of the localization step** (`gaussian_completing_square`,
   `annulus_phase_bound`, `log_scale_exponential`).
4. **Exponent bookkeeping of the contradiction** (`ratio_tendsto_atTop`,
   `no_uniform_estimate_of_ratio`, `parameter_choice`). If the left side is at least `c N^a`, the
   data are at most `C₁ N^b` with `b < a`, and the source is at most
   `C₂ N^{a−δ₁}(log N)^k + C₃ N^{a−δ₂}` with `δᵢ > 0`, then no constant bounds the left side by
   data plus source. For every Sobolev pair `(s, σ)` the parameters `K, K', J` of the construction
   can be chosen to produce such exponent gaps.

The hypotheses of item 4 are the analytic PDE estimates of the paper proof. They are inputs, not
the conclusion: the conclusion (failure of the estimate) is derived.
-/

namespace ReleaseProofs.S004LocalizedHadamard

open ReleaseProofs.S004CharacteristicCone
open Complex MeasureTheory Filter Topology

/-! ## 1. The exact localized beam of the frozen principal part in `1+1` dimensions -/

/-- Complex Gaussian-beam phase `φ(t, y) = i y + w t − (y − i w t)²/2`. -/
noncomputable def beamPhase (w : ℂ) (t y : ℝ) : ℂ :=
  I * y + w * t - ((y : ℂ) - I * w * t) ^ 2 / 2

/-- `φₜ = w + i w (y − i w t)`. -/
noncomputable def beamPhaseT (w : ℂ) (t y : ℝ) : ℂ := w + I * w * ((y : ℂ) - I * w * t)

/-- `φ_y = i − (y − i w t)`. -/
noncomputable def beamPhaseY (w : ℂ) (t y : ℝ) : ℂ := I - ((y : ℂ) - I * w * t)

theorem beamPhase_hasDerivAt_t (w : ℂ) (t y : ℝ) :
    HasDerivAt (fun s : ℝ => beamPhase w s y) (beamPhaseT w t y) t := by
  have hs : HasDerivAt (fun s : ℝ => (s : ℂ)) 1 t := by
    simpa using (hasDerivAt_id t).ofReal_comp
  have h1 : HasDerivAt (fun s : ℝ => w * (s : ℂ)) (w * 1) t := hs.const_mul w
  have h2 : HasDerivAt (fun s : ℝ => (y : ℂ) - I * w * (s : ℂ)) (-(I * w * 1)) t := by
    simpa using (hs.const_mul (I * w)).const_sub (y : ℂ)
  have h3 := (h2.pow 2).div_const 2
  have h4 := ((h1.const_add (I * (y : ℂ))).sub h3)
  unfold beamPhase beamPhaseT
  convert h4 using 1
  norm_num
  ring

theorem beamPhase_hasDerivAt_y (w : ℂ) (t y : ℝ) :
    HasDerivAt (fun x : ℝ => beamPhase w t x) (beamPhaseY w t y) y := by
  have hx : HasDerivAt (fun x : ℝ => (x : ℂ)) 1 y := by
    simpa using (hasDerivAt_id y).ofReal_comp
  have h1 : HasDerivAt (fun x : ℝ => I * (x : ℂ)) (I * 1) y := hx.const_mul I
  have h2 : HasDerivAt (fun x : ℝ => (x : ℂ) - I * w * t) 1 y := hx.sub_const _
  have h3 := (h2.pow 2).div_const 2
  have h4 := (h1.add_const (w * (t : ℂ))).sub h3
  unfold beamPhase beamPhaseY
  convert h4 using 1
  · push_cast
    ring

/-- **First-order transport identity** `φₜ + i w φ_y = 0`. -/
theorem beam_transport (w : ℂ) (t y : ℝ) : beamPhaseT w t y + I * w * beamPhaseY w t y = 0 := by
  unfold beamPhaseT beamPhaseY
  have hI : I * I = -1 := I_mul_I
  linear_combination w * hI

/-- **Exact eikonal equation**: `P(φₜ, φ_y) = 0` (with `Z = φ_y²`) whenever `w² = μ`. -/
theorem beam_eikonal {κ θ γ : ℝ} (hk : κ ≠ 0) (hD : 0 ≤ 4 * κ * θ - γ ^ 2) {w : ℂ}
    (hw : w ^ 2 = hadamardRatio κ θ γ) (t y : ℝ) :
    laplaceSymbol κ θ γ (beamPhaseT w t y) (beamPhaseY w t y ^ 2) = 0 := by
  have hroot := hadamardRatio_root hk hD
  have hT : beamPhaseT w t y = -(I * w * beamPhaseY w t y) := by
    linear_combination beam_transport w t y
  have hI : I ^ 2 = -1 := I_sq
  rw [hT]
  unfold laplaceSymbol
  have key : (κ : ℂ) * (beamPhaseY w t y ^ 2 - (-(I * w * beamPhaseY w t y)) ^ 2) ^ 2 +
      (γ : ℂ) * (beamPhaseY w t y ^ 2 - (-(I * w * beamPhaseY w t y)) ^ 2) *
        beamPhaseY w t y ^ 2 + (θ : ℂ) * (beamPhaseY w t y ^ 2) ^ 2 =
      beamPhaseY w t y ^ 4 * ((κ : ℂ) * (1 - I ^ 2 * w ^ 2) ^ 2 +
        (γ : ℂ) * (1 - I ^ 2 * w ^ 2) + θ) := by ring
  rw [key, hI, hw]
  have : (1 - (-1) * hadamardRatio κ θ γ) = hadamardRatio κ θ γ + 1 := by ring
  rw [this, hroot, mul_zero]

/-- The beam `exp(N φ)`. -/
noncomputable def beam (N : ℝ) (w : ℂ) (t y : ℝ) : ℂ := Complex.exp ((N : ℂ) * beamPhase w t y)

theorem beam_hasDerivAt_t (N : ℝ) (w : ℂ) (t y : ℝ) :
    HasDerivAt (fun s => beam N w s y) ((N : ℂ) * beamPhaseT w t y * beam N w t y) t := by
  have := ((beamPhase_hasDerivAt_t w t y).const_mul (N : ℂ)).cexp
  unfold beam
  convert this using 1
  ring

theorem beam_hasDerivAt_y (N : ℝ) (w : ℂ) (t y : ℝ) :
    HasDerivAt (fun x => beam N w t x) ((N : ℂ) * beamPhaseY w t y * beam N w t y) y := by
  have := ((beamPhase_hasDerivAt_y w t y).const_mul (N : ℂ)).cexp
  unfold beam
  convert this using 1
  ring

/-- **The beam is annihilated by the first-order factor** `∂ₜ + i w ∂_y`. -/
theorem beam_first_order_annihilated (N : ℝ) (w : ℂ) (t y : ℝ) :
    (N : ℂ) * beamPhaseT w t y * beam N w t y +
      I * w * ((N : ℂ) * beamPhaseY w t y * beam N w t y) = 0 := by
  have h := beam_transport w t y
  linear_combination ((N : ℂ) * beam N w t y) * h

/-- The first-order factor divides the frozen symbol:
`P(ζ₀, ζ₁²) = κ (ζ₀ + iwζ₁)(ζ₀ − iwζ₁)(ζ₀² + μ̄ ζ₁²)`. -/
theorem laplaceSymbol_first_order_factor {κ θ γ : ℝ} (hk : κ ≠ 0) (hD : 0 ≤ 4 * κ * θ - γ ^ 2)
    {w : ℂ} (hw : w ^ 2 = hadamardRatio κ θ γ) (z0 z1 : ℂ) :
    laplaceSymbol κ θ γ z0 (z1 ^ 2) =
      (κ : ℂ) * (z0 + I * w * z1) * (z0 - I * w * z1) *
        (z0 ^ 2 + hadamardRatioConj κ θ γ * z1 ^ 2) := by
  rw [laplaceSymbol_factor hk hD]
  have hI : I ^ 2 = -1 := I_sq
  have : z0 ^ 2 + hadamardRatio κ θ γ * z1 ^ 2 = (z0 + I * w * z1) * (z0 - I * w * z1) := by
    rw [← hw]
    linear_combination (w ^ 2 * z1 ^ 2) * hI
  rw [this]
  ring

/-! ## 2. Exact `L²` norm and growth of the beam -/

theorem beamPhase_re (w : ℂ) (t y : ℝ) :
    (beamPhase w t y).re = t * w.re - ((y + t * w.im) ^ 2 - (t * w.re) ^ 2) / 2 := by
  unfold beamPhase
  simp only [Complex.sub_re, Complex.add_re, Complex.mul_re, Complex.mul_im, Complex.I_re,
    Complex.I_im, Complex.ofReal_re, Complex.ofReal_im, Complex.div_re, Complex.sub_im,
    pow_two]
  norm_num
  ring

theorem beam_norm_sq (N : ℝ) (w : ℂ) (t y : ℝ) :
    ‖beam N w t y‖ ^ 2 =
      Real.exp (2 * N * t * w.re + N * t ^ 2 * w.re ^ 2) * Real.exp (-N * (y + t * w.im) ^ 2) := by
  unfold beam
  rw [Complex.norm_exp, Complex.re_ofReal_mul, beamPhase_re, ← Real.exp_add, sq,
    ← Real.exp_add]
  congr 1
  ring

/-- For `N > 0` the squared modulus of the beam is integrable in `y` at every time. -/
theorem beam_sq_integrable {N : ℝ} (hN : 0 < N) (w : ℂ) (t : ℝ) :
    Integrable (fun y : ℝ => ‖beam N w t y‖ ^ 2) := by
  simp_rw [beam_norm_sq]
  exact ((integrable_exp_neg_mul_sq hN).comp_add_right (t * w.im)).const_mul _

/-- **Closed-form `L²` norm of the beam**: for `N > 0` the squared modulus is integrable and
`∫ |exp(Nφ(t,y))|² dy = √(π/N) · exp(2Nt Re w + N t² (Re w)²)`. -/
theorem beam_norm_sq_integral {N : ℝ} (hN : 0 < N) (w : ℂ) (t : ℝ) :
    Integrable (fun y : ℝ => ‖beam N w t y‖ ^ 2) ∧
    ∫ y : ℝ, ‖beam N w t y‖ ^ 2 =
      Real.sqrt (Real.pi / N) * Real.exp (2 * N * t * w.re + N * t ^ 2 * w.re ^ 2) := by
  refine ⟨beam_sq_integrable hN w t, ?_⟩
  simp_rw [beam_norm_sq]
  rw [integral_const_mul]
  have htrans : ∫ y : ℝ, Real.exp (-N * (y + t * w.im) ^ 2) =
      ∫ y : ℝ, Real.exp (-N * y ^ 2) :=
    integral_add_right_eq_self (fun y : ℝ => Real.exp (-N * y ^ 2)) (t * w.im)
  rw [htrans, integral_gaussian]
  ring

/-- Growth factor of the squared norm between times `0` and `t`. -/
theorem beam_growth_factor {N : ℝ} (hN : 0 < N) (w : ℂ) (t : ℝ) :
    ∫ y : ℝ, ‖beam N w t y‖ ^ 2 =
      Real.exp (2 * N * t * w.re + N * t ^ 2 * w.re ^ 2) * ∫ y : ℝ, ‖beam N w 0 y‖ ^ 2 := by
  rw [(beam_norm_sq_integral hN w t).2, (beam_norm_sq_integral hN w 0).2]
  have h0 : (2 * N * 0 * w.re + N * 0 ^ 2 * w.re ^ 2) = 0 := by ring
  rw [h0, Real.exp_zero, mul_one, mul_comm]

/-- On the logarithmic scale `t = K log N / N`, `exp(2 N t c) = N^{2Kc}`. -/
theorem log_scale_exponential {N : ℝ} (hN : 0 < N) (K c : ℝ) :
    Real.exp (2 * N * (K * Real.log N / N) * c) = N ^ (2 * K * c) := by
  rw [Real.rpow_def_of_pos hN]
  congr 1
  field_simp

/-- **Polynomial-in-`N` growth on the shrinking time scale `K log N / N`**: the squared `L²` norm
of the beam grows at least by `N^{2K Re w}`, for every `K ≥ 0`. -/
theorem beam_growth_on_log_scale {N : ℝ} (hN : 0 < N) (w : ℂ) (K : ℝ) :
    N ^ (2 * K * w.re) * ∫ y : ℝ, ‖beam N w 0 y‖ ^ 2 ≤
      ∫ y : ℝ, ‖beam N w (K * Real.log N / N) y‖ ^ 2 := by
  rw [beam_growth_factor hN w (K * Real.log N / N), ← log_scale_exponential hN K w.re]
  apply mul_le_mul_of_nonneg_right _ (integral_nonneg fun y => by positivity)
  apply Real.exp_le_exp.mpr
  nlinarith [sq_nonneg (K * Real.log N / N), sq_nonneg w.re, hN.le,
    mul_nonneg hN.le (mul_nonneg (sq_nonneg (K * Real.log N / N)) (sq_nonneg w.re))]

/-! ## 3. Phase inequalities of the localization step -/

/-- Completing the square: `−N x² + B x ≤ B²/(4N)` for `N > 0` (upper Gaussian bound with a
linear drift term). -/
theorem gaussian_completing_square {N : ℝ} (hN : 0 < N) (B x : ℝ) :
    -N * x ^ 2 + B * x ≤ B ^ 2 / (4 * N) := by
  rw [le_div_iff₀ (by positivity)]
  nlinarith [sq_nonneg (2 * N * x - B)]

/-- Annulus bound for the cutoff region `ρ₁ ≤ r ≤ ρ₂`: the localized phase satisfies
`−r²/2 + C t r ≤ −ρ₁²/2 + C t ρ₂` (`C, t ≥ 0`). -/
theorem annulus_phase_bound {r ρ₁ ρ₂ C t : ℝ} (hρ₁ : 0 ≤ ρ₁) (h1 : ρ₁ ≤ r) (h2 : r ≤ ρ₂)
    (hC : 0 ≤ C) (ht : 0 ≤ t) :
    -r ^ 2 / 2 + C * t * r ≤ -ρ₁ ^ 2 / 2 + C * t * ρ₂ := by
  have hsq : ρ₁ ^ 2 ≤ r ^ 2 := pow_le_pow_left₀ hρ₁ h1 2
  have hCt : 0 ≤ C * t := mul_nonneg hC ht
  nlinarith [mul_le_mul_of_nonneg_left h2 hCt]

/-! ## 4. Exponent bookkeeping of the contradiction -/

theorem log_pow_mul_rpow_neg_tendsto_zero (k : ℕ) {δ : ℝ} (hδ : 0 < δ) :
    Tendsto (fun N : ℝ => Real.log N ^ k * N ^ (-δ)) atTop (𝓝 0) := by
  have h := (isLittleO_log_rpow_rpow_atTop (k : ℝ) hδ).tendsto_div_nhds_zero
  refine h.congr' ?_
  filter_upwards [eventually_gt_atTop (0 : ℝ)] with N hN
  rw [Real.rpow_natCast, Real.rpow_neg hN.le, div_eq_mul_inv]

/-- **Exponent gaps force the ratio to infinity.** -/
theorem ratio_tendsto_atTop {a b δ₁ δ₂ c C₁ C₂ C₃ : ℝ} (k : ℕ) (hab : b < a) (h₁ : 0 < δ₁)
    (h₂ : 0 < δ₂) (hc : 0 < c) (hC₁ : 0 < C₁) (hC₂ : 0 ≤ C₂) (hC₃ : 0 ≤ C₃) :
    Tendsto (fun N : ℝ => c * N ^ a /
      (C₁ * N ^ b + C₂ * N ^ (a - δ₁) * Real.log N ^ k + C₃ * N ^ (a - δ₂))) atTop atTop := by
  set den : ℝ → ℝ := fun N => C₁ * N ^ (-(a - b)) + C₂ * (Real.log N ^ k * N ^ (-δ₁)) +
    C₃ * N ^ (-δ₂) with hden
  have hlim : Tendsto den atTop (𝓝 0) := by
    have t1 := (tendsto_rpow_neg_atTop (sub_pos.mpr hab)).const_mul C₁
    have t2 := (log_pow_mul_rpow_neg_tendsto_zero k h₁).const_mul C₂
    have t3 := (tendsto_rpow_neg_atTop h₂).const_mul C₃
    simpa [hden] using (t1.add t2).add t3
  have hposev : ∀ᶠ N in atTop, 0 < den N := by
    filter_upwards [eventually_gt_atTop (1 : ℝ)] with N hN
    have hN0 : 0 < N := by linarith
    have hlog : 0 ≤ Real.log N := Real.log_nonneg hN.le
    have e1 : 0 < C₁ * N ^ (-(a - b)) := mul_pos hC₁ (Real.rpow_pos_of_pos hN0 _)
    have e2 : 0 ≤ C₂ * (Real.log N ^ k * N ^ (-δ₁)) :=
      mul_nonneg hC₂ (mul_nonneg (pow_nonneg hlog k) (Real.rpow_nonneg hN0.le _))
    have e3 : 0 ≤ C₃ * N ^ (-δ₂) := mul_nonneg hC₃ (Real.rpow_nonneg hN0.le _)
    simp only [hden]
    linarith
  have hwithin : Tendsto den atTop (𝓝[>] 0) :=
    tendsto_nhdsWithin_iff.mpr ⟨hlim, hposev⟩
  have hinv := (tendsto_inv_nhdsGT_zero.comp hwithin).const_mul_atTop hc
  refine hinv.congr' ?_
  filter_upwards [eventually_gt_atTop (1 : ℝ)] with N hN
  have hN0 : 0 < N := by linarith
  have hNa : 0 < N ^ a := Real.rpow_pos_of_pos hN0 a
  have r1 : N ^ b = N ^ a * N ^ (-(a - b)) := by
    rw [← Real.rpow_add hN0]; ring_nf
  have r2 : N ^ (a - δ₁) = N ^ a * N ^ (-δ₁) := by
    rw [← Real.rpow_add hN0]; ring_nf
  have r3 : N ^ (a - δ₂) = N ^ a * N ^ (-δ₂) := by
    rw [← Real.rpow_add hN0]; ring_nf
  simp only [Function.comp, hden]
  rw [r1, r2, r3]
  field_simp

/-- If a ratio of positive quantities tends to infinity, no constant bounds it. -/
theorem no_uniform_estimate_of_ratio {f g : ℝ → ℝ}
    (h : Tendsto (fun N => f N / g N) atTop atTop) (hg : ∀ᶠ N in atTop, 0 < g N) :
    ¬ ∃ C : ℝ, ∀ᶠ N in atTop, f N ≤ C * g N := by
  rintro ⟨C, hC⟩
  have hev := (h.eventually (eventually_gt_atTop C)).and (hC.and hg)
  obtain ⟨N, hN1, hN2, hN3⟩ := hev.exists
  rw [lt_div_iff₀ hN3] at hN1
  linarith

/-- **The contradiction of the localized construction.** If a family of candidate solutions
satisfies a lower bound `lhs ≥ c N^a`, data bound `data ≤ C₁ N^b` with `b < a`, and residual
(source) bound `src ≤ C₂ N^{a−δ₁}(log N)^k + C₃ N^{a−δ₂}` with `δᵢ > 0`, then no constant `C`
gives `lhs ≤ C (data + src)` for all large `N`. -/
theorem localized_estimate_fails {lhs data src : ℝ → ℝ} {a b δ₁ δ₂ c C₁ C₂ C₃ : ℝ} (k : ℕ)
    (hab : b < a) (h₁ : 0 < δ₁) (h₂ : 0 < δ₂) (hc : 0 < c) (hC₁ : 0 < C₁) (hC₂ : 0 ≤ C₂)
    (hC₃ : 0 ≤ C₃)
    (hlhs : ∀ᶠ N in atTop, c * N ^ a ≤ lhs N)
    (hdata : ∀ᶠ N in atTop, 0 ≤ data N ∧ data N ≤ C₁ * N ^ b)
    (hsrc : ∀ᶠ N in atTop, 0 ≤ src N ∧
      src N ≤ C₂ * N ^ (a - δ₁) * Real.log N ^ k + C₃ * N ^ (a - δ₂)) :
    ¬ ∃ C : ℝ, ∀ᶠ N in atTop, lhs N ≤ C * (data N + src N) := by
  rintro ⟨C, hC⟩
  have hratio := ratio_tendsto_atTop k hab h₁ h₂ hc hC₁ hC₂ hC₃
  -- compare with the bounding expression
  set U : ℝ → ℝ := fun N => C₁ * N ^ b + C₂ * N ^ (a - δ₁) * Real.log N ^ k + C₃ * N ^ (a - δ₂)
    with hU
  have hUpos : ∀ᶠ N in atTop, 0 < U N := by
    filter_upwards [eventually_gt_atTop (1 : ℝ)] with N hN
    have hN0 : 0 < N := by linarith
    have hlog : 0 ≤ Real.log N := Real.log_nonneg hN.le
    have e1 : 0 < C₁ * N ^ b := mul_pos hC₁ (Real.rpow_pos_of_pos hN0 _)
    have e2 : 0 ≤ C₂ * N ^ (a - δ₁) * Real.log N ^ k :=
      mul_nonneg (mul_nonneg hC₂ (Real.rpow_nonneg hN0.le _)) (pow_nonneg hlog k)
    have e3 : 0 ≤ C₃ * N ^ (a - δ₂) := mul_nonneg hC₃ (Real.rpow_nonneg hN0.le _)
    simp only [hU]
    linarith
  apply no_uniform_estimate_of_ratio hratio hUpos
  refine ⟨max C 0, ?_⟩
  filter_upwards [hC, hlhs, hdata, hsrc] with N h1 h2 h3 h4
  have hsum : data N + src N ≤ U N := by simp only [hU]; linarith [h3.2, h4.2]
  have hnn : 0 ≤ data N + src N := add_nonneg h3.1 h4.1
  calc c * N ^ a ≤ lhs N := h2
    _ ≤ C * (data N + src N) := h1
    _ ≤ max C 0 * (data N + src N) := mul_le_mul_of_nonneg_right (le_max_left _ _) hnn
    _ ≤ max C 0 * U N := mul_le_mul_of_nonneg_left hsum (le_max_right _ _)

/-- **Parameter choice.** For every Sobolev pair `(s, σ)` with `s, σ ≥ 0` and every growth
constant `c > 0`, there are a time-scale constant `K`, a `t`-truncation order `K'` and a number of
WKB corrections `J` with the three exponent gaps `K c > 3 + s + σ`, `K' > 4 + s + σ`,
`J > 2 + s + σ` used in the contradiction. -/
theorem parameter_choice {s σ c : ℝ} (hs : 0 ≤ s) (hσ : 0 ≤ σ) (hc : 0 < c) :
    ∃ K : ℝ, ∃ Kp J : ℕ, 0 < K ∧ 3 + s + σ < K * c ∧ 4 + s + σ < Kp ∧ 2 + s + σ < J := by
  refine ⟨(3 + s + σ) / c + 1, ⌈4 + s + σ⌉₊ + 1, ⌈2 + s + σ⌉₊ + 1, by positivity, ?_, ?_, ?_⟩
  · rw [add_mul, div_mul_cancel₀ _ hc.ne', one_mul]
    linarith
  · push_cast
    linarith [Nat.le_ceil (4 + s + σ)]
  · push_cast
    linarith [Nat.le_ceil (2 + s + σ)]

/-! ## 5. Packaged statement -/

/-- **Localized Hadamard beam for every admissible S004 coefficient triple.** There is a growing
simple Hadamard root `w`; the complex Gaussian-beam phase solves the frozen eikonal equation
exactly; the beam's squared `L²` norm has the closed form above and, on the time scale
`K log N / N`, grows by at least `N^{2K Re w}` relative to its initial value. -/
theorem localized_hadamard_beam {κ θ γ : ℝ} (h : Admissible κ θ γ) :
    ∃ w : ℂ, w ^ 2 = hadamardRatio κ θ γ ∧ 0 < w.re ∧
      laplaceSymbol κ θ γ w (-1) = 0 ∧ laplaceSymbolDeriv κ θ γ w (-1) ≠ 0 ∧
      (∀ t y : ℝ, beamPhaseT w t y + I * w * beamPhaseY w t y = 0 ∧
        laplaceSymbol κ θ γ (beamPhaseT w t y) (beamPhaseY w t y ^ 2) = 0) ∧
      (∀ N : ℝ, 0 < N → ∀ t : ℝ, Integrable (fun y : ℝ => ‖beam N w t y‖ ^ 2) ∧
        ∫ y : ℝ, ‖beam N w t y‖ ^ 2 =
          Real.sqrt (Real.pi / N) * Real.exp (2 * N * t * w.re + N * t ^ 2 * w.re ^ 2)) ∧
      (∀ N : ℝ, 0 < N → ∀ K : ℝ, N ^ (2 * K * w.re) * ∫ y : ℝ, ‖beam N w 0 y‖ ^ 2 ≤
        ∫ y : ℝ, ‖beam N w (K * Real.log N / N) y‖ ^ 2) := by
  have hk : κ ≠ 0 := h.1.ne'
  have hD : 0 ≤ 4 * κ * θ - γ ^ 2 := by linarith [h.2.2]
  obtain ⟨w, hw, hpos, hroot, hsimple⟩ := exists_growing_simple_root h
  exact ⟨w, hw, hpos, hroot, hsimple,
    fun t y => ⟨beam_transport w t y, beam_eikonal hk hD hw t y⟩,
    fun N hN t => beam_norm_sq_integral hN w t,
    fun N hN K => beam_growth_on_log_scale hN w K⟩

end ReleaseProofs.S004LocalizedHadamard
