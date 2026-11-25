import Mathlib
import ReleaseProofs.R18_FiniteScale
import ReleaseProofs.R16_PaisUhlenbeck
import ReleaseProofs.R16_GhostObstruction
import ReleaseProofs.S004CharacteristicCone
import ReleaseProofs.S004HarmonicGhostForcing

/-!
# S004 Phase XXXI: exact second-harmonic ghost resonance — classification and certificate

Amplitude-mode dispersion of the corrected S004 action around a static homogeneous background
(repository Q6/Q7, `R18.inversePropagator`): `D(z,q) = (z−q) − m² − K(z−q)² + Gq(z−q) − Tq²`
with `z = ω²`, `q = k²`, `K = F(ρ₀)κ`, `G = F(ρ₀)γ`, `T = F(ρ₀)θ`, `m² = −2ρ₀V''(ρ₀)`. In the real
window the poles are `R18.lowerPole` (ordinary, positive Pais–Uhlenbeck energy) and
`R18.upperPole` (ghost, negative energy).

**Second-harmonic ghost resonance**: an ordinary standing wave `(k, ω₋(k))` forces the `2k` mode at
frequency `2ω₋(k)` (`S004HarmonicGhostForcing.sigma2Raw`); it is resonant with the ghost iff
`resonanceGap = z₊(4k²) − 4 z₋(k²) = 0`.

Results:

* `resonanceGap_formula`: the `q` and `Gq` terms cancel exactly:
  `2K · resonanceGap = √D(4q) + 4√D(q) − 3`.
* `resonance_exists_of_band`, `resonance_band_of_resonance`: for `γ = 0` (`G = 0`), `K, T > 0`,
  a resonance with both `k` and `2k` in the real window exists **iff**
  `9/25 ≤ δ² ≤ 3/5`, `δ² := 1 − 4Km²`, at the closed form `B q² = δ² − ((3+5δ²)/8)²`.
* Certified instance (`certified_*`): actual weight `F(u) = u/(u+1)²` (`ε = 1`, `α = 0`) at
  `ρ₀ = 1/2` (`F = 2/9`, `F' = 4/27`), `κ = 1/2`, `θ = 1`, `γ = 0`, `V''(ρ₀) = −459/400`
  (`δ = 7/10`): `k*⁴ = 53703/204800`, `D(k*²) = (109/160)²`, `D(4k*²) = (11/40)²`, exact resonance,
  `k* ∈ [1/100, 37/50]`, the resonant `2k*` frequency is the UPPER (negative-energy) pole, and
  `σ₂ = −4131·√(1/2)/1600 ≠ 0` when `V'''(ρ₀) = 0`; `σ₂ = 0` iff `V'''(ρ₀) = 4131/800`.
* `certified_no_periodic_second_order`: at the certified resonance the `cos 2kz cos 2ωt`
  component of the second-order Poincaré–Lindstedt equation has no solution: the ordinary-mode
  Lyapunov (standing-wave) family does not exist to second order.
* `certified_third_harmonic_elliptic`: `3k*` lies outside the real window (no real frequency).
* `original_candidate_no_resonance`: the earlier candidate `V''(ρ₀) = −1/2` (`δ² = 7/9`) admits NO
  resonance at any `k` with `k, 2k` in the window — that claim is refuted.
-/

namespace ReleaseProofs.S004ResonanceCertificate

open ReleaseProofs.R18
open ReleaseProofs.S004CharacteristicCone
open ReleaseProofs.S004HarmonicGhostForcing

/-- `z₊(4q) − 4 z₋(q)`: second-harmonic resonance function into the ghost branch. -/
noncomputable def resonanceGap (K G T m2 q : ℝ) : ℝ :=
  upperPole K G T m2 (4 * q) - 4 * lowerPole K G T m2 q

theorem resonanceGap_formula (K G T m2 q : ℝ) (hK : K ≠ 0) :
    2 * K * resonanceGap K G T m2 q =
      Real.sqrt (poleDiscriminant K G T m2 (4 * q)) +
        4 * Real.sqrt (poleDiscriminant K G T m2 q) - 3 := by
  unfold resonanceGap upperPole lowerPole
  field_simp
  ring

theorem disc_G_zero (K T m2 q : ℝ) :
    poleDiscriminant K 0 T m2 q = (1 - 4 * K * m2) - 4 * K * T * q ^ 2 := by
  unfold poleDiscriminant
  ring

/-- **Existence on the band.** For `G = 0`, `K, T > 0` and `9/25 ≤ δ² ≤ 3/5` there is an exact
resonance with `k²` and `(2k)²` in the real window. -/
theorem resonance_exists_of_band {K T m2 : ℝ} (hK : 0 < K) (hT : 0 < T)
    (hlo : 9 / 25 ≤ 1 - 4 * K * m2) (hhi : 1 - 4 * K * m2 ≤ 3 / 5) :
    ∃ q : ℝ, 0 ≤ q ∧ resonanceGap K 0 T m2 q = 0 ∧
      0 ≤ poleDiscriminant K 0 T m2 q ∧ 0 ≤ poleDiscriminant K 0 T m2 (4 * q) := by
  set d := 1 - 4 * K * m2 with hd
  set s := d - ((3 + 5 * d) / 8) ^ 2 with hs
  have hs0 : 0 ≤ s := by rw [hs]; nlinarith
  have hB : 0 < 4 * K * T := by positivity
  refine ⟨Real.sqrt (s / (4 * K * T)), Real.sqrt_nonneg _, ?_, ?_, ?_⟩
  all_goals
    have hq2 : Real.sqrt (s / (4 * K * T)) ^ 2 = s / (4 * K * T) :=
      Real.sq_sqrt (div_nonneg hs0 hB.le)
  · have h1 : poleDiscriminant K 0 T m2 (Real.sqrt (s / (4 * K * T))) =
        ((3 + 5 * d) / 8) ^ 2 := by
      rw [disc_G_zero, ← hd, hq2]
      field_simp
      rw [hs]
      ring
    have h4 : poleDiscriminant K 0 T m2 (4 * Real.sqrt (s / (4 * K * T))) =
        ((3 - 5 * d) / 2) ^ 2 := by
      rw [disc_G_zero, ← hd, mul_pow, hq2]
      field_simp
      rw [hs]
      ring
    have hgap := resonanceGap_formula K 0 T m2 (Real.sqrt (s / (4 * K * T))) hK.ne'
    rw [h1, h4, Real.sqrt_sq (by linarith), Real.sqrt_sq (by linarith)] at hgap
    have h2K : 2 * K ≠ 0 := by positivity
    have : 2 * K * resonanceGap K 0 T m2 (Real.sqrt (s / (4 * K * T))) = 0 := by
      rw [hgap]
      ring
    exact (mul_eq_zero.mp this).resolve_left h2K
  · rw [disc_G_zero, ← hd, hq2]
    field_simp
    rw [hs]
    nlinarith
  · rw [disc_G_zero, ← hd, mul_pow, hq2]
    field_simp
    rw [hs]
    nlinarith

/-- **No resonance off the band.** If a resonance occurs with `k²`, `(2k)²` in the real window
(`G = 0`, `K, T > 0`), then `9/25 ≤ δ² ≤ 3/5`. -/
theorem resonance_band_of_resonance {K T m2 q : ℝ} (hK : 0 < K) (hT : 0 < T)
    (h1 : 0 ≤ poleDiscriminant K 0 T m2 q) (h4 : 0 ≤ poleDiscriminant K 0 T m2 (4 * q))
    (hres : resonanceGap K 0 T m2 q = 0) :
    9 / 25 ≤ 1 - 4 * K * m2 ∧ 1 - 4 * K * m2 ≤ 3 / 5 := by
  set d := 1 - 4 * K * m2 with hd
  set r1 := Real.sqrt (poleDiscriminant K 0 T m2 q) with hr1
  set r4 := Real.sqrt (poleDiscriminant K 0 T m2 (4 * q)) with hr4
  have hr1s : r1 ^ 2 = poleDiscriminant K 0 T m2 q := Real.sq_sqrt h1
  have hr4s : r4 ^ 2 = poleDiscriminant K 0 T m2 (4 * q) := Real.sq_sqrt h4
  have hr1n : 0 ≤ r1 := Real.sqrt_nonneg _
  have hr4n : 0 ≤ r4 := Real.sqrt_nonneg _
  have hsum : r4 + 4 * r1 = 3 := by
    have := resonanceGap_formula K 0 T m2 q hK.ne'
    rw [hres, mul_zero] at this
    linarith
  have hd1 : r1 ^ 2 = d - 4 * K * T * q ^ 2 := by rw [hr1s, disc_G_zero]
  have hd4 : r4 ^ 2 = d - 4 * K * T * (4 * q) ^ 2 := by rw [hr4s, disc_G_zero]
  have hB : 0 ≤ 4 * K * T * q ^ 2 := by positivity
  have hrel : 16 * r1 ^ 2 - r4 ^ 2 = 15 * d := by rw [hd1, hd4]; ring
  have hr1v : 8 * r1 = 5 * d + 3 := by
    have : r4 = 3 - 4 * r1 := by linarith
    rw [this] at hrel
    nlinarith
  constructor
  · nlinarith
  · nlinarith

/-! ## Certified instance with the actual corrected weight -/

/-- The corrected S004 weight at `ε = 1`, `α = 0`, `u = 1/2` equals `2/9`. -/
theorem certified_weight_value : R16.correctedWeight (1 / 2) 1 0 = 2 / 9 := by
  unfold R16.correctedWeight R16.regularizedDenominator
  norm_num

/-- Its slope there is `F'(1/2) = 4/27 > 0`. -/
theorem certified_weight_slope :
    HasDerivAt (fun u => R16.correctedWeight u 1 0) (4 / 27) (1 / 2) := by
  have hfun : (fun u => R16.correctedWeight u 1 0) = fun u => u / (u + 1) ^ 2 := by
    funext u
    unfold R16.correctedWeight R16.regularizedDenominator
    norm_num
  rw [hfun]
  have hden : HasDerivAt (fun u : ℝ => (u + 1) ^ 2) (2 * (1 / 2 + 1) ^ 1 * 1) (1 / 2) := by
    have := ((hasDerivAt_id (1 / 2 : ℝ)).add_const 1).pow 2
    simpa using this
  have hdiv := (hasDerivAt_id (1 / 2 : ℝ)).div hden (by norm_num)
  convert hdiv using 1
  norm_num

/-- Certified data: `K = F(ρ₀)κ = 1/9`, `T = F(ρ₀)θ = 2/9`, `m² = −2ρ₀V''(ρ₀) = 459/400`. -/
noncomputable def cK : ℝ := 2 / 9 * (1 / 2)
noncomputable def cT : ℝ := 2 / 9 * 1
noncomputable def cM2 : ℝ := -2 * (1 / 2) * (-459 / 400)
/-- `k*² = √(53703/204800)`. -/
noncomputable def qStar : ℝ := Real.sqrt (53703 / 204800)
noncomputable def kStar : ℝ := Real.sqrt qStar

theorem qStar_sq : qStar ^ 2 = 53703 / 204800 := Real.sq_sqrt (by norm_num)

theorem certified_disc_k : poleDiscriminant cK 0 cT cM2 qStar = (109 / 160) ^ 2 := by
  rw [disc_G_zero, qStar_sq]
  unfold cK cT cM2
  norm_num

theorem certified_disc_2k : poleDiscriminant cK 0 cT cM2 (4 * qStar) = (11 / 40) ^ 2 := by
  rw [disc_G_zero, mul_pow, qStar_sq]
  unfold cK cT cM2
  norm_num

/-- **Exact resonance at the certified point.** -/
theorem certified_resonance : resonanceGap cK 0 cT cM2 qStar = 0 := by
  have h := resonanceGap_formula cK 0 cT cM2 qStar (by unfold cK; norm_num)
  rw [certified_disc_k, certified_disc_2k, Real.sqrt_sq (by norm_num),
    Real.sqrt_sq (by norm_num)] at h
  have hK : (2 : ℝ) * cK ≠ 0 := by unfold cK; norm_num
  have : 2 * cK * resonanceGap cK 0 cT cM2 qStar = 0 := by rw [h]; norm_num
  exact (mul_eq_zero.mp this).resolve_left hK

/-- The certified resonance lies in the candidate window `k* ∈ [1/100, 37/50]`. -/
theorem certified_window : 1 / 100 ≤ kStar ∧ kStar ≤ 37 / 50 := by
  have hq0 : 0 ≤ qStar := Real.sqrt_nonneg _
  have hk4 : kStar ^ 4 = 53703 / 204800 := by
    have : kStar ^ 4 = (kStar ^ 2) ^ 2 := by ring
    rw [this, kStar, Real.sq_sqrt hq0, qStar_sq]
  have hk0 : 0 ≤ kStar := Real.sqrt_nonneg _
  constructor
  · by_contra h
    push_neg at h
    have : kStar ^ 4 < (1 / 100) ^ 4 := pow_lt_pow_left₀ h hk0 (by norm_num)
    rw [hk4] at this
    norm_num at this
  · by_contra h
    push_neg at h
    have : (37 / 50 : ℝ) ^ 4 < kStar ^ 4 := pow_lt_pow_left₀ h (by norm_num) (by norm_num)
    rw [hk4] at this
    norm_num at this

/-- At `2k*` the two real poles are distinct and the resonant frequency `4 z₋(k*²)` is the UPPER
pole. -/
theorem certified_resonant_branch :
    lowerPole cK 0 cT cM2 (4 * qStar) < upperPole cK 0 cT cM2 (4 * qStar) ∧
      4 * lowerPole cK 0 cT cM2 qStar = upperPole cK 0 cT cM2 (4 * qStar) := by
  constructor
  · unfold lowerPole upperPole
    rw [certified_disc_2k, Real.sqrt_sq (by norm_num)]
    unfold cK
    norm_num
  · have := certified_resonance
    unfold resonanceGap at this
    linarith

/-- **Ghost identification.** In the Pais–Uhlenbeck normalization with `K > 0`, the single-mode
energy of the upper pole at `2k*` is negative for every nonzero amplitude. -/
theorem certified_upper_mode_negative_energy (Y : ℝ) (hY : Y ≠ 0) :
    R16PU.puHamiltonian cK (cK * (lowerPole cK 0 cT cM2 (4 * qStar) +
        upperPole cK 0 cT cM2 (4 * qStar)))
      (cK * (lowerPole cK 0 cT cM2 (4 * qStar) * upperPole cK 0 cT cM2 (4 * qStar)))
      0 (Y / (lowerPole cK 0 cT cM2 (4 * qStar) - upperPole cK 0 cT cM2 (4 * qStar))) 0
      (-upperPole cK 0 cT cM2 (4 * qStar) *
        (Y / (lowerPole cK 0 cT cM2 (4 * qStar) - upperPole cK 0 cT cM2 (4 * qStar)))) < 0 := by
  have hlt := certified_resonant_branch.1
  have hne : lowerPole cK 0 cT cM2 (4 * qStar) ≠ upperPole cK 0 cT cM2 (4 * qStar) := hlt.ne
  rw [(R16PU.pu_single_mode_values cK _ _ Y hne).2]
  have hK : 0 < cK := by unfold cK; norm_num
  apply div_neg_of_pos_of_neg
  · exact mul_pos hK (by positivity)
  · linarith

/-- On-shell link: the real-sector linear symbol is twice `R18.inversePropagator` with
`K = f₀κ`, `G = f₀γ`, `T = f₀θ`, `m² = −2A²v₂`. -/
theorem linearSymbol_eq_inversePropagator (A f0 v2 κ θ γ k ω : ℝ) :
    linearSymbolRaw A f0 v2 κ θ γ k ω =
      2 * inversePropagator (f0 * κ) (f0 * γ) (f0 * θ) (-2 * A ^ 2 * v2) (ω ^ 2) (k ^ 2) := by
  unfold linearSymbolRaw inversePropagator
  ring

/-- Exact on-shell value `ω*² − k*² = 459/320` on the ordinary branch at the certified point. -/
theorem certified_ordinary_offset : lowerPole cK 0 cT cM2 qStar - qStar = 459 / 320 := by
  unfold lowerPole
  rw [certified_disc_k, Real.sqrt_sq (by norm_num)]
  unfold cK
  norm_num

/-- **Exact nonzero second-harmonic coupling at the certified resonance** (`V'''(ρ₀) = 0`):
`σ₂ = −4131·A/1600` with `A = √(1/2)`, `ω² = z₋(k*²)`, `k² = q*`. -/
theorem certified_sigma2 (k ω : ℝ) (hk : k ^ 2 = qStar)
    (hω : ω ^ 2 = lowerPole cK 0 cT cM2 qStar) :
    sigma2Raw (Real.sqrt (1 / 2)) (4 / 27) (-459 / 400) 0 (1 / 2) 1 0 k ω =
      -4131 * Real.sqrt (1 / 2) / 1600 := by
  rw [sigma2_factor]
  have hX : ω ^ 2 - k ^ 2 = 459 / 320 := by
    rw [hω, hk]
    exact certified_ordinary_offset
  have hY2 : (k ^ 2) ^ 2 = 53703 / 204800 := by rw [hk, qStar_sq]
  have hA2 : Real.sqrt (1 / 2) ^ 2 = 1 / 2 := Real.sq_sqrt (by norm_num)
  unfold symbolForm
  rw [hX, hY2, hA2]
  ring

/-- **Exact cancellation surface.** At the certified resonance, `σ₂ = 0` iff `V'''(ρ₀) = 4131/800`. -/
theorem certified_sigma2_cancellation_iff (k ω v3 : ℝ) (hk : k ^ 2 = qStar)
    (hω : ω ^ 2 = lowerPole cK 0 cT cM2 qStar) :
    sigma2Raw (Real.sqrt (1 / 2)) (4 / 27) (-459 / 400) v3 (1 / 2) 1 0 k ω = 0 ↔
      v3 = 4131 / 800 := by
  rw [sigma2_factor]
  have hX : ω ^ 2 - k ^ 2 = 459 / 320 := by
    rw [hω, hk]
    exact certified_ordinary_offset
  have hY2 : (k ^ 2) ^ 2 = 53703 / 204800 := by rw [hk, qStar_sq]
  have hA2 : Real.sqrt (1 / 2) ^ 2 = 1 / 2 := Real.sq_sqrt (by norm_num)
  have hApos : 0 < Real.sqrt (1 / 2) := Real.sqrt_pos.mpr (by norm_num)
  unfold symbolForm
  rw [hX, hY2, hA2]
  constructor
  · intro h
    have : Real.sqrt (1 / 2) / 2 * (v3 - 4131 / 800) = 0 := by linarith
    rcases mul_eq_zero.mp this with h0 | h0
    · linarith
    · linarith
  · intro h
    rw [h]
    ring

/-- **No periodic second-order correction at the certified resonance.** The `cos 2kz cos 2ωt`
component of the second-order Poincaré–Lindstedt equation is `(linear symbol at (2k, 2ω))·c + σ₂ = 0`;
at the certified resonance the symbol vanishes while `σ₂ ≠ 0`, so no amplitude `c` exists. -/
theorem certified_no_periodic_second_order (k ω : ℝ) (hk : k ^ 2 = qStar)
    (hω : ω ^ 2 = lowerPole cK 0 cT cM2 qStar) :
    ¬ ∃ c : ℝ,
      linearSymbolRaw (Real.sqrt (1 / 2)) (2 / 9) (-459 / 400) (1 / 2) 1 0 (2 * k) (2 * ω) * c +
        sigma2Raw (Real.sqrt (1 / 2)) (4 / 27) (-459 / 400) 0 (1 / 2) 1 0 k ω = 0 := by
  rintro ⟨c, hc⟩
  have hA2 : Real.sqrt (1 / 2) ^ 2 = 1 / 2 := Real.sq_sqrt (by norm_num)
  have hsym : linearSymbolRaw (Real.sqrt (1 / 2)) (2 / 9) (-459 / 400) (1 / 2) 1 0
      (2 * k) (2 * ω) = 0 := by
    rw [linearSymbol_eq_inversePropagator, hA2]
    have hz : (2 * ω) ^ 2 = upperPole cK 0 cT cM2 (4 * qStar) := by
      rw [mul_pow, hω]
      have := certified_resonant_branch.2
      linarith
    have hq : (2 * k) ^ 2 = 4 * qStar := by rw [mul_pow, hk]; norm_num
    rw [hz, hq]
    have hon := upper_pole_on_shell cK 0 cT cM2 (4 * qStar) (by unfold cK; norm_num)
      (by rw [certified_disc_2k]; positivity)
    have hK : (2 / 9 : ℝ) * (1 / 2) = cK := by unfold cK; ring
    have hT : (2 / 9 : ℝ) * 1 = cT := by unfold cT; ring
    have hG : (2 / 9 : ℝ) * 0 = 0 := by ring
    have hM : -2 * (1 / 2 : ℝ) * (-459 / 400) = cM2 := by unfold cM2; ring
    rw [hK, hT, hG, hM, hon]
    ring
  rw [hsym, zero_mul, zero_add, certified_sigma2 k ω hk hω] at hc
  have hApos : 0 < Real.sqrt (1 / 2) := Real.sqrt_pos.mpr (by norm_num)
  nlinarith

/-- **The third harmonic is in the elliptic band**: `D(9k*²) < 0`, so the `3k*` mode has no real
frequency (its frequencies are complex: Hadamard growth). -/
theorem certified_third_harmonic_elliptic :
    poleDiscriminant cK 0 cT cM2 (9 * qStar) < 0 ∧
      ¬ ∃ z : ℝ, inversePropagator cK 0 cT cM2 z (9 * qStar) = 0 := by
  have hneg : poleDiscriminant cK 0 cT cM2 (9 * qStar) < 0 := by
    rw [disc_G_zero, mul_pow, qStar_sq]
    unfold cK cT cM2
    norm_num
  exact ⟨hneg, no_real_frequency_root cK 0 cT cM2 (9 * qStar) hneg⟩

/-- **The earlier candidate is refuted.** With the same weight data and `V''(ρ₀) = −1/2`
(`m² = 1/2`, `δ² = 7/9 > 3/5`) there is no second-harmonic ghost resonance at any `k` with `k²`
and `(2k)²` in the real window. -/
theorem original_candidate_no_resonance (q : ℝ)
    (h1 : 0 ≤ poleDiscriminant cK 0 cT (1 / 2) q)
    (h4 : 0 ≤ poleDiscriminant cK 0 cT (1 / 2) (4 * q)) :
    resonanceGap cK 0 cT (1 / 2) q ≠ 0 := by
  intro hres
  have := (resonance_band_of_resonance (by unfold cK; norm_num) (by unfold cT; norm_num)
    h1 h4 hres).2
  unfold cK at this
  norm_num at this

end ReleaseProofs.S004ResonanceCertificate
