import Mathlib
import ReleaseProofs.R18_ExactMinimizer

/-!
# Exact S004 physical-admissibility consequences

This file strengthens the real-pole non-tachyonic result to both signs of G
on the *strict* curvature domain. It separately records an exact phase/
amplitude mismatch and the hypotheses needed to extend the conditional
tangent minimum across the whole nonnegative-q real-pole interval.

None of this removes the upper negative-residue ghost or UV complex poles.
-/

namespace ReleaseProofs.R18Admissibility

open ReleaseProofs.R18
open ReleaseProofs.R18Canonical
open ReleaseProofs.R18Classification
open ReleaseProofs.R18ExactMinimum

/-- Strict curvature positivity and nonnegative mass force the real-pole
    factor (1+Gq) to be positive at every nonzero spatial q, even if G<0.
    Requiring G≥0 is therefore unnecessary in the no-tachyon theorem. -/
theorem real_pole_base_positive_all_G
    (K G T massSq q : ℝ)
    (hK : 0 < K) (hB : 0 < curvatureGap K G T)
    (hm : 0 ≤ massSq) (hq : 0 < q)
    (hd : 0 ≤ poleDiscriminant K G T massSq q) :
    0 < 1 + G * q := by
  have hproduct : 0 ≤ curvatureGap K G T * q ^ 2 :=
    mul_nonneg hB.le (sq_nonneg q)
  have hmass : 0 ≤ 4 * K * massSq := by
    have hk4 : 0 ≤ 4 * K := by positivity
    exact mul_nonneg hk4 hm
  have hex := discriminant_expansion K G T massSq q
  by_contra hn
  have hbad : 1 + G * q ≤ 0 := le_of_not_gt hn
  nlinarith

/-- Both real squared frequencies are strictly positive for q>0, any G,
    K>0,T>0,4KT-G²>0,m²≥0. The upper real root remains a ghost. -/
theorem real_poles_non_tachyonic_all_G
    (K G T massSq q : ℝ)
    (hK : 0 < K) (hT : 0 < T)
    (hB : 0 < curvatureGap K G T)
    (hm : 0 ≤ massSq) (hq : 0 < q)
    (hd : 0 ≤ poleDiscriminant K G T massSq q) :
    0 < lowerPole K G T massSq q ∧
      0 < upperPole K G T massSq q := by
  have hb := real_pole_base_positive_all_G
    K G T massSq q hK hB hm hq hd
  have hq2 : 0 < q ^ 2 := sq_pos_of_pos hq
  have hmT : 0 < massSq + T * q ^ 2 := by
    have hTq : 0 < T * q ^ 2 := mul_pos hT hq2
    linarith
  have hterm : 0 < 4 * K * (massSq + T * q ^ 2) :=
    mul_pos (by positivity) hmT
  have hs : (Real.sqrt (poleDiscriminant K G T massSq q)) ^ 2 =
      poleDiscriminant K G T massSq q := Real.sq_sqrt hd
  have hnon : 0 ≤ Real.sqrt (poleDiscriminant K G T massSq q) :=
    Real.sqrt_nonneg _
  have hdiff :
      (Real.sqrt (poleDiscriminant K G T massSq q)) ^ 2 <
        (1 + G * q) ^ 2 := by
    rw [hs]
    unfold poleDiscriminant
    nlinarith [hterm]
  have hless :
      Real.sqrt (poleDiscriminant K G T massSq q) < 1 + G * q := by
    nlinarith [hdiff]
  have hnum : 0 < (1 + G * q -
      Real.sqrt (poleDiscriminant K G T massSq q)) / (2 * K) :=
    div_pos (sub_pos.mpr hless) (by positivity)
  have hl : 0 < lowerPole K G T massSq q := by
    unfold lowerPole
    linarith
  have hmore : 0 ≤ Real.sqrt (poleDiscriminant K G T massSq q) / K :=
    div_nonneg hnon hK.le
  have hgap : upperPole K G T massSq q -
       lowerPole K G T massSq q =
      Real.sqrt (poleDiscriminant K G T massSq q) / K := by
    unfold upperPole lowerPole
    field_simp [ne_of_gt hK]
    ring
  constructor
  · exact hl
  · linarith

/-- The unchanged S004 static phase channel has an exactly gapless pole. -/
theorem phase_pole_gapless_at_zero (K G T : ℝ)
    (hK : 0 < K) :
    lowerPole K G T 0 0 = 0 := by
  have hd : poleDiscriminant K G T 0 0 = 1 := by
    unfold poleDiscriminant
    ring
  unfold lowerPole
  rw [hd, Real.sqrt_one]
  ring

/-- A strictly positive amplitude mass separates the amplitude from the
    gapless phase already at zero frequency and momentum. This defeats a
    mode-by-mode identification with two identical copies of a single real
    spatial-only effective dispersion without additional mode selection. -/
theorem static_amplitude_phase_kernel_inequivalent
    (K G T massSq : ℝ) (hm : massSq ≠ 0) :
    inversePropagator K G T massSq 0 0 ≠
      inversePropagator K G T 0 0 0 := by
  unfold inversePropagator
  simpa using (neg_ne_zero.mpr hm)

/-- The monotone physical spatial stiffness cannot equal the independent
    competing-gradient symbol at all q if the latter has negative linear
    stiffness -a with a>0 and the former retains +q. -/
theorem no_identical_static_spatial_symbol
    (K G T massSq mu a b : ℝ)
    (ha : 0 < a)
    (hall : ∀ q : ℝ,
      staticStiffness K G T massSq q = mu - a * q + b * q ^ 2) :
    False := by
  have h0 := hall 0
  have hp := hall 1
  have hn := hall (-1)
  unfold staticStiffness at h0 hp hn
  norm_num at h0 hp hn
  linarith

/-- Given the explicit positive-tangent-at-zero criterion and A≥0,
    the F2 exact minimizer is globally minimizing on ALL q≥0 inside the
    real-pole region. The tangent criterion is written as a real inequality,
    not an unannounced physics assumption. -/
theorem exact_minimum_global_on_nonnegative_real_window
    (K G T massSq q : ℝ)
    (hK : 0 < K) (hB : 0 < curvatureGap K G T)
    (hN : 0 ≤ discrVertex K G T massSq)
    (hA : 0 ≤ linearQ K G)
    (hzero : 0 <
       centerRoot K G T massSq -
        linearQ K G * candidateQ K G T massSq)
    (hq : 0 ≤ q)
    (hdq : 0 ≤ poleDiscriminant K G T massSq q) :
    lowerPole K G T massSq (candidateQ K G T massSq) ≤
      lowerPole K G T massSq q := by
  have hAq : 0 ≤ linearQ K G * q := mul_nonneg hA hq
  have htangent :
      0 < Real.sqrt (poleDiscriminant K G T massSq
            (candidateQ K G T massSq)) +
            linearQ K G * (q - candidateQ K G T massSq) := by
    rw [candidateQ_discriminant K G T massSq hB hN]
    rw [Real.sqrt_sq (centerRoot_nonneg K G T massSq)]
    nlinarith
  exact exact_candidate_is_minimum_in_window
    K G T massSq q hK hB hN hdq htangent

theorem exact_minimum_unique_on_nonnegative_real_window
    (K G T massSq q : ℝ)
    (hK : 0 < K) (hB : 0 < curvatureGap K G T)
    (hN : 0 ≤ discrVertex K G T massSq)
    (hA : 0 ≤ linearQ K G)
    (hzero : 0 <
       centerRoot K G T massSq -
        linearQ K G * candidateQ K G T massSq)
    (hq : 0 ≤ q)
    (hdq : 0 ≤ poleDiscriminant K G T massSq q)
    (heq : lowerPole K G T massSq q =
      lowerPole K G T massSq (candidateQ K G T massSq)) :
    q = candidateQ K G T massSq := by
  have hAq : 0 ≤ linearQ K G * q := mul_nonneg hA hq
  have htangent :
      0 < Real.sqrt (poleDiscriminant K G T massSq
           (candidateQ K G T massSq)) +
           linearQ K G * (q - candidateQ K G T massSq) := by
    rw [candidateQ_discriminant K G T massSq hB hN]
    rw [Real.sqrt_sq (centerRoot_nonneg K G T massSq)]
    nlinarith
  exact exact_candidate_unique_in_window K G T massSq q
    hK hB hN hdq htangent heq

end ReleaseProofs.R18Admissibility
