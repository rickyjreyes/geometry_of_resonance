import Mathlib
import ReleaseProofs.R18_F1ToF2

/-!
# Exact algebraic boundary classification for issue 18

The strict S004 curvature action uses K>0, T>0 and 4KT-G²>0.
These identities identify how the static polynomial changes on the
BOUNDARY of that region. They do NOT assert health on the boundary,
nor take a singular limit of a ghost pole as K tends to zero.
-/

namespace ReleaseProofs.R18Boundaries

open ReleaseProofs.R18
open ReleaseProofs.R18Classification
open ReleaseProofs.R18ExactMinimum

theorem zero_curvature_gap_discriminant
    (K G T massSq q : ℝ)
    (hB : curvatureGap K G T = 0) :
    poleDiscriminant K G T massSq q =
      1 - 4 * K * massSq + 2 * G * q := by
  rw [discriminant_expansion, hB]
  ring

theorem zero_higher_time_inverse_propagator
    (G T massSq z q : ℝ) :
    inversePropagator 0 G T massSq z q =
      (1 + G * q) * (z - q) - massSq - T * q ^ 2 := by
  unfold inversePropagator
  ring

theorem zero_background_linear_kernel
    (massSq z q : ℝ) :
    inversePropagator 0 0 0 massSq z q = z - q - massSq := by
  unfold inversePropagator
  ring

theorem zero_background_frequency_monotone
    (massSq q₁ q₂ : ℝ) (h : q₁ < q₂) :
    massSq + q₁ < massSq + q₂ := by
  linarith

/-- A purely spatial S004 boundary (K=G=0) with nonnegative bending
    T retains a strictly increasing frequency on every nonnegative q. -/
theorem spatial_boundary_no_positive_q_minimum
    (T massSq q₁ q₂ : ℝ)
    (hT : 0 ≤ T) (hq₁ : 0 ≤ q₁) (h12 : q₁ < q₂) :
    massSq + q₁ + T * q₁ ^ 2 <
      massSq + q₂ + T * q₂ ^ 2 := by
  have hq₂ : 0 ≤ q₂ := le_trans hq₁ h12.le
  have hdiff : 0 < q₂ - q₁ := sub_pos.mpr h12
  have hsum : 0 ≤ q₂ + q₁ := add_nonneg hq₂ hq₁
  have hsq : 0 ≤ q₂ ^ 2 - q₁ ^ 2 := by
    nlinarith [mul_nonneg hdiff.le hsum]
  have hterm : 0 ≤ T * (q₂ ^ 2 - q₁ ^ 2) :=
    mul_nonneg hT hsq
  nlinarith

/-- The sign of the F2 candidate is exactly the sign of its numerator,
    and no special sign assumption on G is hidden in the formula. -/
theorem candidate_finite_q_positive_iff
    (K G T massSq : ℝ) (hB : 0 < curvatureGap K G T) :
    0 < candidateQ K G T massSq ↔
      linearQ K G * centerRoot K G T massSq < G := by
  have hi : curvatureGap K G T * candidateQ K G T massSq =
      G - linearQ K G * centerRoot K G T massSq := by
    have hs := candidateQ_shift K G T massSq hB
    linarith
  constructor
  · intro h
    have hm : 0 < curvatureGap K G T * candidateQ K G T massSq :=
      mul_pos hB h
    linarith
  · intro h
    exact candidateQ_positive K G T massSq hB h

theorem candidate_finite_q_nonpositive_iff
    (K G T massSq : ℝ) (hB : 0 < curvatureGap K G T) :
    candidateQ K G T massSq ≤ 0 ↔
      G ≤ linearQ K G * centerRoot K G T massSq := by
  constructor
  · intro h
    by_contra hn
    have hp : linearQ K G * centerRoot K G T massSq < G :=
      lt_of_not_ge hn
    have hpos := (candidate_finite_q_positive_iff K G T massSq hB).mpr hp
    linarith
  · intro h
    by_contra hn
    have hp : 0 < candidateQ K G T massSq := lt_of_not_ge hn
    have hcrit := (candidate_finite_q_positive_iff K G T massSq hB).mp hp
    linarith

end ReleaseProofs.R18Boundaries
