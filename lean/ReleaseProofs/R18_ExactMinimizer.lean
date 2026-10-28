import Mathlib
import ReleaseProofs.R18_Classification

/-!
# The exact F2 candidate, proved from the untruncated S004 Q6 polynomial

This is an oscillatory-frequency minimizer on a real-pole tangent-positive
window, not a growing-mode instability or a physical wavelength prediction.
-/

namespace ReleaseProofs.R18ExactMinimum

open ReleaseProofs.R18
open ReleaseProofs.R18Classification

def linearQ (K G : ℝ) : ℝ := 2 * K + G

noncomputable def centerRoot (K G T massSq : ℝ) : ℝ :=
  Real.sqrt (discrVertex K G T massSq /
    (linearQ K G ^ 2 + curvatureGap K G T))

noncomputable def candidateQ (K G T massSq : ℝ) : ℝ :=
  (G - linearQ K G * centerRoot K G T massSq) /
    curvatureGap K G T

theorem centerRoot_nonneg (K G T massSq : ℝ) :
    0 ≤ centerRoot K G T massSq := by
  unfold centerRoot
  exact Real.sqrt_nonneg _

theorem centerRoot_sq (K G T massSq : ℝ)
    (hB : 0 < curvatureGap K G T)
    (hN : 0 ≤ discrVertex K G T massSq) :
    (linearQ K G ^ 2 + curvatureGap K G T) *
      centerRoot K G T massSq ^ 2 = discrVertex K G T massSq := by
  have hden : 0 < linearQ K G ^ 2 + curvatureGap K G T := by
    nlinarith [sq_nonneg (linearQ K G)]
  have hdiv : 0 ≤ discrVertex K G T massSq /
      (linearQ K G ^ 2 + curvatureGap K G T) :=
    div_nonneg hN hden.le
  have hs := Real.sq_sqrt hdiv
  unfold centerRoot
  rw [hs]
  field_simp [ne_of_gt hden]

theorem candidateQ_shift (K G T massSq : ℝ)
    (hB : 0 < curvatureGap K G T) :
    curvatureGap K G T * candidateQ K G T massSq - G =
      -linearQ K G * centerRoot K G T massSq := by
  unfold candidateQ
  field_simp [ne_of_gt hB]
  ring

theorem candidateQ_discriminant (K G T massSq : ℝ)
    (hB : 0 < curvatureGap K G T)
    (hN : 0 ≤ discrVertex K G T massSq) :
    poleDiscriminant K G T massSq (candidateQ K G T massSq) =
      centerRoot K G T massSq ^ 2 := by
  have hc := centerRoot_sq K G T massSq hB hN
  have hs := candidateQ_shift K G T massSq hB
  have hi := discriminant_vertex_identity K G T massSq
    (candidateQ K G T massSq)
  rw [hs] at hi
  have hz : curvatureGap K G T *
      (poleDiscriminant K G T massSq (candidateQ K G T massSq) -
        centerRoot K G T massSq ^ 2) = 0 := by
    nlinarith [hi, hc]
  exact (mul_eq_zero.mp hz).resolve_left (ne_of_gt hB)

theorem candidateQ_discriminant_nonneg (K G T massSq : ℝ)
    (hB : 0 < curvatureGap K G T)
    (hN : 0 ≤ discrVertex K G T massSq) :
    0 ≤ poleDiscriminant K G T massSq (candidateQ K G T massSq) := by
  rw [candidateQ_discriminant K G T massSq hB hN]
  exact sq_nonneg _

theorem candidateQ_stationary (K G T massSq : ℝ)
    (hB : 0 < curvatureGap K G T)
    (hN : 0 ≤ discrVertex K G T massSq) :
    G - curvatureGap K G T * candidateQ K G T massSq =
      linearQ K G *
        Real.sqrt (poleDiscriminant K G T massSq
          (candidateQ K G T massSq)) := by
  have hd := candidateQ_discriminant K G T massSq hB hN
  have hc : 0 ≤ centerRoot K G T massSq := centerRoot_nonneg _ _ _ _
  have hroot : Real.sqrt
      (poleDiscriminant K G T massSq (candidateQ K G T massSq)) =
        centerRoot K G T massSq := by
    rw [hd]
    exact Real.sqrt_sq hc
  rw [hroot]
  have hs := candidateQ_shift K G T massSq hB
  linarith

/-- This explicit condition is equivalent to the F1 positive-q selector
    inequality under the additional positive-mass regime in the source. -/
theorem candidateQ_positive (K G T massSq : ℝ)
    (hB : 0 < curvatureGap K G T)
    (hpositive : linearQ K G * centerRoot K G T massSq < G) :
    0 < candidateQ K G T massSq := by
  unfold candidateQ
  exact div_pos (sub_pos.mpr hpositive) hB

theorem exact_candidate_is_minimum_in_window (K G T massSq q : ℝ)
    (hK : 0 < K) (hB : 0 < curvatureGap K G T)
    (hN : 0 ≤ discrVertex K G T massSq)
    (hdq : 0 ≤ poleDiscriminant K G T massSq q)
    (hTangent : 0 <
      Real.sqrt (poleDiscriminant K G T massSq (candidateQ K G T massSq)) +
        linearQ K G * (q - candidateQ K G T massSq)) :
    lowerPole K G T massSq (candidateQ K G T massSq) ≤
      lowerPole K G T massSq q := by
  exact lower_pole_minimum_on_tangent_window
    K G T massSq q (candidateQ K G T massSq) (linearQ K G)
    hK hB rfl
    (candidateQ_stationary K G T massSq hB hN)
    (candidateQ_discriminant_nonneg K G T massSq hB hN) hdq hTangent

theorem exact_candidate_unique_in_window (K G T massSq q : ℝ)
    (hK : 0 < K) (hB : 0 < curvatureGap K G T)
    (hN : 0 ≤ discrVertex K G T massSq)
    (hdq : 0 ≤ poleDiscriminant K G T massSq q)
    (hTangent : 0 <
      Real.sqrt (poleDiscriminant K G T massSq (candidateQ K G T massSq)) +
        linearQ K G * (q - candidateQ K G T massSq))
    (heq : lowerPole K G T massSq q =
      lowerPole K G T massSq (candidateQ K G T massSq)) :
    q = candidateQ K G T massSq := by
  exact lower_pole_unique_on_tangent_window
    K G T massSq q (candidateQ K G T massSq) (linearQ K G)
    hK hB rfl
    (candidateQ_stationary K G T massSq hB hN)
    (candidateQ_discriminant_nonneg K G T massSq hB hN) hdq hTangent heq

end ReleaseProofs.R18ExactMinimum
