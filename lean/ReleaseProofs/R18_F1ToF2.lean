import Mathlib
import ReleaseProofs.R18_Admissibility

/-!
# F1 ==> exact F2: sufficient physical sector and full real-window minimum

These lemmas turn the analytical slope condition G > (2K+G)*sqrt(d(0))
into the explicit positivity and tangent hypotheses required by the existing
exact (untruncated) minimizer. They assume the real-pole gap at q=0 is
strictly positive and A=2K+G is nonnegative. They do not address the ghost.
-/

namespace ReleaseProofs.R18F1ToF2

open ReleaseProofs.R18
open ReleaseProofs.R18Classification
open ReleaseProofs.R18ExactMinimum
open ReleaseProofs.R18Admissibility

private theorem f1_square_comparison
    (A B G s c : ℝ)
    (hB : 0 < B) (hA : 0 ≤ A) (hs : 0 ≤ s) (hc : 0 ≤ c)
    (hcenter : (A ^ 2 + B) * c ^ 2 = G ^ 2 + B * s ^ 2)
    (hF1 : A * s < G) : A * c < G := by
  have hAs : 0 ≤ A * s := mul_nonneg hA hs
  have hG : 0 < G := lt_of_le_of_lt hAs hF1
  have hsquares : (A * s) ^ 2 < G ^ 2 := by nlinarith
  have hgap : 0 < G ^ 2 - A ^ 2 * s ^ 2 := by
    nlinarith [hsquares]
  have hiden :
      (A ^ 2 + B) * (G ^ 2 - A ^ 2 * c ^ 2) =
        B * (G ^ 2 - A ^ 2 * s ^ 2) := by
    calc
      _ = (A ^ 2 + B) * G ^ 2 -
            A ^ 2 * ((A ^ 2 + B) * c ^ 2) := by ring
      _ = (A ^ 2 + B) * G ^ 2 -
            A ^ 2 * (G ^ 2 + B * s ^ 2) := by rw [hcenter]
      _ = _ := by ring
  have hden : 0 < A ^ 2 + B := by nlinarith [sq_nonneg A]
  have hpositive : 0 < B * (G ^ 2 - A ^ 2 * s ^ 2) :=
    mul_pos hB hgap
  have hgap2 : 0 < G ^ 2 - A ^ 2 * c ^ 2 := by
    by_contra hh
    have hn : G ^ 2 - A ^ 2 * c ^ 2 ≤ 0 := le_of_not_gt hh
    have hm := mul_nonpos_of_nonneg_of_nonpos hden.le hn
    linarith
  have hAc : 0 ≤ A * c := mul_nonneg hA hc
  nlinarith

private theorem tangent_at_zero_from_strict_gap
    (A B G s c : ℝ)
    (hB : 0 < B) (hA : 0 ≤ A) (hG : 0 ≤ G)
    (hs : 0 < s) (hc : 0 ≤ c)
    (hcenter : (A ^ 2 + B) * c ^ 2 = G ^ 2 + B * s ^ 2) :
    0 < (A ^ 2 + B) * c - A * G := by
  have hden : 0 < A ^ 2 + B := by nlinarith [sq_nonneg A]
  have hinner : 0 < G ^ 2 + (A ^ 2 + B) * s ^ 2 := by
    have hp := mul_pos hden (sq_pos_of_pos hs)
    nlinarith [sq_nonneg G]
  have hiden :
      ((A ^ 2 + B) * c) ^ 2 - (A * G) ^ 2 =
        B * (G ^ 2 + (A ^ 2 + B) * s ^ 2) := by
    calc
      _ = (A ^ 2 + B) * ((A ^ 2 + B) * c ^ 2) -
            (A * G) ^ 2 := by ring
      _ = (A ^ 2 + B) * (G ^ 2 + B * s ^ 2) -
            (A * G) ^ 2 := by rw [hcenter]
      _ = _ := by ring
  have hpositive := mul_pos hB hinner
  have hleft : 0 ≤ (A ^ 2 + B) * c := mul_nonneg hden.le hc
  have hright : 0 ≤ A * G := mul_nonneg hA hG
  nlinarith

/-- The physically stated F1 negative-initial-slope inequality guarantees
    a positive exact F2 wave number, a nondegenerate pole at that candidate,
    and one global minimizer across q>=0 intersect the real-pole region.
    No independent physical length or healthy upper-branch statement follows.
-/
theorem f1_implies_exact_positive_global_minimum
    (K G T massSq : ℝ)
    (hK : 0 < K) (hB : 0 < curvatureGap K G T)
    (hd0 : 0 < 1 - 4 * K * massSq)
    (hA : 0 ≤ linearQ K G)
    (hF1 : linearQ K G * Real.sqrt (1 - 4 * K * massSq) < G) :
    0 < candidateQ K G T massSq ∧
      (∀ q : ℝ, 0 ≤ q →
        0 ≤ poleDiscriminant K G T massSq q →
          lowerPole K G T massSq (candidateQ K G T massSq) ≤
            lowerPole K G T massSq q) ∧
      (∀ q : ℝ, 0 ≤ q →
        0 ≤ poleDiscriminant K G T massSq q →
          lowerPole K G T massSq q =
            lowerPole K G T massSq (candidateQ K G T massSq) →
          q = candidateQ K G T massSq) := by
  let A := linearQ K G
  let B := curvatureGap K G T
  let s := Real.sqrt (1 - 4 * K * massSq)
  let c := centerRoot K G T massSq
  have hs : 0 < s := Real.sqrt_pos.2 hd0
  have hc : 0 ≤ c := centerRoot_nonneg K G T massSq
  have hG : 0 < G := by
    have hAs : 0 ≤ A * s := mul_nonneg hA hs.le
    exact lt_of_le_of_lt hAs hF1
  have hN : 0 ≤ discrVertex K G T massSq := by
    have hBD : 0 < curvatureGap K G T * (1 - 4 * K * massSq) :=
      mul_pos hB hd0
    unfold discrVertex
    nlinarith [sq_nonneg G]
  have hcenter :
      (A ^ 2 + B) * c ^ 2 = G ^ 2 + B * s ^ 2 := by
    have hcv := centerRoot_sq K G T massSq hB hN
    have hs2 := Real.sq_sqrt (le_of_lt hd0)
    change (linearQ K G ^ 2 + curvatureGap K G T) *
       centerRoot K G T massSq ^ 2 =
         G ^ 2 + curvatureGap K G T * s ^ 2
    rw [hs2]
    simpa [discrVertex] using hcv
  have hCandidate :
      A * c < G :=
    f1_square_comparison A B G s c hB hA hs.le hc hcenter hF1
  have hqStar : 0 < candidateQ K G T massSq :=
    candidateQ_positive K G T massSq hB hCandidate
  have htangentSquare :=
    tangent_at_zero_from_strict_gap A B G s c
      hB hA hG.le hs hc hcenter
  have hzero :
      0 < centerRoot K G T massSq -
        linearQ K G * candidateQ K G T massSq := by
    have hrat :
        c - A * candidateQ K G T massSq =
          ((A ^ 2 + B) * c - A * G) / B := by
      dsimp [A, B, c]
      unfold candidateQ
      field_simp [ne_of_gt hB]
      ring
    rw [hrat]
    exact div_pos htangentSquare hB
  constructor
  · exact hqStar
  · constructor
    · intro q hq hdq
      exact exact_minimum_global_on_nonnegative_real_window
        K G T massSq q hK hB hN hA hzero hq hdq
    · intro q hq hdq heq
      exact exact_minimum_unique_on_nonnegative_real_window
        K G T massSq q hK hB hN hA hzero hq hdq heq


/-- The F1 amplitude regime with m²≥0 has a genuinely *positive*
    oscillatory squared frequency at the exact minimizer. It is not an
    exponentially growing finite-q mode; the upper pole is still a ghost. -/
theorem f1_minimum_has_positive_oscillatory_frequency
    (K G T massSq : ℝ)
    (hK : 0 < K) (hT : 0 < T)
    (hB : 0 < curvatureGap K G T)
    (hm : 0 ≤ massSq)
    (hd0 : 0 < 1 - 4 * K * massSq)
    (hA : 0 ≤ linearQ K G)
    (hF1 : linearQ K G * Real.sqrt (1 - 4 * K * massSq) < G) :
    0 < candidateQ K G T massSq ∧
      0 < lowerPole K G T massSq (candidateQ K G T massSq) ∧
      0 < upperPole K G T massSq (candidateQ K G T massSq) := by
  have hf := f1_implies_exact_positive_global_minimum
    K G T massSq hK hB hd0 hA hF1
  have hN : 0 ≤ discrVertex K G T massSq := by
    have hBD : 0 < curvatureGap K G T * (1 - 4 * K * massSq) :=
      mul_pos hB hd0
    unfold discrVertex
    nlinarith [sq_nonneg G]
  have hdisc : 0 ≤ poleDiscriminant K G T massSq
      (candidateQ K G T massSq) :=
    candidateQ_discriminant_nonneg K G T massSq hB hN
  have hpositive := ReleaseProofs.R18Admissibility.real_poles_non_tachyonic_all_G
    K G T massSq (candidateQ K G T massSq)
    hK hT hB hm hf.1 hdisc
  exact ⟨hf.1, hpositive.1, hpositive.2⟩

end ReleaseProofs.R18F1ToF2
