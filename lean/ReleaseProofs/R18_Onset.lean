import Mathlib
import ReleaseProofs.R18_Boundaries

/-!
# Strong static finite-q onset obstruction for either sign of amplitude mass

With K>0, B=4KT-G²>0, a zero-frequency crossing at q>0
forces massSq<0. But that same mass parameter already makes the q=0
lower real squared-frequency pole negative (tachyonic).
Thus NO *primary* finite-q static zero-frequency onset is possible
in this entire strict static homogeneous sector, including m²<0.
This is not a no-go for rotating/inhomogeneous backgrounds or modified actions.
-/

namespace ReleaseProofs.R18Onset

open ReleaseProofs.R18
open ReleaseProofs.R18Canonical

theorem finite_q_zero_requires_negative_mass
    (K G T massSq q : ℝ)
    (hK : 0 < K)
    (hB : 0 < curvatureGap K G T)
    (hq : 0 < q)
    (hzero : inversePropagator K G T massSq 0 q = 0) :
    massSq < 0 := by
  have hc : 0 < K + G + T := by
    apply spatial_coefficient_pos K G T hK
    simpa [curvatureGap] using hB
  have hq2 : 0 ≤ q ^ 2 := sq_nonneg q
  have ht : 0 ≤ (K + G + T) * q ^ 2 :=
    mul_nonneg hc.le hq2
  have hi : inversePropagator K G T massSq 0 q =
      -(massSq + q + (K + G + T) * q ^ 2) := by
    unfold inversePropagator
    ring
  rw [hi] at hzero
  linarith

theorem negative_amplitude_mass_tachyonic_at_q_zero
    (K G T massSq : ℝ)
    (hK : 0 < K)
    (hm : massSq < 0) :
    lowerPole K G T massSq 0 < 0 := by
  have hkm : 0 < (4 * K) * (-massSq) :=
    mul_pos (by positivity) (neg_pos.mpr hm)
  have hd : 1 < poleDiscriminant K G T massSq 0 := by
    unfold poleDiscriminant
    nlinarith
  have hsnon : 0 ≤ Real.sqrt (poleDiscriminant K G T massSq 0) :=
    Real.sqrt_nonneg _
  have hs2 : Real.sqrt (poleDiscriminant K G T massSq 0) ^ 2 =
      poleDiscriminant K G T massSq 0 :=
    Real.sq_sqrt (by linarith)
  have hs : 1 < Real.sqrt (poleDiscriminant K G T massSq 0) := by
    nlinarith
  have hneg : (1 - Real.sqrt (poleDiscriminant K G T massSq 0)) /
      (2 * K) < 0 :=
    div_neg_of_neg_of_pos (sub_neg.mpr hs) (by positivity)
  simpa [lowerPole] using hneg

/-- A positive finite-q *static* onset, if present, cannot be the first
    unstable squared-frequency mode: the homogeneous q=0 mode is tachyonic.
    The hypothesis is an exact root of the original S004 Q6 polynomial. -/
theorem no_primary_finite_q_static_onset
    (K G T massSq q : ℝ)
    (hK : 0 < K)
    (hB : 0 < curvatureGap K G T)
    (hq : 0 < q)
    (hzero : inversePropagator K G T massSq 0 q = 0) :
    lowerPole K G T massSq 0 < 0 := by
  exact negative_amplitude_mass_tachyonic_at_q_zero K G T massSq hK
    (finite_q_zero_requires_negative_mass K G T massSq q hK hB hq hzero)

end ReleaseProofs.R18Onset
