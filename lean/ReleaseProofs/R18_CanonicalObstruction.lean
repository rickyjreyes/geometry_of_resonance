import Mathlib
import ReleaseProofs.R18_FiniteScale

/-!
# Issue 18: static S004 selection obstruction

Strict assumptions are those of the literal corrected S004 action evaluated at a
nonzero *static homogeneous* on-shell background. K,G,T include F(u0)>0.
The full rotating-carrier Hessian is separately recorded as Q1–Q5.
No statement here replaces the fourth-time-order parent action.
-/

namespace ReleaseProofs.R18Canonical

open ReleaseProofs.R18

def staticStiffness (K G T massSq q : ℝ) : ℝ :=
  massSq + q + (K + G + T) * q ^ 2

/-- Positive curvature form implies positive *static* spatial stiffness. -/
theorem spatial_coefficient_pos (K G T : ℝ)
    (hK : 0 < K) (hB : 0 < 4 * K * T - G ^ 2) :
    0 < K + G + T := by
  have hs : 0 ≤ (2 * K + G) ^ 2 := sq_nonneg _
  have hp : 0 < 4 * K * (K + G + T) := by nlinarith
  by_contra hh
  have hh' : K + G + T ≤ 0 := le_of_not_gt hh
  have hk : 0 ≤ 4 * K := by positivity
  have hn := mul_nonpos_of_nonneg_of_nonpos hk hh'
  linarith

theorem static_stiffness_positive (K G T massSq q : ℝ)
    (hK : 0 < K) (hB : 0 < 4 * K * T - G ^ 2)
    (hm : 0 ≤ massSq) (hq : 0 < q) :
    0 < staticStiffness K G T massSq q := by
  have hc : 0 < K + G + T := spatial_coefficient_pos K G T hK hB
  have ht : 0 ≤ (K + G + T) * q ^ 2 :=
    mul_nonneg hc.le (sq_nonneg q)
  unfold staticStiffness
  linarith

/-- The exact static inverse propagator never vanishes at nonzero q. -/
theorem no_static_finite_q_onset (K G T massSq q : ℝ)
    (hK : 0 < K) (hB : 0 < 4 * K * T - G ^ 2)
    (hm : 0 ≤ massSq) (hq : 0 < q) :
    inversePropagator K G T massSq 0 q < 0 := by
  have hp := static_stiffness_positive K G T massSq q hK hB hm hq
  have hi : inversePropagator K G T massSq 0 q =
      -staticStiffness K G T massSq q := by
    unfold inversePropagator staticStiffness
    ring
  linarith

/-- The phase mode has no positive-q static onset, including massSq=0. -/
theorem phase_no_finite_q_onset (K G T q : ℝ)
    (hK : 0 < K) (hB : 0 < 4 * K * T - G ^ 2)
    (hq : 0 < q) :
    inversePropagator K G T 0 0 q < 0 :=
  no_static_finite_q_onset K G T 0 q hK hB (le_refl 0) hq

/-- On the real-pole interval with G>=0 and nonnegative amplitude mass,
    the lower squared-frequency root is strictly positive at every q>0. -/
theorem lower_real_pole_positive (K G T massSq q : ℝ)
    (hK : 0 < K) (hT : 0 < T) (hG : 0 ≤ G)
    (hm : 0 ≤ massSq) (hq : 0 < q)
    (hd : 0 ≤ poleDiscriminant K G T massSq q) :
    0 < lowerPole K G T massSq q := by
  let d := poleDiscriminant K G T massSq q
  have hgd : 0 ≤ G * q := mul_nonneg hG hq.le
  have hbase : 0 < 1 + G * q := by linarith
  have hq2 : 0 < q ^ 2 := sq_pos_of_pos hq
  have hmass : 0 < massSq + T * q ^ 2 := by
    have hp := mul_pos hT hq2
    linarith
  have hterm : 0 < 4 * K * (massSq + T * q ^ 2) := by
    have hh : 0 < 4 * K := by positivity
    exact mul_pos hh hmass
  have hs : (Real.sqrt d) ^ 2 = d := Real.sq_sqrt hd
  have hsnon : 0 ≤ Real.sqrt d := Real.sqrt_nonneg _
  have hdiff : (Real.sqrt d) ^ 2 < (1 + G * q) ^ 2 := by
    rw [hs]
    dsimp [d, poleDiscriminant]
    nlinarith [hterm]
  have hless : Real.sqrt d < 1 + G * q := by
    nlinarith [hdiff]
  have hquot : 0 < (1 + G * q - Real.sqrt d) / (2 * K) :=
    div_pos (sub_pos.mpr hless) (by positivity)
  unfold lowerPole
  linarith

/-- A positive low squared frequency is not a growing exponential mode. -/
theorem real_poles_not_tachyonic (K G T massSq q : ℝ)
    (hK : 0 < K) (hT : 0 < T) (hG : 0 ≤ G)
    (hm : 0 ≤ massSq) (hq : 0 < q)
    (hd : 0 ≤ poleDiscriminant K G T massSq q) :
    0 < lowerPole K G T massSq q ∧
    0 < upperPole K G T massSq q := by
  have hl := lower_real_pole_positive K G T massSq q hK hT hG hm hq hd
  have hr : 0 ≤ Real.sqrt (poleDiscriminant K G T massSq q) :=
    Real.sqrt_nonneg _
  have hk2 : 0 < 2 * K := by positivity
  have hdelta : 0 ≤ Real.sqrt (poleDiscriminant K G T massSq q) / K :=
    div_nonneg hr hK.le
  have hdiff : upperPole K G T massSq q - lowerPole K G T massSq q =
      Real.sqrt (poleDiscriminant K G T massSq q) / K := by
    unfold upperPole lowerPole
    field_simp [ne_of_gt hK]
    ring
  constructor
  · exact hl
  · linarith

/-- The original time-derivative ghost remains with any nonzero positive
    higher-derivative weight, including a modified weight W(0)>0. -/
theorem nonzero_weight_has_acceleration_hessian
    (weight kappa : ℝ) (hw : 0 < weight) (hk : 0 < kappa) :
    0 < 4 * (weight * kappa) ^ 2 := by
  have hp : 0 < weight * kappa := mul_pos hw hk
  positivity

end ReleaseProofs.R18Canonical
