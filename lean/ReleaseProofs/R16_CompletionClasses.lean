import Mathlib
import ReleaseProofs.R16_GhostObstruction
import ReleaseProofs.R18_CanonicalObstruction

/-!
Issue #16: a delimited failure class and independent coercive spatial action.
S004's nonzero higher-time Hessian remains nondegenerate under invertible
point transformations. The separate spatial action is not a S004 completion.
-/

namespace ReleaseProofs.R16Completion

def transformedHessianDet (coeff a b c d : ℝ) : ℝ :=
  (coeff * (a ^ 2 + c ^ 2)) * (coeff * (b ^ 2 + d ^ 2)) -
    (coeff * (a*b+c*d)) ^ 2

theorem hessian_transform_identity (coeff a b c d : ℝ) :
    transformedHessianDet coeff a b c d =
      coeff ^ 2 * (a*d-b*c) ^ 2 := by
  unfold transformedHessianDet
  ring

theorem invertible_point_change_cannot_degenerate
    (coeff a b c d : ℝ)
    (hcoeff : coeff ≠ 0) (hjac : a*d-b*c ≠ 0) :
    0 < transformedHessianDet coeff a b c d := by
  rw [hessian_transform_identity]
  exact mul_pos (sq_pos_of_ne_zero hcoeff) (sq_pos_of_ne_zero hjac)

theorem acceleration_kernel_trivial (coeff x y : ℝ)
    (hc : coeff ≠ 0)
    (hx : coeff * x = 0)
    (hy : coeff * y = 0) : x = 0 ∧ y = 0 := by
  constructor
  · exact (mul_eq_zero.mp hx).resolve_left hc
  · exact (mul_eq_zero.mp hy).resolve_left hc

/-!
Separate competing-gradient action:
L = |psi_t|² - mu |psi|² + a |grad psi|²
    - b |Delta psi|² - lambda |psi|⁴/2
with mu>0, a>0, b>0, lambda>=0 and 4*b*mu>a².
-/

def competingFrequencySq (mu a b q : ℝ) : ℝ :=
  mu - a * q + b * q ^ 2

noncomputable def competingQStar (a b : ℝ) : ℝ := a / (2*b)

theorem competing_square_completion (mu a b q : ℝ) :
    4*b*competingFrequencySq mu a b q =
      (2*b*q-a)^2 + (4*b*mu-a^2) := by
  unfold competingFrequencySq
  ring

theorem competing_positive_frequency (mu a b q : ℝ)
    (hb : 0 < b) (hgap : a ^ 2 < 4*b*mu) :
    0 < competingFrequencySq mu a b q := by
  have hs : 0 ≤ (2*b*q-a)^2 := sq_nonneg _
  have hident := competing_square_completion mu a b q
  have hprod : 0 < 4*b*competingFrequencySq mu a b q := by
    linarith
  by_contra hh
  have hn : competingFrequencySq mu a b q ≤ 0 := le_of_not_gt hh
  have hb4 : 0 ≤ 4*b := by positivity
  have hnp := mul_nonpos_of_nonneg_of_nonpos hb4 hn
  linarith

theorem competing_minimality_identity (mu a b q : ℝ)
    (hb : b ≠ 0) :
    competingFrequencySq mu a b q -
      competingFrequencySq mu a b (competingQStar a b) =
      b * (q-competingQStar a b)^2 := by
  unfold competingFrequencySq competingQStar
  field_simp [hb]
  ring

theorem competing_unique_minimum (mu a b q : ℝ)
    (hb : 0 < b)
    (heq : competingFrequencySq mu a b q =
      competingFrequencySq mu a b (competingQStar a b)) :
    q = competingQStar a b := by
  have hid := competing_minimality_identity mu a b q (ne_of_gt hb)
  rw [heq] at hid
  have hzero : b * (q-competingQStar a b)^2 = 0 := by
    linarith
  have hz : (q-competingQStar a b)^2 = 0 :=
    (mul_eq_zero.mp hzero).resolve_left (ne_of_gt hb)
  nlinarith

theorem finite_preferred_q (a b : ℝ)
    (ha : 0 < a) (hb : 0 < b) :
    0 < competingQStar a b := by
  unfold competingQStar
  exact div_pos ha (by positivity)

noncomputable def competingModeHamiltonian (mu a b q x v : ℝ) : ℝ :=
  (v^2 + competingFrequencySq mu a b q*x^2)/2

theorem competing_mode_energy_nonneg (mu a b q x v : ℝ)
    (hb : 0 < b) (hgap : a^2 < 4*b*mu) :
    0 ≤ competingModeHamiltonian mu a b q x v := by
  have hw := (competing_positive_frequency mu a b q hb hgap).le
  have hp : 0 ≤ competingFrequencySq mu a b q*x^2 :=
    mul_nonneg hw (sq_nonneg x)
  unfold competingModeHamiltonian
  nlinarith [sq_nonneg v]

/-- A positive-frequency single-mode defocusing equilibrium has only x=0.
The PDE proof instead uses integration by parts/Plancherel. -/
theorem defocusing_mode_no_nonzero_stationary (mu a b q lambda x : ℝ)
    (hb : 0 < b) (hgap : a^2 < 4*b*mu) (hlambda : 0 ≤ lambda)
    (heq : competingFrequencySq mu a b q*x + lambda*x^3=0) :
    x=0 := by
  by_contra hx
  have hx2 : 0 < x^2 := sq_pos_of_ne_zero hx
  have hw := competing_positive_frequency mu a b q hb hgap
  have hp : 0 < competingFrequencySq mu a b q*x^2 :=
    mul_pos hw hx2
  have hn : 0 ≤ lambda*(x^2)^2 :=
    mul_nonneg hlambda (sq_nonneg _)
  have hmul := congrArg (fun t : ℝ => t * x) heq
  have hz : competingFrequencySq mu a b q*x^2 +
      lambda*(x^2)^2 = 0 := by
    nlinarith [hmul]
  linarith

end ReleaseProofs.R16Completion
