import Mathlib
import ReleaseProofs.R16_CompletionClasses

/-!
# Issue 16: true two-field acceleration determinant and mode-energy bounds

The acceleration-Hessian entries are the fixed-flat S004 values already
derived from the corrected action in the foundational audit. The finite
Fourier-mode energy statements concern the *distinct* spatial-only competing
gradient model. The abstract final theorem assumes a Plancherel/IBP identity;
it does NOT prove that continuum identity from the S004 action.
-/

namespace ReleaseProofs.R16Energy

open ReleaseProofs.R16Completion

/-- The fixed-flat two-real-field acceleration Hessian of the S004 action
    in positive ordinary-kinetic orientation has diagonal -2 F κ, off
    diagonal 0. Its determinant is a literal 2x2 determinant. -/
def s004AccelDet (F kappa : ℝ) : ℝ :=
  (-2 * F * kappa) * (-2 * F * kappa) - 0 * 0

theorem s004_acceleration_det_exact (F kappa : ℝ) :
    s004AccelDet F kappa = 4 * (F * kappa) ^ 2 := by
  unfold s004AccelDet
  ring

theorem s004_acceleration_det_strictly_positive (F kappa : ℝ)
    (hF : 0 < F) (hk : 0 < kappa) :
    0 < s004AccelDet F kappa := by
  rw [s004_acceleration_det_exact]
  have hprod : 0 < F * kappa := mul_pos hF hk
  positivity

/-- The first-order auxiliary constraints Cx=p_x-λ and Cλ=p_λ
    have the invertible 4x4 off-diagonal symplectic bracket represented
    by this explicit linear map on two-component pairs. -/
def auxConstraintBracket (x y : (Fin 2 → ℝ)) : (Fin 2 → ℝ) × (Fin 2 → ℝ) :=
  (fun i => -y i, x)

theorem auxiliary_bracket_no_null_vector (x y : Fin 2 → ℝ)
    (hzero : auxConstraintBracket x y = (fun _ => 0, fun _ => 0)) :
    x = (fun _ => 0) ∧ y = (fun _ => 0) := by
  have hx : x = (fun _ => 0) := by
    funext i
    have hi := congrArg (fun p : (Fin 2 → ℝ) × (Fin 2 → ℝ) => p.2 i) hzero
    simpa [auxConstraintBracket] using hi
  have hy : y = (fun _ => 0) := by
    funext i
    have hi := congrArg (fun p : (Fin 2 → ℝ) × (Fin 2 → ℝ) => p.1 i) hzero
    simp only [auxConstraintBracket, neg_eq_zero] at hi
    exact hi
  exact ⟨hx, hy⟩

/-- A stronger explicitly sufficient parameter range gives an explicit
    H² Fourier-symbol bound. The standard 4 b μ > a² condition proves mere
    positivity; the extra 2 b μ > a² is used only for this simple bound. -/
theorem competing_frequency_h2_lower_bound
    (mu a b c q : ℝ)
    (hb : 0 < b) (hbc : 2 * c ≤ b)
    (hgap : 2 * b * c ≤ 2 * b * mu - a ^ 2) :
    c * (1 + q ^ 2) ≤ competingFrequencySq mu a b q := by
  have hid :
      2 * b * (competingFrequencySq mu a b q - c * (1 + q ^ 2)) =
        (b * q - a) ^ 2 + (2 * b * mu - a ^ 2 - 2 * b * c) +
          (b * (b - 2 * c)) * q ^ 2 := by
    unfold competingFrequencySq
    ring
  have hc : 0 ≤ b * (b - 2 * c) :=
    mul_nonneg hb.le (by linarith)
  have hq : 0 ≤ (b * (b - 2 * c)) * q ^ 2 :=
    mul_nonneg hc (sq_nonneg _)
  have hh : 0 ≤ 2 * b * mu - a ^ 2 - 2 * b * c := by linarith
  have hnon : 0 ≤
      2 * b * (competingFrequencySq mu a b q - c * (1 + q ^ 2)) := by
    rw [hid]
    nlinarith [sq_nonneg (b * q - a)]
  by_contra hn
  have hnegative : competingFrequencySq mu a b q - c * (1 + q ^ 2) < 0 := by
    linarith
  have hp : 0 < 2 * b := by positivity
  have hprod := mul_neg_of_pos_of_neg hp hnegative
  linarith

/-- Coercive lower bound for each finite Fourier mode; no claim is made that
    a nonlinear Fourier truncation is invariant under the full field PDE. -/
theorem competing_mode_h2_coercivity
    (mu a b c q x v : ℝ)
    (hb : 0 < b) (hbc : 2 * c ≤ b)
    (hgap : 2 * b * c ≤ 2 * b * mu - a ^ 2) :
    c * (1 + q ^ 2) * x ^ 2 + v ^ 2 ≤
      2 * competingModeHamiltonian mu a b q x v := by
  have hw := competing_frequency_h2_lower_bound mu a b c q hb hbc hgap
  have hx := mul_le_mul_of_nonneg_right hw (sq_nonneg x)
  unfold competingModeHamiltonian
  nlinarith

theorem finite_fourier_modes_coercive
    {ι : Type*} (modes : Finset ι)
    (q amp velocity : ι → ℝ) (mu a b c : ℝ)
    (hb : 0 < b) (hbc : 2 * c ≤ b)
    (hgap : 2 * b * c ≤ 2 * b * mu - a ^ 2) :
    Finset.sum modes (fun i : ι =>
      c * (1 + (q i) ^ 2) * (amp i) ^ 2 + (velocity i) ^ 2) ≤
      Finset.sum modes (fun i : ι =>
        2 * competingModeHamiltonian mu a b (q i) (amp i) (velocity i)) := by
  apply Finset.sum_le_sum
  intro i hi
  exact competing_mode_h2_coercivity mu a b c (q i) (amp i) (velocity i)
    hb hbc hgap

/-- Abstract consequence of a *supplied* continuum Plancherel/integration
    identity: coercive quadratic form + nonnegative quartic form = 0 forces
    zero L² norm. The continuum integration identity remains an analytic
    (not kernel-verified) input to this lemma. -/
theorem tested_static_energy_identity_forces_zero
    (coercivityConstant l2Squared linearForm quarticForm : ℝ)
    (hc : 0 < coercivityConstant) (hnorm : 0 ≤ l2Squared)
    (hbound : coercivityConstant * l2Squared ≤ linearForm)
    (hquartic : 0 ≤ quarticForm)
    (hstationary : linearForm + quarticForm = 0) :
    l2Squared = 0 := by
  have hnon : 0 ≤ coercivityConstant * l2Squared :=
    mul_nonneg hc.le hnorm
  have hzero : coercivityConstant * l2Squared = 0 := by
    linarith
  rcases mul_eq_zero.mp hzero with h | h
  · exact False.elim ((ne_of_gt hc) h)
  · exact h

end ReleaseProofs.R16Energy
