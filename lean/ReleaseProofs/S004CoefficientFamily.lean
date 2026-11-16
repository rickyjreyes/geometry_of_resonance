import Mathlib
import ReleaseProofs.R16_EnergyAndConstraints
import ReleaseProofs.S004CharacteristicCone

/-!
# S004 Phase XXVII: the S004 coefficient family `(κ, θ, γ)` — no admissible healthy member

The corrected S004 density keeps its weight `F` and potential `V` fixed; the three curvature
coefficients `(κ, θ, γ)` define a family. This module classifies the family against three
health requirements, using only kernel-checked algebra:

* **ghost freedom** needs a degenerate acceleration Hessian at regular points
  (`R16Class.bddBelow_forces_v_zero`, `R16Class.regular_hamiltonian_unbounded`); for the
  two-component S004 Hessian `(−2Fκ)·I₂` this happens iff `κ = 0`
  (`accel_hessian_degenerate_iff`);
* **positive curvature energy** is the source admissibility condition (`Admissible`);
* **hyperbolicity** (necessary for Sobolev well-posedness by the Lax–Mizohata theorem) needs
  real characteristics (`S004CharacteristicCone.real_characteristic_iff`).

Results:

* `admissible_member_obstructed`: every admissible member has, at every regular point, a
  strictly positive acceleration-Hessian determinant (Ostrogradsky ghost), an elliptic
  principal symbol and no strict hyperbolicity.
* `ghostfree_psd_members`: a member with positive semidefinite curvature form and degenerate
  acceleration Hessian is exactly a spatial-only member `κ = γ = 0`, `θ ≥ 0`; such members lie
  outside the admissible region.
* `hyperbolic_region_curvature_indefinite`: wherever `γ² > 4κθ` (necessary for strict
  hyperbolicity) the curvature form takes negative values.
* `psd_hyperbolic_members_keep_ghost`: positive semidefinite members with any real
  characteristic lie on `γ² = 4κθ` and keep the ghost whenever `κ ≠ 0`.

So the smallest change of the source assumptions that removes the ghost inside this family is
`κ = 0` (then `γ = 0` for a nonnegative curvature energy): the time-curvature term must be
deleted. That changes the degree-of-freedom count (`4 → 2` real configuration variables per
point), the dispersion (`2 → 1` branch) and the Lorentz structure of `Θ₀ = −(□ψ)R_ε`.
-/

namespace ReleaseProofs.S004CoefficientFamily

open ReleaseProofs.S004CharacteristicCone

/-- At a regular point (`F > 0`) the S004 two-component acceleration Hessian is degenerate iff
`κ = 0`. -/
theorem accel_hessian_degenerate_iff {F κ : ℝ} (hF : 0 < F) :
    R16Energy.s004AccelDet F κ = 0 ↔ κ = 0 := by
  rw [R16Energy.s004_acceleration_det_exact]
  constructor
  · intro h
    have h2 : (F * κ) ^ 2 = 0 := by linarith
    have h3 : F * κ = 0 := pow_eq_zero_iff two_ne_zero |>.mp h2
    rcases mul_eq_zero.mp h3 with h0 | h0
    · exact absurd h0 hF.ne'
    · exact h0
  · rintro rfl
    ring

/-- With `κ = 0` and `γ ≠ 0` the curvature density reaches `−1`. -/
theorem kappa_zero_density_witness {θ γ : ℝ} (hγ : γ ≠ 0) :
    curvatureDensity 0 θ γ (-(θ + 1) / γ) 1 = -1 := by
  unfold curvatureDensity
  field_simp
  ring

/-- Positive semidefiniteness of the per-component curvature density. -/
def CurvaturePSD (κ θ γ : ℝ) : Prop := ∀ Q P : ℝ, 0 ≤ curvatureDensity κ θ γ Q P

/-- Admissible members have positive semidefinite (indeed definite) curvature density. -/
theorem admissible_psd {κ θ γ : ℝ} (h : Admissible κ θ γ) : CurvaturePSD κ θ γ := by
  intro Q P
  have hsym := symbolForm_eq_curvatureDensity κ θ γ Q (-P)
  rw [neg_neg] at hsym
  rw [← hsym]
  rcases eq_or_ne Q 0 with hQ | hQ
  · rcases eq_or_ne P 0 with hP | hP
    · subst hQ; subst hP; simp [symbolForm]
    · exact (symbolForm_pos_of_admissible h (Or.inr (neg_ne_zero.mpr hP))).le
  · exact (symbolForm_pos_of_admissible h (Or.inl hQ)).le

/-- With `κ = 0`, a positive semidefinite curvature density forces `γ = 0` and `θ ≥ 0`. -/
theorem psd_kappa_zero {θ γ : ℝ} (h : CurvaturePSD 0 θ γ) : γ = 0 ∧ 0 ≤ θ := by
  have hθ : 0 ≤ θ := by simpa [curvatureDensity] using h 0 1
  refine ⟨?_, hθ⟩
  by_contra hγ
  have := h (-(θ + 1) / γ) 1
  rw [kappa_zero_density_witness hγ] at this
  norm_num at this

/-- **Every admissible member is obstructed at every regular point**: Ostrogradsky-nondegenerate
acceleration Hessian, elliptic principal symbol, and no strict hyperbolicity. -/
theorem admissible_member_obstructed {F κ θ γ : ℝ} (hF : 0 < F) (h : Admissible κ θ γ) :
    0 < R16Energy.s004AccelDet F κ ∧
      (∀ τ Y : ℝ, (τ ≠ 0 ∨ Y ≠ 0) → 0 < principalSymbol F κ θ γ τ Y) ∧
      ¬ ∃ s₁ s₂ : ℝ, 0 < s₁ ∧ s₁ < s₂ ∧ reducedQuadratic κ θ γ s₁ = 0 ∧
        reducedQuadratic κ θ γ s₂ = 0 :=
  ⟨R16Energy.s004_acceleration_det_strictly_positive F κ hF h.1,
    fun _ _ hne => admissible_symbol_elliptic hF h hne,
    admissible_not_strictly_hyperbolic h⟩

/-- **Ghost-free, nonnegative-curvature members are exactly the spatial-only members**, and none
of them is admissible. -/
theorem ghostfree_psd_members {F κ θ γ : ℝ} (hF : 0 < F) (hpsd : CurvaturePSD κ θ γ)
    (hdeg : R16Energy.s004AccelDet F κ = 0) :
    κ = 0 ∧ γ = 0 ∧ 0 ≤ θ ∧ ¬ Admissible κ θ γ := by
  have hk : κ = 0 := (accel_hessian_degenerate_iff hF).mp hdeg
  subst hk
  obtain ⟨hγ, hθ⟩ := psd_kappa_zero hpsd
  exact ⟨rfl, hγ, hθ, fun h => lt_irrefl _ h.1⟩

/-- In the region `γ² > 4κθ` (necessary for strict hyperbolicity) the curvature density is
indefinite: it takes a negative value. -/
theorem hyperbolic_region_curvature_indefinite {κ θ γ : ℝ} (h : 4 * κ * θ < γ ^ 2) :
    ∃ Q P : ℝ, curvatureDensity κ θ γ Q P < 0 := by
  rcases lt_trichotomy κ 0 with hk | hk | hk
  · exact ⟨1, 0, by simp [curvatureDensity]; linarith⟩
  · subst hk
    have hγ : γ ≠ 0 := by
      intro h0
      subst h0
      norm_num at h
    refine ⟨-(θ + 1) / γ, 1, ?_⟩
    rw [kappa_zero_density_witness hγ]
    norm_num
  · refine ⟨γ, -2 * κ, ?_⟩
    have : curvatureDensity κ θ γ γ (-2 * κ) = κ * (4 * κ * θ - γ ^ 2) := by
      unfold curvatureDensity
      ring
    rw [this]
    exact mul_neg_of_pos_of_neg hk (by linarith)

/-- Strict hyperbolicity (two distinct positive characteristic ratios) forces an indefinite
curvature density. -/
theorem strictly_hyperbolic_curvature_indefinite {κ θ γ : ℝ} (hk : 0 < κ)
    (hhyp : ∃ s₁ s₂ : ℝ, 0 < s₁ ∧ s₁ < s₂ ∧ reducedQuadratic κ θ γ s₁ = 0 ∧
      reducedQuadratic κ θ γ s₂ = 0) :
    ∃ Q P : ℝ, curvatureDensity κ θ γ Q P < 0 :=
  hyperbolic_region_curvature_indefinite ((two_positive_roots_iff hk).mp hhyp).1

/-- Positive semidefinite members with any real characteristic ratio lie on the boundary
`γ² = 4κθ`, and keep a nondegenerate acceleration Hessian (ghost) when `κ ≠ 0`. -/
theorem psd_hyperbolic_members_keep_ghost {F κ θ γ s : ℝ} (hF : 0 < F) (hk : κ ≠ 0)
    (hpsd : CurvaturePSD κ θ γ) (hs : reducedQuadratic κ θ γ s = 0) :
    γ ^ 2 = 4 * κ * θ ∧ 0 < R16Energy.s004AccelDet F κ := by
  have hdisc : γ ^ 2 ≤ 4 * κ * θ := by
    by_contra hlt
    push_neg at hlt
    obtain ⟨Q, P, hneg⟩ := hyperbolic_region_curvature_indefinite hlt
    exact absurd (hpsd Q P) (not_le.mpr hneg)
  refine ⟨psd_with_real_characteristic_is_boundary hdisc hs, ?_⟩
  rw [R16Energy.s004_acceleration_det_exact]
  have : F * κ ≠ 0 := mul_ne_zero hF.ne' hk
  positivity

end ReleaseProofs.S004CoefficientFamily
