import Mathlib
import ReleaseProofs.R16_GhostObstruction
import ReleaseProofs.R16_EnergyAndConstraints
import ReleaseProofs.R16_OstrogradskyClass

/-!
# S004 regular/nodal dichotomy and the surviving Ostrogradsky obstruction

The corrected S004 coefficient is F(rho) = rho /
(rho + epsilon^2 * exp(-2 alpha rho))^2. For rho >= 0, epsilon != 0,
its acceleration Hessian is nondegenerate precisely on the nonzero-field
patch; the nodal locus has vanishing acceleration Hessian.

On the regular patch, the existing R16 Ostrogradsky class theorem supplies
the unbounded canonical momentum direction. This is a finite canonical
theorem and a necessary obstruction to any healthy reduction. A proposal
to remove that direction must derive additional constraints and establish
preservation by the complete S004 dynamics. Merely restricting the spectrum
or setting a canonical momentum to zero is not such a derivation.
-/

namespace ReleaseProofs.S004Regularity

open ReleaseProofs

/-- The corrected acceleration coefficient vanishes only at the field
node; the positivity of the regulator does not create a degenerate
acceleration Hessian on the nonzero-field sector. -/
theorem weight_zero_iff_node (rho eps alpha : ℝ)
    (hrho : 0 ≤ rho) (heps : eps ≠ 0) :
    R16.correctedWeight rho eps alpha = 0 ↔ rho = 0 := by
  constructor
  · intro h
    rcases lt_or_eq_of_le hrho with hp | hz
    · exact False.elim ((ne_of_gt (R16.corrected_weight_pos rho eps alpha hp heps)) h)
    · exact hz.symm
  · intro h
    subst rho
    exact R16.corrected_weight_at_zero eps alpha

/-- The acceleration Hessian has zero determinant at a field node. -/
theorem node_hessian_degenerate (eps alpha kappa : ℝ) :
    R16Energy.s004AccelDet (R16.correctedWeight 0 eps alpha) kappa = 0 := by
  rw [R16.corrected_weight_at_zero]
  simp [R16Energy.s004AccelDet]

/-- The finite-dimensional S004 acceleration Hessian is invertible
at every nonzero-field regular point (the two real components use
the same nonzero scalar Hessian coefficient). -/
theorem regular_patch_bijective
    {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (rho eps alpha kappa : ℝ)
    (hrho : 0 < rho) (heps : eps ≠ 0) (hkappa : 0 < kappa) :
    Function.Bijective
      ((-2 * R16.correctedWeight rho eps alpha * kappa) •
        LinearMap.id : E →ₗ[ℝ] E) := by
  have hf : 0 < R16.correctedWeight rho eps alpha :=
    R16.corrected_weight_pos rho eps alpha hrho heps
  exact R16Class.s004_regular
    (E := E) (R16.correctedWeight rho eps alpha) kappa
    (ne_of_gt hf) (ne_of_gt hkappa)

/-- Exactly one of the two analytically distinct sectors occurs for rho >= 0:
a nodal, acceleration-degenerate point or a nonzero-field regular point
with a strictly positive 2x2 acceleration determinant. -/
theorem regular_or_nodal
    (rho eps alpha kappa : ℝ)
    (hrho : 0 ≤ rho) (heps : eps ≠ 0) (hkappa : 0 < kappa) :
    (rho = 0 ∧
      R16Energy.s004AccelDet (R16.correctedWeight rho eps alpha) kappa = 0) ∨
    (0 < rho ∧
      0 < R16Energy.s004AccelDet (R16.correctedWeight rho eps alpha) kappa) := by
  rcases lt_or_eq_of_le hrho with hp | hz
  · right
    refine ⟨hp, ?_⟩
    exact R16Energy.s004_acceleration_det_strictly_positive
      (R16.correctedWeight rho eps alpha) kappa
      (R16.corrected_weight_pos rho eps alpha hp heps) hkappa
  · left
    have hzero : rho = 0 := hz.symm
    constructor
    · exact hzero
    · subst rho
      exact node_hessian_degenerate eps alpha kappa

/-- The canonical Ostrogradsky energy is unbounded in both directions
at a nonzero-velocity regular S004 point, for any P0-independent
remainder G. We explicitly include the Hessian determinant certificate
rather than silently identifying the exceptional node with a healthy
open sector. This DOES NOT establish an invariant constrained phase
space or continuum well-posedness. -/
theorem regular_patch_ghost
    {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (rho eps alpha kappa : ℝ) (G : E → E → E → ℝ)
    (z : R16Class.Phase E) (bound : ℝ)
    (hrho : 0 < rho) (heps : eps ≠ 0) (hkappa : 0 < kappa)
    (hv : z.v ≠ 0) :
    0 < R16Energy.s004AccelDet
      (R16.correctedWeight rho eps alpha) kappa ∧
    (∃ t : ℝ, R16Class.ostroForm G (z.shift t) < bound) ∧
    (∃ t : ℝ, bound < R16Class.ostroForm G (z.shift t)) := by
  refine ⟨R16Energy.s004_acceleration_det_strictly_positive
      (R16.correctedWeight rho eps alpha) kappa
      (R16.corrected_weight_pos rho eps alpha hrho heps) hkappa,
    R16Class.ostroForm_unbounded_below G z hv bound,
    R16Class.ostroForm_unbounded_above G z hv bound⟩

end ReleaseProofs.S004Regularity
