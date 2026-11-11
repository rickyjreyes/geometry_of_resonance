import Mathlib
import ReleaseProofs.S004Regularity
import ReleaseProofs.R16_OstrogradskyClass

/-!
# The actual corrected S004 nodal primary-momentum stratum

At rho=0 the corrected curvature coefficient F(rho) vanishes.
The acceleration Legendre map for the original two-real-component
S004 quadratic-in-acceleration sector therefore collapses:

      p2 = W a + b(q,v),   W = -2*kappa*F(0)*id = 0.

In contrast to the regular nonzero-field branch, every acceleration
has the same p2=b(q,v). Thus the nodal primary momentum condition is
exact, and the acceleration is NOT recoverable from p2.

These are actual primary/Legendre statements, NOT a Dirac secondary
constraint algorithm or a claim that the nodal submanifold is invariant
under the nonlinear S004 equations.
-/

namespace ReleaseProofs.S004NodalPrimary

open ReleaseProofs.R16Class

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The corrected S004 acceleration Hessian itself is the ZERO
linear map at the field node, not merely determinant-zero. -/
theorem node_accel_hessian_zero
    (eps alpha kappa : ℝ) :
    ((-2 * ReleaseProofs.R16.correctedWeight 0 eps alpha * kappa) •
      LinearMap.id : E →ₗ[ℝ] E) = 0 := by
  simp [ReleaseProofs.R16.corrected_weight_at_zero]

/-- At the corrected S004 node the acceleration Legendre map sends
EVERY acceleration to the same actual lower-order momentum b(q,v). -/
theorem node_acceleration_momentum_fixed
    (eps alpha kappa : ℝ) (b : E → E → E) (q v a : E) :
    accelMomentum
      ((-2 * ReleaseProofs.R16.correctedWeight 0 eps alpha * kappa) •
        LinearMap.id : E →ₗ[ℝ] E)
      b q v a = b q v := by
  rw [node_accel_hessian_zero]
  simp [accelMomentum]

/-- An arbitrary nodal higher momentum p2 is attainable if and only
if p2=b(q,v). This is an explicit actual-model primary constraint,
not an assumed restriction on the dynamical field. -/
theorem node_primary_momentum_iff
    (eps alpha kappa : ℝ) (b : E → E → E)
    (q v p2 : E) :
    (∃ a : E,
      accelMomentum
        ((-2 * ReleaseProofs.R16.correctedWeight 0 eps alpha * kappa) •
          LinearMap.id : E →ₗ[ℝ] E)
        b q v a = p2) ↔ p2 = b q v := by
  constructor
  · rintro ⟨a, ha⟩
    rw [node_acceleration_momentum_fixed] at ha
    exact ha.symm
  · intro hp
    refine ⟨0, ?_⟩
    rw [node_acceleration_momentum_fixed]
    exact hp.symm

/-- The nodal S004 Legendre map fails injectivity even in one real
component. Thus its momentum condition cannot determine the missing
acceleration; deriving consistent secondary constraints is a
separate nonlinear Dirac problem. -/
theorem node_acceleration_hessian_not_injective
    (eps alpha kappa : ℝ) :
    ¬ Function.Injective
      ((-2 * ReleaseProofs.R16.correctedWeight 0 eps alpha * kappa) •
        LinearMap.id : ℝ →ₗ[ℝ] ℝ) := by
  intro hinj
  have hEq :
      ((-2 * ReleaseProofs.R16.correctedWeight 0 eps alpha * kappa) •
        LinearMap.id : ℝ →ₗ[ℝ] ℝ) (0 : ℝ) =
      ((-2 * ReleaseProofs.R16.correctedWeight 0 eps alpha * kappa) •
        LinearMap.id : ℝ →ₗ[ℝ] ℝ) (1 : ℝ) := by
    simp [ReleaseProofs.R16.corrected_weight_at_zero]
  have hFalse := hinj hEq
  norm_num at hFalse

end ReleaseProofs.S004NodalPrimary
