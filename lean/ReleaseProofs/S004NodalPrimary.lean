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

/-! ## Exact constraint count and necessary nodal-flow conditions -/

/-- The corrected original S004 two-real-component field has exactly
TWO independent primary acceleration-momentum constraints at rho=0,
equal to the full dimension of its Hessian kernel. This is a rank
statement only; secondary constraints and their Poisson algebra
remain to be determined. -/
theorem nodal_primary_constraint_count_two
    (eps alpha kappa : ℝ) :
    Module.finrank ℝ
      (LinearMap.ker
        ((-2 * ReleaseProofs.R16.correctedWeight 0 eps alpha * kappa) •
          LinearMap.id :
          EuclideanSpace ℝ (Fin 2) →ₗ[ℝ]
            EuclideanSpace ℝ (Fin 2))) = 2 := by
  rw [node_accel_hessian_zero, LinearMap.ker_zero, finrank_top]
  exact finrank_euclideanSpace_fin

/-- An S004 field whose BOTH real components remain identically at a
node must have zero first-time-derivative components wherever the
standard kinematic relations q'=v hold. Thus the nodal constraint
surface by itself cannot be declared invariant with arbitrary
nonzero velocities. The theorem requires only actual derivatives,
not a postulated evolution or a toy finite-state flow. -/
theorem all_time_node_requires_zero_velocity
    (x y vx vy : ℝ → ℝ) (t0 : ℝ)
    (hx : ∀ t, x t = 0) (hy : ∀ t, y t = 0)
    (hxd : HasDerivAt x (vx t0) t0)
    (hyd : HasDerivAt y (vy t0) t0) :
    vx t0 = 0 ∧ vy t0 = 0 := by
  have hxzero : x = fun _ => (0 : ℝ) := funext hx
  have hyzero : y = fun _ => (0 : ℝ) := funext hy
  rw [hxzero] at hxd
  rw [hyzero] at hyd
  constructor
  · simpa using hxd.deriv.symm
  · simpa using hyd.deriv.symm

/-- In particular a classical full-field path with nonzero velocity
at a nodal time cannot remain in the nodal stratum for all times.
This is a rigorous necessary kinematic obstruction to nodal
invariance, not a proof of actual nonlinear S004 well-posedness. -/
theorem nonzero_node_velocity_requires_exit
    (x y vx vy : ℝ → ℝ) (t0 : ℝ)
    (hxd : HasDerivAt x (vx t0) t0)
    (hyd : HasDerivAt y (vy t0) t0)
    (hv : vx t0 ≠ 0 ∨ vy t0 ≠ 0) :
    ∃ t, x t ≠ 0 ∨ y t ≠ 0 := by
  by_contra hn
  push_neg at hn
  have hzero := all_time_node_requires_zero_velocity
    x y vx vy t0 (fun t => (hn t).1) (fun t => (hn t).2) hxd hyd
  rcases hv with h | h
  · exact h hzero.1
  · exact h hzero.2

end ReleaseProofs.S004NodalPrimary
