import Mathlib
import ReleaseProofs.S004ChargeNoGo
import ReleaseProofs.R16_OstrogradskyClass

/-!
# Original S004 regular patch: NO acceleration-momentum primary constraint

The corrected nonzero-field higher-derivative S004 acceleration Hessian
is W = (-2 F(|psi|²) kappa) · id on the two real field components.
Because F>0 at nonzero field and kappa>0, the LEGENDRE acceleration
momentum p2 = W a + b(q,v) is bijective onto ALL p2 values.

This eliminates the possibility that the original regular Lagrangian's
acceleration Legendre map itself imposes a primary constraint capable of
removing the Ostrogradsky canonical momentum p1. A constrained
healthy nonlinear sector would require an independently derived extra
physical constraint, not a consequence of the regular acceleration
Hessian or merely fixing the original U(1) charge.

This module is precise about scope: nonzero-field regular canonical
patches only. The degenerate rho=0 nodal stratum and any nonlinear
constraint propagation require an independent Dirac analysis.
-/

namespace ReleaseProofs.S004RegularConstraints

open ReleaseProofs.R16Class

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The acceleration Legendre map of the **actual corrected S004
regular-patch Hessian** is bijective for every q and v and arbitrary
linear acceleration coefficient b. Therefore the regular Lagrangian
admits every acceleration momentum p2 as a phase-space coordinate,
with a UNIQUE acceleration recovering it. No primary acceleration
momentum constraint follows from its Hessian on this patch. -/
theorem regular_patch_all_accel_momenta_unique
    (rho eps alpha kappa : ℝ)
    (b : E → E → E)
    (hrho : 0 < rho) (heps : eps ≠ 0) (hkappa : 0 < kappa) :
    ∀ q v p2 : E, ∃! a : E,
      accelMomentum
        ((-2 * ReleaseProofs.R16.correctedWeight rho eps alpha * kappa) •
          LinearMap.id : E →ₗ[ℝ] E)
        b q v a = p2 := by
  have hbij := ReleaseProofs.S004Regularity.regular_patch_bijective
    (E := E) rho eps alpha kappa hrho heps hkappa
  exact (legendre_regular_iff_bijective
    ((-2 * ReleaseProofs.R16.correctedWeight rho eps alpha * kappa) •
      LinearMap.id : E →ₗ[ℝ] E) b).2 hbij

/-- The original S004 regular-patch acceleration Hessian produces NO
primary Legendre constraint, and its unrestricted Ostrogradsky
canonical Hamiltonian is unbounded both below and above for every
phase point with nonzero velocity, irrespective of lower-order
potential, gradient, or momentum-independent terms G.

This is a rigorously scoped canonical obstruction, NOT a full
nonlinear evolution or nodal-stratum theorem. -/
theorem regular_patch_no_primary_constraint_and_ghost
    (rho eps alpha kappa : ℝ)
    (b : E → E → E) (G : E → E → E → ℝ)
    (z : Phase E) (bound : ℝ)
    (hrho : 0 < rho) (heps : eps ≠ 0) (hkappa : 0 < kappa)
    (hv : z.v ≠ 0) :
    (∀ q v p2 : E, ∃! a : E,
      accelMomentum
        ((-2 * ReleaseProofs.R16.correctedWeight rho eps alpha * kappa) •
          LinearMap.id : E →ₗ[ℝ] E) b q v a = p2) ∧
    (∃ t : ℝ, ostroForm G (z.shift t) < bound) ∧
    (∃ t : ℝ, bound < ostroForm G (z.shift t)) := by
  exact ⟨regular_patch_all_accel_momenta_unique rho eps alpha kappa b
    hrho heps hkappa,
    ostroForm_unbounded_below G z hv bound,
    ostroForm_unbounded_above G z hv bound⟩

end ReleaseProofs.S004RegularConstraints
