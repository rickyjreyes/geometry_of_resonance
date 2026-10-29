import Mathlib
import ReleaseProofs.R16_TwoFieldAffineObstruction

/-!
# Geometry #16: a single linear auxiliary constraint is generically
# insufficient to bound the two-field affine Hamiltonian

Consider a constraint on the two Ostrogradsky canonical momenta:
a*p0+b*p1=d, with a nonzero. If the velocity pair (Q0,Q1)
is transverse to its normal (a,b), in the explicit sense
a*Q1-b*Q0 ≠ 0, the reduced affine Hamiltonian retains a free
unbounded direction along the constraint line.

This is a finite-dimensional canonical obstruction for a specified
class of extra constraints, not a complete Dirac analysis. Constraints
tangent to the velocity pair are outside this theorem's hypotheses.
-/

namespace ReleaseProofs.R16LinearConstraint

open ReleaseProofs.R16TwoField

/-- Explicit surjectivity of the two-field affine energy even after
the additional scalar linear momentum constraint a*p0+b*p1=d.
Nonzero transverse determinant is essential. -/
theorem affine_energy_onto_with_one_linear_constraint
    (Q0 Q1 C a b d target : ℝ)
    (ha : a ≠ 0) (htrans : a * Q1 - b * Q0 ≠ 0) :
    ∃ p0 p1 : ℝ,
      a * p0 + b * p1 = d ∧
      twoFieldAffineEnergy Q0 Q1 p0 p1 C = target := by
  let p1 : ℝ :=
    (target - (d * Q0 / a + C)) * a / (a * Q1 - b * Q0)
  let p0 : ℝ := (d - b * p1) / a
  refine ⟨p0, p1, ?_, ?_⟩
  · dsimp [p0]
    field_simp
    ring
  · dsimp [twoFieldAffineEnergy, p0, p1]
    field_simp
    ring

/-- No finite lower energy bound survives an additional scalar linear
momentum constraint when its gradient is not aligned with the Q pair. -/
theorem affine_energy_unbounded_below_with_one_linear_constraint
    (Q0 Q1 C a b d bound : ℝ)
    (ha : a ≠ 0) (htrans : a * Q1 - b * Q0 ≠ 0) :
    ∃ p0 p1 : ℝ,
      a * p0 + b * p1 = d ∧
      twoFieldAffineEnergy Q0 Q1 p0 p1 C < bound := by
  obtain ⟨p0, p1, hc, he⟩ :=
    affine_energy_onto_with_one_linear_constraint
      Q0 Q1 C a b d (bound - 1) ha htrans
  refine ⟨p0, p1, hc, ?_⟩
  rw [he]
  linarith

/-- The same constrained affine Hamiltonian is unbounded above. -/
theorem affine_energy_unbounded_above_with_one_linear_constraint
    (Q0 Q1 C a b d bound : ℝ)
    (ha : a ≠ 0) (htrans : a * Q1 - b * Q0 ≠ 0) :
    ∃ p0 p1 : ℝ,
      a * p0 + b * p1 = d ∧
      bound < twoFieldAffineEnergy Q0 Q1 p0 p1 C := by
  obtain ⟨p0, p1, hc, he⟩ :=
    affine_energy_onto_with_one_linear_constraint
      Q0 Q1 C a b d (bound + 1) ha htrans
  refine ⟨p0, p1, hc, ?_⟩
  rw [he]
  linarith

end ReleaseProofs.R16LinearConstraint
