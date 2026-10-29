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

/-- If the auxiliary-velocity vector is parallel to the normal of
the imposed linear momentum constraint, the entire constrained
Hamiltonian is actually constant. This is the sharp exceptional
case excluded by the transverse unboundedness theorem. -/
theorem affine_energy_constant_of_parallel
    (Q0 Q1 C a b d lam p0 p1 : ℝ)
    (hQ0 : Q0 = lam * a) (hQ1 : Q1 = lam * b)
    (hc : a * p0 + b * p1 = d) :
    twoFieldAffineEnergy Q0 Q1 p0 p1 C = lam * d + C := by
  unfold twoFieldAffineEnergy
  rw [hQ0, hQ1]
  calc
    lam * a * p0 + lam * b * p1 + C =
        lam * (a * p0 + b * p1) + C := by ring
    _ = lam * d + C := by rw [hc]

/-- Vanishing transverse determinant forces exact collinearity with
the constraint normal when a is nonzero. -/
theorem parallel_of_zero_transverse
    (Q0 Q1 a b : ℝ) (ha : a ≠ 0)
    (htrans : a * Q1 - b * Q0 = 0) :
    Q0 = (Q0 / a) * a ∧ Q1 = (Q0 / a) * b := by
  constructor
  · field_simp
  · field_simp
    nlinarith [htrans]

/-- Exact characterization of the linear-constraint boundary case:
zero determinant makes the constrained energy fixed on every allowed
momentum configuration. -/
theorem affine_energy_fixed_of_zero_transverse
    (Q0 Q1 C a b d p0 p1 : ℝ)
    (ha : a ≠ 0) (htrans : a * Q1 - b * Q0 = 0)
    (hc : a * p0 + b * p1 = d) :
    twoFieldAffineEnergy Q0 Q1 p0 p1 C = (Q0 / a) * d + C := by
  obtain ⟨hQ0, hQ1⟩ := parallel_of_zero_transverse Q0 Q1 a b ha htrans
  exact affine_energy_constant_of_parallel
    Q0 Q1 C a b d (Q0 / a) p0 p1 hQ0 hQ1 hc

/-- Sharp finite-dimensional classification: a single nondegenerate
linear momentum constraint bounds the two-component affine energy
below if and only if its normal is aligned with the velocity pair.
The result assumes unrestricted canonical momenta and does not
establish that such a constraint is dynamically preserved. -/
theorem constrained_energy_bounded_below_iff
    (Q0 Q1 C a b d : ℝ) (ha : a ≠ 0) :
    (∃ L : ℝ, ∀ p0 p1 : ℝ,
        a * p0 + b * p1 = d →
          L ≤ twoFieldAffineEnergy Q0 Q1 p0 p1 C) ↔
      a * Q1 - b * Q0 = 0 := by
  constructor
  · rintro ⟨L, hL⟩
    by_contra hnonzero
    obtain ⟨p0, p1, hc, hlt⟩ :=
      affine_energy_unbounded_below_with_one_linear_constraint
        Q0 Q1 C a b d L ha hnonzero
    exact (not_le_of_gt hlt) (hL p0 p1 hc)
  · intro hzero
    refine ⟨(Q0 / a) * d + C, ?_⟩
    intro p0 p1 hc
    rw [affine_energy_fixed_of_zero_transverse
      Q0 Q1 C a b d p0 p1 ha hzero hc]

end ReleaseProofs.R16LinearConstraint
