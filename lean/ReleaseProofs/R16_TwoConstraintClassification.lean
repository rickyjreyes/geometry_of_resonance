import Mathlib
import ReleaseProofs.R16_LinearConstraintObstruction

/-!
# Geometry #16: independent momentum constraints remove free affine directions

A single generic scalar linear constraint leaves an unbounded free
momentum direction, as proved in R16_LinearConstraintObstruction.
This module proves the complementary finite-dimensional fact:
two independent scalar linear constraints on two canonical momenta
fix those momenta uniquely, and consequently fix any affine energy.

This is a rank/count statement for a 2x2 linear constraint matrix.
It does NOT assert that either constraint is generated or propagated by
the S004 Dirac algorithm, that imposing them preserves S004 dynamics,
or that their constrained PDE phase space is physically healthy.
-/

namespace ReleaseProofs.R16TwoConstraints

open ReleaseProofs.R16TwoField

/-- A nonsingular two-by-two system of momentum constraints has
an explicit solution by Cramer's rule. -/
theorem independent_constraints_have_solution
    (a b c d e f : ℝ) (hdet : a * d - b * c ≠ 0) :
    ∃ p0 p1 : ℝ,
      a * p0 + b * p1 = e ∧ c * p0 + d * p1 = f := by
  refine ⟨(e * d - b * f) / (a * d - b * c),
          (a * f - e * c) / (a * d - b * c), ?_, ?_⟩
  · field_simp [hdet]
    ring
  · field_simp [hdet]
    ring

/-- Invertibility of the constraint matrix excludes two different
canonical-momentum pairs satisfying the same two equations. -/
theorem independent_constraints_unique_momenta
    (a b c d e f p0 p1 q0 q1 : ℝ)
    (hdet : a * d - b * c ≠ 0)
    (hp : a * p0 + b * p1 = e ∧ c * p0 + d * p1 = f)
    (hq : a * q0 + b * q1 = e ∧ c * q0 + d * q1 = f) :
    p0 = q0 ∧ p1 = q1 := by
  have hfirst : a * (p0 - q0) + b * (p1 - q1) = 0 := by
    linarith [hp.1, hq.1]
  have hsecond : c * (p0 - q0) + d * (p1 - q1) = 0 := by
    linarith [hp.2, hq.2]
  have hm1 : d * (a * (p0 - q0) + b * (p1 - q1)) = 0 := by
    rw [hfirst]
    ring
  have hm2 : b * (c * (p0 - q0) + d * (p1 - q1)) = 0 := by
    rw [hsecond]
    ring
  have hp0 : (a * d - b * c) * (p0 - q0) = 0 := by
    nlinarith [hm1, hm2]
  have hm3 : a * (c * (p0 - q0) + d * (p1 - q1)) = 0 := by
    rw [hsecond]
    ring
  have hm4 : c * (a * (p0 - q0) + b * (p1 - q1)) = 0 := by
    rw [hfirst]
    ring
  have hp1 : (a * d - b * c) * (p1 - q1) = 0 := by
    nlinarith [hm3, hm4]
  have hz0 : p0 - q0 = 0 := (mul_eq_zero.mp hp0).resolve_left hdet
  have hz1 : p1 - q1 = 0 := (mul_eq_zero.mp hp1).resolve_left hdet
  constructor <;> linarith

/-- Under two independent prescribed linear momentum constraints,
the otherwise unbounded affine two-field energy becomes a constant.
The constraints are imposed, not derived from a Lagrangian. -/
theorem independent_constraints_fix_affine_energy
    (Q0 Q1 C a b c d e f : ℝ)
    (hdet : a * d - b * c ≠ 0) :
    ∃ E : ℝ, ∀ p0 p1 : ℝ,
      a * p0 + b * p1 = e →
      c * p0 + d * p1 = f →
      twoFieldAffineEnergy Q0 Q1 p0 p1 C = E := by
  obtain ⟨q0, q1, hq0, hq1⟩ :=
    independent_constraints_have_solution a b c d e f hdet
  refine ⟨twoFieldAffineEnergy Q0 Q1 q0 q1 C, ?_⟩
  intro p0 p1 hp0 hp1
  obtain ⟨heq0, heq1⟩ :=
    independent_constraints_unique_momenta
      a b c d e f p0 p1 q0 q1 hdet ⟨hp0, hp1⟩ ⟨hq0, hq1⟩
  rw [heq0, heq1]

end ReleaseProofs.R16TwoConstraints
