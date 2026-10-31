import Mathlib
import ReleaseProofs.R16_AuxiliaryEquivalence

/-!
# Issue #16: two-real-field affine Hamiltonian obstruction

For the regular two-component auxiliary representative, the two
Ostrogradsky canonical momenta enter affinely. At any nonzero auxiliary
velocity pair, *any* prescribed finite energy can be reached by varying
only those canonical momenta, regardless of the fixed potential,
ordinary-kinetic contributions, or higher-velocity momenta.

This is a finite-dimensional algebraic theorem on the declared
two-field principal model. It is not the continuum Dirac constraint
algorithm, a nonlinear S004 soliton theorem, or a proof covering
new derivative-dependent degeneracies.
-/

namespace ReleaseProofs.R16TwoField

open ReleaseProofs.R16Auxiliary

/-- The affine two-field Ostrogradsky momentum contribution plus an
arbitrary fixed term C; the latter may include potentials and all
terms independent of the canonical momenta p0 and p1. -/
def twoFieldAffineEnergy (Q0 Q1 p0 p1 C : ℝ) : ℝ :=
  p0 * Q0 + p1 * Q1 + C

/-- A nonzero two-real-component auxiliary velocity makes the energy
onto the entire real line as its two canonical momenta vary. -/
theorem two_field_affine_energy_onto
    (Q0 Q1 C target : ℝ) (hQ : Q0 ≠ 0 ∨ Q1 ≠ 0) :
    ∃ p0 p1 : ℝ, twoFieldAffineEnergy Q0 Q1 p0 p1 C = target := by
  rcases hQ with h0 | h1
  · refine ⟨(target - C) / Q0, 0, ?_⟩
    unfold twoFieldAffineEnergy
    field_simp [h0]
    ring
  · refine ⟨0, (target - C) / Q1, ?_⟩
    unfold twoFieldAffineEnergy
    field_simp [h1]
    ring

theorem two_field_affine_energy_unbounded_below
    (Q0 Q1 C bound : ℝ) (hQ : Q0 ≠ 0 ∨ Q1 ≠ 0) :
    ∃ p0 p1 : ℝ, twoFieldAffineEnergy Q0 Q1 p0 p1 C < bound := by
  obtain ⟨p0, p1, heq⟩ :=
    two_field_affine_energy_onto Q0 Q1 C (bound - 1) hQ
  refine ⟨p0, p1, ?_⟩
  rw [heq]
  linarith

theorem two_field_affine_energy_unbounded_above
    (Q0 Q1 C bound : ℝ) (hQ : Q0 ≠ 0 ∨ Q1 ≠ 0) :
    ∃ p0 p1 : ℝ, bound < twoFieldAffineEnergy Q0 Q1 p0 p1 C := by
  obtain ⟨p0, p1, heq⟩ :=
    two_field_affine_energy_onto Q0 Q1 C (bound + 1) hQ
  refine ⟨p0, p1, ?_⟩
  rw [heq]
  linarith

/-- Exact decomposition of the two independently rewritten real
principal components into their affine momenta and fixed remainder.
No assumption about the signs of h, k, or the potentials is used. -/
theorem two_auxiliary_hamiltonians_are_affine
    (h k U0 U1 Q0 Q1 p0 p1 pQ0 pQ1 : ℝ) :
    auxiliaryReducedHamiltonian h k U0 Q0 p0 pQ0 +
      auxiliaryReducedHamiltonian h k U1 Q1 p1 pQ1 =
    twoFieldAffineEnergy Q0 Q1 p0 p1
      (-pQ0 ^ 2 / (2 * h) - k * Q0 ^ 2 / 2 + U0 -
       pQ1 ^ 2 / (2 * h) - k * Q1 ^ 2 / 2 + U1) := by
  unfold auxiliaryReducedHamiltonian twoFieldAffineEnergy
  ring

/-- The two-real-field auxiliary Hamiltonian is unbounded below
whenever at least one auxiliary velocity is nonzero. This concerns
the *regular auxiliary rewrite*, not a conjectured new constrained action. -/
theorem two_component_auxiliary_unbounded_below
    (h k U0 U1 Q0 Q1 pQ0 pQ1 bound : ℝ)
    (hQ : Q0 ≠ 0 ∨ Q1 ≠ 0) :
    ∃ p0 p1 : ℝ,
      auxiliaryReducedHamiltonian h k U0 Q0 p0 pQ0 +
        auxiliaryReducedHamiltonian h k U1 Q1 p1 pQ1 < bound := by
  let C : ℝ :=
    -pQ0 ^ 2 / (2 * h) - k * Q0 ^ 2 / 2 + U0 -
      pQ1 ^ 2 / (2 * h) - k * Q1 ^ 2 / 2 + U1
  obtain ⟨p0, p1, hp⟩ :=
    two_field_affine_energy_unbounded_below Q0 Q1 C bound hQ
  refine ⟨p0, p1, ?_⟩
  rw [two_auxiliary_hamiltonians_are_affine]
  exact hp

end ReleaseProofs.R16TwoField
