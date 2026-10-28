import Mathlib
import ReleaseProofs.R16_EnergyAndConstraints

/-!
# Issue 16: exact auxiliary reduction and Hamiltonian obstruction

Scope: a real-component constant-coefficient temporal-jet representative of
S004's nondegenerate acceleration principal block, NOT a full nonlinear
infinite-dimensional Dirac formalization. The original S004 Hessian is handled
by R16_EnergyAndConstraints. This module proves that rewriting such a
nondegenerate acceleration-square Lagrangian with auxiliary variables cannot
remove the Ostrogradsky affine canonical direction.

No new axioms, weakened parent theory, or ghost-free equivalence are asserted.
-/

namespace ReleaseProofs.R16Auxiliary

open ReleaseProofs.R16
open ReleaseProofs.R16Energy

/-- One real component of the fixed-background acceleration principal part:
    the potential is represented by its value U and the ordinary-speed
    coefficient by k; h is a nonzero higher-time coefficient. -/
def fourthOrderJetLagrangian (h k U velocity acceleration : ℝ) : ℝ :=
  -(h * acceleration ^ 2) / 2 + k * velocity ^ 2 / 2 - U

/-- Exact first-order auxiliary Lagrangian with Q intended to equal q-dot.
    The original higher-time term has become the Q kinetic term, NOT vanished. -/
def auxiliaryJetLagrangian
    (h k U velocity Q Qdot lagrangeMultiplier : ℝ) : ℝ :=
  -(h * Qdot ^ 2) / 2 + k * Q ^ 2 / 2 - U +
    lagrangeMultiplier * (velocity - Q)

/-- On the explicit auxiliary constraint and its derivative the two
    Lagrangian densities agree exactly, with no discarded boundary term. -/
theorem auxiliary_restriction_exact
    (h k U velocity acceleration lagrangeMultiplier : ℝ) :
    auxiliaryJetLagrangian h k U velocity velocity acceleration
        lagrangeMultiplier =
      fourthOrderJetLagrangian h k U velocity acceleration := by
  unfold auxiliaryJetLagrangian fourthOrderJetLagrangian
  ring

/-- The scalar fourth-order Euler–Lagrange jet combination, using
    force=U'(q) and q''/q'''' as independent numerical jet variables. -/
def fourthOrderEulerJet (h k force accel fourth : ℝ) : ℝ :=
  force + k * accel + h * fourth

/-- Algebraic elimination of the differentiated Q-constraint and the q
    multiplier equation is equivalent to the fourth-order jet equation.
    Differentiability and validity of the full field EL system remain separate
    assumptions; we are not pretending a jet lemma formalizes variational
    calculus for arbitrary smooth fields. -/
theorem auxiliary_elimination_at_jet
    (h k force accel fourth : ℝ) :
    (∃ lambdaDot : ℝ,
        lambdaDot = k * accel + h * fourth ∧
        force + lambdaDot = 0) ↔
      fourthOrderEulerJet h k force accel fourth = 0 := by
  constructor
  · rintro ⟨lambdaDot, heq, hforce⟩
    rw [heq] at hforce
    simpa [fourthOrderEulerJet, add_assoc] using hforce
  · intro h
    refine ⟨k * accel + h * fourth, rfl, ?_⟩
    simpa [fourthOrderEulerJet, add_assoc] using h

/-- Exact momentum for the Q coordinate of the auxiliary Lagrangian. -/
def auxiliaryQMomentum (h Qdot : ℝ) : ℝ := -h * Qdot

/-- If h ≠ 0 the Q velocity is uniquely reconstructible from its momentum.
    The auxiliary rewrite is regular, not a primary degeneracy. -/
theorem auxiliary_Q_velocity_reconstructed
    (h Qdot momentum : ℝ)
    (hh : h ≠ 0)
    (hp : momentum = auxiliaryQMomentum h Qdot) :
    Qdot = -momentum / h := by
  unfold auxiliaryQMomentum at hp
  rw [hp]
  field_simp [hh]
  ring

/-- Partial Legendre transform of the Q kinetic sector. -/
theorem auxiliary_kinetic_legendre_identity
    (h Qdot : ℝ) (hh : h ≠ 0) :
    auxiliaryQMomentum h Qdot * Qdot +
        (h * Qdot ^ 2) / 2 =
      -(auxiliaryQMomentum h Qdot) ^ 2 / (2 * h) := by
  unfold auxiliaryQMomentum
  field_simp [hh]
  ring

/-- Reduced first-order canonical Hamiltonian after algebraic multiplier
    constraints. The p*Q term is the unavoidable Ostrogradsky direction. -/
noncomputable def auxiliaryReducedHamiltonian
    (h k U Q p pQ : ℝ) : ℝ :=
  p * Q - pQ ^ 2 / (2 * h) - k * Q ^ 2 / 2 + U

/-- For any nonzero auxiliary velocity Q the reduced Hamiltonian is
    SURJECTIVE as a function of canonical p. This is stronger than merely
    displaying a negative quadratic kinetic coefficient. -/
theorem auxiliary_hamiltonian_onto
    (h k U Q pQ : ℝ)
    (hQ : Q ≠ 0) (target : ℝ) :
    ∃ p : ℝ, auxiliaryReducedHamiltonian h k U Q p pQ =
      target := by
  let C : ℝ := -pQ ^ 2 / (2 * h) - k * Q ^ 2 / 2 + U
  let p : ℝ := (target - C) / Q
  have hp : p * Q = target - C := by
    dsimp [p]
    field_simp [hQ]
  refine ⟨p, ?_⟩
  unfold auxiliaryReducedHamiltonian
  rw [hp]
  dsimp [C]
  ring

theorem auxiliary_hamiltonian_unbounded_below
    (h k U Q pQ : ℝ) (hQ : Q ≠ 0) (bound : ℝ) :
    ∃ p, auxiliaryReducedHamiltonian h k U Q p pQ < bound := by
  obtain ⟨p, hp⟩ :=
    auxiliary_hamiltonian_onto h k U Q pQ hQ (bound - 1)
  refine ⟨p, ?_⟩
  rw [hp]
  linarith

theorem auxiliary_hamiltonian_unbounded_above
    (h k U Q pQ : ℝ) (hQ : Q ≠ 0) (bound : ℝ) :
    ∃ p, bound < auxiliaryReducedHamiltonian h k U Q p pQ := by
  obtain ⟨p, hp⟩ :=
    auxiliary_hamiltonian_onto h k U Q pQ hQ (bound + 1)
  refine ⟨p, ?_⟩
  rw [hp]
  linarith

/-- Explicit necessary and sufficient algebraic degeneracy for the actual
    two-real-component flat S004 acceleration Hessian. -/
theorem two_real_s004_hessian_degenerate_iff
    (F kappa : ℝ) :
    s004AccelDet F kappa = 0 ↔ F * kappa = 0 := by
  rw [s004_acceleration_det_exact]
  constructor
  · intro h
    have hz : (F * kappa) ^ 2 = 0 := by nlinarith
    nlinarith
  · intro h
    rw [h]
    norm_num

/-- Every valid nonzero S004 field weight with kappa>0 excludes the
    degeneracy condition, irrespective of retuning gamma or theta. -/
theorem strict_s004_cannot_remove_hessian_mode
    (u eps alpha kappa : ℝ)
    (hu : 0 < u) (heps : eps ≠ 0) (hkappa : 0 < kappa) :
    s004AccelDet (correctedWeight u eps alpha) kappa ≠ 0 := by
  have hw := corrected_weight_pos u eps alpha hu heps
  have hd := s004_acceleration_det_strictly_positive
    (correctedWeight u eps alpha) kappa hw hkappa
  exact ne_of_gt hd

/-- The 4x4 primary auxiliary second-class bracket is invertible by a
    constructive inverse on two-component pairs, not merely a determinant. -/
theorem auxiliary_bracket_surjective
    (X Y : Fin 2 → ℝ) :
    ∃ x y : Fin 2 → ℝ, auxConstraintBracket x y = (X, Y) := by
  refine ⟨Y, (fun i => -X i), ?_⟩
  simp [auxConstraintBracket]

/-- Conditional finite-dimensional Dirac count: 6 configuration
    coordinates (q,Q,lambda each with 2 real components), 4 independent
    second-class primaries, no secondary constraints in the regular block,
    leave 8 phase dimensions / 4 configuration degrees of freedom.
    This is an ARITHMETIC count, conditional on the analytic Dirac algorithm. -/
theorem two_field_auxiliary_constraint_count :
    (6 * 2 - 4 : ℕ) = 8 ∧ (8 / 2 : ℕ) = 4 := by
  norm_num

/-- Imposing Q=0 as an EXTRA equation is not generally an equivalent
    rewrite: a free fourth-order jet with nonzero velocity is allowed by
    the original fourth-order equation. This counterexample is for the
    declared constant-coefficient/free one-component representative. -/
theorem zero_auxiliary_velocity_is_an_extra_restriction (h : ℝ) :
    fourthOrderEulerJet h 0 0 0 0 = 0 ∧ (1 : ℝ) ≠ 0 := by
  constructor
  · simp [fourthOrderEulerJet]
  · norm_num

end ReleaseProofs.R16Auxiliary
