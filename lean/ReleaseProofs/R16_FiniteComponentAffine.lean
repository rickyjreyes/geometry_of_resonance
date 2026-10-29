import Mathlib
import ReleaseProofs.R16_TwoFieldAffineObstruction

/-!
# Issue 16: finite-component canonical-momentum obstruction

For any positive finite number of unconstrained real canonical momentum
coordinates, a nonzero auxiliary velocity vector Q makes an affine
Hamiltonian p · Q + C surjective onto the real energy axis.

This generalizes the verified one- and two-component *regular auxiliary*
algebra. It is deliberately NOT a continuum Dirac classification, a
proof that constrained momenta can vary freely, or a no-go theorem for
all conceivable degenerate higher-derivative actions.
-/

namespace ReleaseProofs.R16FiniteComponent

/-- The finite-dimensional Ostrogradsky affine contribution. The
remainder C is independent of the unconstrained canonical momenta. -/
def affineHamiltonian (n : ℕ) (Q p : Fin n → ℝ) (C : ℝ) : ℝ :=
  (∑ i : Fin n, p i * Q i) + C

/-- A single nonzero velocity component is enough to realize ANY
prescribed finite energy by varying only its conjugate momentum. -/
theorem affineHamiltonian_surjective_at_nonzero_velocity
    (n : ℕ) (Q : Fin n → ℝ) (C target : ℝ)
    (hQ : ∃ i : Fin n, Q i ≠ 0) :
    ∃ p : Fin n → ℝ, affineHamiltonian n Q p C = target := by
  obtain ⟨j, hj⟩ := hQ
  let p : Fin n → ℝ := fun i =>
    if i = j then (target - C) / Q j else 0
  refine ⟨p, ?_⟩
  have hs : (∑ i : Fin n, p i * Q i) =
      ((target - C) / Q j) * Q j := by
    simp [p, Finset.sum_ite_eq']
  unfold affineHamiltonian
  rw [hs]
  field_simp [hj]
  ring

/-- No lower energy bound exists for the unconstrained finite-component
affine canonical Hamiltonian whenever Q has a nonzero entry. -/
theorem affineHamiltonian_unbounded_below
    (n : ℕ) (Q : Fin n → ℝ) (C bound : ℝ)
    (hQ : ∃ i : Fin n, Q i ≠ 0) :
    ∃ p : Fin n → ℝ, affineHamiltonian n Q p C < bound := by
  obtain ⟨p, hp⟩ :=
    affineHamiltonian_surjective_at_nonzero_velocity n Q C (bound - 1) hQ
  refine ⟨p, ?_⟩
  rw [hp]
  linarith

/-- No upper energy bound exists either. -/
theorem affineHamiltonian_unbounded_above
    (n : ℕ) (Q : Fin n → ℝ) (C bound : ℝ)
    (hQ : ∃ i : Fin n, Q i ≠ 0) :
    ∃ p : Fin n → ℝ, bound < affineHamiltonian n Q p C := by
  obtain ⟨p, hp⟩ :=
    affineHamiltonian_surjective_at_nonzero_velocity n Q C (bound + 1) hQ
  refine ⟨p, ?_⟩
  rw [hp]
  linarith

/-- An exact criterion: the affine Hamiltonian can be independent of
all canonical momenta only when the entire velocity vector vanishes. -/
theorem affineHamiltonian_constant_iff_zero_velocity
    (n : ℕ) (Q : Fin n → ℝ) (C : ℝ) :
    (∀ p : Fin n → ℝ, affineHamiltonian n Q p C = C) ↔
      ∀ i : Fin n, Q i = 0 := by
  constructor
  · intro hc i
    by_contra hi
    obtain ⟨p, hp⟩ :=
      affineHamiltonian_surjective_at_nonzero_velocity
        n Q C (C + 1) ⟨i, hi⟩
    have he := hc p
    rw [he] at hp
    linarith
  · intro hz p
    unfold affineHamiltonian
    simp [hz]

end ReleaseProofs.R16FiniteComponent
