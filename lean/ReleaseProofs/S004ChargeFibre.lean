import Mathlib
import ReleaseProofs.S004Regularity

/-!
# S004 / A2 fixed-charge Ostrogradsky direction

On a regular nonzero-field canonical patch, the literal A2 Hamiltonian
contains P0 dot v while its true global U(1) charge is
Q = P0 dot Jx + P1 dot Jv. The radial momentum shift
  P0 -> P0 + t*x
preserves the exact charge for every field and velocity and changes
the canonical Hamiltonian by t*(x dot v). Thus, whenever x dot v != 0,
the unconstrained regular charge level has no energy lower or upper bound.

The remainder is the exact P0-independent part of the canonical Hamiltonian
at fixed spatial derivatives, fields, velocities and higher momentum.
This theorem is **not** a continuum patch theorem or a disproof of every
possible dynamically invariant healthy reduction.
-/

namespace ReleaseProofs.S004ChargeFibre

def phaseCharge (x1 x2 v1 v2 p01 p02 p11 p12 : ℝ) : ℝ :=
  p01 * (-x2) + p02 * x1 + p11 * (-v2) + p12 * v1

def canonicalJetEnergy (sigma rest v1 v2 p01 p02 : ℝ) : ℝ :=
  p01 * v1 + p02 * v2 + sigma * (v1 ^ 2 + v2 ^ 2) + rest

theorem phaseCharge_shift
    (x1 x2 v1 v2 p01 p02 p11 p12 t : ℝ) :
    phaseCharge x1 x2 v1 v2
      (p01 + t * x1) (p02 + t * x2) p11 p12 =
    phaseCharge x1 x2 v1 v2 p01 p02 p11 p12 := by
  unfold phaseCharge
  ring

theorem canonicalJetEnergy_shift
    (sigma rest x1 x2 v1 v2 p01 p02 t : ℝ) :
    canonicalJetEnergy sigma rest v1 v2
      (p01 + t * x1) (p02 + t * x2) =
    canonicalJetEnergy sigma rest v1 v2 p01 p02 +
      t * (x1 * v1 + x2 * v2) := by
  unfold canonicalJetEnergy
  ring

/-- The actual fixed-charge fibre is unbounded **below**, provided
the velocity has a nonzero radial component. -/
theorem phaseCharge_fibre_unbounded_below
    (sigma rest x1 x2 v1 v2 p01 p02 p11 p12 bound : ℝ)
    (hr : x1 * v1 + x2 * v2 ≠ 0) :
    ∃ t : ℝ,
      phaseCharge x1 x2 v1 v2 (p01 + t * x1) (p02 + t * x2) p11 p12 =
        phaseCharge x1 x2 v1 v2 p01 p02 p11 p12 ∧
      canonicalJetEnergy sigma rest v1 v2
        (p01 + t * x1) (p02 + t * x2) < bound := by
  refine ⟨(bound - canonicalJetEnergy sigma rest v1 v2 p01 p02 - 1) /
      (x1 * v1 + x2 * v2), ?_, ?_⟩
  · exact phaseCharge_shift x1 x2 v1 v2 p01 p02 p11 p12 _
  · rw [canonicalJetEnergy_shift]
    rw [div_mul_cancel₀ _ hr]
    linarith

/-- The same charge fibre is unbounded **above**. -/
theorem phaseCharge_fibre_unbounded_above
    (sigma rest x1 x2 v1 v2 p01 p02 p11 p12 bound : ℝ)
    (hr : x1 * v1 + x2 * v2 ≠ 0) :
    ∃ t : ℝ,
      phaseCharge x1 x2 v1 v2 (p01 + t * x1) (p02 + t * x2) p11 p12 =
        phaseCharge x1 x2 v1 v2 p01 p02 p11 p12 ∧
      bound < canonicalJetEnergy sigma rest v1 v2
        (p01 + t * x1) (p02 + t * x2) := by
  refine ⟨(bound - canonicalJetEnergy sigma rest v1 v2 p01 p02 + 1) /
      (x1 * v1 + x2 * v2), ?_, ?_⟩
  · exact phaseCharge_shift x1 x2 v1 v2 p01 p02 p11 p12 _
  · rw [canonicalJetEnergy_shift]
    rw [div_mul_cancel₀ _ hr]
    linarith

end ReleaseProofs.S004ChargeFibre
