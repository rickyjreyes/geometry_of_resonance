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


/-! ## Physical mass is not the canonical U(1) charge -/

/-- The true Noether charge can be prescribed arbitrarily at a fixed
nonzero field amplitude by changing the canonical momentum P0 alone.
Consequently the Noether constraint is not pointwise equivalent to
an L2 mass constraint on the field. -/
theorem arbitrary_charge_at_fixed_field
    (x1 x2 target : ℝ) (hx : 0 < x1 ^ 2 + x2 ^ 2) :
    ∃ p01 p02 : ℝ,
      phaseCharge x1 x2 0 0 p01 p02 0 0 = target := by
  have hn : x1 ^ 2 + x2 ^ 2 ≠ 0 := ne_of_gt hx
  refine ⟨-(target * x2) / (x1 ^ 2 + x2 ^ 2),
    (target * x1) / (x1 ^ 2 + x2 ^ 2), ?_⟩
  unfold phaseCharge
  field_simp [hn]
  ring

/-- In particular, the same unit field has zero canonical phase charge
when all momenta and velocities are zero. Thus squared field amplitude
cannot be silently identified with the literal S004/A2 Noether charge. -/
theorem noether_charge_not_field_mass :
    phaseCharge 1 0 0 0 0 0 0 0 = 0 ∧
      (1 : ℝ) ^ 2 + (0 : ℝ) ^ 2 = 1 := by
  norm_num [phaseCharge]


/-- Fixed actual Noether charge does not yield even a local canonical
energy minimum: every positive radial-momentum perturbation radius
contains a charge-preserving direction of strict descent whenever
the field and velocity have a nonzero radial pairing.

This is an exact canonical-jet statement. In particular it does not
assume that all possible constrained submanifolds are invariant. -/
theorem fixed_charge_arbitrarily_small_strict_descent
    (sigma rest x1 x2 v1 v2 p01 p02 p11 p12 radius : ℝ)
    (hrad : x1 * v1 + x2 * v2 ≠ 0) (hradius : 0 < radius) :
    ∃ t : ℝ, 0 < |t| ∧ |t| < radius ∧
      phaseCharge x1 x2 v1 v2
        (p01 + t * x1) (p02 + t * x2) p11 p12 =
        phaseCharge x1 x2 v1 v2 p01 p02 p11 p12 ∧
      canonicalJetEnergy sigma rest v1 v2
          (p01 + t * x1) (p02 + t * x2) <
        canonicalJetEnergy sigma rest v1 v2 p01 p02 := by
  rcases lt_or_gt_of_ne hrad with hneg | hpos
  · refine ⟨radius / 2, ?_, ?_, ?_, ?_⟩
    · rw [abs_of_pos (by linarith : 0 < radius / 2)]
      linarith
    · rw [abs_of_pos (by linarith : 0 < radius / 2)]
      linarith
    · exact phaseCharge_shift x1 x2 v1 v2 p01 p02 p11 p12 _
    · rw [canonicalJetEnergy_shift]
      have hmul : (radius / 2) * (x1 * v1 + x2 * v2) < 0 :=
        mul_neg_of_pos_of_neg (by linarith) hneg
      linarith
  · refine ⟨-(radius / 2), ?_, ?_, ?_, ?_⟩
    · rw [abs_of_neg (by linarith : -(radius / 2) < 0)]
      linarith
    · rw [abs_of_neg (by linarith : -(radius / 2) < 0)]
      linarith
    · exact phaseCharge_shift x1 x2 v1 v2 p01 p02 p11 p12 _
    · rw [canonicalJetEnergy_shift]
      have hmul : -(radius / 2) * (x1 * v1 + x2 * v2) < 0 :=
        mul_neg_of_neg_of_pos (by linarith) hpos
      linarith

end ReleaseProofs.S004ChargeFibre
