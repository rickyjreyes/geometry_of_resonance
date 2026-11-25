import Mathlib
import ReleaseProofs.R18_FiniteScale

/-!
# Issue #16: regular higher-time obstruction and a distinct healthy linear rail

The original S004 action is not proved ghost-free. The Ostrogradsky witness below
makes its affine canonical direction unbounded. The positive Hamiltonian result
is for a separate spatial-only quadratic rail, not a repair of S004.
-/

namespace ReleaseProofs.R16

noncomputable def regularizedDenominator (u eps alpha : ℝ) : ℝ :=
  u + eps ^ 2 * Real.exp (-2 * alpha * u)

noncomputable def correctedWeight (u eps alpha : ℝ) : ℝ :=
  u / (regularizedDenominator u eps alpha) ^ 2

theorem denominator_pos (u eps alpha : ℝ)
    (hu : 0 ≤ u) (heps : eps ≠ 0) :
    0 < regularizedDenominator u eps alpha := by
  unfold regularizedDenominator
  have he : 0 < eps ^ 2 := sq_pos_of_ne_zero heps
  have hx : 0 < Real.exp (-2 * alpha * u) := Real.exp_pos _
  have hp := mul_pos he hx
  linarith

theorem corrected_weight_at_zero (eps alpha : ℝ) :
    correctedWeight 0 eps alpha = 0 := by
  simp [correctedWeight]

theorem corrected_weight_pos (u eps alpha : ℝ)
    (hu : 0 < u) (heps : eps ≠ 0) :
    0 < correctedWeight u eps alpha := by
  unfold correctedWeight
  exact div_pos hu (pow_pos (denominator_pos u eps alpha hu.le heps) 2)

/-- Positive coefficient of the nondegenerate acceleration-square term. -/
theorem acceleration_coefficient_pos (u eps alpha kappa : ℝ)
    (hu : 0 < u) (heps : eps ≠ 0) (hkappa : 0 < kappa) :
    0 < 2 * kappa * correctedWeight u eps alpha := by
  have hf := corrected_weight_pos u eps alpha hu heps
  positivity

/-- Fixed nonzero higher-time coordinate q2 yields the Ostrogradsky affine term. -/
noncomputable def ostroHamiltonian (K massSq q1 q2 p1 p2 : ℝ) : ℝ :=
  p1 * q2 - p2 ^ 2 / (2 * K) - q2 ^ 2 / 2 + massSq * q1 ^ 2 / 2

theorem ostro_unbounded_below (K massSq q1 q2 p2 : ℝ)
    (hq2 : q2 ≠ 0) (bound : ℝ) :
    ∃ p1 : ℝ, ostroHamiltonian K massSq q1 q2 p1 p2 < bound := by
  let C : ℝ := -p2 ^ 2 / (2 * K) - q2 ^ 2 / 2 + massSq * q1 ^ 2 / 2
  let p1 : ℝ := (bound - C - 1) / q2
  have hmul : p1 * q2 = bound - C - 1 := by
    dsimp [p1]
    field_simp [hq2]
  refine ⟨p1, ?_⟩
  calc
    ostroHamiltonian K massSq q1 q2 p1 p2 = (bound - C - 1) + C := by
      rw [ostroHamiltonian, hmul]
      dsimp [C]
      ring
    _ < bound := by linarith

theorem ostro_unbounded_above (K massSq q1 q2 p2 : ℝ)
    (hq2 : q2 ≠ 0) (bound : ℝ) :
    ∃ p1 : ℝ, bound < ostroHamiltonian K massSq q1 q2 p1 p2 := by
  let C : ℝ := -p2 ^ 2 / (2 * K) - q2 ^ 2 / 2 + massSq * q1 ^ 2 / 2
  let p1 : ℝ := (bound - C + 1) / q2
  have hmul : p1 * q2 = bound - C + 1 := by
    dsimp [p1]
    field_simp [hq2]
  refine ⟨p1, ?_⟩
  calc
    bound < (bound - C + 1) + C := by linarith
    _ = ostroHamiltonian K massSq q1 q2 p1 p2 := by
      rw [ostroHamiltonian, hmul]
      dsimp [C]
      ring

/-- An auxiliary cross kinetic term is indefinite even if its diagonal h varies. -/
def mixedKineticForm (h c x y : ℝ) : ℝ := h * x ^ 2 + 2 * c * x * y

theorem mixed_kinetic_unbounded_below (h c : ℝ)
    (hc : 0 < c) (bound : ℝ) :
    ∃ y : ℝ, mixedKineticForm h c 1 y < bound := by
  have hc2 : (2 * c) ≠ 0 := mul_ne_zero (by norm_num) (ne_of_gt hc)
  let y : ℝ := (bound - h - 1) / (2 * c)
  have hy : (2 * c) * y = bound - h - 1 := by
    dsimp [y]
    field_simp [hc2]
  refine ⟨y, ?_⟩
  unfold mixedKineticForm
  nlinarith [hy]

/-- A distinct second-time-order spatial rail: no acceleration dependence. -/
noncomputable def spatialRailLagrangian
    (gap bending qStar q amplitude velocity acceleration : ℝ) : ℝ :=
  (velocity ^ 2 -
    R18.spatialRailFrequencySq gap bending qStar q * amplitude ^ 2) / 2

theorem spatial_rail_acceleration_independent
    (gap bending qStar q amplitude velocity a1 a2 : ℝ) :
    spatialRailLagrangian gap bending qStar q amplitude velocity a1 =
      spatialRailLagrangian gap bending qStar q amplitude velocity a2 := rfl

noncomputable def spatialRailHamiltonian
    (gap bending qStar q amplitude momentum : ℝ) : ℝ :=
  (momentum ^ 2 +
    R18.spatialRailFrequencySq gap bending qStar q * amplitude ^ 2) / 2

theorem spatial_rail_hamiltonian_nonneg
    (gap bending qStar q amplitude momentum : ℝ)
    (hgap : 0 < gap) (hb : 0 ≤ bending) :
    0 ≤ spatialRailHamiltonian gap bending qStar q amplitude momentum := by
  have hf : 0 ≤ R18.spatialRailFrequencySq gap bending qStar q :=
    le_of_lt (R18.spatial_rail_positive gap bending qStar q hgap hb)
  have hprod : 0 ≤ R18.spatialRailFrequencySq gap bending qStar q * amplitude ^ 2 :=
    mul_nonneg hf (sq_nonneg amplitude)
  unfold spatialRailHamiltonian
  nlinarith [sq_nonneg momentum]

end ReleaseProofs.R16
