import Mathlib

/-!
# Issue #18: exact algebra of the corrected S004 static dispersion

These are kernel-checked real-algebra statements about Q6/Q7. They do not
supply a physical vacuum, a UV cutoff, or a derivation of the independent
Swift–Hohenberg model from S004.
-/

namespace ReleaseProofs.R18

def curvatureGap (K G T : ℝ) : ℝ := 4 * K * T - G ^ 2

def poleDiscriminant (K G T massSq q : ℝ) : ℝ :=
  (1 + G * q) ^ 2 - 4 * K * (massSq + T * q ^ 2)

theorem discriminant_expansion (K G T massSq q : ℝ) :
    poleDiscriminant K G T massSq q =
      (1 - 4 * K * massSq) + 2 * G * q -
        curvatureGap K G T * q ^ 2 := by
  unfold poleDiscriminant curvatureGap
  ring

def lowerPole (K G T massSq q : ℝ) : ℝ :=
  q + (1 + G * q - Real.sqrt (poleDiscriminant K G T massSq q)) / (2 * K)

def upperPole (K G T massSq q : ℝ) : ℝ :=
  q + (1 + G * q + Real.sqrt (poleDiscriminant K G T massSq q)) / (2 * K)

/-- This is the derivative in z of the Q6 inverse propagator. -/
def poleSlope (K G z q : ℝ) : ℝ := 1 - 2 * K * (z - q) + G * q

theorem lower_pole_slope (K G T massSq q : ℝ) (hK : K ≠ 0) :
    poleSlope K G (lowerPole K G T massSq q) q =
      Real.sqrt (poleDiscriminant K G T massSq q) := by
  unfold poleSlope lowerPole
  field_simp [hK]
  ring

theorem upper_pole_slope (K G T massSq q : ℝ) (hK : K ≠ 0) :
    poleSlope K G (upperPole K G T massSq q) q =
      -Real.sqrt (poleDiscriminant K G T massSq q) := by
  unfold poleSlope upperPole
  field_simp [hK]
  ring

/-- Two distinct real Q7 pole candidates necessarily have opposite residue signs. -/
theorem opposite_residue_signs (K G T massSq q : ℝ)
    (hK : K ≠ 0) (hd : 0 < poleDiscriminant K G T massSq q) :
    0 < poleSlope K G (lowerPole K G T massSq q) q ∧
      poleSlope K G (upperPole K G T massSq q) q < 0 := by
  rw [lower_pole_slope K G T massSq q hK,
      upper_pole_slope K G T massSq q hK]
  have hs : 0 < Real.sqrt (poleDiscriminant K G T massSq q) :=
    Real.sqrt_pos.2 hd
  constructor <;> linarith

/-- A sufficient explicit high-wave-number criterion for complex Q7 poles. -/
theorem discriminant_negative_at_large_q (K G T massSq q : ℝ)
    (hB : 0 < curvatureGap K G T)
    (hG : 0 ≤ G) (hs : 0 ≤ 1 - 4 * K * massSq)
    (hq : 1 ≤ q)
    (hlarge : (1 - 4 * K * massSq) + 2 * G <
      curvatureGap K G T * q) :
    poleDiscriminant K G T massSq q < 0 := by
  have hqpos : 0 < q := by linarith
  have hsmall : 1 - 4 * K * massSq ≤ (1 - 4 * K * massSq) * q := by
    nlinarith
  have hbound :
      ((1 - 4 * K * massSq) + 2 * G) * q <
        curvatureGap K G T * q ^ 2 := by
    have h := mul_lt_mul_of_pos_right hlarge hqpos
    nlinarith
  rw [discriminant_expansion]
  nlinarith

theorem negative_discriminant_has_no_real_square (K G T massSq q : ℝ)
    (hd : poleDiscriminant K G T massSq q < 0) :
    ¬ ∃ s : ℝ, s ^ 2 = poleDiscriminant K G T massSq q := by
  rintro ⟨s, hs⟩
  nlinarith [sq_nonneg s]

/-!
The next declarations are a *different*, explicitly spatial, second-order-in-time
effective model. They provide a healthy linear finite-wavelength mechanism,
not a nonlinear equivalence from the untruncated S004 action.
-/

def spatialRailFrequencySq (gap bending qStar q : ℝ) : ℝ :=
  gap + bending * (q - qStar) ^ 2

theorem spatial_rail_lower_bound (gap bending qStar q : ℝ)
    (hb : 0 ≤ bending) :
    gap ≤ spatialRailFrequencySq gap bending qStar q := by
  unfold spatialRailFrequencySq
  have h := mul_nonneg hb (sq_nonneg (q - qStar))
  linarith

theorem spatial_rail_unique_minimizer (gap bending qStar q : ℝ)
    (hb : 0 < bending)
    (heq : spatialRailFrequencySq gap bending qStar q = gap) :
    q = qStar := by
  have hz : bending * (q - qStar) ^ 2 = 0 := by
    unfold spatialRailFrequencySq at heq
    linarith
  have hsq : (q - qStar) ^ 2 = 0 := by
    rcases mul_eq_zero.mp hz with hh | hh
    · exact False.elim ((ne_of_gt hb) hh)
    · exact hh
  nlinarith

theorem spatial_rail_positive (gap bending qStar q : ℝ)
    (hgap : 0 < gap) (hb : 0 ≤ bending) :
    0 < spatialRailFrequencySq gap bending qStar q := by
  have h := spatial_rail_lower_bound gap bending qStar q hb
  linarith

theorem finite_wavenumber (qStar : ℝ) (hq : 0 < qStar) :
    0 < Real.sqrt qStar ∧ (Real.sqrt qStar) ^ 2 = qStar := by
  exact ⟨Real.sqrt_pos.2 hq, Real.sq_sqrt hq.le⟩

/-- The independent dissipative rail has a positive growth rate at its band center. -/
def dissipativeGrowth (r bending qStar q : ℝ) : ℝ :=
  r - bending * (q - qStar) ^ 2

theorem dissipative_band_center (r bending qStar : ℝ) (hr : 0 < r) :
    0 < dissipativeGrowth r bending qStar qStar := by
  simpa [dissipativeGrowth] using hr

theorem dissipative_band_condition (r bending qStar q : ℝ)
    (hband : bending * (q - qStar) ^ 2 < r) :
    0 < dissipativeGrowth r bending qStar q := by
  unfold dissipativeGrowth
  linarith

end ReleaseProofs.R18
