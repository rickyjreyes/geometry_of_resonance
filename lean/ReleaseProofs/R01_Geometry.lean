import Mathlib

/-! R01: The Geometry of Resonance (DOI: 10.5281/zenodo.15644222).
These are exact scalar-model lemmas, not confinement or spacetime-emergence proofs. -/
namespace ReleaseProofs.R01

def regularizedDenominator (amplitude epsilon : ℝ) : ℝ :=
  amplitude ^ 2 + epsilon ^ 2

theorem regularizedDenominator_pos (amplitude epsilon : ℝ)
    (he : epsilon ≠ 0) : 0 < regularizedDenominator amplitude epsilon := by
  unfold regularizedDenominator
  have he2 : 0 < epsilon ^ 2 := sq_pos_of_ne_zero he
  nlinarith [sq_nonneg amplitude]

def cubicUpdate (r beta x : ℝ) : ℝ := x + r * x - beta * x ^ 3

theorem zero_is_fixed (r beta : ℝ) : cubicUpdate r beta 0 = 0 := by
  simp [cubicUpdate]

theorem curvatureSquare_nonneg (kappa : ℝ) : 0 ≤ kappa ^ 2 := sq_nonneg kappa

end ReleaseProofs.R01
