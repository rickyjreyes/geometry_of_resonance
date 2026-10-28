import Mathlib

/-! R07: Phase-Flux Field (DOI: 10.5281/zenodo.17578766).
This verifies a normalized finite-k selection model; it is not a D4-to-continuum theorem. -/
namespace ReleaseProofs.R07

def finiteBandRate (r damping k0 k : ℝ) : ℝ :=
  r - damping * (k ^ 2 - k0 ^ 2) ^ 2

theorem finiteBandRate_le_peak (r damping k0 k : ℝ)
    (hd : 0 ≤ damping) : finiteBandRate r damping k0 k ≤ r := by
  unfold finiteBandRate
  have h : 0 ≤ damping * (k ^ 2 - k0 ^ 2) ^ 2 :=
    mul_nonneg hd (sq_nonneg _)
  linarith

theorem finiteBandRate_at_peak (r damping k0 : ℝ) :
    finiteBandRate r damping k0 k0 = r := by
  simp [finiteBandRate]

def zeroPreservingStep (r beta x : ℝ) : ℝ :=
  x + r * x - beta * x ^ 3

theorem zero_invariant (r beta : ℝ) :
    zeroPreservingStep r beta 0 = 0 := by
  simp [zeroPreservingStep]

end ReleaseProofs.R07
