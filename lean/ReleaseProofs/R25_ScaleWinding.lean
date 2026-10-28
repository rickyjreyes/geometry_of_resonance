import Mathlib

/-! R25: Scale-winding correspondence (DOI: 10.5281/zenodo.22665914).
Exact phase closure on a nonzero-frequency log coordinate; physical scale
invariance and any unrestricted global bijection are separate claims. -/
namespace ReleaseProofs.R25

noncomputable def logClosureCoordinate (omega : ℝ) (w : ℤ) : ℝ :=
  (2 * Real.pi * (w : ℝ)) / omega

theorem winding_produces_closure (omega : ℝ) (w : ℤ)
    (h : omega ≠ 0) :
    omega * logClosureCoordinate omega w =
      2 * Real.pi * (w : ℝ) := by
  unfold logClosureCoordinate
  field_simp [h]

theorem zero_winding_coordinate (omega : ℝ) :
    logClosureCoordinate omega 0 = 0 := by
  simp [logClosureCoordinate]

theorem closure_additive (omega : ℝ) (m n : ℤ) :
    logClosureCoordinate omega (m + n) =
      logClosureCoordinate omega m + logClosureCoordinate omega n := by
  simp [logClosureCoordinate, Int.cast_add]
  ring

end ReleaseProofs.R25
