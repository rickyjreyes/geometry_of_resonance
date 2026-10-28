import Mathlib

/-! R02: Structure and Derivation of Physical Constants (DOI: 10.5281/zenodo.15596159).
Homogeneous ratios do not independently select an absolute physical scale. -/
namespace ReleaseProofs.R02

noncomputable def dimensionlessRatio (numerator denominator : ℝ) : ℝ :=
  numerator / denominator

theorem simultaneous_scale_invariant (a b scale : ℝ)
    (hb : b ≠ 0) (hs : scale ≠ 0) :
    dimensionlessRatio (scale * a) (scale * b) =
      dimensionlessRatio a b := by
  unfold dimensionlessRatio
  field_simp [hb, hs]
  <;> ring

theorem independent_scale_undetermined (a b : ℝ) (hb : b ≠ 0) :
    dimensionlessRatio (2 * a) (2 * b) = dimensionlessRatio a b := by
  exact simultaneous_scale_invariant a b 2 hb (by norm_num)

end ReleaseProofs.R02
