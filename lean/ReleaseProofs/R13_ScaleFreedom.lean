import Mathlib

/-!
# Geometry #13 — physical confinement scale is an independent input

This module formalizes the exact length dependence of a dimensionless
spectral eigenvalue. It does not assert that no possible revised WCT
action can ever dynamically select an intrinsic scale; it records the
precise scale freedom in the existing conditional spectral conversion.
-/

noncomputable section
namespace ReleaseProofs.R13ScaleFreedom

/-- Convert a dimensionless spatial eigenvalue using a freely chosen
positive reference length L. -/
def physicalEigenvalue (lambda L : ℝ) : ℝ := lambda / L ^ 2

/-- The corresponding positive square-root spectral wavenumber. -/
def spectralWavenumber (lambda L : ℝ) : ℝ :=
  Real.sqrt lambda / L

/-- The physical eigenvalue transforms inversely quadratically when
the reference length is rescaled by a nonzero factor. -/
theorem physicalEigenvalue_rescale
    (lambda L scale : ℝ) (hs : scale ≠ 0) :
    physicalEigenvalue lambda (scale * L) =
      physicalEigenvalue lambda L / scale ^ 2 := by
  unfold physicalEigenvalue
  field_simp

/-- The spectral wavenumber transforms with inverse length under
positive reference-scale changes. -/
theorem spectralWavenumber_rescale
    (lambda L scale : ℝ) :
    spectralWavenumber lambda (scale * L) =
      spectralWavenumber lambda L / scale := by
  unfold spectralWavenumber
  ring

/-- Every positive target wavenumber can be produced by choosing a
positive reference length when the dimensionless eigenvalue is
positive. This proves length dependence, not an internal selection
mechanism. -/
theorem arbitrary_positive_wavenumber_from_free_length
    (lambda target : ℝ) (hLambda : 0 < lambda) (ht : 0 < target) :
    ∃ L : ℝ, 0 < L ∧ spectralWavenumber lambda L = target := by
  have hs : 0 < Real.sqrt lambda := Real.sqrt_pos.2 hLambda
  refine ⟨Real.sqrt lambda / target, div_pos hs ht, ?_⟩
  unfold spectralWavenumber
  field_simp [ne_of_gt ht, ne_of_gt hs]
  all_goals ring

/-- Distinct positive lengths cannot give the same wavenumber for
a strictly positive dimensionless eigenvalue. -/
theorem spectralWavenumber_injective_on_positive_lengths
    (lambda L1 L2 : ℝ)
    (hLambda : 0 < lambda) (hL1 : 0 < L1) (hL2 : 0 < L2)
    (heq : spectralWavenumber lambda L1 =
      spectralWavenumber lambda L2) :
    L1 = L2 := by
  have hs : Real.sqrt lambda ≠ 0 :=
    ne_of_gt (Real.sqrt_pos.2 hLambda)
  dsimp [spectralWavenumber] at heq
  have hcross : Real.sqrt lambda * L2 =
      Real.sqrt lambda * L1 := by
    exact (div_eq_div_iff (ne_of_gt hL1) (ne_of_gt hL2)).mp heq
  exact (mul_left_cancel₀ hs hcross).symm

/-- A constant dimensionless eigenvalue alone cannot produce a
universal fixed numerical physical eigenvalue independent of L. -/
theorem physicalEigenvalue_depends_on_length :
    physicalEigenvalue 1 1 ≠ physicalEigenvalue 1 2 := by
  norm_num [physicalEigenvalue]

end ReleaseProofs.R13ScaleFreedom
