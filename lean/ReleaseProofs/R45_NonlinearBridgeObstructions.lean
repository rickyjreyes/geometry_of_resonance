import Mathlib

/-!
# Geometry #45: rigorous obstructions to inferring a nonlinear
effective theory from one matched quadratic dispersion branch

These are exact algebraic and frequency-support consequences, NOT
a controlled S004-to-PFF reduction and not a full Fourier PDE proof.
The physical action/energy/charge map, nonlinear error estimate, and
UV/ghost control are separate open problems.
-/

namespace ReleaseProofs.R45NonlinearBridge

/-- The potential's cubic amplitude-channel coefficient in E5.
This depends on the third derivative even after the quadratic spectral
coefficients (which depend on V'') have been fixed. -/
def cubicVertex (amplitude Vsecond Vthird : ℝ) : ℝ :=
  -(2 * amplitude * Vsecond + (4 / 3) * amplitude ^ 3 * Vthird)

/-- Varying only the third potential derivative changes the cubic
vertex by an exactly known nonzero multiple on nonzero backgrounds. -/
theorem cubic_vertex_parameter_difference
    (amplitude Vsecond a b : ℝ) :
    cubicVertex amplitude Vsecond a -
      cubicVertex amplitude Vsecond b =
      -(4 / 3) * amplitude ^ 3 * (a - b) := by
  unfold cubicVertex
  ring

/-- A nonzero background determines the cubic potential derivative
uniquely from the cubic vertex once V'' is fixed. Conversely,
different V''' values necessarily produce different cubic vertices,
despite identical V''. -/
theorem cubic_vertex_injective_on_nonzero_background
    (amplitude Vsecond : ℝ) (ha : amplitude ≠ 0) :
    Function.Injective (fun Vthird : ℝ => cubicVertex amplitude Vsecond Vthird) := by
  intro a b hab
  have hdiff : -(4 / 3) * amplitude ^ 3 * (a - b) = 0 := by
    rw [← cubic_vertex_parameter_difference]
    exact sub_eq_zero.mpr hab
  have hfactor : (-(4 / 3) : ℝ) * amplitude ^ 3 ≠ 0 := by
    exact mul_ne_zero (by norm_num) (pow_ne_zero 3 ha)
  have hz : a - b = 0 := (mul_eq_zero.mp hdiff).resolve_left hfactor
  exact sub_eq_zero.mp hz

/-- In a two-mode quintic interaction, three copies of the high
frequency l and two conjugate copies of the low frequency k
generate the algebraic frequency 3l-2k. -/
def quinticFrequency (k l : ℤ) : ℤ := 3 * l - 2 * k

/-- That generated quintic frequency lies strictly outside the
original upper frequency whenever k<l. -/
theorem quinticFrequency_above_max (k l : ℤ) (h : k < l) :
    l < quinticFrequency k l := by
  dsimp [quinticFrequency]
  omega

/-- No finite Fourier coordinate set with a high mode l and a lower
mode k can contain the generated high frequency while retaining l as
its upper bound. This does not assert that all field coefficients
are nonzero; that requires the explicit nonlinear convolution term. -/
theorem quinticFrequency_outside_bounded_support
    (K : Finset ℤ) (k l : ℤ)
    (hkl : k < l) (hmax : ∀ j ∈ K, j ≤ l) :
    quinticFrequency k l ∉ K := by
  intro hin
  have hupper := hmax (quinticFrequency k l) hin
  have hlower := quinticFrequency_above_max k l hkl
  omega

/-- The corresponding complex two-mode quintic interaction coefficient
is nonzero whenever both two-mode amplitudes are nonzero. -/
theorem quintic_extreme_coefficient_nonzero
    (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) :
    b ^ 3 * (star a) ^ 2 ≠ 0 := by
  exact mul_ne_zero (pow_ne_zero 3 hb)
    (pow_ne_zero 2 (star_ne_zero.mpr ha))

end ReleaseProofs.R45NonlinearBridge
