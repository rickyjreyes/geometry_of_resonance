import Mathlib
import ReleaseProofs.R16_PaisUhlenbeck
import ReleaseProofs.S004CharacteristicCone

/-!
# S004 Phase XXVIII: nonlinear second-harmonic forcing of the ghost branch

Repository results (`R16_LinearModeReduction`, `R16_LinearTrajectoryEnergy`) construct
positive-energy **linear** Fourier-mode sectors of the linearized S004 Pais–Uhlenbeck dynamics.
This module records, with kernel-checked algebra, why those sectors are not invariant under the
original nonlinear S004 equations.

**Symbolic input** (`derivations/s004_closure_verify.py`, check family `C13`; exact `sympy`):
real sector `Im ψ = 0` (invariant under the conjugation symmetry), one spatial dimension, static
homogeneous background `x = A > 0` with `V'(A²) = 0`, and the expansion
`x = A + δ cos(kz) cos(ωt) + δ² x₂ + O(δ³)` with `ω` on the ordinary branch. Writing
`f₀ = F(A²)`, `f₁ = F'(A²)`, `v₂ = V''(A²)`, `v₃ = V'''(A²)`:

* the linear symbol is `linearSymbolRaw` (it reproduces the repository's H4/Q7 polynomial);
* the `cos(2kz)` part of the second-order source is `σ₀ + σ₂ cos(2ωt)` with the polynomials
  `sigma0Raw`, `sigma2Raw` below (pasted verbatim from the `sympy` output).

Results:

* `linearSymbol_factor`, `onShell_symbol_identity`: on shell `f₀ · symbolForm = (ω² − k²) + 2A²v₂`.
* `sigma2_factor`: `σ₂ = (A/2)(3v₂ + 2A²v₃ − 9 f₁ · symbolForm κ θ γ (ω²−k²) k²)`.
* `sigma2_neg`: with admissible curvature, positive amplitude mass (`v₂ < 0`), `v₃ ≤ 0`,
  `f₁ ≥ 0` and `k ≠ 0`, `σ₂ < 0`: the second harmonic is forced at EVERY real-window mode.
* `ghost_mode_forced`: in the Pais–Uhlenbeck normal coordinates of the `2k` mode, the ghost
  coordinate `y₂ = ẍ + z₁x` obeys `ÿ₂ + z₂ y₂ = S/K` with the same source `S` as the ordinary
  coordinate; a nonzero second-order source therefore always excites the ghost branch.
* `resonant_secular_solution`, `secular_amplitude_unbounded`: at the resonance
  `(2ω)² = z₂(4k²)` the exact particular solution `(σ/(2KΩ)) t sin(Ωt)` grows without bound.

Scope: this is second-order perturbation theory around a homogeneous background (exact
coefficients). It proves non-invariance of the restricted linear sectors and the existence of a
ghost-pumping channel; it is not a long-time nonlinear instability theorem.
-/

namespace ReleaseProofs.S004HarmonicGhostForcing

open ReleaseProofs.S004CharacteristicCone

/-- Linear symbol of the real-sector S004 EL operator on `exp(ikz − iωt)` around `x = A`
(`V'(A²) = 0`), verbatim from `sympy`. -/
def linearSymbolRaw (A f0 v2 κ θ γ k ω : ℝ) : ℝ :=
  2 * (2 * A ^ 2 * v2 - f0 * γ * k ^ 4 + f0 * γ * k ^ 2 * ω ^ 2 - f0 * k ^ 4 * κ -
    f0 * k ^ 4 * θ + 2 * f0 * k ^ 2 * κ * ω ^ 2 - f0 * κ * ω ^ 4 - k ^ 2 + ω ^ 2)

/-- `cos(2kz)` time-independent second-order source coefficient, verbatim from `sympy`. -/
noncomputable def sigma0Raw (A f1 v2 v3 κ θ γ k ω : ℝ) : ℝ :=
  A * (2 * A ^ 2 * v3 - 9 * f1 * γ * k ^ 4 + 5 * f1 * γ * k ^ 2 * ω ^ 2 - 9 * f1 * k ^ 4 * κ -
    9 * f1 * k ^ 4 * θ + 10 * f1 * k ^ 2 * κ * ω ^ 2 - f1 * κ * ω ^ 4 + 3 * v2) / 2

/-- `cos(2kz) cos(2ωt)` second-order source coefficient, verbatim from `sympy`. -/
noncomputable def sigma2Raw (A f1 v2 v3 κ θ γ k ω : ℝ) : ℝ :=
  A * (2 * A ^ 2 * v3 - 9 * f1 * γ * k ^ 4 + 9 * f1 * γ * k ^ 2 * ω ^ 2 - 9 * f1 * k ^ 4 * κ -
    9 * f1 * k ^ 4 * θ + 18 * f1 * k ^ 2 * κ * ω ^ 2 - 9 * f1 * κ * ω ^ 4 + 3 * v2) / 2

/-- The linear symbol is the repository dispersion polynomial written through `symbolForm`. -/
theorem linearSymbol_factor (A f0 v2 κ θ γ k ω : ℝ) :
    linearSymbolRaw A f0 v2 κ θ γ k ω =
      2 * ((ω ^ 2 - k ^ 2) + 2 * A ^ 2 * v2 - f0 * symbolForm κ θ γ (ω ^ 2 - k ^ 2) (k ^ 2)) := by
  unfold linearSymbolRaw symbolForm
  ring

/-- On shell, `f₀ · symbolForm = (ω² − k²) + 2A²v₂`. -/
theorem onShell_symbol_identity {A f0 v2 κ θ γ k ω : ℝ}
    (hshell : linearSymbolRaw A f0 v2 κ θ γ k ω = 0) :
    f0 * symbolForm κ θ γ (ω ^ 2 - k ^ 2) (k ^ 2) = (ω ^ 2 - k ^ 2) + 2 * A ^ 2 * v2 := by
  rw [linearSymbol_factor] at hshell
  linarith

/-- The double-frequency source factors through the curvature symbol. -/
theorem sigma2_factor (A f1 v2 v3 κ θ γ k ω : ℝ) :
    sigma2Raw A f1 v2 v3 κ θ γ k ω =
      A / 2 * (3 * v2 + 2 * A ^ 2 * v3 - 9 * f1 * symbolForm κ θ γ (ω ^ 2 - k ^ 2) (k ^ 2)) := by
  unfold sigma2Raw symbolForm
  ring

/-- **The second harmonic is always forced** under admissible curvature, positive amplitude mass
(`v₂ < 0`), `v₃ ≤ 0` and `f₁ ≥ 0`, at every mode `k ≠ 0` (on or off shell). -/
theorem sigma2_neg {A f1 v2 v3 κ θ γ k ω : ℝ} (hA : 0 < A) (hv2 : v2 < 0) (hv3 : v3 ≤ 0)
    (hf1 : 0 ≤ f1) (h : Admissible κ θ γ) (hk : k ≠ 0) :
    sigma2Raw A f1 v2 v3 κ θ γ k ω < 0 := by
  rw [sigma2_factor]
  have hp : 0 < symbolForm κ θ γ (ω ^ 2 - k ^ 2) (k ^ 2) :=
    symbolForm_pos_of_admissible h (Or.inr (pow_ne_zero 2 hk))
  have hin : 3 * v2 + 2 * A ^ 2 * v3 - 9 * f1 * symbolForm κ θ γ (ω ^ 2 - k ^ 2) (k ^ 2) < 0 := by
    have h1 : 2 * A ^ 2 * v3 ≤ 0 := by nlinarith [sq_nonneg A]
    have h2 : 0 ≤ 9 * f1 * symbolForm κ θ γ (ω ^ 2 - k ^ 2) (k ^ 2) := by positivity
    linarith
  exact mul_neg_of_pos_of_neg (by positivity) hin

/-- **Ghost-branch forcing.** For the Pais–Uhlenbeck form `K x⁗ + K(z₁+z₂) ẍ + K z₁z₂ x = S` of a
Fourier mode, the ghost normal coordinate `y₂ = ẍ + z₁ x` satisfies `ÿ₂ + z₂ y₂ = S/K`, with the
same source as the ordinary coordinate `y₁ = ẍ + z₂ x`. -/
theorem ghost_mode_forced {K z1 z2 x0 x2 x4 S : ℝ} (hK : K ≠ 0)
    (hS : R16PU.puEuler K (K * (z1 + z2)) (K * (z1 * z2)) x0 x2 x4 = S) :
    (x4 + z1 * x2) + z2 * (x2 + z1 * x0) = S / K ∧
      (x4 + z2 * x2) + z1 * (x2 + z2 * x0) = S / K := by
  obtain ⟨h1, h2⟩ := R16PU.pu_mode_equations K z1 z2 x0 x2 x4
  constructor
  · rw [eq_div_iff hK, mul_comm, h2, hS]
  · rw [eq_div_iff hK, mul_comm, h1, hS]

/-- Off resonance the forced ghost amplitude `σ/(K(z₂ − Ω²))` vanishes iff the source does. -/
theorem forced_ghost_amplitude_ne_zero {σ K z2 Ω : ℝ} (hK : K ≠ 0) (hres : z2 ≠ Ω ^ 2)
    (hσ : σ ≠ 0) : σ / (K * (z2 - Ω ^ 2)) ≠ 0 :=
  div_ne_zero hσ (mul_ne_zero hK (sub_ne_zero.mpr hres))

/-- First derivative of `t ↦ t sin(Ωt)`. -/
theorem secular_first_deriv (Ω t : ℝ) :
    HasDerivAt (fun s => s * Real.sin (Ω * s))
      (Real.sin (Ω * t) + t * (Real.cos (Ω * t) * Ω)) t := by
  have h2 : HasDerivAt (fun s => Real.sin (Ω * s)) (Real.cos (Ω * t) * Ω) t := by
    have := ((hasDerivAt_id t).const_mul Ω).sin
    simpa using this
  have := (hasDerivAt_id t).mul h2
  simpa using this

/-- Second derivative of `t ↦ t sin(Ωt)`. -/
theorem secular_second_deriv (Ω t : ℝ) :
    HasDerivAt (fun s => Real.sin (Ω * s) + s * (Real.cos (Ω * s) * Ω))
      (2 * Ω * Real.cos (Ω * t) - Ω ^ 2 * (t * Real.sin (Ω * t))) t := by
  have hs : HasDerivAt (fun s => Real.sin (Ω * s)) (Real.cos (Ω * t) * Ω) t := by
    have := ((hasDerivAt_id t).const_mul Ω).sin
    simpa using this
  have hc : HasDerivAt (fun s => Real.cos (Ω * s) * Ω) (-Real.sin (Ω * t) * Ω * Ω) t := by
    have := (((hasDerivAt_id t).const_mul Ω).cos).mul_const Ω
    simpa using this
  have := hs.add ((hasDerivAt_id t).mul hc)
  convert this using 1
  simp
  ring

/-- **Exact resonant secular solution.** `y(t) = c · t sin(Ωt)` satisfies
`ÿ + Ω² y = 2cΩ cos(Ωt)`; with `c = σ/(2KΩ)` this is the response of a mode with
`ÿ + Ω²y = (σ/K) cos(Ωt)`, i.e. a ghost coordinate driven exactly at its own frequency. -/
theorem resonant_secular_solution (c Ω t : ℝ) :
    HasDerivAt (fun s => c * (s * Real.sin (Ω * s)))
        (c * (Real.sin (Ω * t) + t * (Real.cos (Ω * t) * Ω))) t ∧
      HasDerivAt (fun s => c * (Real.sin (Ω * s) + s * (Real.cos (Ω * s) * Ω)))
        (c * (2 * Ω * Real.cos (Ω * t) - Ω ^ 2 * (t * Real.sin (Ω * t)))) t ∧
      c * (2 * Ω * Real.cos (Ω * t) - Ω ^ 2 * (t * Real.sin (Ω * t))) +
          Ω ^ 2 * (c * (t * Real.sin (Ω * t))) = 2 * c * Ω * Real.cos (Ω * t) :=
  ⟨(secular_first_deriv Ω t).const_mul c, (secular_second_deriv Ω t).const_mul c, by ring⟩

/-- The resonant amplitude is unbounded: for `c > 0`, `Ω > 0` it exceeds every bound. -/
theorem secular_amplitude_unbounded {c Ω : ℝ} (hc : 0 < c) (hΩ : 0 < Ω) (M : ℝ) :
    ∃ t : ℝ, M < c * (t * Real.sin (Ω * t)) := by
  obtain ⟨n, hn⟩ := exists_nat_gt (M * Ω / (c * (2 * Real.pi)))
  refine ⟨(Real.pi / 2 + n * (2 * Real.pi)) / Ω, ?_⟩
  have hΩt : Ω * ((Real.pi / 2 + n * (2 * Real.pi)) / Ω) = Real.pi / 2 + n * (2 * Real.pi) := by
    field_simp
  rw [hΩt, Real.sin_add_nat_mul_two_pi, Real.sin_pi_div_two, mul_one]
  have hpi : 0 < Real.pi := Real.pi_pos
  have h2 : M * Ω < n * (c * (2 * Real.pi)) := by
    rw [div_lt_iff₀ (by positivity)] at hn
    linarith
  rw [mul_div_assoc', lt_div_iff₀ hΩ]
  nlinarith [Nat.cast_nonneg (α := ℝ) n]


/-- `HasDerivAt` of `t ↦ cos(ω t)` and of its derivative. -/
theorem cos_first_deriv (ω t : ℝ) :
    HasDerivAt (fun s => Real.cos (ω * s)) (-Real.sin (ω * t) * ω) t := by
  have := ((hasDerivAt_id t).const_mul ω).cos
  simpa using this

theorem cos_second_deriv (ω t : ℝ) :
    HasDerivAt (fun s => -Real.sin (ω * s) * ω) (-(ω ^ 2) * Real.cos (ω * t)) t := by
  have := (((hasDerivAt_id t).const_mul ω).sin.neg).mul_const ω
  convert this using 1
  simp
  ring

/-- **Bounded off-resonant excitation (exact).** For `z > 0`, `Ω² ≠ z`, the zero-data response
`y(t) = c/(z − Ω²)·(cos Ωt − cos √z t)` of a normal mode `ÿ + z y = c cos Ωt` satisfies the equation
and the time-uniform bound `|y(t)| ≤ 2|c|/|z − Ω²|`. -/
theorem offresonant_forced_response (c z Ω : ℝ) (hz : 0 < z) (hres : z ≠ Ω ^ 2) (t : ℝ) :
    HasDerivAt (fun s => c / (z - Ω ^ 2) * (Real.cos (Ω * s) - Real.cos (Real.sqrt z * s)))
        (c / (z - Ω ^ 2) * (-Real.sin (Ω * t) * Ω - -Real.sin (Real.sqrt z * t) * Real.sqrt z)) t ∧
      HasDerivAt
        (fun s => c / (z - Ω ^ 2) * (-Real.sin (Ω * s) * Ω - -Real.sin (Real.sqrt z * s) * Real.sqrt z))
        (c / (z - Ω ^ 2) * (-(Ω ^ 2) * Real.cos (Ω * t) - -(Real.sqrt z ^ 2) *
          Real.cos (Real.sqrt z * t))) t ∧
      c / (z - Ω ^ 2) * (-(Ω ^ 2) * Real.cos (Ω * t) - -(Real.sqrt z ^ 2) * Real.cos (Real.sqrt z * t)) +
          z * (c / (z - Ω ^ 2) * (Real.cos (Ω * t) - Real.cos (Real.sqrt z * t))) =
        c * Real.cos (Ω * t) ∧
      |c / (z - Ω ^ 2) * (Real.cos (Ω * t) - Real.cos (Real.sqrt z * t))| ≤ 2 * |c| / |z - Ω ^ 2| := by
  have hden : z - Ω ^ 2 ≠ 0 := sub_ne_zero.mpr hres
  have hsq : Real.sqrt z ^ 2 = z := Real.sq_sqrt hz.le
  refine ⟨((cos_first_deriv Ω t).sub (cos_first_deriv (Real.sqrt z) t)).const_mul _,
    ((cos_second_deriv Ω t).sub (cos_second_deriv (Real.sqrt z) t)).const_mul _, ?_, ?_⟩
  · rw [hsq]
    field_simp
    ring
  · rw [abs_mul, abs_div]
    have hcos : |Real.cos (Ω * t) - Real.cos (Real.sqrt z * t)| ≤ 2 := by
      have h1 := Real.abs_cos_le_one (Ω * t)
      have h2 := Real.abs_cos_le_one (Real.sqrt z * t)
      calc |Real.cos (Ω * t) - Real.cos (Real.sqrt z * t)|
          ≤ |Real.cos (Ω * t)| + |Real.cos (Real.sqrt z * t)| := abs_sub _ _
        _ ≤ 2 := by linarith
    have hpos : 0 < |z - Ω ^ 2| := abs_pos.mpr hden
    calc |c| / |z - Ω ^ 2| * |Real.cos (Ω * t) - Real.cos (Real.sqrt z * t)|
        ≤ |c| / |z - Ω ^ 2| * 2 := mul_le_mul_of_nonneg_left hcos (by positivity)
      _ = 2 * |c| / |z - Ω ^ 2| := by ring

end ReleaseProofs.S004HarmonicGhostForcing
