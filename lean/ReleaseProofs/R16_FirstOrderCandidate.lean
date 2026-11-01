import Mathlib
import ReleaseProofs.R16_PaisUhlenbeck

/-!
# Issue #16 / #45 (Task 1): first-order-in-time candidate class, separate from S004

**Independent replacement model, not S004.** The candidate is the fourth-order cubic–quintic
Schrödinger flow

  `i ∂ₜu = (Δ + k²)² u - β|u|²u + γ|u|⁴u`,  `u(t) ∈ H²(ℝ²; ℂ)`,

with Hamiltonian `H(u) = ‖(Δ+k²)u‖² - (β/2)‖u‖⁴_{L⁴} + (γ/3)‖u‖⁶_{L⁶}` and mass `M(u) = ‖u‖²`.
It comes from the first-order Lagrangian `Im⟨u, ∂ₜu⟩ - H(u)`.

This file proves the exact *structural* facts at the level of one Fourier mode and pointwise
densities:

* `candidate_lagrangian_affine`: the mode Lagrangian is affine in the velocities. The velocity
  Hessian is identically zero and no acceleration occurs. The phase space is `(x, y)` itself, so
  there is no Ostrogradsky momentum `p₁`. **One** real degree of freedom per complex mode,
  against **two** per real field per mode for S004 (`R16PU`).
* `candidate_symbol_nonneg`, `candidate_symbol_zero_iff`, `candidate_symbol_unique_min`: the
  linear frequency `ω(q) = (k² - q)²` (`q = |ξ|²`) is `≥ 0` and vanishes exactly at `q = k²`.
  This is an exact finite-wavelength selection `|ξ| = |k|` in the linear spectrum.
* `candidate_mode_energy_nonneg`: every linear mode has nonnegative energy.
* `candidate_density_lower`: `-(β/2) r² + (γ/3) r³ ≥ -(3β²/(16γ)) r` for `r = |u|² ≥ 0`. This gives
  the mass-shell bound `H ≥ -(3β²/16γ) M`.
* `candidate_vs_s004_dof`: the S004 frequency polynomial in `z = ω²` has degree two when
  `K ≠ 0`. It has two distinct pole branches on the real window, and `R16PU.s004_mode_ghost`
  makes one of them a ghost. The candidate has a single branch.

**Where the continuum results live (not re-proved here).** Existence of constrained minimizers
for `0 < m ≤ m_B = 3β/(γ√(k⁴ + 3β²/(32γ)))`, strict binding, compactness modulo translations,
global `H²` well-posedness with conservation, and orbital stability of the minimizing set are
`ReyesMath.OrbitalStability.main` and `ReyesMath.OrbitalStability.global_flow` in
`rickyjreyes/math`, `lean/MathProofs/OrbitalStability.lean`. Those statements are about exactly
the `H`, `M` and flow written above. Their dependency audit is Task 2. This repository does not
import them, so nothing here is claimed to be kernel-checked *in this build*.

**Non-claims.** No theorem says the candidate is derived from, equivalent to, or a limit of S004
or any WCT action. No physical scale is selected (`k` is a free parameter). No claim is made
about uniqueness, radial symmetry or positivity of minimizers.
-/

namespace ReleaseProofs.R16Candidate

/-- First-order mode Lagrangian for `c = x + i y` with frequency `w`:
`L = ½(x ẏ - y ẋ) - w (x² + y²)`. Sign conventions follow `i ċ = w c` up to orientation. -/
noncomputable def candidateLagrangian (w x y vx vy : ℝ) : ℝ :=
  (1 / 2) * (x * vy - y * vx) - w * (x ^ 2 + y ^ 2)

/-- Affine in the velocities, with zero quadratic part: the velocity Hessian vanishes. -/
theorem candidate_lagrangian_affine (w x y vx vy hx hy : ℝ) :
    candidateLagrangian w x y (vx + hx) (vy + hy) =
      candidateLagrangian w x y vx vy + ((1 / 2) * (x * hy - y * hx)) := by
  unfold candidateLagrangian
  ring

/-- Partial derivatives of `candidateLagrangian`, computed exactly from its definition:
`∂L/∂vx = -y/2`, `∂L/∂vy = x/2`, `∂L/∂x = vy/2 - 2wx`, `∂L/∂y = -vx/2 - 2wy`. -/
theorem candidate_partials (w x y vx vy h : ℝ) :
    candidateLagrangian w x y (vx + h) vy - candidateLagrangian w x y vx vy = (-y / 2) * h ∧
    candidateLagrangian w x y vx (vy + h) - candidateLagrangian w x y vx vy = (x / 2) * h ∧
    candidateLagrangian w (x + h) y vx vy - candidateLagrangian w x y vx vy =
      (vy / 2 - 2 * w * x) * h - w * h ^ 2 ∧
    candidateLagrangian w x (y + h) vx vy - candidateLagrangian w x y vx vy =
      (-vx / 2 - 2 * w * y) * h - w * h ^ 2 := by
  unfold candidateLagrangian
  refine ⟨?_, ?_, ?_, ?_⟩ <;> ring

/-- **First-order Euler–Lagrange system.** With the partials above, the Euler–Lagrange
equations `d/dt(∂L/∂vx) = ∂L/∂x` and `d/dt(∂L/∂vy) = ∂L/∂y` read
`-ẏ/2 = ẏ/2 - 2wx` and `ẋ/2 = -ẋ/2 - 2wy`, i.e. exactly `ẏ = 2wx`, `ẋ = -2wy`. These are first
order: no acceleration and no extra initial data. -/
theorem candidate_euler_lagrange_iff (w x y vx vy : ℝ) :
    (-vy / 2 = vy / 2 - 2 * w * x ∧ vx / 2 = -vx / 2 - 2 * w * y) ↔
      (vy = 2 * w * x ∧ vx = -2 * w * y) := by
  constructor
  · rintro ⟨h1, h2⟩; constructor <;> linarith
  · rintro ⟨h1, h2⟩; constructor <;> linarith

/-- The mode energy `w (x² + y²)` is conserved along the first-order flow:
`d/dt [w (x² + y²)] = w (2x ẋ + 2y ẏ) = 0` when `ẋ = -2wy`, `ẏ = 2wx`. -/
theorem candidate_energy_conserved (w x y : ℝ) :
    w * (2 * x * (-2 * w * y) + 2 * y * (2 * w * x)) = 0 := by ring

/-- Linear Fourier symbol of `(Δ + k²)²`, with `q = |ξ|²`. -/
def candidateSymbol (k q : ℝ) : ℝ := (k ^ 2 - q) ^ 2

theorem candidate_symbol_nonneg (k q : ℝ) : 0 ≤ candidateSymbol k q := sq_nonneg _

theorem candidate_symbol_zero_iff (k q : ℝ) : candidateSymbol k q = 0 ↔ q = k ^ 2 := by
  unfold candidateSymbol
  constructor
  · intro h
    have := pow_eq_zero_iff (n := 2) (by norm_num) |>.mp h
    linarith
  · intro h; rw [h]; ring

/-- Unique global minimiser over `q ≥ 0`, attained at the finite value `q = k²`. -/
theorem candidate_symbol_unique_min (k q : ℝ) (hq : q ≠ k ^ 2) :
    candidateSymbol k (k ^ 2) < candidateSymbol k q := by
  have h0 : candidateSymbol k (k ^ 2) = 0 := (candidate_symbol_zero_iff k _).mpr rfl
  rw [h0]
  rcases (candidate_symbol_nonneg k q).lt_or_eq with h | h
  · exact h
  · exact absurd ((candidate_symbol_zero_iff k q).mp h.symm) hq

/-- Mode energy `ω(q) |c|²`. -/
def candidateModeEnergy (k q x y : ℝ) : ℝ := candidateSymbol k q * (x ^ 2 + y ^ 2)

theorem candidate_mode_energy_nonneg (k q x y : ℝ) : 0 ≤ candidateModeEnergy k q x y :=
  mul_nonneg (candidate_symbol_nonneg k q) (by positivity)

/-- Pointwise density bound behind `H ≥ T - (3β²/16γ) M`. -/
theorem candidate_density_lower (β γ r : ℝ) (hγ : 0 < γ) (hr : 0 ≤ r) :
    -(3 * β ^ 2 / (16 * γ)) * r ≤ -(β / 2) * r ^ 2 + (γ / 3) * r ^ 3 := by
  have hid : -(β / 2) * r ^ 2 + (γ / 3) * r ^ 3 =
      (γ / 3) * r * (r - 3 * β / (4 * γ)) ^ 2 - (3 * β ^ 2 / (16 * γ)) * r := by
    field_simp
    ring
  rw [hid]
  have : 0 ≤ (γ / 3) * r * (r - 3 * β / (4 * γ)) ^ 2 := by positivity
  linarith

/-- The bound is sharp: equality at the density `r = 3β/(4γ)`. -/
theorem candidate_density_sharp (β γ : ℝ) (hγ : 0 < γ) :
    -(β / 2) * (3 * β / (4 * γ)) ^ 2 + (γ / 3) * (3 * β / (4 * γ)) ^ 3 =
      -(3 * β ^ 2 / (16 * γ)) * (3 * β / (4 * γ)) := by
  field_simp
  ring

/-- **Separation from S004 (degree count).** With `K ≠ 0`, the S004 polynomial is genuinely
quadratic in `z = ω²`: two distinct `z` cannot have the same value and the same slope. At a
real-window wave number it has two distinct root branches, and `R16PU` makes one of them carry
negative energy. The candidate's dispersion `ω = (k² - q)²` is a single branch. -/
theorem candidate_vs_s004_dof (K G T massSq q : ℝ) (hK : K ≠ 0)
    (hd : 0 < R18.poleDiscriminant K G T massSq q) :
    R18.lowerPole K G T massSq q ≠ R18.upperPole K G T massSq q ∧
      R18.inversePropagator K G T massSq (R18.lowerPole K G T massSq q) q = 0 ∧
      R18.inversePropagator K G T massSq (R18.upperPole K G T massSq q) q = 0 :=
  ⟨R16PU.s004_poles_distinct K G T massSq q hK hd,
    R18.lower_pole_on_shell K G T massSq q hK hd.le,
    R18.upper_pole_on_shell K G T massSq q hK hd.le⟩

end ReleaseProofs.R16Candidate
