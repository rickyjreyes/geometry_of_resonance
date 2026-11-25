import Mathlib
import ReleaseProofs.R18_FiniteScale

/-!
# Issue #16 (Task 1): exact Pais–Uhlenbeck ghost split of the S004 Fourier mode

Every quadratic, time-reversal-invariant mode Lagrangian in `(x, ẋ, ẍ)` reduces, up to total
derivatives, to

  `L = -(K/2) ẍ² + (A/2) ẋ² - (B/2) x²`.

(`ẋẍ = d/dt(ẋ²/2)`, and `xẍ = d/dt(xẋ) - ẋ²`.) Its Euler–Lagrange equation is
`K x'''' + A x'' + B x = 0`. With `z = ω²`, its frequency polynomial is `-K z² + A z - B`.

* `pu_hamiltonian_eq`, `pu_energy_rate`: the exact Ostrogradsky Hamiltonian on the 4-jet, and
  `dH/dt = ẋ · (Euler–Lagrange)`. So `H` is conserved on solutions.
* `pu_normal_mode_identity`: with `A = K(z₁+z₂)` and `B = K z₁ z₂`,
  `2(z₁ - z₂) H = K[(ẏ₂² + z₂y₂²) - (ẏ₁² + z₁y₁²)]`, where `y₁ = ẍ + z₂x` and `y₂ = ẍ + z₁x`.
  It is an exact polynomial identity on the whole phase space.
* `pu_mode_equations`: on solutions, `ÿ₁ + z₁y₁ = 0` and `ÿ₂ + z₂y₂ = 0`.
* `pu_ghost_unbounded`: for `K ≠ 0` and `z₁ ≠ z₂`, `H` takes values below and above every bound.
  One normal mode always carries negative energy.
* `pu_real_positive_poles`: with `K, A, B > 0` and `A² > 4KB`, both squared frequencies are real,
  positive and distinct. So positive physical frequencies **do not** give a bounded
  Hamiltonian. Linear frequency stability and energy positivity are separate properties.
* `s004_mode_polynomial`: the corrected S004 inverse propagator `R18.inversePropagator` is
  exactly this frequency polynomial with
  `A_q = 1 + 2Kq + Gq` and `B_q = q + m² + (K+G+T)q²`.
* `s004_vieta`, `s004_mode_ghost`: at any wave number with two distinct real poles, the S004 mode
  Hamiltonian splits into opposite-sign normal-mode energies and is unbounded in both directions
  (`K ≠ 0`). This holds for every nonzero overall normalization of the mode Lagrangian.

Scope: linear, single Fourier mode, flat background. This is an exact obstruction to positivity
of the canonical S004 Hamiltonian. It is **not** a proof of nonlinear runaway. A conserved but
indefinite energy does not by itself force growth.
-/

namespace ReleaseProofs.R16PU

/-- `L = -(K/2) a² + (A/2) v² - (B/2) x²`. -/
noncomputable def puLagrangian (K A B x v a : ℝ) : ℝ :=
  -(K / 2) * a ^ 2 + (A / 2) * v ^ 2 - (B / 2) * x ^ 2

/-- `p₂ = ∂L/∂ẍ`. -/
def puP2 (K x2 : ℝ) : ℝ := -K * x2

/-- `p₁ = ∂L/∂ẋ - d/dt ∂L/∂ẍ`, written on the jet `(x, ẋ, ẍ, x''')`. -/
def puP1 (K A x1 x3 : ℝ) : ℝ := A * x1 + K * x3

/-- Ostrogradsky Hamiltonian `H = p₁ẋ + p₂ẍ - L` on the 4-jet `(x₀, x₁, x₂, x₃)`. -/
noncomputable def puHamiltonian (K A B x0 x1 x2 x3 : ℝ) : ℝ :=
  puP1 K A x1 x3 * x1 + puP2 K x2 * x2 - puLagrangian K A B x0 x1 x2

/-- Euler–Lagrange expression `K x'''' + A x'' + B x`. -/
def puEuler (K A B x0 x2 x4 : ℝ) : ℝ := K * x4 + A * x2 + B * x0

theorem pu_hamiltonian_eq (K A B x0 x1 x2 x3 : ℝ) :
    puHamiltonian K A B x0 x1 x2 x3 =
      (A / 2) * x1 ^ 2 + K * x1 * x3 - (K / 2) * x2 ^ 2 + (B / 2) * x0 ^ 2 := by
  unfold puHamiltonian puP1 puP2 puLagrangian
  ring

/-- Chain rule along the jet: `dH/dt = Σ ∂H/∂xᵢ · xᵢ₊₁`, which equals `ẋ · Euler`. -/
theorem pu_energy_rate (K A B x0 x1 x2 x3 x4 : ℝ) :
    B * x0 * x1 + (A * x1 + K * x3) * x2 + (-K * x2) * x3 + (K * x1) * x4 =
      x1 * puEuler K A B x0 x2 x4 := by
  unfold puEuler
  ring

/-- Any nonzero overall normalization `λ` of the Lagrangian rescales `H` by `λ`. -/
theorem pu_hamiltonian_scale (lam K A B x0 x1 x2 x3 : ℝ) :
    puHamiltonian (lam * K) (lam * A) (lam * B) x0 x1 x2 x3 =
      lam * puHamiltonian K A B x0 x1 x2 x3 := by
  rw [pu_hamiltonian_eq, pu_hamiltonian_eq]
  ring

/-- **Exact normal-mode identity** (Pais–Uhlenbeck). -/
theorem pu_normal_mode_identity (K z1 z2 x0 x1 x2 x3 : ℝ) :
    2 * (z1 - z2) * puHamiltonian K (K * (z1 + z2)) (K * (z1 * z2)) x0 x1 x2 x3 =
      K * (((x3 + z1 * x1) ^ 2 + z2 * (x2 + z1 * x0) ^ 2) -
        ((x3 + z2 * x1) ^ 2 + z1 * (x2 + z2 * x0) ^ 2)) := by
  rw [pu_hamiltonian_eq]
  ring

/-- On solutions, `y₁ = ẍ + z₂x` oscillates at `z₁` and `y₂ = ẍ + z₁x` at `z₂`. -/
theorem pu_mode_equations (K z1 z2 x0 x2 x4 : ℝ) :
    K * ((x4 + z2 * x2) + z1 * (x2 + z2 * x0)) =
        puEuler K (K * (z1 + z2)) (K * (z1 * z2)) x0 x2 x4 ∧
      K * ((x4 + z1 * x2) + z2 * (x2 + z1 * x0)) =
        puEuler K (K * (z1 + z2)) (K * (z1 * z2)) x0 x2 x4 := by
  unfold puEuler
  constructor <;> ring

/-- Explicit jets that excite only one normal mode. -/
theorem pu_single_mode_values (K z1 z2 Y : ℝ) (hz : z1 ≠ z2) :
    puHamiltonian K (K * (z1 + z2)) (K * (z1 * z2)) 0 (Y / (z2 - z1)) 0
        (-z1 * (Y / (z2 - z1))) = -(K * Y ^ 2) / (2 * (z1 - z2)) ∧
      puHamiltonian K (K * (z1 + z2)) (K * (z1 * z2)) 0 (Y / (z1 - z2)) 0
        (-z2 * (Y / (z1 - z2))) = (K * Y ^ 2) / (2 * (z1 - z2)) := by
  have h1 : z1 - z2 ≠ 0 := sub_ne_zero.mpr hz
  have h2 : z2 - z1 ≠ 0 := sub_ne_zero.mpr (Ne.symm hz)
  constructor
  · rw [pu_hamiltonian_eq]
    field_simp
    ring
  · rw [pu_hamiltonian_eq]
    field_simp
    ring

/-- **Ghost unboundedness.** With `K ≠ 0` and distinct squared frequencies, `H` is unbounded
below and above. -/
theorem pu_ghost_unbounded (K z1 z2 : ℝ) (hK : K ≠ 0) (hz : z1 ≠ z2) (bound : ℝ) :
    (∃ x0 x1 x2 x3 : ℝ,
        puHamiltonian K (K * (z1 + z2)) (K * (z1 * z2)) x0 x1 x2 x3 < bound) ∧
      (∃ x0 x1 x2 x3 : ℝ,
        bound < puHamiltonian K (K * (z1 + z2)) (K * (z1 * z2)) x0 x1 x2 x3) := by
  set c : ℝ := K / (2 * (z1 - z2)) with hc
  have h1 : z1 - z2 ≠ 0 := sub_ne_zero.mpr hz
  have hc0 : c ≠ 0 := div_ne_zero hK (mul_ne_zero two_ne_zero h1)
  have hval : ∀ Y : ℝ, -(K * Y ^ 2) / (2 * (z1 - z2)) = -(c * Y ^ 2) ∧
      (K * Y ^ 2) / (2 * (z1 - z2)) = c * Y ^ 2 := fun Y => by
    rw [hc]; constructor <;> ring
  -- Pick `Y² = |bound| + 1 / |c|` scale.
  set Y : ℝ := Real.sqrt ((|bound| + 1) / |c|) with hY
  have hcpos : 0 < |c| := abs_pos.mpr hc0
  have hY2 : Y ^ 2 = (|bound| + 1) / |c| := by
    rw [hY, Real.sq_sqrt]; positivity
  have hcY : |c| * Y ^ 2 = |bound| + 1 := by
    rw [hY2]; field_simp
  have hb1 : bound < |bound| + 1 := by linarith [le_abs_self bound]
  have hb2 : -(|bound| + 1) < bound := by linarith [neg_abs_le bound]
  obtain ⟨vA, vB⟩ := pu_single_mode_values K z1 z2 Y hz
  rcases lt_or_gt_of_ne hc0 with hneg | hpos
  · -- c < 0: the second mode is the negative one.
    have habs : |c| = -c := abs_of_neg hneg
    refine ⟨⟨0, Y / (z1 - z2), 0, -z2 * (Y / (z1 - z2)), ?_⟩,
      ⟨0, Y / (z2 - z1), 0, -z1 * (Y / (z2 - z1)), ?_⟩⟩
    · rw [vB, (hval Y).2]; nlinarith
    · rw [vA, (hval Y).1]; nlinarith
  · have habs : |c| = c := abs_of_pos hpos
    refine ⟨⟨0, Y / (z2 - z1), 0, -z1 * (Y / (z2 - z1)), ?_⟩,
      ⟨0, Y / (z1 - z2), 0, -z2 * (Y / (z1 - z2)), ?_⟩⟩
    · rw [vA, (hval Y).1]; nlinarith
    · rw [vB, (hval Y).2]; nlinarith

/-- Root formula for the frequency polynomial `-K z² + A z - B`. -/
noncomputable def puRoot (K A B s : ℝ) : ℝ :=
  (A + s * Real.sqrt (A ^ 2 - 4 * K * B)) / (2 * K)

/-- **Positive real distinct frequencies.** With `K, A, B > 0` and `A² > 4KB`, both roots
`z± = (A ± √(A²-4KB)) / 2K` are positive and distinct. They solve the frequency polynomial and
satisfy `A = K(z₊+z₋)` and `B = K z₊ z₋`. -/
theorem pu_real_positive_poles (K A B : ℝ) (hK : 0 < K) (hA : 0 < A) (hB : 0 < B)
    (hd : 4 * K * B < A ^ 2) :
    let zp := puRoot K A B 1
    let zm := puRoot K A B (-1)
    0 < zm ∧ zm < zp ∧ A = K * (zp + zm) ∧ B = K * (zp * zm) ∧
      -K * zp ^ 2 + A * zp - B = 0 ∧ -K * zm ^ 2 + A * zm - B = 0 := by
  intro zp zm
  set s := Real.sqrt (A ^ 2 - 4 * K * B) with hs
  have hD : 0 < A ^ 2 - 4 * K * B := by linarith
  have hs2 : s ^ 2 = A ^ 2 - 4 * K * B := Real.sq_sqrt hD.le
  have hspos : 0 < s := Real.sqrt_pos.mpr hD
  have hsA : s < A := by
    have : s ^ 2 < A ^ 2 := by rw [hs2]; nlinarith
    nlinarith
  have hK2 : (2 * K) ≠ 0 := by positivity
  have ezp : zp = (A + s) / (2 * K) := by simp [zp, puRoot, hs]
  have ezm : zm = (A - s) / (2 * K) := by simp [zm, puRoot, hs]; ring
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ezm]; apply div_pos <;> linarith
  · rw [ezm, ezp]; apply div_lt_div_of_pos_right _ (by positivity); linarith
  · rw [ezm, ezp]; field_simp; ring
  · rw [ezm, ezp]; field_simp; nlinarith [hs2]
  · rw [ezp]; field_simp; nlinarith [hs2]
  · rw [ezm]; field_simp; nlinarith [hs2]

/-! ## The corrected S004 mode -/

/-- S004 mode coefficients at wave-number variable `q` (with `y = z - q`, as in Q6). -/
def s004A (K G q : ℝ) : ℝ := 1 + 2 * K * q + G * q
def s004B (K G T massSq q : ℝ) : ℝ := q + massSq + (K + G + T) * q ^ 2

/-- The corrected S004 inverse propagator is **exactly** the Pais–Uhlenbeck frequency
polynomial with the coefficients above. -/
theorem s004_mode_polynomial (K G T massSq z q : ℝ) :
    R18.inversePropagator K G T massSq z q =
      -K * z ^ 2 + s004A K G q * z - s004B K G T massSq q := by
  unfold R18.inversePropagator s004A s004B
  ring

/-- Vieta from two distinct real roots of the S004 polynomial. No extra assumption is needed. -/
theorem s004_vieta (K G T massSq q z1 z2 : ℝ) (hz : z1 ≠ z2)
    (h1 : R18.inversePropagator K G T massSq z1 q = 0)
    (h2 : R18.inversePropagator K G T massSq z2 q = 0) :
    s004A K G q = K * (z1 + z2) ∧ s004B K G T massSq q = K * (z1 * z2) := by
  rw [s004_mode_polynomial] at h1 h2
  have hsub : (z1 - z2) * (s004A K G q - K * (z1 + z2)) = 0 := by linear_combination h1 - h2
  have hA : s004A K G q = K * (z1 + z2) := by
    have := (mul_eq_zero.mp hsub).resolve_left (sub_ne_zero.mpr hz)
    linarith
  refine ⟨hA, ?_⟩
  rw [hA] at h1
  linear_combination -h1

/-- **S004 mode ghost.** At any wave number where the corrected S004 polynomial has two distinct
real roots, the canonical mode Hamiltonian, with any nonzero overall normalization `λ`, is
unbounded below and above (`K ≠ 0`). -/
theorem s004_mode_ghost (lam K G T massSq q z1 z2 : ℝ) (hlam : lam ≠ 0) (hK : K ≠ 0)
    (hz : z1 ≠ z2)
    (h1 : R18.inversePropagator K G T massSq z1 q = 0)
    (h2 : R18.inversePropagator K G T massSq z2 q = 0) (bound : ℝ) :
    (∃ x0 x1 x2 x3 : ℝ, puHamiltonian (lam * K) (lam * s004A K G q)
        (lam * s004B K G T massSq q) x0 x1 x2 x3 < bound) ∧
      (∃ x0 x1 x2 x3 : ℝ, bound < puHamiltonian (lam * K) (lam * s004A K G q)
        (lam * s004B K G T massSq q) x0 x1 x2 x3) := by
  obtain ⟨hA, hB⟩ := s004_vieta K G T massSq q z1 z2 hz h1 h2
  have hscale : ∀ x0 x1 x2 x3, puHamiltonian (lam * K) (lam * s004A K G q)
      (lam * s004B K G T massSq q) x0 x1 x2 x3 =
      puHamiltonian (lam * K) (lam * K * (z1 + z2)) (lam * K * (z1 * z2)) x0 x1 x2 x3 := by
    intro x0 x1 x2 x3
    rw [hA, hB, mul_assoc, mul_assoc]
  simp only [hscale]
  exact pu_ghost_unbounded (lam * K) z1 z2 (mul_ne_zero hlam hK) hz bound

/-- In the real-pole region the two Q7 poles are distinct, so `s004_mode_ghost` applies to
them. -/
theorem s004_poles_distinct (K G T massSq q : ℝ) (hK : K ≠ 0)
    (hd : 0 < R18.poleDiscriminant K G T massSq q) :
    R18.lowerPole K G T massSq q ≠ R18.upperPole K G T massSq q := by
  intro h
  unfold R18.lowerPole R18.upperPole at h
  have hs : 0 < Real.sqrt (R18.poleDiscriminant K G T massSq q) := Real.sqrt_pos.mpr hd
  have h2 : (1 + G * q - Real.sqrt (R18.poleDiscriminant K G T massSq q)) / (2 * K) =
      (1 + G * q + Real.sqrt (R18.poleDiscriminant K G T massSq q)) / (2 * K) := by linarith
  have hK2 : 2 * K ≠ 0 := mul_ne_zero two_ne_zero hK
  rw [div_left_inj' hK2] at h2
  linarith

theorem s004_real_window_ghost (lam K G T massSq q : ℝ) (hlam : lam ≠ 0) (hK : K ≠ 0)
    (hd : 0 < R18.poleDiscriminant K G T massSq q) (bound : ℝ) :
    (∃ x0 x1 x2 x3 : ℝ, puHamiltonian (lam * K) (lam * s004A K G q)
        (lam * s004B K G T massSq q) x0 x1 x2 x3 < bound) ∧
      (∃ x0 x1 x2 x3 : ℝ, bound < puHamiltonian (lam * K) (lam * s004A K G q)
        (lam * s004B K G T massSq q) x0 x1 x2 x3) :=
  s004_mode_ghost lam K G T massSq q _ _ hlam hK (s004_poles_distinct K G T massSq q hK hd)
    (R18.lower_pole_on_shell K G T massSq q hK hd.le)
    (R18.upper_pole_on_shell K G T massSq q hK hd.le) bound

end ReleaseProofs.R16PU
