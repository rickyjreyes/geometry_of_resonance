import Mathlib
import ReleaseProofs.R16_PaisUhlenbeck

/-!
# Exact invariant single-mode rails in the LINEAR S004/Pais-Uhlenbeck Fourier system

This is an explicit four-jet vector-field conjugacy, not a ghost-free
reduction of the full nonlinear covariant A2 action.

For the distinct squared frequencies z1,z2 the exact normalized modal ODE is
  x' = v, v' = a, a' = j,
  j' = -(z1+z2)*a-z1*z2*x.
Its Euler expression agrees with the original PU mode with
  A=K(z1+z2), B=K z1 z2, K nonzero.

Each two-dimensional single-mode plane is invariant at the VECTOR FIELD level.
For the z1-mode, the neglected z2 amplitude is measured by
  y2 = a+z1*x, w2 = j+z1*v.
The exact initial-state reconstruction error is y2/(z2-z1),
w2/(z2-z1). The neglected mode's quadratic oscillator energy has
identically zero derivative along the full four-dimensional ODE.

These are necessary ingredients for a controlled reduction. Nonlinear A2
invariance, a functional-analytic PDE reduction and a uniform physical
error bound remain open.
-/

namespace ReleaseProofs.R16LinearModes

structure Jet where
  x : ℝ
  v : ℝ
  a : ℝ
  j : ℝ

/-- First-order vector field equivalent to the actual fourth-order
Pais-Uhlenbeck Euler equation at the two stated roots. -/
def vectorField (z1 z2 : ℝ) (s : Jet) : Jet where
  x := s.v
  v := s.a
  a := s.j
  j := -(z1 + z2) * s.a - (z1 * z2) * s.x

/-- The z-mode invariant plane: x'' = -z x and x''' = -z x'. -/
def InMode (z : ℝ) (s : Jet) : Prop :=
  s.a + z * s.x = 0 ∧ s.j + z * s.v = 0

/-- The ordinary two-dimensional oscillator vector field. -/
def oscillator (z : ℝ) (u : ℝ × ℝ) : ℝ × ℝ :=
  (u.2, -z * u.1)

/-- Embed one oscillator into the full fourth-order canonical jet. -/
def embedMode (z : ℝ) (u : ℝ × ℝ) : Jet where
  x := u.1
  v := u.2
  a := -z * u.1
  j := -z * u.2

theorem embedMode_invariant_constraint (z : ℝ) (u : ℝ × ℝ) :
    InMode z (embedMode z u) := by
  constructor <;> simp [InMode, embedMode]

/-- Exact equality of vector fields on the z1-mode. Thus the linear
single-mode reduction does NOT alter the fourth-order mode equations. -/
theorem vectorField_conjugate_first (z1 z2 : ℝ) (u : ℝ × ℝ) :
    vectorField z1 z2 (embedMode z1 u) =
      embedMode z1 (oscillator z1 u) := by
  cases u with
  | mk x v =>
    ext <;> simp [vectorField, embedMode, oscillator] <;> ring

theorem vectorField_conjugate_second (z1 z2 : ℝ) (u : ℝ × ℝ) :
    vectorField z1 z2 (embedMode z2 u) =
      embedMode z2 (oscillator z2 u) := by
  cases u with
  | mk x v =>
    ext <;> simp [vectorField, embedMode, oscillator] <;> ring

/-- The z1-mode plane is tangent to the full fourth-order vector field. -/
theorem first_mode_tangent (z1 z2 : ℝ) (s : Jet)
    (hs : InMode z1 s) :
    InMode z1 (vectorField z1 z2 s) := by
  obtain ⟨ha, hj⟩ := hs
  constructor
  · exact hj
  · change (-(z1 + z2) * s.a - z1 * z2 * s.x) + z1 * s.a = 0
    linear_combination (-z2) * ha

/-- The second single-mode plane is likewise tangent. -/
theorem second_mode_tangent (z1 z2 : ℝ) (s : Jet)
    (hs : InMode z2 s) :
    InMode z2 (vectorField z1 z2 s) := by
  obtain ⟨ha, hj⟩ := hs
  constructor
  · exact hj
  · change (-(z1 + z2) * s.a - z1 * z2 * s.x) + z2 * s.a = 0
    linear_combination (-z1) * ha

/-- The vector field solves the exact original PU fourth-order
Euler expression, not an unrelated second-order oscillator. -/
theorem exact_pu_euler (K z1 z2 : ℝ) (s : Jet) :
    R16PU.puEuler K (K * (z1 + z2)) (K * (z1 * z2))
      s.x s.a (vectorField z1 z2 s).j = 0 := by
  simp [R16PU.puEuler, vectorField]
  ring

/-- Defect from the z1-mode, following the z2 oscillator equation. -/
def otherMode (z1 : ℝ) (s : Jet) : ℝ × ℝ :=
  (s.a + z1 * s.x, s.j + z1 * s.v)

/-- The omitted mode evolves by the exact z2 oscillator vector field. -/
theorem otherMode_intertwines (z1 z2 : ℝ) (s : Jet) :
    otherMode z1 (vectorField z1 z2 s) =
      oscillator z2 (otherMode z1 s) := by
  apply Prod.ext
  · simp [otherMode, vectorField, oscillator]
  · simp [otherMode, vectorField, oscillator]
    ring

/-- The z1 readout reconstructed from the fourth-order jet. -/
noncomputable def readout (z1 z2 : ℝ) (s : Jet) : ℝ × ℝ :=
  ((s.a + z2 * s.x) / (z2 - z1),
   (s.j + z2 * s.v) / (z2 - z1))

/-- Exact reconstruction error in displacement. -/
theorem displacement_reconstruction_error
    (z1 z2 : ℝ) (hz : z2 ≠ z1) (s : Jet) :
    s.x - (readout z1 z2 s).1 =
      -(otherMode z1 s).1 / (z2 - z1) := by
  have hden : z2 - z1 ≠ 0 := sub_ne_zero.mpr hz
  simp only [readout, otherMode, Prod.fst]
  field_simp
  ring

/-- Exact reconstruction error in velocity. -/
theorem velocity_reconstruction_error
    (z1 z2 : ℝ) (hz : z2 ≠ z1) (s : Jet) :
    s.v - (readout z1 z2 s).2 =
      -(otherMode z1 s).2 / (z2 - z1) := by
  have hden : z2 - z1 ≠ 0 := sub_ne_zero.mpr hz
  simp only [readout, otherMode, Prod.snd]
  field_simp
  ring

/-- Squared reconstruction error, with no big-O suppression and with
the mode separation appearing in the denominator. -/
theorem squared_reconstruction_error (z1 z2 : ℝ)
    (hz : z2 ≠ z1) (s : Jet) :
    (s.x - (readout z1 z2 s).1) ^ 2 +
        (s.v - (readout z1 z2 s).2) ^ 2 =
      ((otherMode z1 s).1 ^ 2 + (otherMode z1 s).2 ^ 2) /
        (z2 - z1) ^ 2 := by
  rw [displacement_reconstruction_error z1 z2 hz,
    velocity_reconstruction_error z1 z2 hz]
  ring

/-- The omitted z2-mode energy. -/
def neglectedEnergy (z1 z2 : ℝ) (s : Jet) : ℝ :=
  (otherMode z1 s).2 ^ 2 + z2 * (otherMode z1 s).1 ^ 2

/-- Exact energy derivative along the linear PU vector field, without
assuming positivity or stability of the chosen frequencies. -/
theorem neglectedEnergy_directional_zero (z1 z2 : ℝ) (s : Jet) :
    2 * (otherMode z1 s).2 * (otherMode z1 (vectorField z1 z2 s)).2 +
      2 * z2 * (otherMode z1 s).1 *
        (otherMode z1 (vectorField z1 z2 s)).1 = 0 := by
  rw [otherMode_intertwines]
  simp only [otherMode, oscillator, Prod.fst, Prod.snd]
  ring

/-- The omitted-mode energy is nonnegative when its squared frequency
is nonnegative, and is zero only for a zero omitted mode if z2 > 0. -/
theorem neglectedEnergy_nonnegative
    (z1 z2 : ℝ) (s : Jet) (hz : 0 ≤ z2) :
    0 ≤ neglectedEnergy z1 z2 s := by
  unfold neglectedEnergy
  positivity

end ReleaseProofs.R16LinearModes
