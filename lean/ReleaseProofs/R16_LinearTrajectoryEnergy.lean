import Mathlib
import ReleaseProofs.R16_LinearModeReduction

/-!
# Linear PU/S004 dynamics: conserved omitted-mode energy and all-time error bound

This module upgrades the exact four-jet projection identities of
R16_LinearModeReduction to actual C¹ solutions of the original
four-dimensional fourth-order *linear Fourier-mode* evolution.

No initial-data smallness, finite-time truncation, or assumption
that the unstable canonical degree of freedom is absent is hidden:
the omitted oscillator's exact energy is constant on every full
solution; when its squared frequency z2>=1 and z1!=z2, the full
physical displacement/velocity and the projected z1 oscillator
remain uniformly close for all real times, with an explicit gap
factor 1/(z2-z1)^2 and initial omitted energy.

This is NOT a nonlinear A2/WCT PDE reduction or a proof of
ghost-freedom of the full covariant action.
-/

namespace ReleaseProofs.R16LinearModes

/-- Exact C¹ solution predicate for the original PU four-jet ODE.
Its last component is equivalent to the fourth-order PU Euler equation
under K != 0 by the existing exact_pu_euler lemma. -/
def IsTrajectory (z1 z2 : ℝ) (s : ℝ → Jet) : Prop :=
  ∀ t : ℝ,
    HasDerivAt (fun τ => (s τ).x) (s t).v t ∧
    HasDerivAt (fun τ => (s τ).v) (s t).a t ∧
    HasDerivAt (fun τ => (s τ).a) (s t).j t ∧
    HasDerivAt (fun τ => (s τ).j)
      (-(z1 + z2) * (s t).a - z1 * z2 * (s t).x) t

/-- A full four-jet trajectory projects to an EXACT omitted z2 oscillator.
This uses all four of the actual fourth-order evolution equations. -/
theorem omittedMode_trajectory
    (z1 z2 : ℝ) (s : ℝ → Jet)
    (hs : IsTrajectory z1 z2 s) (t : ℝ) :
    HasDerivAt
      (fun τ => (otherMode z1 (s τ)).1)
      ((otherMode z1 (s t)).2) t ∧
    HasDerivAt
      (fun τ => (otherMode z1 (s τ)).2)
      (-z2 * (otherMode z1 (s t)).1) t := by
  obtain ⟨hx, hv, ha, hj⟩ := hs t
  constructor
  · have h := ha.add (hx.const_mul z1)
    simpa [otherMode] using h
  · have h := hj.add (hv.const_mul z1)
    convert h using 1 <;>
      simp [otherMode] <;>
      ring

/-- Along every exact linear four-jet solution the omitted oscillator's
quadratic energy has derivative ZERO at every real time. -/
theorem neglectedEnergy_hasDerivAt_zero
    (z1 z2 : ℝ) (s : ℝ → Jet)
    (hs : IsTrajectory z1 z2 s) (t : ℝ) :
    HasDerivAt
      (fun τ => neglectedEnergy z1 z2 (s τ)) 0 t := by
  obtain ⟨hy, hw⟩ := omittedMode_trajectory z1 z2 s hs t
  have h := (hw.pow 2).add ((hy.pow 2).const_mul z2)
  convert h using 1 <;>
    simp [neglectedEnergy, otherMode] <;>
    ring

/-- The omitted mode's energy is constant on every real-time interval.
Unlike a formal vector-field tangent statement, this is a global
trajectory theorem, requiring no unproved time-existence claim:
the full C¹ solution is explicitly assumed. -/
theorem neglectedEnergy_conserved
    (z1 z2 : ℝ) (s : ℝ → Jet)
    (hs : IsTrajectory z1 z2 s) (t : ℝ) :
    neglectedEnergy z1 z2 (s t) =
      neglectedEnergy z1 z2 (s 0) := by
  let E : ℝ → ℝ := fun τ => neglectedEnergy z1 z2 (s τ)
  have hderiv : ∀ τ : ℝ, HasDerivAt E 0 τ := by
    intro τ
    exact neglectedEnergy_hasDerivAt_zero z1 z2 s hs τ
  have hdifferentiable : Differentiable ℝ E :=
    fun τ => (hderiv τ).differentiableAt
  have hz : ∀ τ : ℝ, deriv E τ = 0 :=
    fun τ => (hderiv τ).deriv
  exact is_const_of_deriv_eq_zero hdifferentiable hz t 0

/-- **Actual trajectory-level error control in the linear S004/PU mode.**
For all real times t, squared position/velocity error after projection
onto the invariant z1 oscillator is bounded by initial omitted-mode
energy divided by the EXACT squared spectral gap.

This does not assume the deleted mode is set to zero. Its amplitude
and the gap determine the approximation quality. -/
theorem uniform_trajectory_error_bound
    (z1 z2 : ℝ) (s : ℝ → Jet)
    (hs : IsTrajectory z1 z2 s)
    (hz : z2 ≠ z1) (hfreq : 1 ≤ z2) (t : ℝ) :
    ((s t).x - (readout z1 z2 (s t)).1) ^ 2 +
        ((s t).v - (readout z1 z2 (s t)).2) ^ 2 ≤
      neglectedEnergy z1 z2 (s 0) / (z2 - z1) ^ 2 := by
  have hbound :=
    reduction_error_le_neglected_energy z1 z2 (s t) hz hfreq
  rw [neglectedEnergy_conserved z1 z2 s hs t] at hbound
  exact hbound

/-! ## Concrete global solution and amplitude-annulus non-invariance -/

/-- An explicit smooth global solution of the exact PU four-jet
equations at z1=1,z2=2, confined to the positive-energy
z1 linear oscillator sector. -/
noncomputable def unitCosineTrajectory (t : ℝ) : Jet where
  x := Real.cos t
  v := -Real.sin t
  a := -Real.cos t
  j := Real.sin t

/-- The full fourth-order linear evolution has an explicit entire-time
classical trajectory; the IsTrajectory predicate is nonvacuous. -/
theorem unitCosineTrajectory_isTrajectory :
    IsTrajectory 1 2 unitCosineTrajectory := by
  intro t
  refine ⟨?_, ?_, ?_, ?_⟩
  · simpa [unitCosineTrajectory] using Real.hasDerivAt_cos t
  · simpa [unitCosineTrajectory] using (Real.hasDerivAt_sin t).neg
  · simpa [unitCosineTrajectory] using (Real.hasDerivAt_cos t).neg
  · convert Real.hasDerivAt_sin t using 1 <;>
      simp [unitCosineTrajectory] <;>
      ring

/-- The positive-energy invariant real oscillator passes through a
zero-field node despite starting with strictly positive field density.
A nonvanishing-amplitude restriction must therefore be justified by
additional dynamics; it is not preserved by this real linear sector. -/
theorem unitCosineTrajectory_crosses_node :
    (unitCosineTrajectory 0).x = 1 ∧
      (unitCosineTrajectory (Real.pi / 2)).x = 0 := by
  constructor
  · simp [unitCosineTrajectory]
  · simp [unitCosineTrajectory, Real.cos_pi_div_two]

/-- For every proposed strictly positive scalar squared-amplitude floor
delta<=1, the explicit FULL PU solution violates that floor at some time. -/
theorem unitCosineTrajectory_no_positive_amplitude_floor
    (delta : ℝ) (hdelta : 0 < delta) (hd1 : delta ≤ 1) :
    delta ≤ (unitCosineTrajectory 0).x ^ 2 ∧
      ¬ ∀ t : ℝ, delta ≤ (unitCosineTrajectory t).x ^ 2 := by
  constructor
  · simpa [unitCosineTrajectory] using hd1
  · intro hall
    have hnode := (unitCosineTrajectory_crosses_node).2
    have hlow := hall (Real.pi / 2)
    rw [hnode] at hlow
    norm_num at hlow
    linarith

end ReleaseProofs.R16LinearModes
