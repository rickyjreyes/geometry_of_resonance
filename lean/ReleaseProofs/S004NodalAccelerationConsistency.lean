import Mathlib
import ReleaseProofs.S004NodalHolonomic
import ReleaseProofs.S004NodalPrimary

/-!
# Phase XXIV: second-order kinematic consistency at the S004 node

The original corrected S004 nodal PRIMARY constraint surface is
q=0 and p2=b(q,v). The existing verified analysis proves that an
entirely nodal classical path obeying qdot=v must satisfy v=0.

This module adds the genuinely distinct second-order requirement:
if the same classical path satisfies vdot=a, then its acceleration
must vanish as well. Thus a nonzero acceleration cannot be part of
a permanently nodal classical path, even though the nodal primary
acceleration Legendre map alone admits arbitrary accelerations.

These are kinematic necessary conditions. They are NOT an
original-model nonlinear Dirac algorithm, a proof of nodal dynamical
invariance or a healthy-sector stability theorem.
-/

namespace ReleaseProofs.S004NodalAccelerationConsistency

open ReleaseProofs.R16Class
open ReleaseProofs.S004NodalEnergyObstruction
open ReleaseProofs.S004NodalHolonomic

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The acceleration of an entirely nodal genuine classical path
must be zero when BOTH actual kinematic derivative identities hold.
The primary nodal Legendre map alone does not impose this condition. -/
theorem permanently_nodal_path_acceleration_zero
    (b : E → E → E) (z : ℝ → Phase E) (accel : ℝ → E)
    (hPrimary : ∀ t, z t ∈ nodalPrimarySurface b)
    (hQdot : ∀ t,
      HasDerivAt (fun s => (z s).q) (z t).v t)
    (hVdot : ∀ t,
      HasDerivAt (fun s => (z s).v) (accel t) t) :
    ∀ t, accel t = 0 := by
  have hHol := permanently_nodal_path_is_holonomic
    b z hPrimary hQdot
  have hvel : (fun s => (z s).v) = (fun _ => (0 : E)) := by
    funext t
    exact (hHol t).2.1
  intro t
  have hd := hVdot t
  rw [hvel] at hd
  simpa using hd.deriv.symm

/-- The true nodal kinematics require both zero canonical velocity
and zero acceleration, not merely the degenerate nodal acceleration
Hessian. -/
theorem permanently_nodal_path_velocity_and_acceleration_zero
    (b : E → E → E) (z : ℝ → Phase E) (accel : ℝ → E)
    (hPrimary : ∀ t, z t ∈ nodalPrimarySurface b)
    (hQdot : ∀ t,
      HasDerivAt (fun s => (z s).q) (z t).v t)
    (hVdot : ∀ t,
      HasDerivAt (fun s => (z s).v) (accel t) t) :
    ∀ t, (z t).v = 0 ∧ accel t = 0 := by
  intro t
  exact ⟨(permanently_nodal_path_is_holonomic
    b z hPrimary hQdot t).2.1,
    permanently_nodal_path_acceleration_zero
      b z accel hPrimary hQdot hVdot t⟩

/-- Any genuine classical trajectory with a nonzero acceleration at
even ONE time cannot remain on the original S004 nodal primary
constraint surface for all real times. This is an exit NECESSITY,
not an existence or flow-preservation assertion. -/
theorem nonzero_acceleration_requires_nodal_exit
    (b : E → E → E) (z : ℝ → Phase E) (accel : ℝ → E)
    (hQdot : ∀ t,
      HasDerivAt (fun s => (z s).q) (z t).v t)
    (hVdot : ∀ t,
      HasDerivAt (fun s => (z s).v) (accel t) t)
    (t0 : ℝ) (hne : accel t0 ≠ 0) :
    ∃ t : ℝ, z t ∉ nodalPrimarySurface b := by
  by_contra hn
  push_neg at hn
  have hzero := permanently_nodal_path_acceleration_zero
    b z accel hn hQdot hVdot t0
  exact hne hzero

end ReleaseProofs.S004NodalAccelerationConsistency
