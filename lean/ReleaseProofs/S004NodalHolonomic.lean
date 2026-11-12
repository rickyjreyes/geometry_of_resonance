import Mathlib
import ReleaseProofs.S004NodalEnergyObstruction
import ReleaseProofs.S004NodalPrimary

/-!
# Phase XXII: actual nodal kinematics and the holonomic energy fibre

The corrected S004 nodal acceleration Legendre equation enforces the
primary condition p₂=b(q,v) at q=0. This by itself admits v ≠ 0
and the Ostrogradsky energy is unbounded above and below.

A CLASSICAL path which remains at the field node for all times must
also satisfy v=q̇=0. This module identifies exactly the resulting
holonomic nodal subset and proves that the entire unrestricted p₁
fibre has one canonical energy value G(0,0,b(0,0)).

The kinematic result uses real HasDerivAt, not a fabricated Hamiltonian
flow. It neither derives the secondary Dirac constraints nor proves
invariance, well-posedness or physical stability of this subset.
-/

namespace ReleaseProofs.S004NodalHolonomic

open ReleaseProofs.R16Class
open ReleaseProofs.S004NodalEnergyObstruction

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- A strictly nodal classical field path must have q=0, q̇=v=0,
and the exact corrected nodal higher momentum p₂=b(0,0).
The first momentum p₁ is deliberately NOT restricted. -/
def nodalHolonomicSurface (b : E → E → E) : Set (Phase E) :=
  {z | z.q = 0 ∧ z.v = 0 ∧ z.p2 = b 0 0}

/-- Genuine kinematic nodal data lie on the exact S004 nodal
PRIMARY constraint surface. This implication is not reversible. -/
theorem nodal_holonomic_is_primary
    (b : E → E → E) :
    nodalHolonomicSurface b ⊆ nodalPrimarySurface b := by
  intro z hz
  rcases hz with ⟨hq, hv, hp⟩
  change z.q = 0 ∧ z.p2 = b z.q z.v
  constructor
  · exact hq
  · simpa only [hq, hv] using hp

/-- At a classical stationary node the canonical Ostrogradsky
term ⟪p₁,v⟫ really vanishes, regardless of p₁. The remaining
value is exactly G(0,0,b(0,0)); it need not be nonnegative. -/
theorem nodal_holonomic_energy_exact
    (b : E → E → E) (G : E → E → E → ℝ)
    (z : Phase E) (hz : z ∈ nodalHolonomicSurface b) :
    ostroForm G z = G 0 0 (b 0 0) := by
  rcases hz with ⟨hq, hv, hp⟩
  simp [ostroForm, hq, hv, hp]

/-- The canonical Hamiltonian image of the whole holonomic nodal
set is exactly a singleton, not just bounded. This is a
kinematic energy-fibre result; it does not show the set is an
invariant subset of the nonlinear S004 Hamiltonian flow. -/
theorem nodal_holonomic_energy_image
    (b : E → E → E) (G : E → E → E → ℝ) :
    ostroForm G '' nodalHolonomicSurface b =
      {G 0 0 (b 0 0)} := by
  ext e
  constructor
  · rintro ⟨z, hz, rfl⟩
    simp [nodal_holonomic_energy_exact b G z hz]
  · intro he
    rw [Set.mem_singleton_iff] at he
    subst e
    let z : Phase E := ⟨0, 0, 0, b 0 0⟩
    refine ⟨z, ?_, ?_⟩
    · simp [z, nodalHolonomicSurface]
    · exact nodal_holonomic_energy_exact b G z
        (by simp [z, nodalHolonomicSurface])

/-- A CLASSICAL trajectory staying entirely in the original nodal
primary surface, and satisfying its honest kinematics q̇=v, must
also stay in the holonomic zero-velocity subset. Neither the
primary constraint nor a zero Hessian alone implies this. -/
theorem permanently_nodal_path_is_holonomic
    (b : E → E → E) (z : ℝ → Phase E)
    (hPrimary : ∀ t, z t ∈ nodalPrimarySurface b)
    (hKinematic : ∀ t,
      HasDerivAt (fun s => (z s).q) (z t).v t) :
    ∀ t, z t ∈ nodalHolonomicSurface b := by
  have hq : (fun s => (z s).q) = (fun _ => (0 : E)) := by
    funext t
    exact (hPrimary t).1
  intro t
  have hderiv := hKinematic t
  rw [hq] at hderiv
  have hv : (z t).v = 0 := by
    simpa using hderiv.deriv.symm
  refine ⟨(hPrimary t).1, hv, ?_⟩
  have hp := (hPrimary t).2
  simpa only [(hPrimary t).1, hv] using hp

/-- On any classical path that stays in the corrected original
nodal primary surface, the Ostrogradsky energy is a single
fixed value. This is not a statement about whether such a
nontrivial or invariant dynamical path exists. -/
theorem permanently_nodal_path_energy_constant
    (b : E → E → E) (G : E → E → E → ℝ)
    (z : ℝ → Phase E)
    (hPrimary : ∀ t, z t ∈ nodalPrimarySurface b)
    (hKinematic : ∀ t,
      HasDerivAt (fun s => (z s).q) (z t).v t) :
    ∀ t, ostroForm G (z t) = G 0 0 (b 0 0) := by
  intro t
  exact nodal_holonomic_energy_exact b G (z t)
    (permanently_nodal_path_is_holonomic b z hPrimary hKinematic t)

end ReleaseProofs.S004NodalHolonomic
