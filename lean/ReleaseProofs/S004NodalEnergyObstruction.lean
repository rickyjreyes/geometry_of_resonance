import Mathlib
import ReleaseProofs.S004NodalPrimary
import ReleaseProofs.R16_OstrogradskyClass

/-!
# Corrected S004 nodal primary surface: energy obstruction

At the corrected S004 field node q=0 (both real components), the
acceleration Hessian vanishes and the exact primary momentum condition
is p2=b(q,v). This module characterizes the node with its genuine
Legendre image and applies the canonical Ostrogradsky momentum direction
WHILE ENFORCING BOTH q=0 AND THE NODAL MOMENTUM CONSTRAINT.

For any nonzero canonical velocity v, the primary nodal constraint set
still admits unbounded energy above and below under free p1-shifts.
Consequently primary nodal degeneracy alone does not eliminate the ghost.

This is a canonical primary-constraint obstruction, not a claim that
arbitrary nonzero-velocity nodal points satisfy the secondary Dirac
constraints or evolve consistently under the full nonlinear field PDE.
-/

namespace ReleaseProofs.S004NodalEnergyObstruction

open ReleaseProofs.R16Class
open ReleaseProofs.S004NodalPrimary

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Both real field components vanish at the S004 node and the exact
primary acceleration momentum satisfies the nodal Legendre equation.
No restriction on p1 follows from this primary condition. -/
def nodalPrimarySurface (b : E → E → E) : Set (Phase E) :=
  {z | z.q = 0 ∧ z.p2 = b z.q z.v}

/-- Membership in the nodal primary surface is equivalent to the
actual corrected S004 nodal acceleration Legendre equation, together
with q=0. The acceleration itself is unconstrained at the node. -/
theorem nodal_primary_surface_exact
    (eps alpha kappa : ℝ) (b : E → E → E) (z : Phase E) :
    z ∈ nodalPrimarySurface b ↔
      z.q = 0 ∧ ∃ a : E,
        accelMomentum
          ((-2 * ReleaseProofs.R16.correctedWeight 0 eps alpha * kappa) •
            LinearMap.id : E →ₗ[ℝ] E)
          b z.q z.v a = z.p2 := by
  change (z.q = 0 ∧ z.p2 = b z.q z.v) ↔ _
  constructor
  · rintro ⟨hq, hp⟩
    exact ⟨hq, (node_primary_momentum_iff eps alpha kappa b
      z.q z.v z.p2).mpr hp⟩
  · rintro ⟨hq, hp⟩
    exact ⟨hq, (node_primary_momentum_iff eps alpha kappa b
      z.q z.v z.p2).mp hp⟩

/-- The exact nodal PRIMARY constraints do not involve p1 and
therefore cannot obstruct its canonical translation along v. -/
theorem nodal_primary_surface_shift
    (b : E → E → E) :
    ∀ z ∈ nodalPrimarySurface b, ∀ t : ℝ,
      z.shift t ∈ nodalPrimarySurface b := by
  intro z hz t
  simpa [nodalPrimarySurface, Phase.shift] using hz

/-- Even when q=0 and p2=b(q,v) are ENFORCED, the canonical
Ostrogradsky energy on the S004 nodal PRIMARY surface cannot be
bounded below if a nonzero velocity is admitted. This does not
assert that such velocities survive a secondary Dirac algorithm. -/
theorem nodal_primary_surface_not_bddBelow
    (b : E → E → E) (G : E → E → E → ℝ)
    (v : E) (hv : v ≠ 0) :
    ¬ BddBelow (ostroForm G '' nodalPrimarySurface b) := by
  intro hb
  let z : Phase E := ⟨0, v, 0, b 0 v⟩
  have hz : z ∈ nodalPrimarySurface b := by
    simp [z, nodalPrimarySurface]
  have hvz : z.v ≠ 0 := by simpa [z] using hv
  have hzero := bddBelow_forces_v_zero G (nodalPrimarySurface b)
    (nodal_primary_surface_shift b) hb z hz
  exact hvz hzero

/-- The same exact nodal primary surface also has NO upper energy
bound if it contains a canonical phase point with nonzero velocity. -/
theorem nodal_primary_surface_not_bddAbove
    (b : E → E → E) (G : E → E → E → ℝ)
    (v : E) (hv : v ≠ 0) :
    ¬ BddAbove (ostroForm G '' nodalPrimarySurface b) := by
  intro hb
  let z : Phase E := ⟨0, v, 0, b 0 v⟩
  have hz : z ∈ nodalPrimarySurface b := by
    simp [z, nodalPrimarySurface]
  have hvz : z.v ≠ 0 := by simpa [z] using hv
  have hzero := bddAbove_forces_v_zero G (nodalPrimarySurface b)
    (nodal_primary_surface_shift b) hb z hz
  exact hvz hzero

end ReleaseProofs.S004NodalEnergyObstruction
