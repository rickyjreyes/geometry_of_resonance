import Mathlib
import ReleaseProofs.S004NodalHolonomic
import ReleaseProofs.S004NodalEnergyObstruction

/-!
# Phase XXIII: exact energy classification for nodal p₁-translation invariant sectors

This completes a *restricted* but nontrivial S004 canonical classification:
a subset of the actual nodal primary Legendre surface whose constraints
are preserved by all translations of p₁ along v has bounded Ostrogradsky
energy above or below precisely when every point has zero velocity.
The zero-velocity subset lies on the genuine nodal holonomic fibre and
has the single energy value G(0,0,b(0,0)).

The theorem covers arbitrary additional constraints involving q, v,
and p₂, since those cannot obstruct p₁ translations. Hence no such
constraint set containing v≠0 removes the canonical unboundedness.
This does NOT derive the secondary Dirac algorithm of the nonlinear
S004 action. Its eventual secondary constraints may involve p₁, and
their consistency/physical invariance must be analyzed separately.
-/

namespace ReleaseProofs.S004NodalShiftClassification

open ReleaseProofs.R16Class
open ReleaseProofs.S004NodalEnergyObstruction
open ReleaseProofs.S004NodalHolonomic

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Exact classification, for any constraint sector contained in the
corrected S004 nodal primary surface and invariant under all p₁ shifts:
bounded-below energy is EQUIVALENT to zero velocity throughout the sector.

The forward direction uses the proven Ostrogradsky unboundedness
witness, while the converse uses the actual nodal Legendre equation,
not an assumed positivity of a different Hamiltonian. -/
theorem nodal_shift_sector_bddBelow_iff_holonomic
    (b : E → E → E) (G : E → E → E → ℝ)
    (C : Set (Phase E))
    (hPrimary : C ⊆ nodalPrimarySurface b)
    (hShift : ∀ z ∈ C, ∀ t : ℝ, z.shift t ∈ C) :
    BddBelow (ostroForm G '' C) ↔
      C ⊆ nodalHolonomicSurface b := by
  constructor
  · intro hb z hz
    have hv : z.v = 0 :=
      bddBelow_forces_v_zero G C hShift hb z hz
    obtain ⟨hq, hp⟩ := hPrimary hz
    change z.q = 0 ∧ z.v = 0 ∧ z.p2 = b 0 0
    refine ⟨hq, hv, ?_⟩
    simpa only [hq, hv] using hp
  · intro hHol
    refine ⟨G 0 0 (b 0 0), ?_⟩
    rintro e ⟨z, hz, rfl⟩
    rw [nodal_holonomic_energy_exact b G z (hHol hz)]

/-- The upper-bounded condition is *equally* restrictive.
Both signs of the unbounded affine p₁ direction are controlled. -/
theorem nodal_shift_sector_bddAbove_iff_holonomic
    (b : E → E → E) (G : E → E → E → ℝ)
    (C : Set (Phase E))
    (hPrimary : C ⊆ nodalPrimarySurface b)
    (hShift : ∀ z ∈ C, ∀ t : ℝ, z.shift t ∈ C) :
    BddAbove (ostroForm G '' C) ↔
      C ⊆ nodalHolonomicSurface b := by
  constructor
  · intro hb z hz
    have hv : z.v = 0 :=
      bddAbove_forces_v_zero G C hShift hb z hz
    obtain ⟨hq, hp⟩ := hPrimary hz
    change z.q = 0 ∧ z.v = 0 ∧ z.p2 = b 0 0
    refine ⟨hq, hv, ?_⟩
    simpa only [hq, hv] using hp
  · intro hHol
    refine ⟨G 0 0 (b 0 0), ?_⟩
    rintro e ⟨z, hz, rfl⟩
    rw [nodal_holonomic_energy_exact b G z (hHol hz)]

/-- Every translation-invariant nodal sector has bounded energy in
both directions iff it lies entirely in the zero-velocity fibre. -/
theorem nodal_shift_sector_two_sided_iff_holonomic
    (b : E → E → E) (G : E → E → E → ℝ)
    (C : Set (Phase E))
    (hPrimary : C ⊆ nodalPrimarySurface b)
    (hShift : ∀ z ∈ C, ∀ t : ℝ, z.shift t ∈ C) :
    (BddBelow (ostroForm G '' C) ∧
      BddAbove (ostroForm G '' C)) ↔
      C ⊆ nodalHolonomicSurface b := by
  rw [nodal_shift_sector_bddBelow_iff_holonomic b G C hPrimary hShift,
      nodal_shift_sector_bddAbove_iff_holonomic b G C hPrimary hShift]
  simp

/-- A nodal primary constraint restricted by *any* additional
predicate depending only on (q,v,p₂). These are exactly independent
of p₁; the full nonlinear constraint algorithm is not asserted. -/
def nodalLowerMomentumSector
    (b : E → E → E) (P : E → E → E → Prop) : Set (Phase E) :=
  {z | z ∈ nodalPrimarySurface b ∧ P z.q z.v z.p2}

/-- Explicit S004 no-go for arbitrary p₁-independent nodal
constraint predicates: boundedness below is possible exactly
when they exclude every nonzero-velocity nodal state. -/
theorem lower_momentum_sector_bddBelow_iff_holonomic
    (b : E → E → E) (G : E → E → E → ℝ)
    (P : E → E → E → Prop) :
    BddBelow (ostroForm G '' nodalLowerMomentumSector b P) ↔
      nodalLowerMomentumSector b P ⊆ nodalHolonomicSurface b := by
  apply nodal_shift_sector_bddBelow_iff_holonomic b G
  · intro z hz
    exact hz.1
  · intro z hz t
    change z.shift t ∈ nodalPrimarySurface b ∧
      P (z.shift t).q (z.shift t).v (z.shift t).p2
    constructor
    · exact nodal_primary_surface_shift b z hz.1 t
    · simpa only [Phase.shift] using hz.2

end ReleaseProofs.S004NodalShiftClassification
