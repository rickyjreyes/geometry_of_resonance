import Mathlib
import ReleaseProofs.S004ChargeFibre
import ReleaseProofs.S004Regularity

/-!
# Corrected regular S004 canonical no-go at EVERY prescribed Noether charge

Earlier formalization identified radial Ostrogradsky momentum shifts that
preserve actual U(1) canonical charge and change energy without bound.
Here the prescribed value of the canonical charge is NO LONGER the
charge of an arbitrarily chosen initial phase point: every real
charge Q has a nonempty canonical momentum fibre at every nonzero
field, and on every such fibre with nonzero radial velocity the
canonical Hamiltonian is unbounded both below and above.

Thus fixing the literal Noether charge alone cannot produce an
energetically healthy regular S004/A2 theory. Any healthy *candidate*
sector must impose additional constraints restricting the radial P0
direction or the radial velocity. This does not rule out all possible
independently derived nonlinear constrained sectors or resolve their
invariance under the full field evolution.
-/

namespace ReleaseProofs.S004ChargeNoGo

open ReleaseProofs.S004ChargeFibre

/-- For every nonzero real complex field, every velocity, every
higher canonical momentum and *every prescribed real Noether charge*,
the correct A2/S004 charge condition has a canonical P0 solution.
The construction does not identify Noether charge with field mass. -/
theorem arbitrary_noether_charge_at_fixed_full_jet
    (x1 x2 v1 v2 p11 p12 Q : ℝ)
    (hx : 0 < x1 ^ 2 + x2 ^ 2) :
    ∃ p01 p02 : ℝ,
      phaseCharge x1 x2 v1 v2 p01 p02 p11 p12 = Q := by
  let rho : ℝ := x1 ^ 2 + x2 ^ 2
  let target : ℝ := Q + p11 * v2 - p12 * v1
  have hrho : rho ≠ 0 := ne_of_gt (by simpa [rho] using hx)
  refine ⟨-(target * x2) / rho, (target * x1) / rho, ?_⟩
  unfold phaseCharge
  dsimp [target, rho]
  field_simp [show x1 ^ 2 + x2 ^ 2 ≠ 0 from hrho]
  ring

/-- For ANY specified Noether charge and regular canonical data with
nonzero radial velocity, the fixed-charge energy set has no lower bound.
The remainder rest may contain all P0-independent spatial, potential,
P1, Laplacian and regulator terms: its exact value is irrelevant. -/
theorem no_lower_bound_at_any_fixed_charge
    (sigma rest x1 x2 v1 v2 p11 p12 Q : ℝ)
    (hx : 0 < x1 ^ 2 + x2 ^ 2)
    (hrad : x1 * v1 + x2 * v2 ≠ 0) :
    ¬ BddBelow
      {e : ℝ | ∃ p01 p02 : ℝ,
        phaseCharge x1 x2 v1 v2 p01 p02 p11 p12 = Q ∧
        canonicalJetEnergy sigma rest v1 v2 p01 p02 = e} := by
  intro hb
  obtain ⟨bound, hbound⟩ := hb
  obtain ⟨p01,p02,hq⟩ :=
    arbitrary_noether_charge_at_fixed_full_jet
      x1 x2 v1 v2 p11 p12 Q hx
  obtain ⟨t, hcharge, henergy⟩ :=
    phaseCharge_fibre_unbounded_below sigma rest x1 x2
      v1 v2 p01 p02 p11 p12 bound hrad
  have hmem : canonicalJetEnergy sigma rest v1 v2
        (p01 + t * x1) (p02 + t * x2) ∈
      {e : ℝ | ∃ a b : ℝ,
        phaseCharge x1 x2 v1 v2 a b p11 p12 = Q ∧
        canonicalJetEnergy sigma rest v1 v2 a b = e} := by
    exact ⟨p01 + t * x1, p02 + t * x2, hcharge.trans hq, rfl⟩
  have hlow := hbound hmem
  linarith

/-- The same arbitrary-charge regular S004 canonical fibres are never
bounded above. In particular no fixed literal phase charge selects a
positive-energy branch of the unrestricted canonical theory. -/
theorem no_upper_bound_at_any_fixed_charge
    (sigma rest x1 x2 v1 v2 p11 p12 Q : ℝ)
    (hx : 0 < x1 ^ 2 + x2 ^ 2)
    (hrad : x1 * v1 + x2 * v2 ≠ 0) :
    ¬ BddAbove
      {e : ℝ | ∃ p01 p02 : ℝ,
        phaseCharge x1 x2 v1 v2 p01 p02 p11 p12 = Q ∧
        canonicalJetEnergy sigma rest v1 v2 p01 p02 = e} := by
  intro hb
  obtain ⟨bound, hbound⟩ := hb
  obtain ⟨p01,p02,hq⟩ :=
    arbitrary_noether_charge_at_fixed_full_jet
      x1 x2 v1 v2 p11 p12 Q hx
  obtain ⟨t, hcharge, henergy⟩ :=
    phaseCharge_fibre_unbounded_above sigma rest x1 x2
      v1 v2 p01 p02 p11 p12 bound hrad
  have hmem : canonicalJetEnergy sigma rest v1 v2
        (p01 + t * x1) (p02 + t * x2) ∈
      {e : ℝ | ∃ a b : ℝ,
        phaseCharge x1 x2 v1 v2 a b p11 p12 = Q ∧
        canonicalJetEnergy sigma rest v1 v2 a b = e} := by
    exact ⟨p01 + t * x1, p02 + t * x2, hcharge.trans hq, rfl⟩
  have hhigh := hbound hmem
  linarith

/-- A single exact theorem resolves the regular unrestricted
fixed-charge energy question, in BOTH directions, for every Q.
This is a canonical-jet no-go; its passage to a full nonlinear PDE
sector requires genuine constrained phase-space analysis. -/
theorem unrestricted_regular_charge_fibre_no_go
    (sigma rest x1 x2 v1 v2 p11 p12 Q : ℝ)
    (hx : 0 < x1 ^ 2 + x2 ^ 2)
    (hrad : x1 * v1 + x2 * v2 ≠ 0) :
    ¬ BddBelow
      {e : ℝ | ∃ p01 p02 : ℝ,
        phaseCharge x1 x2 v1 v2 p01 p02 p11 p12 = Q ∧
        canonicalJetEnergy sigma rest v1 v2 p01 p02 = e} ∧
    ¬ BddAbove
      {e : ℝ | ∃ p01 p02 : ℝ,
        phaseCharge x1 x2 v1 v2 p01 p02 p11 p12 = Q ∧
        canonicalJetEnergy sigma rest v1 v2 p01 p02 = e} := by
  exact ⟨no_lower_bound_at_any_fixed_charge sigma rest x1 x2 v1 v2
    p11 p12 Q hx hrad,
    no_upper_bound_at_any_fixed_charge sigma rest x1 x2 v1 v2
    p11 p12 Q hx hrad⟩

/-- **Combined corrected S004 regular-patch fixed-charge no-go.**
At an actual nonzero two-component field, nonzero regulator epsilon and
positive higher-derivative kappa, the corrected S004 acceleration
Hessian has strictly positive determinant. At that same field and any
prescribed literal canonical U(1) charge Q, if radial velocity is
nonzero, the unrestricted Ostrogradsky canonical energy is
simultaneously unbounded above and below.

This theorem bridges the *actual corrected nonlinear S004 Hessian*
to the exact fixed-Noether-charge obstruction, rather than treating
regularity and energy pathology as unrelated surrogate claims.
It remains a finite canonical-jet theorem, not a claim about any
separately constrained nonlinear PDE sector. -/
theorem regular_patch_at_every_charge_no_go
    (eps alpha kappa sigma rest x1 x2 v1 v2 p11 p12 Q : ℝ)
    (heps : eps ≠ 0) (hk : 0 < kappa)
    (hx : 0 < x1 ^ 2 + x2 ^ 2)
    (hrad : x1 * v1 + x2 * v2 ≠ 0) :
    0 < ReleaseProofs.R16Energy.s004AccelDet
      (ReleaseProofs.R16.correctedWeight
        (x1 ^ 2 + x2 ^ 2) eps alpha) kappa ∧
    (¬ BddBelow
      {e : ℝ | ∃ p01 p02 : ℝ,
        phaseCharge x1 x2 v1 v2 p01 p02 p11 p12 = Q ∧
        canonicalJetEnergy sigma rest v1 v2 p01 p02 = e}) ∧
    (¬ BddAbove
      {e : ℝ | ∃ p01 p02 : ℝ,
        phaseCharge x1 x2 v1 v2 p01 p02 p11 p12 = Q ∧
        canonicalJetEnergy sigma rest v1 v2 p01 p02 = e}) := by
  have hweight := ReleaseProofs.R16.corrected_weight_pos
    (x1 ^ 2 + x2 ^ 2) eps alpha hx heps
  have hhessian :=
    ReleaseProofs.R16Energy.s004_acceleration_det_strictly_positive
      (ReleaseProofs.R16.correctedWeight
        (x1 ^ 2 + x2 ^ 2) eps alpha) kappa hweight hk
  exact ⟨hhessian,
    (unrestricted_regular_charge_fibre_no_go
      sigma rest x1 x2 v1 v2 p11 p12 Q hx hrad).1,
    (unrestricted_regular_charge_fibre_no_go
      sigma rest x1 x2 v1 v2 p11 p12 Q hx hrad).2⟩

end ReleaseProofs.S004ChargeNoGo
