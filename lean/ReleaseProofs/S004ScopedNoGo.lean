import Mathlib
import ReleaseProofs.S004RegularConstraints
import ReleaseProofs.S004ChargeNoGo
import ReleaseProofs.S004CharacteristicCone
import ReleaseProofs.S004NodalDiracClosure
import ReleaseProofs.S004CoefficientFamily

/-!
# S004 Phase XXIX: single kernel-checked interface for the scoped no-go (issues #16, #53)

One theorem collecting, for the corrected S004 action on a fixed flat background with inertial
foliation and the source admissibility condition `κ > 0, θ > 0, γ² < 4κθ`, `ε ≠ 0`:

1. **Regular canonical obstruction** at every nonzero-field point (existing Phases XVIII–XIX):
   strictly positive acceleration-Hessian determinant, bijective acceleration Legendre map (no
   primary constraint), and canonical energy unbounded above and below along `p₁`-shifts at every
   phase point with nonzero velocity.
2. **Fixed-charge obstruction** (existing): the same two-sided unboundedness on every fibre of
   prescribed U(1) Noether charge when the radial velocity is nonzero.
3. **Spacetime ellipticity** (Phase XXV): the principal symbol of the linearized operator at the
   same point is strictly positive at every real nonzero covector.
4. **Exact Hadamard roots** (Phase XXV): the frozen principal part has, at every spatial
   frequency `n > 0`, an exact root with `Re λ = n c`, `c > 0` fixed.
5. **Nodal closure** (Phase XXVI): the nodal invariance constraints leave exactly the vacuum
   point, and they are all second class (bracket determinant `1`).
6. **Family classification** (Phase XXVII): no ghost-free member of the S004 coefficient family
   with nonnegative curvature energy is admissible.

What remains analytical (NOT kernel-checked): the Lax–Mizohata step from items 3–4 at one point to
ill-posedness of the variable-coefficient/nonlinear Cauchy problem, and the infinite-dimensional
Dirac count `H2`. See `docs/s004_closure/`.
-/

namespace ReleaseProofs.S004ScopedNoGo

open ReleaseProofs.R16Class
open ReleaseProofs.S004CharacteristicCone

/-- **Scoped S004 no-go interface.** -/
theorem s004_scoped_no_go
    (eps alpha kappa theta gamma : ℝ) (heps : eps ≠ 0)
    (hadm : Admissible kappa theta gamma)
    -- a regular point of the field: two real components with nonzero density
    (x1 x2 : ℝ) (hx : 0 < x1 ^ 2 + x2 ^ 2)
    -- canonical data for items 1 and 2
    (b : EuclideanSpace ℝ (Fin 2) → EuclideanSpace ℝ (Fin 2) → EuclideanSpace ℝ (Fin 2))
    (G : EuclideanSpace ℝ (Fin 2) → EuclideanSpace ℝ (Fin 2) → EuclideanSpace ℝ (Fin 2) → ℝ)
    (z : Phase (EuclideanSpace ℝ (Fin 2))) (hv : z.v ≠ 0) (bound : ℝ)
    (sigma rest v1 v2 p11 p12 Q : ℝ) (hrad : x1 * v1 + x2 * v2 ≠ 0) :
    -- (1) regular canonical obstruction
    (0 < R16Energy.s004AccelDet (R16.correctedWeight (x1 ^ 2 + x2 ^ 2) eps alpha) kappa) ∧
    (∀ q v p2 : EuclideanSpace ℝ (Fin 2), ∃! a : EuclideanSpace ℝ (Fin 2),
      accelMomentum
        ((-2 * R16.correctedWeight (x1 ^ 2 + x2 ^ 2) eps alpha * kappa) •
          LinearMap.id) b q v a = p2) ∧
    (∃ t : ℝ, ostroForm G (z.shift t) < bound) ∧
    (∃ t : ℝ, bound < ostroForm G (z.shift t)) ∧
    -- (2) fixed-charge obstruction
    (¬ BddBelow {e : ℝ | ∃ p01 p02 : ℝ,
        S004ChargeFibre.phaseCharge x1 x2 v1 v2 p01 p02 p11 p12 = Q ∧
        S004ChargeFibre.canonicalJetEnergy sigma rest v1 v2 p01 p02 = e}) ∧
    -- (3) spacetime ellipticity at the same regular point
    (∀ (τ : ℝ) (ξ : EuclideanSpace ℝ (Fin 3)), (τ ≠ 0 ∨ ξ ≠ 0) →
      0 < principalSymbol (R16.correctedWeight (x1 ^ 2 + x2 ^ 2) eps alpha)
        kappa theta gamma τ (‖ξ‖ ^ 2)) ∧
    -- (4) exact Hadamard roots with growth linear in the frequency
    (∃ c : ℝ, 0 < c ∧ ∀ n : ℝ, 0 < n → ∃ lam : ℂ,
      frozenSymbolLambda kappa theta gamma lam (n ^ 2) = 0 ∧ lam.re = n * c) ∧
    -- (5) nodal closure: the invariant nodal set is the vacuum point
    ({w : Phase (EuclideanSpace ℝ (Fin 2)) |
        w.q = 0 ∧ w.v = 0 ∧ w.p2 = 0 ∧ w.p1 = (2 : ℝ) • w.v} = {⟨0, 0, 0, 0⟩}) ∧
    (S004NodalDiracClosure.nodalBracketMatrix.det = 1) ∧
    -- (6) no ghost-free nonnegative-curvature member of the family is admissible
    (∀ κ' θ' γ' : ℝ, S004CoefficientFamily.CurvaturePSD κ' θ' γ' →
      R16Energy.s004AccelDet (R16.correctedWeight (x1 ^ 2 + x2 ^ 2) eps alpha) κ' = 0 →
      ¬ Admissible κ' θ' γ') := by
  have hF : 0 < R16.correctedWeight (x1 ^ 2 + x2 ^ 2) eps alpha :=
    R16.corrected_weight_pos _ eps alpha hx heps
  obtain ⟨hdet, -, -⟩ := S004ChargeNoGo.regular_patch_at_every_charge_no_go
    eps alpha kappa sigma rest x1 x2 v1 v2 p11 p12 Q heps hadm.1 hx hrad
  obtain ⟨hleg, hlow, hup⟩ :=
    S004RegularConstraints.regular_patch_no_primary_constraint_and_ghost
      (x1 ^ 2 + x2 ^ 2) eps alpha kappa b G z bound hx heps hadm.1 hv
  refine ⟨hdet, hleg, hlow, hup, ?_, ?_, admissible_hadamard_growth hadm,
    S004NodalDiracClosure.nodal_invariant_constraint_set,
    S004NodalDiracClosure.nodalBracketMatrix_det, ?_⟩
  · exact (S004ChargeNoGo.regular_patch_at_every_charge_no_go
      eps alpha kappa sigma rest x1 x2 v1 v2 p11 p12 Q heps hadm.1 hx hrad).2.1
  · intro τ ξ hne
    exact admissible_elliptic_covector hF hadm τ ξ hne
  · intro κ' θ' γ' hpsd hdeg
    exact (S004CoefficientFamily.ghostfree_psd_members hF hpsd hdeg).2.2.2

end ReleaseProofs.S004ScopedNoGo
