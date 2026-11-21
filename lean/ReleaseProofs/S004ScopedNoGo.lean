import Mathlib
import ReleaseProofs.S004RegularConstraints
import ReleaseProofs.S004ChargeNoGo
import ReleaseProofs.S004CharacteristicCone
import ReleaseProofs.S004NodalDiracClosure
import ReleaseProofs.S004CoefficientFamily
import ReleaseProofs.S004LocalizedHadamard

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

Phase XXXIII adds `s004_scoped_no_go_localization`: the kernel-checked ingredients (simple growing
root, exact localized beam with closed-form `L²` growth, exponent bookkeeping) of the
self-contained proof that the linearized variable-coefficient Cauchy problem is Sobolev ill-posed
near every regular point (`docs/s004_closure/VARIABLE_COEFFICIENT_ILLPOSEDNESS.md`).

What remains analytical (NOT kernel-checked): the Sobolev and transport-hierarchy estimates of that
proof (written out in full there; no external theorem is cited), the nonlinear consequences, and
the infinite-dimensional Dirac count `H2`. See `docs/s004_closure/`.
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

/-- **Phase XXXIII: localization ingredient at the same regular point.** With the source
hypotheses only (`ε ≠ 0`, admissibility, one regular point): the weight is positive, the
linearized principal symbol is elliptic, there is a simple growing Hadamard root `w`, the complex
Gaussian-beam phase solves the frozen eikonal equation exactly, the beam's squared `L²` norm has
a closed form and grows by at least `N^{2K Re w}` on the time scale `K log N / N`, and for every
Sobolev pair the construction parameters have positive exponent gaps.

The PDE passage from these ingredients to failure of every Sobolev estimate for the linearized
variable-coefficient Cauchy problem is the paper proof in
`docs/s004_closure/VARIABLE_COEFFICIENT_ILLPOSEDNESS.md`; its final contradiction step is
`S004LocalizedHadamard.localized_estimate_fails`. -/
theorem s004_scoped_no_go_localization
    (eps alpha kappa theta gamma : ℝ) (heps : eps ≠ 0)
    (hadm : Admissible kappa theta gamma) (x1 x2 : ℝ) (hx : 0 < x1 ^ 2 + x2 ^ 2) :
    0 < R16.correctedWeight (x1 ^ 2 + x2 ^ 2) eps alpha ∧
    (∀ (τ : ℝ) (ξ : EuclideanSpace ℝ (Fin 3)), (τ ≠ 0 ∨ ξ ≠ 0) →
      0 < principalSymbol (R16.correctedWeight (x1 ^ 2 + x2 ^ 2) eps alpha)
        kappa theta gamma τ (‖ξ‖ ^ 2)) ∧
    (∃ w : ℂ, w ^ 2 = hadamardRatio kappa theta gamma ∧ 0 < w.re ∧
      laplaceSymbol kappa theta gamma w (-1) = 0 ∧
      laplaceSymbolDeriv kappa theta gamma w (-1) ≠ 0 ∧
      (∀ t y : ℝ, S004LocalizedHadamard.beamPhaseT w t y +
          Complex.I * w * S004LocalizedHadamard.beamPhaseY w t y = 0 ∧
        laplaceSymbol kappa theta gamma (S004LocalizedHadamard.beamPhaseT w t y)
          (S004LocalizedHadamard.beamPhaseY w t y ^ 2) = 0) ∧
      (∀ N : ℝ, 0 < N → ∀ t : ℝ,
        MeasureTheory.Integrable (fun y : ℝ => ‖S004LocalizedHadamard.beam N w t y‖ ^ 2) ∧
        ∫ y : ℝ, ‖S004LocalizedHadamard.beam N w t y‖ ^ 2 =
          Real.sqrt (Real.pi / N) * Real.exp (2 * N * t * w.re + N * t ^ 2 * w.re ^ 2)) ∧
      (∀ N : ℝ, 0 < N → ∀ K : ℝ,
        N ^ (2 * K * w.re) * ∫ y : ℝ, ‖S004LocalizedHadamard.beam N w 0 y‖ ^ 2 ≤
          ∫ y : ℝ, ‖S004LocalizedHadamard.beam N w (K * Real.log N / N) y‖ ^ 2)) ∧
    (∀ s σ c : ℝ, 0 ≤ s → 0 ≤ σ → 0 < c →
      ∃ K : ℝ, ∃ Kp J : ℕ, 0 < K ∧ 3 + s + σ < K * c ∧ 4 + s + σ < Kp ∧ 2 + s + σ < J) := by
  have hF : 0 < R16.correctedWeight (x1 ^ 2 + x2 ^ 2) eps alpha :=
    R16.corrected_weight_pos _ eps alpha hx heps
  exact ⟨hF, fun τ ξ hne => admissible_elliptic_covector hF hadm τ ξ hne,
    S004LocalizedHadamard.localized_hadamard_beam hadm,
    fun _ _ _ hs hσ hc => S004LocalizedHadamard.parameter_choice hs hσ hc⟩

end ReleaseProofs.S004ScopedNoGo
