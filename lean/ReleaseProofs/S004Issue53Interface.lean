import Mathlib
import ReleaseProofs.S004ScopedNoGo
import ReleaseProofs.S004StratifiedConstraint
import ReleaseProofs.S004ResonanceCertificate
import ReleaseProofs.S004LocalizedHadamard

/-!
# S004 Phase XXXII: consolidated interface for issue #53

One kernel-checked statement for the corrected S004 action (fixed flat background, inertial
foliation, two real components, physical orientation `L_phys = −L_orig`), organised by the eight
items of the #53 consolidation request. Hypotheses: only the source parameters
(`ε ≠ 0`, admissibility `κ > 0, θ > 0, γ² < 4κθ`) and one regular point `x₁² + x₂² > 0`.
No dynamical or physical conclusion is assumed.

1. Strata: nondegenerate Hessian at the regular point; zero Hessian at the node.
2. Constraints: the tangency condition of the Legendre constraint fixes the jerk uniquely at the
   regular point (no secondary constraint) and is exactly `p₁ = 2v` on the nodal stratum; the
   nodal constraint bracket has determinant `1` (second class).
3. Preservation: the invariant nodal set is the single vacuum point.
4. Boundedness: a `p₁`-shift-invariant set with energy bounded below carries only static
   classical trajectories.
5. Characteristics: spacetime ellipticity at the regular point; exact Hadamard roots.
6. Nonlinear ghost excitation: any nonzero second-order source forces the ghost coordinate.
7. Certified exact second-harmonic ghost resonance in the window `[1/100, 37/50]`, and the
   resulting second-order obstruction to a periodic (Lyapunov) continuation.
8. Excluded recovery inside the coefficient family: no ghost-free nonnegative-curvature member is
   admissible.

Phase XXXIII adds `s004_issue53_interface_with_localization`: items (1)–(8) plus (9) the simple
growing Hadamard root and the exact localized Gaussian beam with closed-form `L²` growth, and (10)
the exponent-gap contradiction step of the localized construction.

What these interfaces do NOT contain (analytical or open, see `docs/s004_closure/`): the Sobolev
and transport-hierarchy estimates of the variable-coefficient construction (proved on paper in
`VARIABLE_COEFFICIENT_ILLPOSEDNESS.md`), the infinite-dimensional Dirac count, nonlinear
instability (norm inflation is not claimed), dynamical-metric completions.
-/

namespace ReleaseProofs.S004Issue53Interface

open ReleaseProofs.R16Class
open ReleaseProofs.S004CharacteristicCone
open ReleaseProofs.S004StratifiedConstraint
open ReleaseProofs.S004ResonanceCertificate

/-- Two real field components. -/
abbrev E2 := EuclideanSpace ℝ (Fin 2)

theorem s004_issue53_interface
    (eps alpha kappa theta gamma : ℝ) (heps : eps ≠ 0) (hadm : Admissible kappa theta gamma)
    (x1 x2 : ℝ) (hx : 0 < x1 ^ 2 + x2 ^ 2) :
    -- (1) strata
    (0 < R16Energy.s004AccelDet (R16.correctedWeight (x1 ^ 2 + x2 ^ 2) eps alpha) kappa) ∧
    (((-2 * R16.correctedWeight 0 eps alpha * kappa) • LinearMap.id : E2 →ₗ[ℝ] E2) = 0) ∧
    -- (2) constraints
    (∀ (Fdot : ℝ) (v a b bdot p1 : E2), ∃! j : E2,
      tangencyResidual kappa (2 * kappa + gamma) (R16.correctedWeight (x1 ^ 2 + x2 ^ 2) eps alpha)
        Fdot v a b bdot j p1 = 0) ∧
    (∀ v a b bdot j p1 : E2,
      tangencyResidual kappa (2 * kappa + gamma) 0 0 v a b bdot j p1 = 0 ↔ p1 = (2 : ℝ) • v) ∧
    (S004NodalDiracClosure.nodalBracketMatrix.det = 1) ∧
    -- (3) preservation
    ({w : Phase E2 | w.q = 0 ∧ w.v = 0 ∧ w.p2 = 0 ∧ w.p1 = (2 : ℝ) • w.v} = {⟨0, 0, 0, 0⟩}) ∧
    -- (4) boundedness
    (∀ (G : E2 → E2 → E2 → ℝ) (S : Set (Phase E2)), (∀ w ∈ S, ∀ t : ℝ, w.shift t ∈ S) →
      BddBelow (ostroForm G '' S) → ∀ z : ℝ → Phase E2, (∀ t, z t ∈ S) →
      (∀ t, HasDerivAt (fun s => (z s).q) (z t).v t) → ∀ t, (z t).q = (z 0).q) ∧
    -- (5) characteristics
    (∀ (τ : ℝ) (ξ : EuclideanSpace ℝ (Fin 3)), (τ ≠ 0 ∨ ξ ≠ 0) →
      0 < principalSymbol (R16.correctedWeight (x1 ^ 2 + x2 ^ 2) eps alpha)
        kappa theta gamma τ (‖ξ‖ ^ 2)) ∧
    (∃ c : ℝ, 0 < c ∧ ∀ n : ℝ, 0 < n → ∃ lam : ℂ,
      frozenSymbolLambda kappa theta gamma lam (n ^ 2) = 0 ∧ lam.re = n * c) ∧
    -- (6) nonlinear ghost excitation
    (∀ K z1 z2 y0 y2 y4 S : ℝ, K ≠ 0 →
      R16PU.puEuler K (K * (z1 + z2)) (K * (z1 * z2)) y0 y2 y4 = S →
      (y4 + z1 * y2) + z2 * (y2 + z1 * y0) = S / K) ∧
    -- (7) certified resonance and second-order periodic obstruction
    (resonanceGap cK 0 cT cM2 qStar = 0 ∧ 1 / 100 ≤ kStar ∧ kStar ≤ 37 / 50) ∧
    (∀ k ω : ℝ, k ^ 2 = qStar → ω ^ 2 = R18.lowerPole cK 0 cT cM2 qStar →
      ¬ ∃ c : ℝ,
        S004HarmonicGhostForcing.linearSymbolRaw (Real.sqrt (1 / 2)) (2 / 9) (-459 / 400)
            (1 / 2) 1 0 (2 * k) (2 * ω) * c +
          S004HarmonicGhostForcing.sigma2Raw (Real.sqrt (1 / 2)) (4 / 27) (-459 / 400) 0
            (1 / 2) 1 0 k ω = 0) ∧
    -- (8) excluded recovery inside the coefficient family
    (∀ κ' θ' γ' : ℝ, S004CoefficientFamily.CurvaturePSD κ' θ' γ' →
      R16Energy.s004AccelDet (R16.correctedWeight (x1 ^ 2 + x2 ^ 2) eps alpha) κ' = 0 →
      ¬ Admissible κ' θ' γ') := by
  have hF : 0 < R16.correctedWeight (x1 ^ 2 + x2 ^ 2) eps alpha :=
    R16.corrected_weight_pos _ eps alpha hx heps
  refine ⟨R16Energy.s004_acceleration_det_strictly_positive _ kappa hF hadm.1,
    S004NodalPrimary.node_accel_hessian_zero eps alpha kappa,
    fun Fdot v a b bdot p1 =>
      tangency_regular_unique_jerk (Fdot := Fdot) hadm.1.ne' hF.ne' v a b bdot p1,
    fun v a b bdot j p1 => tangency_nodal_iff kappa (2 * kappa + gamma) v a b bdot j p1,
    S004NodalDiracClosure.nodalBracketMatrix_det,
    S004NodalDiracClosure.nodal_invariant_constraint_set,
    fun G S hS hb z hzS hkin => shift_invariant_bounded_sector_is_static G S hS hb z hzS hkin,
    fun τ ξ hne => admissible_elliptic_covector hF hadm τ ξ hne,
    admissible_hadamard_growth hadm,
    fun K z1 z2 y0 y2 y4 S hK hS => (S004HarmonicGhostForcing.ghost_mode_forced hK hS).1,
    ⟨certified_resonance, certified_window⟩,
    fun k ω hk hω => certified_no_periodic_second_order k ω hk hω,
    fun κ' θ' γ' hpsd hdeg =>
      (S004CoefficientFamily.ghostfree_psd_members hF hpsd hdeg).2.2.2⟩

/-- **Phase XXXIII: the #53 interface with the localization ingredients.** Items (1)–(8) are
exactly the Phase XXXII interface. Item (9) adds, at the same source hypotheses: a simple growing
Hadamard root, the exact complex Gaussian-beam solution of the frozen eikonal equation and its
closed-form `L²` growth on the time scale `K log N / N`. Item (10) adds the contradiction step of the
localized construction: positive exponent gaps exist for every Sobolev pair, and such gaps make
every estimate `lhs ≤ C (data + source)` fail. The analytic PDE bounds that feed item (10) are proved
on paper in `docs/s004_closure/VARIABLE_COEFFICIENT_ILLPOSEDNESS.md`; they are hypotheses of
`localized_estimate_fails`, not conclusions. -/
theorem s004_issue53_interface_with_localization
    (eps alpha kappa theta gamma : ℝ) (heps : eps ≠ 0) (hadm : Admissible kappa theta gamma)
    (x1 x2 : ℝ) (hx : 0 < x1 ^ 2 + x2 ^ 2) :
    (-- (1) strata
    (0 < R16Energy.s004AccelDet (R16.correctedWeight (x1 ^ 2 + x2 ^ 2) eps alpha) kappa) ∧
    (((-2 * R16.correctedWeight 0 eps alpha * kappa) • LinearMap.id : E2 →ₗ[ℝ] E2) = 0) ∧
    -- (2) constraints
    (∀ (Fdot : ℝ) (v a b bdot p1 : E2), ∃! j : E2,
      tangencyResidual kappa (2 * kappa + gamma) (R16.correctedWeight (x1 ^ 2 + x2 ^ 2) eps alpha)
        Fdot v a b bdot j p1 = 0) ∧
    (∀ v a b bdot j p1 : E2,
      tangencyResidual kappa (2 * kappa + gamma) 0 0 v a b bdot j p1 = 0 ↔ p1 = (2 : ℝ) • v) ∧
    (S004NodalDiracClosure.nodalBracketMatrix.det = 1) ∧
    -- (3) preservation
    ({w : Phase E2 | w.q = 0 ∧ w.v = 0 ∧ w.p2 = 0 ∧ w.p1 = (2 : ℝ) • w.v} = {⟨0, 0, 0, 0⟩}) ∧
    -- (4) boundedness
    (∀ (G : E2 → E2 → E2 → ℝ) (S : Set (Phase E2)), (∀ w ∈ S, ∀ t : ℝ, w.shift t ∈ S) →
      BddBelow (ostroForm G '' S) → ∀ z : ℝ → Phase E2, (∀ t, z t ∈ S) →
      (∀ t, HasDerivAt (fun s => (z s).q) (z t).v t) → ∀ t, (z t).q = (z 0).q) ∧
    -- (5) characteristics
    (∀ (τ : ℝ) (ξ : EuclideanSpace ℝ (Fin 3)), (τ ≠ 0 ∨ ξ ≠ 0) →
      0 < principalSymbol (R16.correctedWeight (x1 ^ 2 + x2 ^ 2) eps alpha)
        kappa theta gamma τ (‖ξ‖ ^ 2)) ∧
    (∃ c : ℝ, 0 < c ∧ ∀ n : ℝ, 0 < n → ∃ lam : ℂ,
      frozenSymbolLambda kappa theta gamma lam (n ^ 2) = 0 ∧ lam.re = n * c) ∧
    -- (6) nonlinear ghost excitation
    (∀ K z1 z2 y0 y2 y4 S : ℝ, K ≠ 0 →
      R16PU.puEuler K (K * (z1 + z2)) (K * (z1 * z2)) y0 y2 y4 = S →
      (y4 + z1 * y2) + z2 * (y2 + z1 * y0) = S / K) ∧
    -- (7) certified resonance and second-order periodic obstruction
    (resonanceGap cK 0 cT cM2 qStar = 0 ∧ 1 / 100 ≤ kStar ∧ kStar ≤ 37 / 50) ∧
    (∀ k ω : ℝ, k ^ 2 = qStar → ω ^ 2 = R18.lowerPole cK 0 cT cM2 qStar →
      ¬ ∃ c : ℝ,
        S004HarmonicGhostForcing.linearSymbolRaw (Real.sqrt (1 / 2)) (2 / 9) (-459 / 400)
            (1 / 2) 1 0 (2 * k) (2 * ω) * c +
          S004HarmonicGhostForcing.sigma2Raw (Real.sqrt (1 / 2)) (4 / 27) (-459 / 400) 0
            (1 / 2) 1 0 k ω = 0) ∧
    -- (8) excluded recovery inside the coefficient family
    (∀ κ' θ' γ' : ℝ, S004CoefficientFamily.CurvaturePSD κ' θ' γ' →
      R16Energy.s004AccelDet (R16.correctedWeight (x1 ^ 2 + x2 ^ 2) eps alpha) κ' = 0 →
      ¬ Admissible κ' θ' γ')) ∧
    -- (9) localization: simple growing root and the exact localized beam
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
    -- (10) the contradiction step: exponent gaps exist and defeat every uniform estimate
    (∀ s σ c : ℝ, 0 ≤ s → 0 ≤ σ → 0 < c →
      ∃ K : ℝ, ∃ Kp J : ℕ, 0 < K ∧ 3 + s + σ < K * c ∧ 4 + s + σ < Kp ∧ 2 + s + σ < J) ∧
    (∀ (lhs data src : ℝ → ℝ) (a b δ₁ δ₂ c C₁ C₂ C₃ : ℝ) (k : ℕ), b < a → 0 < δ₁ → 0 < δ₂ →
      0 < c → 0 < C₁ → 0 ≤ C₂ → 0 ≤ C₃ →
      (∀ᶠ N in Filter.atTop, c * N ^ a ≤ lhs N) →
      (∀ᶠ N in Filter.atTop, 0 ≤ data N ∧ data N ≤ C₁ * N ^ b) →
      (∀ᶠ N in Filter.atTop, 0 ≤ src N ∧
        src N ≤ C₂ * N ^ (a - δ₁) * Real.log N ^ k + C₃ * N ^ (a - δ₂)) →
      ¬ ∃ C : ℝ, ∀ᶠ N in Filter.atTop, lhs N ≤ C * (data N + src N)) :=
  ⟨s004_issue53_interface eps alpha kappa theta gamma heps hadm x1 x2 hx,
    S004LocalizedHadamard.localized_hadamard_beam hadm,
    fun _ _ _ hs hσ hc => S004LocalizedHadamard.parameter_choice hs hσ hc,
    fun _ _ _ _ _ _ _ _ _ _ _ k hab h₁ h₂ hc hC₁ hC₂ hC₃ hl hd hs =>
      S004LocalizedHadamard.localized_estimate_fails k hab h₁ h₂ hc hC₁ hC₂ hC₃ hl hd hs⟩

end ReleaseProofs.S004Issue53Interface
