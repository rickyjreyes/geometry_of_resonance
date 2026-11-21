import Mathlib
import ReleaseProofs.S004NodalDiracClosure
import ReleaseProofs.S004ChargeFibre

/-!
# S004 Phase XXX: the stratified constraint algorithm (unified formalism)

The acceleration Hessian of the corrected S004 density changes rank on the nodal set, so Dirac's
algorithm (which presupposes constant rank on an open region) cannot be applied literally across
strata. The standard rank-independent framework is the Skinner–Rusk unified formalism on
`W = {(q, v, a, p₁, p₂)}` with energy `E = ⟪p₁,v⟫ + ⟪p₂,a⟫ − L`:

* primary constraint (from `∂E/∂a = 0`): `p₂ = ∂L/∂a = F(ρ)·(A b − 2κ a)`;
* dynamics: `q̇ = v`, `v̇ = a`, `ṗ₂ = ∂L/∂v − p₁ = 2v − p₁`, `ṗ₁ = ∂L/∂q`;
* tangency of the primary constraint: `p₁ = 2v − d/dt[F(Ab − 2κa)]`, i.e.
  `tangencyResidual = p₁ − 2v + Ḟ(Ab − 2κa) + F(Aḃ − 2κȧ) = 0`.

The jerk `ȧ` enters ONLY through `−2κF·ȧ` (`= ∂²L/∂a² · ȧ`; independent symbolic check in
`derivations/s004_closure_verify.py`). Consequently (this module):

* `tangency_regular_unique_jerk`: where `F ≠ 0` the tangency condition is solved by a unique jerk
  for EVERY `(v, a, b, ḃ, p₁)`: it is a multiplier condition and creates no secondary constraint
  (regular stratum: no constraints, `4` real configuration degrees of freedom per point).
* `tangency_nodal_iff`: where `F = 0` and `Ḟ = 0` the jerk drops out and the tangency condition
  is exactly the secondary constraint `p₁ = 2v`.
* `tangency_jerk_independent_iff`: the condition is jerk-independent (a genuine constraint) iff
  `κF = 0`: the rank change of the Hessian is exactly the change of constraint type.
* `nodal_path_tangency_iff`: along every path differentiable at a nodal instant, `F = Ḟ = 0`
  there (nodal Fermat lemma), so the tangency condition reduces to `p₁ = 2v` at that instant.
* `shift_invariant_bounded_sector_is_static`: on the regular stratum no constraint is generated,
  so any sector defined without reference to `p₁` is invariant under `p₁ ↦ p₁ + t v`; if the
  canonical energy is bounded below on such a sector, every classical trajectory staying in it is
  static (`q` constant).

Conventions: Lean `Phase.p1` is conjugate to `q`, `Phase.p2` to `v`; physical orientation
`L_phys = −L_orig`, so `∂L/∂v = 2v` (in `L_orig` orientation the constraint reads `p₁ = −2v`;
in the `p₀, p₁` naming of `docs/foundational/GHOST_AND_CONSTRAINT_ANALYSIS.md` it is `p₀ = 2v`).
-/

namespace ReleaseProofs.S004StratifiedConstraint

open ReleaseProofs.R16Class

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Unified-formalism tangency residual of the S004 Legendre constraint at one point, with weight
value `F`, its time derivative `Fdot`, velocity `v`, acceleration `a`, Laplacian `b`, its time
derivative `bdot`, jerk `j` and first momentum `p1`. -/
def tangencyResidual (kappa A F Fdot : ℝ) (v a b bdot j p1 : E) : E :=
  p1 - (2 : ℝ) • v + Fdot • (A • b - (2 * kappa) • a) + F • (A • bdot - (2 * kappa) • j)

/-- **Regular stratum: multiplier, not constraint.** If `κF ≠ 0` the tangency condition has
exactly one solution `j` for every `(v, a, b, ḃ, p₁)`. -/
theorem tangency_regular_unique_jerk {kappa A F Fdot : ℝ} (hk : kappa ≠ 0) (hF : F ≠ 0)
    (v a b bdot p1 : E) :
    ∃! j : E, tangencyResidual kappa A F Fdot v a b bdot j p1 = 0 := by
  have hc : (2 * kappa * F) ≠ 0 := mul_ne_zero (mul_ne_zero two_ne_zero hk) hF
  set r : E := p1 - (2 : ℝ) • v + Fdot • (A • b - (2 * kappa) • a) + F • (A • bdot) with hr
  have key : ∀ j : E, tangencyResidual kappa A F Fdot v a b bdot j p1 =
      r - (2 * kappa * F) • j := by
    intro j
    simp only [tangencyResidual, hr, smul_sub, smul_smul]
    rw [show F * (2 * kappa) = 2 * kappa * F by ring]
    abel
  refine ⟨(2 * kappa * F)⁻¹ • r, ?_, ?_⟩
  · beta_reduce
    rw [key, smul_smul, mul_inv_cancel₀ hc, one_smul, sub_self]
  · intro j hj
    rw [key, sub_eq_zero] at hj
    rw [hj, smul_smul, inv_mul_cancel₀ hc, one_smul]

/-- **Nodal stratum: the tangency condition is the secondary constraint `p₁ = 2v`**, for every
acceleration, Laplacian, its rate and jerk. -/
theorem tangency_nodal_iff (kappa A : ℝ) (v a b bdot j p1 : E) :
    tangencyResidual kappa A 0 0 v a b bdot j p1 = 0 ↔ p1 = (2 : ℝ) • v := by
  simp only [tangencyResidual, zero_smul, add_zero]
  exact sub_eq_zero

/-- **Rank change = change of constraint type.** For a nonzero direction, the tangency residual
is independent of the jerk iff `κF = 0`. -/
theorem tangency_jerk_independent_iff (kappa A F Fdot : ℝ) (v a b bdot p1 : E) (e : E)
    (he : e ≠ 0) :
    (∀ j j' : E, tangencyResidual kappa A F Fdot v a b bdot j p1 =
        tangencyResidual kappa A F Fdot v a b bdot j' p1) ↔ kappa * F = 0 := by
  have hdiff : ∀ j j' : E, tangencyResidual kappa A F Fdot v a b bdot j p1 -
      tangencyResidual kappa A F Fdot v a b bdot j' p1 = (2 * kappa * F) • (j' - j) := by
    intro j j'
    simp only [tangencyResidual, smul_sub, smul_smul]
    rw [show F * (2 * kappa) = 2 * kappa * F by ring]
    abel
  constructor
  · intro h
    have h0 := hdiff 0 e
    rw [h 0 e, sub_self, sub_zero] at h0
    rcases smul_eq_zero.mp h0.symm with hc | hc
    · have : (2 : ℝ) * (kappa * F) = 0 := by linarith [hc]
      exact (mul_eq_zero.mp this).resolve_left two_ne_zero
    · exact absurd hc he
  · intro hkF j j'
    have hc : 2 * kappa * F = 0 := by rw [mul_assoc, hkF, mul_zero]
    have := hdiff j j'
    rw [hc, zero_smul, sub_eq_zero] at this
    exact this

/-- **Along any path at a nodal instant** (`q t₀ = 0`, `q` differentiable there) the tangency
condition, evaluated with the actual weight `F(‖q t₀‖²) = 0` and its actual derivative `0`, is
equivalent to `p₁ = 2v`. -/
theorem nodal_path_tangency_iff (kappa A eps alpha t₀ : ℝ) (q : ℝ → E) (heps : eps ≠ 0)
    (hq0 : q t₀ = 0) (hq : DifferentiableAt ℝ q t₀) (v a b bdot j p1 : E) :
    tangencyResidual kappa A (R16.correctedWeight (‖q t₀‖ ^ 2) eps alpha)
        (deriv (fun t => R16.correctedWeight (‖q t‖ ^ 2) eps alpha) t₀) v a b bdot j p1 = 0 ↔
      p1 = (2 : ℝ) • v := by
  rw [S004NodalDiracClosure.nodal_weight_zero q t₀ eps alpha hq0,
    (S004NodalDiracClosure.nodal_weight_hasDerivAt_zero q t₀ eps alpha heps hq0 hq).deriv]
  exact tangency_nodal_iff kappa A v a b bdot j p1

/-- **Regular-sector invariant-set classification.** Let `S` be invariant under the `p₁`-shifts
`p₁ ↦ p₁ + t v` (true for every sector defined without reference to `p₁`; the regular stratum
generates no constraint involving `p₁`), and let an Ostrogradsky-form energy be bounded below on
`S`. Then every classical path `z` staying in `S` with the genuine kinematics `q̇ = v` is static. -/
theorem shift_invariant_bounded_sector_is_static (G : E → E → E → ℝ) (S : Set (Phase E))
    (hS : ∀ w ∈ S, ∀ t : ℝ, w.shift t ∈ S) (hb : BddBelow (ostroForm G '' S))
    (z : ℝ → Phase E) (hzS : ∀ t, z t ∈ S)
    (hkin : ∀ t, HasDerivAt (fun s => (z s).q) (z t).v t) :
    ∀ t, (z t).q = (z 0).q := by
  have hv : ∀ t, (z t).v = 0 := fun t => bddBelow_forces_v_zero G S hS hb (z t) (hzS t)
  have hconst : ∀ t, HasDerivAt (fun s => (z s).q) 0 t := fun t => by
    have := hkin t
    rwa [hv t] at this
  intro t
  exact is_const_of_deriv_eq_zero
    (fun s => (hconst s).differentiableAt) (fun s => (hconst s).deriv) t 0

/-- **Charged states cannot cross the node regularly.** At a nodal instant of any path whose
position, acceleration and Laplacian are differentiable there, the actual U(1) Noether charge
`Q = ⟪p₁, Jx⟫ + ⟪p₂, Jv⟫` (`S004ChargeFibre.phaseCharge`, two real components) vanishes, because
`x = 0` and `p₂ = 0` there. Since `Q` is conserved along classical solutions (Noether), a solution
with `Q ≠ 0` never passes through the node with this regularity. -/
theorem nodal_instant_charge_vanishes (kappa A eps alpha t₀ : ℝ)
    (q v a b : ℝ → EuclideanSpace ℝ (Fin 2)) (j c : EuclideanSpace ℝ (Fin 2))
    (heps : eps ≠ 0) (hq0 : q t₀ = 0) (hq : DifferentiableAt ℝ q t₀)
    (ha : HasDerivAt a j t₀) (hb : HasDerivAt b c t₀) :
    S004ChargeFibre.phaseCharge ((q t₀) 0) ((q t₀) 1) ((v t₀) 0) ((v t₀) 1)
      ((S004NodalDiracClosure.firstMomentumPath kappa A eps alpha q v a b t₀) 0)
      ((S004NodalDiracClosure.firstMomentumPath kappa A eps alpha q v a b t₀) 1)
      ((S004NodalDiracClosure.accelMomentumPath kappa A eps alpha q a b t₀) 0)
      ((S004NodalDiracClosure.accelMomentumPath kappa A eps alpha q a b t₀) 1) = 0 := by
  obtain ⟨h2, -⟩ := S004NodalDiracClosure.nodal_instant_momenta kappa A eps alpha t₀ q v a b j c
    heps hq0 hq ha hb
  rw [h2, hq0]
  simp [S004ChargeFibre.phaseCharge]

end ReleaseProofs.S004StratifiedConstraint
