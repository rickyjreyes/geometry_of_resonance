import Mathlib

/-!
# Issue #16 (Task 1): class-wide Ostrogradsky obstruction and necessary degeneracy

This module works with `n` real fields at once: a finite-dimensional real inner-product
space `E`. It covers the whole class of Lagrangians **quadratic in the acceleration**

  `L(q, v, a) = ½⟪a, W a⟫ + ⟪b(q,v), a⟫ + c(q,v)`

with a symmetric acceleration Hessian `W : E →ₗ[ℝ] E` and **arbitrary** functions `b`, `c`.
For the fixed-flat S004 background, `W = (-2Fκ)•id` on two real fields (see
`R16_EnergyAndConstraints.s004AccelDet`). That instance is `s004_regular` below.

Results:

* `quadratic_lagrangian_expansion`: the exact second-order expansion in `a`. It identifies the
  acceleration momentum `p₂ = W a + b` and the Hessian `W` with no approximation.
* `ostroForm_unbounded_below/above`: any Hamiltonian of Ostrogradsky form
  `H = ⟪p₁, v⟫ + G(q, v, p₂)`, with `G` arbitrary, is unbounded above and below at every
  phase point with `v ≠ 0`.
* `bddBelow_forces_v_zero`: **necessary degeneracy condition.** Take any set of phase points
  that is invariant under shifting `p₁` along `v`. If the Hamiltonian is bounded below on that
  set, then `v = 0` throughout it. So a healthy reduction needs constraints that involve `p₁`.
* `legendre_regular_iff_bijective`, `surjective_iff_det_ne_zero`: the acceleration momentum map
  can be solved for every `p₂` exactly when `W` is invertible, i.e. `det W ≠ 0`.
* `regular_hamiltonian_unbounded`: for invertible `W`, the canonical Ostrogradsky Hamiltonian is
  defined on the whole phase space and is unbounded in both directions.
* `primary_constraint_iff`, `primary_count`: for singular symmetric `W`, a momentum is attainable
  exactly when `⟪p₂ - b, e⟫ = 0` for every `e ∈ ker W`. That gives `finrank (ker W)`
  independent primary constraints, and none of them involves `p₁`.
* `primary_surface_unbounded`: on the primary constraint surface the reduced Hamiltonian still
  has Ostrogradsky form. Without secondary constraints that involve `p₁`, it is still unbounded.

Scope: these are exact finite-dimensional canonical statements about a declared Lagrangian
class. They do not do the full field-theoretic Dirac algorithm for an arbitrary degenerate
action. They also do not construct any healthy completion of S004.
-/

namespace ReleaseProofs.R16Class

open RealInnerProductSpace Module

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Ostrogradsky phase point `(q, v, p₁, p₂)`: the velocity `v = q̇` is promoted to a coordinate,
and `p₁`, `p₂` are its conjugate momenta. -/
structure Phase (E : Type*) where
  q : E
  v : E
  p1 : E
  p2 : E

/-- Shift `p₁` by `t • v`. -/
def Phase.shift (z : Phase E) (t : ℝ) : Phase E :=
  ⟨z.q, z.v, z.p1 + t • z.v, z.p2⟩

/-- A Hamiltonian of Ostrogradsky form: linear in `p₁` with coefficient `v`, plus any `G`. -/
def ostroForm (G : E → E → E → ℝ) (z : Phase E) : ℝ :=
  ⟪z.p1, z.v⟫ + G z.q z.v z.p2

theorem ostroForm_shift (G : E → E → E → ℝ) (z : Phase E) (t : ℝ) :
    ostroForm G (z.shift t) = ostroForm G z + t * ‖z.v‖ ^ 2 := by
  simp only [ostroForm, Phase.shift, inner_add_left, real_inner_smul_left,
    real_inner_self_eq_norm_sq]
  ring

theorem ostroForm_unbounded_below (G : E → E → E → ℝ) (z : Phase E) (hv : z.v ≠ 0)
    (bound : ℝ) : ∃ t : ℝ, ostroForm G (z.shift t) < bound := by
  have hn : 0 < ‖z.v‖ ^ 2 := by positivity
  refine ⟨(bound - ostroForm G z - 1) / ‖z.v‖ ^ 2, ?_⟩
  rw [ostroForm_shift, div_mul_cancel₀ _ hn.ne']
  linarith

theorem ostroForm_unbounded_above (G : E → E → E → ℝ) (z : Phase E) (hv : z.v ≠ 0)
    (bound : ℝ) : ∃ t : ℝ, bound < ostroForm G (z.shift t) := by
  have hn : 0 < ‖z.v‖ ^ 2 := by positivity
  refine ⟨(bound - ostroForm G z + 1) / ‖z.v‖ ^ 2, ?_⟩
  rw [ostroForm_shift, div_mul_cancel₀ _ hn.ne']
  linarith

/-- **Necessary degeneracy condition.** Let a set of phase points be closed under
`p₁ ↦ p₁ + t v`. If an Ostrogradsky-form Hamiltonian is bounded below on it, every point of the
set has `v = 0`. Contrapositive: a bounded-below sector with nonzero velocities needs
constraints that are not invariant under `p₁`-translation, i.e. constraints involving `p₁`. -/
theorem bddBelow_forces_v_zero (G : E → E → E → ℝ) (C : Set (Phase E))
    (hC : ∀ z ∈ C, ∀ t : ℝ, z.shift t ∈ C)
    (hb : BddBelow (ostroForm G '' C)) : ∀ z ∈ C, z.v = 0 := by
  intro z hz
  by_contra hv
  obtain ⟨m, hm⟩ := hb
  obtain ⟨t, ht⟩ := ostroForm_unbounded_below G z hv m
  have := hm ⟨z.shift t, hC z hz t, rfl⟩
  linarith

theorem bddAbove_forces_v_zero (G : E → E → E → ℝ) (C : Set (Phase E))
    (hC : ∀ z ∈ C, ∀ t : ℝ, z.shift t ∈ C)
    (hb : BddAbove (ostroForm G '' C)) : ∀ z ∈ C, z.v = 0 := by
  intro z hz
  by_contra hv
  obtain ⟨m, hm⟩ := hb
  obtain ⟨t, ht⟩ := ostroForm_unbounded_above G z hv m
  have := hm ⟨z.shift t, hC z hz t, rfl⟩
  linarith

/-! ## The quadratic-in-acceleration class -/

/-- `W` is symmetric with respect to the inner product. -/
def Symm (W : E →ₗ[ℝ] E) : Prop := ∀ x y : E, ⟪W x, y⟫ = ⟪x, W y⟫

/-- `L(q, v, a) = ½⟪a, W a⟫ + ⟪b(q,v), a⟫ + c(q,v)`. -/
noncomputable def quadLagrangian (W : E →ₗ[ℝ] E) (b : E → E → E) (c : E → E → ℝ)
    (q v a : E) : ℝ :=
  (1 / 2) * ⟪a, W a⟫ + ⟪b q v, a⟫ + c q v

/-- The acceleration momentum `p₂ = ∂L/∂a`. -/
def accelMomentum (W : E →ₗ[ℝ] E) (b : E → E → E) (q v a : E) : E := W a + b q v

/-- **Exact** expansion in the acceleration. The linear term is `⟪p₂, h⟫` and the quadratic term
is `½⟪h, W h⟫`, so the gradient is `p₂` and the Hessian is `W`. No remainder is left over. -/
theorem quadratic_lagrangian_expansion (W : E →ₗ[ℝ] E) (hW : Symm W) (b : E → E → E)
    (c : E → E → ℝ) (q v a h : E) :
    quadLagrangian W b c q v (a + h) =
      quadLagrangian W b c q v a + ⟪accelMomentum W b q v a, h⟫ +
        (1 / 2) * ⟪h, W h⟫ := by
  simp only [quadLagrangian, accelMomentum, map_add, inner_add_left, inner_add_right]
  have h1 : ⟪h, W a⟫ = ⟪W a, h⟫ := real_inner_comm _ _
  have h2 : ⟪a, W h⟫ = ⟪W a, h⟫ := (hW a h).symm
  rw [h1, h2]
  ring

/-- Legendre regularity in the acceleration: every `p₂` is attained by exactly one `a` at every
`(q, v)` if and only if `W` is bijective. -/
theorem legendre_regular_iff_bijective (W : E →ₗ[ℝ] E) (b : E → E → E) :
    (∀ q v p2 : E, ∃! a : E, accelMomentum W b q v a = p2) ↔ Function.Bijective W := by
  constructor
  · intro h
    have h0 := h 0 0
    refine ⟨fun x y hxy => ?_, fun y => ?_⟩
    · obtain ⟨a, -, ha⟩ := h0 (W x + b 0 0)
      have hx := ha x rfl
      have hy := ha y (by simp [accelMomentum, hxy])
      rw [hx, hy]
    · obtain ⟨a, ha, -⟩ := h0 (y + b 0 0)
      exact ⟨a, by simpa [accelMomentum] using ha⟩
  · rintro ⟨hinj, hsurj⟩ q v p2
    obtain ⟨a, ha⟩ := hsurj (p2 - b q v)
    refine ⟨a, by simp [accelMomentum, ha], fun a' ha' => hinj ?_⟩
    simp only [accelMomentum] at ha'
    rw [ha]
    exact eq_sub_of_add_eq ha'

/-- In finite dimension, solvability for every `p₂` is the same as `det W ≠ 0`. -/
theorem surjective_iff_det_ne_zero [FiniteDimensional ℝ E] (W : E →ₗ[ℝ] E) :
    Function.Surjective W ↔ LinearMap.det W ≠ 0 := by
  constructor
  · intro hs
    have hi : Function.Injective W := LinearMap.injective_iff_surjective.mpr hs
    have hu : IsUnit W := (Module.End.isUnit_iff W).mpr ⟨hi, hs⟩
    exact (LinearMap.isUnit_det W hu).ne_zero
  · intro hd
    exact (W.equivOfDetNeZero hd).surjective

/-- The canonical Ostrogradsky Hamiltonian of a Legendre-regular quadratic Lagrangian. The
acceleration is recovered as `a(p₂) = W⁻¹(p₂ - b)`. -/
noncomputable def regularHamiltonian (We : E ≃ₗ[ℝ] E) (b : E → E → E) (c : E → E → ℝ)
    (z : Phase E) : ℝ :=
  let a := We.symm (z.p2 - b z.q z.v)
  ⟪z.p1, z.v⟫ + ⟪z.p2, a⟫ - quadLagrangian (We : E →ₗ[ℝ] E) b c z.q z.v a

theorem regularHamiltonian_isOstroForm (We : E ≃ₗ[ℝ] E) (b : E → E → E) (c : E → E → ℝ) :
    ∃ G : E → E → E → ℝ, regularHamiltonian We b c = ostroForm G :=
  ⟨fun q v p2 =>
      ⟪p2, We.symm (p2 - b q v)⟫ -
        quadLagrangian (We : E →ₗ[ℝ] E) b c q v (We.symm (p2 - b q v)),
    by funext z; simp only [regularHamiltonian, ostroForm]; ring⟩

/-- The recovered acceleration really reproduces the given momentum. -/
theorem regular_momentum_inverts (We : E ≃ₗ[ℝ] E) (b : E → E → E) (q v p2 : E) :
    accelMomentum (We : E →ₗ[ℝ] E) b q v (We.symm (p2 - b q v)) = p2 := by
  simp [accelMomentum]

/-- **Regular class obstruction.** When `W` is invertible, the canonical Hamiltonian is
unbounded below and above at every phase point with `q̇ ≠ 0`, whatever `b` and `c` are. -/
theorem regular_hamiltonian_unbounded (We : E ≃ₗ[ℝ] E) (b : E → E → E) (c : E → E → ℝ)
    (z : Phase E) (hv : z.v ≠ 0) (bound : ℝ) :
    (∃ t : ℝ, regularHamiltonian We b c (z.shift t) < bound) ∧
      (∃ t : ℝ, bound < regularHamiltonian We b c (z.shift t)) := by
  obtain ⟨G, hG⟩ := regularHamiltonian_isOstroForm We b c
  rw [hG]
  exact ⟨ostroForm_unbounded_below G z hv bound, ostroForm_unbounded_above G z hv bound⟩

/-! ## Singular acceleration Hessian: primary constraints -/

section Singular

variable [FiniteDimensional ℝ E]

omit [FiniteDimensional ℝ E] in
theorem range_orthogonal_eq_ker (W : E →ₗ[ℝ] E) (hW : Symm W) :
    (LinearMap.range W)ᗮ = LinearMap.ker W := by
  ext e
  simp only [Submodule.mem_orthogonal, LinearMap.mem_range, LinearMap.mem_ker,
    forall_exists_index, forall_apply_eq_imp_iff]
  constructor
  · intro h
    have h1 : ⟪W (W e), e⟫ = 0 := h (W e)
    rw [hW] at h1
    exact inner_self_eq_zero.mp h1
  · intro h a
    rw [hW, h, inner_zero_right]

/-- A momentum `p₂` is attainable at `(q, v)` iff it satisfies the primary constraints
`⟪p₂ - b(q,v), e⟫ = 0` for every null direction `e ∈ ker W`. -/
theorem primary_constraint_iff (W : E →ₗ[ℝ] E) (hW : Symm W) (b : E → E → E) (q v p2 : E) :
    (∃ a, accelMomentum W b q v a = p2) ↔
      ∀ e ∈ LinearMap.ker W, ⟪p2 - b q v, e⟫ = 0 := by
  have hr : p2 - b q v ∈ LinearMap.range W ↔ p2 - b q v ∈ (LinearMap.ker W)ᗮ := by
    rw [← range_orthogonal_eq_ker W hW, Submodule.orthogonal_orthogonal]
  constructor
  · rintro ⟨a, rfl⟩ e he
    have hmem : W a ∈ (LinearMap.ker W)ᗮ := by
      rw [← range_orthogonal_eq_ker W hW, Submodule.orthogonal_orthogonal]
      exact ⟨a, rfl⟩
    have : accelMomentum W b q v a - b q v = W a := by simp [accelMomentum]
    rw [this, real_inner_comm]
    exact (Submodule.mem_orthogonal _ _).mp hmem e he
  · intro h
    have hmem : p2 - b q v ∈ (LinearMap.ker W)ᗮ := by
      rw [Submodule.mem_orthogonal]
      intro e he
      rw [real_inner_comm]
      exact h e he
    obtain ⟨a, ha⟩ := hr.mpr hmem
    exact ⟨a, by simp [accelMomentum, ha]⟩

/-- Count: the attainable momenta form an affine subspace of dimension `rank W`. The number of
independent primary constraints is `finrank (ker W) = n - rank W`. -/
theorem primary_count (W : E →ₗ[ℝ] E) :
    finrank ℝ (LinearMap.range W) + finrank ℝ (LinearMap.ker W) = finrank ℝ E :=
  LinearMap.finrank_range_add_finrank_ker W

/-- The primary constraint surface. -/
def primarySurface (W : E →ₗ[ℝ] E) (b : E → E → E) : Set (Phase E) :=
  {z | ∃ a, accelMomentum W b z.q z.v a = z.p2}

omit [FiniteDimensional ℝ E] in
/-- The primary surface does not involve `p₁`, so it is invariant under `p₁`-shifts. -/
theorem primarySurface_shift (W : E →ₗ[ℝ] E) (b : E → E → E) :
    ∀ z ∈ primarySurface W b, ∀ t : ℝ, z.shift t ∈ primarySurface W b := by
  rintro z ⟨a, ha⟩ t
  exact ⟨a, ha⟩

omit [FiniteDimensional ℝ E] in
/-- On the primary surface, the value `⟪p₂, a⟫ - L(q,v,a)` does not depend on which
acceleration `a` in the fibre produces `p₂`. This makes the reduced Hamiltonian well defined. -/
theorem reduced_value_fibre_independent (W : E →ₗ[ℝ] E) (hW : Symm W) (b : E → E → E)
    (c : E → E → ℝ) (q v a a' : E) (h : W a = W a') :
    ⟪accelMomentum W b q v a, a⟫ - quadLagrangian W b c q v a =
      ⟪accelMomentum W b q v a, a'⟫ - quadLagrangian W b c q v a' := by
  simp only [accelMomentum, quadLagrangian, inner_add_left]
  have e1 : ⟪a, W a⟫ = ⟪W a, a⟫ := real_inner_comm _ _
  have e2 : ⟪a', W a'⟫ = ⟪W a', a'⟫ := real_inner_comm _ _
  have e3 : ⟪W a, a'⟫ = ⟪W a', a'⟫ := by rw [h]
  have e4 : ⟪W a, a⟫ = ⟪W a', a'⟫ := by
    rw [hW, h, ← hW, h]
  rw [e1, e2, e3, e4]
  ring

omit [FiniteDimensional ℝ E] in
/-- **Degenerate class obstruction.** Take any Hamiltonian on the primary surface that has
Ostrogradsky form. (Every choice of reduced Hamiltonian built from the Legendre transform does,
by `reduced_value_fibre_independent`.) It is not bounded below on the primary surface as soon as
the surface contains a point with `q̇ ≠ 0`. So a singular Hessian alone does not remove the
ghost: secondary constraints involving `p₁` are needed. -/
theorem primary_surface_unbounded (W : E →ₗ[ℝ] E) (b : E → E → E) (G : E → E → E → ℝ)
    (z : Phase E) (hz : z ∈ primarySurface W b) (hv : z.v ≠ 0) :
    ¬ BddBelow (ostroForm G '' primarySurface W b) := fun hb =>
  hv (bddBelow_forces_v_zero G _ (primarySurface_shift W b) hb z hz)

end Singular

/-! ## S004 instance -/

/-- The fixed-flat S004 two-field Hessian `W = (-2Fκ)•id` is invertible when `Fκ ≠ 0`. So S004
belongs to the Legendre-regular class, and `regular_hamiltonian_unbounded` applies to it. -/
theorem s004_regular (F kappa : ℝ) (hF : F ≠ 0) (hk : kappa ≠ 0) :
    Function.Bijective ((-2 * F * kappa) • LinearMap.id : E →ₗ[ℝ] E) := by
  have hc : (-2 * F * kappa) ≠ 0 := by
    have : (-2 : ℝ) ≠ 0 := by norm_num
    exact mul_ne_zero (mul_ne_zero this hF) hk
  refine ⟨fun x y hxy => ?_, fun y => ⟨(-2 * F * kappa)⁻¹ • y, ?_⟩⟩
  · simp only [LinearMap.smul_apply, LinearMap.id_apply] at hxy
    exact smul_right_injective E hc hxy
  · simp only [LinearMap.smul_apply, LinearMap.id_apply, smul_smul]
    rw [mul_inv_cancel₀ hc, one_smul]

theorem s004_symm (F kappa : ℝ) : Symm ((-2 * F * kappa) • LinearMap.id : E →ₗ[ℝ] E) := by
  intro x y
  simp [real_inner_smul_left, real_inner_smul_right]

end ReleaseProofs.R16Class
