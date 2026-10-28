import Mathlib
import ReleaseProofs.R18_CanonicalObstruction

/-!
# Exact real-pole classification and conditional finite-q extremum

In this file K=F(u₀)κ, G=F(u₀)γ, T=F(u₀)θ are the
*literal static homogeneous S004* coefficients. None of the separate
spatial-only rail assumptions is imported as a parent-model equivalence.
-/

namespace ReleaseProofs.R18Classification

open ReleaseProofs.R18

def discrVertex (K G T massSq : ℝ) : ℝ :=
  G ^ 2 + curvatureGap K G T * (1 - 4 * K * massSq)

/-- Completing the square in q describes ALL real-pole intervals. -/
theorem discriminant_vertex_identity (K G T massSq q : ℝ) :
    curvatureGap K G T * poleDiscriminant K G T massSq q =
      discrVertex K G T massSq -
        (curvatureGap K G T * q - G) ^ 2 := by
  unfold curvatureGap poleDiscriminant discrVertex
  ring

/-- On the strict curvature domain, a real pole is exactly a point inside
    the completed-square interval (possibly an endpoint). -/
theorem real_pole_iff_vertex_interval (K G T massSq q : ℝ)
    (hB : 0 < curvatureGap K G T) :
    0 ≤ poleDiscriminant K G T massSq q ↔
      (curvatureGap K G T * q - G) ^ 2 ≤
        discrVertex K G T massSq := by
  have hi := discriminant_vertex_identity K G T massSq q
  constructor
  · intro hd
    have hp : 0 ≤ curvatureGap K G T *
        poleDiscriminant K G T massSq q :=
      mul_nonneg hB.le hd
    nlinarith
  · intro hbound
    by_contra hn
    have hd : poleDiscriminant K G T massSq q < 0 :=
      lt_of_not_ge hn
    have hneg : curvatureGap K G T *
        poleDiscriminant K G T massSq q < 0 :=
      mul_neg_of_pos_of_neg hB hd
    nlinarith

/-- Negative vertex discriminant excludes real squared frequencies everywhere. -/
theorem everywhere_complex_if_vertex_negative (K G T massSq q : ℝ)
    (hB : 0 < curvatureGap K G T)
    (hN : discrVertex K G T massSq < 0) :
    poleDiscriminant K G T massSq q < 0 := by
  have hi := discriminant_vertex_identity K G T massSq q
  have hs : 0 ≤ (curvatureGap K G T * q - G) ^ 2 := sq_nonneg _
  by_contra hn
  have hd : 0 ≤ poleDiscriminant K G T massSq q :=
    le_of_not_gt hn
  have hp : 0 ≤ curvatureGap K G T *
      poleDiscriminant K G T massSq q :=
    mul_nonneg hB.le hd
  linarith

/-- A real-pole endpoint has exactly a repeated pole when the vertex is zero. -/
theorem vertex_zero_collision (K G T massSq q : ℝ)
    (hB : 0 < curvatureGap K G T)
    (hN : discrVertex K G T massSq = 0) :
    poleDiscriminant K G T massSq q = 0 ↔
      curvatureGap K G T * q = G := by
  have hi := discriminant_vertex_identity K G T massSq q
  constructor
  · intro hd
    rw [hd, hN] at hi
    have hz : (curvatureGap K G T * q - G) ^ 2 = 0 := by
      linarith
    nlinarith
  · intro heq
    have hz : curvatureGap K G T *
        poleDiscriminant K G T massSq q = 0 := by
      rw [hN, sub_eq_zero.mpr heq] at hi
      simpa using hi
    exact (mul_eq_zero.mp hz).resolve_left (ne_of_gt hB)

/-- This gives an exact, symbolically checkable UV-complex criterion. -/
theorem uv_complex_beyond_vertex (K G T massSq q : ℝ)
    (hB : 0 < curvatureGap K G T)
    (hfar : discrVertex K G T massSq <
      (curvatureGap K G T * q - G) ^ 2) :
    poleDiscriminant K G T massSq q < 0 := by
  have hi := discriminant_vertex_identity K G T massSq q
  by_contra hn
  have hd : 0 ≤ poleDiscriminant K G T massSq q :=
    le_of_not_gt hn
  have hp : 0 ≤ curvatureGap K G T *
      poleDiscriminant K G T massSq q :=
    mul_nonneg hB.le hd
  linarith

/-- Exact algebra for a tangent to the square root of the discriminant,
    at a proposed stationary point. No derivative was truncated. -/
theorem tangent_gap_identity (K G T massSq q qStar A : ℝ)
    (hstat : G - curvatureGap K G T * qStar =
      A * Real.sqrt (poleDiscriminant K G T massSq qStar))
    (hdStar : 0 ≤ poleDiscriminant K G T massSq qStar)
    (hdq : 0 ≤ poleDiscriminant K G T massSq q) :
    (Real.sqrt (poleDiscriminant K G T massSq qStar) +
       A * (q - qStar)) ^ 2 -
       (Real.sqrt (poleDiscriminant K G T massSq q)) ^ 2 =
         (A ^ 2 + curvatureGap K G T) * (q - qStar) ^ 2 := by
  have hshift :
      poleDiscriminant K G T massSq q =
        poleDiscriminant K G T massSq qStar +
          2 * (G - curvatureGap K G T * qStar) * (q - qStar) -
          curvatureGap K G T * (q - qStar) ^ 2 := by
    unfold poleDiscriminant curvatureGap
    ring
  have hsqStar := Real.sq_sqrt hdStar
  have hsqQ := Real.sq_sqrt hdq
  have hstep := congrArg (fun x : ℝ => 2 * x * (q - qStar)) hstat
  rw [hsqStar, hsqQ]
  nlinarith [hshift, hstep]

/-- A stationary lower pole minimizes frequency on every real-pole window
    where its tangent is positive; an explicit *global domain* condition is
    retained, not silently inferred from the polynomial alone. -/
theorem lower_pole_minimum_on_tangent_window
    (K G T massSq q qStar A : ℝ)
    (hK : 0 < K) (hB : 0 < curvatureGap K G T)
    (hA : A = 2 * K + G)
    (hstat : G - curvatureGap K G T * qStar =
      A * Real.sqrt (poleDiscriminant K G T massSq qStar))
    (hdStar : 0 ≤ poleDiscriminant K G T massSq qStar)
    (hdq : 0 ≤ poleDiscriminant K G T massSq q)
    (hpositiveTangent :
      0 < Real.sqrt (poleDiscriminant K G T massSq qStar) +
        A * (q - qStar)) :
    lowerPole K G T massSq qStar ≤ lowerPole K G T massSq q := by
  have hi := tangent_gap_identity K G T massSq q qStar A hstat hdStar hdq
  have hcoeff : 0 ≤ A ^ 2 + curvatureGap K G T := by
    nlinarith [sq_nonneg A]
  have hprod : 0 ≤ (A ^ 2 + curvatureGap K G T) * (q - qStar) ^ 2 :=
    mul_nonneg hcoeff (sq_nonneg _)
  have hsq : 0 ≤ Real.sqrt (poleDiscriminant K G T massSq q) :=
    Real.sqrt_nonneg _
  have hrootLe :
      Real.sqrt (poleDiscriminant K G T massSq q) ≤
      Real.sqrt (poleDiscriminant K G T massSq qStar) +
        A * (q - qStar) := by
    nlinarith [hi]
  have hdiff : 2 * K *
      (lowerPole K G T massSq q - lowerPole K G T massSq qStar) =
      Real.sqrt (poleDiscriminant K G T massSq qStar) +
        A * (q - qStar) -
        Real.sqrt (poleDiscriminant K G T massSq q) := by
    unfold lowerPole
    rw [hA]
    field_simp [ne_of_gt hK]
    ring
  have hnon : 0 ≤ 2 * K *
      (lowerPole K G T massSq q - lowerPole K G T massSq qStar) := by
    linarith
  nlinarith

theorem lower_pole_unique_on_tangent_window
    (K G T massSq q qStar A : ℝ)
    (hK : 0 < K) (hB : 0 < curvatureGap K G T)
    (hA : A = 2 * K + G)
    (hstat : G - curvatureGap K G T * qStar =
      A * Real.sqrt (poleDiscriminant K G T massSq qStar))
    (hdStar : 0 ≤ poleDiscriminant K G T massSq qStar)
    (hdq : 0 ≤ poleDiscriminant K G T massSq q)
    (hpositiveTangent :
      0 < Real.sqrt (poleDiscriminant K G T massSq qStar) +
        A * (q - qStar))
    (heq : lowerPole K G T massSq q =
      lowerPole K G T massSq qStar) :
    q = qStar := by
  have hi := tangent_gap_identity K G T massSq q qStar A hstat hdStar hdq
  have hdiff : 2 * K *
      (lowerPole K G T massSq q - lowerPole K G T massSq qStar) =
      Real.sqrt (poleDiscriminant K G T massSq qStar) +
        A * (q - qStar) -
        Real.sqrt (poleDiscriminant K G T massSq q) := by
    unfold lowerPole
    rw [hA]
    field_simp [ne_of_gt hK]
    ring
  rw [heq, sub_self, mul_zero] at hdiff
  have hroot : Real.sqrt (poleDiscriminant K G T massSq qStar) +
        A * (q - qStar) =
        Real.sqrt (poleDiscriminant K G T massSq q) := by
    linarith
  rw [hroot] at hi
  have hcoeff : 0 < A ^ 2 + curvatureGap K G T := by
    nlinarith [sq_nonneg A]
  have hzero : (q - qStar) ^ 2 = 0 := by
    have hproduct : (A ^ 2 + curvatureGap K G T) * (q - qStar) ^ 2 = 0 := by
      linarith
    exact (mul_eq_zero.mp hproduct).resolve_left (ne_of_gt hcoeff)
  nlinarith


/-- Exact strict real-pole window: no positivity of the mass parameter is
    assumed. This includes any disconnected positive-q islands when present. -/
theorem distinct_real_pole_iff_strict_vertex (K G T massSq q : ℝ)
    (hB : 0 < curvatureGap K G T) :
    0 < poleDiscriminant K G T massSq q ↔
      (curvatureGap K G T * q - G) ^ 2 <
        discrVertex K G T massSq := by
  have hi := discriminant_vertex_identity K G T massSq q
  constructor
  · intro hd
    have hp : 0 < curvatureGap K G T *
        poleDiscriminant K G T massSq q := mul_pos hB hd
    linarith
  · intro hd
    by_contra hn
    have hnon : poleDiscriminant K G T massSq q ≤ 0 := le_of_not_gt hn
    have hp : curvatureGap K G T *
        poleDiscriminant K G T massSq q ≤ 0 :=
      mul_nonpos_of_nonneg_of_nonpos hB.le hnon
    linarith

/-- The precise pole-collision surface, including both endpoints of a
    bounded real-pole interval when the vertex is positive. -/
theorem collision_iff_vertex_boundary (K G T massSq q : ℝ)
    (hB : 0 < curvatureGap K G T) :
    poleDiscriminant K G T massSq q = 0 ↔
      (curvatureGap K G T * q - G) ^ 2 =
        discrVertex K G T massSq := by
  have hi := discriminant_vertex_identity K G T massSq q
  constructor
  · intro hd
    rw [hd] at hi
    linarith
  · intro heq
    have hp : curvatureGap K G T *
        poleDiscriminant K G T massSq q = 0 := by
      linarith
    exact (mul_eq_zero.mp hp).resolve_left (ne_of_gt hB)

/-- The UV discriminant is strictly negative beyond the exact right
    collision endpoint, irrespective of the sign of G. -/
theorem uv_complex_beyond_upper_collision (K G T massSq q : ℝ)
    (hB : 0 < curvatureGap K G T)
    (hN : 0 ≤ discrVertex K G T massSq)
    (hq : (G + Real.sqrt (discrVertex K G T massSq)) /
        curvatureGap K G T < q) :
    poleDiscriminant K G T massSq q < 0 := by
  have hs : (Real.sqrt (discrVertex K G T massSq)) ^ 2 =
      discrVertex K G T massSq := Real.sq_sqrt hN
  have hr : 0 ≤ Real.sqrt (discrVertex K G T massSq) :=
    Real.sqrt_nonneg _
  have hmul := (div_lt_iff₀ hB).mp hq
  have hdif : Real.sqrt (discrVertex K G T massSq) <
      curvatureGap K G T * q - G := by
    nlinarith
  have hsq :
      discrVertex K G T massSq <
        (curvatureGap K G T * q - G) ^ 2 := by
    nlinarith [hs]
  exact uv_complex_beyond_vertex K G T massSq q hB hsq

end ReleaseProofs.R18Classification
