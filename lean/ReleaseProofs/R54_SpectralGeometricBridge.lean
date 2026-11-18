import Mathlib
import ReleaseProofs.R13_ScaleFreedom

/-!
# R54: explicit spectral/curve mass bridge and mismatch budget

A positive spectral eigenvalue lambda of a named operator produces the
conditional wavenumber sqrt(lambda)/L after an independently supplied
positive length L. Separately, a regular curve produces the curvature-
torsion statistic sigma=sqrt(kappa^2+tau^2). No theorem of the frozen
WCT model equates these inputs for arbitrary fields or curves.

This module proves the exact constant-curvature/torsion-free reduction
and a quantitative mismatch bound for positive normalized weights.
A concrete mismatch witness refutes an unrestricted spectral=geometric
identity. This neither constructs a confined PDE eigenstate nor fixes
the physical length scale, and it makes no particle-mass prediction.
-/

noncomputable section

namespace ReleaseProofs.R54SpectralGeometric

open ReleaseProofs.R13ScaleFreedom

/-- Physical inverse-length curve invariant, whenever curvature and
torsion are defined by a sufficiently regular nondegenerate curve. -/
def sigma (kappa tau : ℝ) : ℝ := Real.sqrt (kappa ^ 2 + tau ^ 2)

/-- Spectral mass conversion with independently specified length L. -/
def spectralMass (hbar c lambda L : ℝ) : ℝ :=
  (hbar / c) * spectralWavenumber lambda L

/-- Normalized-weight geometric mass candidate from supplied curve samples. -/
def geometricMass {n : ℕ} (hbar c : ℝ)
    (w kappa tau : Fin n → ℝ) : ℝ :=
  (hbar / c) * ∑ i, w i * sigma (kappa i) (tau i)

/-- The precise constant-curvature locked-loop reduction when torsion
is zero and nonnegative curvature equals the spectral wavenumber. -/
theorem sigma_zero_torsion (kappa : ℝ) (hk : 0 ≤ kappa) :
    sigma kappa 0 = kappa := by
  simp [sigma, Real.sqrt_sq_eq_abs, abs_of_nonneg hk]

/-- The constant-curvature circular case matches the spectral mass,
provided curvature itself is *supplied* as sqrt(lambda)/L.
This does not derive that relationship from the original PDE. -/
theorem constant_circle_mass_bridge
    (hbar c lambda L : ℝ) (hL : 0 < L) :
    (hbar / c) * sigma (spectralWavenumber lambda L) 0 =
      spectralMass hbar c lambda L := by
  have hk : 0 ≤ spectralWavenumber lambda L := by
    unfold spectralWavenumber
    exact div_nonneg (Real.sqrt_nonneg _) hL.le
  rw [sigma_zero_torsion _ hk]
  rfl

/-- Positive normalized weights bound the mismatch from an independently
chosen spectral wavenumber by the mean absolute curve discrepancy. -/
theorem weighted_constant_mismatch
    {n : ℕ} (w values : Fin n → ℝ) (k : ℝ)
    (hw : ∀ i, 0 ≤ w i) (hsum : (∑ i, w i) = 1) :
    |(∑ i, w i * values i) - k| ≤
      ∑ i, w i * |values i - k| := by
  have hid :
      (∑ i, w i * values i) - k =
        ∑ i, w i * (values i - k) := by
    calc
      (∑ i, w i * values i) - k =
          (∑ i, w i * values i) - (∑ i, w i) * k := by
        rw [hsum]
        ring
      _ = ∑ i, (w i * values i - w i * k) := by
          rw [Finset.sum_sub_distrib, Finset.sum_mul]
      _ = ∑ i, w i * (values i - k) := by
          apply Finset.sum_congr rfl
          intro i hi
          ring
  rw [hid]
  calc
    |∑ i, w i * (values i - k)| ≤
        ∑ i, |w i * (values i - k)| :=
      Finset.abs_sum_le_sum_abs _ _
    _ = ∑ i, w i * |values i - k| := by
      apply Finset.sum_congr rfl
      intro i hi
      rw [abs_mul, abs_of_nonneg (hw i)]

/-- Exact mismatch bound for the frozen two mass formulas, under a
genuinely certified normalization and nonnegative weights. -/
theorem spectral_geometric_mass_error
    {n : ℕ} (hbar c lambda L : ℝ)
    (w kappa tau : Fin n → ℝ)
    (hw : ∀ i, 0 ≤ w i) (hsum : (∑ i, w i) = 1) :
    |geometricMass hbar c w kappa tau -
        spectralMass hbar c lambda L| ≤
      |hbar / c| *
        ∑ i, w i *
          |sigma (kappa i) (tau i) - spectralWavenumber lambda L| := by
  unfold geometricMass spectralMass
  rw [← mul_sub, abs_mul]
  exact mul_le_mul_of_nonneg_left
    (weighted_constant_mismatch w (fun i => sigma (kappa i) (tau i))
      (spectralWavenumber lambda L) hw hsum)
    (abs_nonneg _)

/-- Exact counterexample: arbitrary geometric data cannot be
universally identified with a separately supplied spectral value.
Here the geometric wavenumber is 1 and the spectral wavenumber is 2. -/
theorem unrestricted_spectral_curve_identity_false :
    geometricMass (1 : ℝ) 1
      (fun _ : Fin 1 => 1)
      (fun _ => 1)
      (fun _ => 0) ≠ spectralMass 1 1 4 1 := by
  norm_num [geometricMass, spectralMass, sigma, spectralWavenumber]

end ReleaseProofs.R54SpectralGeometric
