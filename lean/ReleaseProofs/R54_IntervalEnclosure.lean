import ReleaseProofs.R54_SpectralGeometricBridge

/-!
# R54: certified interval propagation for the spectral/curve mass bridge

This module combines an independently provided interval for a nonnegative
spectral eigenvalue and the already checked geometric mismatch budget.
The eigenvalue endpoints and curve discrepancy are supplied inputs,
*not* certified here from a PDE eigenfunction or finite-element residual.
The frozen physical length and conversion factor are kept explicit.

No particle-mass predictions or validation of a self-bound state follows.
-/

noncomputable section

namespace ReleaseProofs.R54SpectralGeometric

open ReleaseProofs.R13ScaleFreedom

/-- The source-matched conversion sqrt(lambda)/L is monotone in a
nonnegative confined spectral eigenvalue, provided L>0 and hbar/c>=0.
This identifies exactly which input interval endpoints yield a
one-sided mass enclosure. -/
theorem spectralMass_monotone
    (hbar c L lambdaLo lambdaHi : ℝ)
    (hhbar : 0 ≤ hbar) (hc : 0 < c) (hL : 0 < L)
    (hinterval : lambdaLo ≤ lambdaHi) :
    spectralMass hbar c lambdaLo L ≤
      spectralMass hbar c lambdaHi L := by
  unfold spectralMass spectralWavenumber
  exact mul_le_mul_of_nonneg_left
    (div_le_div_of_nonneg_right (Real.sqrt_le_sqrt hinterval) hL.le)
    (div_nonneg hhbar hc.le)

/-- Both endpoints of an independently certified eigenvalue interval
enclose the conversion at its actual (also supplied) eigenvalue. -/
theorem spectralMass_eigenvalue_interval
    (hbar c L lambdaLo lambda lambdaHi : ℝ)
    (hhbar : 0 ≤ hbar) (hc : 0 < c) (hL : 0 < L)
    (hlo : lambdaLo ≤ lambda) (hhi : lambda ≤ lambdaHi) :
    spectralMass hbar c lambdaLo L ≤ spectralMass hbar c lambda L ∧
    spectralMass hbar c lambda L ≤ spectralMass hbar c lambdaHi L := by
  exact ⟨spectralMass_monotone hbar c L lambdaLo lambda hhbar hc hL hlo,
    spectralMass_monotone hbar c L lambda lambdaHi hhbar hc hL hhi⟩

/-- A rigorous two-sided *geometric-mass* enclosure whenever a supplied
eigenvalue enclosure and the nonnegative normalized-weight discrepancy
are valid. This is an error-propagation theorem, not an eigenvalue
certificate: no source discretization or eigenstate construction is
hidden in the hypotheses. -/
theorem geometricMass_interval_from_eigenvalue_enclosure
    {n : ℕ} (hbar c L lambdaLo lambda lambdaHi : ℝ)
    (w kappa tau : Fin n → ℝ)
    (hhbar : 0 ≤ hbar) (hc : 0 < c) (hL : 0 < L)
    (hlo : lambdaLo ≤ lambda) (hhi : lambda ≤ lambdaHi)
    (hw : ∀ i, 0 ≤ w i) (hsum : (∑ i, w i) = 1) :
    let discrepancy := |hbar / c| *
      ∑ i, w i *
        |sigma (kappa i) (tau i) - spectralWavenumber lambda L|
    spectralMass hbar c lambdaLo L - discrepancy ≤
      geometricMass hbar c w kappa tau ∧
    geometricMass hbar c w kappa tau ≤
      spectralMass hbar c lambdaHi L + discrepancy := by
  dsimp
  have hspec := spectralMass_eigenvalue_interval
    hbar c L lambdaLo lambda lambdaHi hhbar hc hL hlo hhi
  have hdiff := abs_le.mp
    (spectral_geometric_mass_error hbar c lambda L w kappa tau hw hsum)
  constructor <;> linarith

end ReleaseProofs.R54SpectralGeometric
