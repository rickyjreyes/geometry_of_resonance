import Mathlib

/-!
# Geometry #46 — exact rotating homogeneous Q5 determinant normal form

This is the original corrected S004 rotating carrier's quadratic Q5
amplitude/phase matrix, in its reported physical-sign convention.
The results are *algebraic spectral structure*, not a claim of
primary finite-q instability, ghost removal, or a healthy nonlinear
wave-length selector. Background on-shell conditions and stability
classification remain separate obligations.
-/

noncomputable section
namespace ReleaseProofs.R46RotatingQ5

/-- On a homogeneous rotating carrier, the shifted kinetic argument. -/
def shifted (omega q z : ℝ) : ℝ := z - q + omega ^ 2

/-- The phase-channel Q5 diagonal symbol, with unexpanded background inputs. -/
def phaseB (F kappa theta gamma u Fprime omega q Vprime z : ℝ) : ℝ :=
  -(shifted omega q z) - Vprime +
    F * (kappa * ((shifted omega q z) ^ 2 + 4 * omega ^ 2 * z)
      + theta * q ^ 2 - gamma * q * (shifted omega q z))
    + u * kappa * omega ^ 4 * Fprime

/-- The amplitude-channel Q5 diagonal symbol. -/
def amplitudeA (F kappa theta gamma u Fprime Fsecond
    omega q Vprime Vsecond z : ℝ) : ℝ :=
  phaseB F kappa theta gamma u Fprime omega q Vprime z
    - 2 * u * Vsecond
    + 4 * u * Fprime *
        (kappa * omega ^ 2 * shifted omega q z
          - gamma * omega ^ 2 * q / 2)
    + 2 * u ^ 2 * kappa * omega ^ 4 * Fsecond

/-- The real coefficient of the off-diagonal i omega frequency coupling. -/
def couplingC (F kappa gamma u Fprime omega q z : ℝ) : ℝ :=
  -2 + 4 * F * kappa * shifted omega q z
    - 2 * F * gamma * q + 4 * u * Fprime * kappa * omega ^ 2

/-- Determinant of the full rotating amplitude/phase matrix,
before the physically irrelevant global factor four. -/
def rotatingDet (F kappa theta gamma u Fprime Fsecond
    omega q Vprime Vsecond z : ℝ) : ℝ :=
  amplitudeA F kappa theta gamma u Fprime Fsecond
    omega q Vprime Vsecond z *
  phaseB F kappa theta gamma u Fprime omega q Vprime z
    - omega ^ 2 * z *
      couplingC F kappa gamma u Fprime omega q z ^ 2

private def xZero (omega q : ℝ) : ℝ := omega ^ 2 - q

/-- The exact coefficient of z in the phase polynomial. -/
def bOne (F kappa gamma omega q : ℝ) : ℝ :=
  -1 + F * (kappa * (2 * xZero omega q + 4 * omega ^ 2) - gamma * q)

/-- The constant term of the phase polynomial. -/
def bZero (F kappa theta gamma u Fprime omega q Vprime : ℝ) : ℝ :=
  -xZero omega q - Vprime +
    F * (kappa * (xZero omega q) ^ 2 + theta * q ^ 2
      - gamma * q * xZero omega q)
    + u * kappa * omega ^ 4 * Fprime

def aOne (F kappa gamma u Fprime omega q : ℝ) : ℝ :=
  bOne F kappa gamma omega q + 4 * u * Fprime * kappa * omega ^ 2

def aZero (F kappa theta gamma u Fprime Fsecond omega q
    Vprime Vsecond : ℝ) : ℝ :=
  bZero F kappa theta gamma u Fprime omega q Vprime
    - 2 * u * Vsecond +
    4 * u * Fprime * (kappa * omega ^ 2 * xZero omega q
      - gamma * omega ^ 2 * q / 2) +
    2 * u ^ 2 * kappa * omega ^ 4 * Fsecond

def cZero (F kappa gamma u Fprime omega q : ℝ) : ℝ :=
  -2 + 4 * F * kappa * xZero omega q -
    2 * F * gamma * q + 4 * u * Fprime * kappa * omega ^ 2

theorem phaseB_quadratic
    (F kappa theta gamma u Fprime omega q Vprime z : ℝ) :
    phaseB F kappa theta gamma u Fprime omega q Vprime z =
      (F * kappa) * z ^ 2 +
      bOne F kappa gamma omega q * z +
      bZero F kappa theta gamma u Fprime omega q Vprime := by
  unfold phaseB bOne bZero xZero shifted
  ring

theorem amplitudeA_quadratic
    (F kappa theta gamma u Fprime Fsecond omega q Vprime Vsecond z : ℝ) :
    amplitudeA F kappa theta gamma u Fprime Fsecond
      omega q Vprime Vsecond z =
      (F * kappa) * z ^ 2 +
      aOne F kappa gamma u Fprime omega q * z +
      aZero F kappa theta gamma u Fprime Fsecond
        omega q Vprime Vsecond := by
  unfold amplitudeA phaseB aOne aZero bOne bZero xZero shifted
  ring

theorem couplingC_linear
    (F kappa gamma u Fprime omega q z : ℝ) :
    couplingC F kappa gamma u Fprime omega q z =
      4 * (F * kappa) * z + cZero F kappa gamma u Fprime omega q := by
  unfold couplingC cZero xZero shifted
  ring

/-- Every Q5-type product of quadratic diagonal symbols with a
linear rotating coupling is exactly quartic in its formal variable,
and its z^4 coefficient is h^2 irrespective of omega and all
lower-order coefficients. -/
theorem generic_rotating_quartic_identity
    (h a b c d e omega z : ℝ) :
    (h * z ^ 2 + a * z + b) *
        (h * z ^ 2 + c * z + d) -
        omega ^ 2 * z * (4 * h * z + e) ^ 2 =
      h ^ 2 * z ^ 4 +
        (h * (a + c) - 16 * omega ^ 2 * h ^ 2) * z ^ 3 +
        (h * (b + d) + a * c - 8 * omega ^ 2 * h * e) * z ^ 2 +
        (a * d + b * c - omega ^ 2 * e ^ 2) * z +
        b * d := by
  ring

/-- Exact rotating Q5 determinant with its actual source-action
coefficients; no small-q or small-frequency expansion was taken. -/
theorem rotatingDet_exact_quartic
    (F kappa theta gamma u Fprime Fsecond omega q Vprime Vsecond z : ℝ) :
    rotatingDet F kappa theta gamma u Fprime Fsecond
      omega q Vprime Vsecond z =
    let h := F * kappa
    let a := aOne F kappa gamma u Fprime omega q
    let b := aZero F kappa theta gamma u Fprime Fsecond
        omega q Vprime Vsecond
    let c := bOne F kappa gamma omega q
    let d := bZero F kappa theta gamma u Fprime omega q Vprime
    let e := cZero F kappa gamma u Fprime omega q
    h ^ 2 * z ^ 4 +
      (h * (a + c) - 16 * omega ^ 2 * h ^ 2) * z ^ 3 +
      (h * (b + d) + a * c - 8 * omega ^ 2 * h * e) * z ^ 2 +
      (a * d + b * c - omega ^ 2 * e ^ 2) * z +
      b * d := by
  unfold rotatingDet
  rw [amplitudeA_quadratic, phaseB_quadratic, couplingC_linear]
  exact generic_rotating_quartic_identity
    (F * kappa)
    (aOne F kappa gamma u Fprime omega q)
    (aZero F kappa theta gamma u Fprime Fsecond omega q Vprime Vsecond)
    (bOne F kappa gamma omega q)
    (bZero F kappa theta gamma u Fprime omega q Vprime)
    (cZero F kappa gamma u Fprime omega q)
    omega z

/-- In the strict nonzero-curvature regime the rotating polynomial
has a positive coefficient of the fourth power of squared frequency.
The degree-eight-in-omega mode content cannot be silently omitted. -/
theorem rotating_leading_coefficient_positive
    (F kappa : ℝ) (hF : 0 < F) (hk : 0 < kappa) :
    0 < (F * kappa) ^ 2 := by
  exact sq_pos_of_pos (mul_pos hF hk)

end ReleaseProofs.R46RotatingQ5
