# Adversarial splitting analysis and executed diagnostics

Date: 2026-10-22. **STRICT_BINDING_RESTRICTED.** The distinction between a
counterexample to a proof shortcut and a counterexample to strict binding is
essential. No finite-mass counterexample to the frozen functional's binding
inequality was established.

The exact functional and theorem range are in
[SUBADDITIVITY_PROOF_OR_COUNTEREXAMPLE.md](SUBADDITIVITY_PROOF_OR_COUNTEREXAMPLE.md).
All coefficients in the actual-model trials are unchanged. Parameter limits
below are explicitly boundary controls, not replacements for the D4 problem.

## A1. Analytical attempts to force dichotomy

| Candidate | Method | Result |
|---|---|---|
| Two compact near-minimizers translated arbitrarily far apart | ANALYTICAL | Energies add; proves weak subadditivity. They are minimizing for the total mass only if equality of the infima is independently proved. |
| An equality split at a fixed positive mass | ANALYTICAL, conditional premise | Produces a dichotomizing minimizing sequence. Such equality is excluded in S2's range and remains unresolved outside it. |
| Infimum negative but unattained at frozen coefficients | UNRESOLVED outside the proved range | No construction achieves the actual infimum. Ruled out for 0<m<=m_* by C1. |
| Binding only below a critical mass | UNRESOLVED | m_* is a sufficient analytical bound, not a demonstrated critical threshold. |
| A vanishing shell-carrier sequence | ANALYTICAL | Energy tends to zero; cannot minimize because I(m)<0 for each fixed positive m. |
| A high-frequency remainder with vanishing L² mass | ANALYTICAL | Its nonlinear powers vanish under the H² bound; the remaining squared A energy is nonnegative. It does not supply an unaccounted negative profile. |
| Splits with one component tending to zero | ANALYTICAL limiting observation | The binding gap tends to zero by continuity. This prevents a uniform gap over all splits, but is not equality at a fixed positive split. |
| Macroscopic components as m tends to infinity | ANALYTICAL limiting observation | The gap divided by m tends to zero. No finite-mass equality follows. |

To justify the last two statements, define the nonnegative gap

    Delta(m,mu)=I(mu)+I(m-mu)-I(m).

Continuity gives Delta(m,mu)->0 as mu->0 or mu->m. S1 proves strict positivity
for every fixed split in its range; it does not claim one split-independent
positive lower bound. The vanishing mass component in this endpoint limit
is different from dichotomy, which requires two fixed positive limiting masses.

Here are further analytical bounds, proved without assuming strict binding:

    -Cm < I(m)<0 for every m>0,
    I(m)/m -> 0 as m->0,
    I(m)/m -> -C as m->infinity,  C=3beta²/(16gamma).

**Strict lower bound at finite mass.** If I(m)=-Cm, a minimizing sequence
satisfies T(u_n)->0 by square completion. Its negative energy excludes
vanishing, so translating gives a nonzero weak H² profile u. Weak continuity
of A implies Au=0. Its L² Fourier transform would be supported on the
measure-zero circle |xi|=k, forcing u=0. This contradiction proves I(m)>-Cm.

**Small mass.** On a negative-energy minimizing sequence, F3 gives
||u_n||_infinity²<=K_inf m. Thus Q(u_n)<=K_inf m² and
I(m)>=-beta K_inf m²/2. Combined with I(m)<0, this proves I(m)/m->0.

**Large mass.** Set rho0=3beta/(4gamma). Let chi_R be a smooth radial cutoff,
one on the disk of radius R, zero outside radius R+1, with uniformly bounded
first and second derivatives in the annulus. The carrier field
u_R=sqrt(rho0) chi_R exp(i k x) has mass rho0*pi R²+O(R). The squared A energy
and the square-completed potential defect are O(R), since both are supported
in the annulus. Hence H(u_R)=-C M(u_R)+O(R). Choose a monotone cutoff so M(u_R)
is continuous and increasing; all sufficiently large prescribed masses occur.
Together with the lower bound this gives -Cm<I(m)<=-Cm+O(sqrt(m)). For any
fixed 0<theta<1, it follows that

    0<=Delta(m,theta*m)/m -> 0.

These limits explain plausible near-equality regimes without asserting a
finite critical mass or nonattainment.

## A2. Explicit failures of two universal scaling shortcuts

Let the normalized carrier Gaussian have mass m, equal widths w, and carrier
wave number q. Its moments are

    Q/m = m/(2pi w²),  S/m = m²/(3pi²w⁴),
    K/m = q²+1/w²,
    D/m = q⁴+4q²/w²+2/w⁴.

For the actual frozen beta=gamma and k=1, take w=10.

1. **Amplitude shortcut.** Let m=150pi and q=1. Then

       H/m = 101/5000-beta/8 < 0,
       d/dtheta [H(sqrt(theta)u)/theta]_(theta=1) /m = beta/8 > 0.

   Negative energy therefore does not imply the pointwise amplitude inequality
   used for a pure focusing power. The executed numerical check gives
   H=-231.5717669... and derivative +241.0907927.... The signs also follow
   exactly from the dyadic beta and the displayed rational coefficients.

2. **Density-preserving spatial dilation shortcut.** Let m=75pi and q²=3/4.
   Then K/m=19/25, D/m=5927/10000 and

       H/m = 727/10000-beta/8 < 0,
       d/dtheta [H(u(./sqrt(theta)))/theta]_(theta=1) /m
           = 1673/5000 > 0.

   The diagnostic gives H=-103.4158624... and derivative +78.8382676....
   The carrier must also be dilated; holding it fixed would test a different
   transformation.

These profiles are admissible whole-space fields. They are not demonstrated
minimizers. Accordingly they refute the proposed universal trial inequalities
and do **not** refute strict subadditivity of the infimum.

## A3. Degenerate limits

| Boundary control | Analytical result | Relation to frozen D4 |
|---|---|---|
| beta=0, gamma>=0, k>0 | I(m)=0 for every m>0; equality holds for all splits; no minimizer | Outside beta>0; not a counterexample at the frozen coefficients |
| gamma=0, beta>0, k>0, dimension two | Strict binding for all positive masses | A different boundary equation; shows m_* is not a sharp critical mass |
| k=0, beta,gamma>0, dimension two | Strict binding for all positive masses | Shell disappears; not the k=1 problem |

For beta=0, H>=0 and broad carrier packets reach zero. Equality would require
Au=0, impossible for a nonzero L² field. This is an **unattained zero** infimum,
not an example of an unattained negative infimum for the actual model.

For gamma=0, the two-dimensional interpolation inequality
Q<=M^(3/2)||Delta u||/4 (from ||u||_infinity²<=||u||_2||Delta u||_2/4)
gives fixed-mass coercivity after absorbing the negative gradient term.
The negative Gaussian construction still works. Along a minimizing sequence
Q is bounded below, and amplitude scaling strictly decreases H/theta by
beta(theta-1)Q/2 for every theta>1. The uniform gap proves all-mass binding.

For k=0, the same Gaussian dilation has a negative R^(-2) term and positive
R^(-4) term, proving negative infima for every mass. Density-preserving
dilation lowers H/theta by (1-theta^(-2))D. Along a negative minimizing
sequence, Q is bounded below; Q<=M^(3/2)sqrt(D)/4 bounds D away from zero.
This again proves a uniform strict gap and all-mass binding. The interpolation
constant follows from weighted Fourier Cauchy–Schwarz with a+|xi|⁴ and
optimization over a>0. Neither boundary proof is used to fill the gap at k=1.

## A4. Executed finite-dimensional diagnostics

The script [run_diagnostics.py](../../audit/strict_binding/run_diagnostics.py)
records every result in [diagnostics.json](../../audit/strict_binding/diagnostics.json).
It executes:

- An independent adaptive radial quadrature for the Fourier constant J.
- Fourteen prescribed masses, each with three deterministic Gaussian starts.
  The two widths and carrier are trial parameters; beta,gamma,k are fixed.
- Seven two-component mass partitions at m_D4.
- Twenty-seven separation/relative-phase comparisons of two Gaussian packets,
  with the total mass renormalized before comparing energies.
- The two explicit negative-energy scaling counterexamples above.

The Gaussian search is a bounded finite-dimensional trial optimization. It
has no lower bound on I and no certificate of global optimality, even within
the trial family. The stored optimizer flags and evaluation counts are
diagnostic. No PDE propagation, continuum stationary root calculation, or
new interval root/coercivity campaign was executed.

Selected optimized Gaussian trial energies (approximate upper bounds):

| Norm m | Trial H |
|---|---:|
| 1 | -0.0000890620 |
| 2 | -0.00295080 |
| m_* | -0.00553710 |
| m_D4 | -0.01007250 |
| 4 | -0.0999699 |
| 16 | -5.73641 |
| 256 | -145.38778 |

All seven sampled split-trial sums exceed the selected total-mass trial
energy. That is **not** a proof that I(m)<I(mu)+I(m-mu): both split values are
upper bounds and can overestimate the true infima. The receipt explicitly
marks every such comparison `certifies_infimum_comparison: false`.

The two-packet interaction differences have both signs (approximately
-0.193847 through +0.623370 in the recorded family). At separation 64 with
width 4, all four tested phases agree with additive energy to <1e-12.
This verifies the expected approach to additivity of these trial profiles;
it does not establish a dichotomizing minimizing sequence. The kinetic cross
term includes its complex phase, and the norm cross term is normalized. An
independent full-field Gaussian-Hermite quadrature checks this implementation.

## A5. Meaning of the adversarial outcome

No actual-coefficient equality split, negative unattained infimum, or sharp
critical mass was proved. The analytical sufficient range and central D4
split interval survive the adversarial checks. The remaining small-component
splits at m_D4, and the universal assertion over all m>0, remain OPEN.
The correct primary label is therefore **STRICT_BINDING_RESTRICTED**.
