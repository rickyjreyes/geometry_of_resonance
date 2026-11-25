# Theory and transformation ledger

Relation status and transformation status answer different questions. An **ALREADY_DERIVED** conditional identity can still require a **HYPOTHESIS** before it applies to an experiment. “Already” means it exists in the cited corpus or completed preceding audit; it does not automatically mean it predicted an empirical value before observation. Chronology is explicit below.

Sources P01–P22 and S004 refer to the pinned [preceding derivation ledger](../cross_paper/CROSS_PAPER_DERIVATION_LEDGER.md) and [source inventory](../../audit/cross_paper/SOURCE_LEDGER.json). P05 is the October 6 phase/window-covariance revision. Empirical sources are pinned in this audit's source ledger. The indexed corpus was searched across all requested relation classes, including current revisions; unavailable or unindexed documents are outside the coverage claim.

| Relation | Relation status | Transformation status | Mathematical content and physical limit |
|---|---|---|---|
| Angular frequency versus cycles | ALREADY_DERIVED | DERIVED | f_log=nu/(2pi); not an integer unless independent closure holds |
| DSI period | ALREADY_DERIVED | DERIVED conditionally | lambda=exp(2pi/nu) for a fundamental one-cycle log mode; periodic re-expression does not prove physical DSI |
| Scale-winding theorem, P03 | ALREADY_DERIVED | DERIVED conditionally | nu L=2pi n and lambda=exp(L/n); requires a physical log eigenmode and appropriate boundary-phase closure |
| General power coordinate | ALREADY_DERIVED | DERIVED | y=c x^p gives ell_y=p ell_x+constant, nu_y=nu_x/p and lambda_y=lambda_x^p |
| Mass to dimuon q2 | ALREADY_DERIVED | PHYSICALLY_JUSTIFIED | Actual source uses q2=m_mumu^2, so k_q2=omega_m/2; cross-channel waveform equality is additional physics |
| Spectroscopic wavenumber to photon energy | ALREADY_DERIVED | PHYSICALLY_JUSTIFIED | E=hc sigma is multiplication by a constant, hence unchanged log-frequency magnitude |
| Mass to energy | ALREADY_DERIVED | PHYSICALLY_JUSTIFIED | E=Mc^2 changes reference/phase, not log-frequency; it does not equate a population spectrum to a collider response |
| Root mass to mass | ALREADY_DERIVED | DERIVED, but application conditional | m=r^2 gives nu_m=nu_r/2 and lambda_m=lambda_r^2; none of the primary measured coordinates is root mass |
| Reciprocal variable | ALREADY_DERIVED | DERIVED | nu changes sign, lambda inverts; magnitudes coincide, so this is not an extra good fit |
| Unit or reference normalization | ALREADY_DERIVED | DERIVED | x→c x translates ell and shifts phase only; cannot tune frequency |
| Physical width covariance, P05 | PROPOSED physical rule | HYPOTHESIS over a conditional identity | With independently fixed n, nu'=nu L/L'; changing an analysis crop is not changing the physical boundary |
| Spatial-to-log phase map | ALREADY_DERIVED in preceding audit | DERIVED chain rule | nu=[dTheta/ds]/[d ln X/ds]; constant spatial carrier is not automatically constant log-frequency |
| Corrected S004 finite-k branch | ALREADY_DERIVED in preceding audit | DERIVED in controlled regime | q_star=-C2/(2C4) at retained EFT order; parameter-dependent spatial L^-2 scale, not dimensionless nu |
| Universal k_star R | PROPOSED | HYPOTHESIS | Dimensionless, but R and the observable phase map must be independently supplied; dimensional compatibility is insufficient |
| PFF band center, P02 | ALREADY_DERIVED within chosen rail | DERIVED conditionally | sigma=r+a k²-b k⁴, k_star²=a/(2b); S004 damping/preparation and nonlinear completion remain additional premises |
| Shell/radial quantization, P02/P06/P10 | PROPOSED physical application | HYPOTHESIS | Discrete spectra require an actual operator, state, domain and boundary conditions; a fourth-order Hessian is not automatically a chosen second-order radial potential |
| Curvature/dispersion mass, P07 | PROPOSED physical identification | HYPOTHESIS | A normalized gap can set an effective mass on the same branch; curvature, rest energy and response spectra need the physical normalization |
| Loop rest-energy/locking, P08/P22 | ALREADY_DERIVED conditional construction | HYPOTHESIS for S004 embedding | E_lock,min=(2pi n-S_sigma)^2/I_w selects nearest n only for a supplied loop and weight, not for four unsolved systems |
| 2–3–6 norm geometry, P05 | ALREADY_DERIVED | DERIVED | T†T=6 and singular value sqrt6 are exact finite-dimensional geometry; an iterated physical dilation operator is not thereby obtained |
| Root-mass sqrt6 gain as physical recurrence | PROPOSED | HYPOTHESIS | If genuine, mass gain is 6, not sqrt6; recursion and observable embedding remain unproved |
| CMS lambda≈sqrt6 | NUMERICAL_ANALOGY | HYPOTHESIS | Explicitly retrospective in P05; physical scale assignment not derived |
| NIST lambda≈sqrt(3/2) | NUMERICAL_ANALOGY | HYPOTHESIS | Exact comparison is reproducible; no same-action selector assigns this geometric norm to atomic line density |
| LHCb a≈sqrt(3/2) | NUMERICAL_ANALOGY | HYPOTHESIS | May paper §15.2 explicitly leaves why a regional dilation should equal this norm open |
| lambda_CMS≈2 lambda_NIST | NUMERICAL_ANALOGY | HYPOTHESIS | Ratio follows from the two chosen geometric comparators; dividing lambda by two is not the mass/root-mass power rule |
| Harmonic/subharmonic sectors | PROPOSED | HYPOTHESIS | nu_n=n nu_1 implies lambda_n=lambda_1^(1/n); nu_n=nu_1/n implies lambda_n=lambda_1^n; n cannot be chosen for agreement |
| Universal frequency equality U0/U1 | NEW_CANDIDATE | HYPOTHESIS | Explicit retrospective low-flexibility stress models; not the S004 prediction |
| CMS phase convention to LHCb | ALREADY_DERIVED | DERIVED | A cos(omega ell_m-phi_CMS) maps to A cos((omega/2)ell_q2+phi_LHCb) with phi_LHCb=-phi_CMS |
| Fit separate exponents/integers after seeing targets | NEW_CANDIDATE negative control | POST_HOC_NOT_ALLOWED | Labels in null diagnostics quantify flexibility only; not physical sectors |
| Treat LHCb a as lambda and compute 2pi/ln(a) | NUMERICAL_ANALOGY | POST_HOC_NOT_ALLOWED | Wrong operational type unless a new physical bridge is derived |
| lambda=exp(2pi n/L) in the proposed example menu | NEW_CANDIDATE | POST_HOC_NOT_ALLOWED as DSI identity | With closure, exponent is nu, whereas the period is exp(L/n)=exp(2pi/nu); these are not the same formula |
| Four arbitrary multipliers | NUMERICAL_ANALOGY | POST_HOC_NOT_ALLOWED | Saturates four positive targets; held-out multiplier remains unconstrained |

## Start from the verified S004 branch

In the inherited conventions, q=k_spatial², z=omega_time², K=F(u0)kappa, G=F(u0)gamma, T=F(u0)theta and m²=-2u0 V''(u0), with V'(u0)=0. The lower pole has the expansion

\[
z_-=C_0+C_2q+C_4q^2+O(q^3),\quad
s=\sqrt{1-4Km^2},\quad C_0=\frac{1-s}{2K},
\]
\[
C_2=1-\frac{GC_0}{s},\qquad
C_4=\frac{T}{s}+\frac{G^2m^2}{s^3},\qquad
k_\star^2=-\frac{C_2}{2C_4}.
\]

The controlled witness has m²=M², K=delta/M², G=g/M², T=h/(delta M²), delta<<1, g>1, h>g²/4. Then k_star²≈delta M²(g-1)/(2h), with the inherited low-energy cutoff and error control. The preceding audit also derives the exact pole minimum. This is real mathematical progress. It does not fix delta, g, h, M from the four experiments, choose their backgrounds, create a localized bound state, or select an observable recurrence. The retained low-energy positive-residue branch does not erase the upper ghost pole of the literal higher-time action.

The PFF overdamped coefficient correspondence r=-Gamma C0, a=-Gamma C2, b=Gamma C4 shares this band center once damping is independently assumed. The stable witness is least damped at finite k, not automatically an unstable growing mode or a confined nonlinear solution.

## Why the same spatial scale does not yet yield one log-frequency

For a spatial phase Theta(s)=k_star s and a positive observable X(s),

\[
\nu(s)=\frac{k_\star}{d\ln X/ds}.
\]

For X=s this is k_star s, not a constant. A constant nu would require an appropriate exponential X(s), or a different derived phase profile. In each physical system the map X, response weighting, active endpoints and sector must be obtained independently. Selecting an R so that k_star R equals a measured nu only fits a missing conversion.

An ordinary fixed-radius radial ladder m_n proportional to n/R has adjacent ratios (n+1)/n, not a constant DSI ratio. Eigenvalue, spatial wavenumber, effective mass and angular log-frequency cannot be relabeled as one another.

## Chronology

The LHCb scale result is in a May 11 repository commit; the readable paper is dated May 13, original release May 9. Canonical Fe II results are preserved by May 25 and the paper is dated May 28. CMS froze its discovered frequency August 28. The October 6 P05 table explicitly calls the LHCb/CMS/NIST proximities retrospective motivation. The completed controlled S004 reduction is October 7. Neither recent derivation can retroactively turn those values into predictions. Older mathematical identities remain valid; their physical applications and numeric scale assignments require separate provenance.
