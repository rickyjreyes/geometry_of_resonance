# Adversarial splitting analysis

Date: 2026-10-23. Analytical classification and the actual finite campaigns
are kept separate. No trial-energy comparison is called a constrained-infimum
comparison. The new receipts are in `audit/d4_binding`.

| Candidate or attack | Kind | Outcome and limitation |
|---|---|---|
| Separated compact near-minimizers | Analytical | Exact mass/energy addition proves ordinary subadditivity only |
| Dichotomizing minimizing sequence at a fixed interior split | Analytical | Would contradict the proved strict inequality at m_D4; excluded |
| Widely separated Gaussian packets | Numerical | Energy approaches the sum of single trial energies; no matching lower bound on infima |
| Overlapping complex packets | Numerical and algebraic | Interaction defects have both signs; overlap alone supplies no universal strict comparison |
| Highly asymmetric split, including mu/m=10^-18 | Analytical with certified scalar samples | F(mu)>0 on the entire continuous half-interval; finite samples corroborate constants |
| Dilute components and endpoint near-additivity | Analytical | Delta tends to zero at an endpoint; this does not establish interior equality |
| Broad vanishing profiles | Analytical | Energy tends to zero while I(m)<0 for every fixed m>0; not minimizing |
| Universal amplitude decrease for all fields | REFUTED, explicit trial | A narrow Gaussian at m_D4 has positive amplitude slope; this does not refute the adjusted-sequence theorem |
| Negative-energy amplitude shortcut | REFUTED, inherited trial | PR #35's large-mass negative-energy counterexample is preserved and rerun |
| Original Fourier weight estimate | Analytical limitation | Its sufficient mass threshold is below m_D4 |
| Tuning the original Fourier weight | Numerical exploration | Best found threshold remains below m_D4; global optimality of the scan is not certified |
| Positive sextic term defeating V3 | Analytical | Controlled on adjusted near-minimizers by S/Q<=sigma*M/4 within the stated range |
| Loss of strictness in the infimum limit | Analytical | V3 supplies a positive sequence-uniform lower defect at each fixed pair |
| Failure of the endpoint chemical-potential gap | Analytical | lambda0=0 and the frozen limsup energy slope has an explicit negative upper bound |
| Competing stationary/minimizing branches | Unresolved branch identification | No stationary PDE branch campaign was executed; the variational theorem covers all minimizing sequences |
| Negative but unattained infimum at m_D4 | Analytical | Excluded by all-split compactness; no claim for every larger mass |
| A sharp critical mass | OPEN | m_B is sufficient, not asserted necessary; failure beyond it does not prove equality |
| Degenerate beta=0 | Analytical, outside hypotheses | H>=0, spreading annular Fourier trials give I=0, and all splits are equality; positivity of beta is essential |
| Degenerate gamma=0 or k=0 | Outside theorem hypotheses | Present proof does not apply; no conclusion transferred by an unjustified limit |

The new deterministic campaign uses 240 anisotropic complex chirped Gaussians
over six masses, including m_D4, m_B and mass 4. Every profile is dilated
exactly using V1. The code checks energy decrease, the scalar virial residual
and the new ratio estimate. A negative adjusted amplitude slope is required
only within the proved sufficient range. Independent quadrature reintegrates
a chirped dilated profile to check the transformation.

There are 59 normalized two-profile trials: mass fractions 1/2, 1/10 and
10^-6, five separations and four relative phases. The exactly cancelling
zero profile is excluded. The inherited Gaussian-pair evaluator is explicitly
identified and independently cross-checked by new coordinate quadrature.
All records mark `certifies_infimum_comparison=false`.

Twenty certified scalar samples of F include fractions 10^-1 through 10^-18,
the exact old interval junction and m/2. Three independent radial quadratures
check the Fourier constant. The proof of positivity on the whole interval is
analytical, not sampling. Exact rational arithmetic and outward enclosures
decide the D4 threshold and endpoint signs.

Twelve new tests include a mass 1% above m_D4 that fails the sufficient gate,
the complex chirp transformation, independent pair integration, endpoint
coverage, interval zero-division rejection and the unadjusted shortcut
counterexample. This higher-mass gate failure is not a binding counterexample.

The workspace restarted before the earlier uncommitted package could be
published. The proof was restored from the retained derivation and the
campaigns were freshly rerun. The committed receipts describe these restored
files; unavailable earlier receipts are not reconstructed or counted as
published evidence. All existing PR #33–#35 files, including failed historical
attempts, are preserved unchanged.

No physical-spectrum export, PDE propagation, original D4 interval root
campaign, external mathematical review or proof-assistant formalization is
claimed. All five original rational-replay gates are separately rerun against
freshly extracted pinned sources.
