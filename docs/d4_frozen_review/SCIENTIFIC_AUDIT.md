# Frozen D4 scientific audit and remaining obligations

Date: 2026-10-29, America/Los_Angeles.
Reviewed main: `7264a760bee05fea89a1e42249b49f39bd078b78`.

**Outcome A is confirmed. Every interior split at the exact frozen mass is
strictly bound.** The resolution predates this review: PR #36 was merged on
2026-10-23 at `5a4c58a4206005ad9d46ee2951069186490b01f1`, with proof head
`31f322c94d2ef32b20acea741884514806025c85`. It is present in the reviewed main.
The request's PR #33–#35 baseline omits that subsequent result.

The [complete reconstruction and new estimate](PROOF.md) prove

\[
I_2(\mu)+I_2(m_{D4}-\mu)-I_2(m_{D4})
 >\min(\mu,m_{D4}-\mu)/400000,\quad 0<\mu<m_{D4}.
\]

This new rational corollary quantifies the existing all-split theorem. It is
not presented as the first resolution of a gap that remained open on main.

## Baseline and review findings

| Item | Finding |
|---|---|
| Frozen model | Same complex H²(R²) Hamiltonian, exact coefficient dyadics, squared integral mass and k=1 |
| PR #35 central splits | Valid; exact endpoints 2m−2m_* and 2m_*−m, with the complement historically open |
| PR #36 adjusted-sequence argument | Valid under the stated positive coefficients and two-dimensional hypotheses |
| Infimum strictness | A positive gap uniform along each minimizing sequence precedes the infimum limit |
| Frozen threshold | Exact rational 9−ρm²>0; m_B≈2.550345636351549 exceeds m≈2.544696894857120 |
| New continuum certificate | Four positive rational Bernstein coefficients prove the new bound on every component mass in [m/2,m] |
| Endpoint intervals | Covered by the same continuous-interval theorem, then symmetry; no numerical junction gap |
| Vanishing and spreading | Excluded for minimizing sequences by I₂(s)<0 at each fixed positive mass |
| Dichotomy at a positive split | Excluded by the strict deficit; separated trials do not furnish an equality |
| Endpoint limiting equality | The deficit tends to zero as a component tends to zero; this is compatible with strict binding at every interior split |
| Whole-space compactness and existence | Follow from coercivity, nonvanishing, Hilbert and separate Brezis–Lieb splittings, and the now-verified all-split bound |
| Whole-space orbital stability | Full minimizing set only; the proof separately verifies global H² flow, continuity and M/H conservation |
| Periodic D4 certificate | Preserved and rational basin verifier freshly replayed; not used to identify the whole-space minimizer |
| Canonical WCT action | Remains a distinct model; its ghost/UV obstruction is not repaired by this theorem |

## Verification actually executed

The [recorded local run](../../audit/d4_frozen_review/evidence/execution.json)
passes eight command groups:

1. Fresh exact certificate generation.
2. Independent reconstruction of the committed certificate.
3. The same certificate verification with Python optimization enabled.
4. Ten independent tests, including a second cubic interpolation calculation,
   hexadecimal normalization fixtures, tiny-component arithmetic at 10^-200,
   a rejected excessive gap, and corrupted-parameter/scope negative controls.
5. Ten symbolic identities reconstructed without importing inherited
   derivation code, including the Fourier integral and infimum-defect algebra.
6. The complete inherited 15-group public Phase V reproduction, including
   48 derivation tests, 20/14/38/31 inherited symbolic identities, 35 cross-paper
   identities, the existing adversarial campaigns, prospective tests and all
   historical integrity checks.
7. Fresh convention reconstruction from seven authenticated, pinned
   `wct-pde` files, with matching SHA256 and Git-blob hashes.
8. All five exact rational gates of the existing periodic basin verifier.

These are reproducibility and algebra checks, not a machine verification of
every functional-analytic argument. The Bernstein certificate proves its
scalar polynomial is positive on the full interval; the analytic connection
to constrained infima is supplied in PROOF.md. Existing Gaussian diagnostics
are retained as diagnostics and are not used as lower bounds on infima.

The original interval root/coercivity campaigns were not rerun. Their source
receipts and private input arrays remain unchanged. Public CI does not have
those private files: it reruns all public checks and validates the recorded
private-source replay. The seven private blobs are not added to this public
repository. Only their already-public pins and fresh verification receipts
are included.

Two environment setup attempts encountered a missing SymPy installation,
once before dependency installation and once after the runtime changed.
Neither reached mathematical verification. Installing the pinned dependencies
in a persistent task directory resolved this; the complete run above passed.
The older historical failure records have not been modified or removed.

## Remaining obligations, by exact scope

| Question | Status after this review |
|---|---|
| Binding for every 0<μ<m_D4 | **PROVED; no remaining split interval** |
| Quantitative bound in this review | **PROVED**, with exact continuum scalar certificate |
| Existence and compactness at m_D4 | **PROVED**, for the full complex admissible class |
| Stability of the full whole-space minimizing set | **PROVED**, using the separately verified dynamical hypotheses |
| All-positive-mass binding or sharp threshold | OPEN; m_B remains sufficient, not necessary |
| Uniqueness, symmetry or one-orbit nondegeneracy | OPEN |
| Whole-space minimizer equals the periodic D4 root | OPEN; no expanding-domain branch identification |
| Explicit whole-space orbital basin radius | OPEN; the periodic basin is a separate theorem |
| Particle rest mass, empirical spectrum or healthy canonical parent | NOT ESTABLISHED |
| External review or full proof-assistant verification of this report | NOT CLAIMED |

No tracking issue is created or closed. The review branch is intended for
review, not automatic merge. Historical manifests and results are preserved.
