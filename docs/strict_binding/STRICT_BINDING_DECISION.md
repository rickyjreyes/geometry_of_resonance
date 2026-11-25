# Phase IV decision: strict binding and whole-space compactness

Date: 2026-10-22, America/Los_Angeles.
**Primary classification: STRICT_BINDING_RESTRICTED.**

For the unchanged complex cubic–quintic Hamiltonian on R², strict binding
is proved for every split at every mass

    0 < m <= m_* = 4 beta k_*² /
                       [gamma (k_*⁴+3 beta²/(16 gamma))].

At the frozen D4 coefficients this is

    m_* = 72057594037927936/31838933052907969
        = 2.2631912293727647... .

The all-positive-mass assertion is not established or refuted. The pinned D4
mass is 2.54469689485712..., which lies above this sufficient bound. Its
whole-space existence and stability therefore remain OPEN.

## What is now established

| Question | Result | Exact scope |
|---|---|---|
| Frozen functional and norm | VERIFIED reconstruction | H=||(Delta+k_*²)u||²-beta||u||_4⁴/2+gamma||u||_6⁶/3; M=||u||_2²; full complex H²(R²) |
| Coercivity, finite infima, continuity | PROVED | All beta,gamma,k_*>0 and all m>0 |
| Negative infimum | PROVED, PR #34 result independently reconstructed | All m>0 in two dimensions, using anisotropic carrier Gaussians |
| Strict decrease of I(b)/b relative to I(a)/a | PROVED | 0<a<b and a+b<=2m_* |
| Strict binding | PROVED within restriction | m+max(mu,m-mu)<=2m_*; in particular all splits for 0<m<=m_* |
| Precompactness modulo translations | PROVED | Every minimizing sequence, 0<m<=m_* |
| Whole-space constrained minimizer | PROVED | Nonempty full complex minimizing set for 0<m<=m_* |
| Euler–Lagrange equation and regularity | PROVED | Smooth localized profiles with a real negative multiplier in the same range |
| Orbital stability | PROVED | Full minimizing set in H², including small mass-changing complex perturbations, 0<m<=m_* |
| All splits at m_D4 | OPEN | Central interval proved; small-component intervals remain open |
| Whole-space existence/stability at m_D4 | OPEN | No transfer from its periodic certificate |
| Universal assertion for all m>0 | OPEN | No frozen-coefficient counterexample established |

“Proved” refers to the analytic arguments in these reports, not Lean
certification or external peer review. A finite-dimensional trial search is
reported separately and does not replace any infimum comparison.

The new estimate is an explicit Fourier Cauchy–Schwarz bound,

    ||u||_infinity² <= (3/(16k_*²))
                        [||(Delta+k_*²)u||²+k_*⁴||u||²].

Combined with the energy lower bound, it controls the sextic/quartic ratio
on minimizing sequences. This supplies a uniform strict amplitude-scaling
gap. The proof handles the equality boundary a+b=2m_* by retaining the
strictly negative limiting energy, and does not assume minimizers exist.

## What the actual D4 norm gains

At m=m_D4 the result excludes all splits with

    2m_D4-2m_* <= mu <= 2m_*-m_D4,

approximately [0.5630113309687108, 1.9816855638884092]. The exact rational
endpoints, not rounded decimals, define the interval. By symmetry the
remaining obligation is

    0 < mu < 2m_D4-2m_*.

This narrows the splitting problem but does not satisfy its all-splits
quantifier. The highest-value next task at the frozen norm is to prove or
refute binding on that remaining interval, including control as mu->0.

## Three mathematical approaches and adversarial outcome

Amplitude scaling succeeds in the proved range. Concentration compactness
and profile extraction then supply actual whole-space existence and
minimizing-set stability. Direct separated/overlapping comparisons establish
weak additivity and expose sign-changing overlap terms, but do not settle
the uncovered splits.

Explicit negative-energy fields refute two unrestricted scaling shortcuts.
Degenerate coefficient limits, two-profile sequences, very small component
masses, large masses, and negative nonattainment were examined. The actual
functional has no established counterexample to strict binding. The bound
m_* is not a discovered critical mass. All diagnostic comparisons of unknown
infima remain marked as upper-bound comparisons.

## Reproducibility and preserved evidence

The dedicated branch starts at frozen PR #34 head
`dff2463e4dae6c1ee4fe0c4ebc6125bf90c9f49e`, stacked on PR #33 head
`0385386e3daff64b44e5c5062f8d578ea68d146d`. All existing tracked files are
preserved. New claims, sources, commands, receipts and hashes are additive in
`audit/strict_binding`; prior claim matrices keep their historical statuses.

The new work has 14 symbolic identities, exact-rational threshold/split
checks, 12 focused tests, and the explicitly bounded diagnostic campaign.
The execution manifest records fresh inherited regression runs and the D4
exact-rational replay. Hosted outcomes are reported on the new draft PR at
its actual published head, not inferred from local checks.

The D4 source repository is private. Full local reproduction requires the
five authorized, hash-pinned inputs and executes all five rational gates.
Public CI reruns the public mathematical checks and validates that recorded
full-local evidence; it does not execute the private-source D4 replay.
The first hosted download failure and the corrected scope are preserved in
the reproducibility record. A public-CI receipt cannot replace the canonical
full-local execution manifest.

The D4 rational replay uses the pinned candidate, root and eight-sector
coercivity receipts at wct-pde `051b6eca10c46b8e038993d60cabb20715194a20`.
Its original interval root/coercivity campaigns are inherited evidence and
are not claimed as newly executed. Its fixed-torus basin, coefficients,
normalization and numerical thresholds are unchanged.

No issue closure, merge, physical particle spectrum, empirical mass fit,
healthy-parent derivation, or whole-space identification of the D4 root is
part of this result. MODEL_FAMILY_NOT_YET_UNIFIED and MASS_BRIDGE_OBSTRUCTED
remain the physical interpretation boundaries from PR #34.

Read [the proof and functional reconstruction](SUBADDITIVITY_PROOF_OR_COUNTEREXAMPLE.md),
[compactness and stability](WHOLE_SPACE_COMPACTNESS.md),
[adversarial analysis](ADVERSARIAL_SPLITTING_TESTS.md),
[claim matrix](../../audit/strict_binding/claim_matrix.json),
[source ledger](../../audit/strict_binding/source_ledger.json),
[execution manifest](../../audit/strict_binding/execution.json), and
[artifact hashes](../../audit/strict_binding/manifest.json).
