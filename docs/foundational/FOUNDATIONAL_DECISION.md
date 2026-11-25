# Foundational closure decision

Date: 2026-10-20. **Primary classification: `FOUNDATIONAL_OBSTRUCTION_CONFIRMED`.**
Scope: the currently defined, untruncated corrected S004 scalar action with
fixed flat metric/foliation, nonzero regular field sector and strictly positive
curvature quadratic form. This is not a no-go theorem for every possible WCT
model or an assertion that the separate periodic results are invalid.

The key new result is an explicit high-frequency Cauchy obstruction: around
every allowed nonzero static homogeneous solution, its linearized modes grow
as exp(c_uv |k| tau) with c_uv>0. Therefore no fixed Sobolev four-jet solution
map is continuous at positive time, even allowing finite derivative loss.
A complete regular-sector constraint count also shows why its two extra real
degrees of freedom are not gauge constraints. These strengthen the known
opposite-residue and fixed-charge energy obstructions.

## Phase decisions and substantive work

| Phase | Primary classification / outcome | Acceptance obligations met | Unmet or blocked obligations |
|---|---|---|---|
| 0 | Baseline reconciled | Current default heads, active relevant PRs, issues, primary equations and proof scopes pinned; stale D4 handoff identified | No claim of exhaustive unpublished-source coverage |
| 1 | `FOUNDATIONAL_GHOST_OBSTRUCTION_CONFIRMED` | Independent full flat EL, quadratic reconstruction, both Hessians, first/second-class constraint count, canonical H/Q, residues, UV proof and controls | Healthy nonlinear/quantum EFT, dynamic g/n constraint analysis and physical cutoff not supplied |
| 2 | `MODEL_FAMILY_NOT_YET_UNIFIED` | Exact, conditional, asymptotic and non-equivalent arrows separated; one-component quadratic remainder replayed | Nonlinear S004→PFF/Hamiltonian/charge/domain map open |
| 3 | `EXISTENCE_OPEN` for literal S004 stationary localized states | Positive-spatial dilation obstruction incorporated; periodic #24/#25/#27 preserved and basin replayed | Healthy self-bound S004 state, arbitrary-V stationary theorem, whole-space compactness and physical boundary selection |
| 4 | `MASS_BRIDGE_OBSTRUCTED` | Conditional separated-field effective mass preserved; exact weighted-variance/amplitude-gradient equality theorem proved | Independent operator, length, normalization, particle assignment and inertia |
| 5 | Conditional research specification only | Distinct dimensions, norms, lifetime/refinement prerequisites specified | No common derived healthy model or dimension-dependent lifetime theorem |
| 6 | Auditable local verification and CI workflow | Symbolic, analytic, limiting-case and numerical-mode checks; pinned source/evidence hashes | No new PDE campaign, external peer review, Lean proof or physical observation claimed |

## New versus inherited results

**New here:** H2's explicit second-class constraint elimination/count; H4's
unbounded high-frequency growth and Sobolev discontinuity sequence; M1's exact
equality conditions and explicit nonconstant locked-curve counterexample for a
declared periodic operator; an
integrated current source/claim/acceptance matrix and corrected live D4 handoff.

**Independently reconstructed, already established:** S004's bulk equation,
full homogeneous quadratic Hessian and residues; eigen #16's fixed-charge
canonical minimum obstruction; the controlled single-real-component linear
EFT; corrected positive-spatial dilation obstruction and mass-scale ambiguity.
The D4 rational basin replay is a new execution of an existing verifier, not
a new existence theorem. Source hashes and revisions are in the
[source ledger](../../audit/foundational/source_ledger.json).

The [cross-model matrix](CROSS_MODEL_DERIVATION.md),
[localization audit](LOCALIZED_STATE_EXISTENCE.md),
[physical-health proof](GHOST_AND_CONSTRAINT_ANALYSIS.md), and
[mass audit](GEOMETRIC_MASS_CLOSURE.md) supply the complete definitions,
assumptions, proofs, failure conditions and reproduction commands. The
[claim matrix](../../audit/foundational/claim_matrix.json) records each status
without promoting one model's theorem to another.

## Research maturity: evidence first, no artificial score increase

The supplied WCT-RMI numbers are provisional judgments, not measured
probabilities or percentages of truth. No anchored scoring rubric mapping a
particular new theorem or obstruction to a numerical increment was supplied.
Accordingly all seven numbers are **carried forward provisionally**, with
their changed evidence assessment below. This is an updated scorecard, not
an invented quantitative re-rating. In particular the physical-health score
must not be interpreted as a 35% chance that S004 is healthy.

| Metric | Weight | Supplied baseline | Carried forward | Evidence now governing reconsideration |
|---|---:|---:|---:|---|
| Axiomatic consistency | 6% | 55 | 55 | Exact action/EL recovered; general h requires adjoint; multiple foundational actions still distinguished |
| Symmetry & physical health | 7% | 35 | 35 | Global phase is not local gauge; ghost and now explicit UV Sobolev obstruction; no health acceptance gate passed |
| Mathematical closure | 8% | 45 | 45 | New rigorous negative closure and scoped mass theorem; parent nonlinear well-posedness not achieved |
| Confinement & stability | 10% | 55 | 55 | Genuine periodic root/orbital basin survives; positive-spatial self-binding fails; no physical transfer |
| Dimensional selection | 4% | 35 | 35 | Sobolev threshold preserved; no new matched lifetime theorem |
| Cross-model derivation | 8% | 30 | 30 | Controlled quadratic arrow verified; full nonlinear/dissipative/charge links still open |
| Geometric mass & inertia | 8% | 40 | 40 | Exact conditional identities clarified; physical scale/energy/inertia remain unselected |

These weights total 51%, so their weighted subtotal is not a full WCT-RMI
score. Better documentation, more tests and a negative theorem do not by
themselves increase positive physical maturity. A future numerical re-rating
needs predeclared anchored criteria, with the negative health result respected.

## Executed verification and material limits

The local [execution receipt](../../audit/foundational/execution.json) records
commands, outcomes, versions and hashes. At this revision:

- 31 new symbolic identities/dimensional checks passed, including direct EL,
  real quadratic symbols, Legendre inversion, charge-preserving variation and
  principal-symbol factorization.
- 13 focused unittest cases passed, comprising eight existing cases and five
  new UV/remainder/boundary/locked-curve tests.
- All 35 preexisting cross-paper symbolic checks passed in a temporary copy;
  the historical checked-in output was not overwritten.
- Eight independently solved time-domain companion polynomials agree with
  exact poles; their k^-2 asymptotic approach and the B=0 Lorentzian control
  pass. These are algebraic mode checks, not time-propagated PDE experiments.
- Four arbitrary-precision low-branch Taylor comparisons satisfy the proven
  remainder bound. The theorem supplies the interval-wide error control.
- Nineteen supporting files match remote Git blob hashes; the exact-rational
  D4 basin replay passes with pinned root/coercivity inputs and all five gates.

The full root/coercivity interval campaigns and frozen nine-attempt eigen
search were **not rerun**. Their immutable receipts and limitations were read.
  The eigen read-only verifier additionally passed its frozen hash, nine-attempt,
  profile-provenance and failed-gate checks.
No numerical values from a command that was merely proposed are reported as
new experiments. Hosted CI status is recorded separately in the PR; local
PASS is not substituted for a hosted result.

## Remaining blockers, ranked by scientific impact

1. **Healthy foundational dynamics:** S004's nondegenerate extra modes and
   UV principal symbol require a justified different completion or a controlled
   physical sector. A numerical non-excitation claim cannot meet this gate.
   [#16](https://github.com/rickyjreyes/geometry_of_resonance/issues/16).
2. **Nonlinear parent-to-effective map:** actual Q/H, gapless phase, cutoff,
   nonlinear vertices, field/time/domain/unit correspondence and time-uniform
   error control; dissipation needs its own physical mechanism.
   [#18](https://github.com/rickyjreyes/geometry_of_resonance/issues/18).
3. **Self-consistent whole-space binding in a surviving healthy model:**
   exceed the correct vanishing threshold, exclude splitting, then prove
   stability. Positive scalar and corrected positive-spatial routes already
   have scoped negative theorems. [eigen #7](https://github.com/rickyjreyes/eigen/issues/7),
   [eigen #12](https://github.com/rickyjreyes/eigen/issues/12).
4. **Physical scale, action normalization and inertial observable:**
   [#12](https://github.com/rickyjreyes/geometry_of_resonance/issues/12),
   [#13](https://github.com/rickyjreyes/geometry_of_resonance/issues/13),
   [#14](https://github.com/rickyjreyes/geometry_of_resonance/issues/14),
   [#27](https://github.com/rickyjreyes/geometry_of_resonance/issues/27).
5. **Tensor/gauge completion:** dynamical fields, constraints, normalization
   and generic response; independent g or a gauge identity is not emergence.
   [#11](https://github.com/rickyjreyes/geometry_of_resonance/issues/11),
   [#15](https://github.com/rickyjreyes/geometry_of_resonance/issues/15).
6. **Dimension-dependent lifetimes after a common model is fixed:**
   [dimension #3](https://github.com/rickyjreyes/dimension/issues/3),
   [#5](https://github.com/rickyjreyes/dimension/issues/5),
   [#7](https://github.com/rickyjreyes/dimension/issues/7). No broad scan warranted yet.

The independent numerical tasks wct-pde #21–23 remain valuable within that
model but cannot repair the failed foundational arrow. Issue states are not
closed by this audit. The dedicated branch is
`agent/foundational-closure-2026-10-20`; its PR is for review and must remain
unmerged unless separately authorized.
