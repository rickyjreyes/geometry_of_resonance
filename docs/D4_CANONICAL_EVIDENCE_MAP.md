# Canonical D4 evidence handoff (2026-10-14)

**Status:** documentation crosswalk for [#29](https://github.com/rickyjreyes/geometry_of_resonance/issues/29). This is an additive revision of the living repository, not an amendment to the original dated Zenodo releases. Machine-readable source of truth for this handoff: [audit/d4_handoff/claims.json](../audit/d4_handoff/claims.json). A PASS in one row cannot promote any other row.

## Frozen identity and mathematical model

The designation **D4 is dihedral square symmetry in two spatial dimensions, not four spatial dimensions**. The independent experimental Hamiltonian rail on the periodic square (\mathbb R^2/(42\mathbb Z)^2) is

```math
i u_t=(\Delta+1)^2u-\beta|u|^2u+\gamma|u|^4u,\qquad
\beta=\gamma=4.0928844146232946.
```

The target mass is (2.54469689485712); the recorded frequency is approximately (-0.360518510946102). Frozen grid states: (N=112,140,168). The finite-grid quotient target is (G_N(t)=\|P\exp(t A_N)P\|_{2\to2}), where (P) removes only phase and two translations and the input space still contains nonsymmetric and mass-normal perturbations. Padé and split endpoint maps approximate this target; neither is itself the exact exponential or a continuous-time maximum. Consult the source report for exact source/state/vector/configuration hashes; do not reconstruct them from rounded values here.

- **Immutable release:** [d4-pade-numerical-closure-v1](https://github.com/rickyjreyes/wct-pde/releases/tag/d4-pade-numerical-closure-v1), 200 members / 199 inventoried payloads. Archive SHA256: `db9cfbbbb4a4b9d696bf4d0af6c2eb125e8795acfb8505491628eb0238f47b64`. Tag object `c7c6fa9f4f81419bf2a73ea561913d185b778cb1`, peeled commit `d5335dc0fab77e14b66feb7e1a943c2c31eeaa19`.
- **Scoped campaign report (immutable revision):** [wct-pde D4 comprehensive closure report @ 26296d5](https://github.com/rickyjreyes/wct-pde/blob/26296d5c40857c37adca6a1aae47930b3598c4ff/docs/D4_COMPREHENSIVE_CLOSURE_REPORT.md); associated [PR #26](https://github.com/rickyjreyes/wct-pde/pull/26), still open at handoff. [Windows production audit run 38441784132](https://github.com/rickyjreyes/wct-pde/actions/runs/38441784132), fresh [continuum CI run 38512669787](https://github.com/rickyjreyes/wct-pde/actions/runs/38512669787). The production audit checked recovered bytes/provenance without propagation.
- **Recovery evidence:** [indexed source and receipts at 26296d5](https://github.com/rickyjreyes/wct-pde/tree/26296d5c40857c37adca6a1aae47930b3598c4ff/docs/evidence/d4_production_recovery). Both earlier and revised reports remain historical evidence; do not replace the earlier eleven-gate FAIL with the scoped PASS.

## Separate claim-level decisions

| Claim | Classification as of handoff | Exactly what is, and is not, established |
|---|---|---|
| #19 production provenance / #20 release | **PASS / both closed** | Recovered source, state, solver, optimizer and reference arrays; verified release/checksums. Not a new propagation or physical theorem. |
| Padé convergence at (N=140,t=100) | **PASS, finite numerical protocol** | (G=1.3690254532021537) under frozen temporal/spatial/start checks; not an exact leading operator norm or all-time bound. |
| Historical aggregate | **FAIL, preserved** | Original eleven-gate aggregate FAIL remains in the record with disaggregated causes; later recovery does not retroactively pass it. |
| #21 split/Padé optimized equivalence | **OPEN** | Eight recovered-direction trajectories satisfy fine comparison on *those directions*. Historical split gain (2.3902336510107576) at dt=.00625 drops to (1.0000039150347957) at dt=.0015625, versus Padé (0.9999996472181292). Independently optimized refined norms remain unverified. |
| #22 refined envelope | **OPEN** | First t=5 sample (G=1.5136573217842118), residual (1.816591138012199\times10^{-11}); separate accuracy, other sample times, peak and local refinements incomplete. |
| #23 exhaustive leading gain | **OPEN** | No sharp exhaustive production enclosure near 1.369 or continuous-time supremum certificate. |
| #24 computed branch to continuum | **CONDITIONAL, open** | Proposed outward 128-bit Arb contraction application (reported (Y_0\le2.398622\times10^{-13},Z_0\le0.903773844573,r=10^{-8},L\le1401.108126)) passed its computational inequalities but still needs independent review of norms, tails, inverse and root linkage. |
| #25 all-time nonlinear stability of *this* root | **CONDITIONAL, open** | Proposed full-parity, mass-penalized constrained (H^2) application, reported lower bound (0.0008267789943812908), awaits independent certificate/proof review and prerequisite #24. No arbitrary-amplitude, whole-space or numerical-basin claim. |
| [wct-fem#8](https://github.com/rickyjreyes/wct-fem/issues/8) | **INDEPENDENT REPLICATION PENDING** | The independent same-state D4 FEM work item is open. Other FEM states (e.g. beta=gamma=16, mass 20, L=24) must not substitute for this replication, including contrary outcomes. |
| [wct-lean#21](https://github.com/rickyjreyes/wct-lean/issues/21) | **FORMAL APPLICATION PENDING** | Conditional theorem/contract work does not instantiate computed-D4 certificates until exact hypotheses are independently validated. |

## Bidirectional paper / claim map

| Canonical record | Supported within tested scope | Related but conditional / model-dependent | Not established by D4 evidence |
|---|---|---|---|
| [*The Geometry of Resonance* (original DOI 10.5281/zenodo.15644222)](https://doi.org/10.5281/zenodo.15644222) | A separate, explicitly specified confined nonlinear wave *example* possesses an audited periodic numerical stationary branch and finite-protocol propagation/convergence evidence. | General motivation for confinement, spectral localization, and a physical microdynamics-to-effective-model reduction. | Identity with the corrected covariant quotient action; a unique vacuum, self-bound whole-space particle, ghost-free Lorentzian sector, selected physical constants or a general WCT existence/stability theorem. |
| [*Phase–Flux Field (PFF)* (original DOI 10.5281/zenodo.17578766)](https://doi.org/10.5281/zenodo.17578766) | Evidence that a *distinct* finite-band experimental PDE can be audited reproducibly with declared symmetry and finite-grid diagnostics. | Compare the PFF finite-k and D4-to-continuum proposals only after an explicit operator, field, clock, domain, boundary and coefficient map. | PFF's **four-node D4 primer** equals the **D4 dihedral square symmetry** state; automatic proof of PFF continuum limit, flow equivalence or PFF-specific stability. |
| [DERIVATION_CLOSURE.md](../DERIVATION_CLOSURE.md) and [corrected action](../WCT_CORRECTED_VARIATIONAL_CLOSURE_DRAFT.md) | Their own scoped algebraic variational statements remain separate. | Cross-model comparison and possible future reduction. | #16 ghost freedom and #18 canonical finite-k selection cannot follow from periodic finite-time D4 gain. |

**Promotion gate:** Only reclassify a row after its *own* acceptance criteria pass at a cited immutable revision. Record numerical PASS, independent replication, conditional theorem, validated application, and physical interpretation separately. Subsequent null, contrary or failed replication outcomes must be appended, not erased. The original Zenodo PDFs, original dates and frozen WCT-PDE release are unchanged.

**Return path:** [geometry #29](https://github.com/rickyjreyes/geometry_of_resonance/issues/29) ↔ this map/JSON ↔ [PDE #19–#26](https://github.com/rickyjreyes/wct-pde/pull/26), [FEM #8](https://github.com/rickyjreyes/wct-fem/issues/8), [Lean #21](https://github.com/rickyjreyes/wct-lean/issues/21), and foundational [#16](https://github.com/rickyjreyes/geometry_of_resonance/issues/16)/[#18](https://github.com/rickyjreyes/geometry_of_resonance/issues/18).
