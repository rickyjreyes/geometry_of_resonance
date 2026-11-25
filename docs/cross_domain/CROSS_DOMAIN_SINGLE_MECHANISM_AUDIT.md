# Cross-domain single-mechanism audit

**Verdict: NUMERICAL_CONVERGENCE_ONLY.** The controlled S004 finite spatial scale remains derived. No low-parameter physical chain was recovered that independently selects the four systems' observable coordinates, domains and sectors and predicts their frozen structures.

This work follows `FINITE_SCALE_DERIVED_BUT_CLOSURE_INCOMPLETE` at commit `caba0a5d034b1a347e60ffcd36dd7302f2c3e10d`. The analysis protocol was published separately at `b75989b60eeb9874aae1f4d2562cf35447e3862f` on `agent/wct-cross-domain-single-mechanism` before this audit's new numerical comparisons. Historical values were already known, so the work is retrospective. Audit work spans October 7–8, 2026 UTC.

**A. Frozen values and their actual meanings**

| Domain | Exact primary value | Meaning |
|---|---:|---|
| GWTC | 9.602325620315224 | Angular frequency in ln(source-frame chirp mass/M_sun), selected on training data |
| CMS | 7.025825825825827 | Frozen angular frequency in ln(dimuon mass/GeV) |
| NIST Fe II | 31.3265306122449 | Canonical line-density angular frequency in ln(wavenumber/(cm^-1)) |
| LHCb | 1.2282873602374818 | Zero-intercept scale fitted between selected regional winding triplets; **not a DSI period** |

GWTC's KDE variant 9.794117647058822 and imposed future/V4 target 9.7 remain separate. Co II is a within-domain control: 30.24489795918367 at 120/160 bins and 30.213085234093636 at 200. LHCb k1=7.61054 is a separately documented dominant low-frequency reference fixed in later fits, not the primary a target. All exact provenance, units, uncertainty status and freeze chronology are in [EMPIRICAL_TARGET_LEDGER.md](EMPIRICAL_TARGET_LEDGER.md).

**B. Which quantities are legitimately comparable**

The first three primary quantities are angular log-frequency magnitudes. Unit normalization changes phase, not frequency. For the actual LHCb q2=m_mumu² coordinate, frequency expressed in log mass is twice frequency expressed in log q2. Thus a separately labeled four-frequency reference panel is possible with k1. Equality across those distinct physical responses remains a hypothesis. No derived map converts the primary regional slope a into a DSI period, radial eigenvalue or log-frequency.

**C. Candidate single mechanism**

The intended candidate is shared S004 action → controlled lower amplitude branch → stable physical domain/state → independently selected n → observable log phase. Only the initial finite-spatial-scale reduction is presently established. The conditional scale-winding theorem nu L=2pi n is exact when its physical premises hold; those premises are not supplied by an analysis-window cycle count. U0/U1 are useful equality stress tests, U2 needs independent sectors and widths, and U3/U4 lack a completed forward map.

**D. Shared parameters**

U0/U1 fit one common continuous scale, no per-domain multiplier and no integer labels. U2 with common physical width would have one continuous parameter plus four independently supplied sectors; these assignments are absent. S004 has five named bare couplings (kappa,gamma,theta,epsilon,alpha) plus the coefficients of a potential that has not been fixed. Its nominal four effective homogeneous branch parameters K,G,T,m² are not automatically common across different backgrounds. The full effective count is therefore unresolved, not zero. Four freely fitted conversion lengths alone would saturate four frequency targets. [degrees_of_freedom.json](../../audit/cross_domain/degrees_of_freedom.json) distinguishes counts, independent inputs and output rank.

**E. Required system-specific inputs**

Each system needs an independently specified background/preparation, physical boundary/domain rule, sector selector and observable response. These must connect a spatial field scale respectively to a merger mass population, an inclusive dimuon residual, an atomic transition-line density and an exclusive candidate/sideband spectrum. Catalog extrema, bin centers, resonance vetoes and fitted triplets do not provide these physical inputs.

**F. Sector/winding assignments**

None of the four sectors is independently determined. GWTC's fitted cycles are about 6.79, CMS's reported retained-center cycles about 4.57, Fe II's about 10.74 and LHCb k1's about 5.79. Rounding these is not sector selection. The primary admissible integer range is empty; the integer searches below are negative controls only. Existing locked LHCb combs include both integer and noninteger bases chosen from the same studied data.

**G. GWTC prediction and residual**

The S004 physical prediction is unavailable. Supplementary U1 predicts 14.9629898903793 versus 9.602325620315224: signed error +5.36066427006407, fractional error +55.826729%.

**H. CMS prediction and residual**

The S004 physical prediction is unavailable. Supplementary U1 predicts 16.6052237696928 versus 7.025825825825827: signed error +9.57939794386694, fractional error +136.345508%.

**I. NIST prediction and residual**

The S004 physical prediction is unavailable. Supplementary U1 predicts 10.0888057204914 versus 31.3265306122449: signed error -21.2377248917535, fractional error -67.794692%.

**J. LHC/LHCb prediction and residual**

The primary slope a has no derived four-domain prediction. For the distinct k1 reference, supplementary U1 predicts 6.41650751150959 versus 7.61054: signed error -1.19403248849041, fractional error -15.689195%.

No primary four-domain S004 prediction exists to normalize by an error bar. To give the valid equality hypotheses their strongest direct test, the separately declared reference-frequency panel uses GWTC V1, CMS, Fe II, and LHCb k1. U1 maps these to common mass/energy-log coordinates, fits the geometric mean of three training values, freezes it, and reverses the fixed coordinate map to predict the fourth:

| Held-out domain | Primary physical prediction | Supplementary U1 reference prediction | Reference fractional error |
|---|---|---:|---:|
| G. GWTC | Unavailable: physical map/sector unresolved | 14.9629898903793 | +55.826729% |
| H. CMS | Unavailable: physical map/sector unresolved | 16.6052237696928 | +136.345508% |
| I. NIST Fe II | Unavailable: physical map/sector unresolved | 10.0888057204914 | -67.794692% |
| J. LHCb | Unavailable: physical map/sector unresolved; a has no frequency map | 6.41650751150959 (k1 frequency) | -15.689195% |


For J, the native observed reference is k1=7.61054; the prediction in that row is not a prediction of a=1.2282873602374818. Signed and absolute errors, all full-fit predictions and full precision are recorded in [cross_validation_results.json](../../audit/cross_domain/cross_validation_results.json). There are no calibrated frequency sigmas in the cited target records, so no z residuals are invented. The paper's approximate heavy-tailed bootstrap scale for a is retained separately and does not fix this type mismatch.

**K. Leave-one-domain-out results**

All four folds were computed for every U0/U1 source-choice stress panel: 3 GWTC variants × 2 atomic targets × 3 LHCb references/optima × 2 models = **36 fits, 144 predictions**. The primary U0 native-frequency errors are +23.6798%, +87.5876%, -74.4386%, +68.6216%. Every alternate panel is disclosed in [LEAVE_ONE_DOMAIN_OUT.md](LEAVE_ONE_DOMAIN_OUT.md), with no best-source substitution. All U0–U4 primary-anchor fold obstructions are also recorded explicitly. U2–U4 and the partial geometric analogy have no eligible physical four-domain prediction; unknown held-out inputs are not filled by observing the held-out value.

**L. Look-elsewhere / transformation freedom**

The negative control searches integers 1..12, sensitivities 1..6 and 1..24, power-coordinate menus, harmonic/subharmonic direction and both declared coordinate panels. It repeats the entire selection on 10000 synthetic independent quartets per support. Optimized observed RMS log residual is 0.00999620; 7319/10000 log-uniform[1,100] quartets and 9918/10000 on [5,50] fit at least as well. Add-one fractions are 0.7319268073 and 0.9918008199. These arbitrary-support algorithm diagnostics are **not empirical p-values** and do not calibrate historical pipeline/experiment selection. Nominal search entries total 478,349,768, heavily redundant; the whole-selection Monte Carlo handles that dependence within the diagnostic.

**M. Which domains support the common mechanism**

No domain currently supplies causal support for the complete shared mechanism through an independently derived forward map. CMS and GWTC contain documented fixed-mode tests within their pipelines; Fe II/Co II contain recurring line-density structure; LHCb contains structured candidate and sideband responses. These motivate the question. The attractive CMS/Fe period ratio 2.001161449490961 and near-sqrt(3/2) comparisons remain reproducible **numerical analogies**. They do not provide a GWTC assignment or turn the regional LHCb slope into the same operational quantity.

**N. Which domain breaks the tested route**

LHCb a blocks the type identification in the primary four-number convergence claim. NIST strongly disrupts simple frequency equality; CMS has the largest fractional U1 held-out error. Separately, the previously frozen CMS→LHCb frequency/phase/positive-sign template failed with A_hat=0, q=0 and p=1. Its post-unblinding free-phase response was positive at the imported frequency but boundary-limited. That narrow template failure is informative; it does not prove absence of the frequency or falsify every possible S004 completion. No failing domain is removed.

**O. Exact verdict**

`NUMERICAL_CONVERGENCE_ONLY`

There is no supported four-domain physical mechanism and no proof that multiple mechanisms are required. The newer action-level mathematical progress is preserved. The reason for this verdict is the unresolved typed physical bridge and sector assignment, reinforced by poor low-flexibility held-out predictions and substantial search freedom—not merely that four raw numbers differ.

**P. Exactly one next derivation**

Derive the **S004-to-observable scale-winding map** for a fully specified stable localized branch, fixing the physical endpoints, observable coordinate and integer sector from independent inputs before using any of the four empirical target values. The output must be a definite log-frequency or independently constrained finite prediction set with approximation error. A length, endpoint, harmonic or geometric norm fitted to an observed target does not close this derivation.

## Deliverable index

| Record | File |
|---|---|
| Target definitions, provenance and variants | [EMPIRICAL_TARGET_LEDGER.md](EMPIRICAL_TARGET_LEDGER.md), [CROSS_DOMAIN_TARGET_LEDGER.md](CROSS_DOMAIN_TARGET_LEDGER.md), [cross_domain_targets.json](../../audit/cross_domain/cross_domain_targets.json) |
| Existing and proposed transformations | [THEORY_TRANSFORMATION_LEDGER.md](THEORY_TRANSFORMATION_LEDGER.md) |
| Invariants and physical identifiability | [UNIVERSAL_INVARIANT_SEARCH.md](UNIVERSAL_INVARIANT_SEARCH.md) |
| Independent sector/domain inputs | [SECTOR_ASSIGNMENT_AUDIT.md](SECTOR_ASSIGNMENT_AUDIT.md) |
| Prediction folds and all sensitivities | [LEAVE_ONE_DOMAIN_OUT.md](LEAVE_ONE_DOMAIN_OUT.md), [cross_validation_results.json](../../audit/cross_domain/cross_validation_results.json) |
| Search-aware null diagnostics | [NULL_MODEL_AND_TRIALS_AUDIT.md](NULL_MODEL_AND_TRIALS_AUDIT.md), [null_diagnostics.json](../../audit/cross_domain/null_diagnostics.json) |
| Candidate and parameter records | [candidate_unified_mechanisms.json](../../audit/cross_domain/candidate_unified_mechanisms.json), [degrees_of_freedom.json](../../audit/cross_domain/degrees_of_freedom.json) |
| Failures and final decision | [FAILURE_MODES.md](FAILURE_MODES.md), [DECISION_CROSS_DOMAIN.md](DECISION_CROSS_DOMAIN.md) |
| Source pins, chronology and reproducibility | [SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md), [REPRODUCE.md](REPRODUCE.md) |

The workflow uses only preserved summaries and methodological sources. No source empirical analysis was rerun or modified. The ATLAS holdout remains unopened. Every pre-existing blob on the protocol branch is subject to an exact preservation check before publication.
