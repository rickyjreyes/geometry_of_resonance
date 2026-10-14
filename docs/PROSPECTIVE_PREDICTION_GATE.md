# Target-blind prospective WCT prediction gate (#17)

**Current state: NOT PREREGISTERED / HOLDOUT NOT EXECUTED.** This is a protocol form and a blocking checklist, **not** a filled-in preregistration or a permission to inspect a holdout.

First, the theory must derive a measurable *dimensionless* observable from one specified operator, background, coupling/units, prepared domain and detector forward response. The [physical-domain audit](FOUNDATIONAL_CLOSURE_VERDICT.md) currently says `WCT_ACTIVE_DOMAIN_UNDERDETERMINED`; the [spectral mapping audit](SPECTRAL_OBSERVABLE_MAPPING.md) says `SPECTRAL_MEASUREMENT_MAP_NOT_DERIVED`. Do not select a familiar fitted peak and assert a newly derived target.

Freeze all fields below in a separately committed machine-readable specification, then record that commit/hash and archive the untouched holdout list/manifest by SHA256 **before** any holdout inspection:

| Field required at freeze | Acceptance |
|---|---|
| Theory and observable | Equations, derivation, assumptions, physical measurement/estimator forward map, units and a single numerical target or interval with a propagated predeclared uncertainty |
| Data identity | Immutable source releases and exact calibration/train/validation/holdout identifiers and content hashes; disjointness and prior-access declaration |
| Transform/exclusions | All coordinate/binnings, windows, cuts, nuisance families, blind tags, background alternatives, estimator, optimizer seeds and numerical tolerances |
| Null and multiplicity | Full null-generating procedure, predictive model check, nuisance selection, trial correction and empirical type-I error protocol |
| Power/injection | Injected effect sizes, acceptance/sensitivity and explicit FAIL/invalidation conditions before unblinding |
| Decision | Fixed effect sign/phase/frequency if relevant, statistic, numerical PASS/NULL/FAIL thresholds, uncertainty reporting and a one-time unblinding rule |
| Provenance | Hash of spec and analysis source, dependency lock, external independent reviewer/second signoff and immutable output storage |

If null/background alternatives predict held-out control data better than the preferred signal-producing background, the significance claim must be downgraded or invalidated under the frozen rule; do not choose significance-enhancing nuisance models retrospectively. If a frozen protocol is invalid, document the reason and create a **new** experiment; retain the original failed or invalidated record. A calibration or in-sample detector residual is not automatically a fundamental WCT prediction. The separate GWTC/CMS historical freezes, if cited, retain their original scopes and background limitations.

After passing the freeze gate, calibration and injection gate, execute the held-out computation **once** and publish effects, uncertainty, null comparison, failures and deviations. No claim of execution is made by this document.
