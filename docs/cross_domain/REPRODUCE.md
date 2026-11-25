# Reproduce the cross-domain audit

The code performs summary arithmetic and synthetic negative controls. It does not ingest events, scan spectra, download experimental data, or touch the ATLAS holdout.

From the repository root, with Python, NumPy 2.3.5 and mpmath 1.3.0 installed:

```bash
python audit/cross_domain/run_audit.py
python audit/cross_domain/verify_audit.py
```

`--skip-nulls` recomputes the deterministic predictions and derived quantities without rerunning the already preserved synthetic diagnostic. The default full run uses two sets of 10000 four-tuples, PCG64 seed 20261007. The same coordinate/label/search family is evaluated on each observed and null quartet, including the omnibus selection. No source-choice sensitivity result is chosen to replace the primary reference panel.

Inputs are the exact string-valued numerical records in `cross_domain_targets.json`; observed inputs are never rounded before calculations. mpmath uses 80 digits for transformations, geometric-mean fits and residuals. This preserves numerical reproducibility, not extra experimental precision. Synthetic diagnostic optimization uses NumPy float64 from those original values. Software versions are recorded in `execution_environment.json`.

The global optimizer minimizes sum_i min_j (x_ij-mu)^2. At fixed mu each domain selects its nearest candidate x_ij. The combined one-dimensional nearest-choice boundaries partition the real line; the algorithm sweeps every resulting assignment and evaluates its best shared mean. A global minimizer must appear among these assignments. Direct residual recomputation avoids cancellation in cumulative variance. `verify_audit.py` compares against exhaustive enumerations on small caps, checks exact synthetic harmonics, nesting/reciprocal duplicates, every held-out prediction, and an explicit held-out-value perturbation that must not change its prediction.

## Source-byte verification

The published source ledger supplies immutable URLs and both hashes for all 62 cited repository files. If those files are mirrored under a directory with layout `SOURCE_DIR/GWTC/...`, `SOURCE_DIR/wct-cms/...`, `SOURCE_DIR/NIST/...`, and `SOURCE_DIR/LHC/...`, run:

```bash
python audit/cross_domain/verify_audit.py --source-dir SOURCE_DIR
```

This verifies Git blob SHA1 and SHA256 and re-extracts each exact source JSON pointer or literal. The audit performed this check against its retrieved source bytes. Source files are not silently fetched by the verification script. The original paper retrieval hashes identify returned extracted text, not PDF/TeX bytes or a visual-layout review.

## Additive branch preservation

`BASELINE_MANIFEST.json` records all 167 protocol-branch blobs, based on commit b75989b60eeb9874aae1f4d2562cf35447e3862f. Given a complete recursive Git tree JSON for the final candidate tree:

```bash
python audit/cross_domain/verify_audit.py --source-dir SOURCE_DIR --published-tree TREE.json
```

Every existing path must retain its original blob SHA. This check includes the previously completed derivation audit and the protocol. New files are confined to `docs/cross_domain/` and `audit/cross_domain/`. Publication updates only the dedicated branch, using the expected previous commit as a lease. No empirical repository, old target artifact, or protected ATLAS result is edited.

## Outputs and scope

`cross_validation_results.json` contains 20 blocked primary-anchor folds plus 36 reference-frequency fits/144 held-out predictions. `derived_quantities.json` contains exact-source period re-expressions, cycle counts and published partial geometric analogies. `null_diagnostics.json` contains all 96 family/support rows and both omnibus results. `verification.json` records the checks actually run. `ARTIFACT_MANIFEST.json` records final deliverable hashes and intentionally excludes itself.

The Markdown reports are the reviewed interpretation of these frozen outputs, not additional empirical computations. The decision is `NUMERICAL_CONVERGENCE_ONLY`; missing physical predictions remain null. The absence of legitimate primary frequency error bars is not repaired with bin spread, grid spacing, a p-value or a prospective acceptance band.
