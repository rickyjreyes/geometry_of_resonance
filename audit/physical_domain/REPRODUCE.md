# Reproduce the foundational audit

From the root of the audit branch, using Python 3.12:

```bash
python -m venv /tmp/wct-physical-domain-venv
/tmp/wct-physical-domain-venv/bin/python -m pip install -r audit/physical_domain/requirements.txt
/tmp/wct-physical-domain-venv/bin/python audit/physical_domain/verify_symbolic.py
/tmp/wct-physical-domain-venv/bin/python audit/physical_domain/validate_records.py
```

On Windows, use the corresponding virtual environment `Scripts/python.exe`. Neither script downloads data, runs a PDE solver, evaluates a collider spectrum, or reads a holdout. The symbolic script writes `symbolic_results.json`; the record validator writes `VALIDATION.json`. Run without Python's `-O` option: assertions are intentional audit checks.

The symbolic checks cover polar decomposition, corrected weight differentiation and amplitude order, a legacy denominator witness, PFF compatibility, finite-band symbols, exact Hamiltonian plane waves, fixed-loop locking and sector costs, radius degeneracy, covariance/coordinate identities, an exact estimator counterexample, Z3 norms, operator gain, Cartan eigenvalues and Willmore scale degeneracy. Written proofs provide the function-space and minimization hypotheses. A zero residual is not a proof of all-time stability, a physical scale, or an observable correspondence.

The committed reference was produced with Python 3.12.14, SymPy 1.14.0 and mpmath 1.3.0. A different Python patch version changes the metadata string even if all mathematics agrees. The validator checks the reference manifest before replacing its validation record; reruns that change recorded metadata should be reviewed as differences rather than silently changing the manifest.

Read `SOURCE_LEDGER.json` for all 48 GitHub files at immutable commits, their verified Git blob hashes and SHA256 hashes, and all eight manuscript identities. Seven manuscript extractions were read in full; the 313-page foundational PDF was read in the specified relevant windows. Manuscript hashes identify the returned extracted-content arrays, not the original binary files. No original-binary hash or inaccessible solver-state provenance is invented. The paper sources remain the user's identified manuscripts; this audit does not publish duplicate full manuscripts.

`prior_vs_new.json` distinguishes published results from this audit's deductions and conditional hypotheses. `dependency_graph.json`, `assumptions.json`, `derivation_status.json`, `prediction.json` and `decision.json` preserve the unresolved classifications. In particular, every required prediction field stays null.

`BASELINE_BLOBS.json` records every baseline blob and mode. `ARTIFACT_SHA256.json` hashes every new artifact except itself. It is the publication reference, not a self-updating acceptance mechanism. Publication verification compares the remote tree against both the full baseline and exact bytes of every addition, and checks the branch head after a lease-protected fast-forward. The final commit hash is reported externally rather than embedded in the commit itself.

The frozen ATLAS repository, scientific gates, completion scores and unresolved issue states are not modified. This audit's completion is distinct from foundational physical closure.
