# Phase V reproduction and evidence scope

Entry point: [D4_STRICT_BINDING_DECISION.md](../../docs/d4_binding/D4_STRICT_BINDING_DECISION.md).
Frozen base: PR #35 at `65499cabca7e177286b6541fa1a5a00d51a50ba5`.
All changes are additions; historical PR #33–#35 evidence is unchanged.

## Public reproduction

```sh
python -m pip install sympy==1.14.0 mpmath==1.3.0 numpy==2.3.5 scipy==1.16.2
python audit/d4_binding/validate_records.py
python audit/d4_binding/reproduce.py --public-ci --output-dir /tmp/d4-binding-public-fresh
```

Use Python 3.12 and a fresh output directory. Public mode runs 15 groups:
14 mathematical/regression groups and validation of the canonical recorded
full-local evidence. It does **not** execute the private-source D4 verifier.
Its receipt explicitly marks `D4_rational_replay_executed=false`. Hosted CI
has exactly this scope and uploads its newly generated outputs.

## Full authorized replay

The five replay inputs and two supporting reconstruction sources are listed
by immutable revision, Git blob SHA1 and SHA256 in `source_ledger.json`.
Obtain those seven files from your authorized access to private
`rickyjreyes/wct-pde` at `051b6eca10c46b8e038993d60cabb20715194a20`.
A checkout or extracted directory preserving their paths is sufficient.

```sh
python audit/d4_binding/reproduce.py --d4-source-root /absolute/path/to/pinned/wct-pde --output-dir /tmp/d4-binding-full-fresh
```

Full mode runs 17 groups: 20 new identities; 240-profile and 59-pair new
diagnostics; all 48 derivation tests (36 inherited, 12 new); PR #35's 14,
PR #34's 38 and PR #33's 31 identities; the inherited adversarial campaign;
35 cross-paper identities; three inherited integrity checks; historical D4
scope; prospective self-test and five tests; pinned input preparation;
source-convention reconstruction; and all five original D4 rational gates.

Missing or mismatched sources fail closed. Only five pinned replay inputs are
copied into the fresh run directory. The two other files independently fix
the actual root operator `(Delta+1)^2` and audit its normalization. The root
validator matches the root receipt's original hash; the pinned audit document
differs from its earlier receipt input. Both document hashes are recorded.
The original interval root/coercivity campaigns are not rerun.

This phase's sources were freshly extracted through authenticated `fetch_file`
with base64 transport and independent byte/Git hash checks. The full recorded
run additionally supplies `--source-extraction-receipt` with
`source_extraction.json`; its hash is in `execution.json`. A future local
replay without this option does not claim new remote extraction. Private
source blobs and the candidate array are not published to this public repo.

## Evidence and limitations

- `verification.json`: independent symbolic identities and exact threshold,
  rational square-root and Machin-pi enclosures. Fractions are authoritative;
  decimal strings only display them.
- `diagnostics.json`: ordinary numerical trials and certified scalar samples
  of the analytical endpoint bound; no inferred numerical continuum infima.
- `source_ledger.json`: 106 preserved inherited file pins, the frozen stack,
  private input/supporting pins and literature access scope.
- `source_verification.json`: exact source conventions and operator checks.
- `d4_basin_replay.json`: five newly executed rational gates.
- `execution.json` and logs: actual commands, UTC times, dependency versions,
  return codes and hashes. Temporary paths describe provenance; the runner
  recreates outputs under the selected directory.
- `claim_matrix.json` and `proof_dependencies.json`: scoped proof statuses;
  validation rejects cycles or existence feeding into endpoint arguments.
- `manifest.json`: hashes every new deliverable except itself.

The canonical receipt is a full-local successful 17-group run. A public-CI
receipt cannot replace it. A workspace restart lost earlier uncommitted
Phase V artifacts; the restored files and campaigns were newly verified.
Unavailable earlier receipts are neither reconstructed nor claimed as
committed evidence. Existing historical evidence, including prior failed
attempts, is preserved byte-for-byte.

Internal independent reconstruction is not external mathematical review.
CI checks reproducibility, not all analytic proof steps. No original interval
campaign, stationary PDE search, propagation, physical spectrum, issue
creation, issue closure or merge is part of this verification.
