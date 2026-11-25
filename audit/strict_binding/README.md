# Phase IV reproducibility

Scientific entry point:
[STRICT_BINDING_DECISION.md](../../docs/strict_binding/STRICT_BINDING_DECISION.md).

Python 3.12 with the actual pinned dependencies:

```sh
python -m pip install sympy==1.14.0 mpmath==1.3.0 numpy==2.3.5 scipy==1.16.2
python audit/foundational/validate_records.py
python audit/foundational_recovery/validate_records.py
python audit/strict_binding/validate_records.py
python audit/strict_binding/reproduce.py --output-dir /tmp/strict-binding-fresh --d4-source-root /absolute/path/to/authorized/pinned/wct-pde/files
```

Choose a fresh output directory. The full runner requires authorized local
copies of five source/input blobs at the pinned private wct-pde revision.
It checks all five SHA256 hashes before running the existing exact-rational
D4 verifier. The input directory need not be a Git checkout. Missing or
altered inputs fail the full replay; there is no automatic fallback.
It does not execute the original root/coercivity interval campaigns.

Public CI runs a separately labeled scope, requiring no private inputs:

```sh
python audit/strict_binding/reproduce.py --public-ci --output-dir /tmp/strict-binding-public
```

This runs the eleven public mathematical/regression groups plus the recorded
evidence validator. It checks the hashes, five recorded D4 gate results and
full-local execution provenance, but **does not execute the private-source
D4 replay**. Its receipt explicitly records `D4_rational_replay_executed=false`.
The committed canonical execution must instead be a successful full-local
13-group run with that field true; a public-CI receipt cannot replace it.

`execution.json` records the successful local run's actual commands, times,
return codes and output hashes. Its local temporary paths are execution
provenance, not additional dependencies: the runner reconstructs them under
the requested output directory. `manifest.json` hashes the complete new
artifact set and excludes itself. `source_ledger.json` pins 56 inherited files
and the five D4 source/input files. The new claim matrix supplements both
historical matrices without rewriting their earlier conclusions.

The successful run includes 36 derivation tests (24 inherited, 12 new), five
prospective-gate tests and their fail-closed self-test; 14 Phase IV, 38 Phase III,
31 PR #33 and 35 cross-paper symbolic checks; the specified adversarial
diagnostics; both inherited integrity validators; the historical D4 scope
validator; and all five rational basin gates.

The [initial failed setup attempt](development_attempt/README.md) is retained
with its actual receipt and logs. A manifest-schema assumption prevented the
source ledger from being generated, so source preparation failed after the
first eleven command groups had passed. Both manifest schemas were then read
correctly and the complete 13-group run passed. It was a packaging failure,
not a failed mathematical assertion or a successful D4 replay.

Symbolic and numerical checks are corroboration. The functional-analytic
proofs are in the reports. Gaussian trial optimizations supply upper bounds,
not continuum minimum values. No physical spectrum, new stationary root,
PDE propagation, external peer review, or formal proof certificate is claimed.

The [first hosted run](ci_attempt.json) passed the eleven public groups, then
failed with HTTP 404 on an unauthenticated download from the private wct-pde
repository. The correction makes the two scopes explicit and requires local
input access for full replay. Private source blobs are not copied into this
public repository. The complete local replay was rerun after this correction.

Hosted CI repeats the public checks and validates the recorded full-local
evidence, uploading fresh receipts for that scope. The draft PR records the
exact published head and actual CI links.
