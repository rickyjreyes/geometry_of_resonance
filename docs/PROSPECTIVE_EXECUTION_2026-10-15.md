# Issue #17 execution and obstruction

Baseline main: `31d346e7b62c5a61042a57fdab9c5bec3e825e95` (PRs #30/#31 merged).
The original requirement is a derived numerical prediction, frozen choices,
calibration/injection validation, and exactly one untouched-holdout execution.
No numerical prediction or untouched holdout has been supplied by the existing
theory-to-measurement chain. Classification: **BLOCKED_MATHEMATICAL**.

The accepted physical-domain audit is reused without changing its conclusions:
`WCT_ACTIVE_DOMAIN_UNDERDETERMINED` and `SPECTRAL_MEASUREMENT_MAP_NOT_DERIVED`.
A phase-closure identity does not fix the measured residual frequency. The
separate response constructions in the cross-paper work do not supply the
missing physical domain, preparation/response coupling, or a numerical target
for an actual reserved observation. No existing CMS/GWTC fit is relabeled unseen.

The administrative gate now rejects nonfinite/reversed target intervals,
booleans/strings masquerading as numbers, and malformed object shapes. An
optional local artifact verifier checks declared source-manifest, derivation,
forward-map, and protocol SHA-256 bytes, including consistency with the declared
source digest. It rejects unknown roles, duplicate roles, path escapes and
digest corruption. Holdout outcomes are not a supported role. Even a complete
metadata/byte pass does not establish a scientifically derived prediction,
independent signoff, or an immutable preregistration.

Executed commands:

```bash
python -m unittest discover -s audit/prospective -p 'test_*.py'
python audit/prospective/freeze_gate.py --self-test
python audit/physical_domain/validate_records.py
python audit/prospective/freeze_gate.py --verify-artifacts .
```

Results: five adversarial tests pass; the existing gate self-test passes;
17 record checks and 30 symbolic checks pass with reference hashes verified.
The final command intentionally exits 2 with the exact missing prerequisites
in `audit/prospective/execution_2026-10-15.json`. It does not run calibration,
injections or a holdout. Source and result hashes are in the neighboring receipt.

Closure recommendation: keep #17 open. First establish a physical observable
and numerical target from specified theory/domain/response inputs. Then obtain
independent target-access review, freeze disjoint IDs and all choices, complete
null/injection calibration, and execute the reserved observation exactly once.
