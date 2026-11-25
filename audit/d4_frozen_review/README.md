# Frozen D4 independent review and rational gap certificate

Start with [the proof](../../docs/d4_frozen_review/PROOF.md) and
[the scientific audit](../../docs/d4_frozen_review/SCIENTIFIC_AUDIT.md).
The qualitative frozen binding theorem was already merged in PR #36.
This package independently reconstructs it and certifies the additional bound
`I(mu)+I(m-mu)-I(m) > min(mu,m-mu)/400000` for every interior split.

## Minimal exact certificate, no third-party dependencies

```sh
python audit/d4_frozen_review/certify_gap.py --check audit/d4_frozen_review/certificate.json
python -O audit/d4_frozen_review/certify_gap.py --check audit/d4_frozen_review/certificate.json
python -m unittest discover -v -s audit/d4_frozen_review -p 'test_*.py'
```

The program uses integers and `fractions.Fraction` for all sign decisions.
It decodes hexadecimal dyadics directly, proves coarse π bounds by alternating
series, checks a rational square-root upper bound by squaring, and checks four
positive Bernstein coefficients on the entire interval. It imports none of
the historical derivation or interval code. The separate interpolation test
reconstructs the coefficients from the factored polynomial at four rational
nodes. Those four values identify a cubic exactly; this is not a four-point
sampling argument for positivity.

## Complete public reproduction

```sh
python -m pip install sympy==1.14.0 mpmath==1.3.0 numpy==2.3.5 scipy==1.16.2
python audit/d4_frozen_review/validate_records.py
python audit/d4_frozen_review/run_review.py --output-dir /tmp/d4-frozen-review-fresh
```

Choose an output directory that does not exist. Python 3.12 is used in CI.
The runner executes six public review groups; one contains the complete
15-group inherited Phase V reproduction. It never overwrites old receipts.
The public run does not claim to execute private-source periodic verification.

## Authorized local source and periodic replay

With the seven files at the exact `wct-pde` revision
`051b6eca10c46b8e038993d60cabb20715194a20`, preserving their paths:

```sh
python audit/d4_frozen_review/run_review.py \
  --output-dir /tmp/d4-frozen-review-local-fresh \
  --d4-source-root /absolute/path/to/pinned/wct-pde
```

The expected paths and hashes remain in
[the unchanged source ledger](../d4_binding/source_ledger.json).
This adds source-convention verification and all five periodic rational basin
gates, for eight review groups. It does not rerun the original interval
root/coercivity campaigns. The whole-space proof does not use the periodic
root as an existence premise.

## Recorded evidence

- `certificate.json`: deterministic exact certificate for the full interval.
- `evidence/execution.json` and logs: actual successful eight-group local run.
- `evidence/baseline_execution.json`: nested inherited 15-group public receipt.
- `evidence/identities.json`: ten freshly reconstructed symbolic identities.
- `evidence/source_extraction.json` and `source_verification.json`: fresh
  authenticated retrieval metadata and checks for the seven pinned files.
- `evidence/d4_basin_replay.json`: the five newly executed periodic gates.
- `manifest.json`: hashes the new review files and pins the inherited evidence.

The baseline's generated diagnostics are reproducible with its unchanged
runner; their historical committed originals remain in `audit/d4_binding` and
`audit/strict_binding`. This package records the new nested execution receipt
and logs instead of duplicating those large unchanged diagnostic payloads.
Private input files are not committed. Internal review and executable
certificates are distinguished from external peer review and full formalization.
