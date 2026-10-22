# Initial execution attempt

The first ledger-generation command assumed both inherited manifests used a `files` array. PR #33 instead uses a `sha256` mapping. Ledger generation failed with `KeyError: files`; the runner was started before that setup result was inspected. Its first eleven command groups passed, then pinned-source preparation failed because the new source ledger did not yet exist. These are the original command receipt and logs.

The generator was corrected to read both schemas, all 56 inherited pins were captured, and the complete run was repeated in a fresh directory. No mathematical test failure was hidden, no historical file changed, and this failed attempt is not counted as a successful D4 replay.
