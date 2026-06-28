#!/usr/bin/env python3
"""Verify deterministic reproducibility artifacts against expected SHA-256 values."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", type=Path, default=Path("reproducibility/EXPECTED_HASHES.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("reproducibility/artifacts"))
    args = parser.parse_args()

    expected = json.loads(args.expected.read_text(encoding="utf-8"))
    failures: list[str] = []
    actual: dict[str, str] = {}
    for name, wanted in sorted(expected.items()):
        path = args.artifact_dir / name
        if not path.is_file():
            failures.append(f"missing artifact: {path}")
            continue
        got = sha256(path)
        actual[name] = got
        if got != wanted:
            failures.append(f"{name}: expected {wanted}, got {got}")

    unexpected = sorted(path.name for path in args.artifact_dir.iterdir() if path.is_file() and path.name not in expected)
    if unexpected:
        failures.append("unexpected artifacts: " + ", ".join(unexpected))

    print(json.dumps(actual, indent=2, sort_keys=True))
    if failures:
        raise SystemExit("Reproducibility verification failed:\n- " + "\n- ".join(failures))
    print(f"Verified {len(actual)} deterministic artifacts.")


if __name__ == "__main__":
    main()
