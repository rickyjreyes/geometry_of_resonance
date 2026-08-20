#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import urllib.request
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "WCT_FULL_EQUATION_LIST_CORRECTED.md"
REF = os.environ.get("WCT_SYMPY_REF", "main")
RAW = f"https://raw.githubusercontent.com/rickyjreyes/wct-sympy/{REF}/equations"
ID_RE = re.compile(r"^##\s+((?:M|E|CLE|CM|TOP|CORR)\d+[A-Z]?|G1|EX|EY|EZ|FA)\s+[—-]\s+", re.M)
STATUS_RE = re.compile(r"^\*\*(?:Status|Current effective status):\*\*\s+.*$", re.M)
BASELINE_RE = re.compile(r"^\*\*Baseline status:\*\*\s+.*(?:\n|$)", re.M)
PROVENANCE_RE = re.compile(r"^\*\*Status provenance:\*\*.*(?:\n|$)", re.M)
MARK = {"PASS": "✅ `PASS`", "CONDITIONAL": "⚠️ `CONDITIONAL`", "DEFINITION": "◻️ `DEFINITION`", "OPEN": "○ `OPEN`", "FAIL": "❌ `FAIL`"}
EXPECTED = {"PASS": 68, "CONDITIONAL": 18, "DEFINITION": 26, "OPEN": 30, "FAIL": 0}
OLD_TOTALS = r"$$59\ {\rm PASS} + 27\ {\rm CONDITIONAL} + 26\ {\rm DEFINITION} + 30\ {\rm OPEN} + 0\ {\rm FAIL} = 142.$$"
NEW_TOTALS = r"$$68\ {\rm PASS} + 18\ {\rm CONDITIONAL} + 26\ {\rm DEFINITION} + 30\ {\rm OPEN} + 0\ {\rm FAIL} = 142.$$"


def rows(name: str) -> list[list[str]]:
    request = urllib.request.Request(f"{RAW}/{name}", headers={"User-Agent": "wct-status-sync"})
    with urllib.request.urlopen(request, timeout=45) as response:
        value = yaml.safe_load(response.read().decode("utf-8")) or []
    return [[str(part) for part in row] for row in value]


def main() -> None:
    baseline_rows = rows("full_registry.yaml")
    override_rows = rows("derived_overrides.yaml")
    baseline = {object_id: status for object_id, _checker, status in baseline_rows}
    effective = dict(baseline)
    checkers: dict[str, str] = {}
    for object_id, checker, status in override_rows:
        if object_id not in baseline:
            raise RuntimeError(f"Unknown override ID {object_id}")
        effective[object_id] = status
        checkers[object_id] = checker
    counts = Counter(effective.values())
    normalized = {status: counts.get(status, 0) for status in EXPECTED}
    if len(effective) != 142 or normalized != EXPECTED:
        raise RuntimeError(f"Invalid effective registry: total={len(effective)}, counts={normalized}")

    text = DOC.read_text(encoding="utf-8")
    if NEW_TOTALS not in text:
        if OLD_TOTALS not in text:
            raise RuntimeError("Canonical totals line not found")
        text = text.replace(OLD_TOTALS, NEW_TOTALS, 1)

    note = """## Effective-status provenance

The formulas, names, assumptions, and historical baseline classifications originate in this canonical document. The **current effective status** shown for each object is generated from `wct-sympy/equations/full_registry.yaml` plus the higher-precedence `wct-sympy/equations/derived_overrides.yaml` layer. Changed objects retain their baseline status and override checker directly below the effective status.

A SymPy `PASS` reports success of its assigned check under declared assumptions. It is not automatically a Lean proof or empirical validation. Verification kind, Lean declarations, empirical state, and complete provenance are published in the compiled machine-readable registry.

"""
    marker = "## Current totals\n"
    if "## Effective-status provenance" not in text:
        if marker not in text:
            raise RuntimeError("Current totals heading not found")
        text = text.replace(marker, note + marker, 1)

    matches = list(ID_RE.finditer(text))
    ids = [match.group(1) for match in matches]
    if len(ids) != 142 or len(set(ids)) != 142 or set(ids) != set(effective):
        raise RuntimeError("Canonical object headings do not match the 142-object registry")

    chunks: list[str] = []
    cursor = 0
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        chunks.append(text[cursor:start])
        section = text[start:end]
        object_id = match.group(1)
        section = BASELINE_RE.sub("", section)
        section = PROVENANCE_RE.sub("", section)
        status_match = STATUS_RE.search(section)
        if not status_match:
            raise RuntimeError(f"{object_id}: status line not found")
        lines = [f"**Current effective status:** {MARK[effective[object_id]]}"]
        if effective[object_id] != baseline[object_id]:
            lines.append(f"**Baseline status:** {MARK[baseline[object_id]]}")
            lines.append(f"**Status provenance:** `derived_overrides.yaml` via `{checkers[object_id]}`.")
        block = "\n\n".join(lines)
        section = section[:status_match.start()] + block + section[status_match.end():]
        chunks.append(section)
        cursor = end
    chunks.append(text[cursor:])
    output = "".join(chunks)

    if NEW_TOTALS not in output:
        raise RuntimeError("Effective totals missing after generation")
    for object_id in ("E5", "E15", "E32", "E41", "E50", "CLE5", "CLE8", "TOP3", "CORR2"):
        heading = re.search(rf"^##\s+{object_id}\s+[—-].*$", output, re.M)
        if not heading:
            raise RuntimeError(f"Missing {object_id}")
        next_heading = re.search(r"^##\s+", output[heading.end():], re.M)
        end = heading.end() + next_heading.start() if next_heading else len(output)
        if f"**Current effective status:** {MARK[effective[object_id]]}" not in output[heading.start():end]:
            raise RuntimeError(f"{object_id}: wrong effective status")

    DOC.write_text(output, encoding="utf-8")
    changed = sum(effective[key] != baseline[key] for key in effective)
    print(f"Synchronized 142 objects; {changed} changed objects retain baseline provenance.")


if __name__ == "__main__":
    main()
