#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import urllib.request
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "WCT_FULL_EQUATION_LIST_CORRECTED.md"
SYMPY_REF = os.environ.get("WCT_SYMPY_REF", "main")
BASE_URL = f"https://raw.githubusercontent.com/rickyjreyes/wct-sympy/{SYMPY_REF}/equations/full_registry.yaml"
OVERRIDE_URL = f"https://raw.githubusercontent.com/rickyjreyes/wct-sympy/{SYMPY_REF}/equations/derived_overrides.yaml"
ID_RE = re.compile(r"^##\s+((?:M|E|CLE|CM|TOP|CORR)\d+[A-Z]?|G1|EX|EY|EZ|FA)\s+[—-]\s+", re.M)
STATUS_RE = re.compile(r"^\*\*(?:Status|Current effective status):\*\*\s+.*$", re.M)
BASELINE_RE = re.compile(r"^\*\*Baseline status:\*\*\s+.*(?:\n|$)", re.M)
STATUS_MARKS = {
    "PASS": "✅ `PASS`",
    "CONDITIONAL": "⚠️ `CONDITIONAL`",
    "DEFINITION": "◻️ `DEFINITION`",
    "OPEN": "○ `OPEN`",
    "FAIL": "❌ `FAIL`",
}
EXPECTED = {"PASS": 59, "CONDITIONAL": 27, "DEFINITION": 26, "OPEN": 30, "FAIL": 0}


def fetch_rows(url: str) -> list[list[str]]:
    request = urllib.request.Request(url, headers={"User-Agent": "wct-status-sync"})
    with urllib.request.urlopen(request, timeout=45) as response:
        rows = yaml.safe_load(response.read().decode("utf-8")) or []
    if not isinstance(rows, list):
        raise RuntimeError(f"Expected a list from {url}")
    return [[str(value) for value in row] for row in rows]


def load_statuses() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    baseline_rows = fetch_rows(BASE_URL)
    override_rows = fetch_rows(OVERRIDE_URL)
    baseline = {object_id: status for object_id, _checker, status in baseline_rows}
    override_checker = {object_id: checker for object_id, checker, _status in override_rows}
    effective = dict(baseline)
    for object_id, _checker, status in override_rows:
        if object_id not in baseline:
            raise RuntimeError(f"Override references unknown object {object_id}")
        effective[object_id] = status
    if len(effective) != 142:
        raise RuntimeError(f"Expected 142 objects, found {len(effective)}")
    counts = Counter(effective.values())
    normalized = {status: counts.get(status, 0) for status in EXPECTED}
    if normalized != EXPECTED:
        raise RuntimeError(f"Unexpected effective totals: {normalized}")
    return baseline, effective, override_checker


def status_block(object_id: str, baseline: dict[str, str], effective: dict[str, str], checker: dict[str, str]) -> str:
    lines = [f"**Current effective status:** {STATUS_MARKS[effective[object_id]]}"]
    if baseline[object_id] != effective[object_id]:
        lines.append(f"**Baseline status:** {STATUS_MARKS[baseline[object_id]]}")
        lines.append(
            f"**Status provenance:** promoted by `wct-sympy/equations/derived_overrides.yaml` via `{checker[object_id]}`."
        )
    return "\n\n".join(lines)


def replace_object_statuses(text: str, baseline: dict[str, str], effective: dict[str, str], checker: dict[str, str]) -> str:
    matches = list(ID_RE.finditer(text))
    ids = [match.group(1) for match in matches]
    if len(ids) != 142 or len(set(ids)) != 142:
        raise RuntimeError(f"Canonical document must contain 142 unique object headings; found {len(ids)}")
    unknown = sorted(set(ids) - set(effective))
    missing = sorted(set(effective) - set(ids))
    if unknown or missing:
        raise RuntimeError(f"Registry mismatch: unknown={unknown}; missing={missing}")

    parts: list[str] = []
    cursor = 0
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        parts.append(text[cursor:start])
        section = text[start:end]
        object_id = match.group(1)
        section = BASELINE_RE.sub("", section)
        section = re.sub(r"^\*\*Status provenance:\*\*.*(?:\n|$)", "", section, flags=re.M)
        status_match = STATUS_RE.search(section)
        if not status_match:
            raise RuntimeError(f"{object_id}: no status line found")
        section = section[:status_match.start()] + status_block(object_id, baseline, effective, checker) + section[status_match.end():]
        parts.append(section)
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def replace_summary(text: str) -> str:
    text = re.sub(
        r"\$\$51\\ \{\\rm PASS\} \+ 32\\ \{\\rm CONDITIONAL\} \+ 23\\ \{\\rm DEFINITION\} \+ 36\\ \{\\rm OPEN\} = 142\.\$\$",
        "$$59\\ {\\rm PASS} + 27\\ {\\rm CONDITIONAL} + 26\\ {\\rm DEFINITION} + 30\\ {\\rm OPEN} + 0\\ {\\rm FAIL} = 142.$$",
        text,
    )
    provenance = """## Effective-status provenance

The formulas, names, assumptions, and historical baseline classifications originate in this canonical document. The **current effective status** shown for each object is generated from `wct-sympy/equations/full_registry.yaml` plus the higher-precedence `wct-sympy/equations/derived_overrides.yaml` layer. Changed objects retain their baseline status and override checker directly below the effective status.

A SymPy `PASS` reports success of its assigned check under declared assumptions. It is not automatically a Lean proof or empirical validation. Verification kind, Lean declarations, empirical state, and full machine-readable provenance are published in `compiled-registry.json` and `research-corpus.json`.

"""
    marker = "## Current totals\n"
    if "## Effective-status provenance" not in text:
        text = text.replace(marker, provenance + marker, 1)
    return text


def validate_output(text: str, effective: dict[str, str]) -> None:
    for object_id in ("E5", "E9", "E13", "E14", "E18", "E58", "CM9", "CM11", "CM12", "CM13", "CM16", "CM18", "E70"):
        heading = re.search(rf"^##\s+{re.escape(object_id)}\s+[—-].*$", text, flags=re.M)
        if not heading:
            raise RuntimeError(f"Missing heading {object_id}")
        next_heading = re.search(r"^##\s+", text[heading.end():], flags=re.M)
        end = heading.end() + next_heading.start() if next_heading else len(text)
        section = text[heading.start():end]
        expected_line = f"**Current effective status:** {STATUS_MARKS[effective[object_id]]}"
        if expected_line not in section:
            raise RuntimeError(f"{object_id}: expected status line not found")
    if "59\\ {\\rm PASS}" not in text:
        raise RuntimeError("Effective totals were not updated")


def main() -> None:
    baseline, effective, checker = load_statuses()
    text = REGISTRY_PATH.read_text(encoding="utf-8")
    text = replace_summary(text)
    text = replace_object_statuses(text, baseline, effective, checker)
    validate_output(text, effective)
    REGISTRY_PATH.write_text(text, encoding="utf-8")
    changed = sum(1 for object_id in effective if effective[object_id] != baseline[object_id])
    print(f"Synchronized 142 canonical objects; {changed} retain explicit baseline-to-effective provenance.")


if __name__ == "__main__":
    main()
