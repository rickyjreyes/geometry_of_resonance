#!/usr/bin/env python3
"""Strict S004 axiom audit.

Every theorem root enumerated by `#print axioms` in `S004RegularAxiomAudit.lean` (136 roots: the 100
Phase XVIII-XXXII roots plus the 36 Phase XXXIII localization roots) must be reported exactly once
in the Lean log, using only `propext`, `Classical.choice` and `Quot.sound`. The log must contain no
Lean error. The check fails if a root is missing from the audit file, missing from the log,
reported twice, unexpected, or depends on any other axiom (e.g. `sorryAx`).

Usage: check_s004_regular_axioms.py <Lean audit log> [<audit .lean file>]
"""
from pathlib import Path
import re
import sys

EXPECTED = 136
ALLOWED = {"propext", "Classical.choice", "Quot.sound"}

if len(sys.argv) not in (2, 3):
    raise SystemExit("Usage: check_s004_regular_axioms.py <Lean audit log> [<audit .lean file>]")
log = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
audit = Path(sys.argv[2]) if len(sys.argv) == 3 else Path(__file__).with_name("S004RegularAxiomAudit.lean")
roots = re.findall(r"^#print axioms\s+(\S+)\s*$", audit.read_text(encoding="utf-8"), flags=re.M)
if len(roots) != EXPECTED or len(set(roots)) != EXPECTED:
    raise SystemExit(f"Audit file must enumerate {EXPECTED} distinct roots; "
                     f"found {len(roots)} ({len(set(roots))} distinct)")
if re.search(r":\d+:\d+: error", log):
    raise SystemExit("Lean reported an error in the audit log")
reports: dict[str, list[set[str]]] = {}
for name, body in re.findall(r"'([^']+)' depends on axioms:\s*\[([^\]]*)\]", log, flags=re.S):
    reports.setdefault(name, []).append({a.strip() for a in body.split(",") if a.strip()})
for name in re.findall(r"'([^']+)' does not depend on any axioms", log):
    reports.setdefault(name, []).append(set())
missing = [r for r in roots if r not in reports]
if missing:
    raise SystemExit(f"Missing axiom reports for {len(missing)} enumerated roots, e.g. {missing[:5]}")
duplicated = [r for r in roots if len(reports[r]) > 1]
if duplicated:
    raise SystemExit(f"Roots reported more than once: {duplicated[:5]}")
unexpected = sorted(set(reports) - set(roots))
if unexpected:
    raise SystemExit(f"Reports for roots not enumerated in the audit file: {unexpected[:5]}")
bad = {r: sorted(ax - ALLOWED) for r in roots for ax in reports[r] if ax - ALLOWED}
if bad:
    raise SystemExit("Unreviewed axioms: " + repr(bad))
print(f"PASS: all {EXPECTED} enumerated S004 canonical, characteristic, nodal and localization "
      "theorem roots use only propext, Classical.choice, Quot.sound.")
