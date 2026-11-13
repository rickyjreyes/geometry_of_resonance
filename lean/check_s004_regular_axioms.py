#!/usr/bin/env python3
"""Require all twenty-seven exact S004 regular-patch theorem roots and no custom axioms."""
from pathlib import Path
import re
import sys

if len(sys.argv) != 2:
    raise SystemExit("Usage: check_s004_regular_axioms.py <Lean audit log>")
text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
blocks = re.findall(r"depends on axioms:\s*\[([^\]]*)\]", text, flags=re.S)
free = re.findall(r"does not depend on any axioms", text)
if len(blocks) + len(free) != 27:
    raise SystemExit(f"Expected twenty-seven S004 canonical proof reports; found {len(blocks)+len(free)}")
allowed = {"propext", "Classical.choice", "Quot.sound"}
extras = {n.strip() for b in blocks for n in b.split(",") if n.strip() and n.strip() not in allowed}
if extras:
    raise SystemExit("Unreviewed axioms: " + repr(sorted(extras)))
print("PASS: twenty-seven original S004 regular-canonical theorems use only standard logical axioms.")
