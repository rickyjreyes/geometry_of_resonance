#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# This script is intentionally idempotent after the first successful repair.
def replace_required(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"Expected {label} text was not found")
    return text.replace(old, new)


def repair_master() -> None:
    path = ROOT / "WCT_MASTER_EQUATIONS_UPDATED.md"
    text = path.read_text(encoding="utf-8")
    text = replace_required(
        text,
        "$$51\\ \mathrm{PASS} + 32\\ \mathrm{CONDITIONAL} + 23\\ \mathrm{DEFINITION} + 36\\ \mathrm{OPEN} = 142,$$",
        "$$59\\ \mathrm{PASS} + 27\\ \mathrm{CONDITIONAL} + 26\\ \mathrm{DEFINITION} + 30\\ \mathrm{OPEN} + 0\\ \mathrm{FAIL} = 142,$$",
        "stale master totals",
    )
    text = text.replace(
        "with no contradiction remaining in the current encoded specification.",
        "with no contradiction remaining in the current encoded specification. The historical baseline assignments come from `wct-sympy/equations/full_registry.yaml`; effective promotions and reclassifications come from `wct-sympy/equations/derived_overrides.yaml`; and the current public result is compiled in `compiled-registry.json`. A SymPy `PASS` is not automatically a Lean proof or empirical validation.",
        1,
    )
    text = replace_required(
        text,
        "$$32\\ \mathrm{CONDITIONAL} + 36\\ \mathrm{OPEN} = 68$$",
        "$$27\\ \mathrm{CONDITIONAL} + 30\\ \mathrm{OPEN} = 57$$",
        "stale unresolved frontier totals",
    )
    path.write_text(text, encoding="utf-8")


def repair_readme() -> None:
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    old_operator = """```math
\\Theta[\\psi]
= -\\frac{\\nabla^2\\psi}{\\psi + \\varepsilon e^{-\\alpha |\\psi|^2}}
```

This operator regularizes curvature at nodes of the field and supplies the main nonlinear feedback rail."""
    new_operator = """```math
R_\\varepsilon(\\psi)
=
\\frac{\\overline{\\psi}}
{|\\psi|^2+\\varepsilon^2e^{-2\\alpha|\\psi|^2}}
```

```math
\\Theta_\\varepsilon[\\psi]
=-(\\Delta\\psi)R_\\varepsilon(\\psi)
```

For `\\varepsilon>0`, the modulus-squared denominator is strictly positive. For nonzero `\\psi`, the regularized reciprocal approaches `1/\\psi` as `\\varepsilon\\to0`. This removes the historical scalar denominator zero, but it does not establish global PDE existence, uniqueness, regularity, or stability."""
    text = replace_required(text, old_operator, new_operator, "historical README operator")
    old_dim = "WCT gives a stability bound $n\\leq 3$ using Sobolev control, Lyapunov scaling, entropy localization, topology, arbitrary-data evolution, and curvature-feedback divergence."
    new_dim = "The verified mathematical result is the standard $H^2\\to L^\\infty$ Sobolev threshold for integer spatial dimension $n\\leq3$ under the stated domain assumptions. The broader WCT confinement conclusion remains conditional on the $H^2$-based stability route being necessary for the admissible confinement mechanism."
    text = replace_required(text, old_dim, new_dim, "unqualified dimensionality summary")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    repair_master()
    repair_readme()
    print("Repaired master totals, operator definition, and dimensionality boundary.")


if __name__ == "__main__":
    main()
