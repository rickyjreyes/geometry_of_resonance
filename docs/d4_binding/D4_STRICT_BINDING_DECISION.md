# Frozen D4 strict-binding decision

**YES. Primary classification: D4_STRICT_BINDING_PROVED.**
Date: 2026-10-23. Frozen base: PR #35 at
`65499cabca7e177286b6541fa1a5a00d51a50ba5`, stacked on #34 and #33.

For the unchanged Hamiltonian, exact D4 coefficients, squared integral norm
and full complex H²(R²) admissible class,

    I_2(m_D4)<I_2(mu)+I_2(m_D4-mu) for EVERY 0<mu<m_D4.

The new sufficient threshold is

    m_B=3beta/[gamma sqrt(k⁴+3beta²/(32gamma))]
       =2.5503456363515490892...>m_D4=2.5446968948571200286... .

The comparison is decided by a positive exact rational, not decimal
rounding. Exact mass-preserving dilation puts arbitrary near-minimizers on
a scalar virial identity. Square completion and a direct Fourier estimate
then control S/Q. The amplitude comparison has an explicitly positive defect
uniform along each minimizing sequence, so strictness survives taking infima.
No minimizer is assumed in proving this result.

| Requested decision | Result |
|---|---|
| Ordinary subadditivity | PROVED by exact decoupling of compact separated approximants |
| Strict energy per mass | PROVED for every 0<a<b<=m_B |
| Every frozen interior split | PROVED |
| Small-mass coefficient | lambda0=0, from a two-sided infimum bound |
| Left energy slope at m_D4 | Exists and is strictly negative; a derivative-free upper bound is also proved |
| Endpoint interval | Explicit F(mu)>0 covers the entire half-interval, and symmetry covers the rest |
| Precompactness modulo translations | PROVED for every minimizing sequence at m_D4 |
| Whole-space minimizer existence | PROVED |
| Euler–Lagrange regularity | PROVED |
| Orbital H² stability | PROVED for the full minimizing set |

There is no remaining mathematical obstruction to the frozen all-split
claim. The universal all-positive-mass claim remains unproved; the sufficient
threshold is not asserted sharp. Uniqueness, identification with the periodic
root, a physical mass bridge or spectrum, and resolution of S004 are not
consequences. None is silently bundled into this decision.

Read the [complete proof](D4_SUBADDITIVITY_PROOF_OR_COUNTEREXAMPLE.md),
[energy analysis](D4_ENERGY_PER_MASS_ANALYSIS.md),
[endpoint analysis](D4_ENDPOINT_BINDING_ANALYSIS.md),
[compactness proof](D4_WHOLE_SPACE_COMPACTNESS.md),
[stability proof](D4_ORBITAL_STABILITY_STATUS.md),
[adversarial report](D4_ADVERSARIAL_SPLITTING_TESTS.md), and
[dependency graph](D4_PROOF_DEPENDENCY_GRAPH.md).
The [reproduction package](../../audit/d4_binding/README.md) distinguishes
analytic proof, exact arithmetic, diagnostics and actual execution scope.
Internal reconstruction and automated checks are not external mathematical
review or proof-assistant formalization.
