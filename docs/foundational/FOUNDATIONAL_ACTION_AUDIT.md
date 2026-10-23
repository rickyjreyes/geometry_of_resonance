# Authoritative foundational source matrix

Audit date: 2026-10-20. Entry point: [decision](FOUNDATIONAL_DECISION.md).
The governing actions below are distinct mathematical models. Reused coefficient
letters do not identify their units or physical values. This work uses the
current default branches and separately pinned active-PR evidence, not only
the October 14 handoff. The [source ledger](../../audit/foundational/source_ledger.json)
records exact revisions and content hashes; [repository/issue snapshot](../../audit/foundational/repository_snapshot.json)
records the observed state. Original manuscripts and frozen results are unchanged.

## Current revisions and live-PR distinction

| Repository | Default-branch revision inspected | Additional relevant state |
|---|---|---|
| geometry_of_resonance | `5f786e270a3ea8a59703772ba61d615dffe001f4` | PRs #30–32 merged; foundational #11–18 and #27 open; paper-coverage PR #5 open |
| eigen | `957cd8f382ba72f3764941c372222d7ebb58ed7e` | #15 merged at `49a8b52ac433e2c641481f062bf47dbfca2656dc`; #16 merged at current head; issues #3–14 open |
| wct-pde | `ecc2a446cc8fbf2f104bc49aa0a9fdf5a9739876` | #26 remains open at `051b6eca10c46b8e038993d60cabb20715194a20`; #28 merged **into that branch**, not default main |
| dimension | `f61d924f859d79e7ff733c3e30ba8baf4b5ceb81` | Numerical-audit #9 open at `57f73abdced315ef66dc6cc91deb1ff63a9131b0`; nonlinear refinement remains unresolved |
| wct-sympy | `9656863097da396fa2c8d0b8ea190afeabdd05d8` | Symbolic/registry support; continuum and physical bridges remain scoped obligations |
| wct-lean | `dea05da8a6e40361e79ee685a7939a4c6f9a8696` | Open #29–31 include noncoercivity, analytic contracts and local variation work; not merged full-action/PDE proofs |

For wct-pde, #24/#25 were closed on evidence
`e9a2b589243cfc8fa1492b67a02949c9ba660e93`; #27's quantitative basin was
closed on `1bda6eac963bf58be3ff2e48da92716be4e3298c`. Those results are included
in the current #26 head. The issue closure is real even though its containing
PR has not reached main. #21/#22/#23 remain open. The review notes, source
blobs and input receipts were read and hash-matched; the rational basin replay
was executed again. The full interval root/coercivity campaign was not rerun.

## Governing-model matrix

Abbreviations: N=integral|psi|², rho=|psi|²,
F=rho/(rho+epsilon² exp(-2 alpha rho))², h=1/(rho+delta²).
Statements concerning derivatives/integrals require the stated smoothness and
boundary conditions; algebraic invariance alone is not a conservation theorem.

| Model and governing definition | Fields, domain and boundaries | Symmetries / conserved or decreasing quantities | Established result and numerical evidence | Conditional result / outstanding obligation | Pinned source |
|---|---|---|---|---|---|
| S004 corrected action: partial psibar partial psi -V+F[kappa|Box psi|²+theta|Delta_h psi|²+gamma Re(Box psi conjugate(Delta_h psi))] | One complex scalar, fixed Lorentzian g and timelike foliation n; flat inertial specialization analyzed; compact variations/periodic/zero flux | Global U(1), translations/rotations in flat fixed background; actual higher-derivative H,Q; preferred foliation prevents asserting scalar-only Lorentz invariance | Exact EL/quadratic expansion; full regular constraint count; opposite residues; fixed-Q energetic obstruction; new homogeneous Sobolev UV obstruction | V, dynamic g/n, physical normalization, cutoff and nonlinear EFT not selected; node degeneracy needs separate domain | geometry base, corrected action; Q1–Q7; this audit; eigen #16 |
| Scalar-tensor candidate: F_g(rho)R/2-Z partial psibar partial psi -V-beta_Theta|Theta|²/2+S_m | Independent metric plus complex scalar and matter; boundary/gravitational variational data needed | Diffeomorphism covariance if all fields varied, global phase; stress tensor and higher-derivative current | Formal variation and conditional low-energy Einstein limit; scalar metric rank obstruction is separate | Full coupled constraint/ghost audit not supplied by the fixed-metric calculation; independent metric is not emergent tensor theorem | DERIVATION_CLOSURE.md at geometry base |
| Corrected positive spatial M2: E_F=c1||grad psi||²+c2 integral F|Delta psi|² | Complex H²(R^d); fixed positive N for variational question; decay for EL | Global U(1), translations, rotations; no unique evolution supplied | No H²-local constrained minimum in d=1–4; infimum zero unattained; no dilation-admissible regular critical point | It is not S004's full Hamiltonian; weak stationary extensions require separate analysis | eigen #16, Theorem 1; geometry corrected registry |
| Implemented eigen additive energy: integral c1|grad u|²+c2|Delta u/(u+epsilon exp(-alpha u²))|² | Real bounded H², positive-denominator branch on R^d; fixed N | Spatial symmetries; N imposed as variational constraint | Exact d=1–4 dilation obstruction; nine frozen attempts negative; discrete Hessian differences checked | Radial repeated-Laplacian defect remains; no validated physical spectrum or dynamical lifetimes | eigen #15/#16; vacuum.py, vacuum2.py, SCALAR_DILATION_THEOREM.md |
| PFF/real Swift–Hohenberg: A_t=rA-a Delta A-b Delta²A-beta A³ | Real field with declared periodic/decay/natural boundary data | Gradient energy decreases; N generally not conserved | Exact declared gradient variation, finite-k symbol and saturation within its own model | Closed S004 lacks a derived dissipative mechanism; onset and nonlinear coefficients not fixed by its quadratic reduction | geometry CORRECTED_ACTION_TO_PFF_REDUCTION.md |
| Projection-free cubic–quintic Hamiltonian: i psi_t=(Delta+kstar²)²psi-beta rho psi+gamma rho²psi | Complex H²; validated application on length-42 2D torus | H and N conserved; phase/translations | Nearby smooth real-D4 root, full-complex local all-time orbital H² theorem and explicit basin; separate finite-grid Padé result | No parent-action map or whole-space transfer; optimized integrator/envelope/gain issues #21–23 open | wct-pde main implementation; #26 evidence head `051b6eca...` |
| Closure functional: a4||Delta psi||²-a2||grad psi||²+beta||psi||4^4/2+kappa_r integral h|Delta psi|² | Complex fixed-N H²; periodic or whole space distinguished | Declared projected real-gradient flow preserves N and decreases energy | Fixed-N lower bound/coercivity, fixed-shape scale competition, periodic attainment; negative-energy vanishing packets | Whole-space binding below true escape threshold and exclusion of splitting; parent-action match | geometry LOCALIZATION_AND_DOMAIN_SELECTION.md; eigen #16 C18–C23 |
| DS-01: finite-band linear flow + additive curvature RHS -mu_entr psi Delta log(rho+delta) | Complex field on periodic tori in d=2,3,4; numerical denominator floor | N and historical energy are diagnostics, not established invariants | Model and regularity boundaries explicit; nonlinear refinement failure retained | Full variational identity/well-posedness and converged lifetime comparison open | dimension docs/MATH_AND_CLAIMS.md at default head |

No mathematical result in this table supplies a source field for the separate
externally attractive eigen model. Eigen #16 proves binding of its declared
linear exterior operator, conditional on that external source. It is not a
self-bound vacuum state or a derivation of source backreaction.

## Conflicting or stale statements resolved in the living record

1. The additive historical denominator is not phase covariant; the corrected
   denominator is globally phase safe. Neither alone is a local gauge theory.
2. The older curved S004 equation uses Delta_h where its adjoint is required
   unless h satisfies extra conditions. The flat derivation remains exact.
3. Positive curvature quadratic form does not imply a positive Hamiltonian,
   real high-frequency roots, or a well-posed full initial-value problem.
4. A finite oscillatory-frequency minimum is not an unstable PFF growth band.
5. The October 14 D4 map's conditional #24/#25 rows and earlier reports' open
   quantitative basin are historical. The current additive update recognizes
   closed #24/#25/#27 with pinned scope; it retains #21–23 separately.
6. D4 here is square symmetry in two dimensions, never four spatial dimensions.
7. The symbolic/Lean counts denote support for encoded objects or contracts,
   not proofs of physical model equivalence. The open Lean branches do not
   supply a full physical closure or displace the source action.
8. Spectral m=(hbar/c)sqrt(lambda) is a theorem for a specified separated
   parent operator. It does not identify S004, a physical confinement length,
   arbitrary curvature averages, or observed particle labels by itself.

These clarifications are added to current indexes and notes. They do not edit
the dated manuscripts, original experimental observations or frozen thresholds.
