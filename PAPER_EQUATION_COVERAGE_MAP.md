# WCT Paper Equation Coverage Map

**Status:** First-pass audit map / review draft  
**Date:** 2026-06-28  
**Scope:** Maps the papers and branches listed in `READING_ORDER.md` to the canonical 142-object WCT equation registry in `WCT_FULL_EQUATION_LIST_CORRECTED.md` and `WCT_MASTER_EQUATIONS_UPDATED.md`.

This file is **not** a replacement for the canonical equation registry. It is a coverage and provenance checklist.

The goal is to answer:

> Is every major equation from each WCT paper represented by a stable canonical equation ID?

---

## Status labels

| Status | Meaning |
|---|---|
| `REPRESENTED` | Major equations appear to be covered by stable registry IDs. |
| `PARTIAL` | Core equations are represented, but paper-specific equations, parameters, data tests, or protocol formulas need additional mapping. |
| `NOT_CLEAR` | Repository search did not show a clear registry representation. Needs paper extraction. |
| `NOT_REGISTRY_TARGET` | Mostly narrative, prediction-audit, architecture, or non-equation paper; may not require full equation-object mapping. |
| `NEEDS_EXTRACTION` | PDF/paper equation extraction needed before a final claim. |

---

## Current canonical registry summary

The canonical equation registry currently contains 142 objects:

```text
M1-M8, with M6A and M6B separated
E1A, E1B, E2-E82
CLE1-CLE10
G1, EX, EY, EZ, FA
CM1-CM20
TOP1-TOP9
CORR1-CORR6
```

Current symbolic audit totals:

```text
59 PASS + 27 CONDITIONAL + 26 DEFINITION + 30 OPEN + 0 FAIL = 142
```

Important caveat:

A SymPy `PASS` is not a Lean proof and not empirical validation. It only means the assigned symbolic, dimensional, logical, limit, or consistency check passes under declared assumptions.

---

# Paper-by-paper first-pass map

## 1. The Geometry of Resonance: Wave Confinement Theory and the Emergence of Mass, Force, and Spacetime

**Status:** `REPRESENTED`

**Likely registry coverage:**

```text
M2, M6A, M6B, M7
E17-E23
E49
E51-E56
E81-E82
```

**Covered themes:**

- regularized curvature operator;
- WCT energy / Lyapunov candidate;
- curvature-feedback dynamics;
- higher-order operator sector;
- effective metric ansatz;
- effective-mass gap relation;
- curvature commutator and effective potential;
- coherence length and information-geometry tensor.

**Audit note:** Core mathematical objects appear represented. Physical interpretation remains conditional where the registry marks it conditional/open.

---

## 2. Phase-Flux Field (PFF): Axiomatic Substrate for Wave Confinement Theory

**Status:** `REPRESENTED`

**Likely registry coverage:**

```text
M3
E9-E16
E57-E64
```

**Covered themes:**

- phase-flux constitutive relation;
- conservation equation;
- radial shell quantization;
- phase winding;
- finite-band dispersion rail;
- band-pass amplitude evolution;
- band-pass Lyapunov functional;
- modal growth and spectral concentration;
- Swift-Hohenberg shell operator;
- Green kernel;
- annular projection;
- selected wavelength.

**Audit note:** This is one of the strongest registry-covered branches.

---

## 3. Rest Energy from Density-Weighted Loop Curvature: A Covariant Locking Principle

**Status:** `REPRESENTED`

**Likely registry coverage:**

```text
M1
E1A-E8
E6-E7
```

**Covered themes:**

- curvature-rate density;
- curvature spectral rate;
- weighted loop average;
- loop-locking action;
- covariant locking solution;
- effective wavenumber chain;
- rest energy and mass-curvature law;
- corrected weighted-lock identity.

**Audit note:** Core mass-locking structure appears directly represented.

---

## 4. Hard Upper Bound on Spatial Dimensionality in Wave Confinement Theory

**Status:** `REPRESENTED`

**Likely registry coverage:**

```text
M4
E24-E27
E65-E70
TOP1-TOP9
```

**Covered themes:**

- Sobolev embedding threshold;
- critical Sobolev exponent;
- corrected curvature L2 bound;
- finite-energy confinement;
- failure of H2 to L-infinity above three dimensions;
- corrected high-regularity curvature bound;
- dimensional stability criterion;
- topology objects.

**Audit note:** The registry correctly treats this as an embedding/stability criterion, not a complete universal nonlinear stability theorem.

---

## 5. Structure and Derivation of Physical Constants through Wave Confinement

**Status:** `PARTIAL`

**Likely registry coverage:**

```text
E6-E7
E19
E28-E34
E37-E40
E49
E81
CLE9-CLE10
```

**Possibly missing / needs extraction:**

```text
xi parameter origin
sigma parameter origin
beta/theta parameter origin
harmonic mass maps
physical constant derivation tables
fitted-vs-derived status for constants
G-scale relation, if present as a central equation
DNA / Higgs / heavy-particle harmonic relations, if paper-level major equations
```

**Audit note:** The general mass-curvature and coherence machinery is represented, but the constants paper likely needs a dedicated extraction pass because parameter origins and quantitative constant maps are not obviously fully registered.

---

## 6. Self-Emergent Fourier Cymatics: Entropic Eigenmodes out of Chaos

**Status:** `REPRESENTED`

**Likely registry coverage:**

```text
M3
E12-E16
E29-E34
E57-E64
E81
```

**Covered themes:**

- finite-band rail;
- amplitude evolution;
- spectral entropy;
- support-entropy relation;
- entropy-drop pruning;
- spectral energy concentration;
- annular projection;
- coherence length.

**Audit note:** Core equations appear represented. Simulation-specific numerical parameter tables should still be linked as experiment/simulation objects, not necessarily equation objects.

---

## 7. Emergence of Effective Mass: Solenoidal Topology of Vibrational Energy

**Status:** `REPRESENTED / PARTIAL`

**Likely registry coverage:**

```text
M1
E1A-E8
E19
E49
CLE1-CLE10
TOP objects
```

**Possibly missing / needs extraction:**

```text
worked helix/circle examples
solenoidal topology-specific mass examples
SU(2)/U(1) interpretive equations, if presented as equations
ppm-scale cavity prediction equations
```

**Audit note:** The main curvature-to-mass law is represented. Worked examples and topology/gauge interpretation may need paper-specific extraction.

---

## 8. Logarithmic Curvature Flow, Filament Localization, and the Geometric Origin of the Lepton Mass Spectrum

**Status:** `REPRESENTED / PARTIAL`

**Likely registry coverage:**

```text
G1
EX
EY
EZ
FA
CLE sector
Koide files in Lean/SymPy, where applicable
```

**Possibly missing / needs extraction:**

```text
lepton filament spectrum equations
explicit mass-spectrum mapping equations
toroidal spectrum equations
Koide-geometry equations if not represented by Koide formal files
```

**Audit note:** The log transform and Cole-Hopf/diffusion reduction are represented. The lepton-spectrum-specific branch likely needs more precise mapping.

---

## 9. Wave Confinement Theory Predicts the Koide Mass Relation

**Status:** `PARTIAL`

**Known validation presence:**

```text
wct-sympy/scripts/check_koide.py
wct-sympy/tests/test_koide.py
wct-lean/WCTLean/Koide.lean
wct-lean/WCTLean/Models/KoideDerivation.lean
```

**Likely registry coverage:**

```text
mass-curvature basis: E6-E7
curvature spectrum basis: CLE / G / EX-EZ where applicable
```

**Needs registry integration check:**

```text
Koide Q expression
curvature harmonic K(s)
charged-lepton Q = 2/3 statement
effective-spin caveats
mass triplet parameterization
```

**Audit note:** Koide is clearly present in Lean/SymPy files, but it is not clearly exposed as stable IDs inside the 142 canonical registry. This should be upgraded into explicit registry objects or mapped as external validation objects.

---

## 10. Observation of Long-Lived Photon Resonance Confinement in Water Cavities

**Status:** `PARTIAL`

**Likely registry coverage:**

```text
E45
E46
E47
E48
E50
E52-E56
E81
```

**Possibly missing / needs extraction:**

```text
experimental persistence equations
ringdown / Q calculation details
FFT structure definitions
perturbation/re-lock criteria
control conditions
measurement thresholds
```

**Audit note:** General Q, power balance, cavity-matching, and coherence equations are represented. Protocol-specific equations and measured quantities should be mapped as experiment objects.

---

## 11. Prediction & Protocol Ledger: Long-Lived Harmonic State Induction in Photodiodes

**Status:** `PARTIAL / NEEDS_EXTRACTION`

**Likely registry coverage:**

```text
E45
E50
E52-E56
E81
```

**Possibly missing / needs extraction:**

```text
prediction ledger pass/fail criteria
photodiode harmonic induction formulas
measurement windows
threshold equations
protocol-specific observables
```

**Audit note:** This appears more like an experiment/protocol object than a pure equation paper. It should be mapped into prediction and experiment registries.

---

## 12. Self-Emergent Fourier Cymatics repeat entry

**Status:** `REPRESENTED`

Same coverage as paper 6.

---

## 13. JUNO Energy Resolution and Detectability of WCT Ghost-Mode Neutrinos

**Status:** `PARTIAL`

**Likely registry coverage:**

```text
G1
CM12-CM13, if spectrum/peak quantities are used
E30-E34, if entropy/spectral support is used
```

**Possibly missing / needs extraction:**

```text
Gaussian smearing equations
energy-resolution function
log-energy modulation detectability inequalities
JUNO-specific resolvability thresholds
look-elsewhere or null-test formulas
```

**Audit note:** The log-periodic ansatz is represented by G1. Detector-resolution/detectability equations need explicit mapping.

---

## 14. A Curvature-Induced Log-Periodic Deformation of C9(q^2): WCT and the LHCb B0 -> K*0 mu+ mu- Anomaly

**Status:** `NOT_CLEAR / NEEDS_EXTRACTION`

**Possibly related registry coverage:**

```text
G1
spectral/log-periodic machinery
```

**Likely missing / needs extraction:**

```text
delta C9(q^2) ansatz
log-frequency scan equations
null/shuffle test statistics
covariance / residual model equations
```

**Audit note:** No clear stable registry representation found for the C9-specific equations.

---

## 15. Log-Spectral Structure and Koide-Like Winding Geometry in Open-Data B0 -> K*0 mu+ mu- Candidate Spectra

**Status:** `NOT_CLEAR / NEEDS_EXTRACTION`

**Possibly related registry coverage:**

```text
G1
E10-E11
E30-E34
Koide validation files
```

**Likely missing / needs extraction:**

```text
KDE baseline repair equations
active-domain winding n
sideband/signal dilation formulas
Koide-like ratio definitions for spectra
veto stress-test equations
```

**Audit note:** Needs dedicated paper extraction.

---

## 16. Bin-Stable Log-Periodic Structure in Public NIST Atomic Line List

**Status:** `NOT_CLEAR / NEEDS_EXTRACTION`

**Possibly related registry coverage:**

```text
G1
E30-E34
```

**Likely missing / needs extraction:**

```text
bin-stability statistic
log-periodic fit model
null-test statistic
look-elsewhere correction
atomic-line preprocessing equations
```

**Audit note:** Needs dedicated paper extraction and should likely become a data-analysis/protocol object rather than core WCT equation object only.

---

## 17. Discrete Wave-Constrained Computation and Classical Complexity: Turing Equivalence for P and NP

**Status:** `REPRESENTED / PARTIAL`

**Likely registry coverage:**

```text
M5
E35-E40
E71-E76
```

**Covered themes:**

- discrete WCC update;
- fixed point definition;
- bandlimit from energy;
- spatial channel capacity;
- polynomial update bound;
- complexity identification;
- curvature-pruned search space;
- polynomial verification;
- WCC complexity equivalence.

**Possibly missing / needs extraction:**

```text
encoding maps
simulation overhead equations
exact P/NP relation statements
proof obligations for equivalence
```

**Audit note:** Core WCC equations are represented, but exact paper-level computational reductions need proof-object mapping.

---

## 18. P vs NP in Curvature-Bounded Wave Computation: Model-Relative Separation

**Status:** `PARTIAL`

**Likely registry coverage:**

```text
E71-E76
M5
```

**Possibly missing / needs extraction:**

```text
formal model-relative separation statement
finite-size families P_n, NP_n
reduction/cost model equations
oracle/impossibility assumptions
```

**Audit note:** Registry has the skeleton but not necessarily the full paper-level separation structure.

---

## 19. The Classical P vs NP Problem is Mathematically and Physically Ill-Posed

**Status:** `PARTIAL / NOT_REGISTRY_TARGET`

**Likely registry coverage:**

```text
M5
E71-E76
```

**Possibly missing / needs extraction:**

```text
formal critique definitions
physical resource constraint equations
logical reframing statements
```

**Audit note:** Mostly conceptual/formal critique; may need claim mapping more than equation mapping.

---

## 20. WaveLock: A Curvature-Locked One-Way Function Based on Nonlinear PDE Evolution

**Status:** `NOT_CLEAR / NEEDS_EXTRACTION`

**Possibly related registry coverage:**

```text
M7
E17-E18
E35-E40
E71-E76
```

**Likely missing / needs extraction:**

```text
WaveLock map definition
input/output encoding equations
nonlinear PDE evolution used as one-way candidate
avalanche metrics
adversarial test suite metrics
explicit non-security-claim boundaries
```

**Audit note:** This should probably have its own `WL` object family or be mapped as an applied computation branch.

---

## 21. Resonance-Confinement Architecture: A Physically Bounded Substrate for Safe Superintelligence

**Status:** `PARTIAL / NOT_REGISTRY_TARGET`

**Likely registry coverage:**

```text
M5
E35-E40
E71-E80
```

**Possibly missing / needs extraction:**

```text
coherence objective equations
contradiction-control equations
bounded-curvature update constraints
Lyapunov-like stabilization rules
architecture-specific protocol equations
```

**Audit note:** Mostly architecture/specification. It likely needs concept/claim/protocol mapping more than canonical equation mapping.

---

## 22. Recursive AI Drift: A 2025 Prediction Timeline External Validation Audit and Technical Note

**Status:** `NOT_REGISTRY_TARGET`

**Possibly related registry coverage:**

```text
E77-E80, if information/entropy dynamics are invoked
```

**Likely missing / needs extraction:**

```text
prediction-status table
external validation criteria
failure-case taxonomy
```

**Audit note:** This is mainly a prediction audit and historical validation note. It should be represented in prediction/claim registries, not necessarily as an equation-object set.

---

## 23. Nuclear Fusion Tokamak with Self Sustaining Resonance

**Status:** `PARTIAL / NEEDS_EXTRACTION`

**Likely registry coverage:**

```text
E45
E46
E47
E48
E50
E52-E56
```

**Likely missing / needs extraction:**

```text
diagnostics proxy I(t)
BES/reflectometry/CER proxy definitions
P_EM demand formula
zeta accounting fraction
handoff gate inequalities
wall-power latch equation
energy confinement margin equations
Monte Carlo/stress test criteria
SPARC adaptation equations
```

**Audit note:** General Q/power/confinement equations are represented, but the applied fusion-control logic should be mapped into a dedicated applied-control branch.

---

# Highest-priority missing or partial coverage areas

## A. Koide / lepton / harmonic mass branch

Recommended action:

```text
Create explicit registry objects for Koide and lepton-spectrum equations, or create a mapped external-validation family.
```

Candidate IDs:

```text
KDE1-KDE10 or KOIDE1-KOIDE10
```

Minimum objects:

```text
Koide Q expression
Q = 2/3 target statement
curvature harmonic K(s)
mass-triplet parameterization
effective-spin caveat
relation to E6/E7 mass-curvature law
```

## B. Constants / parameter-origin branch

Recommended action:

```text
Create PARAMETER_ORIGIN_MAP.md or registry/parameters.yaml.
```

Minimum objects:

```text
xi
sigma
beta
theta
gamma
r
fitted/derived/postulated classification
source paper location
```

## C. Open-data phenomenology branch

Recommended action:

```text
Create PHENOMENOLOGY_EQUATION_MAP.md or PHENO object family.
```

Minimum objects:

```text
JUNO smearing/detectability
C9(q^2) deformation
B0 -> K* active-domain winding
NIST line-list bin-stability
null tests
look-elsewhere corrections
```

## D. WaveLock branch

Recommended action:

```text
Create WAVELOCK_EQUATION_MAP.md or WL object family.
```

Minimum objects:

```text
WaveLock map
PDE evolution operator
input/output encoding
avalanche metrics
adversarial metrics
security non-claim boundary
```

## E. Fusion control branch

Recommended action:

```text
Create FUSION_CONTROL_EQUATION_MAP.md or FUSION object family.
```

Minimum objects:

```text
I(t) diagnostic proxy
P_EM demand
zeta fraction
handoff gate
wall-power latch
margin inequalities
SPARC adaptation
```

---

# Immediate next step

Before claiming full corpus coverage, create a machine-readable table:

```text
paper_id | title | DOI | equation_label_in_paper | equation_text | canonical_id | representation_status | notes
```

Then generate these reports:

```text
1. missing_major_equations.md
2. ambiguous_equation_mappings.md
3. represented_equations_by_paper.md
4. equations_with_no_paper_source.md
5. paper_equation_coverage_totals.json
```

---

# Current audit conclusion

```text
Core WCT mathematical spine: strongly represented
142-object canonical registry: complete as registry
SymPy audit coverage: complete for registry
Lean coverage: partial/formal scaffold
Full 23-paper equation coverage: not yet proven
Most important gap: paper-by-paper equation extraction and mapping
```
