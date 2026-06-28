# The Geometry of Resonance

**Wave Confinement Theory and the Emergence of Mass, Force, and Spacetime**  
**Author:** Richard J. Reyes  
**Initial release:** April 22, 2025  
**Repository:** <https://github.com/rickyjreyes/geometry_of_resonance>

> Unified wave-confinement field theory, simulations, experiments, spectral tests, computation, and applied resonance-control architecture.

---

## Wave Confinement Theory (WCT)

Wave Confinement Theory (WCT) is a geometric field framework in which mass, force, charge-like structure, computation, and effective spacetime geometry are modeled as emergent consequences of confined oscillatory fields.

The central claim is not that geometry is a passive background. The WCT program treats geometry as an output of sustained resonance, curvature feedback, entropy regulation, and topological locking.

Logic chain:

```text
informational constraints
        ↓
boundary / confinement
        ↓
resonance
        ↓
curvature-locked energy
        ↓
mass, force, spectra, and effective geometry
```

---

## Core WCT Objects

The following objects recur across the WCT volume set.

### Curvature-feedback operator

```math
R_\varepsilon(\psi)
=
\frac{\overline{\psi}}
{|\psi|^2+\varepsilon^2e^{-2\alpha|\psi|^2}}
```

```math
\Theta_\varepsilon[\psi]
=-(\Delta\psi)R_\varepsilon(\psi)
```

For `\varepsilon>0`, the modulus-squared denominator is strictly positive. For nonzero `\psi`, the regularized reciprocal approaches `1/\psi` as `\varepsilon\to0`. This removes the historical scalar denominator zero, but it does not establish global PDE existence, uniqueness, regularity, or stability.

### Curvature / torsion scalar

```math
\sigma(s)=\sqrt{\kappa(s)^2+\tau(s)^2}
```

Here \(\kappa\) is curvature, \(\tau\) is torsion, and \(s\) is arclength along a closed curve \(\Gamma\).

### Density-weighted curvature average

```math
\langle f\rangle_w
= \frac{\oint_\Gamma w(s)f(s)\,ds}{\oint_\Gamma w(s)\,ds}
```

Here \(w(s)\) is the linear energy-density weight along the loop.

### Rest-energy / curvature lock

```math
E_{\mathrm{rest}} = mc^2 = \hbar c\,k_{\mathrm{eff}},
\qquad
m=\frac{\hbar}{c}\langle \sigma\rangle_w
```

This is the mass-law spine used by the loop-curvature, solenoidal-mass, electron, and lepton-spectrum papers.

### Finite-band spectral rail

```math
\sigma(k)=r+a|k|^2-b|k|^4,
\qquad a,b>0
```

This selects a finite spectral shell instead of allowing unrestricted IR or UV growth.

---

## Who This Repository Is For

- **Mathematical physicists** — curvature operators, Lyapunov functionals, dimensional bounds, loop-locking variational principles.
- **Field theorists** — emergent gauge-like phase structure, effective metrics, curvature-derived mass laws, phenomenological collider tests.
- **Experimentalists** — optical cavity confinement, photodiode resonance protocols, spectral/log-periodic tests, tokamak control analogs.
- **Computer scientists** — curvature-bounded computation, model-relative complexity, WaveLock nonlinear-PDE one-way functions.
- **AI researchers** — resonance-confinement architecture, coherence regulation, recursive drift audits.

---

## Main Contributions

### 1. Geometry-of-Resonance framework

A covariant wavefield framework in which mass, force, and effective metric structure arise from nonlinear confinement of \(\psi\), rather than being primitive assumptions.

### 2. Phase–Flux Field substrate

A pre-WCT layer defined only by observable energy density \(u(x,t)\), flux \(S(x,t)\), and phase \(\theta(x,t)\), with conservation, causal flow, finite-k band-pass selection, shell quantization, and D4-to-continuum construction.

### 3. Emergent mass from curvature locking

Multiple papers derive or refine the identity that rest energy is determined by effective loop wavenumber / curvature:

```math
m=\frac{\hbar}{c}\left\langle\sqrt{\kappa^2+\tau^2}\right\rangle_w
```

### 4. Conditional spatial-dimensionality criterion

The verified mathematical result is the standard \(H^2\to L^\infty\) Sobolev threshold for integer spatial dimension \(n\leq3\) under the stated domain assumptions. The broader WCT confinement conclusion remains conditional on the \(H^2\)-based stability route being necessary for the admissible confinement mechanism.

### 5. Pattern formation and spectral contraction

Random or broadband fields evolve toward finite-band spectral support, rings, shells, and eigenmode-like structures under Lyapunov descent and Swift–Hohenberg-type dynamics.

### 6. Experimental photon resonance confinement

Water-cavity and photodiode experiments are used to test long-lived resonance states, harmonic mode persistence, and perturbation-induced lock/re-lock behavior.

### 7. Open-data phenomenology

WCT is tested against log-periodic structures in JUNO-style neutrino spectra, LHCb \(B^0\to K^{*0}\mu^+\mu^-\) anomaly data, public NIST atomic line lists, and candidate-spectrum sidebands.

### 8. Computation, cryptography, and AI extensions

The same curvature-locking logic is extended to curvature-bounded computation, model-relative P/NP arguments, WaveLock nonlinear-PDE commitments, and RCA coherence-bounded AI architecture.

---

## Repository Contents

- **Core/** — WCT kernels, curvature operators, evolution logic, constants, precision utilities.
- **Examples/** — Interferometry, cavity evolution, spectral diagnostics, cosmology-style tests.
- **Papers/** — PDF manuscripts corresponding to the Zenodo records below.
- **SM_Spectrum/** — Spectrum-mapping and particle-mass analysis tools.
- **Wavefield Folding/** — Experiments on folding, curvature concentration, and self-organization.
- **Archive/** — Historical code and earlier experiments retained for auditability.

Zenodo should be used for archival citation. GitHub should be used for current code and repository structure.

---

## Quick Start

```bash
git clone https://github.com/rickyjreyes/geometry_of_resonance.git
cd geometry_of_resonance
```

---

## Recommended Reading Order

### Minimal 30–60 minute path

1. **The Geometry of Resonance** — core overview and Lagrangian.  
2. **Phase–Flux Field** — observable substrate and finite-k selection.  
3. **Rest Energy from Density-Weighted Loop Curvature** — clean mass-locking principle.  
4. **Hard Upper Bound on Spatial Dimensionality** — conditional WCT interpretation built on the verified \(H^2\to L^\infty\) threshold.  
5. **Observation of Long-Lived Photon Resonance Confinement in Water Cavities** — experimental anchor.

### Mathematical backbone

1. **Phase–Flux Field**  
2. **The Geometry of Resonance**  
3. **Rest Energy from Density-Weighted Loop Curvature**  
4. **Emergence of Effective Mass**  
5. **Hard Upper Bound on Spatial Dimensionality**  
6. **Self-Emergent Fourier Cymatics**  
7. **Logarithmic Curvature Flow**  
8. **Koide Mass Relation**

### Empirical / falsification path

1. **Observation of Long-Lived Photon Resonance Confinement in Water Cavities**  
2. **Prediction & Protocol Ledger**  
3. **JUNO Energy Resolution and Detectability of WCT Ghost-Mode Neutrinos**  
4. **C9(q²) / LHCb angular anomaly paper**  
5. **Open-data B⁰ → K*⁰ μ⁺μ⁻ candidate spectra**  
6. **Bin-Stable Log-Periodic Structure in Public NIST Atomic Line List**

### Computation / AI / applied path

1. **Discrete Wave-Constrained Computation and Classical Complexity**  
2. **P vs NP in Curvature-Bounded Wave Computation**  
3. **The Classical P vs NP Problem Is Mathematically and Physically Ill-Posed**  
4. **WaveLock**  
5. **Resonance-Confinement Architecture**  
6. **Recursive AI Drift**

---

## Claim-Status Key

Use this key when reading the corpus:

| Status | Meaning |
|---|---|
| **Core theory** | Defines the WCT operator, field ontology, or master equations. |
| **Derivation** | Produces a symbolic or variational result from prior WCT definitions. |
| **Mathematical closure** | Addresses well-posedness, stability, uniqueness, or dimensional bounds. |
| **Simulation** | Numerically tests WCT dynamics or spectral behavior. |
| **Experiment / protocol** | Physical test, lab protocol, or reproducible measurement path. |
| **Phenomenology** | Applies WCT to public physics datasets or known anomalies. |
| **Architecture / application** | Applies WCT principles to computation, AI, cryptography, or control. |

---

## Citations and Zenodo Releases

### Primary WCT foundations

[1] R. J. Reyes, **“The Geometry of Resonance: Wave Confinement Theory and the Emergence of Mass, Force, and Spacetime,”** Apr. 22, 2025, Zenodo.  
DOI: [10.5281/zenodo.15644222](https://doi.org/10.5281/zenodo.15644222)  
Record: <https://zenodo.org/records/15644222>

[2] R. J. Reyes, **“Structure and Derivation of Physical Constants through Wave Confinement,”** Apr. 26, 2025, Zenodo.  
DOI: [10.5281/zenodo.15596159](https://doi.org/10.5281/zenodo.15596159)  
Record: <https://zenodo.org/records/15596159>

[7] R. J. Reyes, **“Phase-Flux Field (PFF): Axiomatic Substrate for Wave Confinement Theory Zero-Wave Invariance, Finite-k Lyapunov Band-Pass, Shell Quantization, and D4 to Continuum,”** Sep. 08, 2025, Zenodo.  
DOI: [10.5281/zenodo.17578766](https://doi.org/10.5281/zenodo.17578766)  
Record: <https://zenodo.org/records/17578766>

### Mass, geometry, dimensionality, and spectral dynamics

[6] R. J. Reyes, **“Hard Upper Bound on Spatial Dimensionality in Wave Confinement Theory,”** Aug. 13, 2025, Zenodo.  
DOI: [10.5281/zenodo.17081283](https://doi.org/10.5281/zenodo.17081283)  
Record: <https://zenodo.org/records/17081283>

[8] R. J. Reyes, **“Self-Emergent Fourier Cymatics: Entropic Eigenmodes out of Chaos,”** Sep. 16, 2025, Zenodo.  
DOI: [10.5281/zenodo.17732648](https://doi.org/10.5281/zenodo.17732648)  
Record: <https://zenodo.org/records/17732648>

[9] R. J. Reyes, **“Emergence of Effective Mass: Solenoidal Topology of Vibrational Energy,”** Oct. 27, 2025, Zenodo.  
DOI: [10.5281/zenodo.17459463](https://doi.org/10.5281/zenodo.17459463)  
Record: <https://zenodo.org/records/17459463>

[10] R. J. Reyes, **“Rest Energy from Density-Weighted Loop Curvature: A Covariant Locking Principle,”** Nov. 11, 2025, Zenodo.  
DOI: [10.5281/zenodo.20533537](https://doi.org/10.5281/zenodo.20533537)  
Record: <https://zenodo.org/records/20533537>

[14] R. J. Reyes, **“Wave Confinement Theory Predicts the Koide Mass Relation: A Curvature–Harmonic Origin of Fermion Mass Triplets,”** Dec. 10, 2025, Zenodo.  
DOI: [10.5281/zenodo.17887562](https://doi.org/10.5281/zenodo.17887562)  
Record: <https://zenodo.org/records/17887562>

[16] R. J. Reyes, **“Logarithmic Curvature Flow, Filament Localization, and the Geometric Origin of the Lepton Mass Spectrum,”** Mar. 10, 2026, Zenodo.  
DOI: [10.5281/zenodo.18936949](https://doi.org/10.5281/zenodo.18936949)  
Record: <https://zenodo.org/records/18936949>

### Computation, complexity, AI, and cryptography

[3] R. J. Reyes, **“P vs NP in Curvature-Bounded Wave Computation: A Model-Relative P_WCC ≠ NP_WCC Separation,”** May 07, 2025, Zenodo.  
DOI: [10.5281/zenodo.17743607](https://doi.org/10.5281/zenodo.17743607)  
Record: <https://zenodo.org/records/17743607>

[5] R. J. Reyes, **“Resonance-Confinement Architecture: A Physically Bounded Substrate for Safe Superintelligence,”** Jun. 11, 2025, Zenodo.  
DOI: [10.5281/zenodo.17732661](https://doi.org/10.5281/zenodo.17732661)  
Record: <https://zenodo.org/records/17732661>

[12] R. J. Reyes, **“Discrete Wave-Constrained Computation and Classical Complexity: Turing Equivalence for 𝐏 and 𝐍𝐏,”** Nov. 26, 2025, Zenodo.  
DOI: [10.5281/zenodo.17732642](https://doi.org/10.5281/zenodo.17732642)  
Record: <https://zenodo.org/records/17732642>

[13] R. J. Reyes, **“The Classical P vs NP Problem is Mathematically and Physically Ill-Posed,”** Dec. 01, 2025, Zenodo.  
DOI: [10.5281/zenodo.17783074](https://doi.org/10.5281/zenodo.17783074)  
Record: <https://zenodo.org/records/17783074>

[20] R. J. Reyes, **“Recursive AI Drift: A 2025 Prediction Timeline External Validation Audit and Technical Note,”** May 2026, Zenodo.  
DOI: [10.5281/zenodo.20142976](https://doi.org/10.5281/zenodo.20142976)  
Record: <https://zenodo.org/records/20142976>

[22] R. J. Reyes, **“WaveLock: A Curvature-Locked One-Way Function Based on Nonlinear PDE Evolution,”** Dec. 01, 2025, Zenodo.  
DOI: [10.5281/zenodo.19122146](https://doi.org/10.5281/zenodo.19122146)  
Record: <https://zenodo.org/records/19122146>  



### Experiment, protocol, and physical systems

[4] R. J. Reyes, **“Observation of Long-Lived Photon Resonance Confinement in Water Cavities,”** May 17, 2025, Zenodo.  
DOI: [10.5281/zenodo.17206381](https://doi.org/10.5281/zenodo.17206381)  
Record: <https://zenodo.org/records/17206381>

[11] R. J. Reyes, **“JUNO Energy Resolution and Detectability of WCT Ghost-Mode Neutrinos,”** Nov. 20, 2025, Zenodo.  
DOI: [10.5281/zenodo.17715872](https://doi.org/10.5281/zenodo.17715872)  
Record: <https://zenodo.org/records/17715872>

[15] R. J. Reyes, **“Prediction & Protocol Ledger: Long-Lived Harmonic State Induction in Photodiodes,”** Dec. 2025, Zenodo.  
DOI: [10.5281/zenodo.17957713](https://doi.org/10.5281/zenodo.17957713)  
Record: <https://zenodo.org/records/17957713>

[17] R. J. Reyes, **“Nuclear Fusion Tokamak with Self Sustaining Resonance,”** Apr. 14, 2026, Zenodo.  
DOI: [10.5281/zenodo.19578185](https://doi.org/10.5281/zenodo.19578185)  
Record: <https://zenodo.org/records/19578185>

### Collider, atomic-line, and open-data phenomenology

[18] R. J. Reyes, **“A Curvature-Induced Log-Periodic Deformation of C9(q²): Wave Confinement Theory and the LHCb B⁰ → K*⁰ μ⁺μ⁻ Anomaly,”** Apr. 23, 2026, Zenodo.  
DOI: [10.5281/zenodo.19705254](https://doi.org/10.5281/zenodo.19705254)  
Record: <https://zenodo.org/records/19705254>

[19] R. J. Reyes, **“Log-Spectral Structure and Koide-Like Winding Geometry in Open-Data B⁰ → K*⁰ μ⁺μ⁻ Candidate Spectra,”** May 09, 2026, Zenodo.  
DOI: [10.5281/zenodo.20164333](https://doi.org/10.5281/zenodo.20164333)  
Record: <https://zenodo.org/records/20164333>

[21] R. J. Reyes, **“Bin-Stable Log-Periodic Structure in Public NIST Atomic Line List,”** May 28, 2026, Zenodo.  
DOI: [10.5281/zenodo.20435463](https://doi.org/10.5281/zenodo.20435463)  
Record: <https://zenodo.org/records/20435463>

---

## Full Reference List

1. R. J. Reyes, “The Geometry of Resonance: Wave Confinement Theory and the Emergence of Mass, Force, and Spacetime,” Apr. 22, 2025, Zenodo. doi: 10.5281/zenodo.15644222.
2. R. J. Reyes, “Structure and Derivation of Physical Constants through Wave Confinement,” Apr. 26, 2025, Zenodo. doi: 10.5281/zenodo.15596159.
3. R. J. Reyes, “P vs NP in Curvature-Bounded Wave Computation: A Model-Relative P_WCC ≠ NP_WCC Separation,” May 07, 2025, Zenodo. doi: 10.5281/zenodo.17743607.
4. R. J. Reyes, “Observation of Long-Lived Photon Resonance Confinement in Water Cavities,” May 17, 2025, Zenodo. doi: 10.5281/zenodo.17206381.
5. R. J. Reyes, “Resonance-Confinement Architecture: A Physically Bounded Substrate for Safe Superintelligence,” Jun. 11, 2025, Zenodo. doi: 10.5281/zenodo.17732661.
6. R. J. Reyes, “Hard Upper Bound on Spatial Dimensionality in Wave Confinement Theory,” Aug. 13, 2025, Zenodo. doi: 10.5281/zenodo.17081283.
7. R. J. Reyes, “Phase-Flux Field (PFF): Axiomatic Substrate for Wave Confinement Theory Zero-Wave Invariance, Finite-k Lyapunov Band-Pass, Shell Quantization, and D4 to Continuum,” Sep. 08, 2025, Zenodo. doi: 10.5281/zenodo.17578766.
8. R. J. Reyes, “Self-Emergent Fourier Cymatics: Entropic Eigenmodes out of Chaos,” Sep. 16, 2025, Zenodo. doi: 10.5281/zenodo.17732648.
9. R. J. Reyes, “Emergence of Effective Mass: Solenoidal Topology of Vibrational Energy,” Oct. 27, 2025, Zenodo. doi: 10.5281/zenodo.17459463.
10. R. J. Reyes, “Rest Energy from Density-Weighted Loop Curvature: A Covariant Locking Principle,” Nov. 11, 2025, Zenodo. doi: 10.5281/zenodo.20533537.
11. R. J. Reyes, “JUNO Energy Resolution and Detectability of WCT Ghost-Mode Neutrinos,” Nov. 20, 2025, Zenodo. doi: 10.5281/zenodo.17715872.
12. R. J. Reyes, “Discrete Wave-Constrained Computation and Classical Complexity: Turing Equivalence for 𝐏 and 𝐍𝐏,” Nov. 26, 2025, Zenodo. doi: 10.5281/zenodo.17732642.
13. R. J. Reyes, “The Classical P vs NP Problem is Mathematically and Physically Ill-Posed,” Dec. 01, 2025, Zenodo. doi: 10.5281/zenodo.17783074.
14. R. J. Reyes, “Wave Confinement Theory Predicts the Koide Mass Relation: A Curvature–Harmonic Origin of Fermion Mass Triplets,” Dec. 10, 2025, Zenodo. doi: 10.5281/zenodo.17887562.
15. R. J. Reyes, “Prediction & Protocol Ledger: Long-Lived Harmonic State Induction in Photodiodes,” Dec. 2025, Zenodo. doi: 10.5281/zenodo.17957713.
16. R. J. Reyes, “Logarithmic Curvature Flow, Filament Localization, and the Geometric Origin of the Lepton Mass Spectrum,” Mar. 10, 2026, Zenodo. doi: 10.5281/zenodo.18936949.
17. R. J. Reyes, “Nuclear Fusion Tokamak with Self Sustaining Resonance,” Apr. 14, 2026, Zenodo. doi: 10.5281/zenodo.19578185.
18. R. J. Reyes, “A Curvature-Induced Log-Periodic Deformation of C9(q²): Wave Confinement Theory and the LHCb B⁰ → K*⁰ μ⁺μ⁻ Anomaly,” Apr. 23, 2026, Zenodo. doi: 10.5281/zenodo.19705254.
19. R. J. Reyes, “Log-Spectral Structure and Koide-Like Winding Geometry in Open-Data B⁰ → K*⁰ μ⁺μ⁻ Candidate Spectra,” May 09, 2026, Zenodo. doi: 10.5281/zenodo.20164333.
20. R. J. Reyes, “Recursive AI Drift: A 2025 Prediction Timeline External Validation Audit and Technical Note,” May 2026, Zenodo. doi: 10.5281/zenodo.20142976.
21. R. J. Reyes, “Bin-Stable Log-Periodic Structure in Public NIST Atomic Line List,” May 28, 2026, Zenodo. doi: 10.5281/zenodo.20435463.
22. R. J. Reyes, “WaveLock: A Curvature-Locked One-Way Function Based on Nonlinear PDE Evolution,” Dec. 01, 2025, Zenodo. doi: 10.5281/zenodo.19122146.

---

## Keywords

Wave confinement · emergent mass · nonlinear curvature · curvature locking · phase–flux field · topological resonance · covariant field theory · entropy–coherence coupling · finite-band instability · rest energy from curvature · spatial dimensionality bound · Koide relation · log-periodic spectra · curvature-regulated computation · WaveLock · resonance-based AI

---

## Quick Links

- [Geometry of Resonance — Full Paper](https://zenodo.org/records/15644222)
- [Physical Constants through Wave Confinement](https://zenodo.org/records/15596159)
- [GitHub Repository](https://github.com/rickyjreyes/geometry_of_resonance)
