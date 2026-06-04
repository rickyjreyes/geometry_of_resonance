# Wave Confinement Theory – Reading Map

Practical guide to the WCT volume set:  
**what to read, in what order, for which goal.**

This map is organized by **claim type**, not only by date. The goal is to prevent all papers from appearing equal-weight: foundational definitions, mathematical closure, empirical tests, computation, AI, engineering, and open-data phenomenology have different proof status.

---

## 0. If you only have 30–60 minutes

Start here:

1. **The Geometry of Resonance: Wave Confinement Theory and the Emergence of Mass, Force, and Spacetime**  
   *[Zenodo 15644222](https://zenodo.org/records/15644222) – core overview and Lagrangian*  
   DOI: [10.5281/zenodo.15644222](https://doi.org/10.5281/zenodo.15644222)  
   - Role: main WCT overview; curvature-feedback Lagrangian; emergence of mass, force, and effective geometry.  
   - How to read: abstract → introduction → main equations → figures → conclusion.

2. **Phase-Flux Field (PFF): Axiomatic Substrate for Wave Confinement Theory**  
   *[Zenodo 17578766](https://zenodo.org/records/17578766) – zero-wave substrate and finite-k rail*  
   DOI: [10.5281/zenodo.17578766](https://doi.org/10.5281/zenodo.17578766)  
   - Role: defines the pre-WCT substrate: energy density `u`, flux `S`, phase `θ`, conservation, shell quantization, and finite-k Lyapunov band-pass.  
   - How to read: focus on axioms, zero-wave state, band-pass evolution, and shell closure.

3. **Rest Energy from Density-Weighted Loop Curvature: A Covariant Locking Principle**  
   *[Zenodo 20533537](https://zenodo.org/records/20533537) – precise mass-locking paper*  
   DOI: [10.5281/zenodo.20533537](https://doi.org/10.5281/zenodo.20533537)  
   - Role: cleanest rest-energy statement: closed-loop curvature/torsion → effective wavenumber → rest mass.  
   - How to read: definitions → locking action → Euler–Lagrange result → mass law.

4. **Hard Upper Bound on Spatial Dimensionality in Wave Confinement Theory**  
   *[Zenodo 17081283](https://zenodo.org/records/17081283) – dimensional stability bound*  
   DOI: [10.5281/zenodo.17081283](https://doi.org/10.5281/zenodo.17081283)  
   - Role: explains why stable curvature-locked confinement is restricted to `n ≤ 3`.  
   - How to read: Sobolev bound, Lyapunov scaling, entropy/topology routes, final convergence table.

---

## 1. Foundations — canonical WCT spine

Read in this order:

1. **Phase-Flux Field (PFF): Axiomatic Substrate for Wave Confinement Theory Zero-Wave Invariance, Finite-k Lyapunov Band-Pass, Shell Quantization, and D4 to Continuum**  
   *Sep. 08, 2025 — Zenodo 17578766*  
   DOI: [10.5281/zenodo.17578766](https://doi.org/10.5281/zenodo.17578766)  
   - Claim type: axiomatic substrate.  
   - Role: starts below WCT using only observables `(u, S, θ)`, conservation, causal/flux constraints, winding, and band-pass selection.  
   - Look for: zero-wave state, finite-k spectral rail, D4 primer, continuum limit.

2. **The Geometry of Resonance: Wave Confinement Theory and the Emergence of Mass, Force, and Spacetime**  
   *Apr. 22, 2025 — Zenodo 15644222*  
   DOI: [10.5281/zenodo.15644222](https://doi.org/10.5281/zenodo.15644222)  
   - Claim type: core theory overview.  
   - Role: introduces the WCT curvature-feedback Lagrangian, regularized curvature operator, mass/force/geometry interpretation, and effective metric proposal.  
   - Look for: `Θ[ψ]`, `L_WCT`, curvature/entropy balance, effective metric, simulation logic.

3. **Structure and Derivation of Physical Constants through Wave Confinement**  
   *Apr. 26, 2025 — Zenodo 15596159*  
   DOI: [10.5281/zenodo.15596159](https://doi.org/10.5281/zenodo.15596159)  
   - Claim type: constants ansatz / structural derivation.  
   - Role: maps confinement parameters and harmonic structure toward physical constants.  
   - Look for: ξ, σ, β, θ, harmonic mass structure, assumptions behind fitted/derived constants.

4. **Self-Emergent Fourier Cymatics: Entropic Eigenmodes out of Chaos**  
   *Sep. 16, 2025 — Zenodo 17732648*  
   DOI: [10.5281/zenodo.17732648](https://doi.org/10.5281/zenodo.17732648)  
   - Claim type: dynamical/numerical support.  
   - Role: shows random seed → spectral annulus → finite support → eigenmode-like patterns under Lyapunov descent.  
   - Look for: spectral entropy decrease, finite support `K*`, Swift–Hohenberg reduction, curvature-induced gap.

Recommended reading style:

- First pass: abstract, introduction, definitions, main equations, all figures.  
- Second pass: derivations, proofs, and appendices only when implementing or auditing.

---

## 2. Mass, Geometry, and Dimensional Stability

After PFF + Geometry, read:

5. **Rest Energy from Density-Weighted Loop Curvature: A Covariant Locking Principle**  
   *Nov. 11, 2025 — Zenodo 20533537*  
   DOI: [10.5281/zenodo.20533537](https://doi.org/10.5281/zenodo.20533537)  
   - Claim type: mathematical derivation / variational locking.  
   - Role: precise rest-energy law `E_rest = ℏc k_eff`, with `k_eff = ⟨σ⟩_w`, where `σ = sqrt(κ² + τ²)`.  
   - Look for: loop geometry, winding, density weighting, locking action, stationarity, uniqueness, mislock bounds.

6. **Emergence of Effective Mass: Solenoidal Topology of Vibrational Energy**  
   *Oct. 27, 2025 — Zenodo 17459463*  
   DOI: [10.5281/zenodo.17459463](https://doi.org/10.5281/zenodo.17459463)  
   - Claim type: geometric mass model.  
   - Role: connects curved photon-like/solenoidal paths, curvature, torsion, effective refractive index, phase delay, and inertial behavior.  
   - Look for: solenoidal mass law, helix/circle worked examples, SU(2)/U(1) links, ppm-scale cavity prediction.

7. **Hard Upper Bound on Spatial Dimensionality in Wave Confinement Theory**  
   *Aug. 13, 2025 — Zenodo 17081283*  
   DOI: [10.5281/zenodo.17081283](https://doi.org/10.5281/zenodo.17081283)  
   - Claim type: mathematical stability bound.  
   - Role: argues that stable WCT confinement fails for `n > 3`.  
   - Look for: Sobolev embedding `H²(Rⁿ) → L∞`, Lyapunov scaling, entropy divergence, topological unlinking, feedback divergence.

8. **Logarithmic Curvature Flow, Filament Localization, and the Geometric Origin of the Lepton Mass Spectrum**  
   *Mar. 10, 2026 — Zenodo 18936949*  
   DOI: [10.5281/zenodo.18936949](https://doi.org/10.5281/zenodo.18936949)  
   - Claim type: mathematical reduction + mass-spectrum model.  
   - Role: rewrites the curvature operator through `u = ln ψ`, obtains viscous Hamilton–Jacobi/Cole–Hopf structure, then links filament curvature to lepton mass geometry.  
   - Look for: logarithmic transform, diffusion equivalence, topology requirement, filament/toroidal spectrum, Koide geometry.

9. **Wave Confinement Theory Predicts the Koide Mass Relation: A Curvature–Harmonic Origin of Fermion Mass Triplets**  
   *Dec. 10, 2025 — Zenodo 17887562*  
   DOI: [10.5281/zenodo.17887562](https://doi.org/10.5281/zenodo.17887562)  
   - Claim type: mass-triplet phenomenology / geometric derivation.  
   - Role: maps curvature harmonics and spin-dependent geometry to Koide-like mass ratios.  
   - Look for: `Q(s)`, curvature harmonic `K(s)`, charged-lepton `Q = 2/3`, effective-spin caveats.

If the question is **“is WCT a viable mass/geometry story?”**, this section is the critical path.

---

## 3. Dynamics, Patterns, and Cavity Physics

Once the mass/geometry picture is clear, move to dynamics and experiments:

10. **Observation of Long-Lived Photon Resonance Confinement in Water Cavities**  
    *May 17, 2025 — Zenodo 17206381*  
    DOI: [10.5281/zenodo.17206381](https://doi.org/10.5281/zenodo.17206381)  
    - Claim type: experimental report.  
    - Role: reports long-lived photonic/cavity resonance behavior in chilled water cavities.  
    - Look for: FFT structure, persistence after blocking, perturbation/re-lock behavior, controls, repeatability limits.

11. **Prediction & Protocol Ledger: Long-Lived Harmonic State Induction in Photodiodes**  
    *Dec. 2025 — Zenodo 17957713*  
    DOI: [10.5281/zenodo.17957713](https://doi.org/10.5281/zenodo.17957713)  
    - Claim type: protocol / prediction ledger.  
    - Role: records the experimental protocol and prediction ledger for long-lived harmonic induction tests.  
    - Look for: dated predictions, protocol details, pass/fail conditions, reproducibility checklist.

12. **Self-Emergent Fourier Cymatics: Entropic Eigenmodes out of Chaos**  
    *Sep. 16, 2025 — Zenodo 17732648*  
    DOI: [10.5281/zenodo.17732648](https://doi.org/10.5281/zenodo.17732648)  
    - Claim type: simulation / numerical proof-of-concept.  
    - Role: bridges random-field dynamics to spectral selection and locked eigenmodes.  
    - Look for: spectral collapse, Lyapunov monotonicity, finite-k mode selection.

If you are an **experimentalist**, the fast path is: Geometry → PFF → Water Cavities → Protocol Ledger → Cymatics.

---

## 4. Open-Data / Phenomenology Tests

This section contains WCT-motivated tests against public or detector-facing data. Treat these as **phenomenological probes**, not final discoveries.

13. **JUNO Energy Resolution and Detectability of WCT Ghost-Mode Neutrinos**  
    *Nov. 20, 2025 — Zenodo 17715872*  
    DOI: [10.5281/zenodo.17715872](https://doi.org/10.5281/zenodo.17715872)  
    - Claim type: detectability bound / experimental forecasting.  
    - Role: tests whether WCT ghost-mode modulations can survive JUNO resolution and systematic smearing.  
    - Look for: log-energy modulation ansatz, Gaussian smearing, amplitude damping, resolvability inequalities.

14. **A Curvature-Induced Log-Periodic Deformation of C9(q²): Wave Confinement Theory and the LHCb B⁰ → K*⁰ μ⁺μ⁻ Anomaly**  
    *Apr. 23, 2026 — Zenodo 19705254*  
    DOI: [10.5281/zenodo.19705254](https://doi.org/10.5281/zenodo.19705254)  
    - Claim type: phenomenological collider ansatz.  
    - Role: tests whether a WCT log-periodic deformation can improve over a constant `δC9` model.  
    - Look for: `δC9(q²)` ansatz, log-frequency scan, null/shuffle test, covariance limitations.

15. **Log-Spectral Structure and Koide-Like Winding Geometry in Open-Data B⁰ → K*⁰ μ⁺μ⁻ Candidate Spectra**  
    *May 09, 2026 — Zenodo 20164333*  
    DOI: [10.5281/zenodo.20164333](https://doi.org/10.5281/zenodo.20164333)  
    - Claim type: open-data spectral search.  
    - Role: analyzes candidate spectra for log-domain residual structure, active-domain winding, and Koide-like ratios.  
    - Look for: KDE baseline repair, veto stress tests, active-domain `n`, sideband/signal dilation, non-discovery caveats.

16. **Bin-Stable Log-Periodic Structure in Public NIST Atomic Line List**  
    *May 28, 2026 — Zenodo 20435463*  
    DOI: [10.5281/zenodo.20435463](https://doi.org/10.5281/zenodo.20435463)  
    - Claim type: public-dataset spectral test.  
    - Role: searches NIST atomic line data for bin-stable log-periodic structure.  
    - Look for: bin stability, null tests, variable stress, correction for look-elsewhere effects.

---

## 5. Computation and Complexity

For readers with a CS / complexity focus:

17. **Discrete Wave-Constrained Computation and Classical Complexity: Turing Equivalence for P and NP**  
    *Nov. 26, 2025 — Zenodo 17732642*  
    DOI: [10.5281/zenodo.17732642](https://doi.org/10.5281/zenodo.17732642)  
    - Claim type: model equivalence.  
    - Role: defines a discrete wave-constrained computation model and relates it back to classical `P` and `NP`.  
    - Suggested path: model definition → simulation/encoding rules → equivalence statement.

18. **P vs NP in Curvature-Bounded Wave Computation: A Model-Relative P_WCC ≠ NP_WCC Separation**  
    *May 07, 2025 — Zenodo 17743607*  
    DOI: [10.5281/zenodo.17743607](https://doi.org/10.5281/zenodo.17743607)  
    - Claim type: model-relative separation.  
    - Role: argues for separation inside curvature-bounded WCC rather than a direct unrestricted classical proof.  
    - Suggested path: WCC model → curvature bound → separation argument → relation to classical `P`/`NP`.

19. **The Classical P vs NP Problem is Mathematically and Physically Ill-Posed**  
    *Dec. 01, 2025 — Zenodo 17783074*  
    DOI: [10.5281/zenodo.17783074](https://doi.org/10.5281/zenodo.17783074)  
    - Claim type: philosophical / formal critique.  
    - Role: argues that the classical problem omits physical resource constraints and should be reframed through realizable computation.  
    - Suggested path: read after the two WCC papers, not before.

20. **WaveLock: A Curvature-Locked One-Way Function Based on Nonlinear PDE Evolution**  
    *Dec. 01, 2025 — Zenodo 19122146*  
    DOI: [10.5281/zenodo.19122146](https://doi.org/10.5281/zenodo.19122146)  
    - Claim type: empirical cryptographic research artifact.  
    - Role: explores one-way behavior from nonlinear PDE contraction and curvature-locked evolution.  
    - Look for: construction, adversarial test suite, avalanche behavior, explicit non-claim of formal security.  

This section is logically downstream of WCT but can be read independently after Geometry + PFF.

---

## 6. AI, Architecture, and Prediction Audit

For AI / alignment / AGI architecture readers:

21. **Resonance-Confinement Architecture: A Physically Bounded Substrate for Safe Superintelligence**  
    *Jun. 11, 2025 — Zenodo 17732661*  
    DOI: [10.5281/zenodo.17732661](https://doi.org/10.5281/zenodo.17732661)  
    - Claim type: AI architecture proposal.  
    - Role: maps WCT principles into an AI substrate governed by coherence, bounded curvature, contradiction control, and Lyapunov-like stabilization.  
    - Reading tip: treat it as an architecture specification, not as a completed AGI implementation.

22. **Recursive AI Drift: A 2025 Prediction Timeline External Validation Audit and Technical Note**  
    *May 2026 — Zenodo 20142976*  
    DOI: [10.5281/zenodo.20142976](https://doi.org/10.5281/zenodo.20142976)  
    - Claim type: prediction audit / technical note.  
    - Role: audits earlier AI-drift predictions against external events and model behavior.  
    - Look for: dated claims, external validation logic, prediction-status table, failure cases.

---

## 7. Applied Engineering / Fusion Control

23. **Nuclear Fusion Tokamak with Self Sustaining Resonance**  
    *Apr. 14, 2026 — Zenodo 19578185*  
    DOI: [10.5281/zenodo.19578185](https://doi.org/10.5281/zenodo.19578185)  
    - Claim type: applied control architecture.  
    - Role: uses WCT-style confinement/coherence logic as a macroscopic control analogy for tokamak sustainment.  
    - Look for: diagnostics proxy `I`, margin gates, virtual harvest fraction, handoff gate, Monte Carlo/stress tests.  
    - Important reading note: distinguish **accounting latch** from true physical wall-power removal unless harvest hardware exists.

Read this only after the core WCT concepts are clear, because it is an applied engineering branch rather than a foundation paper.

---

## 8. Recommended Paths by Goal

### A. Fast conceptual overview

1. Geometry of Resonance  
2. Phase-Flux Field  
3. Rest Energy from Density-Weighted Loop Curvature  
4. Hard Upper Bound on Spatial Dimensionality

### B. Mathematical audit path

1. Phase-Flux Field  
2. Rest Energy from Density-Weighted Loop Curvature  
3. Hard Upper Bound on Spatial Dimensionality  
4. Logarithmic Curvature Flow  
5. Koide Mass Relation  
6. Discrete Wave-Constrained Computation

### C. Experimental / falsification path

1. Geometry of Resonance  
2. Observation of Long-Lived Photon Resonance Confinement in Water Cavities  
3. Prediction & Protocol Ledger  
4. JUNO Ghost-Mode Detectability  
5. LHCb `C9(q²)` Log-Periodic Deformation  
6. B⁰ → K*⁰ Candidate Spectra  
7. NIST Atomic Line List

### D. Computation / cryptography path

1. Geometry of Resonance  
2. Phase-Flux Field  
3. Discrete Wave-Constrained Computation  
4. P vs NP in Curvature-Bounded WCC  
5. Classical P vs NP Is Ill-Posed  
6. WaveLock

### E. AI / AGI path

1. Geometry of Resonance  
2. Phase-Flux Field  
3. Self-Emergent Fourier Cymatics  
4. Resonance-Confinement Architecture  
5. Recursive AI Drift Audit

---

## 9. Full DOI Reference List

[1] R. J. Reyes, “The Geometry of Resonance: Wave Confinement Theory and the Emergence of Mass, Force, and Spacetime,” Apr. 22, 2025, Zenodo. doi: [10.5281/zenodo.15644222](https://doi.org/10.5281/zenodo.15644222).

[2] R. J. Reyes, “Structure and Derivation of Physical Constants through Wave Confinement,” Apr. 26, 2025, Zenodo. doi: [10.5281/zenodo.15596159](https://doi.org/10.5281/zenodo.15596159).

[3] R. J. Reyes, “P vs NP in Curvature-Bounded Wave Computation: A Model-Relative P_WCC ≠ NP_WCC Separation,” May 07, 2025, Zenodo. doi: [10.5281/zenodo.17743607](https://doi.org/10.5281/zenodo.17743607).

[4] R. J. Reyes, “Observation of Long-Lived Photon Resonance Confinement in Water Cavities,” May 17, 2025, Zenodo. doi: [10.5281/zenodo.17206381](https://doi.org/10.5281/zenodo.17206381).

[5] R. J. Reyes, “Resonance-Confinement Architecture: A Physically Bounded Substrate for Safe Superintelligence,” Jun. 11, 2025, Zenodo. doi: [10.5281/zenodo.17732661](https://doi.org/10.5281/zenodo.17732661).

[6] R. J. Reyes, “Hard Upper Bound on Spatial Dimensionality in Wave Confinement Theory,” Aug. 13, 2025, Zenodo. doi: [10.5281/zenodo.17081283](https://doi.org/10.5281/zenodo.17081283).

[7] R. J. Reyes, “Phase-Flux Field (PFF): Axiomatic Substrate for Wave Confinement Theory Zero-Wave Invariance, Finite-k Lyapunov Band-Pass, Shell Quantization, and D4 to Continuum,” Sep. 08, 2025, Zenodo. doi: [10.5281/zenodo.17578766](https://doi.org/10.5281/zenodo.17578766).

[8] R. J. Reyes, “Self-Emergent Fourier Cymatics: Entropic Eigenmodes out of Chaos,” Sep. 16, 2025, Zenodo. doi: [10.5281/zenodo.17732648](https://doi.org/10.5281/zenodo.17732648).

[9] R. J. Reyes, “Emergence of Effective Mass: Solenoidal Topology of Vibrational Energy,” Oct. 27, 2025, Zenodo. doi: [10.5281/zenodo.17459463](https://doi.org/10.5281/zenodo.17459463).

[10] R. J. Reyes, “Rest Energy from Density-Weighted Loop Curvature: A Covariant Locking Principle,” Nov. 11, 2025, Zenodo. doi: [10.5281/zenodo.20533537](https://doi.org/10.5281/zenodo.20533537).

[11] R. J. Reyes, “JUNO Energy Resolution and Detectability of WCT Ghost-Mode Neutrinos,” Nov. 20, 2025, Zenodo. doi: [10.5281/zenodo.17715872](https://doi.org/10.5281/zenodo.17715872).

[12] R. J. Reyes, “Discrete Wave-Constrained Computation and Classical Complexity: Turing Equivalence for P and NP,” Nov. 26, 2025, Zenodo. doi: [10.5281/zenodo.17732642](https://doi.org/10.5281/zenodo.17732642).

[13] R. J. Reyes, “The Classical P vs NP Problem is Mathematically and Physically Ill-Posed,” Dec. 01, 2025, Zenodo. doi: [10.5281/zenodo.17783074](https://doi.org/10.5281/zenodo.17783074).

[14] R. J. Reyes, “Wave Confinement Theory Predicts the Koide Mass Relation: A Curvature–Harmonic Origin of Fermion Mass Triplets,” Dec. 10, 2025, Zenodo. doi: [10.5281/zenodo.17887562](https://doi.org/10.5281/zenodo.17887562).

[15] R. J. Reyes, “Prediction & Protocol Ledger: Long-Lived Harmonic State Induction in Photodiodes,” Dec. 2025. doi: [10.5281/zenodo.17957713](https://doi.org/10.5281/zenodo.17957713).

[16] R. J. Reyes, “Logarithmic Curvature Flow, Filament Localization, and the Geometric Origin of the Lepton Mass Spectrum,” Mar. 10, 2026, Zenodo. doi: [10.5281/zenodo.18936949](https://doi.org/10.5281/zenodo.18936949).

[17] R. J. Reyes, “Nuclear Fusion Tokamak with Self Sustaining Resonance,” Apr. 14, 2026, Zenodo. doi: [10.5281/zenodo.19578185](https://doi.org/10.5281/zenodo.19578185).

[18] R. J. Reyes, “A Curvature-Induced Log-Periodic Deformation of C9(q²): Wave Confinement Theory and the LHCb B⁰ → K*⁰ μ⁺μ⁻ Anomaly,” Apr. 23, 2026, Zenodo. doi: [10.5281/zenodo.19705254](https://doi.org/10.5281/zenodo.19705254).

[19] R. J. Reyes, “Log-Spectral Structure and Koide-Like Winding Geometry in Open-Data B⁰ → K*⁰ μ⁺μ⁻ Candidate Spectra,” May 09, 2026, Zenodo. doi: [10.5281/zenodo.20164333](https://doi.org/10.5281/zenodo.20164333).

[20] R. J. Reyes, “Recursive AI Drift: A 2025 Prediction Timeline External Validation Audit and Technical Note,” May 2026. doi: [10.5281/zenodo.20142976](https://doi.org/10.5281/zenodo.20142976).

[21] R. J. Reyes, “Bin-Stable Log-Periodic Structure in Public NIST Atomic Line List,” May 28, 2026, Zenodo. doi: [10.5281/zenodo.20435463](https://doi.org/10.5281/zenodo.20435463).

[22] R. J. Reyes, “WaveLock: A Curvature-Locked One-Way Function Based on Nonlinear PDE Evolution,” Dec. 01, 2025, Zenodo. doi: [10.5281/zenodo.19122146](https://doi.org/10.5281/zenodo.19122146).

---

## 10. How to Map Papers to This Repository

Use this map as both a reading order and a repository lookup table.

Suggested repository grouping:

- **`Papers/Core/`**  
  Geometry of Resonance; Phase-Flux Field; Structure and Derivation of Physical Constants.

- **`Papers/Math/`**  
  Rest Energy; Emergence of Effective Mass; Hard Upper Bound; Logarithmic Curvature Flow; Koide Mass Relation.

- **`Papers/Experiments/`**  
  Water Cavity; Prediction & Protocol Ledger; JUNO Ghost-Mode Detectability.

- **`Papers/OpenData/`**  
  LHCb `C9(q²)`; B⁰ → K*⁰ Candidate Spectra; NIST Atomic Line List.

- **`Papers/Computation/`**  
  Discrete Wave-Constrained Computation; WCC P vs NP; Classical P vs NP Is Ill-Posed; WaveLock.

- **`Papers/AI/`**  
  Resonance-Confinement Architecture; Recursive AI Drift Audit.

- **`Papers/Engineering/`**  
  Nuclear Fusion Tokamak with Self Sustaining Resonance.

Suggested code grouping:

- **`Core/`**: curvature operator, PFF evolution, finite-k Lyapunov band-pass.  
- **`Examples/Cavity/`**: water-cavity and harmonic-state protocols.  
- **`Examples/Cymatics/`**: random → annulus → eigenmode simulations.  
- **`Examples/Mass/`**: loop curvature, solenoidal mass, Koide geometry.  
- **`Examples/OpenData/`**: JUNO, LHCb, NIST analysis scripts.  
- **`Examples/WCC/`**: curvature-bounded computation tests.  
- **`Examples/WaveLock/`**: PDE evolution, hashing, and adversarial tests.

---

## 11. Proof-Status Legend

Use these labels consistently when presenting the corpus:

- **Definition**: introduces symbols, operators, or model objects.  
- **Ansatz**: proposes a structural identification or model form.  
- **Derivation**: follows from stated assumptions by explicit calculation.  
- **Simulation**: numerically demonstrates behavior under chosen dynamics.  
- **Experiment**: reports physical measurement or protocol data.  
- **Open-data test**: tests WCT-motivated structure against public datasets.  
- **Prediction ledger**: records dated claims and pass/fail conditions.  
- **Architecture**: proposes an implementation or system design.  
- **Speculative extension**: extrapolates WCT beyond the directly validated core.

Core discipline:

```text
Definition ≠ Ansatz ≠ Derivation ≠ Simulation ≠ Experiment ≠ Prediction
```

