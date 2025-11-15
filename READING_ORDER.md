# Wave Confinement Theory – Reading Map

Practical guide to the WCT volume set:  
**what to read, in what order, for which goal.**

---

## 0. If you only have 30–60 minutes

Start here:

1. **The Geometry of Resonance**  
   *[Zenodo 15644222](https://zenodo.org/records/15644222) – core overview and Lagrangian*  
   - Role: main idea, curvature-feedback Lagrangian, how mass / force / geometry emerge.  
   - How to read: abstract → introduction → figures → conclusion.

2. **Structure and Derivation of Physical Constants through Wave Confinement**  
   *[Zenodo 15596159](https://zenodo.org/records/15596159) – how “constants” arise from structure*  
   - Role: how ξ, σ, β, θ and related quantities are fixed by confinement rules.  
   - How to read: focus on conceptual sections; skip detailed fits on first pass.

---

## 1. Foundations (must-read for everything else)

Read in this order:

1. **Phase–Flux Field (PFF)**  
   *[Zenodo 17578766](https://zenodo.org/records/17578766)*  
   - Role: defines the substrate (u, S, θ), conservation, and the finite-k Lyapunov band-pass. 
   - Look for: why random fields collapse to a narrow band before any “mass” is defined.

2. **The Geometry of Resonance**  
   *[Zenodo 15644222](https://zenodo.org/records/15644222)*  
   - Role: introduces the curvature-regularized Lagrangian and effective metric.  
   - Look for: how the curvature operator, effective metric, and confinement functional fit together.

Recommended reading style:
- First pass: abstract, intro, main equations, all figures.  
- Second pass: derivations / appendices only if you need to implement or prove things.

---

## 2. Mass & Geometry (how inertia and dimension emerge)

After PFF + GoR, read:

3. **Rest Energy from Density-Weighted Loop Curvature**  
   *[Zenodo 17579059](https://zenodo.org/records/17579059)*  
   - Role: precise mass law m = (ℏ/𝑐)⟨σ⟩_w on a closed loop.
   - Look for: the variational lock, uniqueness proof, and mislock error bounds.

4. **Emergence of Effective Mass: Solenoidal Topology of Vibrational Energy**  
   *[Zenodo 17459463](https://zenodo.org/records/17459463)*  
   - Role: connects curved photon-like paths, torsion, and solenoidal topology to effective mass.  
   - Look for: how geometric invariants along the path map to inertial behavior.

5. **Hard Upper Bound on Spatial Dimensionality in Wave Confinement Theory**  
   *[Zenodo 17081283](https://zenodo.org/records/17081283)*  
   - Role: shows why stable self-localized confinement prefers n ≤ 3.
   - Look for: the key inequality and scaling that break down for n > 3.

If you care primarily about **“is this a viable mass/geometry story?”**, these three papers plus the foundations are the critical set.

---

## 3. Dynamics, Patterns, and Cavity Physics

Once the mass/geometry picture is clear, move to dynamics:

6. **Resonant Cavity of Vector Fields**  
   *[Zenodo 17371795](https://zenodo.org/records/17371795)*  
   - Role: Swift–Hohenberg-like reduction for vector fields, finite-band instability, mode competition.  
   - Look for: how discrete modes and toroidal/spinor-like structures appear from the band-pass.

7. **Self-Emergent Fourier Cymatics: Entropic Eigenmodes out of Chaos**  
   *[Zenodo 17578796](https://zenodo.org/records/17578796)*  
   - Role: numerical proof-of-concept for random → annulus → coherent ring via curvature + dissipation.  
   - Look for: spectral collapse, ring formation, Lyapunov descent behavior.

8. **Observation of Long-Lived Photon Resonance Confinement in Water Cavities**  
   *[Zenodo 17206381](https://zenodo.org/records/17206381)*  
   - Role: experimental evidence; how actual cavity data matches the confinement predictions.  
   - Look for: long-lived modes, FFT structures, and how they map back to the theory.

If you’re an **experimentalist**, a good path is: Overview → PFF → Water Cavities → then RCVF / Cymatics as needed.

---

## 4. Computation and Complexity

For readers with a CS / complexity focus:

9. **A Formal Proof of P ≠ NP via Curvature-Regulated Wave Computation**  
   *[Zenodo 17081273](https://zenodo.org/records/17081273)*  
   - Role: constructs a curvature-machine model and a time-hierarchy-style separation.  
   - Suggested path:
     - Read intro + model definition.  
     - Skim the diagonal argument to see where curvature constraints enter.  
     - Use the appendices only if you want to check every combinatorial step.

This paper is logically self-contained; it only needs high-level familiarity with WCT (from the Overview + GoR).

---

## 5. AI and Architecture

For AI / alignment readers:

10. **Resonance-Confinement Architecture (RCA)**  
    *[Zenodo 15659978](https://zenodo.org/records/15659978)*  
    - Role: maps WCT principles into an AI substrate focused on coherence and bounded curvature rather than unbounded loss minimization.  
    - Reading tip: treat it as an architecture spec; you only need core WCT concepts (coherence, curvature, Lyapunov descent), not every field-theory derivation.

---

## 6. How to Map Papers to This Repository

- The **`Papers/`** directory contains PDFs matching the Zenodo IDs listed above.  
- The **`Core/`** and **`Examples/`** directories contain code that implements:
  - PFF evolution and Lyapunov band-pass (related to PFF and GoR).  
  - Cavity and ring formation (related to RCVF, Fourier Cymatics, and Water Cavities).  
  - Spectral and geometric mass diagnostics (related to Rest Energy and Emergent Mass).

Use this map as:

- A **reading order** for the theory, and  
- A **lookup table** to find the matching code and experiments in this GitHub repository.
