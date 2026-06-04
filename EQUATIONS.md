# Wave Confinement Theory — Canonical Equation Families (E1–E82)

**Richard J. Reyes — Master Reference (v1.0, Nov 2025)**

This document is the authoritative catalogue of the core equations used across the WCT research volumes. Each entry contains: (i) plain definition, (ii) symbolic form, (iii) context (paper/module).


---

## 0. Symbol Table

| Symbol | Definition |
|--------|------------|
| $\psi(x,t)$ | Wavefield (complex) |
| $\Theta[\psi]$ | Curvature-feedback operator |
| $\kappa, \tau$ | Curvature, torsion of loop |
| $\sigma_{\text{dens}}(s) = \kappa(s)^2 + \tau(s)^2$ | Curvature-rate density along $\Gamma$ (units $L^{-2}$) |
| $\sigma(s) = \sqrt{\kappa(s)^2 + \tau(s)^2}$ | Curvature spectral rate (inverse-length scale, units $L^{-1}$) |
| $w(s)$ | Energy-density weight along loop |
| $A(x,t)$ | Band-pass amplitude |
| $\hat{A}_k$ | Fourier mode of $A$ |
| $H_k$ | Spectral entropy |
| $\mathcal{E}[\psi]$ | WCT Lyapunov functional |
| $\Lambda$ | Winding-number Lagrange multiplier |
| $k_{\text{eff}}$ | Effective wavenumber |
| $\alpha(n)$ | $\alpha$-Drop exponent |
| $U$ | Discrete WCC update rule |
| $k_{\max}$ | Energy-limited bandlimit |
| $N_{\text{lanes}}$ | Spatial channel capacity (modes/lanes) |

---

## A. Rest Energy, Curvature, Loop Locking

*(Rest Energy / Solenoidal Mass)*

### E1a — Curvature-rate density

Curvature-plus-torsion density along a loop $\Gamma$:

$$\sigma_{\text{dens}}(s) = \kappa(s)^2 + \tau(s)^2$$

---

### E1b — Curvature spectral rate (inverse-length)

Inverse-length curvature scale used for locking / effective wavenumber:

$$\sigma(s) = \sqrt{\kappa(s)^2 + \tau(s)^2}$$

---

### E2 — Weighted loop average

Density-weighted average of a scalar $f$ along $\Gamma$:

$$\langle f \rangle_w = \frac{\oint_\Gamma w(s)\, f(s)\, ds}{\oint_\Gamma w(s)\, ds}$$

---

### E3 — Loop-locking action

Phase–curvature locking with winding constraint:

$$S_{\text{eff}}[\varphi] = \oint_\Gamma w(s) \left(\partial_s \varphi(s) - \sigma(s)\right)^2 ds + \Lambda \left( \oint_\Gamma \partial_s \varphi(s)\, ds - 2\pi n \right)$$

---

### E4 — Covariant locking (Euler–Lagrange)

Phase gradient with density correction:

$$\partial_s \varphi(s) = \sigma(s) + \frac{\alpha}{w(s)}$$

where

$$\alpha = \frac{2\pi n - \oint_\Gamma \sigma(s)\, ds}{\oint_\Gamma \frac{ds}{w(s)}}$$

---

### E5 — Effective wavenumber identity

Equivalence of winding, loop-length, and curvature average:

$$k_{\text{eff}} = \frac{2\pi|n|}{L_s} = \frac{1}{L_s} \oint_\Gamma \sigma(s)\, ds = \langle\sigma\rangle_w$$

where $L_s := \oint_\Gamma ds$

---

### E6 — Mass–curvature law

Rest energy and mass from effective curvature:

$$E_{\text{rest}} = \hbar c\, k_{\text{eff}}$$

$$m = \frac{\hbar}{c} \langle\sigma\rangle_w$$


---

### E7 — Solenoidal mass law

Mass from curvature–torsion combination along $\Gamma$:

$$m = \frac{\hbar}{c} \left\langle \sqrt{\kappa^2 + \tau^2} \right\rangle_\Gamma$$

---

### E8 — Density-weighted lock identity

Integral identity for a locked loop:

$$\oint_\Gamma w\, \partial_s \varphi\, ds = \oint_\Gamma w\, \sigma\, ds + \frac{2\pi\alpha \oint_\Gamma ds}{\oint_\Gamma \frac{ds}{w(s)}}$$

---

## B. Phase–Flux Field & Cymatic Rails

### E9 — Phase–flux constitutive axiom

Phase flux proportional to amplitude:

$$\mathbf{S}(x,t) = u(x,t)\, \nabla\theta(x,t)$$

---

### E10 — Radial shell quantization

Radial wavenumber quantization between shells:

$$\int_{r_1}^{r_2} k_r(r)\, dr = 2\pi n$$

---

### E11 — PFF vorticity

Topological phase winding number:

$$m(\gamma) = \frac{1}{2\pi} \oint_\gamma \nabla\theta \cdot d\boldsymbol{\ell} \in \mathbb{Z}$$

---

### E12 — Finite-band dispersion rail

Band-limited growth rate:

$$\sigma(k) = r + a|k|^2 - b|k|^4, \quad \text{with } a, b > 0$$

---

### E13 — Band-pass amplitude evolution

Nonlinear band-pass evolution:

$$\partial_t A = \sigma(-i\nabla) A - \beta |A|^2 A$$

---

### E14 — Band-pass Lyapunov functional

Energy functional for band-pass dynamics:

$$\mathcal{E}[A] = \int \left( -r|A|^2 - a|\nabla A|^2 + b|\Delta A|^2 + \frac{\beta}{2}|A|^4 \right) dx$$

---

### E15 — Modal growth bound

Amplitude-square inequality per mode:

$$\frac{d}{dt} |\hat{A}_k|^2 \leq 2\sigma(k) |\hat{A}_k|^2 - c|\hat{A}_k|^4$$

---

### E16 — Randomness → spectral concentration

Growth of initial broadband noise:

$$P_k(t) \propto P_k(0)\, e^{2\sigma(k)t}$$

$$\arg\max_k P_k(t) \to k^*$$

---

## C. Curvature Feedback & Lyapunov Dynamics

### E17 — Curvature operator

Nonlinear curvature-feedback operator:

$$\Theta[\psi] = -\frac{\nabla^2\psi}{\psi + \varepsilon\, e^{-\alpha|\psi|^2}}$$

---

### E18 — WCT Lyapunov functional

Curvature-plus-gradient energy:

$$\mathcal{E}[\psi] = \int \left( c_1 |\nabla\psi|^2 + c_2 |\Theta[\psi]|^2 \right) dx$$

---

### E19 — Spectral-gap ↔ curvature invariant

Scaling relation between curvature and gap:

$$\Delta^* \sim \langle\sigma\rangle_w^2$$

---

### E20 — Higher-order cavity Lagrangian

Fourth-order cavity model (schematic):

$$\mathcal{L} = f(\psi) \left( \kappa S^2 + \theta P^2 - \gamma S P - \lambda \psi^2 \right)$$

---

### E21 — 4th-order cavity Euler–Lagrange (schematic)

Generalized EL for Lagrangians with second derivatives:

$$\frac{\delta\mathcal{L}}{\delta\psi} = \frac{\partial\mathcal{L}}{\partial\psi} - \partial_\mu \left( \frac{\partial\mathcal{L}}{\partial(\partial_\mu \psi)} \right) + \partial_\mu \partial_\nu \left( \frac{\partial\mathcal{L}}{\partial(\partial_\mu \partial_\nu \psi)} \right) = 0$$

---

### E22 — Effective metric

Matter-coupled metric deformation:

$$g_{\mu\nu}^{\text{eff}} = \eta_{\mu\nu} + \lambda \frac{\partial_\mu \psi\, \partial_\nu \psi}{\rho c^2} + \delta\eta_{\mu\nu} \frac{W_\psi}{W_0}$$

---

### E23 — Enthalpic curvature relation

Local enthalpy as energy plus curvature gradient:

$$h(\psi) \propto W_\psi + \chi |\nabla\psi|^2$$

---

## D. Dimensionality & Functional Bounds

### E24 — $n \leq 3$ stability bound

Embedding condition for pointwise control:

$$H^2(\mathbb{R}^n) \hookrightarrow L^\infty(\mathbb{R}^n) \Rightarrow n \leq 3$$

---

### E25 — Subcritical nonlinearity constraint

Nonlinearity exponent below critical Sobolev threshold:

$$p < p_c(n)$$

---

### E26 — Curvature norm bound

Curvature control via Sobolev norm:

$$\|\Theta[\psi]\|_{L^\infty} \leq C \|\psi\|_{H^2}$$

---

### E27 — Finite-energy confinement

WCT finite-energy condition:

$$\int_{\mathbb{R}^n} \left( |\nabla\psi|^2 + |\Theta[\psi]|^2 \right) dx < \infty$$

---

## E. α-Drop, Entropy Reduction, Pruning

### E28 — α-Drop exponent

Exponent defined from multiplicative pruning:

$$\alpha(n) = 1 + \frac{1}{n} \sum_{t=1}^{m(n)} \log_2 q_t(n) + \beta(n)$$

with $q_t(n) = \dfrac{M_t + 1}{M_t}$

---

### E29 — Entropy-drop pruning

State-count decay per step:

$$M_{t+1} \leq e^{-\Delta_t} M_t$$

---

### E30 — Spectral entropy

Entropy in Fourier space:

$$H_k(t) = -\sum_k P_k(t) \log P_k(t)$$

---

### E31 — Band-pass entropy drop

Curvature-induced entropy decrease (heuristic):

$$\Delta_t \gtrsim c_0 \left( k^{*2} \Delta_t \right)$$

---

### E32 — $\alpha < 1$ curvature-bounded search

Sub-exponential effective exploration:

$$\limsup_{n\to\infty} \alpha(n) < 1$$

---

### E33 — Support shrinkage

Support size controlled by entropy:

$$K_t \leq e^{H_k(t)}$$

---

### E34 — Energy–entropy conversion

Curvature energy vs. entropy change:

$$\Delta E_t \geq \lambda\, \Delta H_k(t)$$

---

## F. WCC, Channel Capacity, P vs NP

### E35 — Curvature-locked fixed point

Locked WCT configuration:

$$W_\psi = -\frac{\nabla^2\psi}{\psi + \varepsilon\, e^{-\alpha|\psi|^2}}$$

$$\frac{d}{dt} S[\psi] \to 0$$

$$\nabla W_\psi \to 0$$

---

### E36 — Discrete WCC update

Local update rule with neighbourhood $N(x)$:

$$\psi^{(t+1)}(x) = U\left( \psi^{(t)}(x),\, \{\psi^{(t)}(y)\}_{y \in N(x)} \right)$$

---

### E37 — Bandlimit from energy

Maximal wavenumber from energy bound:

$$k_{\max} = C_1 \frac{E_{\max}}{\hbar c}$$

---

### E38 — Spatial channel capacity

Max mode count in volume $V$:

$$N_{\text{lanes}} \leq C_2\, V\, k_{\max}^3$$

---

### E39 — Time-step polynomial bound

Max update steps for input size $n$:

$$T_{\max}(n) \leq C_3\, n^d$$

---

### E40 — $P_{\text{WCC}} / NP_{\text{WCC}}$

Model-relative identification:

$$P_{\text{WCC}} = P$$

$$NP_{\text{WCC}} = NP$$

---

### E41 — Curvature-bounded configuration count

Configuration count under curvature rails:

$$|C_{\text{curv}}(n)| \leq 2^{\alpha(n)\, n}, \quad \text{with } \alpha(n) < 1$$

---

### E42 — Θ-information identity

Coherence information decay:

$$\frac{d}{dt} I_{\text{coh}}[\psi] \propto -|\Theta[\psi]|^2$$

---

### E43 — Curvature–entropy tradeoff

Spectral entropy decay from curvature:

$$\frac{d}{dt} H_k(t) \leq -\mu\, |\Theta[\psi]|^2$$

---

## G. Resonant Cavity & Tokamak Scaling

### E44 — Θ-eigenmode quantization

Curvature eigenmodes:

$$\Theta[\psi_n] = \lambda_n \psi_n$$

---

### E45 — Effective Q-factor

Quality factor with loss region $\gamma_{\text{loss}}$:

$$Q_{\text{eff}} = \omega \frac{\int u\, dV}{\int_{\gamma_{\text{loss}}} u\, dV}$$

---

### E46 — Plasma–cavity curvature match

Matched curvature averages:

$$\langle\sigma\rangle_{w,\text{plasma}} \approx \langle\sigma\rangle_{w,\text{cavity}}$$

---

### E47 — Power balance with curvature losses

Input vs loss and fusion:

$$P_{\text{in}} = P_{\text{loss}}(\psi) + P_{\text{fusion}}(\psi)$$

---

### E48 — Stability via curvature gap

Core–edge curvature gap criterion:

$$\Delta\sigma = \langle\sigma\rangle_{\text{core}} - \langle\sigma\rangle_{\text{edge}} > \Delta_{\text{crit}}$$

---

# Supplemental Wave Confinement Theory Equation Families (E49–E82)

*(Second-tier structural laws across Geometry of Resonance, Self-Emergent Cymatics, Enthalpic Aether, Randomness, Dimensionality, and P v. NP papers.)*

---

## H. Geometry-of-Resonance Extensions (Curvature, Phase) — E49–E56

### E49 — Curvature-modified Helmholtz effective mass

Gap-induced effective mass and spectrum:

$$m_{\text{eff}}^2 = \Delta^* c^2$$

$$\omega_j^2 = c^2 \lambda_j + \Delta^*$$

---

### E50 — Phase-coherence functional

Global phase coherence measure:

$$\mathcal{C}[\psi] = \int_\Omega |\nabla\theta|^{-1} |\psi|^2\, dx$$

---

### E51 — Curvature–gradient commutator

Non-commutativity of curvature and gradient (schematic):

$$[\Theta, \nabla]\psi = \nabla\left(\frac{\nabla^2\psi}{\psi + \cdots}\right) - \frac{\nabla^2(\nabla\psi)}{\psi + \cdots}$$

("…" uses the same denominator structure as in $\Theta[\psi]$ from E17.)

---

### E52 — Nonlinear curvature gain/loss balance

Global curvature gain vs gradient loss:

$$G_\sigma = \int |\Theta|^2\, dx$$

$$L_\sigma = \int |\nabla\psi|^2\, dx$$


### E53 — Local curvature pressure-like density

Curvature penalty density (tied to the Lyapunov weight from E18):

$$p_\Theta(x) := c_2\, |\Theta[\psi](x)|^2$$

where $c_2$ is the curvature weight in the Lyapunov functional:

$$\mathcal{E}[\psi] = \int \left( c_1 |\nabla\psi|^2 + c_2 |\Theta[\psi]|^2 \right) dx$$

---

### E54 — Resonance-lock condition (stationary attractor)

Locked resonance state:

$$\partial_t \psi = 0$$

$$\delta\mathcal{E}[\psi] = 0$$

$$\nabla\Theta = 0$$

---

### E55 — Curvature-induced effective potential

Potential modified by curvature energy:

$$V_{\text{eff}}(\psi) = V(|\psi|^2) + \kappa\, |\Theta[\psi]|^2$$

---

### E56 — Phase-wall formation criterion

Phase-wall vs bulk curvature:

$$|\nabla\theta| \sim \sigma_{\text{wall}} \gg \langle\sigma\rangle_w$$

---

## I. Self-Emergent Fourier Cymatics (Swift–Hohenberg Structure) — E57–E64

### E57 — Swift–Hohenberg operator representation

Band-selective operator:

$$\mathcal{SH}[A] = (k^{*2} + \Delta)^2 A$$

---

### E58 — Band-selective Green's kernel

Spectral Green's function:

$$G(k) = \frac{1}{r + a(k^2 - k^{*2})^2}$$

---

### E59 — Projection onto dominant annulus

Let the dominant annulus be (discrete FFT spectrum):

$$
\mathcal{A}^{\star}
:= \{\, k \in Z^{d} : \lvert k \rvert - k_{\star} \le \Delta k \,\}
$$



Then the band-projection of $A$ is:

$$
(P_{k_{\star}} A)(x)
:= \sum_{k \in \mathcal{A}^{\star}} \hat{A}_{k}\, e^{i\,k\cdot x}
$$



---


### E60 — Center-manifold amplitude equation

Reduced amplitude dynamics:

$$\partial_T \mathcal{A} = \mu \mathcal{A} - g|\mathcal{A}|^2 \mathcal{A}$$

---

### E61 — Pattern formation threshold

Critical growth parameter:

$$r_c = \min_k \left[ a(k^2 - k^{*2})^2 \right] = 0$$

---

### E62 — Spectral energy concentration

Energy fraction in dominant modes:

$$\eta(t) = \frac{\sum_{k \in \mathcal{A}^*} |\hat{A}_k|^2}{\sum_k |\hat{A}_k|^2}$$

---

### E63 — Entropic mode selection

Entropy-weighted mode preference:

$$k^* = \arg\min_k \left[ H_k + \lambda\, C_\Theta(k) \right]$$

---

### E64 — Pattern wavelength from curvature

Characteristic scale from curvature balance:

$$\lambda^* = \frac{2\pi}{k^*} = 2\pi \sqrt{\frac{b}{a}}$$

---

## J. Dimensionality Bounds & Sobolev Structure — E65–E70

### E65 — Critical Sobolev exponent

Dimension-dependent critical power:

$$p_c(n) = \frac{n + 2}{n - 2} \quad (n > 2)$$

---

### E66 — Gagliardo–Nirenberg interpolation

Interpolation inequality:

$$\|u\|_{L^p} \leq C \|\nabla u\|_{L^2}^\theta \|u\|_{L^2}^{1-\theta}$$

---

### E67 — Embedding failure for $n > 3$

Counterexample existence:

$$\exists\, \psi \in H^2(\mathbb{R}^n) : \psi \notin L^\infty(\mathbb{R}^n) \quad \text{if } n > 3$$

---

### E68 — Energy concentration bound

Localized energy control:

$$\int_{B_R} |\nabla\psi|^2\, dx \leq C R^{n-2} \|\psi\|_{H^1}^2$$

---

### E69 — Curvature regularity requirement

Minimum regularity for bounded curvature:

$$\psi \in H^2 \Rightarrow \Theta[\psi] \in L^\infty \quad \text{iff } n \leq 3$$

---

### E70 — Dimensional stability criterion

Combined bound:

$$n \leq 3 \iff H^2 \hookrightarrow L^\infty \text{ and } p < p_c(n)$$

---

## K. P vs NP Computational Bounds — E71–E76

### E71 — Physical computation bound

Spacetime resource limit:

$$T \cdot V \cdot k_{\max}^3 \leq C_{\text{phys}}$$

---

### E72 — Curvature-pruned search space

Effective configuration count:

$$|S_{\text{eff}}(n)| \leq 2^{\alpha(n) n}$$

---

### E73 — Polynomial verification

Verifier constraint:

$$V(x, w) \in P \quad \text{for } |w| = \text{poly}(|x|)$$

---

### E74 — Curvature separation conjecture

Gap between classes:

$$\inf_n \left[ \frac{\log |NP_n|}{\log |P_n|} \right] > 1$$

---

### E75 — Physical oracle impossibility

No polynomial-time curvature oracle:

$$\nexists\, O : O(\psi) = \arg\min_\psi \mathcal{E}[\psi] \text{ in poly time}$$

---

### E76 — WCC complexity equivalence

Model-theoretic identification:

$$P_{\text{WCC}} = P \implies \text{WCC captures physical computation}$$

---

## L. Entropy & Information Dynamics — E77–E82

### E77 — Mutual information decay

Information loss rate:

$$\frac{d}{dt} I(\psi; \psi_0) \leq -\gamma\, \mathcal{E}_\Theta[\psi]$$

---

### E78 — Fisher information bound

Curvature–Fisher relation:

$$\mathcal{I}_F[\psi] \geq c\, \int |\Theta[\psi]|^2\, dx$$

---

### E79 — Entropy production rate

Irreversibility measure:

$$\dot{\Sigma} = \frac{d H_k}{dt} + \frac{\mathcal{E}_\Theta}{T_{\text{eff}}}$$

---

### E80 — Landauer bound analog

Minimum energy per entropy reduction:

$$\Delta E \geq k_B T_{\text{eff}} \ln 2 \cdot \Delta H$$

---

### E81 — Coherence length from entropy

Characteristic scale:

$$\xi_{\text{coh}} = \left( \frac{\mathcal{E}[\psi]}{H_k} \right)^{1/2}$$

---

### E82 — Information–geometry duality

Curvature as information metric:

$$g_{ij}^{(\text{info})} = \langle \partial_i \Theta\, \partial_j \Theta \rangle$$

---

# Curvature-Locking Equations (CLE1–CLE10)

## CLE1 — Curvature-Locking Functional (Toroidal Form)

$$S[\psi] = \int_\mathcal{M} \left[ |\nabla\psi|^2 + |W_\psi - \sigma_\star|^2 \right] \sqrt{g}\, d^3x$$

$$W_\psi := -\frac{\nabla^2\psi}{\psi}$$

---

## CLE2 — Euler–Lagrange (Curvature-Lock)

$$-\nabla^2\psi + (W_\psi - \sigma_\star) \cdot \frac{\nabla^2\psi}{\psi^2} = 0$$

---

## CLE3 — Curvature-Locking Condition

$$W_\psi = \sigma_\star \quad \text{(spatially uniform curvature)}$$

---

## CLE4 — Effective Equation for Locked ψ

$$-\nabla^2\psi = \sigma_\star \cdot \psi$$

---

## CLE5 — Laplacian on Torus (Flat Embedding)

$$\nabla^2\psi = \frac{1}{R^2} \partial_\theta^2 \psi + \frac{1}{r^2} \partial_\phi^2 \psi$$

---

## CLE6 — Separation Ansatz

$$\psi(\theta, \phi) = f(\theta)\, g(\phi)$$

Yields:

$$\frac{f''(\theta)}{f(\theta)} + \frac{R^2}{r^2} \cdot \frac{g''(\phi)}{g(\phi)} = -\sigma_\star R^2$$

---

## CLE7 — Reduced Angular ODE (Thin-Torus Limit)

$$f''(\theta) + \sigma_\star^2 f(\theta) = 0$$

Only smooth $2\pi$-periodic solution under curvature locking is constant $f$.

---

## CLE8 — ψ-Electron Eigenmode Solution

$$\psi(\theta, \phi) = A\, e^{i\phi}$$

Unique curvature-locked toroidal eigenmode.

---

## CLE9 — Electron Radius from Curvature

$$R = \frac{1}{\sigma_\star}$$

For the electron: $R \approx 386.3$ fm.

---

## CLE10 — Curvature Scalar Identity

$$W_\psi = -\frac{\nabla^2\psi}{\psi} = \sigma_\star^2$$

Ties together curvature scalar, eigenmode equation, and feedback-collapsed ψ-electron solution.

---

## G1 — Ghost-mode modulation (JUNO phenomenology)

$$\delta_g(E) = A_g \cos\left( k_\ell \ln\frac{E}{E_0} + \phi \right)$$

---




# WCT Topology & Spectral Emergence — Core Equations

## 0. Scope (What Is Uniquely WCT)

This document includes **only equations and principles that are unique to Wave Confinement Theory (WCT)**.

**Explicitly excluded**:
- Pure Frenet–Serret definitions
- Classical knot invariants (Alexander, Jones, etc.)
- Elastic rod models
- Gauge, holonomy, or Berry-phase postulates

**Included**:
- Irreversible curvature flow as a selection principle
- Spectral (Fourier-loop) formulation of physical structure
- Dynamical (not assumed) emergence of topology
- Energy–topology correspondence verified numerically

This file contains **only WCT-specific structure**. Standard mathematics is used only as substrate.

---

## 1. Configuration Space (WCT Postulate)

A physical structure is represented as a **closed spectral loop**:

\[
\gamma(s) =
\sum_{k=1}^{K}
\left[
a_k \cos(ks) + b_k \sin(ks)
\right],
\quad s \in [0, 2\pi)
\]

This representation is **not a mathematical convenience**.

**WCT Postulate 1**  
> All physical structure is a superposition of closed rotational modes.

**Clarification (Emergent Basis)**  
The Fourier-loop basis is not assumed a priori. Empirically, irreversible curvature flow drives all smooth closed curves toward band-limited rotational modes. The spectral basis is therefore **emergent**, not axiomatic.

---

## 2. WCT Energy Functional (Minimal Form)

\[
\boxed{
\mathcal{E}_{\text{WCT}}[\gamma]
=
\int \kappa^2 \, ds
+
\alpha \sum_k k^p \left( |a_k|^2 + |b_k|^2 \right)
+
V_{\text{SA}}[\gamma]
}
\]

**Terms**:
- \( \int \kappa^2 ds \): curvature confinement
- spectral penalty: ultraviolet suppression
- \( V_{\text{SA}} \): self-avoidance (steric exclusion)

**Uniqueness**:
- No gauge fields
- No imposed topology
- No quantization postulates

This functional is implemented directly in WCT CuPy simulations.

**Observed Constraint**  
> Stability requires \( p > 1 \).  
> For \( p \le 1 \), ultraviolet cascade destroys all confined structures under curvature flow.

Thus, spectral suppression is not optional—it is a **necessity condition** for persistence.

---

## 3. Irreversible Curvature Flow (Core WCT Principle)

Structure evolves by irreversible energy descent:

\[
\boxed{
\frac{\partial \gamma}{\partial t}
=
- \frac{\delta \mathcal{E}_{\text{WCT}}}{\delta \gamma}
}
\]

Along this flow:

\[
\boxed{
\frac{d\mathcal{E}_{\text{WCT}}}{dt}
=
-
\left\|
\frac{\delta \mathcal{E}_{\text{WCT}}}{\delta \gamma}
\right\|^2
\le 0
}
\]

Equality holds only at fixed points.

**WCT Principle**  
> Physical structure is defined by what survives irreversible curvature descent.  
> No inverse flow exists.

This irreversibility defines the physical arrow of time within WCT.

---

## 4. Emergent Topology Criterion (Central Equation)

A quantity \( I[\gamma] \) corresponds to **physical topology** iff:

\[
\boxed{
\exists \; t \to \infty :
\quad I[\gamma_t] = \text{const}
\quad \text{and} \quad
\frac{d\mathcal{E}}{dt} < 0
}
\]

**Interpretation**:
- Smoothly removable → not physical
- Arbitrarily deformable under descent → not physical
- Survives all irreversible flow → emergent topology

Topology is therefore **defined dynamically**, not postulated.

---

## 5. WCT Codimension (Dynamical Definition)

\[
\boxed{
\text{codim}_{\text{WCT}}(\gamma)
=
\text{minimum number of singular events required to reach the unknot}
}
\]

This is **not** classical manifold codimension.  
It is a **dynamical obstruction count**.

| Structure   | \( \text{codim}_{\text{WCT}} \) | Explanation                    |
|------------|----------------------------------|--------------------------------|
| Unknot     | 0                                | No obstruction                 |
| Helix      | 0                                | Smooth untwisting              |
| Trefoil    | 1                                | One self-intersection required |
| Hopf link  | 2                                | Two linked components          |
| Borromean  | 3                                | Collective obstruction         |

Only singular events can reduce codimension.

---

## 6. Spectral Topology Invariant (Observed)

Define normalized curvature energy density:

\[
\boxed{
\epsilon_\kappa
=
\frac{1}{L}
\int \kappa^2 ds
}
\]

### Empirical Result

| Structure  | \( \epsilon_\kappa \) band |
|-----------|----------------------------|
| Unknot    | lowest                     |
| Figure-8  | intermediate               |
| Trefoil   | higher                     |

**Key Result**  
> Distinct topologies occupy disjoint curvature-energy bands.

No band structure is imposed.

**Negative Result**  
> If curvature flow is made reversible, or if spectral suppression is removed, no band structure appears.

Thus, discreteness is **dynamically generated**.

---

## 7. Mass Proxy (WCT Prediction)

\[
\boxed{
m_{\text{WCT}}
\;\sim\;
\epsilon_\kappa
\quad
\text{(within a fixed topological class)}
}
\]

Mass arises from **irreducible curvature confinement**, not from scalar fields or symmetry breaking.

Absolute mass scales are addressed only after topological class fixation.

---

## 8. Failure of Holonomy (Formal)

Holonomy:
\[
H = \int \tau ds
\]

For all \( \varepsilon > 0 \), there exists a smooth deformation \( \delta \gamma \) such that:

\[
|\Delta H| < \varepsilon
\]

Therefore:

\[
\boxed{
\text{Holonomy is not a WCT invariant}
}
\]

**Generalization**  
> Any quantity continuously deformable under curvature descent cannot define particle identity.

This excludes gauge, phase, and holonomy-based invariants.

---

## 9. Protein ↔ Particle Correspondence (WCT-Only Bridge)

Proteins obey:
- identical curvature flow
- identical self-avoidance
- identical spectral suppression

Thus:

\[
\boxed{
\text{Knotted proteins}
\;\leftrightarrow\;
\text{stable WCT excitations}
}
\]

Observed:
- unknotted proteins relax to lowest bands
- knotted proteins remain trapped in higher bands

**Boundary Condition**  
This correspondence applies only to systems governed by irreversible curvature flow with steric self-avoidance.

---

## 10. Testable Predictions (Status)

### P1 — Universality  
All unknotted loops collapse to the same band  
**Status**: verified

### P2 — Knot Stability  
Knots cannot reach the unknot band without singularity  
**Status**: verified

### P3 — Band Quantization  
Distinct knot types occupy disjoint bands  
**Status**: emerging (additional PDB data required)

### P4 — UV Necessity  
Removing spectral suppression destroys stability  
**Status**: verified

---

## 11. Final WCT Statement

\[
\boxed{
\textbf{Topology is not a kinematic assumption.}
\quad
\textbf{Topology is the residue of irreversible curvature dynamics.}
}
\]

This is the core equation-level contribution of Wave Confinement Theory.





---

## EX — Logarithmic Field Representation

### Plain definition

Introduce a logarithmic representation of the wavefield to expose the geometric structure of the curvature operator.

### Symbolic form

Logarithmic transform

u = ln ψ

Inverse relation

ψ = e^u

Gradient relation

∇ψ = e^u ∇u

Laplacian relation

∇²ψ = e^u ( ∇²u + |∇u|² )

Curvature ratio identity

(∇²ψ) / ψ = ∇²u + |∇u|²

### Context

This identity converts the curvature operator

Θ[ψ] = − ∇²ψ / ψ

into the nonlinear geometric form

Θ[ψ] = − ( ∇²u + |∇u|² )

revealing a Hamilton–Jacobi curvature flow.

---

## EY — Log–Curvature Evolution Equation

### Plain definition

Curvature-driven relaxation expressed in logarithmic coordinates.

### Symbolic form

∂t u = ∇²u + |∇u|²

### Context

This equation is a viscous Hamilton–Jacobi equation describing curvature-driven smoothing and filament sharpening. It governs the local curvature flow of the wavefield.

---

## EZ — Cole–Hopf Linearization

### Plain definition

The nonlinear logarithmic curvature flow admits an exact linearization.

### Symbolic form

Substitution

ψ = e^u

produces

∂t ψ = ∇² ψ

### Context

Thus the nonlinear curvature evolution is equivalent to diffusion of the underlying wavefield.

This implies that local nonlinear curvature dynamics alone cannot maintain stable confined structures. Persistent localization therefore requires global geometric or topological constraints.

---

## FA — Filament Localization Condition

### Plain definition

Localization of curvature filaments occurs when the gradient magnitude of the logarithmic field matches the curvature scale of the core curve.

### Symbolic form

|∇u| ~ κ

where

κ = curvature of the filament core curve.

### Context

This condition links geometric curvature to the formation of localized filamentary wave structures. These filaments provide the geometric basis for curvature-locked particle modes.

---






# 🌌 Wave Confinement Theory (WCT) — Cosmology Equation Set (CM1–CM20)

> **Module:** `WCT Cosmology Core`  
> **Scope:** CMB Spectrum, Primordial Evolution, Sound Horizon Physics  
> **Reference:** Addendum to `EQUATIONS.md`, citing `Geometry of Resonance` and `WCT Cosmology Notebook v2`

---

## 📘 Overview

This document defines the **complete minimal equation set** (CM1–CM18) for cosmological modeling using Wave Confinement Theory (WCT) instead of General Relativity (GR). These equations:

- Replace inflation and Friedmann dynamics
- Generate CMB acoustic peaks from curvature principles
- Are partially implemented in your current WCT simulator

---

## CM1 — Fundamental Field Evolution

$$i\, \partial_t \psi = -\Theta[\psi] \cdot J[\psi]$$

$$\Theta[\psi] = -\frac{\Delta\psi}{\psi + \varepsilon \cdot e^{-\alpha|\psi|^2}}$$

$$J[\psi] = |\psi|^2 \cdot \nabla^2\psi \cdot \varepsilon_{\text{vac}}$$

---

## CM2 — Curvature-Spectral Tilt

$$P_{\text{prim}}(k) \sim k^{-\alpha_{\text{WCT}}}$$

$$n_s - 1 = -\alpha_{\text{WCT}}$$

$$\alpha_{\text{WCT}} = -\frac{d(\ln|\Theta(k)|)}{d(\ln k)}$$

---

## CM3 — Gravitational Potential from Θ

$$\Phi(k, t) = -C_\Phi \cdot \frac{\Theta(k, t)}{k^2}$$

---

## CM4 — Horizon-Entry Potential Decay

$$\partial_t \Phi(k, t) = -\Gamma(k, t) \cdot \Phi(k, t)$$

$$\Gamma(k, t) = \left| \frac{\partial_t \Theta(k, t)}{\Theta(k, t)} \right|$$

---

## CM5 — WCT Analog Oscillators

$$\ddot{\delta}_\gamma + c_s^2(t) \cdot k^2 \cdot \delta_\gamma = -k^2 \cdot \Phi$$

$$\ddot{\delta}_b + \mathcal{R}(t) \cdot c_s^2(t) \cdot k^2 \cdot \delta_\gamma = -k^2 \cdot \Phi$$

$$\mathcal{R}(t) = \frac{E_{\text{comp}}}{E_{\text{rad}}}$$

---

## CM6 — Sound Speed from Curvature Feedback

$$c_s^2(t) = \frac{1}{3(1 + \mathcal{R}(t))} \left[ 1 - \beta_{\text{curv}} \cdot \frac{E_{\text{curv}}(t)}{E_{\text{tot}}} \right]$$

---

## CM7 — Curvature Diffusion (Silk Analog)

$$\partial_t \delta_\gamma \to \partial_t \delta_\gamma - D_{\text{curv}}(t) \cdot k^2 \cdot \delta_\gamma$$

$$D_{\text{curv}}(t) = \frac{\langle|\nabla\psi|^2\rangle}{\langle|\psi|^2\rangle}$$

---

## CM8 — Initial Conditions (Sachs–Wolfe Form)

$$\delta_\gamma(0) = \delta_b(0) = -2 \cdot \Phi(k, 0)$$

$$\Phi(k, 0) = 2 \cdot C_\Phi \cdot \frac{\Theta(k, 0)}{k^2}$$

---

## CM9 — First-Order Mode Evolution

$$\dot{\delta}_\gamma = v_\gamma$$

$$\dot{v}_\gamma = -c_s^2 \cdot k^2 \cdot \delta_\gamma - k^2 \cdot \Phi$$

$$\dot{\delta}_b = v_b$$

$$\dot{v}_b = -\mathcal{R}(t) \cdot c_s^2 \cdot k^2 \cdot \delta_\gamma - k^2 \cdot \Phi$$

---

## CM10 — Tight Coupling Drag

$$\delta_b \leftarrow (1 - \varepsilon_{\text{drag}}) \cdot \delta_b + \varepsilon_{\text{drag}} \cdot \delta_\gamma$$

$$\varepsilon_{\text{drag}}(t) = \frac{E_{\text{exch}}}{E_{\text{comp}}}$$

---

## CM11 — Curvature Damping Envelope

$$D(k) = \exp\left( -\frac{k^2}{k_D^2} \right)$$

$$k_D^{-2} = \int_0^{t^*} D_{\text{curv}}(t)\, dt$$

---

## CM12 — Dimensionless Power Spectrum

$$\Delta^2(k) = \frac{k^3}{2\pi^2} \cdot P(k)$$

---

## CM13 — Peak Metrics

$$r_{21} = \frac{P(k_2)}{P(k_1)}, \quad r_{31} = \frac{P(k_3)}{P(k_1)}$$

$$s_{21} = \frac{k_2}{k_1}, \quad s_{31} = \frac{k_3}{k_1}$$

---

## CM14 — Peak Interpretation

- Fast Θ decay → $s_{ij}$ ↑
- High compression → $r_{31}$ ↑
- High radiative energy → $r_{21}$ ↓

---

## CM15 — Angular Scaling from $a_{\text{WCT}}$

$$k \to \frac{k}{a_{\text{WCT}}(t)}$$

$$a_{\text{WCT}}(t) = \left[ \frac{E_{\text{curv}}(0)}{E_{\text{curv}}(t)} \right]^{1/3}$$

---

## CM16 — Horizon Scale (WCT Form)

$$R_{\text{hor}}(t) = \int_0^t c_s(t')\, dt'$$

$$k_{\text{hor}} = \frac{2\pi}{R_{\text{hor}}}$$

---

## CM17 — Curvature Energy Conservation

$$E_{\text{curv}}(t) + E_{\text{grad}}(t) = E_{\text{tot}}$$

---

## CM18 — Closure Law (WCT Minimal Set)

$$\{ \text{CM1} + \text{CM2} + \text{CM3} + \text{CM4} + \text{CM5} + \text{CM7} \}$$

---

## CM19 — Acoustic Horizon from Θ

$$c_s(t) = \sqrt{ \frac{\partial P_{\text{curv}}}{\partial \rho_{\text{curv}}} }$$

---

## CM20 — Θ-Based Expansion Law

$$H(t) = \frac{\dot{a}_{\text{WCT}}}{a_{\text{WCT}}} = \sqrt{ \frac{\rho_\Theta(t)}{3|K|} }$$

---













# Wave Confinement Theory — Canonical Closure Layer

**Richard J. Reyes — WCT Master Framework**

This section defines the **five canonical closure equations** of Wave Confinement Theory (WCT).  
They organize the entire equation corpus (E1–E82, CLE1–CLE10, CM1–CM20) into a unified hierarchy.

These equations correspond to the same structural role that:

- the **Einstein equation** plays in General Relativity,
- the **Lagrangian + symmetry structure** plays in Quantum Field Theory.

They are not new assumptions.  
Each equation already exists in the WCT corpus, but here they are **centralized into a single canonical layer**.

---

# C1 — Master Action Principle

The dynamics of the wavefield are defined by the curvature-regularized WCT action:

$$
S[\psi] = \int L_{\text{WCT}}(\psi,\partial\psi,\Theta[\psi])\, d^4x
$$

with Lagrangian density

$$
L_{\text{WCT}}
=
|\partial_\mu \psi|^2
-
V(|\psi|^2)
+
\kappa
\left(
\frac{\Box \psi}{g(\psi)}
\right)^2
+
\theta \, \Theta[\psi]^2
+
\gamma
\left(
\frac{\Box \psi}{g(\psi)}
\right)
\Theta[\psi]
$$

where

$$
g(\psi) = \psi + \epsilon e^{-\alpha |\psi|^2}
$$

and

$$
\Theta[\psi] =
-\frac{\nabla^2\psi}{\psi + \epsilon e^{-\alpha|\psi|^2}}
$$

This action generates the field equations governing curvature-confined wave dynamics.

---

# C2 — Noether Current Conservation

Global phase symmetry

$$
\psi \rightarrow e^{i\alpha}\psi
$$

implies a conserved current

$$
\partial_\mu J^\mu = 0
$$

with

$$
J^\mu = \mathrm{Im}(\psi^* \partial^\mu \psi)
$$

This conservation law governs:

- probability density
- charge conservation
- global phase symmetry

and underlies several entropy and coherence relations in WCT.

---

# C3 — Phase Quantization (Loop Winding)

Closed phase loops satisfy the quantization condition

$$
\oint_\Gamma \nabla \theta \cdot d\ell = 2\pi n,
\quad n \in \mathbb{Z}.
$$

Equivalently,

$$
n = \frac{1}{2\pi} \oint \nabla \arg(\psi)\cdot d\ell.
$$

This quantization condition defines:

- topological winding
- particle identity
- curvature-locked eigenmodes.

It leads directly to the effective wavenumber relation

$$
k_{\text{eff}} = \frac{2\pi |n|}{L_s}.
$$

---

# C4 — Dispersion Relation

Wave excitations in WCT obey a curvature-modified dispersion relation

$$
\omega^2 = c^2 (k^2 + k_{\text{eff}}^2).
$$

Using

$$
k_{\text{eff}} = \frac{mc}{\hbar}
$$

this becomes the relativistic massive dispersion law

$$
\omega^2 = c^2 k^2 + \frac{m^2 c^4}{\hbar^2}.
$$

Curvature corrections modify this relation through the invariant

$$
W_\psi = -\frac{\nabla^2\psi}{\psi}.
$$
# C5 — Renormalization / Scale Flow

### Plain definition

The effective confinement curvature of WCT changes with observation scale.  
Renormalization is therefore defined first as geometric scale flow, and second as induced coupling flow.

### Symbolic form

Define the scale-dependent effective curvature

$$
\Sigma(\mu) := \langle \sigma \rangle_\mu.
$$

Its beta function is

$$
\beta_{\Sigma} := \mu \frac{d\Sigma}{d\mu}.
$$

Equivalently,

$$
\frac{d\Sigma}{d\ln \mu}
=
\beta_{\Sigma}(\Sigma;\kappa,\alpha,\theta,\gamma,\dots).
$$

Since

$$
m(\mu)=\frac{\hbar}{c}\Sigma(\mu),
$$

mass runs as

$$
\mu \frac{dm}{d\mu}
=
\frac{\hbar}{c}\beta_{\Sigma}.
$$

The action couplings also run:

$$
\beta_i := \mu \frac{d g_i}{d\mu},
\qquad
g_i \in \{\kappa,\alpha,\theta,\gamma,\dots\}.
$$

A scale-stable confinement sector satisfies

$$
\beta_{\Sigma}=0,
\qquad
\beta_i=0.
$$

### Context

This equation defines the renormalization structure of WCT:  
the confinement spectrum itself runs with scale, and the action couplings follow as effective parameters.

# Canonical Closure Structure

Together the five equations form the **closure layer of WCT**:

| Layer | Equation |
|------|----------|
| Action principle | C1 |
| Symmetry & conservation | C2 |
| Topological quantization | C3 |
| Wave propagation | C4 |
| Scale dependence | C5 |

All remaining equations in the WCT corpus (E1–E82, CLE1–CLE10, CM1–CM20) derive from or operate within this closure structure.

---

# Relationship to the Full WCT Equation Corpus

The closure equations govern the entire theory architecture:

| Domain | Derived Sections |
|------|------|
| Particle mass | E1–E8 |
| Phase-flux dynamics | E9–E16 |
| Curvature feedback | E17–E23 |
| Dimensional stability | E24–E27 |
| Entropy dynamics | E28–E34 |
| Computation limits | E35–E43 |
| Resonant cavities | E44–E48 |
| Topology emergence | spectral WCT module |
| Cosmology | CM1–CM20 |

Thus the WCT framework follows the hierarchy








# Correction Notes and Canonical Alignment

This section records notation, closure, and consistency corrections applied to the preceding analysis. No new physical assumptions are introduced. All changes align the presentation with the canonical Wave Confinement Theory (WCT) equation set (E17–E18, E30, E34, E43).

---

### 1. Lyapunov Energy Functional

Earlier drafts referenced the curvature penalty

$$E_\Theta = \int |\Theta[\psi]|^2\,dx$$

as a stability proxy. This has been corrected to the full WCT Lyapunov candidate

$$\mathcal{E}_{\text{WCT}}[\psi] = \int \left( |\nabla\psi|^2 + |\Theta[\psi]|^2 \right) dx,$$

consistent with the canonical formulation (E18). The quantity $E_\Theta$ is retained only as the curvature component within $\mathcal{E}_{\text{WCT}}$.

---

### 2. Spectral Curvature Closure

In the spectral analysis, the denominator of $\Theta[\psi]$ was approximated by a constant amplitude scale. This approximation is now made explicit via the effective closure

$$D_{\text{eff}}^2 := \langle |\psi|^2 \rangle + \varepsilon^2,$$

leading to the curvature cost

$$C_\Theta(k) = \frac{k^4}{D_{\text{eff}}^2}.$$

This clarifies that the $k^4$ spectral penalty arises from a mean–amplitude (weak–intermittency) closure rather than an exact identity.

---

### 3. Symbol Disambiguation in the Spectral Functional

To avoid collision with the nonlinear saturation parameter $\alpha$ appearing in the definition of $\Theta[\psi]$, the spectral free-energy weights have been relabeled:

$$\alpha \to \lambda_\Theta, \qquad \beta \to \lambda_{\text{ex}}.$$

No change in functional form or interpretation is implied.

---

### 4. Macro–Micro Control Parameter

The regime parameter governing entropy-dominated versus curvature-dominated behavior is redefined in observable terms as

$$\Xi = \frac{\int k^4 \rho(k)\, dk}{H}, \qquad H = -\sum_k P_k \log P_k,$$

eliminating reliance on an undefined "noise temperature" and aligning the definition with the spectral entropy (E30).

---

### 5. Entropy–Curvature Relation

The empirical entropy–curvature coupling is stated in its canonically bounded form

$$\frac{dH}{dt} \leq -\mu\, |\Theta[\psi]|^2,$$

consistent with the curvature–entropy tradeoff (E43) and its integrated variant (E34).

---

### 6. Isoelectronic Flow Consistency

The imaginary-time isoelectronic evolution equation is explicitly identified as a radially reduced, imaginary-time sector of the unified curvature–wavefield equation (UWCT), with ultraviolet smoothing and norm enforcement included for numerical well-posedness. No additional dynamical primitives are introduced.

---

### 7. Scope of Corrections

All corrections in this section are:

- notational or definitional clarifications,
- explicit statements of previously implicit closures,
- alignments with the canonical WCT master equations.

They do not alter the qualitative or quantitative conclusions of the empirical results, but ensure internal consistency, symbol hygiene, and direct traceability to the WCT core equation architecture.





---

# End of WCT Equations Master Document
