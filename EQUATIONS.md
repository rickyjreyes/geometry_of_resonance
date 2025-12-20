# Wave Confinement Theory — Canonical Equation Families (E1–E82)
**Richard J. Reyes — Master Reference (v1.0, Nov 2025)**  

This document is the authoritative catalogue of the core equations used across the WCT research volumes.  
Each entry contains: (i) plain definition, (ii) symbolic form, (iii) context (paper/module).

---

## 0. Symbol Table

| Symbol | Definition |
|--------|------------|
| ψ(x,t) | Wavefield (complex) |
| Θ[ψ] | Curvature-feedback operator |
| κ, τ | Curvature, torsion of loop |
| σ = κ² + τ² | Curvature-rate scalar |
| w(s) | Energy-density weight along loop |
| A(x,t) | Band-pass amplitude |
| Â_k | Fourier mode of A |
| H_k | Spectral entropy |
| E[ψ] | WCT Lyapunov functional |
| Λ | Winding-number Lagrange multiplier |
| k_eff | Effective wavenumber |
| α(n) | α-Drop exponent |
| U | Discrete WCC update rule |
| k_max | Energy-limited bandlimit |
| N_lanes | Spatial channel capacity (modes/lanes) |

---

## A. Rest Energy, Curvature, Loop Locking  
*(Rest Energy / Solenoidal Mass)*

**E1 — Curvature-rate scalar**  
Curvature-plus-torsion invariant along a loop Γ:  
σ(s) = κ(s)² + τ(s)²

**E2 — Weighted loop average**  
Density-weighted average of a scalar f along Γ:  
⟨f⟩_w = (∮_Γ w(s) f(s) ds) / (∮_Γ w(s) ds)

**E3 — Loop-locking action**  
Phase–curvature locking with winding constraint:  
S_eff[φ] = ∮_Γ w(s) (∂_s φ(s) − σ(s))² ds + Λ ( ∮_Γ ∂_s φ(s) ds − 2πn )

**E4 — Covariant locking (Euler–Lagrange)**  
Phase gradient with density correction:  
∂_s φ(s) = σ(s) + α / w(s)

with  
α = ( 2πn − ∮_Γ σ(s) ds ) / ( ∮_Γ ds / w(s) )

**E5 — Effective wavenumber identity**  
Equivalence of winding, loop-length, and curvature average:  
k_eff = 2π|n| / L_s = (1 / L_s) ∮_Γ σ(s) ds = ⟨σ⟩_w

where L_s := ∮_Γ ds

**E6 — Mass–curvature law**  
Rest energy and mass from effective curvature:  
E_rest = ℏc k_eff  
m = (ℏ / c) ⟨σ⟩_w

**E7 — Solenoidal mass law**  
Mass from curvature–torsion combination along Γ:  
m = (ℏ / c) ⟨κ² + τ²⟩_Γ

**E8 — Density-weighted lock identity**  
Integral identity for a locked loop:  
∮_Γ w ∂_s φ ds = ∮_Γ w σ ds + 2πα ( ∮_Γ ds ) / ( ∮_Γ ds / w(s) )

---

## B. Phase–Flux Field & Cymatic Rails

**E9 — Phase–flux constitutive axiom**  
Phase flux proportional to amplitude:  
S(x,t) = u(x,t) ∇θ(x,t)

**E10 — Radial shell quantization**  
Radial wavenumber quantization between shells:  
∫_{r₁}^{r₂} k_r(r) dr = 2πn

**E11 — PFF vorticity**  
Topological phase winding number:  
m(γ) = (1 / 2π) ∮_γ ∇θ · dℓ ∈ ℤ

**E12 — Finite-band dispersion rail**  
Band-limited growth rate:  
σ(k) = r + a|k|² − b|k|⁴, with a, b > 0

**E13 — Band-pass amplitude evolution**  
Nonlinear band-pass evolution:  
∂_t A = σ(−i∇) A − β |A|² A

**E14 — Band-pass Lyapunov functional**  
Energy functional for band-pass dynamics:  
E[A] = ∫ ( −r|A|² − a|∇A|² + b|ΔA|² + (β/2)|A|⁴ ) dx

**E15 — Modal growth bound**  
Amplitude-square inequality per mode:  
d/dt |Â_k|² ≤ 2 σ(k) |Â_k|² − c |Â_k|⁴

**E16 — Randomness → spectral concentration**  
Growth of initial broadband noise:  
P_k(t) ∝ P_k(0) e^{2σ(k)t}  
arg max_k P_k(t) → k*

---

## C. Curvature Feedback & Lyapunov Dynamics

**E17 — Curvature operator**  
Nonlinear curvature-feedback operator:  
Θ[ψ] = − ∇²ψ / ( ψ + ε e^{−α|ψ|²} )

**E18 — WCT Lyapunov functional**  
Curvature-plus-gradient energy:  
E[ψ] = ∫ ( c₁ |∇ψ|² + c₂ |Θ[ψ]|² ) dx

**E19 — Spectral-gap ↔ curvature invariant**  
Scaling relation between curvature and gap:  
Δ* ∼ ⟨σ⟩_w²

**E20 — Higher-order cavity Lagrangian**  
Fourth-order cavity model (schematic):  
L = f(ψ) ( κ S² + θ P² − γ S P − λ ψ² )

**E21 — 4th-order cavity Euler–Lagrange (schematic)**  
Generalized EL for Lagrangians with second derivatives:  
δL/δψ = ∂L/∂ψ − ∂_μ ( ∂L/∂(∂_μ ψ) ) + ∂_μ ∂_ν ( ∂L/∂(∂_μ ∂_ν ψ) ) = 0

**E22 — Effective metric**  
Matter-coupled metric deformation:  
g_μν^eff = η_μν + λ ( ∂_μ ψ ∂_ν ψ ) / (ρ c²) + δ η_μν ( W_ψ / W₀ )

**E23 — Enthalpic curvature relation**  
Local enthalpy as energy plus curvature gradient:  
h(ψ) ∝ W_ψ + χ |∇ψ|²

---

## D. Dimensionality & Functional Bounds

**E24 — n ≤ 3 stability bound**  
Embedding condition for pointwise control:  
H²(ℝⁿ) ↪ L^∞(ℝⁿ) ⇒ n ≤ 3

**E25 — Subcritical nonlinearity constraint**  
Nonlinearity exponent below critical Sobolev threshold:  
p < p_c(n)

**E26 — Curvature norm bound**  
Curvature control via Sobolev norm:  
‖Θ[ψ]‖_{L^∞} ≤ C ‖ψ‖_{H²}

**E27 — Finite-energy confinement**  
WCT finite-energy condition:  
∫_{ℝⁿ} ( |∇ψ|² + |Θ[ψ]|² ) dx < ∞

---

## E. α-Drop, Entropy Reduction, Pruning

**E28 — α-Drop exponent**  
Exponent defined from multiplicative pruning:  
α(n) = 1 + (1/n) ∑_{t=1}^{m(n)} log₂ q_t(n) + β(n)

with q_t(n) = (M_t + 1) / M_t

**E29 — Entropy-drop pruning**  
State-count decay per step:  
M_{t+1} ≤ e^{−Δ_t} M_t

**E30 — Spectral entropy**  
Entropy in Fourier space:  
H_k(t) = − ∑_k P_k(t) log P_k(t)

**E31 — Band-pass entropy drop**  
Curvature-induced entropy decrease (heuristic):  
Δ_t ≳ c₀ ( k*² Δ_t )

**E32 — α < 1 curvature-bounded search**  
Sub-exponential effective exploration:  
limsup_{n→∞} α(n) < 1

**E33 — Support shrinkage**  
Support size controlled by entropy:  
K_t ≤ e^{H_k(t)}

**E34 — Energy–entropy conversion**  
Curvature energy vs. entropy change:  
ΔE_t ≥ λ ΔH_k(t)

---

## F. WCC, Channel Capacity, P vs NP

**E35 — Curvature-locked fixed point**  
Locked WCT configuration:  
W_ψ = − ∇²ψ / ( ψ + ε e^{−α|ψ|²} )  
d/dt S[ψ] → 0  
∇W_ψ → 0

**E36 — Discrete WCC update**  
Local update rule with neighbourhood N(x):  
ψ^{(t+1)}(x) = U( ψ^{(t)}(x), { ψ^{(t)}(y) }_{y ∈ N(x)} )

**E37 — Bandlimit from energy**  
Maximal wavenumber from energy bound:  
k_max = C₁ E_max / (ℏc)

**E38 — Spatial channel capacity**  
Max mode count in volume V:  
N_lanes ≤ C₂ V k_max³

**E39 — Time-step polynomial bound**  
Max update steps for input size n:  
T_max(n) ≤ C₃ n^d

**E40 — P_WCC / NP_WCC**  
Model-relative identification:  
P_WCC = P  
NP_WCC = NP

**E41 — Curvature-bounded configuration count**  
Configuration count under curvature rails:  
|C_curv(n)| ≤ 2^{α(n) n}, with α(n) < 1

**E42 — Θ-information identity**  
Coherence information decay:  
d/dt I_coh[ψ] ∝ − |Θ[ψ]|²

**E43 — Curvature–entropy tradeoff**  
Spectral entropy decay from curvature:  
d/dt H_k(t) ≤ − μ |Θ[ψ]|²

---

## G. Resonant Cavity & Tokamak Scaling

**E44 — Θ-eigenmode quantization**  
Curvature eigenmodes:  
Θ[ψ_n] = λ_n ψ_n

**E45 — Effective Q-factor**  
Quality factor with loss region γ_loss:  
Q_eff = ω ( ∫ u dV ) / ( ∫_{γ_loss} u dV )

**E46 — Plasma–cavity curvature match**  
Matched curvature averages:  
⟨σ⟩_{w,plasma} ≈ ⟨σ⟩_{w,cavity}

**E47 — Power balance with curvature losses**  
Input vs loss and fusion:  
P_in = P_loss(ψ) + P_fusion(ψ)

**E48 — Stability via curvature gap**  
Core–edge curvature gap criterion:  
Δσ = ⟨σ⟩_core − ⟨σ⟩_edge > Δ_crit

---

# Supplemental Wave Confinement Theory Equation Families (E49–E82)

*(Second-tier structural laws across Geometry of Resonance, Self-Emergent Cymatics, Enthalpic Aether, Randomness, Dimensionality, and P v. NP papers.)*

---

## H. Geometry-of-Resonance Extensions (Curvature, Phase) — E49–E56

**E49 — Curvature-modified Helmholtz effective mass**  
Gap-induced effective mass and spectrum:  
m_eff² = Δ* c²  
ω_j² = c² λ_j + Δ*

**E50 — Phase-coherence functional**  
Global phase coherence measure:  
C[ψ] = ∫_Ω |∇θ|^{−1} |ψ|² dx

**E51 — Curvature–gradient commutator**  
Non-commutativity of curvature and gradient (schematic):  
[Θ, ∇] ψ = ∇ ( ∇²ψ / (ψ + …) ) − ∇²(∇ψ) / (ψ + …)

("…" uses the same denominator structure as in Θ[ψ] from E17.)

**E52 — Nonlinear curvature gain/loss balance**  
Global curvature gain vs gradient loss:  
G_σ = ∫ |Θ|² dx  
L_σ = ∫ |∇ψ|² dx

**E53 — Local curvature pressure**  
Curvature "pressure" density:  
p_Θ(x) = |Θ[ψ](x)|²

**E54 — Resonance-lock condition (stationary attractor)**  
Locked resonance state:  
∂_t ψ = 0  
δE[ψ] = 0  
∇Θ = 0

**E55 — Curvature-induced effective potential**  
Potential modified by curvature energy:  
V_eff(ψ) = V(|ψ|²) + κ |Θ[ψ]|²

**E56 — Phase-wall formation criterion**  
Phase-wall vs bulk curvature:  
|∇θ| ∼ σ_wall ≫ ⟨σ⟩_w

---

## I. Self-Emergent Fourier Cymatics (Swift–Hohenberg Structure) — E57–E64

**E57 — Swift–Hohenberg operator representation**  
Band-selective operator:  
SH[A] = (k*² + Δ)² A

**E58 — Band-selective Green's kernel**  
Spectral Green's function:  
G(k) = 1 / ( r + a (k² − k*²)² )

**E59 — Projection onto dominant annulus**  
Projection onto shell around k*:  
P_{k*} A = ∑_{k ∈ A*} Â_k e^{ik·x}

**E60 — Center-manifold amplitude equation**  
Reduced amplitude dynamics:  
∂_t A = μ A − g |A|² A + O(|A|⁴)

**E61 — Modal competition inequality**  
Competitive exclusion condition:  
g_{ij} > 0 ⇒ A_i A_j → exclusion

**E62 — Vortex-charge conservation**  
Topological charge conservation:  
∑_{i ∈ V} m_i = const

**E63 — Phase-lattice quantization**  
Phase from discrete vortex charges:  
θ(x) = ∑_i m_i arg(x − x_i)

**E64 — Locked modal support**  
Set of locked modes:  
K* = { k : ∂_t |Â_k|² = 0, ∂_t² |Â_k|² < 0 }

---

## J. Randomness & Structured Noise (α–θ–β Law) — E65–E72

**E65 — Heavy-tail emission law**  
Power-law tail for returns:  
P(|r| > x) ∼ x^{−α}

**E66 — Volatility recursion**  
GARCH-type recursion:  
σ_t² = ω + α r_{t−1}² + β σ_{t−1}²

**E67 — Extremal index**  
Clustering of extremes:  
θ = lim_{u → u*} P(max r_t ≤ u) / ( P(r_t ≤ u) )ⁿ

**E68 — Hawkes branching ratio**  
Self-excitation parameter:  
n = κ γ

**E69 — 1/f spectral slope**  
Power spectrum scaling:  
S(f) ∼ f^{−β}

**E70 — α–θ–β cross-domain relation**  
Heuristic cross-domain relation:  
β = 1 − 2/α + c (1 − θ)

**E71 — Curvature-bias estimator**  
Composite bias indicator:  
B_curv = ( β + 2/α ) + c (θ − 1)

**E72 — Phase-noise diffusion**  
Phase diffusion constant:  
D_θ = lim_{t → ∞} E[ (θ(t) − θ(0))² ] / (2t)

---

## K. Enthalpic Aether / Rotating Rings — E73–E78

**E73 — Enthalpic drive term**  
Driven–damped rotating NLS:  
i ∂_t ψ = −(1/2) ∇²ψ + V ψ + g |ψ|² ψ − i γ ψ + Ω L_z ψ

**E74 — Angular momentum density**  
Local angular momentum about z-axis:  
ℓ_z = Im( ψ̄ ∂_θ ψ )

**E75 — Rotation-lock plateau condition**  
Locked angular momentum:  
∂_t L_z → 0, with L_z > 0

**E76 — Annular confinement radius**  
Most-probable radius:  
r* = arg max_r |ψ(r)|²

**E77 — Ring-width scaling**  
Width vs interaction scale:  
w ∼ (g n₀)^{−1/2}

**E78 — Chirality selection inequality**  
Net positive rotation:  
∫ Im( ψ̄ ∂_θ ψ ) dx > 0

---

## L. Dimensionality, Sobolev Machinery — E79–E82

**E79 — Gagliardo–Nirenberg curvature bound**  
Interpolated curvature control:  
‖Θ‖_{L^∞} ≤ C ‖ψ‖_{H²}^{1−λ} ‖ψ‖_{L^∞}^λ

**E80 — Blow-up exclusion (subcritical regime)**  
Uniform H² bound in n ≤ 3:  
‖ψ(t)‖_{H²} ≤ K, for n ≤ 3

**E81 — Embedding slope constraint**  
Hölder continuity from Sobolev embedding:  
H²(Ω) ↪ C^{0,α}(Ω), with α = 1 − n/2

**E82 — Dimensional curvature-scaling heuristic**  
Scaling of curvature norm with radius:  
‖Θ‖_{L^p} ∼ R^{−(2 + n/p)}

---

# Section CL — Curvature-Locked ψ–Electron & Grand Transform Architecture

The CL-series contains structural WCT equations not present in E1–E82.

---

## CL1 — Curvature-Induced Displacement Field (Charge Operator)

**Field definitions**
- ψ(x): field
- ρ(x): density
- Θ[ψ]: curvature operator
- χ(ρ,Θ): curvature–phase susceptibility
- n̂: outward normal
- S: enclosing surface

**Equations**
- D_curv(x) := χ(ρ(x), Θ[ψ(x)]) · n̂(x)
- q_eff := ∮_S D_curv · dS

---

## CL2 — Dimensionless Eigenvalue System (α, e, m_e, λ_C)

**Definitions**
- R = ν λ_C
- a = μ r_e
- η := a / R = (μ/ν) α

**Energy closure**
- (E_curv + E_EM) / (m_e c²) = (1/ν) F_geo(η) + (1/(2μ)) κ_EM(η) = 1

---

## CL3 — Shell Quantization Condition

Q(a/R, n) = 0

---

## CL4 — Eigenmode-Selection Principle

η = η_*  
F_geo′(η_*) + … = 0

Electron corresponds to n = 1, lowest shell mode.

---

## CL5 — ψ-Field Emergent Metric

g_eff^{μν} = η^{μν} + κ ( ∂^μψ ∂^νψ̄ + ∂^μψ̄ ∂^νψ ) / |ψ|²

---

## CL6 — Curvature-Locking ⇒ H² Regularity

**Assume**
- Θ_min ≤ |Θ[ψ]| ≤ Θ_max
- |ψ| ≥ ψ_min
- |g(ψ)| ≥ g_min
- E_WCT[ψ] < ∞

**Then**  
ψ ∈ H²(ℝᵈ)

---

## CL7 — Curvature–Flux Charge Quantization

n_flux := (1/e) ∮_S D_curv · dS  
q_eff = e · n_flux

---

## CL8 — Effective ψ–Electron Lagrangian (Toroidal Reduction)

L_eff = w(s) ( ∂_s φ(s) − σ(s) )² + κ |Θ[ψ]|² + γ (□ψ / g(ψ)) Θ[ψ]

---

## CL9 — Shell–Shell Interaction Force

**Energy**  
E_int(R) = (q₁ q₂)/(4πε₀ R) − A exp(−R/R_a) + B exp(−R/R_r)

**Force**  
F_R = − dE_int / dR

---

## CL10 — Dimensional Unwinding Lemma (d > 3)

For d > 3:  
∃ ψ̃ ∈ H²(ℝᵈ) : ‖ψ̃ − ψ‖_{H²} < ε and n_lock(ψ̃) = 0

---

## CL11 — Semantic Curvature Complexity (WCC Measure)

**Complexity functional**  
C_Θ(f, n) = inf_M sup_{|x| ≤ n} ∑_{t=0}^{T(n)} Φ(ψ_x^t)

**Curvature energy density**  
Φ(ψ) = ∑_y ( |∇ψ(y)|² + |Θ[ψ](y)|² ) ΔV

---

## CL12 — Gauge Connection from Internal Phase Rotors (GTA)

**Definitions**
- U = exp(i θ^a T^a)
- T^a: generators

**Connection**  
A_μ^a T^a = −(i/g) (∂_μ U) U^{−1}

---

# Section CLE — ψ-Electron Collapse & Curvature-Locking Equations

The CLE-series contains all previously missing equations that finalize the curvature-locked ψ-electron derivation.

---

## CLE1 — Feedback Derivative (Locking Condition)

F′_fb(σ) = −2σ

Derived from the variational collapse condition under Θ[ψ] = σ(ψ).

---

## CLE2 — Integrated Feedback Functional

F_fb(σ) = −σ² + C

Set C = 0.

---

## CLE3 — Curvature–Feedback Cancellation

σ² + F_fb(σ) = 0

Eliminates curvature energy from the ψ-electron, leaving only gradient energy.

---

## CLE4 — ψ-Electron Eigenmode Equation (Helmholtz Form)

∇²ψ = −σ_★² ψ

---

## CLE5 — Toroidal Laplacian (Geometric Reduction)

∇²ψ = (1/r) ∂_θ(r ∂_θ ψ) + 1/(R + r cos θ)² ∂_φ² ψ

---

## CLE6 — Separation Ansatz

ψ(θ,φ) = f(θ) e^{imφ}

ψ-electron corresponds to the minimal winding m = 1.

---

## CLE7 — Reduced Angular ODE (Thin-Torus Limit)

f″(θ) + σ_★² f(θ) = 0

Only smooth 2π-periodic solution under curvature locking is constant f.

---

## CLE8 — ψ-Electron Eigenmode Solution

ψ(θ,φ) = A e^{iφ}

Unique curvature-locked toroidal eigenmode.

---

## CLE9 — Electron Radius from Curvature

R = 1/σ_★

For the electron: R ≈ 386.3 fm.

---

## CLE10 — Curvature Scalar Identity

W_ψ = −∇²ψ / ψ = σ_★²

Ties together curvature scalar, eigenmode equation, and feedback-collapsed ψ-electron solution.

---

## G1 — Ghost-mode modulation (JUNO phenomenology)

δ_g(E) = A_g cos(k_ℓ ln(E/E₀) + φ)

---

# 🌌 Wave Confinement Theory (WCT) — Cosmology Equation Set (CM1–CM18)

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

# 🔷 WCT Cosmology Equation Set (CM1–CM20)

> All physics derives from ψ and Θ[ψ]. No GR, no Λ, no SM plasma.  
> One curvature field. One locking operator. Fully closed dynamics.

---

## CM1 — Fundamental Field Evolution

i ∂_t ψ = −Θ[ψ] · J[ψ]  
Θ[ψ] = −Δψ / (ψ + ε · e^{−α|ψ|²})  
J[ψ] = |ψ|² · ∇²ψ · ε_vac

---

## CM2 — Curvature-Spectral Tilt

P_prim(k) ∼ k^{−α_WCT}  
n_s − 1 = −α_WCT  
α_WCT = −d(ln|Θ(k)|)/d(ln k)

---

## CM3 — Gravitational Potential from Θ

Φ(k, t) = −C_Φ · Θ(k, t) / k²

---

## CM4 — Horizon-Entry Potential Decay

∂_t Φ(k, t) = −Γ(k, t) · Φ(k, t)  
Γ(k, t) = |∂_t Θ(k, t) / Θ(k, t)|

---

## CM5 — WCT Analog Oscillators

δ̈_γ + c_s²(t) · k² · δ_γ = −k² · Φ  
δ̈_b + ℛ(t) · c_s²(t) · k² · δ_γ = −k² · Φ  
ℛ(t) = E_comp / E_rad

---

## CM6 — Sound Speed from Curvature Feedback

c_s²(t) = [1 / (3(1 + ℛ(t)))] · [1 − β_curv · E_curv(t) / E_tot]

---

## CM7 — Curvature Diffusion (Silk Analog)

∂_t δ_γ → ∂_t δ_γ · D_curv(t) · k² · δ_γ  
D_curv(t) = ⟨|∇ψ|²⟩ / ⟨|ψ|²⟩

---

## CM8 — Initial Conditions (Sachs–Wolfe Form)

δ_γ(0) = δ_b(0) = −2 · Φ(k, 0)  
Φ(k, 0) = 2 · C_Φ · Θ(k, 0) / k²

---

## CM9 — First-Order Mode Evolution

δ̇_γ = v_γ  
v̇_γ = −c_s² · k² · δ_γ − k² · Φ  
δ̇_b = v_b  
v̇_b = −ℛ(t) · c_s² · k² · δ_γ − k² · Φ

---

## CM10 — Tight Coupling Drag

δ_b ← (1 − ε_drag) · δ_b + ε_drag · δ_γ  
ε_drag(t) = E_exch / E_comp

---

## CM11 — Curvature Damping Envelope

D(k) = exp(−k² / k_D²)  
k_D⁻² = ∫₀^{t*} D_curv(t) dt

---

## CM12 — Dimensionless Power Spectrum

Δ²(k) = (k³ / 2π²) · P(k)

---

## CM13 — Peak Metrics

r₂₁ = P(k₂)/P(k₁),  r₃₁ = P(k₃)/P(k₁)  
s₂₁ = k₂ / k₁,  s₃₁ = k₃ / k₁

---

## CM14 — Peak Interpretation

Fast Θ decay → s_{ij} ↑  
High compression → r₃₁ ↑  
High radiative energy → r₂₁ ↓

---

## CM15 — Angular Scaling from a_WCT

k → k / a_WCT(t)  
a_WCT(t) = [E_curv(0) / E_curv(t)]^{1/3}

---

## CM16 — Horizon Scale (WCT Form)

R_hor(t) = ∫₀^t c_s(t′) dt′  
k_hor = 2π / R_hor

---

## CM17 — Curvature Energy Conservation

E_curv(t) + E_grad(t) = E_tot

---

## CM18 — Closure Law (WCT Minimal Set)

{ CM1 + CM2 + CM3 + CM4 + CM5 + CM7 }

---

## CM19 — Acoustic Horizon from Θ

c_s(t) = √( ∂P_curv / ∂ρ_curv )

---

## CM20 — Θ-Based Expansion Law

H(t) = ȧ_WCT / a_WCT = √( ρ_Θ(t) / 3|K| )

---

# End of WCT Equations Master Document
