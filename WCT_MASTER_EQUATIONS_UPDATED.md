# Wave Confinement Theory — Audited Master Equation Architecture

**Richard J. Reyes**  
**Canonical master reference aligned with the 142-object WCT equation registry**

Wave Confinement Theory (WCT) is a proposed geometric wavefield framework in which curvature, phase, topology, spectral selection, and finite physical resources are used to model confined structures.

This document is an architectural index. It does not by itself prove the physical validity of WCT. The canonical equation corpus is divided into:

- 9 master objects: `M1`–`M5`, `M6A`, `M6B`, `M7`, `M8`;
- 83 canonical equations: `E1A`, `E1B`, `E2`–`E82`;
- 10 curvature-locking equations: `CLE1`–`CLE10`;
- 20 cosmology equations: `CM1`–`CM20`;
- 5 logarithmic and ghost-mode equations: `G1`, `EX`, `EY`, `EZ`, `FA`;
- 9 topology objects: `TOP1`–`TOP9`;
- 6 correction objects: `CORR1`–`CORR6`.

Thus,

$$
9+83+10+20+5+9+6=142.
$$

## Audit semantics

| Status | Meaning |
|---|---|
| `PASS` | The encoded algebraic, dimensional, limiting, or consistency check succeeds under its stated assumptions. |
| `CONDITIONAL` | Additional domain, sign, regularity, counting, or model assumptions are required. |
| `DEFINITION` | The object is a definition or ansatz, not a theorem. |
| `OPEN` | Resolution requires analysis, simulation, or experiment beyond the current symbolic audit. |
| `FAIL` | The encoded statement is contradicted by algebra, dimensions, logic, or a counterexample. |

Current corpus totals are

$$
51\ \mathrm{PASS}
+
32\ \mathrm{CONDITIONAL}
+
23\ \mathrm{DEFINITION}
+
36\ \mathrm{OPEN}
=
142,
$$

with no contradiction remaining in the current encoded specification.

---

# 1. Master Equation Architecture

## M1 — Curvature-Locking Functional

**Status:** `CONDITIONAL`

Let:

- \(\Gamma\) be a closed loop;
- \(s\) be arc length along \(\Gamma\);
- \(\varphi(s)\) be the phase;
- \(w(s)\ge 0\) be a loop weight;
- \(\kappa(s)\) be curvature;
- \(\tau(s)\) be torsion;
- \(\sigma(s)\) be the inverse-length curvature rate.

Define

$$
\sigma(s)=\sqrt{\kappa(s)^2+\tau(s)^2}.
$$

The locking functional is

$$
S_{\mathrm{lock}}[\varphi]
=
\oint_\Gamma
w(s)\left(\partial_s\varphi(s)-\sigma(s)\right)^2\,ds.
$$

With the winding constraint

$$
\oint_\Gamma \partial_s\varphi\,ds=2\pi n,
\qquad n\in\mathbb Z,
$$

the stationary phase profile is

$$
\partial_s\varphi(s)
=
\sigma(s)+\frac{\alpha_{\mathrm{lock}}}{w(s)},
$$

where

$$
\alpha_{\mathrm{lock}}
=
\frac{
2\pi n-\oint_\Gamma \sigma(s)\,ds
}{
\oint_\Gamma w(s)^{-1}\,ds
}.
$$

The weighted curvature average is

$$
\langle \sigma\rangle_w
=
\frac{
\oint_\Gamma w(s)\sigma(s)\,ds
}{
\oint_\Gamma w(s)\,ds
}.
$$

The mass identification

$$
m=\frac{\hbar}{c}\langle\sigma\rangle_w
$$

is conditional on the physical phase-curvature locking hypothesis.

**Primary descendants:** `E1A`–`E8`, `CLE1`–`CLE10`.

---

## M2 — Nonsingular Curvature Operator and Lyapunov Candidate

**Status:** `PASS` for denominator positivity and regularized reciprocal consistency; the full dynamical Lyapunov claim remains conditional on the chosen evolution.

Let:

- \(\psi(x,t)\in\mathbb C\) be the wavefield;
- \(\varepsilon>0\) be the node-regularization scale;
- \(\alpha\in\mathbb R\) be the saturation parameter;
- \(\Delta\) be the spatial Laplacian.

Define the modulus-squared regularized reciprocal

$$
R_\varepsilon(\psi)
=
\frac{
\overline{\psi}
}{
|\psi|^2+\varepsilon^2e^{-2\alpha|\psi|^2}
}.
$$

Its denominator satisfies

$$
|\psi|^2+\varepsilon^2e^{-2\alpha|\psi|^2}>0
\qquad
(\varepsilon>0).
$$

Define the nonsingular curvature operator

$$
\Theta_\varepsilon[\psi]
=
-(\Delta\psi)\,R_\varepsilon(\psi).
$$

For nonzero \(\psi\),

$$
R_\varepsilon(\psi)\longrightarrow\frac{1}{\psi}
\qquad
\text{as }\varepsilon\to0^+.
$$

Define the WCT energy candidate

$$
\mathcal E_{\mathrm{WCT}}[\psi]
=
\int_\Omega
\left(
|\nabla\psi|^2
+
|\Theta_\varepsilon[\psi]|^2
\right)\,dx.
$$

For an actual gradient flow

$$
\partial_t\psi
=
-\frac{\delta \mathcal E_{\mathrm{WCT}}}{\delta\overline{\psi}},
$$

one obtains formally

$$
\frac{d\mathcal E_{\mathrm{WCT}}}{dt}
=
-
\left\|
\frac{\delta\mathcal E_{\mathrm{WCT}}}
{\delta\overline{\psi}}
\right\|^2
\le0.
$$

This descent identity depends on the exact dynamical and functional-analytic setup; therefore the global Lyapunov interpretation is conditional.

**Primary descendants:** `E17`–`E23`, parts of `E44`–`E56`, `CLE1`–`CLE10`.

---

## M3 — Finite-Band Spectral Selector

**Status:** `PASS`

Let:

- \(A(x,t)\) be a pattern-forming amplitude;
- \(\mu\) be the linear growth parameter;
- \(g>0\) be the nonlinear saturation coefficient;
- \(b>0\) be the fourth-order damping coefficient;
- \(k_\star>0\) be the selected wavenumber.

The corrected Swift–Hohenberg form is

$$
\partial_tA
=
\mu A
-
g|A|^2A
-
b(\Delta+k_\star^2)^2A.
$$

For a Fourier mode \(A_k e^{ik\cdot x}\),

$$
(\Delta+k_\star^2)^2
\mapsto
(k_\star^2-|k|^2)^2,
$$

so the linear growth rate is

$$
\lambda(k)
=
\mu
-
b(|k|^2-k_\star^2)^2.
$$

Hence modes away from \(|k|=k_\star\) are damped.

The equivalent quartic rail

$$
\sigma(k)=r+a|k|^2-b|k|^4,
\qquad a,b>0,
$$

has stationary maximum

$$
k_\star=\sqrt{\frac{a}{2b}},
$$

and therefore

$$
\lambda_\star
=
\frac{2\pi}{k_\star}
=
2\pi\sqrt{\frac{2b}{a}}.
$$

**Primary descendants:** `E12`–`E16`, `E57`–`E64`.

---

## M4 — Dimensional Stability Threshold

**Status:** `PASS` for the Sobolev embedding threshold.

Let:

- \(n\) be spatial dimension;
- \(H^2(\Omega)\) be the second-order Sobolev space;
- \(L^\infty(\Omega)\) be the space of essentially bounded functions.

For a bounded smooth domain \(\Omega\subset\mathbb R^n\),

$$
H^2(\Omega)\hookrightarrow L^\infty(\Omega)
$$

whenever

$$
2>\frac n2.
$$

For integer \(n\), this gives

$$
n\le3.
$$

This proves an embedding threshold. It does not by itself prove that every WCT equilibrium is stable for \(n\le3\), nor that every equilibrium is unstable for \(n\ge4\).

Two distinct curvature estimates must be separated.

### \(L^2\) curvature estimate

If

$$
\psi\in H^2(\Omega)
$$

and

$$
|D_\varepsilon(\psi)|\ge\delta>0,
$$

then

$$
\|\Theta_\varepsilon[\psi]\|_{L^2}
\le
\delta^{-1}\|\Delta\psi\|_{L^2}.
$$

### \(L^\infty\) curvature estimate

To control \(\Theta_\varepsilon[\psi]\) pointwise, sufficient higher regularity is

$$
\psi\in H^s(\Omega),
\qquad
s>\frac n2+2.
$$

Under the corresponding denominator bound,

$$
\Theta_\varepsilon[\psi]\in L^\infty(\Omega).
$$

**Primary descendants:** `E24`–`E27`, `E65`–`E70`.

---

## M5 — Curvature-Bounded Computation

**Status:** `CONDITIONAL`

Let:

- \(\psi^{(t)}(x)\) be the state at discrete time \(t\);
- \(N(x)\) be the finite neighborhood of site \(x\);
- \(U\) be the local update map.

The WCC update is

$$
\psi^{(t+1)}(x)
=
U\!\left(
\psi^{(t)}(x),
\{\psi^{(t)}(y)\}_{y\in N(x)}
\right).
$$

A finite curvature-resource condition may be written

$$
\sum_y
\left(
|\nabla\psi^{(t)}(y)|^2
+
|\Theta_\varepsilon[\psi^{(t)}(y)]|^2
\right)\Delta V
\le C_\Theta.
$$

This defines a resource-bounded computational model.

The identifications

$$
P_{\mathrm{WCC}}=P,
\qquad
NP_{\mathrm{WCC}}=NP
$$

remain conditional on an explicit encoding, simulation theorem, and polynomial-overhead equivalence between the WCC machine and a standard Turing model.

The corrected pruning exponent uses retained fractions

$$
\rho_t(n)\in(0,1]
$$

rather than factors greater than one:

$$
\alpha(n)
=
1+
\frac1n
\sum_{t=1}^{m(n)}
\log_2\rho_t(n)
+
\beta(n).
$$

To obtain

$$
\alpha(n)<1,
$$

one must impose

$$
\beta(n)
<
-
\frac1n
\sum_{t=1}^{m(n)}
\log_2\rho_t(n).
$$

Configuration-count consequences remain conditional.

**Primary descendants:** `E28`–`E43`, `E71`–`E76`.

---

## M6A — Unified Linear Operator

**Status:** `DEFINITION`

Let:

- \(c_1,c_2,c_3,c_4\) be real coefficients;
- \(c_2>0\) enforce ultraviolet damping;
- \(\sigma\) be an inverse-length curvature scale;
- \(k_\star\) be the selected spectral radius;
- \(m\in\mathbb Z\) be a winding label;
- \(R>0\) be a geometric scale;
- \(n\) be spatial dimension;
- \(p>0\) be a scaling exponent.

Define

$$
\mathcal L_{\mathrm{WCT}}
=
c_1(\Delta+\sigma^2)
-
c_2(\Delta+k_\star^2)^2
+
ic_3m
+
c_4R^{-(2+n/p)}.
$$

The negative fourth-order sign is required so that

$$
-c_2(|k|^2-k_\star^2)^2
$$

damps off-shell modes.

This operator is a unifying linear ansatz. It is not a proof that every linear equation in the corpus is uniquely derived from it.

---

## M6B — Nonlinear Curvature Operator

**Status:** `OPEN`

Define

$$
\mathcal N_{\mathrm{curv}}[\psi]
=
-(\Delta\psi)
\frac{
\overline{\psi}
}{
|\psi|^2+\varepsilon^2e^{-2\alpha|\psi|^2}
}.
$$

The operator is nonsingular for \(\varepsilon>0\).

It is registered as a proposed nonlinear curvature primitive. The claims that it is unique, that it follows uniquely from one action, or that every nonlinear WCT equation reduces to it remain open theorem obligations.

---

## M7 — Full Curvature–Wavefield Evolution

**Status:** `PASS` for the encoded sign and dimensional consistency checks; physical closure remains conditional.

Let:

- \(g\) be the cubic coupling;
- \(\eta\) be the stochastic amplitude;
- \(\xi(t)\) be an external stochastic process;
- \(\circ\) denote Stratonovich multiplication.

The unified evolution ansatz is

$$
\begin{aligned}
\partial_t\psi
={}&
\mathcal N_{\mathrm{curv}}[\psi]
+
g|\psi|^2\psi
+
c_1(\Delta+\sigma^2)\psi \\
&-
c_2(\Delta+k_\star^2)^2\psi
+
ic_3m\psi
+
c_4R^{-(2+n/p)}\psi
+
\eta\psi\circ\xi(t),
\end{aligned}
$$

with

$$
c_2>0.
$$

This equation combines nonlinear curvature feedback, cubic saturation, Helmholtz structure, finite-band damping, winding, dimensional scaling, and stochastic forcing.

The equation is a master ansatz. Individual sectors require independent domain, regularity, existence, uniqueness, stability, and empirical validation.

---

## M8 — Curvature–Acoustic Cosmology System

**Status:** `OPEN`

The cosmology sector couples:

- a curvature field;
- an effective potential;
- photon-like and matter-like acoustic modes;
- diffusion;
- peak observables;
- an effective expansion law.

A representative open system is

$$
\Phi(k,t)
=
-C_\Phi\frac{\Theta_\varepsilon(k,t)}{k^2},
$$

$$
\partial_t\Phi(k,t)
=
-\Gamma(k,t)\Phi(k,t),
$$

$$
\dot\delta_\gamma=v_\gamma,
$$

$$
\dot v_\gamma
=
-c_s^2(t)k^2\delta_\gamma
-k^2\Phi
-D_{\mathrm{curv}}(t)k^2\delta_\gamma,
$$

$$
\dot\delta_b=v_b,
$$

$$
\dot v_b
=
-\mathcal R(t)c_s^2(t)k^2\delta_\gamma
-k^2\Phi.
$$

Here:

- \(\Phi\) is the proposed curvature-generated potential;
- \(\delta_\gamma\) is the photon-like perturbation;
- \(\delta_b\) is the matter-like perturbation;
- \(c_s\) is an effective sound speed;
- \(\mathcal R\) is a compression-to-radiation ratio;
- \(D_{\mathrm{curv}}\) is an effective diffusion coefficient.

These equations are phenomenological. The current audit does not establish that `CM1`–`CM20` are derived from `M7`, that they replace the Friedmann system, or that they reproduce calibrated cosmological data.

**Primary descendants:** `CM1`–`CM20`.

---

# 2. Master-to-Family Index

| Master object | Status | Core role | Main descendants |
|---|---|---|---|
| `M1` | `CONDITIONAL` | Loop locking and mass identification | `E1A`–`E8`, `CLE1`–`CLE10` |
| `M2` | `PASS` | Nonsingular curvature operator and energy candidate | `E17`–`E23`, `E44`–`E56` |
| `M3` | `PASS` | Finite-band spectral selection | `E12`–`E16`, `E57`–`E64` |
| `M4` | `PASS` | Sobolev dimensional threshold | `E24`–`E27`, `E65`–`E70` |
| `M5` | `CONDITIONAL` | Resource-bounded computation | `E28`–`E43`, `E71`–`E76` |
| `M6A` | `DEFINITION` | Unified linear ansatz | Linear sectors across the E-series |
| `M6B` | `OPEN` | Proposed nonlinear curvature primitive | Nonlinear curvature sectors |
| `M7` | `PASS` | Unified mixed evolution ansatz | Cross-family operator synthesis |
| `M8` | `OPEN` | Curvature-acoustic cosmology | `CM1`–`CM20` |

No separate `CL1`–`CL12` family is included in the current 142-object canonical registry. Any future `CL` family must receive stable IDs, formulas, checker assignments, and statuses before being called canonical.

---

# 3. Additional Registered Sectors

## Ghost-mode modulation — `G1`

**Status:** `PASS` as a bounded phenomenological ansatz.

Let:

- \(A_g\) be amplitude;
- \(k_\ell\) be logarithmic frequency;
- \(E>0\) be energy;
- \(E_0>0\) be reference energy;
- \(\phi\) be phase.

Define

$$
\delta_g(E)
=
A_g
\cos\!\left(
k_\ell\ln\frac{E}{E_0}+\phi
\right).
$$

Then

$$
|\delta_g(E)|\le |A_g|.
$$

Detector smearing and statistical detectability are separate empirical questions.

## Logarithmic field sector — `EX`, `EY`, `EZ`, `FA`

For \(\psi>0\), define

$$
u=\ln\psi.
$$

Then

$$
\nabla\psi=e^u\nabla u,
$$

$$
\Delta\psi
=
e^u\left(
\Delta u+|\nabla u|^2
\right),
$$

and therefore

$$
\frac{\Delta\psi}{\psi}
=
\Delta u+|\nabla u|^2.
$$

The viscous Hamilton–Jacobi equation

$$
\partial_tu
=
\Delta u+|\nabla u|^2
$$

is equivalent under \(\psi=e^u\) to

$$
\partial_t\psi=\Delta\psi.
$$

The filament relation

$$
|\nabla u|\sim\kappa
$$

remains conditional.

## Topology sector — `TOP1`–`TOP9`

These objects distinguish:

- definitions of spectral loops and WCT codimension;
- conditional claims about irreversible curvature descent;
- open claims about topology-energy bands and protein-particle correspondence.

Empirical simulation observations must remain separated from proved topological invariants.

## Correction sector — `CORR1`–`CORR6`

These objects record:

- the full Lyapunov candidate;
- the weak-intermittency spectral closure;
- symbol disambiguation;
- the observable macro-micro parameter;
- open entropy-curvature dynamics;
- open isoelectronic-flow closure.

---

# 4. Canonical Repository Roles

| Resource | Role |
|---|---|
| `MASTER_EQUATIONS.md` | Master architecture and scope |
| `EQUATIONS.md` | Canonical equation-family index |
| `equations/full_registry.yaml` | Stable IDs, checkers, and expected statuses |
| `wct_sympy/full_checks_*.py` | Executable SymPy audits |
| `interoperability/lean_map.yaml` | Links to existing Lean declarations |
| `rickyjreyes/wct-lean` | Kernel-checked definitions and theorems |

A SymPy `PASS` is not a Lean proof. The label `PROVED` should be reserved for declarations accepted by Lean.

---

# 5. Scope Statement

The corrected hierarchy is

$$
\text{master object}
\longrightarrow
\text{equation family}
\longrightarrow
\text{SymPy audit}
\longrightarrow
\text{Lean declaration where available}
\longrightarrow
\text{simulation or experiment where required}.
$$

The current specification is internally audit-clean. It is not analytically or empirically complete.

The remaining frontier is

$$
32\ \mathrm{CONDITIONAL}
+
36\ \mathrm{OPEN}
=
68
$$

claims requiring additional mathematical assumptions, PDE analysis, formal proof, numerical calibration, or experiment.

---

# 6. Citation

When using a specific equation, cite the corresponding WCT paper and stable equation ID rather than citing only this architecture document.

Relevant paper families include:

- Rest Energy from Density;
- Emergence of Effective Mass;
- Curvature-Locked \(\psi\)-Field Solitons;
- Phase-Flux Field;
- Resonant Cavity of Vector Fields;
- Self-Emergent Fourier Cymatics;
- Hard Upper Bound on Spatial Dimensionality;
- WCT computation and drift papers;
- JUNO ghost-mode phenomenology;
- WCT cosmology notebooks and addenda.

---

# End of Audited WCT Master Equation Architecture
