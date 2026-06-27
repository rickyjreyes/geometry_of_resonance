# Wave Confinement Theory — Corrected Full Equation Registry

**Richard J. Reyes — Canonical audit-aligned reference**

**Revision:** 2.0  
**Date:** June 27, 2026  
**Supersedes:** *Canonical Equation Families v1.0, November 2025*

This document replaces the earlier equation catalogue with the current audit-clean specification. It contains all **142 registered objects**:

- $M1$–$M8$, with $M6A$ and $M6B$ separated;
- $E1A$, $E1B$, $E2$–$E82$;
- $CLE1$–$CLE10$;
- $G1$, $EX$, $EY$, $EZ$, $FA$;
- $CM1$–$CM20$;
- $TOP1$–$TOP9$;
- $CORR1$–$CORR6$.

## Audit meaning

| Status | Meaning |
|---|---|
| ✅ **PASS** | The encoded algebraic, dimensional, logical, limit, or counterexample check passes under its declared assumptions. |
| ⚠️ **CONDITIONAL** | Additional domain, sign, regularity, counting, model, or empirical assumptions are required. |
| ◻️ **DEFINITION** | A definition, ansatz, or bookkeeping object; not a theorem. |
| ○ **OPEN** | Requires analysis, simulation, formal proof, or experiment beyond the present symbolic audit. |
| ❌ **FAIL** | A current encoded statement is contradicted. There are no remaining `FAIL` entries in this revision. |

A SymPy `PASS` is **not** a Lean proof or empirical validation. `PROVED` should be reserved for declarations accepted by Lean.

## Current totals

$$51\ {\rm PASS} + 32\ {\rm CONDITIONAL} + 23\ {\rm DEFINITION} + 36\ {\rm OPEN} = 142.$$

## Principal corrections from v1.0

1. The node-sensitive reciprocal is now modulus-squared and nonsingular for $\varepsilon>0$.
2. The weighted locking identity uses $+\alpha_{\rm lock}L_s$.
3. Swift-Hohenberg fourth-order damping has the correct negative sign.
4. $H^2$ gives the stated $L^2$ curvature estimate; $L^\infty$ control requires higher regularity.
5. Alpha-drop uses retained fractions $\rho_t\in(0,1]$.
6. The entropy-support inequality is $e^H\le K$.
7. Quality factor uses loss **power**, and fusion is a source in the energy balance.
8. The effective-mass law is $m_{\rm eff}^2=\hbar^2\Delta_\omega^\star/c^4$.
9. The selected wavelength is $2\pi\sqrt{2b/a}$.
10. Coherence length is a spectral second-moment or gradient-ratio scale.
11. The CLE Euler-Lagrange equation includes its fourth-order term.
12. The CLE convention is $W_\psi=\sigma_\star^2$ and $R=1/\sigma_\star$.
13. Periodic angular modes form an integer family; the torus mode is not unique without added selection principles.

## Core symbols

| Symbol | Definition |
|---|---|
| $\psi(x,t)$ | Complex wavefield |
| $\Theta_\varepsilon[\psi]$ | Nonsingular regularized curvature operator |
| $\kappa,\tau$ | Curve curvature and torsion, units $L^{-1}$ |
| $\sigma_{\rm dens}=\kappa^2+\tau^2$ | Curvature-rate density, units $L^{-2}$ |
| $\sigma=\sqrt{\kappa^2+\tau^2}$ | Curvature spectral rate, units $L^{-1}$ |
| $w(s)$ | Nonnegative loop weight |
| $A(x,t)$ | Finite-band amplitude |
| $\widehat A_k$ | Fourier coefficient |
| $H_k$ | Normalized spectral Shannon entropy |
| $k_\star$ | Selected finite-band wavenumber |
| $\alpha_{\rm lock}$ | Locking integration constant |
| $\alpha$ | Curvature-regularizer parameter |
| $\alpha(n)$ | Alpha-drop exponent; distinguished by its argument |
| $\mathcal E$ | Energy or Lyapunov candidate |
| $U$ | Discrete WCC update rule |

---


# Master systems


## M1 — Curvature-locking functional

**Status:** ⚠️ `CONDITIONAL`

Let
$$\sigma(s):=\sqrt{\kappa(s)^2+\tau(s)^2}, \qquad \langle f\rangle_w:= \frac{\oint_\Gamma w f\,ds}{\oint_\Gamma w\,ds}, \quad \oint_\Gamma w\,ds>0.$$
Define
$$S_{\rm lock}[\varphi] = \oint_\Gamma w(s)\bigl(\partial_s\varphi-\sigma\bigr)^2\,ds.$$
Under phase-curvature locking and the stated winding assumptions,
$$m=\frac{\hbar}{c}\langle\sigma\rangle_w.$$


## M2 — Nonsingular curvature operator and Lyapunov candidate

**Status:** ✅ `PASS`

For $\varepsilon>0$, define
$$R_\varepsilon(\psi) := \frac{\overline{\psi}} {|\psi|^2+\varepsilon^2e^{-2\alpha|\psi|^2}},$$
$$\Theta_\varepsilon[\psi] := -(\Delta\psi)R_\varepsilon(\psi).$$
The denominator is strictly positive for every complex $\psi$. For $\psi\neq0$,
$$R_\varepsilon(\psi)\longrightarrow \frac1\psi \qquad(\varepsilon\to0).$$
A Lyapunov candidate is
$$\mathcal E_{\rm WCT}[\psi] = \int_\Omega \left( |\nabla\psi|^2+|\Theta_\varepsilon[\psi]|^2 \right)\,dx.$$


## M3 — Finite-band spectral selector

**Status:** ✅ `PASS`

$$\partial_tA = \mu A-g|A|^2A-b(\Delta+k_\star^2)^2A, \qquad b>0.$$
Its Fourier growth symbol contains
$$-b\bigl(|k|^2-k_\star^2\bigr)^2,$$
so sufficiently off-shell ultraviolet modes are damped.


## M4 — Dimensional stability threshold

**Status:** ✅ `PASS`

For a standard Sobolev domain,
$$H^2(\Omega)\hookrightarrow L^\infty(\Omega) \quad\text{when}\quad 2>\frac n2.$$
For integer spatial dimension,
$$n\le3.$$
This is an embedding threshold, not by itself a complete nonlinear-stability theorem.


## M5 — Curvature-bounded computation

**Status:** ⚠️ `CONDITIONAL`

$$\psi^{(t+1)}(x) = U\!\left( \psi^{(t)}(x), \{\psi^{(t)}(y):y\in N(x)\} \right).$$
Complexity claims require a specified encoding, precision model, update cost, and finite curvature-resource bound.


## M6A — Unified linear operator

**Status:** ◻️ `DEFINITION`

$$\mathcal L_{\rm WCT} = c_1(\Delta+\sigma^2) -c_2(\Delta+k_\star^2)^2 +i\,c_3m +c_4R^{-(2+n/p)}, \qquad c_2>0.$$


## M6B — Nonlinear curvature operator

**Status:** ○ `OPEN`

$$\mathcal N_{\rm curv}[\psi] = -(\Delta\psi) \frac{\overline{\psi}} {|\psi|^2+\varepsilon^2e^{-2\alpha|\psi|^2}}.$$
The operator is well-defined for $\varepsilon>0$; uniqueness of this nonlinear closure remains open.


## M7 — Full curvature-wavefield equation

**Status:** ✅ `PASS`

$$\partial_t\psi = \mathcal N_{\rm curv}[\psi] +g|\psi|^2\psi +c_1(\Delta+\sigma^2)\psi -c_2(\Delta+k_\star^2)^2\psi +i\,c_3m\psi +c_4R^{-(2+n/p)}\psi +\eta\psi\circ\xi(t), \qquad c_2>0.$$
The explicit negative fourth-order term supplies finite-band ultraviolet damping. Other dynamical claims require separate analysis.


## M8 — Curvature-acoustic cosmology system

**Status:** ○ `OPEN`

Representative closure relations are
$$\Phi(k,t)=-C_\Phi\frac{\Theta(k,t)}{k^2},$$
$$\delta_g(E) = A_g\cos\!\left( k_\ell\ln\frac{E}{E_0}+\phi \right).$$
The full cosmology sector remains open pending derivation, parameter closure, and calibrated data tests.


# A. Rest energy, curvature, and loop locking


## E1A — Curvature-rate density

**Status:** ✅ `PASS`

$$\sigma_{\rm dens}(s)=\kappa(s)^2+\tau(s)^2, \qquad [\sigma_{\rm dens}]=L^{-2}.$$


## E1B — Curvature spectral rate

**Status:** ✅ `PASS`

$$\sigma(s)=\sqrt{\kappa(s)^2+\tau(s)^2}, \qquad [\sigma]=L^{-1}.$$


## E2 — Weighted loop average

**Status:** ✅ `PASS`

For $w(s)\ge0$ and $\oint_\Gamma w\,ds>0$,
$$\langle f\rangle_w = \frac{\oint_\Gamma w(s)f(s)\,ds} {\oint_\Gamma w(s)\,ds}.$$


## E3 — Loop-locking action

**Status:** ✅ `PASS`

$$S_{\rm eff}[\varphi] = \oint_\Gamma w(\partial_s\varphi-\sigma)^2\,ds + \Lambda \left( \oint_\Gamma\partial_s\varphi\,ds-2\pi n \right), \qquad n\in\mathbb Z.$$
For $w\ge0$, the mismatch term is nonnegative.


## E4 — Covariant locking solution

**Status:** ✅ `PASS`

Stationarity gives
$$\partial_s\varphi = \sigma+\frac{\alpha_{\rm lock}}{w},$$
where
$$\alpha_{\rm lock} = \frac{ 2\pi n-\oint_\Gamma\sigma\,ds }{ \oint_\Gamma ds/w },$$
assuming $w>0$ and $\oint_\Gamma ds/w<\infty$.


## E5 — Effective-wavenumber chain

**Status:** ⚠️ `CONDITIONAL`

Define
$$L_s:=\oint_\Gamma ds, \qquad k_{\rm wind}:=\frac{2\pi|n|}{L_s}, \qquad k_\sigma:=\frac1{L_s}\oint_\Gamma\sigma\,ds.$$
Then
$$k_{\rm wind}=k_\sigma=\langle\sigma\rangle_w$$
only under compatible orientation, exact integrated locking, and a weight condition such as uniform $w$. Without those assumptions the three quantities are distinct.


## E6 — Mass-curvature law

**Status:** ✅ `PASS`

$$E_{\rm rest}=\hbar c\,k_{\rm eff}, \qquad m=\frac{\hbar}{c}k_{\rm eff}.$$
When E5 applies,
$$m=\frac{\hbar}{c}\langle\sigma\rangle_w.$$


## E7 — Solenoidal mass law

**Status:** ✅ `PASS`

$$m = \frac{\hbar}{c} \left\langle \sqrt{\kappa^2+\tau^2} \right\rangle_\Gamma.$$
The averaging measure must be specified; the dimensional relation is valid.


## E8 — Corrected weighted-lock identity

**Status:** ✅ `PASS`

Substituting E4 gives
$$\boxed{ \oint_\Gamma w\,\partial_s\varphi\,ds = \oint_\Gamma w\,\sigma\,ds + \alpha_{\rm lock}L_s }.$$
The earlier extra factor
$$2\pi\oint ds/\oint ds/w$$
was incorrect.


# B. Phase-flux and finite-band selection


## E9 — Phase-flux constitutive relation

**Status:** ◻️ `DEFINITION`

$$\mathbf S(x,t)=u(x,t)\nabla\theta(x,t).$$
A conservation equation, when imposed, is
$$\partial_tu+\nabla\cdot\mathbf S=0.$$
The constitutive relation itself is a definition.


## E10 — Radial shell quantization

**Status:** ✅ `PASS`

$$\int_{r_1}^{r_2}k_r(r)\,dr=2\pi n, \qquad n\in\mathbb Z.$$
Both sides are dimensionless.


## E11 — Phase winding

**Status:** ✅ `PASS`

$$m(\gamma) = \frac1{2\pi} \oint_\gamma\nabla\theta\cdot d\boldsymbol\ell \in\mathbb Z,$$
provided $\psi\neq0$ on the loop and the phase is continuous modulo $2\pi$.


## E12 — Finite-band dispersion rail

**Status:** ✅ `PASS`

To avoid conflict with curvature $\sigma$, write the growth rate as
$$\lambda_{\rm grow}(k) = r+a|k|^2-b|k|^4, \qquad a,b>0.$$
Its nonzero stationary maximum is
$$k_\star=\sqrt{\frac{a}{2b}}.$$
Equivalently,
$$\lambda_{\rm grow}(k) = \mu-b(|k|^2-k_\star^2)^2, \qquad \mu=r+\frac{a^2}{4b}.$$


## E13 — Band-pass amplitude evolution

**Status:** ⚠️ `CONDITIONAL`

A form consistent with E12 is
$$\partial_tA = (r-a\Delta-b\Delta^2)A-\beta|A|^2A,$$
or equivalently
$$\partial_tA = \mu A-b(\Delta+k_\star^2)^2A-\beta|A|^2A.$$
Boundary conditions and the sign of $\beta$ must be declared.


## E14 — Band-pass Lyapunov functional

**Status:** ⚠️ `CONDITIONAL`

For the centered Swift-Hohenberg form,
$$\mathcal E[A] = \int_\Omega \left[ -\mu|A|^2 +b|(\Delta+k_\star^2)A|^2 +\frac{\beta}{2}|A|^4 \right]dx.$$
Gradient-flow descent requires compatible boundary conditions and normalization conventions.


## E15 — Modal growth bound

**Status:** ⚠️ `CONDITIONAL`

$$\frac{d}{dt}|\widehat A_k|^2 \le 2\lambda_{\rm grow}(k)|\widehat A_k|^2 -c|\widehat A_k|^4, \qquad c>0.$$
The quartic modal estimate requires a model-specific nonlinear projection bound.


## E16 — Linear spectral concentration

**Status:** ✅ `PASS`

For the linearized dynamics,
$$P_k(t)=P_k(0)e^{2\lambda_{\rm grow}(k)t}.$$
If the maximizing shell is isolated and initially populated,
$$\operatorname*{arg\,max}_kP_k(t)\to k_\star.$$


# C. Curvature feedback and Lyapunov dynamics


## E17 — Nonsingular curvature-feedback operator

**Status:** ✅ `PASS`

$$R_\varepsilon(\psi) = \frac{\overline\psi} {|\psi|^2+\varepsilon^2e^{-2\alpha|\psi|^2}},$$
$$\boxed{ \Theta_\varepsilon[\psi] = -(\Delta\psi)R_\varepsilon(\psi) }.$$
For $\varepsilon>0$, the denominator is strictly positive for all complex $\psi$. This replaces
$$-\Delta\psi/(\psi+\varepsilon e^{-\alpha|\psi|^2})$$,
whose denominator can vanish.


## E18 — WCT Lyapunov candidate

**Status:** ⚠️ `CONDITIONAL`

$$\mathcal E[\psi] = \int_\Omega \left( c_1|\nabla\psi|^2 +c_2|\Theta_\varepsilon[\psi]|^2 \right)dx.$$
It is nonnegative when $c_1,c_2\ge0$. Lyapunov descent still requires a specified evolution equation satisfying $d\mathcal E/dt\le0$.


## E19 — Gap-curvature scaling

**Status:** ⚠️ `CONDITIONAL`

Separate the spatial and frequency gaps:
$$\Delta_k^\star\sim\langle\sigma\rangle_w^2, \qquad [\Delta_k^\star]=L^{-2},$$
$$\Delta_\omega^\star:=c^2\Delta_k^\star, \qquad [\Delta_\omega^\star]=T^{-2}.$$
The proportionality constant and spectral derivation remain model-dependent.


## E20 — Higher-order cavity quadratic sector

**Status:** ✅ `PASS`

Let
$$S:=\frac{\Box\psi}{g(\psi)}, \qquad P:=\frac{\Delta\psi}{g(\psi)}.$$
The quadratic sector is
$$Q(S,P)=\kappa S^2+\theta P^2-\gamma SP.$$
It is positive semidefinite when
$$\kappa\ge0,\qquad \theta\ge0,\qquad \gamma^2\le4\kappa\theta.$$
Potential and prefactor terms require separate sign assumptions.


## E21 — Second-derivative Euler-Lagrange equation

**Status:** ✅ `PASS`

For $\mathcal L(\psi,\partial\psi,\partial^2\psi)$,
$$\frac{\delta\mathcal L}{\delta\psi} = \frac{\partial\mathcal L}{\partial\psi} -\partial_\mu \frac{\partial\mathcal L}{\partial(\partial_\mu\psi)} +\partial_\mu\partial_\nu \frac{\partial\mathcal L} {\partial(\partial_\mu\partial_\nu\psi)} =0.$$


## E22 — Effective metric ansatz

**Status:** ⚠️ `CONDITIONAL`

$$g_{\mu\nu}^{\rm eff} = \eta_{\mu\nu} +\lambda_g \frac{\partial_\mu\overline\psi\,\partial_\nu\psi} {\rho c^2} +\delta_g\,\eta_{\mu\nu}\frac{W_\psi}{W_0}.$$
The coefficients must carry the units needed to make both corrections dimensionless, and signature/nondegeneracy must be checked.


## E23 — Enthalpic curvature relation

**Status:** ⚠️ `CONDITIONAL`

$$h(\psi) = C_h\left( W_\psi+\chi|\nabla\psi|^2 \right).$$
The constants must reconcile dimensions and the relation requires a constitutive derivation.


# D. Dimensionality and functional bounds


## E24 — Sobolev embedding threshold

**Status:** ✅ `PASS`

$$H^2(\Omega)\hookrightarrow L^\infty(\Omega) \quad\Longrightarrow\quad 2>\frac n2.$$
For integer $n$,
$$n\le3.$$


## E25 — Critical Sobolev exponent

**Status:** ◻️ `DEFINITION`

For $n>2$,
$$p_c(n)=\frac{n+2}{n-2}, \qquad p<p_c(n)$$
is the stated subcriticality condition.


## E26 — Corrected curvature $L^2$ bound

**Status:** ✅ `PASS`

If $\psi\in H^2(\Omega)$ and
$$|R_\varepsilon(\psi(x))|\le\delta^{-1} \quad\text{a.e.},$$
then
$$\boxed{ \|\Theta_\varepsilon[\psi]\|_{L^2} \le \delta^{-1}\|\Delta\psi\|_{L^2} }.$$
An $L^\infty$ curvature bound does not follow from $H^2$ in arbitrary dimension; see E69.


## E27 — Finite-energy confinement

**Status:** ◻️ `DEFINITION`

$$\int_{\mathbb R^n} \left( |\nabla\psi|^2 + |\Theta_\varepsilon[\psi]|^2 \right)dx <\infty.$$


# E. Alpha-drop, entropy reduction, and pruning


## E28 — Corrected alpha-drop exponent

**Status:** ✅ `PASS`

Let $\rho_t(n)\in(0,1]$ be retained fractions. Define
$$\alpha(n) = 1+\frac1n\sum_{t=1}^{m(n)}\log_2\rho_t(n) +\beta(n).$$
Then $\alpha(n)<1$ requires
$$\beta(n) < -\frac1n\sum_t\log_2\rho_t(n).$$
This replaces $q_t=(M_t+1)/M_t>1$, which could not produce $\alpha<1$ with $\beta\ge0$.


## E29 — Entropy-drop pruning

**Status:** ✅ `PASS`

$$M_{t+1}\le e^{-\Delta_t}M_t, \qquad \Delta_t\ge0.$$
Iteration gives
$$M_T \le M_0\exp\!\left(-\sum_{t=0}^{T-1}\Delta_t\right).$$


## E30 — Spectral entropy

**Status:** ✅ `PASS`

For $P_k\ge0$ and $\sum_kP_k=1$,
$$H_k=-\sum_kP_k\ln P_k.$$
For support size $K$,
$$0\le H_k\le\ln K.$$


## E31 — Conditional entropy-production bound

**Status:** ⚠️ `CONDITIONAL`

Define the entropy drop
$$\Delta H_t:=H_k(t)-H_k(t+1).$$
A noncircular model-specific bound has the form
$$\Delta H_t\ge c_0\mathcal D_t, \qquad \mathcal D_t\ge0.$$
The dissipation functional $\mathcal D_t$ must be derived from the dynamics; the earlier self-referential expression was removed.


## E32 — Subexponential exploration condition

**Status:** ⚠️ `CONDITIONAL`

$$\limsup_{n\to\infty}\alpha(n)<1.$$
This follows only if the retained-fraction and $\beta(n)$ bounds in E28 hold uniformly with sufficient margin.


## E33 — Corrected support-entropy relation

**Status:** ✅ `PASS`

For a distribution supported on $K_t$ modes,
$$H_k(t)\le\ln K_t,$$
hence
$$\boxed{ e^{H_k(t)}\le K_t }.$$
The previous inequality $K_t\le e^{H_k(t)}$ was reversed.


## E34 — Energy-entropy conversion

**Status:** ○ `OPEN`

For entropy reduction
$$\Delta H_k:=H_{\rm before}-H_{\rm after}\ge0,$$
the proposed cost relation is
$$\Delta E_{\rm cost}\ge\lambda\,\Delta H_k, \qquad \lambda>0.$$
A derivation of $\lambda$ remains open.


# F. WCC, channel capacity, and complexity


## E35 — Curvature-locked fixed point

**Status:** ◻️ `DEFINITION`

A stationary locked configuration satisfies
$$\partial_t\psi=0, \qquad \nabla\Theta_\varepsilon[\psi]=0, \qquad \frac{d}{dt}S[\psi]=0.$$


## E36 — Discrete WCC update

**Status:** ◻️ `DEFINITION`

$$\psi^{(t+1)}(x) = U\!\left( \psi^{(t)}(x), \{\psi^{(t)}(y)\}_{y\in N(x)} \right).$$


## E37 — Bandlimit from energy

**Status:** ✅ `PASS`

$$k_{\max} = C_1\frac{E_{\max}}{\hbar c},$$
with dimensionless $C_1$.


## E38 — Spatial channel capacity

**Status:** ✅ `PASS`

In three spatial dimensions,
$$N_{\rm lanes} \le C_2Vk_{\max}^3,$$
with dimensionless $C_2$.


## E39 — Polynomial update bound

**Status:** ◻️ `DEFINITION`

$$T_{\max}(n)\le C_3n^d.$$
This defines the assumed computational resource class.


## E40 — WCC complexity identification

**Status:** ⚠️ `CONDITIONAL`

$$P_{\rm WCC}\cong P, \qquad NP_{\rm WCC}\cong NP.$$
The identification requires an explicit encoding and polynomial simulation in both directions.


## E41 — Curvature-bounded configuration count

**Status:** ⚠️ `CONDITIONAL`

$$|C_{\rm curv}(n)| \le 2^{\alpha(n)n}, \qquad \alpha(n)<1.$$
E28 alone does not prove this counting bound; an injective coding or combinatorial argument is required.


## E42 — Theta-information relation

**Status:** ○ `OPEN`

$$\frac{d}{dt}I_{\rm coh}[\psi] = -\lambda_I \int_\Omega|\Theta_\varepsilon[\psi]|^2dx.$$
The information functional and coupling $\lambda_I$ require derivation.


## E43 — Curvature-entropy tradeoff

**Status:** ○ `OPEN`

$$\frac{dH_k}{dt} \le -\mu \int_\Omega|\Theta_\varepsilon[\psi]|^2dx, \qquad \mu>0.$$
This remains an analytic/empirical claim.


# G. Cavity, effective mass, and phase structure


## E44 — Theta eigenmode problem

**Status:** ◻️ `DEFINITION`

$$\Theta_\varepsilon[\psi_n] = \lambda_n\psi_n.$$
Because $\Theta_\varepsilon$ is nonlinear, the spectral problem and normalization must be specified carefully.


## E45 — Corrected quality factor

**Status:** ✅ `PASS`

$$\boxed{ Q_{\rm eff} = \omega\frac{U}{P_{\rm loss}} }$$
where
$$U=\int_\Omega u\,dV$$
is stored energy and $P_{\rm loss}$ is loss power. An integral of energy density over a loss region is not a power unless a loss rate is included.


## E46 — Plasma-cavity curvature match

**Status:** ○ `OPEN`

$$\langle\sigma\rangle_{w,\rm plasma} \approx \langle\sigma\rangle_{w,\rm cavity}.$$
A measurable matching tolerance and transfer mechanism remain open.


## E47 — Corrected power balance

**Status:** ✅ `PASS`

$$\boxed{ \frac{dW}{dt} = P_{\rm in} + P_{\rm fusion} - P_{\rm loss} - P_{\rm out} }.$$
At stationarity,
$$P_{\rm in}+P_{\rm fusion} = P_{\rm loss}+P_{\rm out}.$$
Fusion is a source, not a loss term.


## E48 — Curvature-gap stability criterion

**Status:** ⚠️ `CONDITIONAL`

$$\Delta\sigma = \langle\sigma\rangle_{\rm core} - \langle\sigma\rangle_{\rm edge} > \Delta_{\rm crit}.$$
The threshold and direction of the inequality must be calibrated to a specified stability observable.


## E49 — Corrected effective-mass gap law

**Status:** ✅ `PASS`

If
$$\omega_j^2=c^2\lambda_j+\Delta_\omega^\star, \qquad [\Delta_\omega^\star]=T^{-2},$$
then comparison with
$$\omega^2=c^2k^2+\frac{m_{\rm eff}^2c^4}{\hbar^2}$$
gives
$$\boxed{ m_{\rm eff}^2 = \frac{\hbar^2}{c^4}\Delta_\omega^\star }.$$


## E50 — Phase-coherence functional

**Status:** ⚠️ `CONDITIONAL`

$$\mathcal C[\psi] = \int_\Omega \frac{|\psi|^2}{|\nabla\theta|}\,dx.$$
The definition requires a regularization or lower bound
$$|\nabla\theta|\ge\delta>0$$
on the integration region.


## E51 — Curvature-gradient commutator

**Status:** ✅ `PASS`

For a smooth scalar denominator $D\neq0$, define
$$\Theta_D[\psi]:=-\frac{\Delta\psi}{D}.$$
Then, where $\nabla\Delta=\Delta\nabla$,
$$[\nabla,\Theta_D]\psi := \nabla(\Theta_D[\psi])-\Theta_D[\nabla\psi] = \frac{\Delta\psi}{D^2}\nabla D.$$
This is the exact commutator generated by the spatially varying denominator.


## E52 — Curvature gain and gradient loss

**Status:** ◻️ `DEFINITION`

$$G_\sigma:=\int_\Omega|\Theta_\varepsilon[\psi]|^2dx, \qquad L_\sigma:=\int_\Omega|\nabla\psi|^2dx.$$


## E53 — Curvature pressure density

**Status:** ✅ `PASS`

$$p_\Theta(x) := c_2|\Theta_\varepsilon[\psi](x)|^2.$$
It is the local curvature contribution to E18.


## E54 — Resonance-lock condition

**Status:** ⚠️ `CONDITIONAL`

$$\partial_t\psi=0, \qquad \delta\mathcal E[\psi]=0, \qquad \nabla\Theta_\varepsilon[\psi]=0.$$
Simultaneous satisfaction requires existence and regularity results.


## E55 — Curvature-induced effective potential

**Status:** ◻️ `DEFINITION`

$$V_{\rm eff}(\psi) = V(|\psi|^2) + \kappa|\Theta_\varepsilon[\psi]|^2.$$


## E56 — Phase-wall criterion

**Status:** ⚠️ `CONDITIONAL`

$$|\nabla\theta|_{\rm wall} \sim \sigma_{\rm wall} \gg \langle\sigma\rangle_w.$$
The comparison scale and wall-detection threshold must be defined.


# H. Swift-Hohenberg and spectral projection


## E57 — Swift-Hohenberg shell operator

**Status:** ✅ `PASS`

$$\mathcal{SH}[A] = (\Delta+k_\star^2)^2A.$$
Its Fourier symbol is
$$(|k|^2-k_\star^2)^2.$$


## E58 — Band-selective Green kernel

**Status:** ⚠️ `CONDITIONAL`

For
$$\mathcal L=r+a(\Delta+k_\star^2)^2,$$
the Fourier Green kernel is
$$G(k) = \frac1{r+a(|k|^2-k_\star^2)^2}.$$
It is defined only away from zeros of the denominator.


## E59 — Projection onto a dominant annulus

**Status:** ✅ `PASS`

$$\mathcal A^\star := \left\{ k\in\mathbb Z^d: \bigl||k|-k_\star\bigr|\le\Delta k \right\},$$
$$(P_{k_\star}A)(x) = \sum_{k\in\mathcal A^\star} \widehat A_ke^{ik\cdot x}.$$
With a fixed annulus,
$$P_{k_\star}^2=P_{k_\star}.$$


## E60 — Center-manifold amplitude equation

**Status:** ◻️ `DEFINITION`

$$\partial_T\mathcal A = \mu\mathcal A-g|\mathcal A|^2\mathcal A.$$


## E61 — Pattern-formation threshold

**Status:** ✅ `PASS`

In the continuum,
$$r_c = \min_k a(|k|^2-k_\star^2)^2 = 0.$$
For a discrete domain, equality requires an admissible mode on the selected shell.


## E62 — Spectral energy concentration

**Status:** ✅ `PASS`

For nonzero total spectral energy,
$$\eta(t) = \frac{ \sum_{k\in\mathcal A^\star}|\widehat A_k|^2 }{ \sum_k|\widehat A_k|^2 }, \qquad 0\le\eta(t)\le1.$$


## E63 — Entropic mode selection

**Status:** ◻️ `DEFINITION`

$$k_\star = \operatorname*{arg\,min}_k \left[ H_k+\lambda_\Theta C_\Theta(k) \right].$$


## E64 — Corrected selected wavelength

**Status:** ✅ `PASS`

From E12,
$$k_\star=\sqrt{\frac{a}{2b}}.$$
Therefore
$$\boxed{ \lambda_\star = \frac{2\pi}{k_\star} = 2\pi\sqrt{\frac{2b}{a}} }.$$


# I. Sobolev structure and dimensional bounds


## E65 — Critical Sobolev exponent

**Status:** ✅ `PASS`

For $n>2$,
$$p_c(n)=\frac{n+2}{n-2}.$$


## E66 — Gagliardo-Nirenberg interpolation

**Status:** ⚠️ `CONDITIONAL`

$$\|u\|_{L^p} \le C \|\nabla u\|_{L^2}^{\theta} \|u\|_{L^2}^{1-\theta}.$$
The allowed $p,\theta,n$, domain, and boundary assumptions must be specified.


## E67 — Failure of $H^2	o L^\infty$ above three dimensions

**Status:** ✅ `PASS`

For $n>3$, the embedding $H^2\hookrightarrow L^\infty$ fails in general; equivalently, there exist $H^2$ functions not controlled in $L^\infty$.


## E68 — Localized energy estimate

**Status:** ⚠️ `CONDITIONAL`

A model-dependent localized estimate is
$$\int_{B_R}|\nabla\psi|^2dx \le CR^{n-2}\|\psi\|_{H^1}^2.$$
Its exponent and constant require a precise scaling regime and domain hypotheses.


## E69 — Corrected high-regularity curvature bound

**Status:** ✅ `PASS`

If
$$\psi\in H^s(\Omega), \qquad s>\frac n2+2,$$
and the regularized reciprocal is uniformly bounded, then Sobolev embedding gives
$$\Theta_\varepsilon[\psi]\in L^\infty(\Omega).$$
This replaces the false general claim $H^2\Rightarrow\Theta\in L^\infty$.


## E70 — Dimensional stability criterion

**Status:** ⚠️ `CONDITIONAL`

Use the one-way criterion
$$n\le3, \qquad H^2\hookrightarrow L^\infty, \qquad p<p_c(n)$$
as separate hypotheses for the proposed stability analysis. They are not a proven biconditional characterization of all stable WCT solutions.


# J. Computational resource bounds


## E71 — Physical computation resource bound

**Status:** ⚠️ `CONDITIONAL`

$$TVk_{\max}^3 \le C_{\rm phys}.$$
The constant must have units of time, and the interpretation depends on the update rate, precision, and physical encoding.


## E72 — Curvature-pruned search space

**Status:** ⚠️ `CONDITIONAL`

$$|S_{\rm eff}(n)| \le 2^{\alpha(n)n}.$$
A counting theorem linking the physical pruning process to discrete configurations is required.


## E73 — Polynomial verification

**Status:** ◻️ `DEFINITION`

$$V(x,w)\in P, \qquad |w|=\operatorname{poly}(|x|).$$


## E74 — Curvature separation conjecture

**Status:** ○ `OPEN`

$$\inf_n \frac{\log|NP_n|}{\log|P_n|} >1.$$
The finite-size families $P_n,NP_n$ must first be defined.


## E75 — Physical-oracle impossibility

**Status:** ○ `OPEN`

$$\nexists\, O: O(\psi)=\operatorname*{arg\,min}_\psi\mathcal E[\psi] \quad\text{in polynomial time}.$$
This is a complexity claim requiring a formal computational model and reduction.


## E76 — WCC complexity equivalence

**Status:** ⚠️ `CONDITIONAL`

$$P_{\rm WCC}=P \quad\Longrightarrow\quad \text{WCC polynomially simulates the declared physical-computation model}.$$
The implication requires explicit translations and cost bounds.


# K. Entropy and information dynamics


## E77 — Mutual-information decay

**Status:** ○ `OPEN`

$$\frac{d}{dt}I(\psi_t;\psi_0) \le -\gamma\mathcal E_\Theta[\psi_t].$$
The probability law, channel, and regularity assumptions remain open.


## E78 — Fisher-information curvature bound

**Status:** ○ `OPEN`

$$\mathcal I_F[\psi] \ge c\int_\Omega|\Theta_\varepsilon[\psi]|^2dx.$$
A common probability density and geometric derivation are required.


## E79 — Entropy-production rate

**Status:** ◻️ `DEFINITION`

$$\dot\Sigma = \frac{dH_k}{dt} + \frac{\mathcal E_\Theta}{T_{\rm eff}}.$$
The sign convention and units must be fixed when used physically.


## E80 — Landauer-type bound

**Status:** ⚠️ `CONDITIONAL`

If $\Delta H_{\rm bits}$ is measured in bits,
$$\Delta E \ge k_BT_{\rm eff}\ln2\, \Delta H_{\rm bits}.$$
For entropy in nats, omit the factor $\ln2$.


## E81 — Corrected coherence length

**Status:** ✅ `PASS`

With normalized spectral weights $p_k$,
$$\boxed{ \xi_{\rm coh} = \left( \sum_kp_k|k|^2 \right)^{-1/2} }.$$
Equivalently, when the integrals exist,
$$\boxed{ \xi_{\rm coh} = \sqrt{ \frac{\int_\Omega|\psi|^2dx} {\int_\Omega|\nabla\psi|^2dx} } }.$$
This replaces $\sqrt{\mathcal E/H_k}$, which did not generally have units of length.


## E82 — Information-geometry tensor

**Status:** ◻️ `DEFINITION`

$$g_{ij}^{({\rm info})} = \left\langle \partial_i\Theta_\varepsilon\, \partial_j\Theta_\varepsilon \right\rangle.$$
Positive definiteness and coordinate invariance require additional conditions.


# Curvature-locking equations


## CLE1 — Curvature-locking functional

**Status:** ◻️ `DEFINITION`

Use the inverse-length convention $[\sigma_\star]=L^{-1}$:
$$S[\psi] = \int_\mathcal M \left[ |\nabla\psi|^2 + |W_\psi-\sigma_\star^2|^2 \right]\sqrt g\,d^3x, \qquad W_\psi:=-\frac{\Delta\psi}{\psi}.$$


## CLE2 — Corrected curvature-lock Euler-Lagrange equation

**Status:** ✅ `PASS`

For the real one-dimensional reduction
$$q:=-\frac{\psi_{xx}}{\psi}-\sigma_\star^2,$$
the generalized Euler-Lagrange equation is
$$\boxed{ q\frac{\psi_{xx}}{\psi^2} -\psi_{xx} -\frac{d^2}{dx^2}\left(\frac q\psi\right) =0 }.$$
The last fourth-order term was missing from the earlier expression.


## CLE3 — Curvature-locking condition

**Status:** ◻️ `DEFINITION`

$$W_\psi=\sigma_\star^2.$$
Both sides have units $L^{-2}$.


## CLE4 — Locked-field equation

**Status:** ✅ `PASS`

$$-\Delta\psi = \sigma_\star^2\psi.$$


## CLE5 — Thin/product-torus Laplacian

**Status:** ⚠️ `CONDITIONAL`

Under a flat product or thin-torus approximation,
$$\Delta\psi \approx \frac1{R^2}\partial_\theta^2\psi + \frac1{r^2}\partial_\phi^2\psi.$$
The exact embedded-torus Laplace-Beltrami operator contains metric-dependent terms.


## CLE6 — Separation ansatz

**Status:** ✅ `PASS`

For
$$\psi(\theta,\phi)=f(\theta)g(\phi),$$
CLE4 and CLE5 give
$$\frac{f''}{f} + \frac{R^2}{r^2}\frac{g''}{g} = -\sigma_\star^2R^2.$$


## CLE7 — Periodic angular mode family

**Status:** ✅ `PASS`

The periodic reduced equation
$$f''+m^2f=0$$
has the full family
$$\boxed{ f(\theta)=A\cos(m\theta)+B\sin(m\theta), \qquad m\in\mathbb Z_{\ge0} }.$$
The constant solution is only the $m=0$ member, not the unique periodic solution.


## CLE8 — Selected torus eigenmode

**Status:** ⚠️ `CONDITIONAL`

$$\psi(\theta,\phi)=Ae^{i\phi}$$
is one admissible winding-one mode. Uniqueness requires additional lowest-mode, chirality, normalization, phase, and boundary-selection principles.


## CLE9 — Electron radius from curvature

**Status:** ✅ `PASS`

$$R=\frac1{\sigma_\star}.$$
For $\sigma_\star=m_ec/\hbar$,
$$R=\frac{\hbar}{m_ec}\approx386.16\ {\rm fm}.$$


## CLE10 — Curvature scalar chain

**Status:** ✅ `PASS`

$$\boxed{ W_\psi = -\frac{\Delta\psi}{\psi} = \sigma_\star^2 }, \qquad R=\sigma_\star^{-1}.$$


# Logarithmic and ghost equations


## G1 — Log-periodic ghost modulation

**Status:** ✅ `PASS`

For $E>0$ and $E_0>0$,
$$\delta_g(E) = A_g\cos\!\left( k_\ell\ln\frac E{E_0}+\phi \right),$$
with
$$|\delta_g(E)|\le|A_g|.$$


## EX — Logarithmic field representation

**Status:** ✅ `PASS`

For a positive real field $\psi>0$, let
$$u=\ln\psi, \qquad \psi=e^u.$$
Then
$$\nabla\psi=e^u\nabla u,$$
$$\Delta\psi=e^u(\Delta u+|\nabla u|^2),$$
and
$$\frac{\Delta\psi}{\psi} = \Delta u+|\nabla u|^2.$$


## EY — Log-curvature evolution

**Status:** ✅ `PASS`

If
$$\partial_tu = \Delta u+|\nabla u|^2,$$
then the logarithmic dynamics are equivalent to the diffusion equation in EZ, provided $\psi=e^u>0$.


## EZ — Cole-Hopf reduction

**Status:** ✅ `PASS`

With
$$\psi=e^u,$$
EY gives
$$\partial_t\psi = e^u\partial_tu = e^u(\Delta u+|\nabla u|^2) = \Delta\psi.$$
Thus
$$\boxed{\partial_t\psi=\Delta\psi}.$$


## FA — Filament-localization condition

**Status:** ⚠️ `CONDITIONAL`

$$|\nabla u| \sim \kappa_{\rm core}.$$
A norm, tolerance, scale, and dynamical derivation are required.


# Curvature-acoustic cosmology


## CM1 — Fundamental field evolution

**Status:** ○ `OPEN`

$$i\partial_t\psi = -\Theta_\varepsilon[\psi]\,J[\psi],$$
$$J[\psi] = |\psi|^2\Delta\psi\,\varepsilon_{\rm vac}.$$
Coefficient dimensions and derivation remain open.


## CM2 — Curvature-spectral tilt

**Status:** ○ `OPEN`

$$P_{\rm prim}(k)\propto k^{-\alpha_{\rm WCT}},$$
$$n_s-1=-\alpha_{\rm WCT},$$
$$\alpha_{\rm WCT} = -\frac{d\ln|\Theta(k)|}{d\ln k}.$$


## CM3 — Potential from curvature

**Status:** ○ `OPEN`

$$\Phi(k,t) = -C_\Phi\frac{\Theta(k,t)}{k^2}.$$


## CM4 — Horizon-entry potential decay

**Status:** ○ `OPEN`

$$\partial_t\Phi = -\Gamma\Phi,$$
$$\Gamma(k,t) = \left| \frac{\partial_t\Theta(k,t)}{\Theta(k,t)} \right|,$$
on the domain $\Theta\neq0$.


## CM5 — Curvature-acoustic oscillators

**Status:** ○ `OPEN`

$$\ddot\delta_\gamma +c_s^2k^2\delta_\gamma = -k^2\Phi,$$
$$\ddot\delta_b +\mathcal R\,c_s^2k^2\delta_\gamma = -k^2\Phi,$$
$$\mathcal R = \frac{E_{\rm comp}}{E_{\rm rad}}.$$


## CM6 — Sound speed from curvature feedback

**Status:** ○ `OPEN`

$$c_s^2(t) = \frac1{3(1+\mathcal R(t))} \left[ 1-\beta_{\rm curv} \frac{E_{\rm curv}(t)}{E_{\rm tot}} \right].$$
Positivity requires the bracketed factor to be nonnegative.


## CM7 — Curvature diffusion

**Status:** ○ `OPEN`

A phenomenological damping replacement is
$$\dot\delta_\gamma = v_\gamma - D_{\rm curv}(t)k^2\delta_\gamma,$$
$$D_{\rm curv}(t) = \frac{\langle|\nabla\psi|^2\rangle} {\langle|\psi|^2\rangle}.$$


## CM8 — Initial conditions

**Status:** ○ `OPEN`

Use CM3 consistently:
$$\delta_\gamma(0)=\delta_b(0)=-2\Phi(k,0),$$
$$\Phi(k,0) = -C_\Phi\frac{\Theta(k,0)}{k^2}.$$
The earlier extra factor and opposite sign were removed.


## CM9 — First-order mode system

**Status:** ○ `OPEN`

$$\dot\delta_\gamma=v_\gamma, \qquad \dot v_\gamma=-c_s^2k^2\delta_\gamma-k^2\Phi,$$
$$\dot\delta_b=v_b, \qquad \dot v_b=-\mathcal R c_s^2k^2\delta_\gamma-k^2\Phi.$$


## CM10 — Tight-coupling drag

**Status:** ○ `OPEN`

$$\delta_b \leftarrow (1-\varepsilon_{\rm drag})\delta_b + \varepsilon_{\rm drag}\delta_\gamma,$$
$$\varepsilon_{\rm drag} = \frac{E_{\rm exch}}{E_{\rm comp}}, \qquad 0\le\varepsilon_{\rm drag}\le1.$$


## CM11 — Curvature damping envelope

**Status:** ○ `OPEN`

$$D(k) = \exp\!\left(-\frac{k^2}{k_D^2}\right),$$
$$k_D^{-2} = \int_0^{t_\star}D_{\rm curv}(t)\,dt.$$
Dimensional consistency requires the time-dependent diffusion coefficient to carry units $L^2/T$.


## CM12 — Dimensionless power spectrum

**Status:** ○ `OPEN`

$$\Delta^2(k) = \frac{k^3}{2\pi^2}P(k).$$


## CM13 — Peak metrics

**Status:** ○ `OPEN`

$$r_{21} = \frac{P(k_2)}{P(k_1)}, \qquad r_{31} = \frac{P(k_3)}{P(k_1)},$$
$$s_{21} = \frac{k_2}{k_1}, \qquad s_{31} = \frac{k_3}{k_1}.$$


## CM14 — Peak-response interpretation

**Status:** ○ `OPEN`

Proposed qualitative relations:
$$\text{faster }\Theta\text{ decay}\Rightarrow s_{ij}\uparrow,$$
$$\text{larger compression}\Rightarrow r_{31}\uparrow,$$
$$\text{larger radiative fraction}\Rightarrow r_{21}\downarrow.$$


## CM15 — WCT angular scaling

**Status:** ○ `OPEN`

$$k_{\rm phys} = \frac{k}{a_{\rm WCT}(t)},$$
$$a_{\rm WCT}(t) = \left[ \frac{E_{\rm curv}(0)} {E_{\rm curv}(t)} \right]^{1/3}.$$


## CM16 — Acoustic horizon

**Status:** ○ `OPEN`

$$R_{\rm hor}(t) = \int_0^tc_s(t')\,dt',$$
$$k_{\rm hor} = \frac{2\pi}{R_{\rm hor}}.$$


## CM17 — Curvature-energy closure

**Status:** ○ `OPEN`

$$E_{\rm curv}(t) + E_{\rm grad}(t) = E_{\rm tot},$$
for a closed sector with no external source or loss.


## CM18 — Minimal cosmology closure set

**Status:** ○ `OPEN`

$$\mathfrak C_{\rm min} = \{\mathrm{CM1},\mathrm{CM2},\mathrm{CM3}, \mathrm{CM4},\mathrm{CM5},\mathrm{CM7}\}.$$
This is a bookkeeping closure, not a derivation of cosmology.


## CM19 — Acoustic speed from curvature equation of state

**Status:** ○ `OPEN`

$$c_s^2 = \frac{\partial P_{\rm curv}} {\partial\rho_{\rm curv}},$$
where the derivative must be taken along a specified thermodynamic or dynamical path.


## CM20 — Theta-based expansion ansatz

**Status:** ○ `OPEN`

$$H(t) = \frac{\dot a_{\rm WCT}}{a_{\rm WCT}} = \sqrt{ \frac{\rho_\Theta(t)}{3|K|} }.$$
The constant $K$ must carry the units needed for $H^2$, and the equation requires independent derivation.


# Topology and spectral emergence


## TOP1 — Closed spectral-loop representation

**Status:** ◻️ `DEFINITION`

$$\gamma(s) = \sum_{k=1}^{K} \left[ a_k\cos(ks)+b_k\sin(ks) \right], \qquad s\in[0,2\pi).$$
This is the chosen configuration representation; emergence of the basis is a separate empirical claim.


## TOP2 — WCT loop-energy functional

**Status:** ◻️ `DEFINITION`

$$\mathcal E_{\rm loop}[\gamma] = \int_\gamma\kappa^2ds + \alpha_{\rm UV} \sum_kk^p(|a_k|^2+|b_k|^2) + V_{\rm SA}[\gamma].$$


## TOP3 — Irreversible gradient flow

**Status:** ⚠️ `CONDITIONAL`

$$\partial_t\gamma = -\frac{\delta\mathcal E_{\rm loop}}{\delta\gamma}.$$
For a differentiable gradient flow,
$$\frac{d\mathcal E_{\rm loop}}{dt} = - \left\| \frac{\delta\mathcal E_{\rm loop}}{\delta\gamma} \right\|^2 \le0.$$
Well-posedness and treatment of self-avoidance require assumptions.


## TOP4 — Emergent-topology criterion

**Status:** ○ `OPEN`

A proposed physical invariant $I[\gamma]$ satisfies
$$I[\gamma_t]\to I_\infty$$
while
$$\frac{d\mathcal E}{dt}<0$$
along nonsingular descent. The equivalence between this persistence and physical topology remains open.


## TOP5 — WCT dynamical codimension

**Status:** ◻️ `DEFINITION`

$$\operatorname{codim}_{\rm WCT}(\gamma) := \text{minimum number of singular events required to reach the unknot}.$$
This is not manifold codimension.


## TOP6 — Spectral topology bands

**Status:** ○ `OPEN`

Define
$$\epsilon_\kappa = \frac1L\int_\gamma\kappa^2ds.$$
The claim that distinct knot types occupy disjoint asymptotic $\epsilon_\kappa$ bands requires broader numerical and experimental validation.


## TOP7 — Topological mass proxy

**Status:** ⚠️ `CONDITIONAL`

Within a fixed topological and normalization class,
$$m_{\rm WCT} \propto \epsilon_\kappa.$$
The proportionality constant and absolute scale require calibration.


## TOP8 — Holonomy non-invariance

**Status:** ○ `OPEN`

For
$$H_\tau[\gamma] = \int_\gamma\tau\,ds,$$
the proposed non-invariance statement is that smooth admissible deformations can change $H_\tau$ continuously. A general theorem for the declared flow class remains open.


## TOP9 — Protein-particle structural correspondence

**Status:** ○ `OPEN`

$$\text{knotted protein states} \longleftrightarrow \text{stable WCT loop excitations}$$
is a proposed analogy restricted to irreversible curvature flow with self-avoidance and spectral suppression. It is not an established physical equivalence.


# Canonical correction layer


## CORR1 — Full Lyapunov candidate

**Status:** ◻️ `DEFINITION`

$$\mathcal E_{\rm WCT}[\psi] = \int \left( |\nabla\psi|^2 + |\Theta_\varepsilon[\psi]|^2 \right)dx.$$
The curvature term alone is only one component.


## CORR2 — Mean-amplitude spectral closure

**Status:** ⚠️ `CONDITIONAL`

Under a weak-intermittency mean-amplitude approximation,
$$D_{\rm eff}^2 := \langle|\psi|^2\rangle+\varepsilon^2,$$
$$C_\Theta(k) \approx \frac{k^4}{D_{\rm eff}^2}.$$
This is an approximation, not an exact identity.


## CORR3 — Spectral-weight notation

**Status:** ◻️ `DEFINITION`

Use
$$\lambda_\Theta$$
for curvature cost and
$$\lambda_{\rm ex}$$
for auxiliary/exploration weight, avoiding collision with the regularizer parameter $\alpha$.


## CORR4 — Macro-micro control parameter

**Status:** ◻️ `DEFINITION`

$$\Xi = \frac{\int k^4\rho(k)\,dk}{H}, \qquad H=-\sum_kP_k\ln P_k,$$
on the domain $H>0$.


## CORR5 — Entropy-curvature coupling

**Status:** ○ `OPEN`

$$\frac{dH}{dt} \le -\mu \int|\Theta_\varepsilon[\psi]|^2dx.$$
This remains the same open claim as E43 unless derived for a specified evolution.


## CORR6 — Isoelectronic-flow alignment

**Status:** ○ `OPEN`

The imaginary-time isoelectronic flow is proposed as a reduced sector of M7 with ultraviolet smoothing and norm enforcement. A derivation of the reduction and its error bounds remains open.

# Audit index

## ✅ PASS (51)

`M2`, `M3`, `M4`, `M7`, `E1A`, `E1B`, `E2`, `E3`, `E4`, `E6`, `E7`, `E8`, `E10`, `E11`, `E12`, `E16`, `E17`, `E20`, `E21`, `E24`, `E26`, `E28`, `E29`, `E30`, `E33`, `E37`, `E38`, `E45`, `E47`, `E49`, `E51`, `E53`, `E57`, `E59`, `E61`, `E62`, `E64`, `E65`, `E67`, `E69`, `E81`, `CLE2`, `CLE4`, `CLE6`, `CLE7`, `CLE9`, `CLE10`, `G1`, `EX`, `EY`, `EZ`

## ⚠️ CONDITIONAL (32)

`M1`, `M5`, `E5`, `E13`, `E14`, `E15`, `E18`, `E19`, `E22`, `E23`, `E31`, `E32`, `E40`, `E41`, `E48`, `E50`, `E54`, `E56`, `E58`, `E66`, `E68`, `E70`, `E71`, `E72`, `E76`, `E80`, `CLE5`, `CLE8`, `FA`, `TOP3`, `TOP7`, `CORR2`

## ◻️ DEFINITION (23)

`M6A`, `E9`, `E25`, `E27`, `E35`, `E36`, `E39`, `E44`, `E52`, `E55`, `E60`, `E63`, `E73`, `E79`, `E82`, `CLE1`, `CLE3`, `TOP1`, `TOP2`, `TOP5`, `CORR1`, `CORR3`, `CORR4`

## ○ OPEN (36)

`M6B`, `M8`, `E34`, `E42`, `E43`, `E46`, `E74`, `E75`, `E77`, `E78`, `CM1`, `CM2`, `CM3`, `CM4`, `CM5`, `CM6`, `CM7`, `CM8`, `CM9`, `CM10`, `CM11`, `CM12`, `CM13`, `CM14`, `CM15`, `CM16`, `CM17`, `CM18`, `CM19`, `CM20`, `TOP4`, `TOP6`, `TOP8`, `TOP9`, `CORR5`, `CORR6`

# Source alignment

This revision was aligned to the current canonical repository architecture:

- `equations/full_registry.yaml` — 142 IDs and current statuses;
- `MASTER_EQUATIONS.md` — $M1$–$M8$;
- `EQUATIONS.md` — corrected family-level equations;
- `FULL_COVERAGE.md` — contradiction-resolution policy;
- `wct-lean` — kernel-checked subset and compiled correction theorems.

The historical forms that failed remain relevant to chronology, but they are not the current canonical equations. A zero current `FAIL` count means that known contradictory forms were replaced or weakened; it does not convert conditional or open claims into proofs.

---

# End of corrected WCT equation registry
