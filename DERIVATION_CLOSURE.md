# WCT GR and Fundamental-Constants Derivation Closure

**Author:** Richard J. Reyes  
**Document type:** mathematical closure / obstruction audit  
**Status:** exact internal derivations where stated; conditional low-energy completion where assumptions are stated; no claim of empirical validation

## 1. Purpose

This note closes as much of the Wave Confinement Theory (WCT) derivation chain as can be closed from one consistent action. It also records the precise points where the historical formulation cannot support a conclusion without an additional field, coupling, or physical scale.

The purpose is to separate:

1. exact algebraic and variational consequences of WCT definitions;
2. conditional low-energy consequences of a covariant completion;
3. phenomenological identifications requiring measurement;
4. statements impossible for the present field content.

The main results are:

- a phase-covariant node regularization;
- the correct higher-derivative Noether-current formula;
- a corrected loop-locking derivation;
- an exact confinement-eigenvalue derivation of effective mass;
- a covariant scalar-tensor completion with a controlled GR limit;
- a rank obstruction for the historical one-complex-scalar metric ansatz;
- a classification of which constants can be predicted absolutely and which can only be predicted as ratios.

## 2. Phase-covariant curvature operator

Let

\[
q:=|\psi|^2,
\qquad
D_\varepsilon(q):=q+\varepsilon^2e^{-2\alpha q},
\qquad
\varepsilon>0,
\quad \alpha>0.
\]

Define

\[
R_\varepsilon(\psi)
:=\frac{\bar\psi}{D_\varepsilon(|\psi|^2)}
\]

and

\[
\Theta_g[\psi]
:=-(\Box_g\psi)R_\varepsilon(\psi).
\tag{1}
\]

For every complex \(\psi\),

\[
D_\varepsilon(|\psi|^2)
=|\psi|^2+\varepsilon^2e^{-2\alpha|\psi|^2}>0,
\]

so the denominator has no finite-field zero. For \(\psi\neq0\),

\[
\lim_{\varepsilon\to0}R_\varepsilon(\psi)
=\frac{\bar\psi}{|\psi|^2}
=\frac1\psi.
\]

Under the global phase transformation

\[
\psi\mapsto e^{i\beta}\psi,
\]

we have

\[
R_\varepsilon\mapsto e^{-i\beta}R_\varepsilon,
\qquad
\Box_g\psi\mapsto e^{i\beta}\Box_g\psi,
\]

and therefore

\[
\Theta_g[\psi]\mapsto\Theta_g[\psi].
\]

The historical denominator

\[
\psi+\varepsilon e^{-\alpha|\psi|^2}
\]

is not phase covariant because its two summands transform differently. It cannot support the claimed generic global \(U(1)\) symmetry. Equation (1) supersedes it in the symmetry-preserving covariant sector.

## 3. Minimal covariant completion

Let \(g_{\mu\nu}\) be a Lorentzian metric and \(q=|\psi|^2\). Consider

\[
S[g,\psi,\Phi_m]
=
\int d^4x\sqrt{-g}
\left[
\frac12F(q)R
-Z(q)g^{\mu\nu}\nabla_\mu\bar\psi\nabla_\nu\psi
-V(q)
-\frac{\beta_\Theta}{2}|\Theta_g[\psi]|^2
\right]
+S_m[g,\Phi_m].
\tag{2}
\]

Here \(F(q)>0\) is the gravitational stiffness, \(Z(q)>0\) is the two-derivative kinetic coefficient, \(V(q)\) is the confinement-vacuum potential, and \(S_m\) contains other matter fields.

Equation (2) is a covariant low-energy completion. It is not yet a proof that a full metric emerges from one scalar. That distinction is addressed in Section 7.

The finite-\(k\) Swift-Hohenberg rail used in pattern-formation simulations should not be silently inserted into the fundamental Lorentzian action. A purely spatial fourth-order operator requires a preferred foliation or an additional timelike field. It remains an effective nonequilibrium sector unless that covariance structure is supplied.

## 4. Metric field equation

Define

\[
T^{(m)}_{\mu\nu}
:=-\frac{2}{\sqrt{-g}}
\frac{\delta S_m}{\delta g^{\mu\nu}}
\]

and

\[
T^{(\Theta)}_{\mu\nu}
:=-\frac{2}{\sqrt{-g}}
\frac{\delta}{\delta g^{\mu\nu}}
\int d^4x\sqrt{-g}
\left(-\frac{\beta_\Theta}{2}|\Theta_g|^2\right).
\]

Varying (2) with respect to \(g^{\mu\nu}\) gives

\[
F(q)G_{\mu\nu}
=
T^{(m)}_{\mu\nu}
+T^{(0)}_{\mu\nu}
+T^{(\Theta)}_{\mu\nu}
+\nabla_\mu\nabla_\nu F
-g_{\mu\nu}\Box_gF,
\tag{3}
\]

where

\[
T^{(0)}_{\mu\nu}
=
Z(q)
\left(
\nabla_\mu\bar\psi\nabla_\nu\psi
+\nabla_\nu\bar\psi\nabla_\mu\psi
\right)
-g_{\mu\nu}
\left[
Z(q)\nabla_\rho\bar\psi\nabla^\rho\psi+V(q)
\right].
\tag{4}
\]

The expanded expression for \(T^{(\Theta)}_{\mu\nu}\) is lengthy because \(\Theta_g\) contains second derivatives and connection dependence. Its functional-derivative definition is exact and does not discard metric-variation terms.

Diffeomorphism invariance gives the consistency identity

\[
\nabla^\mu
\left[
T^{(m)}_{\mu\nu}
+T^{(0)}_{\mu\nu}
+T^{(\Theta)}_{\mu\nu}
\right]=0
\]

when the \(\psi\) equation of motion holds.

## 5. Higher-derivative phase current

For

\[
\delta\psi=i\epsilon\psi,
\qquad
\delta\bar\psi=-i\epsilon\bar\psi,
\]

a Lagrangian depending on \(\psi\), \(\nabla\psi\), and \(\nabla\nabla\psi\) has generalized current

\[
J^\mu
=
\frac{\partial\mathcal L}{\partial(\nabla_\mu\psi)}\delta\psi
+
\frac{\partial\mathcal L}{\partial(\nabla_\mu\nabla_\nu\psi)}
\nabla_\nu\delta\psi
-
\nabla_\nu
\left(
\frac{\partial\mathcal L}{\partial(\nabla_\mu\nabla_\nu\psi)}
\right)
\delta\psi
+\text{complex conjugate}.
\tag{5}
\]

On shell,

\[
\nabla_\mu J^\mu=0.
\tag{6}
\]

The familiar current

\[
J^\mu_{(2)}
=2Z(q)\,\mathrm{Im}(\bar\psi\nabla^\mu\psi)
\]

is only the leading two-derivative contribution. The curvature term adds the remaining terms in (5). Therefore the elementary current alone is not the full conserved current of the higher-derivative action.

## 6. Controlled GR limit

Let \(q_0\) be a homogeneous locked vacuum satisfying

\[
F_0:=F(q_0)>0,
\qquad
V'(q_0)=0,
\qquad
\nabla_\mu q_0=0.
\]

Assume amplitude and curvature-sector fluctuations have positive gaps \(M_q\) and \(M_\Theta\), and consider energies

\[
E\ll M_q,
\qquad
E\ll M_\Theta.
\]

Then

\[
\nabla_\mu F=0,
\qquad
T^{(\Theta)}_{\mu\nu}
=\mathcal O(E^2/M_\Theta^2),
\]

and

\[
T^{(0)}_{\mu\nu}
=-V(q_0)g_{\mu\nu}
+\mathcal O(E^2/M_q^2).
\]

Equation (3) becomes

\[
G_{\mu\nu}
+\Lambda_{\mathrm{eff}}g_{\mu\nu}
=
8\pi G_{\mathrm{eff}}T^{(m)}_{\mu\nu}
+
\mathcal O
\left(
\frac{E^2}{M_q^2},
\frac{E^2}{M_\Theta^2}
\right),
\tag{7}
\]

with

\[
G_{\mathrm{eff}}
=\frac{1}{8\pi F_0},
\qquad
\Lambda_{\mathrm{eff}}
=\frac{V(q_0)}{F_0}
\tag{8}
\]

in natural units.

### GR-limit theorem

Under these assumptions, the covariant WCT completion (2) reproduces Einstein gravity with a cosmological constant below the amplitude and curvature-feedback gaps, with calculable higher-derivative corrections.

Linearizing

\[
g_{\mu\nu}=\eta_{\mu\nu}+h_{\mu\nu}
\]

for a static weak source yields

\[
\nabla^2\Phi=4\pi G_{\mathrm{eff}}\rho
\]

up to the same suppressed corrections. The Newtonian limit therefore follows from the field equation rather than from identifying a scalar curvature diagnostic directly with the Newtonian potential.

This is a complete low-energy recovery statement for the covariant completion. It does not prove that the metric tensor itself is generated by the historical one-field metric ansatz.

## 7. Rank obstruction to the historical metric ansatz

A frequently used WCT metric ansatz has the schematic form

\[
g^{\mathrm{eff}}_{\mu\nu}
=\eta_{\mu\nu}
+B(q)\,\mathrm{Re}
\left(
\nabla_\mu\bar\psi\nabla_\nu\psi
\right)
+C(q,\Theta)\eta_{\mu\nu}.
\tag{9}
\]

Write \(\psi=a+ib\). Its anisotropic deformation is

\[
h^{\mathrm{aniso}}_{\mu\nu}
=B
\left(
\nabla_\mu a\nabla_\nu a
+\nabla_\mu b\nabla_\nu b
\right).
\]

This is a sum of two rank-one matrices, so

\[
\mathrm{rank}(h^{\mathrm{aniso}})\le2.
\tag{10}
\]

A generic four-dimensional metric perturbation has no such restriction. The independent tensor degrees of freedom of GR cannot be represented by (9) for arbitrary sources and wave configurations.

Therefore the generic implication

\[
\text{one complex scalar effective metric}
\Longrightarrow
\text{full general relativity}
\]

is false.

A complete theory requires at least one of:

1. an independent metric or tetrad collective field;
2. enough additional order-parameter fields to span a generic symmetric tensor sector;
3. a microscopic mechanism whose low-energy collective variable is a full metric or tetrad rather than (9).

The completion in Section 3 takes the first route. A stronger emergence claim requires the third route and remains open.

## 8. Exact confinement derivation of effective mass

Let a massless parent field live on a separated domain \(M_4\times K\) and satisfy

\[
\Box_{4+d}\Psi=0.
\]

Let

\[
\Psi(x,y)=\phi_n(x)\chi_n(y)
\]

with confined eigenmode

\[
-\Delta_K\chi_n=\lambda_n\chi_n,
\qquad
\lambda_n\ge0.
\tag{11}
\]

Separation gives

\[
(\Box_4-\lambda_n)\phi_n=0.
\tag{12}
\]

Comparison with the Klein-Gordon equation yields the exact effective mass

\[
m_n=\frac{\hbar}{c}\sqrt{\lambda_n}.
\tag{13}
\]

For a circular loop of length \(L=2\pi R\),

\[
\lambda_n
=\left(\frac{2\pi n}{L}\right)^2
=\frac{n^2}{R^2},
\]

so

\[
m_n
=\frac{2\pi\hbar|n|}{cL}
=\frac{\hbar|n|}{cR}.
\tag{14}
\]

For the single-winding circular mode, \(\kappa=1/R\), and

\[
m_1=\frac{\hbar}{c}\kappa.
\tag{15}
\]

Equation (15) is an exact constant-curvature special case of the WCT mass-curvature law. For general loops, the exact quantity is \(\sqrt{\lambda_n}\), determined by the full confined eigenproblem. Replacing it with

\[
\left\langle\sqrt{\kappa^2+\tau^2}\right\rangle_w
\]

requires an additional geometric approximation or theorem.

## 9. Corrected loop-locking derivation

Consider

\[
S_{\mathrm{lock}}[\varphi]
=
\oint_\Gamma
w(s)(\varphi'(s)-\sigma(s))^2ds
+\Lambda
\left(
\oint_\Gamma\varphi'(s)ds-2\pi n
\right).
\tag{16}
\]

Variation with respect to \(\varphi\) gives

\[
\frac{d}{ds}
\left[w(s)(\varphi'-\sigma)\right]=0,
\]

hence

\[
\varphi'(s)=\sigma(s)+\frac{C}{w(s)}.
\tag{17}
\]

The winding constraint fixes

\[
C
=
\frac{2\pi n-\oint_\Gamma\sigma ds}
{\oint_\Gamma ds/w(s)}.
\tag{18}
\]

The minimum mismatch action is

\[
S_{\min}
=
\frac{
(2\pi n-\oint_\Gamma\sigma ds)^2
}
{\oint_\Gamma ds/w(s)}.
\tag{19}
\]

Exact pointwise phase-curvature locking occurs if and only if

\[
\oint_\Gamma\sigma ds=2\pi n,
\tag{20}
\]

for which \(C=0\) and \(\varphi'=\sigma\).

The commonly written chain

\[
\frac{2\pi n}{L}
=\frac1L\oint\sigma ds
=\langle\sigma\rangle_w
\]

is not generally valid. The first equality requires (20), and the second requires constant \(w\) or another condition equating weighted and unweighted averages.

Thus the loop functional supplies an exact mismatch penalty and an exact-lock criterion. It does not by itself prove the density-weighted mass law for arbitrary weights and geometry.

## 10. Fundamental constants

### 10.1 Dimensionless predictions

A nondimensionalized WCT model can predict:

- eigenvalue ratios \(\lambda_n/\lambda_m\);
- mass ratios
  \[
  \frac{m_n}{m_m}
  =\sqrt{\frac{\lambda_n}{\lambda_m}};
  \]
- dimensionless coupling ratios;
- topological integers and selection rules;
- dimensionless deviations from a baseline theory.

These become genuine predictions only when mode assignments and parameters are frozen before comparison with held-out data.

### 10.2 Speed \(c\)

A Lorentzian action can select a universal characteristic speed. Once units are chosen, it is denoted \(c\). The numerical SI value \(299\,792\,458\ \mathrm{m/s}\) is tied to unit definitions and is not an independent dimensionless prediction.

WCT may derive or postulate a universal causal speed, but cannot generate its unit-dependent decimal value from a dimensionless PDE.

### 10.3 Planck's constant \(\hbar\)

The classical action does not derive the absolute quantum of action. \(\hbar\) enters through quantization, for example

\[
Z=\int\mathcal D\psi\,e^{iS/\hbar},
\]

or canonical commutators. A derivation of \(\hbar\) requires a microscopic quantization mechanism and an absolute action scale. Until then, equation (13) uses \(\hbar\) as the frequency-to-energy conversion.

### 10.4 Newton's constant \(G\)

From (8),

\[
G_{\mathrm{eff}}
=\frac{1}{8\pi F(q_0)}.
\]

If

\[
F(q)=\xi q,
\]

then

\[
G_{\mathrm{eff}}
=\frac{1}{8\pi\xi q_0}.
\tag{21}
\]

This converts the gravitational coupling into a vacuum-amplitude relation. It is a numerical prediction only if WCT independently fixes both \(\xi\) and the dimensionful vacuum scale \(q_0\) without calibration to measured \(G\).

### 10.5 Cosmological constant \(\Lambda\)

The low-energy relation is

\[
\Lambda_{\mathrm{eff}}
=\frac{V(q_0)}{F(q_0)}.
\tag{22}
\]

This is exact within the covariant completion. It is not a numerical prediction if \(V(q_0)\), \(q_0\), or the unit scale is chosen from observed vacuum energy.

### 10.6 Fine-structure constant \(\alpha_{\mathrm{EM}}\)

Global phase symmetry quantizes winding but does not fix the electromagnetic coupling magnitude. To derive

\[
\alpha_{\mathrm{EM}}
=\frac{e^2}{4\pi\hbar c},
\]

WCT must supply a local \(U(1)\) gauge sector, its kinetic normalization, and the normalized coupling of a confined mode to the gauge field. Without these ingredients, a numerical value is a fit or phenomenological identification.

## 11. Closure status

| Target | Result |
|---|---|
| Node-safe operator | Completed algebraically |
| Global \(U(1)\) invariance | Completed for corrected operator |
| Higher-derivative Noether current | General exact formula supplied |
| Loop-locking variation | Completed; equality conditions corrected |
| Effective mass from confinement | Completed as eigenvalue theorem |
| Curvature-average mass law | Exact for constant-curvature locked cases; conditional generally |
| GR low-energy limit | Completed for covariant scalar-tensor completion |
| Full GR from historical one-scalar metric | Ruled out generically by rank obstruction |
| Numerical \(G\) | Conditional on independently fixed scale and coupling |
| Numerical \(\Lambda\) | Conditional on independently predicted vacuum energy |
| Numerical \(\hbar\) | Not derived by the current classical theory |
| Numerical \(c\) | Universal speed can be selected; SI decimal is unit-defined |
| Numerical \(\alpha_{\mathrm{EM}}\) | Requires a normalized local gauge sector |

## 12. Remaining work

1. **Tensor emergence:** derive a full metric or tetrad collective mode from WCT microdynamics.
2. **Spectrum theorem:** bound confined eigenvalues in terms of curvature and torsion for admissible WCT geometries.
3. **Vacuum-scale selection:** obtain \(q_0\) and the confinement length without observed-constant calibration.
4. **Quantization:** define the quantum measure or canonical structure fixing the action scale.
5. **Gauge normalization:** construct the local \(U(1)\) sector and derive its coupling normalization.
6. **Stability:** prove the physical sector avoids unacceptable higher-derivative instabilities, or construct a degenerate or auxiliary-field completion.
7. **Prospective prediction:** freeze parameters and predict a held-out dimensionless observable before measurement.

## 13. Conclusion

WCT supports a mathematically controlled effective-mass derivation and a controlled GR low-energy limit after correcting the phase regularization and supplying a genuine tensor gravitational sector. A single complex scalar alone does not generate arbitrary spacetime geometry.

The strongest defensible conclusion is:

> Confined WCT eigenmodes generate effective masses through their spectral eigenvalues, and a covariant WCT scalar-tensor completion reduces to Einstein gravity at low energy. Absolute dimensionful constants require an independently selected physical scale, while full emergent gravity requires a tensor collective mode beyond the historical one-scalar metric ansatz.
