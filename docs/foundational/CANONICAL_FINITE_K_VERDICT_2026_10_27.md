# #18 — Canonical finite-wavelength verdict for corrected S004

**Decision:** `PARAMETER_DEPENDENT_DISPERSION_MINIMUM_ONLY; PHYSICAL_SELECTOR_OBSTRUCTED_IN_LITERAL_STRICT_S004`.

The unchanged real S004 action is **not** a dynamically healthy first-principles finite-band pattern selector in the strict parameter region described here. This is a **scoped** obstruction, not a theorem that no healthy WCT-inspired action can exist. The two distinct surviving claims are (i) a conditional amplitude oscillation-frequency minimum in the untruncated parent and (ii) a *separately assumed* spatial-only effective selector.

## Inputs, conventions, exact second variation

Source: `WCT_CORRECTED_VARIATIONAL_CLOSURE_DRAFT.md` §§1–6; `docs/cross_paper/QUADRATIC_ACTION_AND_DISPERSION.md`, equations Q1–Q7; and `docs/foundational/GHOST_AND_CONSTRAINT_ANALYSIS.md`, H1–H4.

On a fixed flat Lorentzian foliation in units `tau=c0*t`, use signature `(-,+,+,+)`, `Box=-partial_tau^2+Delta`, `u=|psi|^2`, `D=u+eps^2 exp(-2 alpha u)`, `F=u/D^2`,
`Q=Box psi`, `P=Delta psi` and
`H=kappa |Q|^2+theta |P|^2+gamma Re(Q conjugate(P))`.
The original Lagrangian is `L_orig=partial conjugate(psi) partial psi - V(u)+F(u) H`; use `L_phys=-L_orig` to orient the ordinary kinetic term positively without changing the poles.

For an arbitrary on-shell smooth background `psi0`, put `psi=psi0+eta`, `u1=2 Re(conj(psi0)eta)`, `u2=|eta|^2`, `H0=H(psi0)`, `H2=H(eta)`, and

```text
H1=2 Re[ kappa conj(Q0) Box eta + theta conj(P0) Delta eta
         + gamma/2*(conj(Q0) Delta eta + conj(P0) Box eta) ].
L2_orig=partial conj(eta) partial eta - V' u2 - (V''/2)u1^2
        +F H2+F' u1 H1+(F' u2+(F''/2)u1^2)H0.             (Q1)
```

These formulas include both real components and all time derivatives. For inhomogeneous `psi0`, Fourier modes couple; no diagonal `M(omega,k)` is asserted.

For a homogeneous carrier `psi0=A0 exp(-i Omega tau+i Kvec.x)`, the **full**, not scalar-truncated, Fourier Hessian is Q4 (reproduced exactly in the cited source). Define `d=Omega^2-|Kvec|^2`, `p0=-|Kvec|^2`, `ell=Omega omega-Kvec.k`, `z=omega^2`, `q=|k|^2`, `E=diag(1,0)`, and
```text
Qmat=[[z-q+d, 2 i ell],[-2 i ell,z-q+d]]
Pmat=[[-q-|Kvec|^2,-2 i Kvec.k],[2 i Kvec.k,-q-|Kvec|^2]]
J=(kappa*d+gamma*p0/2) Qmat+(theta*p0+gamma*d/2) Pmat
Kmat=-Qmat-V' I-2u V'' E
    +F[kappa Qmat^2+theta Pmat^2+gamma/2*(Qmat Pmat+Pmat Qmat)]
    +2u F'*(E J+J E)+u*h0*F' I+2u^2*h0*F'' E
h0=kappa*d^2+theta*p0^2+gamma*d*p0
V'=-d+h0*(F+u F') (background stationarity).
M_orig=2 Kmat; M_phys=-2 Kmat; det(M)=4 det(Kmat).
```
For `Kvec=0` and `Omega != 0`, Q5 is an exact 2x2 amplitude–phase matrix with determinant `4*(A*B-Omega^2*z*C^2)`, quartic in `z`. Its high-time Hessian is nondegenerate whenever `F*kappa>0`. Neither a branch's reality nor a stable finite band follows universally from curvature positivity.

## Static homogeneous classification and *new explicit no-onset consequence*

Now take `Omega=Kvec=0`, `u=A0^2>0`, `V'(u)=0`, and
`K=F*kappa>0`, `G=F*gamma`, `T=F*theta>0`, `B=4*K*T-G^2>0`.
The amplitude block has `m^2=-2u V''(u)`; the phase block has `m^2=0`. Both satisfy the **exact**, not truncated, Q6/Q7 system

```text
D(z,q)=(z-q)-m^2-K*(z-q)^2+G*q*(z-q)-T*q^2
d(q)=1-4*K*m^2+2*G*q-B*q^2
z_plus/minus(q)=q+(1+G*q +/- sqrt(d(q)))/(2*K).
```

On `d(q)>0`, the two poles have opposite residues (`partial_z D(z_minus)=sqrt(d)`, `partial_z D(z_plus)=-sqrt(d)`); at `d=0`, they collide. With `B>0`, `d(q)<0` at sufficiently large `q`, giving complex time-frequency poles and H4's high-frequency Sobolev obstruction. This fails foundational health even where the low pole is positive.

**Static no-onset theorem.** Under `K>0`, `T>0`, `B>0`, `m^2>=0`, `q>0`:
```text
K+G+T = ((2*K+G)^2+B)/(4*K) > 0,
D(0,q)=-m^2-q-(K+G+T)*q^2<0.
```
Therefore no nonzero finite-`q` *zero-frequency static bifurcation* exists in this sector. In the real-pole window with `G>=0`, `m^2>=0`, both `z_minus,z_plus>0` for `q>0`; there is no *tachyonic* finite band there. For a general negative `G`, the connected real-pole interval from `q=0` likewise has no zero of `D(0,q)` and cannot cross to a negative pole without a collision. This topological-continuity observation is an analytical argument; the Lean module directly certifies the stated `G>=0` subset, rather than claiming a formalized global continuation theorem.

**All static branch features:** for the *amplitude* mode, if `0<4*K*m^2<1`, put `s=sqrt(1-4*K*m^2)`, `a0=2*K+G`. Its lower branch on `d>0` satisfies
```text
z_minus'(q)=[a0-(G-B*q)/sqrt(d)]/(2*K)
z_minus''(q)=(B*s^2+G^2)/(2*K*d^(3/2))>0.
```
A unique positive interior minimum occurs **iff** `G*C0>s`, where `C0=(1-s)/(2*K)`. Its exact location and collision boundary are F2 in `docs/cross_paper/FINITE_SCALE_SELECTION.md`. This is a positive-frequency minimum, not an instability, a localized eigenmode or a physical length prediction. If `G*C0<=s`, strict convexity forbids an interior low-branch minimum. On the *phase* branch `m^2=0`, `z_minus'(0)=1`, and strict convexity on `d>0` rules out an interior positive-`q` minimum. For the upper pole `z_plus'` and `z_plus''` may have extrema, but these belong to the negative-residue branch and **cannot** furnish healthy selection. At `d=0`, derivative formulas are singular; treat the collision as a separate boundary. At `d<0`, roots are complex, not physical real oscillatory extrema.



**Upper branch extremum (not a healthy selector).** On the same connected real-pole interval from `q=0`, the upper pole is *strictly concave*,
```text
z_plus'(q)=[a0+(G-B*q)/sqrt(d)]/(2*K)
z_plus''(q)=-(B*s^2+G^2)/(2*K*d^(3/2)) < 0.
```
Because `z_plus'` tends to negative infinity at the collision, it has a unique interior **maximum** iff `a0+G/s>0`; otherwise there is no positive-`q` upper-branch stationary point. The upper branch has a negative residue in either case. For the lower branch the counterpart is a unique **minimum** iff `a0-G/s<0`, which is equivalent to F1's `G*C0>s`. No other stationary points exist on these connected real-pole intervals because the second derivatives have strict signs.

**Disconnected real-pole islands.** The above derivative-at-zero classification assumes `d(0)=s²>0`. If `d0=1-4Km²<0`, a bounded positive-`q` real-pole island can occur only if `G>0` and `G²+B*d0>0`, in which case the real intervals are bounded by the two positive collisions `q_±=(G±sqrt(G²+B*d0))/B`. There is already a complex-frequency instability at `q=0` in this case, and the model is not a healthy background. The double-root cases `d0=0` or `G²+B*d0=0` require their own Jordan/collision analysis and are not called stable finite bands.

**Zero background** `F(0)=0`: there is no linear fourth-spatial-order term. **Replacing** the weight with a positive `W(0)` adds it back but also restores an invertible higher-time Hessian `2*kappa*W(0)*I2` if `kappa>0`; it does not cure the ghost.

## Other models and physical selection

- An independent spatial-only, second-order-in-time action `L=|psi_tau|^2 - mu|psi|^2 + a|grad psi|^2-b|Delta psi|^2 - lambda |psi|^4/2` can have a finite spectral center `q_star=a/(2b)` if `a,b>0`. Its positive-energy gate is `4*b*mu>a^2`. The sign of the quadratic gradient term has changed; **this is a new declared model, not a reduction of S004**. See the companion #16 note.
- The older spatial-only S004 boundary with `kappa=gamma=0`, `theta>0` has `z=q+Tq^2+m^2`, with no positive-`q` minimum for `m^2>=0`.
- Swift–Hohenberg/PFF **dissipative growth** needs separate damping or a bath/coarse-graining theorem; it is neither the S004 Q6 pole dynamics nor the Hamiltonian rail.
- In historical `MASTER_EQUATIONS.md` Master Equation 3 uses a **plus** sign in `+b*(Delta+k_star^2)^2` while describing short-wave suppression. For `b>0` that sign produces unbounded high-`k` growth; suppression requires a **minus** sign, independently of all S004 conclusions.

**Canonical classification:** `S004_LINEAR_STATIC_FINITE_BAND_NOT_ESTABLISHED`; `S004_AMPLITUDE_FREQUENCY_MINIMUM_DERIVED_CONDITIONAL`; `S004_STRICT_FULL_SYSTEM_UNHEALTHY`; `SPATIAL_COMPETING_GRADIENT_RAIL_INDEPENDENT_MODEL`; `S004_TO_PFF_NONLINEAR_REDUCTION_OPEN`. No physical wavelength or unique background follows from the unchanged action without independently fixing its parameters, preparation/constraints, and UV completion.

## Verification and proof boundary

`lean/ReleaseProofs/R18_CanonicalObstruction.lean`: static coefficient positivity, exact zero-frequency inverse-propagator obstruction, phase specialization, no tachyonic real poles in the `G>=0` window, and nondegenerate acceleration Hessian for modified positive weights. Builds atop the verified Q6/Q7 identities of PR #42.

`derivations/canonical_16_18_verify.py`: symbolic reconstruction of static pole and the **full rotating carrier Q4/Q5** matrix, exact Hessian/constraint determinants, and competing-gradient alternative, without treating numerical roots as theorems.

**Outstanding gates**: justified background and `V`, physical length, cutoff, nonlinear constrained/ghost-free parent, full global inhomogeneous background analysis, and nonlinear physical-sector reduction. Preserve #18 OPEN even if the mathematical scoped obstruction passes verification.


## Lean completion supplement: exact pole trichotomy and tangent theorem

The new `lean/ReleaseProofs/R18_Classification.lean` derives an exact completed-square invariant for **all** real `q`, without limiting the sign of `G` or `m²`. Put `B=4KT-G²>0`, `d0=1-4Km²` and `N=G²+B*d0`:

```text
B*d(q) = N - (B*q-G)^2.
d(q)>0  iff (B*q-G)^2<N   (distinct real z-poles).
d(q)=0  iff (B*q-G)^2=N   (collision).
d(q)<0  iff (B*q-G)^2>N   (complex z-poles).
```

Thus `N<0` gives no real pole at any spatial wave number; `N=0` permits only the degenerate vertex `q=G/B`; `N>0` admits the single complete real-pole interval `(G-sqrt(N))/B < q < (G+sqrt(N))/B`, intersected with the physical `q>=0`. The right endpoint `(G+sqrt(N))/B` yields an exact UV-complex criterion for every larger `q`. A lower-frequency minimum found inside a real interval cannot continue through a collision as a real stable branch.

The module also gives a **derivative-free finite-q minimum certificate**: if `qStar` is real with `d(qStar)>=0`, `A=2K+G` and `G-B*qStar=A*sqrt(d(qStar))`, then for any real-pole `q` satisfying `sqrt(d(qStar))+A*(q-qStar)>0`,
```text
[sqrt(d(qStar))+A*(q-qStar)]^2-sqrt(d(q))^2
    =(A^2+B)*(q-qStar)^2.
```
Consequently `z_minus(qStar)<=z_minus(q)`, with equality only for `q=qStar`. This explicitly retains the positive-tangent domain hypothesis; neither the lemma nor its proof silently assumes an unrestricted action-level EFT reduction. The analytical F1/F2 source supplies the explicit parameter-dependent `qStar` and its collision-domain placement. The completed-square Lean proof is a stronger direct uniqueness check in the declared tangent-positive real-pole window, not a new experimental scale prediction.

The independent competing-gradient alternative is formalized only under a **different namespace and action** in `R16_CompletionClasses.lean`. No Lean axiom asserting equivalence to S004 has been added. Full covariant/rotating-background stability, dynamic foliation constraints, cutoff preparation and nonlinear S004-to-PFF derivation remain explicit open barriers.

**Verification:** both `.github/workflows/release-lean.yml` and the independently retrying pinned `.github/workflows/geometry-scoped-lean.yml` run the entire `lean/ReleaseProofs.lean` library; the exact-symmetry and symbolic checks run in `.github/workflows/canonical-16-18.yml`. Do not describe new declarations as kernel-checked until the check for the *latest commit* succeeds.
