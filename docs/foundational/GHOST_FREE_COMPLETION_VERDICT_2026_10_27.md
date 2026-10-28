# #16 — Scoped obstruction and independently healthy competing-gradient alternative

**Status:** `S004_REGULAR_NONDEGENERATE_GHOST_UNAVOIDABLE` within the defined class; `SCALAR_TENSOR_A1_GHOST_REMAINS`; `SPATIAL_RAIL_LINEAR_HEALTH_AND_ENERGY_PROVED_AS_NEW_MODEL`; `WCT_FOUNDATIONAL_COMPLETION_OPEN`.

**Never identify the new competing-gradient model with S004.** This document supplements, rather than revises, the frozen action, PR #33's H1–H4 analysis, PR #34's A1/A2 analysis, and PR #42's earlier Lean witnesses.

## 1. Parent and exact canonical structure

The corrected strict S004 action on fixed flat metric and inertial foliation is
```text
L_orig = partial_mu(conj psi) partial^mu psi - V(|psi|^2)
       + F(|psi|^2) [kappa |Box psi|^2 + theta |Delta psi|^2
                     + gamma Re(Box psi conjugate(Delta psi))]
F(u)=u/(u+eps^2 exp(-2 alpha u))^2.
```

Use `L_phys=-L_orig` for the ordinary positive time-kinetic orientation (no pole or energy-sign relative change). For `u>0`, `eps>0`, `kappa>0`, `F(u)>0`. Because `Box psi=-ddot psi+Delta psi`, the real Cartesian acceleration Hessian is proportional to `-2F*kappa*I2`, determinant `4(F*kappa)^2>0`. Mixed `gamma` and spatial `theta` terms are **linear** or independent of accelerations and cannot cancel this Hessian. This holds locally for rotating, static and inhomogeneous regular backgrounds; the precise linear dispersion depends on the background.

The direct Ostrogradsky phase space contains `(x1,x2,v1,v2,px1,px2,pv1,pv2)`. The equivalent first-order auxiliary construction introduces `lambda_i(dot x_i-v_i)` and canonical `p_x-lambda=0`, `p_lambda=0`. Their 4x4 Poisson bracket matrix is
```text
[[0,-I2],[I2,0]]
```
with determinant 1: all four auxiliary primary constraints are **second class**. Preservation fixes multipliers; the regular acceleration Legendre map adds no primary scalar constraint. Hence the two real Cartesian fields together have **four configuration degrees of freedom**, rather than two, per space point. This is H2 in the existing foundational analysis; new symbolic checks independently reproduce the bracket determinant.

The reduced Ostrogradsky Hamiltonian contains `px·v` as an affine, unbounded canonical direction for any `v!=0`. In a single-real-sector mechanical normalization it is `p1*q2-p2^2/(2K)-q2^2/2+m^2*q1^2/2`, and is unbounded above and below (the earlier PR #42 Lean theorem). A mere auxiliary-field rewrite cannot make the solution space smaller without extra independently justified constraints.

### New limited failure-class theorem

Let `C=F(u)*kappa != 0`, and let `P` be the Jacobian of any *regular invertible point transformation* in the two real-field coordinates. The acceleration Hessian transforms by congruence, with
```text
det(H_transformed) = 4*C^2*(det(P))^2 > 0.
```
The new Lean proof checks the exact Gram-determinant identity for a generic 2x2 Jacobian, and positivity for `C!=0,det(P)!=0`. Consequently no **invertible derivative-free field reparametrization** can produce the primary acceleration degeneracy absent in S004. This theorem is *not* a no-go theorem for singular redefinitions, derivative-dependent transformations, new gauge constraints or specifically engineered DHOST couplings.

### Already-established high-frequency gate

At a nonzero static on-shell homogeneous background set `K=F*kappa>0`, `G=F*gamma`, `T=F*theta>0`, `B=4KT-G^2>0` and `m^2=-2uV''` (amplitude) or zero (phase). The exact inverse-propagator poles Q6–Q7 have discriminant `1-4Km^2+2Gq-Bq^2`. For `q->infinity`, the strict `B>0` forces complex poles and growth `exp(c_uv|k|tau)`, `c_uv>0`, as rigorously argued in existing H4. This excludes fixed-Sobolev continuous linear evolution at any positive time on the specified homogeneous background. It is not a nonlinear blowup theorem about every background.

For real nondegenerate poles, `partial_z D(z_minus)=+sqrt(d)`, `partial_z D(z_plus)=-sqrt(d)`. The high pole is an opposite-residue ghost even below the collision; deleting it by a cutoff is an *EFT preparation assumption*, not a constraint. Full nonzero rotating-carrier Q5 remains quartic in `z`, with nonzero high-time Hessian, and is not replaced by the simpler static block.

## 2. Completion-candidate decision table

| Route | Exact mathematical result | Gate |
| --- | --- | --- |
| Keep strict corrected S004 | Nonzero 2x2 acceleration Hessian, four real configuration DOF, affine unbounded Hamiltonian, H4 UV obstruction | **FAIL** |
| Vary only gamma and theta while kappa F>0 | Acceleration Hessian independent of them | **FAIL** |
| Replace F(u) by positive W(u), including W(0)>0 | Hessian remains 2*kappa*W*I2 where W>0 | **FAIL** as a ghost cure |
| Invertible point change of complex field | Nonzero Hessian determinant multiplied by det(P)^2 | **FAIL** as a degeneracy cure |
| Introduce exact auxiliary variables | Four second-class constraints remove only artificial variables, no physical ghost | **FAIL** |
| Previously proposed Einstein-frame scalar–tensor completion | Full PR #34 A1 kinetic inertia (2 positive, 2 negative), 2 tensor + 4 scalar configuration DOF | **FAIL** for that published candidate |
| kappa=gamma=0, theta>0 (spatial S004 boundary) | Two DOF and nonnegative H for U>=0, monotone q+Tq²+m²; node-dependent H² noncoercivity | **CONDITIONAL**, no finite selector |
| Restrict static S004 to positive-residue low pole | Valid only inside declared quadratic spectral window; nonlinear excluded-mode control missing | **NOT A COMPLETION** |
| Add specific DHOST, dynamical foliation or new gauge sector | No complete new source action and constraints supplied | **OPEN**, no universal no-go |
| Independent competing-gradient spatial action below | Positive mode frequencies, Hamiltonian coercivity and finite q-star, with no fourth-time poles | **HEALTHY MODEL CLASS**, **NOT S004** |

## 3. Explicit alternative — declared new action

As an **independent mathematical/physical modeling proposal**, motivated by competition between long-wavelength aggregation (negative gradient energy) and short-wavelength bending stiffness, use
```text
S_alt = integral d tau d^d x {
   |partial_tau psi|^2 - mu |psi|^2
   +a |grad psi|^2 -b |Delta psi|^2
   -(lambda/2)|psi|^4
}
```
with real parameters `b>0`, `a>0`, `lambda>=0`, and **strict gap** `4*b*mu>a^2`. Unlike S004, this alters the quadratic spatial gradient sign and supplies independent `mu,a,b,lambda`. It is U(1)-invariant, real, local, and second-order-in-time. The principal equation, ignoring harmless overall convention factors from varying a complex field, is exactly
```text
partial_tau^2 psi+mu psi+a Delta psi+b Delta^2 psi
     +lambda |psi|^2 psi=0.
```
Its two real Cartesian fields have velocity Hessian `2 I2`, two configuration degrees of freedom, and **no extra Ostrogradsky poles**. Linearizing around `psi=0`:
```text
omega^2(q) = mu-a*q+b*q^2
           = b*(q-a/(2b))^2 + mu-a^2/(4b).
q_star=a/(2b)>0, k_star=sqrt(q_star), ell_star=2*pi/k_star.
```
The strict gap yields **positive real oscillatory frequencies for all q>=0**; no exponential growth band is claimed. A stationary finite-frequency *spectral preference* is selected by model inputs `a,b`, not a universal WCT prediction.

The conserved Hamiltonian, for smooth decaying or periodic fields and appropriate variations, is
```text
H_alt = integral (|psi_tau|^2
   +mu |psi|^2 -a |grad psi|^2 +b |Delta psi|^2
   +(lambda/2)|psi|^4) dx.
```
Plancherel gives quadratic energy `int [mu-aq+bq²] |hat psi(k)|² dk`. Since `mu-aq+bq²>=delta:=mu-a²/(4b)>0` and `b>0`, its symbol is bounded below by `c*(1+q²)` for some `c>0` (continuity on compact q and asymptotic bq²). Hence `H_alt>=c||psi||_{H²}²+||psi_tau||_2²`, with the nonnegative quartic contribution. This is a **nonlinear coercive energy** for this new model; conservation follows from time translation of the action on regular solutions. Standard semilinear beam-energy methods give local evolution for energy data in dimensions `d<=3` (H² embeds into L-infinity) and conservation prevents finite-time blowup of this energy norm, subject to the usual rigorous solution construction and domain/regularity hypotheses. That PDE theory is an analytical application, not a Lean-kernel theorem proved here.

**Localized equilibrium obstruction in this candidate.** For a time-independent finite-energy solution `phi` of `(mu+a Delta+b Delta²)phi+lambda|phi|²phi=0`, multiply by `conj phi`, take real part and integrate. This yields
```text
int (mu-aq+bq²)|hat phi|² dk+lambda int |phi|⁴ dx=0.
```
Every term is nonnegative and the first is strictly positive unless `phi=0`. Therefore **no nontrivial stationary localized equilibrium** exists in this strictly positive defocusing candidate. The zero vacuum is Lyapunov stable in the finite-energy topology by the conserved coercive Hamiltonian, but it does not fulfill the issue's nontrivial localized-state gate. Time-harmonic Q-ball-like states at other frequencies, boundary-confined modes, or a different saturating/focusing nonlinearity require new work and are **not ruled out** by the static theorem.

The alternative can be viewed as a candidate physical pattern medium (competing spatial interactions), but there is **no established first-principles map** from the corrected S004 curvature quotient, preferred foliation, conserved charge, or a microscopic resonant cavity to the new coefficients. It is not an accepted WCT foundational completion.

## 4. Verified and missing proof layers

- **Lean:** `ReleaseProofs.R16Completion` proves the rank-preservation identity for a generic 2x2 point Jacobian, trivial acceleration kernel, competing-gradient square completion/unique finite-q minimizer/positive squared frequency, nonnegative per-mode energy and an explicit no-nonzero-static-single-mode witness. Earlier R16/R18 results from PR #42 are imported; they are not renamed as new proofs.
- **Exact symbolic:** `derivations/canonical_16_18_verify.py` reconstructs H2's primary Poisson determinant, the original acceleration Hessian, transformed Gram determinant, exact Q4/Q5 rotating matrix, Q6/Q7 static matrix and the alternative's completed square. These identities are not formal proofs of full infinite-dimensional evolution.
- **Analytical PDE:** Plancherel energy coercivity and nonexistence of a nonzero time-independent H² stationary field for the independent defocusing action. No model transfer or experimentally measured mode is claimed.
- **Remaining:** explicit physically motivated degeneracy constraints that preserve curvature interpretation, nonlinear ghost-free/UV-complete WCT parent, independently fixed physical length, and nontrivial localized spectrally/orbitally stable solution in a completion linked to S004.

Do not close issue #16. This package satisfies the **scoped failure-class** mission but does not meet the issue's original positive health-and-localization acceptance gates.
