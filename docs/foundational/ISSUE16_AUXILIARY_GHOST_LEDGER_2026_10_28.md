# Geometry #16 — Exact auxiliary equivalence, degenerate boundaries, and ghost/stability ledger

Date: 2026-10-28. Repository: `rickyjreyes/geometry_of_resonance`. PR #44, issue #16. Dependencies #41, #42 and #43 are merged into `main`.

## 1. Source action versus finite-dimensional jet representative

**Original corrected S004:** fixed flat metric and preferred inertial foliation, `u=|psi|²`,
```text
L_orig = ∂_μ psi* ∂^μ psi - V(u)
       + F(u)[κ |Box psi|²+θ |Delta psi|²
       + γ Re(Box psi (Delta psi)*)]
F(u)=u/(u+eps² exp(-2 alpha u))².
```
The physical ordinary-time-kinetic orientation is `L_phys=-L_orig`. For a regular nonzero Cartesian real-field background `u>0,eps≠0,κ>0`, `F(u)>0` and the real two-field **acceleration Hessian** is `Hacc=-2Fκ I₂` (previous R16Energy), with determinant `4(Fκ)²>0`. The exact algebraic degeneracy condition **in this original action** is `Fκ=0`. Neither γ nor θ enters the acceleration Hessian; changes to them do not remove this regular high-time mode. Setting κ=0 or reaching F=0 is outside the strict sector, and Hessian degeneracy alone does not certify a healthy field theory.

**New finite-dimensional representative to analyze the auxiliary rewrite:** freeze a single real component and coefficients of the acceleration principal part. Define
```text
L4(q,qdot,qddot) = -(h/2) qddot²+(k/2) qdot²-U(q),  h=2Fκ>0,
Laux(q,Q,λ) = -(h/2) Qdot²+(k/2)Q²-U(q)+λ(qdot-Q).
```
The `R16_AuxiliaryEquivalence.lean` theorem `auxiliary_restriction_exact` proves `Laux|Q=qdot,Qdot=qddot=L4` exactly. It does not replace the complete S004 field action with this simplified representative; the full S004 contains spatial dependence, field-dependent `F`, nontrivial V, the γ-mixing and θ-spatial terms.

For smooth trajectories, varying λ yields `qdot-Q=0`. Varying Q gives `kQ-λ+h Qddot=0`; varying q gives `-U'(q)-λdot=0`. Differentiation and elimination give
```text
U'(q)+k qddot+h q^{(4)}=0,
```
the original fourth-order Euler–Lagrange ODE. The Lean `auxiliary_elimination_at_jet` states the corresponding **exact algebraic jet equivalence**. Derivatives, variation, boundary conditions and multiplier preservation are analytic obligations not proved by that real-variable lemma.

## 2. Canonical reduction and the sharp restricted obstruction

For `h≠0`, `pQ=-h Qdot`, so `Qdot=-pQ/h` is an invertible Legendre map. The auxiliary first-order constraints give `p_q=λ` and `p_λ=0`. Their two-real-component primary bracket is `[[0,-I₂],[I₂,0]]`, invertible; the new Lean `auxiliary_bracket_surjective` proves an explicit inverse map. Assuming the regular multiplier-preservation result of the analytic Dirac algorithm, the two-real-component auxiliary formulation has 6 configuration coordinates and 4 second-class constraints: `12-4=8` physical phase coordinates = **4 configuration DOF**, including two extra ones relative to an ordinary complex second-order field. The arithmetic is Lean-checked; the full infinite-dimensional constraint algorithm is not.

For the one-real-component representative,
```text
Hred(q,Q,p_q,pQ)=p_q Q - pQ²/(2h) - (k/2)Q² + U(q).
```
At **any Q≠0**, the function `p_q↦Hred` is surjective onto ℝ; in particular it is unbounded both above and below. `R16_AuxiliaryEquivalence.lean` constructs for every desired energy E the exact canonical momentum
```text
p_q = [E +pQ²/(2h)+(k/2)Q²-U(q)] / Q.
```
The Lean theorems `auxiliary_hamiltonian_onto`, `auxiliary_hamiltonian_unbounded_below` and `auxiliary_hamiltonian_unbounded_above` prove the claim. This is the precise failure of **exact auxiliary-field rewriting within the regular nondegenerate class** as a ghost-removal mechanism. Adding `Q=0` as an independent multiplier restriction is not equivalent to the representative: even its free case `q(t)=t` solves the fourth-order equation while having `Q=qdot=1`, excluded by the extra constraint. `zero_auxiliary_velocity_is_an_extra_restriction` checks the corresponding jet counterexample.

These statements strengthen but do **not** expand the original S004 no-go into a universal theorem for derivative-dependent field transformations, new gauge sectors, dynamical foliation, specifically tuned DHOST actions, nonlocal projections, or different boundary conditions.

## 3. Candidate and parameter classification

| Route | Actual mathematical status | Action equivalence |
| --- | --- | --- |
| Original strict S004 at Fκ>0 | 2x2 Hessian nondegenerate, affine unbounded Hamiltonian; static pole upper residue negative and UV discriminant complex | Original **unhealthy** regular sector |
| Exact first-order auxiliary variables λ, Q | Same fourth-order jets and unbounded Hred; constraints remove added artificial coordinates | **Equivalent rewrite**, NOT a cure |
| Regular invertible derivative-free point change | det Hnew=4(Fκ)²(det J)²>0 | Equivalent variables, NOT a cure |
| Retune γ,θ with Fκ>0 | Hessian unaffected | Original, NOT a cure |
| Positive replacement weight W(u)>0,κ>0 | New Hessian remains nondegenerate | New weight, NOT a cure |
| Fκ=0 degeneracy / κ=0 | Acceleration-square Hessian vanishes; mixed terms and time dependence may remain; scalar constraints not proven | **Singular/different sector**, health unresolved |
| Extra constraint Q=0 | Excludes original admissible trajectories, so not equivalent without independently justified dynamics | Different constrained action |
| Published scalar–tensor A1 candidate | Indefinite scalar kinetic signature from earlier PR #34 | Specified candidate fails |
| Spatial-only S004 boundary κ=γ=0,θ>0 | Second-time-order but monotone static q+Tq²; not a finite-q selector | Degenerate boundary, NOT strict S004 |
| Independent competing-gradient action | Positive linear frequencies, unique q*=a/(2b)>0, finite Fourier energy bound, defocusing no nonzero static localized equilibrium | **Different theory**, not a ghost-free S004 completion |
| Dynamical gauge/foliation/DHOST completion | No exact healthy action/constraint algebra established | OPEN; restricted proof cannot exclude |

## 4. Independent competing-gradient model — checkable details

The explicitly different action is
```text
S_alt=∫[|ψ_t|²−μ|ψ|²+a|∇ψ|²−b|Δψ|²−(λ/2)|ψ|⁴] dt dx.
```
Its smooth Euler–Lagrange equation is
```text
ψ_tt+μψ+aΔψ+bΔ²ψ+λ|ψ|²ψ=0.
```
The linear dispersion is `ω²=μ−a q+b q²`. If `a,b>0,4bμ>a²`, it has positive squared frequencies and a unique finite-q oscillatory minimum at `q*=a/(2b)`; there is **no growing band** in this regime. For `λ≥0`, its Hamiltonian quadratic Fourier symbol is strictly positive and the quartic is nonnegative. Previously checked finite-mode coercivity and a conditional weak-stationary energy identity provide rigorous **supporting steps**, not an imported continuum nonlinear stability theorem.

For a smooth finite-energy static solution, Plancherel and weak testing yield
```text
∫[μ−a q+b q²]|ψhat|² + λ∫|ψ|⁴ =0.
```
The strict gap and λ≥0 force ψ=0. Thus this specific positive-energy defocusing model provides **no nontrivial static localized equilibrium**. Time-periodic, traveling, focusing or constrained states are not excluded; no full orbitally stable localized state is established.

## 5. Formalization ledger and explicit stopping points

| Claim | Evidence | State |
| --- | --- | --- |
| S004 F(u)>0 for u>0,ε≠0 | `R16_GhostObstruction.corrected_weight_pos` | Lean |
| Actual 2-real acceleration Hessian and determinant | `R16_EnergyAndConstraints.s004_acceleration_det_exact` | Lean |
| Exact degeneracy iff Fκ=0 | `R16_AuxiliaryEquivalence.two_real_s004_hessian_degenerate_iff` | Lean |
| Exclusion of strict Fκ>0 degeneracy | `strict_s004_cannot_remove_hessian_mode` | Lean |
| Regular point Jacobian congruence | `R16_CompletionClasses.invertible_point_change_cannot_degenerate` | Lean |
| Auxiliary Lagrangian matches 4th-order jet constraint | `auxiliary_restriction_exact` | Lean finite representative |
| Auxiliary Euler jet elimination | `auxiliary_elimination_at_jet` | Lean algebra + analytic variation |
| Legendre invertibility for h≠0 | `auxiliary_Q_velocity_reconstructed`, `auxiliary_kinetic_legendre_identity` | Lean |
| Hred surjectivity and both unbounded directions | `auxiliary_hamiltonian_onto`, `auxiliary_hamiltonian_unbounded_* ` | Lean |
| Primary bracket injectivity and surjectivity | `auxiliary_bracket_no_null_vector`, `auxiliary_bracket_surjective` | Lean |
| 4 real configuration DOF in full first-order auxiliary reduction | `two_field_auxiliary_constraint_count` | Conditional Lean arithmetic + analytic Dirac |
| Constraint Q=0 alters admissible dynamics | `zero_auxiliary_velocity_is_an_extra_restriction` | Lean jet counterexample (free representative) |
| Auxiliary action identity, EL, Hamiltonian, two-real determinant | `derivations/ghost_16_auxiliary_verify.py` | Exact symbolic CI |
| Full original nonlinear S004 field variation/Dirac algorithm | H2/earlier foundational analysis | Analytical; not fully Lean |
| Nonlinear localized healthy WCT and spectral/orbital stability | No qualifying completion | **OPEN** |

**Success classification:** a **rigorously delimited no-go** for standard regular auxiliary rewriting of the nondegenerate S004 acceleration principal sector, alongside preserved full strict-sector ghost and UV obstructions. A ghost-free, fully nonlinear WCT completion **does not currently exist as a proved result here**. Do not close issue #16; neither full continuum constrained dynamics nor healthy localized states have been certified. The separated exact scale (#13), nonlinear bridge (#45), and background spectral (#46) obligations remain independent.
