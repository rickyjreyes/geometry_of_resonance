# Physical active-domain derivation

**Result: `PHYSICAL_ACTIVE_DOMAIN_NOT_DERIVED`.** The inspected equations do not identify a finite physical interval in a measured log observable. The following explicit nonuniqueness results establish why the present premises are insufficient. They do not rule out all localized WCT solutions or future physical completions.

## What would constitute a domain map?

The desired map is `W=mathcal W[psi,C]=[ln(x_minus/x0),ln(x_plus/x0)]`, where C is independently measured preparation/environment/boundary data. A spatial domain Omega, a filament core Gamma, a density percentile radius, a Fourier support shell and a measured mass interval are different objects.

One conditional construction is

`Gamma=Extract[psi,C]`, `x=X[psi,C;s]`,

`W=[ln(min_Gamma X/x0),ln(max_Gamma X/x0)]`.

This is only a type-correct template. Neither the physical extractor, map X, endpoint phase identification nor a reason that this image is one active phase interval has been derived. Extrema can coincide, images can be disconnected, and a closed spatial loop does not become a log-mass loop by notation.

## Proposition D1: the locking law does not select a loop radius

Assume the already published fixed-loop locking functional of P04/P08, a planar circle of radius R>0, positive constant line weight, no holonomy, longitudinal winding n=1 and sigma=kappa=1/R. Then q=1/R, `integral q ds=2pi`, and the minimum locking cost is zero **for every R>0**.

Proof: the circumference is 2pi R and the product with 1/R is 2pi. The mismatch is identically zero. If one also imposes the conditional mass law, it gives `M=hbar/(cR)` for every R; it does not select R. Fixing total line energy by choosing w proportional to 1/R leaves the zero mismatch intact. The physical loop size is a continuous family until another independent condition selects it.

This is a counterexample to extracting an intrinsic radius from closure/locking alone. It is not asserted to be a family of stable solutions of the full unclosed action. The full stationary solution is precisely the missing premise.

## Proposition D2: exact PDE solutions do not supply an intrinsic interval

For M4 on a circle of externally specified circumference S, every integer n has an exact plane wave

`psi=A exp(i(q s-omega t))`, `q=2pi n/S`,

`omega=(k_star^2-q^2)^2-beta A^2+gamma A^4`.

Direct substitution proves the statement. The family exists for every S>0 and A>0, with the stated periodic boundary condition. No hard edge or preferred circumference appears in these solutions. Fixing S gives a spectrum conditional on a box; leaving S unspecified leaves a continuous input. Existence of these solutions makes no stability or self-binding claim. Complete initial/boundary data may determine a trajectory; those data do not themselves define an observable domain functional.

## Proposition D3: no fixed-mass free-space ground state for the positive spatial energies alone

Let f be a nonzero smooth compactly supported field on R^d, with fixed positive L2 norm, and define `psi_R(r)=R^(-d/2) f(r/R)`. Then `||psi_R||_2` is constant and `||grad psi_R||_2^2=R^-2 ||grad f||_2^2`.

For P04's companion curvature energy,

`integral |Delta psi_R|^2/(|psi_R|^2+delta^2) <= delta^-2 R^-4 ||Delta f||_2^2`.

For the squared complex-safe curvature `Theta=-(Delta psi)bar(psi)/D(|psi|^2)`, its numerator scales as R^(-d-2); after squaring and integration its contribution is O(R^(-d-4)). Indeed D is uniformly bounded below for R>=1 by `epsilon^2 exp(-2 alpha ||f||_infinity^2)>0`. The legacy positive energy has the already published O(R^-4) curvature bound in the small-amplitude admissible tail (S037).

Thus each positive energy `c1||grad psi||_2^2+c2 E_curvature`, c1,c2>0, has fixed-mass infimum zero. A nonzero L2 field cannot attain it: zero gradient energy would make it constant, and a nonzero constant is not square-integrable on R^d. There is no global finite-radius ground-state minimizer in these unrestricted free-space problems.

Scope: this does not exclude excited/saddle states, externally confined problems, fixed-geometry topological sectors, additional potentials or M4's attractive-cubic/repulsive-quintic model. The proof extends the preserved positive-energy obstruction to the two corrected spatial regularizations; it is not a no-go theorem for every WCT extension.

## Proposition D4: Willmore shape selection leaves size free

For a torus of revolution, `W_Willmore=pi^2/[eta sqrt(1-eta^2)]`, eta=a/R in (0,1). Differentiation gives the unique ratio minimum eta^2=1/2, with value 2pi^2. But `(R,a)->(sR,sa)` leaves eta and the energy unchanged for every s>0.

Even granting the paper's narrow-shell reduction and its fixed profile/norm assumptions, this exact shape ratio cannot select an absolute size or log-observable endpoints. The full energy, including leading area, profile, phase and finite-thickness terms, is a distinct variational problem. This audit does not promote a formal asymptotic shape reduction to a full free-boundary theorem.

## Candidate boundary mechanisms

For clarity, a natural boundary condition can be derived without selecting the boundary's position. On a fixed smooth domain, consider either companion energy `E=integral[|grad psi|^2+G(rho)|Delta psi|^2]`, where `G=1/(rho+delta^2)`, or the corrected spatial energy with `G=F(rho)`. Integration by parts in the variation with respect to bar(psi) gives the boundary term

`integral_boundary [(partial_n psi-partial_n(G Delta psi)) delta bar(psi)+(G Delta psi) partial_n(delta bar(psi))]`.

Free independent value and normal-derivative variations therefore give `G Delta psi=0` and `partial_n psi-partial_n(G Delta psi)=0`. Alternatively, fixing both traces cancels those variations. These are conditional natural boundary conditions on an already specified domain; they give no equation for its location. A free-boundary problem needs an additional domain-displacement variation and a defined interface/environment energy. For M4's Hamiltonian, the corresponding free-trace conditions are `(Delta+k_star^2)psi=0` and `partial_n[(Delta+k_star^2)psi]=0`; the inspected solver instead uses periodic conditions.

| Mechanism | Mathematical consequence available | Domain classification / missing physical input |
|---|---|---|
| Stable localization | A solution may have a density profile and characteristic widths | Stability/branch-dependent; a threshold or quantile is a continuous analysis convention, not a hard physical endpoint. |
| Curvature feedback alone | Positive spatial energies admit the spreading sequence D3 | No intrinsic ground-state radius in that stated class. |
| Phase locking | Fixed-loop minimizer and mismatch cost | Continuous radius degeneracy D1. |
| Stationary action | Bulk Euler–Lagrange equation and boundary terms | Requires a declared domain, boundary data or a free-boundary/interface action. Natural boundary conditions on a chosen boundary do not locate it. |
| Energy/Lyapunov control | Controls norms or monotonicity for the specified flow | Bounds do not uniquely select support; physical-time compatibility must be supplied. |
| Spectral boundary-value problem | Discrete modes on a specified cavity; e.g. q_n=2pi n/S | Conditional on external S, material/interface data and boundary condition. |
| Filament topology | Winding on a prescribed closed, node-free contour | Topology labels a class; metric length and observable endpoints remain free. |
| Bifurcation/finite-band onset | Spatial q_star=sqrt(a/(2b)) and instability band | Coefficients are inputs; no derived map to a measured log-domain. |
| Turning points | Roots of a specified effective dispersion/potential | Potential, eigenvalue and response coordinates must be fixed independently; WKB turning points are not exact hard edges. |
| Coherence loss | Could define a boundary if a physical coherence observable and transition criterion were derived | No such unique criterion is supplied by the inspected equations. |
| Interface of phases | Could give a free-boundary problem with matching/stress conditions | Requires a constitutive interface law and a selected solution branch; currently not derived. |

An analytic decaying profile need not have compact support. Conversely an imposed cavity can have a real physical boundary, but it is an external condition rather than an intrinsically selected particle boundary. No such independently specified cavity-to-log-spectrum experiment is defined in the present corpus.

## Minimum added information

A usable W requires (1) one fully specified physical dynamical/variational model and preparation, including a binding or interface mechanism; (2) a theorem or independently measurable boundary rule selecting a physical set without residual thresholds; and (3) a map from that set to the positive measured variable x, with endpoint matching. Selecting an arbitrary density threshold, radius, geometric scale or winding to obtain closure does not fill these obligations.

The stopping condition is met: the reviewed premises admit the explicit degeneracies above and contain no completed observable-domain bridge. Further empirical peak searches or new large PDE runs cannot identify a functional that has not been physically defined.
