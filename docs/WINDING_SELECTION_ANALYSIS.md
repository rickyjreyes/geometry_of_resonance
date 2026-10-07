# Winding selection analysis

**Predictive classification: `NOT_IDENTIFIABLE`.** Genuine winding and a fixed-sector phase minimizer are well defined under explicit boundary assumptions. A particular physical system's selected integer has not been derived from the full WCT dynamics.

## Established topology versus a descriptive cycle count

If psi is continuous and nonzero on a closed contour Gamma, `psi/|psi|:Gamma~S1 -> S1` has integer degree

`n=(1/(2pi)) integral_Gamma d arg(psi)`.

The degree is invariant under continuous evolution that keeps the contour closed and the field nonzero there. It can change through a zero crossing, a contour/topology change, or external boundary-phase transport. This theorem quantizes possible sectors; it does not choose the sector or prove stability. In a vortex filament the field vanishes on the core axis, so phase must be extracted on a nonzero contour or from a defined modal coefficient, not by unwrapping arg(0).

A measured interval in ell is not automatically a circle. For `phi(ell)=a ell` on an interval of width L, a may be any real number and `Delta phi=aL`. Integer closure needs physical endpoint identification or a boundary condition. The quantity `n_eff=k_fit L/(2pi)` is continuous in a fitted frequency and is not automatically a topological invariant.

## Published fixed-geometry locking result

Use P04/P08's notation with a closed spatial loop of arclength S, positive weight w, normal-frame connection Omega, transverse integer m and longitudinal phase chi. Set

`q=chi'-m Omega`, `sigma=sqrt(kappa^2+tau^2)`,

`Phi_Omega=integral Omega ds`, `J=integral sigma ds`, `I=integral ds/w >0`.

In the declared endpoint/framing convention, `integral q ds=2pi n-m Phi_Omega`. A consistent change of framing changes chi and Omega together; q and the total closure expression are invariant. One must include any seam holonomy rather than treating every covariant phase increment as an integer multiple of 2pi.

The locking functional is

`S_lock=integral w(q-sigma)^2 ds`.

For fixed curve, weight, m and n, Cauchy–Schwarz gives

`S_lock >= (2pi n-m Phi_Omega-J)^2/I`.

Equality is attained uniquely in q when

`q=sigma+C/w`, `C=(2pi n-m Phi_Omega-J)/I`.

Proof: set f=q-sigma, use `(integral f)^2 <= (integral w f^2)(integral 1/w)`, and inspect its equality condition. The phase is unique up to an arbitrary constant, not only integer multiples of 2pi. The second variation is `2 integral w(eta')^2 ds`, positive except for constant eta within that same sector. This is a phase-only variational statement and not a full nonlinear field stability theorem.

Exact lock requires `2pi n=m Phi_Omega+J`. Generic geometry does not satisfy that equality. The density-weighted phase rate is also generally different from the unweighted mean: `mean_w(q)-mean(q)=Cov(w,q)/mean(w)`. A mass law using a weighted rate cannot be equated to `2pi n/S` without weight/holonomy conditions.

## New conditional proposition W1: minimization across integer sectors

Add the extra premises that Gamma,w,m are already fixed, the complete sector-dependent energy is the stated S_lock plus an n-independent term, and the physical preparation can reach a global sector minimum. Define

`zeta=(J+m Phi_Omega)/(2pi)`.

Then the admissible minimum-cost integers are exactly

`N_min={n in Z: |n-zeta|=dist(zeta,Z)}`.

This set has one member away from half-integers and two adjacent members at a half-integer. It follows by minimizing the convex quadratic `(2pi)^2(n-zeta)^2/I`. If additional core, kinetic, circulation, flux or topological energies depend on n, this formula changes; they cannot be omitted without derivation.

This is a conditional finite-branch result, **not** the classification `DERIVED_FINITE_BRANCHES` for the physical WCT problem. Its geometrical inputs, complete energy and inter-sector dynamics are not supplied by the active-domain model.

For a planar circle with m Phi_Omega=0 and sigma=1/R, J=2pi. W1 selects n=1 for this oriented locking convention. The cost is `2pi w (n-1)^2/R`. Every R remains allowed; the integer selection does not close the domain-selection problem.

## New proposition W2: fixed-amplitude phase relaxation cannot select a new sector

For a positive-weight smooth periodic phase-only gradient flow,

`chi_t=2 mu partial_s[w(chi'-m Omega-sigma)]`, mu>0,

with fixed geometry and periodic phase derivatives, differentiate the winding integral. The boundary term vanishes, so `d/dt integral chi' ds=0`. Thus every initial n remains n. The phase converges, if it does, within its sector; energy descent alone does not grant access to the W1 minimum in another sector.

A physical selection law requires specified initial topology, boundary injection or an amplitude/phase-slip mechanism with a derived transition criterion. The constant-positive-amplitude locking model contains no phase slips. Hamiltonian M4 additionally conserves mass and energy; irreversible relaxation to one universal state cannot be inferred from those conservation laws.

## Exact PDE family and energetic limitations

M4 admits the exact plane waves in D2 for all integers on an imposed circle. At fixed total mass M and circumference S their energies satisfy

`H_n/M=((2pi n/S)^2-k_star^2)^2-beta M/(2S)+gamma M^2/(3S^2)`.

Minimizing within this family selects the value `(2pi n/S)^2` nearest the supplied `k_star^2`, with sign degeneracy for nonzero n. Equivalently, n^2 is nearest `(k_star S/(2pi))^2` among integer squares. This restricted comparison is not minimization over every field, and says nothing about a localized toroidal branch's Hessian. It depends on S and k_star, neither of which is an independently derived log-spectral domain input.

## Disconnected domains and matching

A toroidal field may have two independent winding integers. Disconnected observed intervals may have separate phase lifts and unknown relative phases. Summing retained widths does not supply a global winding. One must declare physical connectivity, phase transport across gaps, endpoint twists and whether a contour remains nonzero. An analysis mask creates none of these conditions.

## A falsifiable additional hypothesis, not adopted as a result

`H_sector_min`: a specified full WCT evolution with independently measured fixed Gamma,w,m,holonomy and an established phase-slip mechanism relaxes to the W1 minimizing set, with all other n-dependent energy terms negligible in a controlled limit.

Falsifier: a converged, stationary, admissible solution in that same controlled limit settles outside N_min, or the limiting measured `w(q-sigma)` is not the constant C dictated by closure. Persistent sector trapping would reject the proposed relaxation premise, not the topological theorem. Finite-thickness and measurement error bounds must be derived before a finite-precision experiment is scored. No such new run is authorized here.

The minimum missing result is a reduction of one full physical WCT model to the locking energy and a preparation/transition law, with geometry and boundary data independently fixed. Defining n by rounding an observed frequency is excluded.
