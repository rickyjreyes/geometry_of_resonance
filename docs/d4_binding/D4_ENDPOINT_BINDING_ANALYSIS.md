# Endpoint asymptotics and energy slope

Date: 2026-10-23. See [the energy report](D4_ENERGY_PER_MASS_ANALYSIS.md) for
notation and V1–V4. No minimizer or stationary multiplier is a premise here.

## E1. Actual small-mass asymptotics

V1 adjusts arbitrary minimizing sequences at any mass s>0. V2 gives
Q<=L s², L=sigma/4, on that sequence. Since T,S>=0 and R1 gives negativity,

    -beta*sigma*s²/8<=I(s)<0,
    I(s)/s->0 as s->0+.

Thus `I(s)=lambda0*s+o(s)` with **lambda0=0**. This is a two-sided infimum
estimate, not a trial-state upper bound. PR #35's unadjusted bound separately
gives the same limit. Neither proof uses compactness.

Broad carriers sqrt(s)R^(-1)f(x/R)exp(i k x) have energy tending to zero.
This agrees with the bottom of (|xi|²-k²)². The linear bottom alone would
not control nonlinearities along minimizing sequences; the quadratic lower
bound supplies that missing uniform estimate. At every fixed s>0 the
infimum is negative, so such vanishing sequences are not minimizing.
Attainment for s<=m_B follows later, not as an assumption here.

## E2. One-sided derivatives without a smooth branch

R1 proves local Lipschitz continuity. I is also locally semiconcave. Fix
[a,b] with a>0. Near-minimizers with H<=1 have a uniform H² bound. After
division by sqrt(m), their unit-mass shapes lie in one fixed bounded H² set
V_R. Include all unit-mass shapes of norm at most that bound. Then

    I(m)=inf_(v in V_R)[m T(v)-beta*m²Q(v)/2+gamma*m³S(v)/3]

on the entire interval. The restriction does not change the infimum: it
contains arbitrarily accurate near-minimizers at each m, while all included
shapes yield admissible competitors. The second derivatives
`-beta Q(v)+2gamma*m*S(v)` have a uniform upper bound K. Subtracting Km²/2
makes every member concave, and an infimum of concave functions is concave.
This proves local semiconcavity. Consequently finite left and right
derivatives exist at every positive mass, with I'_-(m)>=I'_+(m). No equality
of them or C¹ branch is assumed. The requested convention is

    I'_-(m)=lim_(mu->0+) [I(m)-I(m-mu)]/mu.

## E3. Strictly negative frozen slope, before compactness

For m=m_D4, b=m-mu, V3 gives

    [I(m)-I(b)]/mu <=[1+(m/b)h(b,m)]e(b),
    h(b,m)=1-(b+m)/(2m_B).

Continuity and b->m imply the derivative-free result

    limsup [I(m)-I(m-mu)]/mu
       <=(2-m/m_B)e(m)
       <=(2-m/m_B)U(m)/m <0.                     (E3)

U is the explicit negative Gaussian bound in
[the subadditivity proof](D4_SUBADDITIVITY_PROOF_OR_COUNTEREXAMPLE.md).
The last quantity is approximately -0.0013068890384450; rational interval
arithmetic verifies its strict negativity. Hence

    liminf I(mu)/mu=0 > limsup [I(m)-I(m-mu)]/mu.

E3 does not require a derivative or minimizer. Using the independently proved
semiconcavity, one can additionally write

    I(m-mu)=I(m)-mu I'_-(m)+o(mu),
    Delta_m(mu)=-I'_-(m)mu+o(mu),
    -I'_-(m)>=(2-m/m_B)(-U(m))/m>0.

No energy derivative has been identified with a hypothetical stationary
frequency. Multipliers are considered only after existence.

## E4. Complete endpoint coverage

The explicit F(mu)>0 in the subadditivity report holds on all 0<mu<=m/2,
and symmetry covers the other half. It crosses the old exact endpoint
2m-2m_* and overlaps the entire inherited central certificate. This finite
interval is not being inferred from an unspecified o(mu) remainder.

Route A already proves strict energy-per-mass monotonicity over the needed
range. E1–E3 provide a separate endpoint conclusion with no appeal to S2,
compactness, existence or stability. The graph records these dependencies.
