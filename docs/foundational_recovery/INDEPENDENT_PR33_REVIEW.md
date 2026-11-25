# Independent reconstruction of PR #33

Date: 2026-10-20 (America/Los_Angeles). Reviewed immutable commit
`0385386e3daff64b44e5c5062f8d578ea68d146d`; it remains an open draft.
**H2, H4 and M1: independently reconstructed and VERIFIED.** No defect in
these three claims was demonstrated. Independence means a fresh derivation
and implementation in this workflow, not an independent human reviewer,
proof assistant, or arithmetic backend. The new implementation does not import
the PR #33 script. It was executed before replaying that script.

## R1. Regular scalar constraints and canonical quantities

Use signature (-,+,...,+), tau=c0 t, x=(Re psi,Im psi), rho=x.x,
D=rho+epsilon^2 exp(-2 alpha rho), f=rho/D^2, epsilon>0.
Put v=x_tau, a=x_tautau, b=Delta x. In the physical orientation L=-L_orig,

    L = v.v-|grad x|^2+V(rho)
        -f[kappa (b-a).(b-a)+theta b.b+gamma (b-a).b].

Assume an open smooth regular patch f>0, kappa!=0, with periodic or compact
variations (or vanishing boundary flux). Then

    L_vv=2 I_2,       L_aa=-2 kappa f I_2.

In particular gamma and theta cannot make this acceleration Hessian singular.
Introduce independent v and lambda with L1=L+lambda.(dot x-v). Coordinates
are (x1,x2,v1,v2,lambda1,lambda2); their 12 momenta/coordinates obey only

    C_i=p_xi-lambda_i=0,    D_i=p_lambdai=0,
    { (C,D), (C,D) } = [[0,-I_2],[I_2,0]] delta(space-space').

This matrix has determinant one. Since

    p_v=-f[2 kappa a-(2 kappa+gamma)b],
    a=((2 kappa+gamma)b-p_v/f)/(2 kappa),

no other primary constraint arises. For H_T=H_c+mu.C+nu.D, consistency gives

    dot D=-v+mu=0,    dot C=-delta H_c/delta x-nu=0.

Both multiplier vectors are fixed: mu=v and nu=-delta H_c/delta x.
There are no secondary or tertiary constraints. Four constraints are second
class, none first class. Eliminating (lambda,p_lambda) leaves canonical
(x,v;p0,p1), eight real phase variables: **four real configuration degrees**.
The auxiliary formulation changes the description, not the solution space.

With A=2 kappa+gamma, p1=p_v and p0=2v-dot p1, direct Legendre inversion gives

    H = integral [p0.v-v.v+|grad x|^2-V
        -|p1|^2/(4 kappa f)+A p1.b/(2 kappa)
        +(theta-gamma^2/(4 kappa))f |b|^2],
    Q = integral [p0.Jx+p1.Jv],       J(x1,x2)=(-x2,x1).

The Q brackets generate simultaneous rotations of x and v, not an arbitrary
space-time-dependent phase. Energy and Q are formal conserved quantities for
smooth solutions with the stated boundary conditions, not existence proofs.
The inherited fixed-Q descending variation continues to apply. No extra
constraint can be obtained by fixing one global conserved number.

Failure boundaries: at f=0 or kappa=0 this Legendre chart is invalid and a
separate analysis is necessary. f=0 occurs at the zero field, not on a
nonzero open field sector. An overall sign reverses both mode signs.
Dynamic metrics or gauge fields are outside this fixed-background count.

## R2. Complete static homogeneous symbol and UV proof

Let psi0=r>0 be constant, with V'(r^2)=0 and finite V''(r^2). This condition is
necessary; the theorem asserts nothing if the chosen potential has no such
background. Perturb psi=r+eta+i xi. Since the background derivatives vanish,
derivatives of f multiply terms of order at least three and do not enter the
quadratic action. V contributes 2 r^2 V'' eta^2 to L_phys and no xi^2 term.
Both sectors therefore have the scalar symbol below, with m^2=-2r^2 V''
for eta and m^2=0 for xi. There is no amplitude/phase mixing on this static
background; the separate rotating-background matrix is not replaced by this.

Write K=f0 kappa, G=f0 gamma, T=f0 theta, q=|k|^2, z=omega^2. Variation of the
real quadratic density before Fourier substitution gives

    -u_tautau+Delta u-m^2 u-K u_tautautautau
      +(2K+G)Delta u_tautau-(K+G+T)Delta^2 u=0.

Consequently

    D(z,q)=(z-q)-m^2-K(z-q)^2+Gq(z-q)-Tq^2,
    P(s,q)=K s^4+[1+(2K+G)q]s^2+q+m^2+(K+G+T)q^2,
    D(-s^2,q)=-P(s,q).

Its exact poles and discriminant are

    z_±=q+[1+Gq ± sqrt(d(q))]/(2K),
    d(q)=1-4Km^2+2Gq-Bq^2,       B=4KT-G^2.

For K,T>0 and B>0, define a=(2K+G)/(2K), b=sqrt(B)/(2K)>0. The principal
symbol is -K(z-aq)^2-Bq^2/(4K), so its two z roots are nonreal and distinct
for all sufficiently large q. They satisfy z_±=(a±ib)q+O(1). The four s
roots include a growing pair with

    Re s(k)=c_UV |k|+O(|k|^-1),
    c_UV=sqrt((sqrt(a^2+b^2)-a)/2)>0.

No sign condition on m^2 is needed for this limit. When d>0 and the poles are
real and simple, D_z(z_-)=sqrt(d) and D_z(z_+)=-sqrt(d). Pole collision does
not create a gauge identity: R1 still has an invertible Legendre map.

**Sobolev proof.** On a torus take a smooth one-mode solution with jet
(1,s_n,s_n^2,s_n^3) exp(i k_n.x+s_n tau), |k_n|->infinity. Normalize its
four-jet norm at a fixed tau0>0 to one. The initial norm is
exp(-Re(s_n)tau0), tending to zero. Any finite change in source/target
Sobolev orders contributes only a polynomial in |k_n| and cannot control
this exponential. A bounded real solution operator would complexify to a
bounded complex one, so complex notation does not evade the real assertion.
On R^d use smooth compact Fourier packets in a growing-root neighborhood,
normalize the final jet, and multiply the initial Fourier jet by
exp(-s(k)tau0). Its initial norm is bounded by a polynomial times
exp(-tau0 inf Re s). This tends to zero as the support escapes to infinity.
Thus no continuous linear extension exists between fixed finite-order Sobolev
jet spaces at positive time. This is linear ill-posedness, not a proof of
nonlinear norm inflation. It also obstructs a nonlinear flow differentiable
at that background with this bounded linearized propagator.
On R^d this is a perturbation problem about a homogeneous background, not a
finite-energy self-bound state. No arbitrary localized-background instability
theorem is inferred.

### Boundaries tested rather than assumed healthy

| Boundary | Actual conclusion |
|---|---|
| B=0, G=T=0, m^2=0 | z=q and q+1/K are real, but opposite residues and R1 remain |
| B=0, G>0 | Large-q poles are real positive; the regular higher-time ghost remains |
| B=0, -2K<G<0 | Imaginary z splitting is O(sqrt(q)); growth is O(1), not the strict-B formula |
| B=0, G=-2K, T=K | D=-Kz^2+z-q-m^2; growth is proportional to |k|^(1/2) |
| B=0, G<-2K | Leading z=(1+G/(2K))q<0; growth is proportional to |k| |
| f0=0 | Higher-derivative quadratic terms vanish; this vacuum does not remove them nearby |
| kappa=0 with nonnegative curvature form | Necessarily gamma=0; a different second-time-order spatial model results |

These boundaries do not falsify H4, whose hypotheses are strict. Twenty-seven
time-polynomial root probes include positive, negative and zero m^2; they are
algebraic diagnostics, not a propagated PDE experiment or the proof itself.

## R3. Mass-average theorem and actual locked curve

Take H=-d^2/ds^2 on H^2_per([0,L]), a smooth nonvanishing eigenfunction
chi=R exp(i phi), R>0, and N=integral R^2. The phase may wind; its derivative
is well-defined. Assume the extra lock phi'=sigma=sqrt(curvature^2+torsion^2)
>=0 and weight w=R^2/N. Periodic integration by parts gives

    lambda N=integral |chi'|^2=integral [(R')^2+R^2 sigma^2].

Subtracting the squared weighted mean proves M1 exactly. Both residual terms
are nonnegative; equality holds iff R and sigma are constant. This conclusion
uses the specified free operator, weight and lock; it is not a statement
about every possible confinement operator or a derivation of the lock.

For chi=cos(s)+i a sin(s), a>0, on length 2pi:

    lambda=1, R^2=cos^2(s)+a^2 sin^2(s), phi'=a/R^2>0,
    N=pi(1+a^2), mean_w(sigma)=2a/(1+a^2).

The unit tangent t=chi/R satisfies t(s+pi)=-t(s). Therefore the integral of
t over a period vanishes and its primitive is a smooth closed unit-speed
planar curve. Since its turning derivative is positive, curvature=phi' and
torsion=0; the claimed lock actually holds. There are no amplitude nodes.
Using integral 1/R^2=2pi/a, the amplitude-gradient contribution is 1-mu and
the variance is mu-mu^2, where mu=2a/(1+a^2). At a=1/2 these are 1/5 and
4/25, giving lambda-mu^2=9/25 and mu=4/5. Neither coordinate closure nor
the domain assumption rescues unconditional equality.

## Independent check boundary and literature

The new symbolic reconstruction includes the complete primary Poisson matrix,
Legendre inversion, actual phase generator, time-domain polynomial, exact
poles, boundary controls and polar identities. Separate quadrature checks
the locked curve and the two nonnegative M1 contributions. Existing scripts
are replayed only after this reconstruction and retain their original bytes.

Ostrogradsky's canonical construction is standard; see Woodard,
[arXiv:1506.02210v2](https://arxiv.org/abs/1506.02210v2). The explicit scalar
count and Fourier proof above, not the citation alone, establish the result
for this action. H2/H4/M1 were first recorded in PR #33; this document claims
reconstruction, not new priority for those results. The expanded B=0 case
classification is an additional scoped deduction.
