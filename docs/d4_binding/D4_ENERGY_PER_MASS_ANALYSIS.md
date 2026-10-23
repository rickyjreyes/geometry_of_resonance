# Frozen D4 energy per mass

Date: 2026-10-23, America/Los_Angeles. Frozen base: PR #35,
`65499cabca7e177286b6541fa1a5a00d51a50ba5`.

## R1. Independent reconstruction

The pinned `wct-pde/src/wct_pde/projection_free.py` defines its integral norm
as `sum(abs(field)**2)*cell_area`, normalizes amplitude by the square root of
the mass ratio, and implements the Hamiltonian below. Its generic coefficient
defaults are not the frozen D4 parameters. The existence/coercivity receipts
and candidate fix the exact dyadic values

    beta = gamma = float.fromhex('0x1.05f1d178c75ebp+2'),
    m_D4 = float.fromhex('0x1.45b8a0bad2714p+1'), k=1.

The actual D4 root validator and its pinned audit document specify
`(Delta+1)^2`, independently fixing k=1. The validator's SHA256 matches the
original root receipt. The audit document at the pinned revision has a
different hash from the receipt's earlier document input; both hashes are
recorded. This distinction is not a rerun of the original interval campaign.

The unchanged independent whole-space problem of PR #34 is

    X=H²(R²; C), A=Delta+k²,
    M=integral |u|², D=||Delta u||², K=||grad u||²,
    T=||Au||²=D-2k²K+k⁴M, Q=integral |u|⁴, S=integral |u|⁶,
    H=T-beta Q/2+gamma S/3,
    I(m)=inf {H(u): u in X, M(u)=m}, I(0)=0.

Integrals use Lebesgue measure, not the mean on the length-42 torus. There is
no extra 1/2 in T, no subtraction of k⁴M, and no restriction to real, radial,
periodic or band-limited fields. The evolution is
`i u_t=A²u-beta|u|²u+gamma|u|⁴u`. No physical mass interpretation is added.

For beta,gamma,k>0 in dimension two, H² embeds into L-infinity and every
finite Lp, and is an algebra. H is continuous and real C¹ on X. With
C=3beta²/(16gamma), the exact pointwise identity

    -beta r²/2+gamma r³/3
      =(gamma/3)r(r-3beta/(4gamma))²-Cr

gives H>=T-CM. Also `||Delta u||<=||Au||+k²||u||`, so fixed mass and an
upper energy bound imply bounded H². This is constrained coercivity; H need
not be positive. K<=sqrt(MD) and T>=0 follow by Fourier integration.

To independently verify negativity, take

    f_ab=(pi*a*b)^(-1/2)exp[-x²/(2a²)-y²/(2b²)],
    u_R=sqrt(m)R^(-1)f_ab(x/R,y/R)exp(i k x).

Then M=m and H=c2/R²+c4/R⁴ with

    c2=2mk²/a²-beta*m²/(4pi*a*b),
    c4=m[3/(4a⁴)+1/(2a²b²)+3/(4b⁴)]
          +gamma*m³/(9pi²a²b²)>0.

Choose a/b>8pi*k²/(beta*m), and R²=-2c4/c2. Thus `-Cm<=I(m)<0` for every
m>0. This proof uses complex anisotropic trials, the whole plane and the
verified squared norm. It does not assume compactness or attainment.

I is locally Lipschitz on positive masses: on a compact positive mass
interval, near-minimizers with H<=1 have uniform H²,T,Q,S bounds. Amplitude
normalization from mass a to b changes H by O(|a-b|) uniformly; reverse a
and b. The lower bound and I<0 give continuity at zero.

The inherited Fourier integral is independently recovered:

    (2pi)^(-2) integral_R² 1/[(|xi|²-k²)²+k⁴] dxi=3/(16k²).

It gives PR #35's K_inf=3(k⁴+C)/(16k²), and
`m_*=3beta/(4gamma K_inf)=4beta*k²/[gamma(k⁴+C)]`.
At frozen coefficients this is exactly
`72057594037927936/31838933052907969=2.2631912293727647...<m_D4`.
Its sequence-uniform proof covers `a+b<=2m_*`, and its central split
interval is `[2m_D4-2m_*,2m_*-m_D4]`. No material error was found in that
restricted theorem or its conditional compactness/stability proof. The old
OPEN classifications remain unchanged historical evidence.

## R2. Scaling and the unsuccessful unrestricted shortcut

For t,lambda>0, exact changes of variables give

    M(sqrt(t)u)=tM,
    H(sqrt(t)u)/t=T-beta*t*Q/2+gamma*t²*S/3;
    u_lambda(x)=lambda*u(lambda*x), M(u_lambda)=M,
    H(u_lambda)=lambda⁴(D+gamma*S/3)
                 -lambda²(2k²K+beta Q/2)+k⁴M;
    v_t(x)=u(x/sqrt(t)), M(v_t)=tM,
    H(v_t)/t=t^(-2)D-2k²*t^(-1)K+k⁴M-beta Q/2+gamma S/3.

The amplitude defect is

    H(sqrt(t)u)/t-H(u)=(t-1)[-beta Q/2+gamma(t+1)S/3].

The positive sextic term prevents a universal profile inequality. Narrow
Gaussians at m_D4 have positive amplitude derivative. PR #35 also has
explicit negative-energy large-mass counterexamples to that shortcut.
Negativity alone does not prove the necessary sign. Its old estimate
S/Q<=K_inf M only guarantees the required sign for a+b<=2m_* (using its
negative-energy refinement at equality). Failure outside that sufficient
range was not an interior equality counterexample.

Varying the old Fourier weight gives
`J_eta=(pi/2+atan(k²/sqrt(eta)))/(4pi sqrt(eta))` and
`S/Q<=J_eta(C+eta)M` on negative-energy profiles. An exploratory numerical
optimization does not reach D4. It is not a certified global optimum over
eta. The next exact argument avoids that loss in the estimate.

## V1. Exact adjustment of arbitrary minimizing sequences

For nonzero u, let P=D+gamma S/3>0 and B=2k²K+beta Q/2>0. The energy is a
quadratic in z=lambda², with unique positive minimum at z0=B/(2P):

    H(u_lambda)-H(u_sqrt(z0))=P(lambda²-z0)².

Writing w=u_sqrt(z0), the exact scalar stationarity identity is

    D(w)-k²K(w)-beta Q(w)/4+gamma S(w)/3=0.       (V)

For ANY mass-a minimizing sequence u_n, its adjusted sequence satisfies
`I(a)<=H(w_n)<=H(u_n)` and hence H(w_n)->I(a). Coercivity bounds w_n in H².
No limit for lambda_n, minimizer, stationary branch or differentiability
of the spatial orbit in H² is assumed. Only the exact scalar polynomial is
differentiated. H and the admissible space remain unchanged.

## V2. The stronger sextic/quartic bound

Set rho=k⁴+3beta²/(32gamma), sigma=sqrt(rho), L=sigma/4, and

    m_B=3beta/(gamma sigma).

Eliminating K from (V), followed by square completion, gives

    D+T+(2gamma/3) integral r(r-3beta/(8gamma))² = rho M,
    r=|w|².

All left summands are nonnegative, so D<=rho M. Independently, Fourier
Cauchy–Schwarz and Parseval give

    ||u||_infinity² <= (D+eta M)/(8sqrt(eta)), eta>0,

because `(2pi)^(-2)integral_R² (|xi|⁴+eta)^(-1) dxi=1/(8sqrt(eta))`.
Choose eta=D/M>0 to get `||u||_infinity²<=sqrt(MD)/4`. Density extends this
from Schwartz functions to all complex H². Thus every adjusted profile obeys

    S/Q<=||w||_infinity²<=L M.                   (R)

No sharpness is asserted. This stronger bound suffices at the frozen norm.

## V3. Quantitative strictness at the infimum

For 0<a<b and a+b<2m_B, put h=1-(a+b)/(2m_B)>0. Then

    e(b)<e(a), e=I/m,
    e(a)-e(b)>=(b-a)h(-I(a))/a²>0.             (E)

Proof: apply V1 to an arbitrary minimizing sequence of mass a. With t=b/a,
(R) bounds the amplitude bracket by `-(beta/2)h Q_n`. Also
`Q_n>=-2H(w_n)/beta`, since T,S>=0. Consequently

    I(b)/t<=H(sqrt(t)w_n)/t
       <=H(w_n)-(t-1)(beta/2)h Q_n
       <=[1+(t-1)h]H(w_n).

Passing to the limit and dividing by a proves
`e(b)<=[1+(b-a)h/a]e(a)<e(a)`, because I(a)<0. The strict defect is bounded
away from zero for each fixed mass pair. Pointwise strictness has therefore
not been lost while taking infima. Any proved trial upper bound U(a)<0
replaces -I(a) by -U(a) in the lower bound for the defect.

In particular e is strictly decreasing for every `0<m<=m_B`, including the
upper mass since a<b<=m_B implies a+b<2m_B. This is an infimum theorem,
not an assertion about amplitude scaling of every admissible profile.

## V4. Exact frozen comparison

At beta=gamma, k=1,

    rho=49853331562389953/36028797018963968,
    m_B=3/sqrt(rho)=2.5503456363515490892...,
    m_D4=2.5446968948571200286... .

Before evaluating a square root, exact arithmetic gives

    9-rho*m_D4²
      =454708712324087584212476662678391978900775591 /
       11417981541647679048466287755595961091061972992 >0.

Thus m_D4<m_B, with margin approximately 0.00564874149442906, and
`h0=1-m_D4/m_B=0.002214892528257459...>0`.
Rational outward enclosures, not printed decimals, verify the gate.
The new range is sufficient; it is not a sharp critical mass or an all-mass
theorem.
