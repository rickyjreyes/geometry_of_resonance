# Frozen D4 binding with a uniform rational split deficit

Date: 2026-10-29, America/Los_Angeles.
Reviewed repository base: `7264a760bee05fea89a1e42249b49f39bd078b78`.

**Outcome A: whole-space strict binding holds at the prescribed frozen mass.**
The qualitative result was already established in merged
[PR #36](https://github.com/rickyjreyes/geometry_of_resonance/pull/36), after
the PR #33–#35 baseline named in the request. This report reconstructs that
argument and adds the following quantitative corollary:

\[
\boxed{I_2(\mu)+I_2(m-\mu)-I_2(m)
  >\frac{\min\{\mu,m-\mu\}}{400000}>0,
  \qquad 0<\mu<m=m_{D4}.}
\tag{1}
\]

The estimate concerns the actual constrained infima. It uses no stationary
profile, fitted numerical infimum, or assumed strict-binding premise. The
analytic reduction is below; its final scalar certificate uses exact
rational arithmetic on a continuous interval. Neither the Python checks nor
this report are a proof-assistant formalization or external peer review.

## 1. Frozen problem and inherited coverage

Let \(X=H^2(\mathbb R^2;\mathbb C)\), with Lebesgue measure and no real,
radial, Fourier-band, momentum or square-symmetry restriction. Write

\[
\begin{aligned}
M(u)&=\int|u|^2,&D(u)&=\|\Delta u\|_2^2,&K(u)&=\|\nabla u\|_2^2,\\
T(u)&=\|(\Delta+k^2)u\|_2^2=D-2k^2K+k^4M,&
Q(u)&=\int|u|^4,&S(u)&=\int|u|^6,\\
H(u)&=T-\frac\beta2 Q+\frac\gamma3 S,&
I_2(s)&=\inf_{u\in X,\ M(u)=s}H(u).
\end{aligned}
\]

The exact parameters are

\[
k=1,\quad
\beta=\gamma=\frac{4608178181141995}{1125899906842624},\qquad
m=\frac{1432536998431173}{562949953421312}.
\tag{2}
\]

These are the dyadics `0x1.05f1d178c75ebp+2` and
`0x1.45b8a0bad2714p+1`. The norm is the squared integral norm, not its square
root or the torus mean. The coefficient of T is one; the k⁴M term remains.
This is the independent Hamiltonian, not the canonical higher-time-derivative
WCT/S004 action. D4 labels the source periodic state's symmetry; it does not
impose that symmetry on this whole-space minimization problem.

The seven pinned source files in `wct-pde` at
`051b6eca10c46b8e038993d60cabb20715194a20` were retrieved and byte-checked
against the Phase V ledger. The generic source defaults are not (2); the
frozen candidate and root/coercivity receipts specify (2).

PR #35 rigorously covered only

\[
[2m-2m_*,\,2m_*-m],\qquad
m_* = \frac{72057594037927936}{31838933052907969},
\]

approximately [0.5630113309687108, 1.9816855638884092]. Its complement was
open in that historical report. PR #36's improvement, reconstructed next,
covers every split, including both endpoint intervals. Historical manifests
and their dated OPEN statements have not been rewritten.

## 2. Finiteness, negativity and exact dilation

For this section take arbitrary \(\beta,\gamma,k,s>0\). With \(r=|u|^2\),

\[
-\frac\beta2r^2+\frac\gamma3r^3
=\frac\gamma3r\left(r-\frac{3\beta}{4\gamma}\right)^2
-\frac{3\beta^2}{16\gamma}r.
\]

Thus H≥T−CM, C=3β²/(16γ). Fixed mass and an upper energy bound control T,
then D by \(\|\Delta u\|_2\leq\|(\Delta+k^2)u\|_2+k^2\|u\|_2\).
They control the full H² norm by Fourier interpolation. In particular
\(I_2(s)>-\infty\), and all minimizing sequences are H²-bounded.

For completeness, every positive mass has negative infimum. Set

\[
f_{ab}(x,y)=(\pi ab)^{-1/2}e^{-x^2/(2a^2)-y^2/(2b^2)},\qquad
u_R=\sqrt{s}R^{-1}f_{ab}(x/R,y/R)e^{ikx}.
\]

Gaussian integration gives M=s and H=c₂/R²+c₄/R⁴, where

\[
c_2=\frac{2sk^2}{a^2}-\frac{\beta s^2}{4\pi ab},\qquad
c_4=s\left(\frac3{4a^4}+\frac1{2a^2b^2}+\frac3{4b^4}\right)
       +\frac{\gamma s^3}{9\pi^2a^2b^2}>0.
\]

Choose a/b>8πk²/(βs). Then c₂<0, and R²=−2c₄/c₂ yields
\(I_2(s)\leq-c_2^2/(4c_4)<0\). This is an admissible complex anisotropic
competitor in the unchanged space.

For any nonzero u, mass-preserving dilation \(u_\lambda(x)=\lambda u(\lambda x)\)
has the exact scalar energy polynomial

\[
H(u_\lambda)=Pz^2-Bz+k^4M,\quad
z=\lambda^2,\quad P=D+\gamma S/3>0,\quad B=2k^2K+\beta Q/2>0.
\]

Its unique positive minimizer is z₀=B/(2P), and the energy difference is
P(z−z₀)². For w=u_{√z₀}, differentiation of this polynomial gives

\[
D(w)-k^2K(w)-\beta Q(w)/4+\gamma S(w)/3=0.\tag{3}
\]

Apply this operation separately to every member of any mass-s minimizing
sequence. The adjusted sequence still minimizes: its energy is between
I₂(s) and the original energy. It is still H²-bounded by coercivity. No
convergence of the dilation factors, differentiability of an H² spatial
orbit, or existence of a minimizer is required.

## 3. The estimate that reaches the frozen mass

Put \(\rho=k^4+3\beta^2/(32\gamma)\) and \(\sigma=\sqrt\rho\). Combining (3)
with T=D−2k²K+k⁴M gives the exact square identity

\[
D+T+\frac{2\gamma}{3}\int r\left(r-\frac{3\beta}{8\gamma}\right)^2
=\rho M.\tag{4}
\]

Every summand on the left is nonnegative; hence D≤ρM on the adjusted
sequence. With the Fourier convention
\(u(x)=(2\pi)^{-2}\int e^{ix\cdot\xi}\widehat u(\xi)\,d\xi\), weighted
Cauchy–Schwarz and Parseval give, for any η>0,

\[
\|u\|_\infty^2\leq
\frac{D+\eta M}{8\sqrt\eta},\qquad
(2\pi)^{-2}\int_{\mathbb R^2}\frac{d\xi}{|\xi|^4+\eta}
=\frac1{4\pi}\int_0^\infty\frac{dt}{t^2+\eta}
=\frac1{8\sqrt\eta}.
\]

The identity first holds for Schwartz fields and extends by H² density.
For nonzero L² fields D>0: otherwise their Fourier transform is supported
at the measure-zero origin and is zero in L². Choosing η=D/M gives

\[
\frac SQ\leq\|u\|_\infty^2\leq\frac{\sqrt{MD}}4
\leq\frac{\sigma M}4\quad\text{on the adjusted sequence}.\tag{5}
\]

Define \(m_B=3\beta/(\gamma\sigma)\). At (2),

\[
\rho=\frac{49853331562389953}{36028797018963968},\qquad
9-\rho m^2=
\frac{454708712324087584212476662678391978900775591}
 {11417981541647679048466287755595961091061972992}>0.
\tag{6}
\]

Consequently \(m<m_B=3/\sqrt\rho=2.550345636351549\ldots\).
This is an exact sign decision, not a comparison of rounded masses.

## 4. Strictness survives taking the infimum

Write e(s)=I₂(s)/s. For 0<a<b with a+b<2m_B, let
\(h=1-(a+b)/(2m_B)>0\) and t=b/a. Along the adjusted mass-a sequence wₙ,
the amplitude identity and (5) imply

\[
\begin{aligned}
I_2(b)/t
&\leq H(\sqrt t\,w_n)/t\\
&=H(w_n)+(t-1)[-\beta Q_n/2+\gamma(t+1)S_n/3]\\
&\leq H(w_n)-(t-1)(\beta/2)hQ_n\\
&\leq[1+(t-1)h]H(w_n).
\end{aligned}
\]

The last inequality uses Qₙ≥−2H(wₙ)/β, since T,S≥0. Passing to the limit,

\[
e(a)-e(b)\geq\frac{(b-a)h[-I_2(a)]}{a^2}>0.\tag{7}
\]

The strictly positive right side is independent of n. It is not an inference
from mere pointwise strictness. Thus e is strictly decreasing on (0,m_B],
and weighting the two component ratios proves strict binding for every
0<s≤m_B and every interior split. In particular the entire frozen question
is resolved affirmatively by the Phase V argument, without the new certificate.

## 5. An explicit bound for the true split deficit

Return to (2). By symmetry take 0<μ≤m/2 and b=m−μ∈[m/2,m). Since
e(μ)≥e(b), (7) at a=b and target mass m gives

\[
\Delta_m(\mu):=I_2(\mu)+I_2(b)-I_2(m)
\geq\frac{m\mu}{b^2}
 \left[1-\frac{(m+b)\sigma}{6}\right][-I_2(b)].\tag{8}
\]

Use the Gaussian from section 2 with longitudinal width 10, transverse
width 1, and optimized R at mass b. It gives I₂(b)≤U(b), where

\[
c_2(b)=b/50-\beta b^2/(40\pi),\quad
c_4(b)=30203b/40000+\beta b^3/(900\pi^2),\quad
U(b)=-c_2(b)^2/(4c_4(b)).
\]

On the entire interval b∈[m/2,m], c₂<0 and c₄>0. The sign c₂<0 follows
from the positive d_min below. This is a valid negative trial upper bound
for every b in the interval, not a comparison of two trial infima.

Let A=30203/40000 and d(b)=βb/(40π)−1/50>0. Then

\[
\frac{\Delta_m(\mu)}\mu\geq
\frac{m[1-(m+b)\sigma/6]d(b)^2}
 {4b[A+\beta b^2/(900\pi^2)]}.\tag{9}
\]

## 6. Four exact positive coefficients certify all component masses

Use rational bounds

\[
p_-=333/106<\pi<355/113=p_+,\qquad
s_+=1176311146708614/10^{15}>\sigma.
\]

The π bounds are verified with twelve-term alternating rational series in
Machin's identity π=16 arctan(1/5)−4 arctan(1/239). The identity follows
from the tangent addition formula: tan(4 arctan(1/5))=120/119 and
tan(4 arctan(1/5)−arctan(1/239))=1, with the angle in (0,π/2).
The alternating remainder bounds are exact. The square-root bound follows
by checking s₊>0 and s₊²−ρ>0 with rational arithmetic.

On [m/2,m] the rational comparison factors obey

\[
\bar h(b)=1-(m+b)s_+/6\geq1-ms_+/3>0,\qquad
\bar d(b)=\beta b/(40p_+)-1/50
\geq\beta m/(80p_+)-1/50>0.
\]

Replacing σ and π with these bounds in the directions shown gives a lower
bound for the right side of (9):

\[
R(b)=\frac{m\bar h(b)\bar d(b)^2}
 {4b[A+\beta b^2/(900p_-^2)]}.
\]

Every denominator and every factor being squared is positive, so the
comparison preserves order. For c=1/400000, R(b)>c is equivalent to
positivity of the cubic

\[
P(b)=m[1-(m+b)s_+/6][\beta b/(40p_+)-1/50]^2
       -4cb[A+\beta b^2/(900p_-^2)].\tag{10}
\]

Put b=m(1+t)/2, t∈[0,1]. Expand exactly in the Bernstein basis:

\[
P(m(1+t)/2)=\sum_{j=0}^3 B_j\binom3j t^j(1-t)^{3-j}.
\]

The four weights are nonnegative and sum to one. The exact coefficients,
stored as fractions in [certificate.json](../../audit/d4_frozen_review/certificate.json),
satisfy these simpler rational lower bounds:

| Coefficient | Strict rational lower bound |
|---|---:|
| B₀ | 28/100000 |
| B₁ | 56/100000 |
| B₂ | 83/100000 |
| B₃ | 29/10000000 |

Thus P>0 on the closed interval, including both ends. There is no mesh,
sampling assumption or unspecified endpoint remainder. Equations (8)–(10)
prove Δₘ(μ)>μ/400000 on the half-interval. Reflection gives (1).

The certificate generator expands the polynomial using rational convolution
and a basis transformation. A separate test reconstructs B₀,…,B₃ by solving
the cubic interpolation system at t=0,1/3,2/3,1 directly from the factored
formula (10). Both calculations agree exactly. A larger proposed constant
1/300000 fails the certificate, as it should; no optimality is asserted for
1/400000.

## 7. Counterexamples, limits and compactness

Ordinary subadditivity holds: approximate two near-minimizers by compact
smooth fields, normalize their masses, and separate their supports by
translation. All local terms, including second derivatives, then add exactly.
Taking approximation errors to zero gives I₂(a+b)≤I₂(a)+I₂(b). Thus a binding
counterexample at a fixed split would have to be an equality. Equation (1)
excludes every such equality at m.

The endpoint limit μ→0 is not a fixed positive split. In fact, along the
adjusted sequence Q≤σs²/4, so
\(-\beta\sigma s^2/8\leq I_2(s)<0\), whence I₂(s)/s→0. I₂ is locally
Lipschitz on positive masses: bounded near-minimizers on a compact positive
mass interval have uniformly bounded T,Q,S, and amplitude rescaling changes
their energies by O(|a−b|); exchange a and b. It follows that Δₘ(μ)→0 at
either endpoint. This is compatible with (1), whose bound also tends to zero.

Broad modulated packets can spread with energy tending to zero. At each
fixed positive mass the infimum is strictly negative, so these packets
cannot be a minimizing sequence. Overlapping packets can have interaction
energies of either sign; no universal overlap-sign shortcut is used.
Negative-energy amplitude-scaling counterexamples in the inherited audit
do not satisfy the sufficient adjusted-sequence argument at the frozen mass.

For clarity, compactness follows as an additional analytic theorem. Take any
mass-m minimizing sequence. Coercivity bounds it in H². If its local L²
mass vanishes uniformly under translations, the local H¹ interpolation
inequality on a bounded-overlap cube cover gives Q→0. Bounded H² gives a
uniform L∞ bound, hence S≤\|u\|∞²Q→0. T≥0 then contradicts I₂(m)<0.

After translations and a subsequence there is a nonzero weak H² limit u,
with strong local Lᵖ convergence for finite p and almost-everywhere
convergence. Let a=M(u), vₙ=uₙ−u in those translated coordinates. Hilbert
splitting gives M(vₙ)=m−a+o(1) and T(uₙ)=T(u)+T(vₙ)+o(1). The Brezis–Lieb
lemma at p=4 and p=6 separately gives H(uₙ)=H(u)+H(vₙ)+o(1). If a<m,
normalizing vₙ to mass m−a changes its energy by o(1), so

\[
I_2(m)\geq H(u)+I_2(m-a)\geq I_2(a)+I_2(m-a),
\]

contrary to (1). Therefore a=m. Now vₙ→0 in L², Q(vₙ),S(vₙ)→0, and energy
splitting plus T(vₙ)≥0 forces H(u)=I₂(m) and T(vₙ)→0. The estimate
\(\|\Delta v_n\|_2\leq\|(\Delta+k^2)v_n\|_2+k^2\|v_n\|_2\)
upgrades convergence to strong H². Thus every minimizing sequence is
precompact modulo translations, and the full minimizing set Gₘ is nonempty.

This supplies the actual hypotheses of the standard concentration-compactness
method; it does not assume weak lower semicontinuity of the focusing energy.
Background: [Lions (1984)](https://numdam.org/item/AIHPC_1984__1_2_109_0/).

## 8. Dynamical consequences and remaining obligations

A minimizer obeys \((\Delta+k^2)^2u-\beta|u|^2u+\gamma|u|^4u=\omega u\),
with real ω. The inverse of (Δ+k²)²+1 gains four derivatives. Since H² in
two dimensions is an algebra, elliptic bootstrapping gives smoothness and
all finite Sobolev regularities; uniform continuity and L² imply spatial
decay to zero. Dilation minimality gives (3), and (5) implies ωm<I₂(m)<0
at the strict interior mass m<m_B. No periodic frequency is substituted.

Orbital stability needs more than binding. For the unchanged evolution
\(iu_t=(\Delta+k^2)^2u-\beta|u|^2u+\gamma|u|^4u\), its linear group is
unitary on H² and its polynomial nonlinearity is locally Lipschitz H²→H².
Duhamel contraction gives local existence, uniqueness, continuous dependence
and the H² blow-up alternative. For smooth data, self-adjoint differentiation
conserves M and H. Smooth approximation and local H² continuous dependence
pass these identities to H² solutions. Higher regularity persists locally
by the tame polynomial product estimate. Conserved M,H and H≥T−CM bound
the H² norm for all time, excluding blow-up in both directions.

Finally suppose initial data approach Gₘ but at some times their solutions
remain a fixed positive H² distance away. Gₘ is uniformly bounded in H²;
continuity of M,H on bounded sets, conservation and coercivity make the
states at those times a bounded sequence whose masses tend to m and energies
to I₂(m). Normalize their masses by factors tending to one. The resulting
minimizing sequence is compact modulo translations, contradicting its
distance from the translation-invariant set Gₘ. This proves full-set orbital
H² stability, including small mass-changing complex perturbations, using
all the required dynamical assumptions explicitly.

**No frozen strict-binding obstruction remains.** The still-open matters
are sharpness of the sufficient mass threshold, all-positive-mass binding,
uniqueness or symmetry of whole-space minimizers, their relation to the
length-42 periodic D4 root, and physical normalization or particle-mass
identification. The canonical S004 obstruction is unaffected. This work
does not assert stability of an arbitrary stationary branch or attraction.
