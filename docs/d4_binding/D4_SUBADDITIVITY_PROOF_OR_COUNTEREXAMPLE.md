# Frozen D4 strict-subadditivity proof

**D4_STRICT_BINDING_PROVED.** Date: 2026-10-23.
Notation and lemmas R1–V4 are in
[D4_ENERGY_PER_MASS_ANALYSIS.md](D4_ENERGY_PER_MASS_ANALYSIS.md).

## S0. Ordinary subadditivity independently

For a,b>0 choose near-minimizers of those masses. Approximate them in H² by
compact smooth complex profiles. Normalize each by the square root of the
prescribed-to-actual mass ratio. The factors tend to one and H is continuous,
so the energy errors tend to zero. Translate the second compact profile until
the supports have positive separation. Derivatives through order two remain
supported inside each support. Mass, T, Q and S then add EXACTLY. Translation
invariance preserves all single-profile quantities, and A is local. There
is no nonlocal interaction term. Let the approximation errors tend to zero:

    I(a+b)<=I(a)+I(b).                            (S0)

For untruncated H² profiles the translated cross terms instead tend to zero
by approximation; the mass is a+b+o(1). Amplitude normalization tending to
one restores a+b and changes energy by o(1) on the bounded H² sequence.
The compact-approximation argument avoids that final correction altogether.

Thus a reversed strict inequality is excluded by the unchanged functional.
A counterexample to binding must prove an interior equality, requiring a
matching lower bound or equivalent complete argument. Separated trials alone
establish only ordinary subadditivity.

## S1. Exact inherited coverage

PR #35's condition `m+max(mu,m-mu)<=2m_*` certifies, at m=m_D4, exactly

    [2m-2m_*,2m_*-m]
      =[0.5630113309687108...,1.9816855638884092...].

Here m and m_* are the exact rationals in R1, not rounded endpoints. Symmetry
reduces the old open region to `0<mu<2m-2m_*`. This was an uncontrolled
sufficient estimate, not demonstrated dichotomy or an equality split.

## S2. All interior splits

**Theorem.** For positive beta,gamma,k in two dimensions, every 0<m<=m_B,
where `m_B=3beta/[gamma sqrt(k⁴+3beta²/(32gamma))]`, satisfies

    I(m)<I(mu)+I(m-mu) for every 0<mu<m.

Proof: V3 gives e(mu)>e(m) and e(m-mu)>e(m). Both weights are positive:

    I(mu)+I(m-mu)=mu e(mu)+(m-mu)e(m-mu)>m e(m)=I(m).

V3 was proved on arbitrary adjusted near-minimizing sequences, with a
sequence-uniform defect, and uses no minimizer at any component mass.
V4 proves the exact frozen norm lies in this range. QED.

Specific larger-mass splits also follow when
`m+max(mu,m-mu)<2m_B`; no such extension is needed for D4. No universal
all-mass result or equality counterexample beyond this range is asserted.

## F. An explicit lower bound through the entire endpoint interval

Let m=m_D4, 0<mu<=m/2, b=m-mu, h=1-(m+b)/(2m_B). Use the admissible
Gaussian of R1 with a=10 and transverse width 1, at mass s. Its optimized
envelope scale gives I(s)<=U(s), where

    c2(s)=s/50-beta*s²/(40pi),
    c4(s)=30203*s/40000+gamma*s³/(900pi²),
    U(s)=-c2(s)²/(4c4(s)).

For s in [m/2,m], c2<0 and c4>0: `beta*m/2>4pi/5`, verified already using
pi<22/7. Thus U<0 throughout this interval. From e(mu)>=e(b) and V3,

    Delta_m(mu)>=F(mu)
       :=m*mu/b² * [1-(m+b)/(2m_B)] * [-U(b)]>0.   (F)

Reflection covers m/2<mu<m. This domain contains the entire formerly open
small-component interval and overlaps the inherited central certificate;
there is no uncertified junction. The analytical sign argument covers a
continuous interval; a finite numerical grid is not its proof.

As mu->0,

    F(mu)=[(1-m/m_B)(-U(m))/m]mu+O(mu²),

with positive coefficient approximately 2.8882216659e-6. Strictness need not
be uniform as the component mass tends to zero.

There is no remaining strict-binding lemma at the frozen norm. Identification
with the periodic D4 root, uniqueness, physical spectrum and the S004 parent
obstruction are separate questions not settled by this theorem.
