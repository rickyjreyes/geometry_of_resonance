# Whole-space compactness

**Precompactness modulo translations: PROVED.**
**Whole-space constrained minimizer existence: PROVED.**
Date: 2026-10-23. These statements hold for every 0<m<=m_B, including m_D4.

## C1. Boundedness and nonvanishing

For any minimizing sequence M(u_n)=m, H(u_n)->I(m), R1 gives bounded H².
If `sup_y integral_(B(y,1))|u_n|²->0`, the local vanishing estimate gives
Q(u_n)->0. Cover by enlarged unit cubes with bounded overlap; the local
H¹/L² interpolation estimate bounds the sum of fourth powers by a constant
times the supremum local L² mass times the bounded global H¹ norm squared.
The uniform H²-to-L-infinity bound then gives S<=||u_n||_infinity² Q->0.
Since T>=0 this contradicts I(m)<0.

Choose translations retaining positive local mass. A subsequence converges
weakly in H², strongly in local finite Lp and almost everywhere to nonzero u
by local Rellich compactness. Set a=M(u), so 0<a<=m.

## C2. No positive-mass remainder

For v_n=u_n(.+y_n)-u, Hilbert splitting yields

    M(v_n)=m-a+o(1), T(u_n)=T(u)+T(v_n)+o(1).

The Brezis–Lieb lemma applies separately at p=4 and p=6 to the bounded,
almost-everywhere convergent complex sequence. Hence Q and S split, and

    H(u_n)=H(u)+H(v_n)+o(1).

We do not assume weak lower semicontinuity of H with its focusing term.
If a<m, normalize v_n to mass m-a by a factor tending to one. Uniform H²
bounds and the amplitude polynomial make its energy change o(1). Then

    I(m)>=H(u)+I(m-a)>=I(a)+I(m-a),

contradicting all-split strict binding. Thus a=m. This excludes dichotomy,
diffuse positive-mass loss and every other positive norm remainder.

## C3. Strong H² convergence

Now v_n->0 in L², while bounded sup norms force Q(v_n),S(v_n)->0. Energy
splitting, H(u)>=I(m), and T(v_n)>=0 force H(u)=I(m) and T(v_n)->0.
Therefore

    ||Delta v_n||<=||A v_n||+k²||v_n||->0,
    ||grad v_n||²<=||v_n|| ||Delta v_n||->0.

The convergence is strong H². Every minimizing sequence has a strongly
convergent subsequence after translations. In particular

    G_m={u in H²(R²;C): M(u)=m,H(u)=I(m)}

is nonempty and compact modulo translations. Translation escape is handled
by those translations, not excluded in the original coordinates. Phase
rotations are included in G_m and U(1) is compact. A high-frequency
zero-mass remainder cannot retain positive squared-A energy. No finite box,
symmetry restriction or altered energy was used.

## Scope

The formerly missing premise of PR #35's compactness theorem is now verified
at m_D4. Equality would permit a separated minimizing sequence; strictness
excludes it. This does not identify G_m with the fixed-torus D4 root, prove
uniqueness or a D4 symmetry, or supply a physical particle spectrum.
