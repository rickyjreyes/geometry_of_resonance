# Geometric scale and operator audit

**Result: exact 2–3–6 geometry is verified; no reviewed derivation makes one of its gains a physical log-spectral dilation eigenvalue.** The October 6 manuscript P05 already distinguishes these claims. No empirical frequencies are inputs here.

## Exact finite-dimensional statements

Let `u=(1,1,1)`, `c_j(phi)=cos(phi+2pi j/3)` for j=0,1,2, and `r=a u+b c`, with a>0. Then

`u·c=0`, `||u||^2=3`, `||c||^2=3/2`,

`Q=||r||^2/(sum r_j)^2=1/3+b^2/(6a^2)`.

Imposing Q=2/3 therefore forces b^2/a^2=2 and gives `||r||^2=6a^2`. These are identities of the parametrization. They do not independently predict Q or identify r as physical mass eigenvalues. Interpreting r_j as positive square roots additionally restricts the allowed phase/branch.

With `mu_i=e_i-u/3`, the defining A2 weights satisfy `||mu_i||^2=2/3`. At the three distinguished phases, the cosine vectors are `(3/2)mu_i` up to ordering. Their six directed weight differences are A2 roots of squared length 2. The Cartan matrix `[[2,-1],[-1,2]]` has eigenvalues 1 and 3. These are actual eigenvalues of a specified mathematical matrix, not evidence that the physical WCT time evolution or mass map contains the same spectrum.

For a fixed phase and positive balance branch, the paper's composed lift is

`T:R -> R^3`, `T(a)=a[u+sqrt(2)c]`.

Thus `T^dagger T=6`, and T has singular value sqrt(6). T is rectangular: it does not have an eigenvector equation `Tv=lambda v` on one physical state space. Repeating T is not even defined without an additional return map. Choosing such a map to obtain a preferred eigenvalue would add the missing dynamical assumption.

## Classification of the proposed scales

| Quantity | Exact origin in the stated normalization | Classification of that statement | Status as a physical scale |
|---|---|---|---|
| sqrt(2) | Positive coefficient ratio b/a at imposed Q=2/3; also conventional root length | Exact mathematical identity | A physical balance or dilation needs a mass-operator theorem. |
| sqrt(3) | Norm of u; norm of each balanced sector divided by a | Exact mathematical identity | Normalization gain, not an independently derived physical eigenvalue. |
| sqrt(3/2) | Norm of the Z3 cosine vector | Exact mathematical identity | Changing vector normalization changes this norm; a measurement calibration is required. |
| sqrt(6) | Balanced root-mass norm/a and singular value of T | Exact mathematical identity | Conditional physical interpretation only if an independent operator acts by this gain on a declared coordinate. |
| 2 | Coefficient-power ratio at Q=2/3; A2 rank and root squared length are separate occurrences | Exact mathematical identities | Equating the separate occurrences physically is a structural analogy without an intertwining map. |
| 3 | Democratic sector power/a^2, number of defining weights, one Cartan eigenvalue | Exact mathematical identities | No derived WCT dilation or stability selection follows. |
| 6 | Total balanced power/a^2, six roots, eigenvalue of T^dagger T | Exact mathematical identities | Conditional squared-coordinate consequence of a root-mass gain; no autonomous mass scaling established. |

The exact A2 weight-space embedding is stronger than a visual resemblance. Its interpretation as physical SU(3) gauge dynamics, color confinement, or a selected braid remains a **structural analogy** until fields, transformations, action and observable correspondence are supplied. The paper's historical nearest-scale comparisons are **retrospective numerical coincidences**, not independent physical derivations; this audit neither recomputes nor selects among them. None of these audited connections reaches **Derived physical relation** for active-domain dilation.

## Root-mass and mass coordinates

If an independently defined operation did give `r_j -> lambda_r r_j`, then `m_j=r_j^2 -> lambda_r^2 m_j`. In particular, lambda_r=sqrt(6) gives lambda_m=6.

Use dimensionless log ratios `ell_r=ln(r/r0)` and `ell_m=ln(m/m0)=2 ell_r` with m0=r0^2. The same oscillation has `k_m=k_r/2`, while its interval width doubles. Hence kL is unchanged. An assumed fundamental one-cycle dilation would give `k_r=2pi/ln(sqrt(6))` and `k_m=2pi/ln(6)`; these are coordinate representations of one conditional statement, not two predictions. Harmonic order, boundary winding and the applicable coordinate must still be fixed independently.

## Thin-shell geometry does not fill the operator gap

P06 distinguishes the toroidal shape–winding quantity `K^2=eta^2 m^2` from the direct separated angular Laplacian cost, which scales as `n^2/R^2+m^2/a^2`. Substituting one for the other changes the physical operator.

For the torus-of-revolution Willmore functional, the shape minimum eta^2=1/2 is exact. The full WCT claim additionally requires a controlled thin-shell reduction with its profile, normalization, cross terms and corrections, selection of m=1, and a mass-operator relation `cos^2(theta_Koide)=K^2`. Those are separate assumptions. Scale invariance leaves the overall radius arbitrary even after selecting eta. See Proposition D4 in the domain report.

The cubic invariant in P05,

`C3=sqrt(6) sum_j(r_j-a)^3/[sum_j(r_j-a)^2]^(3/2)`,

is permutation invariant and, on the b>0 cosine convention, equals cos(3phi). It is undefined at the democratic state b=0. For b<0 the same expression includes sign(b), or one must absorb that sign into the phase convention. This invariant reconstructs orientation given the other triplet invariants; it does not select the charged-lepton orientation. The proposed phase 2/9 is an explicitly empirical-motivated conjectural target in P05 and is not used as an input to this audit.

## Minimum missing physical theorem

One needs an independently specified WCT operator D, its physical state space and boundary conditions, a controlled map from its state to the measured coordinate x, and a theorem that the selected state obeys `x -> lambda x` with a declared lambda. A norm ratio, a rectangular singular value, a dimension count, or a shape-only minimizer supplies none of these alone. The current audit stops at this gap; it does not propose another geometric matching scheme.

Verification: V20 and V24–V30 in [symbolic_results.json](../audit/physical_domain/symbolic_results.json). Source identities and revisions: P05, P06, S003, S004, S028–S030 in [SOURCE_LEDGER.json](../audit/physical_domain/SOURCE_LEDGER.json).
