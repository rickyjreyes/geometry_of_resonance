# Finite scale: derived and parameter dependent

Use Q7 and E1. The amplitude branch has a negative initial q-slope precisely when

\[
G C_0>s,\qquad s=\sqrt{1-4Km^2}.
\tag{F1}
\]

This requires positive G and is compatible with κ>0,θ>0,γ²<4κθ. It does not hold for every allowed S004 parameter set. It never holds for the static homogeneous phase branch, whose initial slope is 1.

## Exact finite minimum

Let B=4KT−G²>0 and a₀=2K+G. In the F1 regime a₀>0. Q7 becomes z₋=(1+a₀q−√d)/(2K), d=s²+2Gq−Bq². On the real-pole interval,

\[
z_-'=\frac{a_0-(G-Bq)/\sqrt d}{2K},\qquad
z_-''=\frac{Bs^2+G^2}{2K d^{3/2}}>0.
\]

Since z₋′(0)<0 and z₋′(G/B)=a₀/(2K)>0, there is one interior minimum. Solving with the unsquared sign condition G−Bq>0 yields

\[
\boxed{q_{\min}=\frac{G-a_0\sqrt{(G^2+Bs^2)/(a_0^2+B)}}{B}>0.}\tag{F2}
\]

The upper real-pole endpoint is q_collision=(G+√(G²+Bs²))/B. The minimum is strictly below it. Thus the finite minimum is not an artifact of truncating at k⁴. At the minimum both real z roots are positive; this is an oscillatory frequency minimum, not an unstable growth band.

## Algebraic witness, not a fit or physical prediction

In units M²=1 choose δ=1/1000,g=2,h=2, giving K=1/1000,G=2,T=2000. This exact rational choice only demonstrates a nonempty parameter regime:

| Quantity | Value |
|---|---:|
| q_min, exact | 0.00025068759420757693 |
| q_star, k⁴ approximation | 0.00025049849448445889 |
| Relative difference | 0.00075432421662419 |
| z_low at the exact minimum | 1.0008759383528104 |
| z_ghost at that same q | 999.501000625251 |

One admissible potential realizing m²>0 is V(u)=−λ(u−u₀)²/2 with λ>0 and m²=2λu₀. This is an allowed example within S004's unspecified V, **not** a potential or vacuum uniquely selected by the corpus. Since F(u₀)>0, arbitrary positive K,T and admissible G can be realized by κ=K/F,θ=T/F,γ=G/F.

## What is and is not predicted

The classification is `DERIVED_PARAMETER_DEPENDENT_SCALE`. Holding δ,g,h fixed while varying M rescales k_min by M; choosing a different background or dimensionless coupling ratios moves it further. No searched source independently freezes the required S004 values. The constants papers use other ansätze, dimensional calibrations, or simulation inputs, and do not supply a matching determination of these coefficients.

The PFF band center and the Geometry EFT minimum were already analytic results within their own rails. This audit adds a coefficient map and an exact minimum of a positive-residue S004 amplitude branch in a controlled window. It does not select a radius, winding number, spectral width, or log frequency.

The high ghost and complex-pole region remain in the untruncated theory. The result is not `UNPHYSICAL_BRANCH` merely because another branch is ghostlike: the identified pole has positive residue and a separated EFT window. Equally, it is not a proof that the complete action is healthy. Falsification conditions include failure of F1, pole collision inside the selected window, or lack of an independently justified EFT cutoff when asserting a physical application.
