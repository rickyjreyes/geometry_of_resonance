# Corrected action to the Geometry EFT kernel

This is a **new deduction from existing ingredients**, not a derivation already written in Geometry. Input: S004, quadratic equations Q6–Q7. Target: Geometry P01 Appendix Y.1, Eq. (504), pp. 310–311:

\[
\mathcal L_0=\tfrac12|\partial_t\psi|^2-\tfrac{c_{\rm EFT}^2}{2}\psi^*(\Delta+k_\star^2)^2\psi-\tfrac{m_0^2}{2}|\psi|^2.
\]

## Exact coefficient map on one physical branch

Take a nonzero, static homogeneous on-shell background with V′=0, m²=−2uV′′>0 and 0<4Km²<1. Set s=√(1−4Km²). Expanding the lower pole in q=k² gives

\[
\boxed{C_0=\frac{1-s}{2K},\qquad C_2=1-\frac{G C_0}{s},\qquad
C_4=\frac{T}{s}+\frac{G^2m^2}{s^3}>0,}\tag{E1}
\]
\[
z_-(q)=C_0+C_2q+C_4q^2+O(q^3).
\]

Hence, if C₂<0,

\[
q_\star=k_\star^2=-\frac{C_2}{2C_4},\quad
\Delta_0=C_0-\frac{C_2^2}{4C_4},\quad
z_-(q)=C_4(q-q_\star)^2+\Delta_0+O(q^3).\tag{E2}
\]

Restore physical time using ω_phys=c₀ω. The target coefficients are

\[
\boxed{c_{\rm EFT}^2=c_0^2C_4,\qquad m_0^2=c_0^2\Delta_0,\qquad k_\star^2=q_\star.}\tag{E3}
\]

[C₀]=L⁻², [C₂]=1, [C₄]=L²; consequently [c_EFT]=L²/T and [m₀]=T⁻¹. The coefficient called c in this fourth-spatial-order EFT cannot dimensionally be identified with light speed without an additional length normalization.

## Action-level reduction, not just a matching polynomial

Exactly,

\[
\mathcal D=-K(z-z_-)(z-z_+)=Z(z,q)(z-z_-),\quad Z=K(z_+-z).
\]

Restrict spectral support to a real-pole window with z well below z₊ and disc uniformly positive. For ξ=√2r, perform the analytic, self-adjoint Fourier-multiplier redefinition ξ=Z⁻¹ᐟ²χ. Then S²=½∫χ*(z−z₋)χ. Taylor expanding z₋ gives the local quadratic target in E3, to the stated error. This transformation is nonlocal before expansion, depends on the selected spectral sector and fails at the high pole. It is **not** a global invertible transformation removing the ghost. Observables must transform with ξ; their residues are not all unity in the original field.

Only **one real amplitude component** has this finite-k kernel. The phase component has C₀=0,C₂=1,C₄=T and no such small-q minimum. Thus the full two-real-component S004 theory is not shown equivalent to the full complex Geometry EFT with two degenerate copies. Adding a second matching mode would require another argument.

The original static stiffness is m²+q+(K+G+T)q² and increases with q in this regime. Its negative-gradient counterpart in E2 arises after frequency-dependent branch normalization; it is not the static energy of the original field. This distinction prevents a false inference of static pattern formation from a dispersion minimum.

## Controlled regime and explicit remainder

Let m²=M², K=δ/M², G=g/M², T=h/(δM²), where 0<δ≪1, g>1 and h>g²/4. For q=δM²Q on a fixed bounded interval of Q≥0,

\[
\frac{z_-}{M^2}=1+\delta[1+(1-g)Q+hQ^2]+O(\delta^2),\qquad
q_{\min}=\delta M^2\frac{g-1}{2h}+O(\delta^2M^2).\tag{E4}
\]

The low frequency is O(M), the ghost frequency O(M/√δ), and the discriminant remains uniformly positive. This hierarchy permits a cutoff between them. The claim is classical quadratic EFT control; it does not prove quantum or nonlinear UV consistency.

More explicitly set B=4KT−G²>0, d(q)=s²+2Gq−Bq² and N=Bs²+G². Then

\[
z_-'''(q)=-\frac{3N(G-Bq)}{2K\,d(q)^{5/2}}.
\]

On 0≤q≤q_max with d≥d_min>0, the E1 remainder has magnitude at most N sup|G−Bq| q³/(4K d_min^{5/2}). In E4 this is O(δ²M²) uniformly on bounded Q. The expansion is not extended to the pole collision or arbitrary large k.

## Nonlinear vertices: why the next match is not automatic

On this static background let u₁=2A₀r, u₂=r²+v² and H₂=H[r+iv]. The original action has the following exact cubic and quartic coefficients (no slow-envelope assumption):

\[
\mathcal L_3=-V''u_1u_2-\frac{V'''}6u_1^3+F'u_1H_2,
\]
\[
\mathcal L_4=-\frac{V''}2u_2^2-\frac{V'''}2u_1^2u_2-\frac{V''''}{24}u_1^4
+(F'u_2+\tfrac12F''u_1^2)H_2.\tag{E5}
\]

S_phys changes both signs together. E5 follows by direct expansion and has O(amplitude⁵) remainder when the required derivatives are bounded. It shows explicitly the derivative interactions that a potential-only cubic saturation model omits. The quadratic normalization must be extended to these vertices; elimination of a high mode also induces exchange terms. The phase mode is gapless and cannot generically be integrated out as a gapped field. A purely real zero-charge subspace is invariant under the real equations, but then S004's U(1) charge is zero, rather than its positive L² norm.

In that real subspace, the physical potential coefficients before nonlinear normalization are

\[
U_3=-[2A_0V''+(4/3)A_0^3V''']r^3,\quad
U_4=-[V''/2+2uV'''+(2/3)u^2V'''']r^4.
\]

V‴ and V⁗ can be varied while preserving all of E1 and F2. Thus the verified dispersion does not determine even the local nonlinear saturation coefficient. This is a concrete remaining physical freedom in the supplied action, not simply an unwritten algebraic step. A selected potential and state, or an independently derived matching condition, is needed before a specific nonlinear localized branch can be asserted.

Status: `CONTROLLED_EFFECTIVE_REDUCTION`, gap class `CROSS_PAPER_REDUCTION_REQUIRED` now closed **for this component and regime**. Remaining parameters: u₀,ε,α,κ,θ,γ,V′′ and c₀/unit normalization; nonlinear closure additionally needs higher derivatives of V and preparation. Falsifier: the direct lower-pole expansion or transformed quadratic action fails to give E1–E3, or the claimed working window reaches disc≤0 or the high pole.
