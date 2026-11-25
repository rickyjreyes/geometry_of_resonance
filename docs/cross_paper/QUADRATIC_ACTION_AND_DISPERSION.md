# Corrected action: complete quadratic expansion and branches

New deduction, 2026-10-07. Primary input: S004 §§1–4, pinned in the preceding audit. No PFF coefficient or observed frequency is an input. Source identifiers are resolved in `audit/cross_paper/SOURCE_LEDGER.json` and `CROSS_PAPER_DERIVATION_LEDGER.md`.

## Conventions and scope

Use the signature (-,+,+,+) required by S004's projector h=g+nn with n²=-1, a fixed flat inertial foliation, and τ=c₀t. Thus □=-∂τ²+Δ. Write

\[
u=|\psi|^2,\quad D=u+\varepsilon^2e^{-2\alpha u},\quad F=uD^{-2},
\quad H=\kappa|Q|^2+\theta|P|^2+\gamma\Re(Q\bar P),\quad Q=\Box\psi,\ P=\Delta\psi.
\]

The literal action is L=∂ψ̄∂ψ−V(u)+FH. Its ordinary time kinetic term is negative in this signature. Orient the ordinary mode positively by using S_phys=−S throughout; this changes neither the equation nor pole locations and does **not** remove relative negative-residue modes. The physical potential is U=−V in this convention. A different signature must transform the entire action consistently.

If [ψ]=A and [x]=[τ]=L, then [u]=[D]=A², [ε]=A, [α]=A⁻², [F]=A⁻², [κ,θ,γ]=A²L², [V]=A²L⁻². The integral may have an additional overall action normalization. Require κ>0, θ>0, γ²<4κθ for the stated curvature form. V is otherwise unspecified in S004.

## Arbitrary stationary background: no lost derivative terms

For any on-shell ψ₀(x,τ), let ψ=ψ₀+η, u₁=2Re(ψ̄₀η), u₂=|η|². Define H₀=H[ψ₀], H₂=H[η] and

\[
H_1=2\Re\{\kappa\bar Q_0\Box\eta+\theta\bar P_0\Delta\eta
+\tfrac\gamma2(\bar Q_0\Delta\eta+\bar P_0\Box\eta)\}.
\]

The exact quadratic coefficient of the original action is

\[
\boxed{\mathcal L_2=\partial\bar\eta\partial\eta-V'u_2-\tfrac12V''u_1^2
+FH_2+F'u_1H_1+(F'u_2+\tfrac12F''u_1^2)H_0.}\tag{Q1}
\]

All scalar functions are evaluated at u₀. This retains amplitude, phase, their mixing, spatial fourth derivatives, mixed space/time derivatives, and fourth time derivatives. The expansion error is O(a³) under η→aη for bounded smooth field jets and C³ weights; it is not a derivative truncation. At zeros of ψ₀ the Cartesian formula remains valid, while polar coordinates fail.

Useful explicit derivatives are F′=D⁻²−2uD′D⁻³ and F″=−4D′D⁻³−2uD″D⁻³+6u(D′)²D⁻⁴, with D′=1−2αε²e⁻²αu and D″=4α²ε²e⁻²αu.

For inhomogeneous backgrounds the Fourier representation is a kernel M(ω,k;ω′,k′). For a time-stationary background only ω is diagonal. A single M(ω,k) is justified only for translation invariance, possibly in a co-rotating frame. The expression below is the general homogeneous plane-wave family, not a claim that every stationary field is homogeneous.

## Boundary terms

Let A_Q=F(κQ+γP/2), A_P=F(θP+γQ/2). The boundary contribution to the original first variation contains, plus its complex conjugate,

\[
\int_{\partial M}\!\left[n_\mu\partial^\mu\psi\,\delta\bar\psi
+A_Q n_\mu\partial^\mu\delta\bar\psi-(n_\mu\partial^\mu A_Q)\delta\bar\psi
+A_P n_i\partial_i\delta\bar\psi-(n_i\partial_i A_P)\delta\bar\psi\right].\tag{Q2}
\]

The quadratic boundary form follows by linearizing these coefficients; none is set to zero before declaring compact support, adequate decay, periodic mathematical test data, or both field and normal-derivative variations fixed. A periodic test domain supplies no physical cavity. Initial/final data for a fourth-time-order equation require more than ψ and ψ̇. In curved space with nonparallel h, use the formal adjoint Δ_h†f=∇μ∇ν(h^{μν}f); replacing it by Δ_h is an extra restriction. The following calculation avoids that restriction by taking fixed flat h.

## Homogeneous plane-wave background and full matrix

Set ψ₀=A₀ exp(−iΩτ+iK·x), u=A₀², d=Ω²−|K|², p₀=−|K|², h₀=κd²+θp₀²+γdp₀. The background equation is

\[
V'=-d+h_0(F+uF').\tag{Q3}
\]

Use ψ=e^{−iΩτ+iK·x}(A₀+r+iv), with real r,v and v=A₀ϑ at linear order in the phase ϑ. Cartesian and polar Hessians agree on shell. Put z=ω², q=|k|², ℓ=Ωω−K·k and

\[
\mathsf Q=\begin{pmatrix}z-q+d&2i\ell\\-2i\ell&z-q+d\end{pmatrix},\quad
\mathsf P=\begin{pmatrix}-q-|K|^2&-2iK\cdot k\\2iK\cdot k&-q-|K|^2\end{pmatrix},\quad E=\operatorname{diag}(1,0),
\]
\[
\mathsf J=(\kappa d+\gamma p_0/2)\mathsf Q+(\theta p_0+\gamma d/2)\mathsf P.
\]

Writing S²_orig=½∫ξ†M_origξ, ξ=(r,v), gives M_orig=2𝕂 with

\[
\boxed{\mathsf K=-\mathsf Q-V'I-2uV''E
+F[\kappa\mathsf Q^2+\theta\mathsf P^2+\tfrac\gamma2\{\mathsf Q,\mathsf P\}]
+2uF'\{E,\mathsf J\}+uh_0F'I+2u^2h_0F''E.}\tag{Q4}
\]

M_phys=−2𝕂 and det M=4 det𝕂. All four entries were independently obtained by differentiating the real-field jet expansion, rather than checking an already assumed matrix against itself. The verification uses a one-dimensional carrier; rotational invariance extends Jk to K·k and J²,k² to their vector norms.

For an explicit determinant for any carrier, write 𝕂=[[A,B],[B*,C]] from (Q4): det M=4(AC−|B|²). At K≠0 the Doppler terms generally make this a degree-eight polynomial in ω, rather than a polynomial in ω². No root is omitted.

## Rotating homogeneous family: amplitude–phase mixing

For K=0, put X=z−q+Ω² and

\[
B=-X-V'+F[\kappa(X^2+4\Omega^2z)+\theta q^2-\gamma qX]+u\kappa\Omega^4F',
\]
\[
A=B-2uV''+4uF'(\kappa\Omega^2X-\gamma\Omega^2q/2)+2u^2\kappa\Omega^4F'',
\]
\[
C=-2+4F\kappa X-2F\gamma q+4uF'\kappa\Omega^2.
\]
\[
\boxed{\mathsf K=\begin{pmatrix}A&i\Omega\omega C\\-i\Omega\omega C&B\end{pmatrix},
\qquad\det M=4(AB-\Omega^2zC^2)=0.}\tag{Q5}
\]

This is quartic in z, with leading coefficient 4(Fκ)². At q=0 the on-shell condition supplies the Goldstone root z=0; the other three roots remain. These roots depend on Ω,V′′,F′,F″ and the couplings. Realness and residues must be checked for a specified rotating background; they are not guaranteed by curvature-form positivity. The higher-time Hessian has magnitude 2Fκ I and is nondegenerate for A₀≠0,κ>0, so there is no constraint here that silently removes the additional modes.

## Static homogeneous family: all poles explicitly

Set Ω=K=0 and V′(u)=0. Define K_c=Fκ>0, G_c=Fγ, T_c=Fθ>0, G_c²<4K_cT_c. Below abbreviate them as K,G,T; they are unrelated to the carrier vector. Let m²=−2uV′′ for the amplitude mode and m²=0 for the phase mode. Then each positive-ordinary-kinetic block, after ξ=√2(r or v), is

\[
\mathcal D(z,q)=y-m^2-Ky^2+Gqy-Tq^2,\quad y=z-q,\quad
S_2=\tfrac12\int\xi^*\mathcal D\xi.\tag{Q6}
\]
\[
\boxed{z_\pm(q)=q+\frac{1+Gq\pm\sqrt{(1+Gq)^2-4K(m^2+Tq^2)}}{2K}.}\tag{Q7}
\]

For each real nondegenerate root, ∂z𝒟(z₋)=+√disc and ∂z𝒟(z₊)=−√disc. Thus the lower pole has positive ordinary residue and the upper pole is ghostlike. There are two such pairs (amplitude and phase). If disc<0 the poles are complex and must be recorded as a high-momentum instability of the untruncated model. At disc=0 the low-energy reduction fails. At q=0 the amplitude pair is real and separated if 0<4Km²<1. The phase pair is 0 and 1/K. Degenerate κ=0 and zero-background limits change differential order and cannot be obtained by dropping a pole at finite κ.

These equations derive a restricted low-energy branch, not a healthy fundamental completion. The existing black-hole bridge P19 §5 independently recognizes the same higher-time ghost issue via an auxiliary field. No ghost is removed from the source action in this audit.

Status: `DERIVED_BY_CROSS_PAPER_COMPOSITION` for the exact new expansion using S004; the identity itself is exact. Falsifier: disagreement with a direct Hessian/linearized EL calculation under these conventions. See the reproducible symbolic checks.
