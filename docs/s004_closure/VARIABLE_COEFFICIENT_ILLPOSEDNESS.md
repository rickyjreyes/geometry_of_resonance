# Variable-coefficient Sobolev ill-posedness of the linearized corrected S004 equations

**Phase XXXIII (issues #53, #16).** This document proves that the linearization of the corrected
S004 Euler–Lagrange equations has an ill-posed Cauchy problem in every pair of Sobolev spaces near
every regular point (`ψ₀ ≠ 0`) of every smooth background. The background may be constant,
nonconstant, time-dependent, or have nodes elsewhere, and it need not solve the field equations.

The proof is self-contained. It does **not** cite the Lax–Mizohata theorem. It uses only Taylor's
formula, the Fourier characterization of Sobolev norms, the Leibniz rule and explicit Gaussian
integrals. The proof uses the same mechanism as Lax (1957) and Mizohata (1961): a localized
high-frequency solution concentrated on a non-real simple characteristic root. Here that
construction is written out with explicit phase, amplitudes, residual and norm estimates for this
operator.

Status labels:

* **LEAN VERIFIED**: kernel-checked in `lean/ReleaseProofs`.
* **ANALYTICALLY PROVED**: complete written proof in this document.
* **NUMERICALLY VERIFIED**: reproduced by `derivations/s004_illposedness_verify.py`.
* **CONDITIONAL**, **REFUTED**, **OPEN**: as stated.

## 0. Results

| # | Statement | Status |
|---|---|---|
| L1 | Structure of the linearization: `L_{ψ₀} = 2F(ρ₀)P(∂)I₂ + Σ_{|α|≤3} A_α ∂^α`, all `d`, any smooth `ψ₀`. The coefficient of `∂^α` involves at most `4−|α|` derivatives of `ψ₀`, at most `F‴` and at most `V″`. | ANALYTICALLY PROVED (§2). Symbolic: d = 1, 2 generic background (C1, I1); three explicit nonconstant backgrounds, exactly (I7). |
| L2 | Admissibility ⟺ ellipticity. Exact factorization `P = κ(λ²+μZ)(λ²+μ̄Z)`. Four distinct simple roots `±w|ξ|, ±w̄|ξ|`, with `Re w > 0` and `∂_λP = 2i w |ξ|³ √(4κθ−γ²) ≠ 0`. | LEAN VERIFIED (`symbolForm_pos_definite_iff`, `laplaceSymbol_factor`, `laplaceSymbolDeriv_at_hadamard_root`, `hadamard_root_simple`, `exists_growing_simple_root`). Symbolic I2. |
| L3 | Exact complex Gaussian-beam phase `φ = iê·x − |x|²/2 + w t S(x) + μt²/2`, `S = (1+2iê·x−|x|²)^{1/2}`. It solves the eikonal equation `P(∇φ) = 0` **exactly** in every dimension. | ANALYTICALLY PROVED (§5.2). Symbolic I4 (d = 1, 2, 3). LEAN VERIFIED for d = 1 (`beam_eikonal`, `beam_transport`). |
| **T1** | **Local Sobolev ill-posedness for every operator `a P(∂)I_n + (order ≤ 3)` with smooth coefficients and `a(0,0) ≠ 0`; hence for `L_{ψ₀}` at every regular point of every smooth background.** The local Cauchy estimate fails for every `s, σ ∈ ℝ`, every `T > 0`, every ball and every constant. | **ANALYTICALLY PROVED** (§4–§5). Lean: bookkeeping and 1D ingredients (§11). NUMERICALLY VERIFIED end to end on three backgrounds (I7). |
| C1 | No bounded linear solution operator with sources, `H^s → H^{−σ}`, that reproduces smooth solutions. | ANALYTICALLY PROVED (§6). |
| T2 | No `C₀`-semigroup for time-independent linearizations (static or co-rotating states). No evolution system with the backward equation on smooth data for time-dependent ones. Both on any data space comparable to Sobolev norms near the point. | ANALYTICALLY PROVED (§6). The evolution-system version assumes the stated generator hypothesis (S1′). |
| T3 | Class A (constant or co-rotating homogeneous background, on- or off-shell): exact Fourier solutions grow like `e^{c|ξ|t}` at fixed `t`. The homogeneous problem is unbounded `H^s → H^{−σ}` for every fixed `t > 0`. | ANALYTICALLY PROVED (§6, Rouché). NUMERICALLY VERIFIED (I6). |
| T4 | No forced nonlinear solution map Fréchet-differentiable at a smooth solution with a regular point. No unforced `C¹` flows from every initial time with a derivative family satisfying (S1′). | ANALYTICALLY PROVED (§8). |
| — | Orbital stability, nonlinear norm inflation, blow-up, Gevrey/analytic-class statements | OPEN (§8, §12). **Not claimed.** |

## 1. Setting

**Density.** The density is the corrected real complex-field density of
`WCT_CORRECTED_VARIATIONAL_CLOSURE_DRAFT.md` §2, on flat `ℝ^{1+d}` with signature `(−,+,…,+)` and
the inertial foliation (`Δ_h = Δ`):

```
L = ∇ψ̄·∇ψ − V(u) + F(u) H,   u = |ψ|²,   F(u) = u / (u + ε² e^{−2αu})²,
H = κ|□ψ|² + θ|Δψ|² + γ Re(□ψ · conj(Δψ)),   □ = −∂ₜ² + Δ.
```

* Parameters: `ε ≠ 0`, `α ∈ ℝ`, `V ∈ C^∞(ℝ)` arbitrary.
* Admissibility: `κ > 0, θ > 0, γ² < 4κθ`.
* `F` is real-analytic on `(0,∞)` with `F > 0` there, and `F(0) = 0`.
* The overall sign of the density (`L_phys = −L_orig`) does not affect anything below, because it
  multiplies the Euler–Lagrange operator by `−1`.

**Euler–Lagrange operator.** Vary `ψ̄` with `ψ` held fixed. Then
`∂H/∂(□ψ̄) = κ□ψ + (γ/2)Δψ` and `∂H/∂(Δψ̄) = θΔψ + (γ/2)□ψ`. Integration by parts gives

```
E(ψ) = □[F(u)(κ□ψ + (γ/2)Δψ)] + Δ[F(u)(θΔψ + (γ/2)□ψ)] + F′(u) H ψ − □ψ − V′(u) ψ.     (1.1)
```

* In real components `ψ = ψ₁ + iψ₂`, the Euler–Lagrange system is
  `E_ℝ(ψ) = 2(Re E(ψ), Im E(ψ))`, because `δS = ∫ 2 Re(E δψ̄)`.
* `sympy.calculus.euler.euler_equations` applied to the density reproduces `E_ℝ`
  (C1, I1, I7).

**Background and linearization.**

* `ψ₀ ∈ C^∞(Ω; ℂ)`, with `Ω ⊂ ℝ^{1+d}` open. No field equation is assumed.
* `L_{ψ₀} v := (d/dε) E_ℝ(ψ₀ + εv)|_{ε=0}` for `v: Ω → ℝ²`.
* A **regular point** is a point with `ψ₀(t₀, x₀) ≠ 0`, i.e. `F(ρ₀) > 0` there, with
  `ρ₀ = |ψ₀|²`.

**Sobolev spaces.**

* `H^s = H^s(ℝ^d; ℂ^n)`, `s ∈ ℝ`, with `‖f‖²_{H^s} = ∫ (1+|ξ|²)^s |f̂(ξ)|² dξ`.
* For integers `m ≥ 0`, `‖f‖_{H^m}` is equivalent to `(Σ_{|β|≤m} ‖∂^β f‖²_{L²})^{1/2}`.
* For `s ≤ s′` and `σ ≤ σ′`, `‖·‖_{H^s} ≤ ‖·‖_{H^{s′}}` and `‖·‖_{H^{−σ′}} ≤ ‖·‖_{H^{−σ}}`.
* Duality: `‖f‖²_{L²} ≤ ‖f‖_{H^{−σ}} ‖f‖_{H^{σ}}`.

## 2. Lemma 1 — structure of the linearized operator (all `d`)

**Lemma 1.** For every smooth `ψ₀` and every `d ≥ 1`,

```
L_{ψ₀} = 2F(ρ₀) P(∂) I₂ + Σ_{|α|≤3} A_α(t,x) ∂^α,     P(∂) = κ□² + γ□Δ + θΔ².      (2.1)
```

* `α` runs over multi-indices in `(t, x)`.
* Each `A_α` is a real `2×2` matrix.
* Its entries are polynomials in `∂^β ψ₀` with `|β| ≤ 4 − |α|`. Their coefficients are built from
  `κ, θ, γ`, `F^{(j)}(ρ₀)` (`j ≤ 3`) and `V′(ρ₀)`, `V″(ρ₀)`.
* Hence `A_α ∈ C^q` whenever `ψ₀ ∈ C^{q+4}`.
* Near a regular point the normalized operator
  `L̃ := (2F(ρ₀))^{-1} L_{ψ₀} = P(∂)I₂ + Σ_{|α|≤3} B_α ∂^α` has a **constant-coefficient principal
  part**, independent of the background.

*Proof.* Every term of (1.1) is a sum of products. In each product the total number of derivatives
is `4` (the `F(u)H` part), `2` (kinetic) or `0` (potential). Linearize, i.e. replace one factor
`ψ` or `ψ̄` by `v` or `v̄`, with `δu = 2 Re(ψ̄₀ v)`. A term with `k` derivatives on `v` then carries
`4 − k` derivatives on background quantities: `ψ₀`, `u₀`, or `F^{(j)}(u₀)` through the chain rule.

Four derivatives fall on `v` only in `□[F(κ□ + (γ/2)Δ)]ψ + Δ[F(θΔ + (γ/2)□)]ψ`, with no
derivative on `F`. That part is `F(u₀)(κ□² + γ□Δ + θΔ²)v`, because `□Δ = Δ□`. In real components
it becomes `2F(ρ₀)P(∂)I₂`.

Derivatives of `F(u₀)` produce `F′, F″, F‴`:

* `□[F(u)X]` with `F` varied gives `F′(u₀)δu`, and two further derivatives on it give at most `F‴`.
* `F′(u)Hψ` varied gives `F″`.

The potential gives `V′(u₀)v + 2V″(u₀) Re(ψ̄₀v) ψ₀`. All coefficients are real in real
components. ∎

*Checks.*

* C1: the principal symbol in `1+1` and `1+2` dimensions for an arbitrary background, i.e. generic
  functions `X₁(t,x), X₂(t,x)`.
* I1: orders, total derivative budget `4−|α|`, `F‴`, `V″`, real coefficients, `1+1` and `1+2`, generic background.
* I7: on each of three explicit nonconstant backgrounds, the fourth-order coefficients equal
  `2f₀(κ, −(2κ+γ), κ+θ+γ)·I₂` with zero off-diagonal blocks. This is verified symbolically and
  exactly, where `f₀ = F(ρ₀(t,x))`.

## 3. Lemma 2 — admissibility, characteristic roots, simplicity

For a complex covector `ζ = (ζ₀, ζ′) ∈ ℂ^{1+d}`, write `Z = ζ′·ζ′ = Σ ζ_j²` (bilinear) and
`P(ζ) := e^{−ζ·(t,x)} P(∂) e^{ζ·(t,x)} = κ(Z − ζ₀²)² + γ(Z − ζ₀²)Z + θZ²`.

* With `ζ₀ = λ` and `ζ′ = iξ` this is
  `P(λ, iξ) = κ(λ²+|ξ|²)² + γ(λ²+|ξ|²)|ξ|² + θ|ξ|⁴`.
* With `λ = iτ` (real frequency) it is the curvature form `κX² − γXY + θY²` at
  `X = τ² − |ξ|²`, `Y = |ξ|²`.

**Lemma 2.** Let `κ > 0, θ > 0, γ² < 4κθ`. Put
`μ = −(2κ+γ)/(2κ) + i√(4κθ−γ²)/(2κ)` and `μ̄` its conjugate.

1. **Ellipticity.** `P(iτ, iξ) > 0` for real `(τ, ξ) ≠ 0`. Conversely, positivity of the
   curvature form for all `(X, Y) ≠ 0` is equivalent to admissibility.
2. **Factorization.** `P(ζ) = κ(ζ₀² + μZ)(ζ₀² + μ̄Z)`.
3. **Roots.**
   * `μ` is non-real, so it has a square root `w` with `Re w > 0` and `Im w ≠ 0`.
   * For `ξ ≠ 0` the roots of `λ ↦ P(λ, iξ)` are `±w|ξ|` and `±w̄|ξ|`.
   * They are four **distinct, simple** roots, none purely imaginary.
   * Two have `Re λ = (Re w)|ξ| > 0`.
4. **Simplicity, quantitatively.** `∂_λ P(w|ξ|, iξ) = 2i √(4κθ−γ²) w |ξ|³ ≠ 0`.

*Proof.*

1. Item 1 is the completed square `4κ(κX² − γXY + θY²) = (2κX − γY)² + (4κθ − γ²)Y²`.
2. For item 2, `μ + μ̄ = −(2κ+γ)/κ` and `κμμ̄ = κ + γ + θ`. Expanding
   `κ(ζ₀² + μZ)(ζ₀² + μ̄Z)` therefore gives
   `κζ₀⁴ − (2κ+γ)ζ₀²Z + (κ+γ+θ)Z² = P(ζ)`.
3. For item 3, `P(λ, iξ) = 0` means `λ² = μ|ξ|²` or `λ² = μ̄|ξ|²`, because `Z = −|ξ|²`.
   * Write `w = a + ib`. Then `μ = a² − b² + 2iab` with `Im μ ≠ 0`, so `ab ≠ 0`.
   * Hence `w, −w, w̄, −w̄` are distinct, and none is purely imaginary.
4. For item 4, `∂_λP = 2λ[2κ(λ²+|ξ|²) + γ|ξ|²]`. At `λ² = μ|ξ|²` this equals
   `2λ|ξ|²(2κ(μ+1) + γ) = 2λ|ξ|² · i√(4κθ−γ²)`. ∎

LEAN VERIFIED:

* Item 1: `S004CharacteristicCone.symbolForm_pos_definite_iff`, `admissible_elliptic_covector`.
* Item 2: `laplaceSymbol_factor`.
* Item 3: `hadamardRatio_root`, `hadamard_sqrt_re_ne_zero`, `exists_growing_simple_root`.
* Item 4: `laplaceSymbolDeriv_at_hadamard_root`, `hadamard_root_simple`,
  `laplaceSymbolDeriv_homogeneous`.

Symbolic: I2.

*Relation to the parameters outside admissibility.* If `γ² > 4κθ` (with `κ > 0`), `μ` is real.
Strictly hyperbolic triples have `μ < 0` and purely imaginary roots, so they give no growth; see
`two_positive_roots_iff` and negative control C4 in I7.

*Conventions.* Two changes of convention are harmless:

* Reversing the signature (`□ ↦ −□`) maps `γ ↦ −γ`. Admissibility and the root set are invariant
  under `γ ↦ −γ` up to relabelling.
* The orientation `L ↦ −L` multiplies `P` by `−1`.

Neither change affects any statement below.

## 4. The theorem

Fix an open set `Ω ⊂ ℝ × ℝ^d` containing `(0, 0)` (translate the base point there), `n ≥ 1`, and

```
L = a(t,x) P(∂) I_n + Σ_{|α|≤3} A_α(t,x) ∂^α,                                           (4.1)
```

with the following hypotheses:

* `a ∈ C^∞(Ω)` and `A_α ∈ C^∞(Ω; ℂ^{n×n})`;
* `(κ, θ, γ)` admissible;
* `a(0,0) ≠ 0`;
* `T₀ > 0` and `r₀ ∈ (0, 1/8]` with `K₀ = [0,T₀] × B̄_{r₀} ⊂ Ω` and `a ≠ 0` on `K₀`.

**Definition 4.1 (local Cauchy estimate).** For `s, σ ∈ ℝ`, `T ∈ (0, T₀]`, `ρ ∈ (0, r₀]` and
`C ≥ 0`, the statement `E(s, σ, T, ρ, C)` is: for every `v ∈ C^∞([0,T] × ℝ^d; ℂ^n)` with
`supp v ⊂ [0,T] × B_ρ`, and every `T′ ∈ (0, T]`,

```
‖v(T′)‖_{H^{−σ}} ≤ C ( Σ_{j=0}^{3} ‖∂ₜ^j v(0)‖_{H^s} + sup_{0≤t≤T′} ‖(Lv)(t)‖_{H^s} ).      (4.2)
```

Here `t = 0` is non-characteristic, since the `∂ₜ⁴` coefficient is `aκ ≠ 0`, and the Cauchy data
are `(v, ∂ₜv, ∂ₜ²v, ∂ₜ³v)`. Estimate (4.2) is the weakest quantitative form of continuous
dependence for the local Cauchy problem:

* it allows any loss of derivatives `s + σ`;
* it allows any time interval, any ball and any constant;
* it involves only smooth, compactly supported functions.

Every Hadamard-type well-posedness statement implies it; see §6.

**Theorem 1 (local Sobolev ill-posedness).** Under the hypotheses above, `E(s, σ, T, ρ, C)` is
false for every `s, σ ∈ ℝ`, `T ∈ (0, T₀]`, `ρ ∈ (0, r₀]` and `C ≥ 0`.

Quantitatively, for every `M ≥ 0` there are `K > 0`, `N₀` and functions `v_N` (`N ≥ N₀`) in the
class of Definition 4.1 with `T_N = K log N / N ≤ T` and

```
‖v_N(T_N)‖_{H^{−σ}} ≥ N^M ( Σ_{j≤3} ‖∂ₜ^j v_N(0)‖_{H^s} + sup_{0≤t≤T_N} ‖L v_N(t)‖_{H^s} ).   (4.3)
```

If `a` and `A_α` are real, the `v_N` can be chosen real-valued (with `N^M` replaced by `N^M/2`).
The same holds backward in time.

**Corollary 1.1 (S004, arbitrary background).** Let `κ > 0, θ > 0, γ² < 4κθ`, `ε ≠ 0`, `α ∈ ℝ`,
`V ∈ C^∞`. Let `ψ₀ ∈ C^∞(Ω)` be **any** smooth background, solution or not, and `(t₀, x₀)` a point
with `ψ₀(t₀, x₀) ≠ 0`. Then the conclusions of Theorem 1 hold for the linearized S004 operator
`L_{ψ₀}` at `(t₀, x₀)`, with real-valued `v`.

*Proof.* By Lemma 1, `L_{ψ₀}` has the form (4.1) with `n = 2` and `a = 2F(ρ₀)`, which is real and
`> 0` near `(t₀, x₀)`, and with real `A_α`. Translate `(t₀, x₀)` to the origin and choose
`T₀, r₀` so that `ψ₀ ≠ 0` on `K₀`. ∎

**Quantifiers, time, scope.**

* **Time interval.** The estimate fails on **every** interval: `T_N → 0`.
* **Indices.** It fails for **every** Sobolev pair, i.e. no finite derivative loss helps.
* **Locality.** It fails in **every** neighbourhood of the regular point.
* **Coefficient regularity.** `C^∞` coefficients, equivalently `ψ₀ ∈ C^∞` near the point. For fixed
  `(s, σ)` the proof uses only finitely many derivatives of the coefficients. We do not track that
  number.
* **Scope.** Flat fixed background metric with the inertial foliation, as in the source action. A
  curved or dynamical metric, or a dynamical foliation, changes the principal part (§12).

## 5. Proof of Theorem 1

Monotonicity of the norms lets us replace `(s, σ)` by integers `s, σ ≥ 0`. Since `a ≠ 0` on `K₀`
and `L v = a L̃ v` with

```
L̃ = a^{-1}L = P(∂) I_n + Σ_{|α|≤3} B_α ∂^α,   B_α = a^{-1} A_α ∈ C^∞(K₀),
```

it suffices to prove (4.3) for `L̃`. Multiplication by a smooth compactly supported function `ã`
with `ã = a` on `K₀` is bounded on `H^s`, and `L v_N` is supported in `B_ρ`. Fix the unit vector
`ê = e₁`.

### 5.1 Constants

* Let `w` be the root of Lemma 2 with `Re w > 0`.
* Put `z(x) = 1 + 2iê·x − |x|²` and `S(x) = z(x)^{1/2}`, principal branch.
* For `|x| ≤ 1/4`: `|z − 1| ≤ 2|x| + |x|² ≤ (9/4)|x| < 9/16`. Hence:
  * `S` is real-analytic on `B_{1/4}`;
  * `Re S > 0`;
  * `|S − 1| = |z − 1| / |S + 1| ≤ (9/4)|x|`;
  * `|S| ≥ (7/16)^{1/2} > 0.6`.
* Put `t₁ = min(T₀, 0.3/|w|)`. Then `|S + wt| ≥ 0.3` on `[0, t₁] × B_{1/4}`.

### 5.2 The exact complex phase (Lemma 3)

```
φ(t, x) = iê·x − |x|²/2 + w t S(x) + μ t²/2,     (t, x) ∈ [0, t₁] × B_{1/4}.                 (5.1)
```

**Lemma 3.** On `[0, t₁] × B_{1/4}`:

* (a) `∇ₓφ = (iê − x)(S + wt)/S`;
* (b) `∇ₓφ·∇ₓφ = −(S + wt)²`;
* (c) `∂ₜφ = w(S + wt)`;
* (d) `P(∇φ) = 0` identically;
* (e) `V₀ := (∂_{ζ₀}P)(∇φ) = 2i√(4κθ−γ²) w (S + wt)³ ≠ 0`;
* (f) `Re φ = −|x|²/2 + t Re(wS) + (t²/2) Re μ`, and in particular `Re φ(t,0) = t Re w + (t²/2) Re μ`;
* (g) `|Re φ(t,x) − Re φ(t,0) + |x|²/2| ≤ C₁ t |x|` with `C₁ = 9|w|/4`;
* (h) `φ(0, x) = iê·x − |x|²/2`.

*Proof.*

* (a) `S² = z` gives `∇S = (iê − x)/S`.
* (b) `(iê − x)·(iê − x) = −z = −S²`.
* (c) `μ = w²`.
* (d) `(∂ₜφ)² + μ∇ₓφ·∇ₓφ = w²(S+wt)² − w²(S+wt)² = 0`, then Lemma 2(2).
* (e) `∂_{ζ₀}P = 2κζ₀[(ζ₀² + μZ) + (ζ₀² + μ̄Z)]`. On `∇φ` the first bracket vanishes and the second
  equals `(μ − μ̄)(S+wt)² = 2i Im μ (S+wt)²`; also `2κ Im μ = √(4κθ−γ²)`.
* (f) and (h) are direct.
* (g) follows from (f) and `|w(S − 1)| ≤ (9/4)|w||x|`. ∎

*Why the formal series terminates.* The general Gaussian-beam method solves `∂ₜφ = H(∇ₓφ)`, with
`H(ζ′) = w(−ζ′·ζ′)^{1/2}`, by a power series in `t`. For the initial phase
`φ(0, ·) = −(x − iê)·(x − iê)/2 − 1/2` this series stops after the `t²` term. The reason is that
the initial phase is a complex "spherical" phase centred at the complex point `iê`, and `H` is
isotropic and homogeneous of degree one.

* The flow only rescales `∇ₓφ`, by the factor `(S + wt)/S`, along the complex radial field
  `iê − x`.
* Hence `H(∇ₓφ) = w(S + wt)` is affine in `t`.

This is the complexification of the fact that spherical wavefronts stay spherical under the
isotropic eikonal flow. Consequences:

* the recursion gives `φ₂ = μ/2` and `φ_j = 0` for `j ≥ 3` (I4 checks the closed form for
  `d = 1, 2, 3`);
* **no truncation error is incurred in the phase**, in any dimension;
* in `d = 1`, `S = 1 + iy` and (5.1) reduces to `φ = iy + wt − (y − iwt)²/2`
  (LEAN `beamPhase`, `beam_eikonal`).

### 5.3 Conjugation (Lemma 4)

**Lemma 4.** For `a ∈ C^∞([0,t₁] × B_{1/4}; ℂ^n)` and `N ≥ 1`,

```
e^{−Nφ} L̃(e^{Nφ} a) = Σ_{k=1}^{4} N^{4−k} Q_k a.                                          (5.2)
```

* Each `Q_k` is a differential operator of order `≤ k` with coefficients in
  `C^∞([0,t₁] × B_{1/4})` that do not depend on `N`.
* `Q₁ a = Σ_{j=0}^{d} V_j ∂_j a + 𝒞 a`, with `V_j = (∂_{ζ_j}P)(∇φ)` and `𝒞` a smooth matrix
  function.

*Proof.* Put `D_j = ∂_j + N∂_jφ`, so that `e^{−Nφ}∂^α(e^{Nφ}a) = D^α a`. The `D_j` commute because
`∂_j∂_kφ = ∂_k∂_jφ`. Hence `e^{−Nφ}P(∂)e^{Nφ} = P(D)` and `e^{−Nφ}B_α∂^αe^{Nφ} = B_α D^α`.

Expand each product `D^β a` in powers of `N`:

* The `N^{|β|}` term is `(∇φ)^β a`. Summed with the coefficients of `P`, the `N⁴` term is
  `P(∇φ)a = 0` by Lemma 3(d).
* The `N^{|β|−1}` terms that put a derivative on `a` take `∂_j` from exactly one factor and `N∂φ`
  from the others, giving `β_j(∇φ)^{β−e_j}∂_j a`. Summed, these give `(∂_{ζ_j}P)(∇φ)∂_j a`.
* The remaining `N³` terms carry no derivative of `a`. Neither does `Σ_{|α|=3}B_α(∇φ)^α a`. These
  form `𝒞 a`.
* Lower powers of `N` have at most `4 − k` derivatives on `a` at power `N^k`.

All coefficients are polynomials in derivatives of `φ` and in the `B_α`. ∎

(I5 checks the `N⁴` and `N³` coefficients exactly for a generic phase and amplitude in 1+1 and 1+2, including
`𝒞 = (1/2)Σ(∂²_{ζ_jζ_k}P)(∇φ)∂_j∂_kφ + Σ_{|α|=3}B_α(∇φ)^α`. Only the structure stated in the lemma
is used below.)

### 5.4 Transport recursion (Lemma 5)

**Lemma 5.** Let `K′ ≥ 1`, `h ∈ C^∞([0,t₁] × B_{1/4}; ℂ^n)` and `a⁽⁰⁾ ∈ C^∞(B_{1/4}; ℂ^n)`. Then
there are `a⁽¹⁾, …, a⁽ᴷ′⁾ ∈ C^∞(B_{1/4}; ℂ^n)` such that `a(t,x) := Σ_{j=0}^{K′} t^j a⁽ʲ⁾(x)`
satisfies

```
Q₁ a − h = t^{K′} r,
```

with `r ∈ C^∞([0,t₁] × B_{1/4})`. Every `x`-derivative of `r` is bounded on `[0,t₁] × B̄_ρ` for
`ρ < 1/4`.

*Proof.* Take `t`-Taylor coefficients at `t = 0`. The coefficient of `t^j` in `Q₁a − h` is

```
(j+1) V₀(0,x) a⁽ʲ⁺¹⁾(x) + G_j(x),
```

where `G_j` is a linear differential expression in `a⁽⁰⁾, …, a⁽ʲ⁾` with coefficients built from the
Taylor coefficients of `V_j, 𝒞, h`. By Lemma 3(e), `V₀(0, x) = 2i√(4κθ−γ²) w S(x)³ ≠ 0`, so we
can solve successively for `a⁽¹⁾, …, a⁽ᴷ′⁾`.

Then `Q₁a − h` is smooth and its first `K′` `t`-Taylor coefficients vanish. Taylor's formula gives

```
r(t,x) = (1/(K′−1)!) ∫₀¹ (1−s)^{K′−1} ∂ₜ^{K′}(Q₁a − h)(st, x) ds.
```

∎

### 5.5 WKB hierarchy and residual (Lemma 6)

Fix `K′, J ≥ 1` and a unit vector `e ∈ ℂ^n`. Define `a₀, …, a_J` by Lemma 5:

* `a₀`: `h = 0`, `a₀(0, ·) = e`;
* `a_m` (`1 ≤ m ≤ J`): `h_m = −Σ_{k=2}^{min(m+1,4)} Q_k a_{m+1−k}`, `a_m(0, ·) = 0`.

Put `A_N = Σ_{m=0}^{J} N^{−m} a_m`.

**Lemma 6.**

```
e^{−Nφ} L̃(e^{Nφ} A_N) = R_N := Σ_{m=0}^{J} N^{3−m} t^{K′} r_m
                              + Σ_{p=J+2}^{J+4} N^{4−p} Σ_{k+m=p, 1≤k≤4, 0≤m≤J} Q_k a_m.   (5.3)
```

For every multi-index `β` there is `C_β` with

```
|∂ₓ^β R_N| ≤ C_β (N³ t^{K′} + N^{2−J})     on [0, t₁] × B̄_ρ,  N ≥ 1.
```

Moreover `|A_N(t,x) − e| ≤ C(t + N^{−1})`.

*Proof.*

1. By (5.2), `e^{−Nφ}L̃(e^{Nφ}A_N) = Σ_{k,m} N^{4−k−m} Q_k a_m`. Group the terms by `p = k + m`.
2. For `p = m′ + 1 ≤ J + 1` the group is `Q₁a_{m′} − h_{m′} = t^{K′} r_{m′}`, by the choice of
   `h_{m′}`.
3. The groups with `p ≥ J+2` carry `N^{4−p} ≤ N^{2−J}`.
4. All coefficients are smooth on the compact set `[0,t₁] × B̄_ρ`, which gives the derivative
   bounds.
5. Finally `a₀(t, x) − e = t ∫₀¹ ∂ₜa₀(st, x) ds` and `|N^{−m}a_m| ≤ C/N` for `m ≥ 1`. ∎

No truncation enters at order `N⁴`, because the phase is exact.

### 5.6 The candidates

* `ρ ∈ (0, r₀]` (so `ρ ≤ 1/8`).
* `χ ∈ C_c^∞(B_ρ)` with `0 ≤ χ ≤ 1` and `χ = 1` on `B_{ρ/2}`.
* `v_N(t, x) = χ(x) e^{Nφ(t,x)} A_N(t, x)` on `[0, t₁] × ℝ^d`, extended by `0`.

Then `v_N ∈ C^∞` with `supp v_N ⊂ [0, t₁] × B_ρ`, and

```
L̃ v_N = χ e^{Nφ} R_N + [L̃, χ](e^{Nφ} A_N).                                              (5.4)
```

The commutator `[L̃, χ]` has order `≤ 3` and coefficients supported in the annulus
`𝔄 = {ρ/2 ≤ |x| ≤ ρ}`. Hence `[L̃, χ](e^{Nφ}A_N) = e^{Nφ} g_N`, with `g_N` supported in `𝔄` and
`‖g_N(t)‖_{C^s} ≤ C N^{3+s}`.

### 5.7 Norm estimates (Lemma 7)

Let `T_N = K log N / N` and `G_N(t) = ∫_{B_ρ} e^{2N Re φ(t,x)} dx`.

**Lemma 7.** There are `N₁` and constants `c₀, C₀, C_m` such that for `N ≥ N₁` and `0 ≤ t ≤ T_N`:

* (i) `‖e^{Nφ(t)} g‖_{H^m} ≤ C_m N^m ‖g‖_{C^m} G_N(t)^{1/2}` for `g` supported in `B̄_ρ`, and the same
  with `G_N` replaced by the integral over `supp g`;
* (ii) `c₀ N^{−d/2} e^{2N Re φ(t,0)} ≤ ∫_{|x|≤N^{−1/2}} e^{2N Re φ(t,x)} dx ≤ G_N(t) ≤ C₀ N^{−d/2} e^{2N Re φ(t,0)}`;
* (iii) `e^{N Re φ(t,x)} ≤ e^{N Re φ(t,0)} e^{−Nρ²/8} N^{C₁ρK}` on `𝔄`;
* (iv) `|N Re φ(t,0) − N t Re w| ≤ K² (log N)² |Re μ| / (2N)`. Hence `e^{N Re φ(t,0)} ≤ 2N^{K Re w}`
  for `t ≤ T_N`, and `e^{N Re φ(T_N,0)} ≥ N^{K Re w}/2`.

*Proof.*

* (i) `∂^β(e^{Nφ}g) = e^{Nφ}(∂ + N∇ₓφ)^β g`. Expand; each term is at most
  `C N^{|β|} |e^{Nφ}| max_{γ≤β}|∂^γ g|`, since the derivatives of `φ` are bounded on the compact set.
  Then use the equivalence of `H^m` norms for integer `m`.
* (ii) By Lemma 3(g), `2N(Re φ(t,x) − Re φ(t,0)) ∈ [−N|x|² − 2C₁K log N |x|, −N|x|² + 2C₁K log N |x|]`.
  * *Lower bound.* On `|x| ≤ N^{−1/2}` the left end is `≥ −2` for `N` large. Integrate.
  * *Upper bound.* Substitute `x = y/√N` and use
    `2C₁K log N |y|/√N ≤ |y|²/2 + 2C₁²K²(log N)²/N`. This gives
    `G_N ≤ e^{2N Re φ(t,0)} N^{−d/2} (2π)^{d/2} e^{2C₁²K²(log N)²/N}`.
* (iii) Lemma 3(g) gives `−|x|²/2 + C₁t|x| ≤ −ρ²/8 + C₁tρ` on `𝔄` (LEAN `annulus_phase_bound`).
  Then use `NT_N = K log N`.
* (iv) Lemma 3(f) (LEAN `log_scale_exponential` for `exp(N T_N c) = N^{Kc}`). ∎

### 5.8 The three estimates

Let `N ≥ N₁` with `T_N ≤ min(T, t₁)` and `N^{−1/2} ≤ ρ/2`.

**Data.** `∂ₜ^j v_N(0) = χ e^{Nφ(0)} [(∂ₜ + N∂ₜφ)^j A_N](0)` and `Re φ(0, 0) = 0`. By Lemma 7(i)–(ii),

```
Σ_{j≤3} ‖∂ₜ^j v_N(0)‖_{H^s} ≤ C N^{3+s−d/4}.                                              (5.5)
```

**Source.** By (5.4), Lemma 6 and Lemma 7, for `0 ≤ t ≤ T_N`,

```
‖L̃ v_N(t)‖_{H^s} ≤ C N^{s−d/4} e^{N Re φ(t,0)} (N³ T_N^{K′} + N^{2−J})
                  + C N^{3+s} e^{N Re φ(t,0)} e^{−Nρ²/8} N^{C₁ρK}.                        (5.6)
```

**Growth.** On `|x| ≤ N^{−1/2}`, `χ = 1` and `|A_N| ≥ 1/2` by Lemma 6. So

```
‖v_N(T_N)‖²_{L²} ≥ (c₀/4) N^{−d/2} e^{2N Re φ(T_N,0)},
‖v_N(T_N)‖_{H^σ} ≤ C N^{σ−d/4} e^{N Re φ(T_N,0)},
```

and by duality

```
‖v_N(T_N)‖_{H^{−σ}} ≥ ‖v_N(T_N)‖²_{L²} / ‖v_N(T_N)‖_{H^σ} ≥ c N^{−σ−d/4} e^{N Re φ(T_N,0)}
                    ≥ (c/2) N^{K Re w − σ − d/4}.                                         (5.7)
```

### 5.9 The contradiction

Divide (5.7) by the sum of (5.5) and the supremum of (5.6). Use Lemma 7(iv) and
`N³T_N^{K′} = K^{K′}(log N)^{K′} N^{3−K′}`. With `Λ = s + σ`,

```
LHS / RHS ≥ c [ N^{3+Λ−K Re w} + K^{K′}(log N)^{K′} N^{3+Λ−K′} + N^{2+Λ−J}
                + N^{3+Λ+d/4+C₁ρK} e^{−Nρ²/8} ]^{−1}.                                    (5.8)
```

Given `M`, choose:

* `K > (3 + Λ + M)/Re w`;
* `K′ > 3 + Λ + M` (integer);
* `J > 2 + Λ + M` (integer).

Each bracket term is then `O(N^{−M−η}(log N)^{K′})` for some `η > 0`. The last term decays faster
than any power of `N`. Hence the ratio exceeds `N^M` for `N ≥ N₀`, which is (4.3). Since `T_N → 0`,
the estimate `E(s, σ, T, ρ, C)` fails for every `T`, `ρ` and `C`.

The exponent bookkeeping is LEAN VERIFIED:

* `S004LocalizedHadamard.ratio_tendsto_atTop`;
* `localized_estimate_fails`;
* `parameter_choice`, which uses the cruder sufficient gap `K′ > 4 + Λ`.

### 5.10 Real-valued solutions

If `a` and `A_α` are real, then `L(Re v) = Re(Lv)` and `‖Re f‖_{H^s} ≤ ‖f‖_{H^s}`, because the
weight `(1+|ξ|²)^s` is even. An estimate (4.2) for real functions, applied to `Re v_N` and
`Im v_N`, gives (4.2) for `v_N` with `2C`. So the failure for `v_N` forces failure for `Re v_N` or
`Im v_N`. ∎ (Theorem 1)

## 6. Solution operators, semigroups and the homogeneous problem

**Corollary 2 (no bounded solution operator with sources).** Fix `s, σ, T, ρ`. There is no map
`𝒮: (g, f) ↦ 𝒮(g, f) ∈ C([0,T]; H^{−σ})` with both of the following properties:

* `sup_t ‖𝒮(g,f)(t)‖_{H^{−σ}} ≤ C(Σ_j ‖g_j‖_{H^s} + sup_t ‖f(t)‖_{H^s})`;
* `𝒮((∂ₜ^jv(0))_j, Lv) = v` for every smooth `v` supported in `[0,T] × B_ρ`.

Neither linearity nor any condition beyond these two is needed. *Proof:* the two properties give
(4.2). ∎

**Theorem 2 (no `C₀`-semigroup; no evolution system).** Write the problem in first-order form
`U = (v, ∂ₜv, ∂ₜ²v, ∂ₜ³v)`, `U′ = 𝒜(t)U + G`, where

* `𝒜(t)U = (u₁, u₂, u₃, −(aκ)^{−1}(L − aκ∂ₜ⁴)[U])`;
* `G = (0, 0, 0, (aκ)^{−1}Lv)`.

Let `X` be a Banach space of 4-tuples of functions on `ℝ^d` with

```
(S2)  c‖u₀‖_{H^{−σ}} ≤ ‖U‖_X ≤ C Σ_j ‖u_j‖_{H^s}   for U ∈ C_c^∞(B_ρ)^4.
```

* (a) Suppose the coefficients are `t`-independent. Then no `C₀`-semigroup `e^{t𝒜_X}` on `X`
  satisfies (S1): `C_c^∞(B_ρ)^4 ⊂ D(𝒜_X)` and `𝒜_X U = 𝒜U` there.
* (b) For `t`-dependent coefficients, there is no strongly continuous, uniformly bounded evolution
  system `U(t, τ)` on `X` satisfying (S1′): for `h ∈ C_c^∞(B_ρ)^4`, `τ ↦ U(t,τ)h` is
  differentiable with `∂_τ U(t,τ)h = −U(t,τ)𝒜(τ)h`.

*Proof.* Let `v` be smooth and supported in `[0,T] × B_ρ`. By (S2) applied to difference
quotients, `V ∈ C¹([0,T]; X)`, `V(t) ∈ D(𝒜)` and `V′ = 𝒜V + G`.

* In case (a), `τ ↦ e^{(t−τ)𝒜}V(τ)` is differentiable with derivative
  `e^{(t−τ)𝒜}(V′ − 𝒜V)(τ) = e^{(t−τ)𝒜}G(τ)`. This is the product rule; it uses `V(τ) ∈ D(𝒜)` and
  the strong continuity and uniform boundedness of the semigroup on `[0,T]`.
* In case (b), the same holds for `τ ↦ U(t,τ)V(τ)`, by (S1′).

Integrating from `0` to `t` gives Duhamel's formula `V(t) = U(t,0)V(0) + ∫₀^t U(t,τ)G(τ) dτ`. Hence

```
‖V(t)‖_X ≤ M(‖V(0)‖_X + T sup ‖G‖_X).
```

By (S2), and since multiplication by `(aκ)^{−1}` is bounded on `H^s` for functions supported in
`B_ρ`, this is (4.2). That contradicts Theorem 1. ∎

Case (a) covers the linearization about every **static or co-rotating** state
`ψ₀ = e^{iωt}φ(x)`, in co-rotating variables, at every point where `φ ≠ 0`. Hypothesis (S1) holds
for every realization of a differential operator whose domain contains the smooth compactly
supported functions near the point.

**Theorem 3 (class A: exact growing solutions at fixed time).** Let
`ψ₀ = √ρ₀ e^{iωt}` with `ρ₀ > 0` and `ω ∈ ℝ`, on- or off-shell. In co-rotating variables the
linearization is a constant-coefficient operator with symbol `M(λ, ξ) = 2F(ρ₀)P(λ, iξ)I₂ + R(λ, ξ)`,
where `R` has degree `≤ 3`. There are `R₀`, `c > 0` and, for each `|ξ| ≥ R₀`, a root `λ(ξ)` of
`det M` with `Re λ(ξ) ≥ (Re w/2)|ξ|` and `λ(ξ)/|ξ| → w`. On the torus,
`u = Re[e^{λ(ξ)t + iξ·x} v(ξ)]`, with `M(λ(ξ), ξ)v(ξ) = 0`, are exact real solutions. For every
fixed `t > 0`,

```
‖u(t)‖_{H^{−σ}} / Σ_j ‖∂ₜ^j u(0)‖_{H^s} ≥ c e^{(Re w/2)|ξ| t} |ξ|^{−3−s−σ} → ∞.
```

*Proof.*

1. By `U(1)` and translation invariance of the density, the linearization in co-rotating
   variables `v = R(ωt)f` has constant coefficients.
2. Rotating `∂ₜ ↦ ∂ₜ + ωJ` changes only terms of order `≤ 3`. So `R` has degree `≤ 3`, and
   `det M(|ξ|ℓ, ξ) = (2F(ρ₀))² |ξ|⁸ [p(ℓ)² + E]` with `|E| ≤ C/|ξ|` for `ℓ` in a compact set, where
   `p(ℓ) = κ(ℓ²+1)² + γ(ℓ²+1) + θ`.
3. Take `0 < r < min(Re w, |Im w|)`. Then `p²` has only the double zero `w` in `|ℓ − w| ≤ r`, and
   `|p²| ≥ m_r > 0` on the circle `|ℓ − w| = r`.
4. By Rouché, for `|ξ| > C/m_r`, `det M` has two zeros with `|ℓ − w| < r`, so
   `Re λ ≥ (Re w − r)|ξ|`. Take `r = Re w/2`.
5. The norm ratio follows from `|λ| ≤ C|ξ|`. ∎

This extends Theorem H4 of `docs/foundational/GHOST_AND_CONSTRAINT_ANALYSIS.md` (static, on-shell)
to co-rotating and off-shell homogeneous backgrounds. I6 checks it numerically for
`(κ, θ, γ) = (1, 2, 1/2)`, `ρ₀ = 0.8`, `ω = 0.7`: `|λ/k − w| = 0.70/k` for `k = 10 … 10⁴`.

## 7. Background classes

| Class | Definition | Result | Status |
|---|---|---|---|
| A | constant or co-rotating homogeneous `ψ₀ = √ρ₀ e^{iωt}`, `ρ₀ > 0` | Theorem 1 at every point. Theorem 3: exact solutions, fixed time, homogeneous problem. Theorem 2(a). | ANALYTICALLY PROVED; NUMERICALLY VERIFIED (I6) |
| B | nonconstant, no nodes on the region considered | Theorem 1 at every point. Theorems 2, 4. | ANALYTICALLY PROVED; NUMERICALLY VERIFIED (I7: B2, B3) |
| C | arbitrary smooth background with one regular point, nodes elsewhere | Theorem 1 at the regular point; nodes are irrelevant | ANALYTICALLY PROVED; NUMERICALLY VERIFIED (I7: B1) |

For class C the result is local. The candidates `v_N` are supported in an arbitrarily small ball
around the regular point, where `F(ρ₀) ≥ c > 0`.

* The degeneration of the operator at a node (where `F(0) = 0` and the order drops) plays no role.
* Any global well-posedness statement, on a domain containing nodes, that reproduces smooth
  compactly supported solutions and is bounded between Sobolev-type norms restricts to (4.2) on
  functions supported near the regular point. Such a statement is therefore impossible.

## 8. Nonlinear implications

**Theorem 4 (no differentiable forced solution map).** Let `ψ₀` be a smooth solution of `E(ψ₀) = 0`
on `[0, T₀] × ℝ^d` with a regular point `(0, 0)`. Let `X, Y, Z` be normed spaces of data, solutions
and sources such that, for functions supported in `B_ρ`:

```
‖u‖_{H^{−σ}} ≤ C‖u‖_Y,   ‖g‖_X ≤ C Σ‖g_j‖_{H^s},   ‖f‖_Z ≤ C‖f‖_{H^s}.
```

Then there is no map `Φ(g, f)`, defined near `(Ψ₀(0), 0)` in `(Ψ₀(0) + X) × C([0,T]; Z)` with values
in `ψ₀ + C([0,T]; Y)`, that has both of the following properties:

* (R) it reproduces smooth solutions: `Φ(Ψ(0), E(ψ)) = ψ` whenever `ψ − ψ₀` is smooth and supported
  in `[0, T] × B_ρ`;
* (D) it is Fréchet differentiable at `(Ψ₀(0), 0)`.

*Proof.* For such `v` and `ψ_ε = ψ₀ + εv`, Taylor's formula gives `E(ψ_ε) = εL_{ψ₀}v + ε²q_ε`, with
`q_ε` bounded in `C([0,T]; H^s)`. So the curve `ε ↦ (Ψ_ε(0), E(ψ_ε))` is differentiable at `0` with
derivative `(V(0), L_{ψ₀}v)`. By (R), `Φ` maps the curve to `ψ_ε`. By the chain rule,
`v = DΦ(V(0), L_{ψ₀}v)`, so `‖v(t)‖_{H^{−σ}} ≤ C(…)`, which is (4.2). This contradicts
Corollary 1.1. ∎

*Unforced flows.* Let `Φ_{t,τ}` be solution maps of `E(ψ) = 0` from every initial time `τ`.
Suppose they are Fréchet differentiable at `Ψ₀(τ)` and their derivatives `U(t,τ)` form an evolution
system satisfying (S1′). Then Theorem 2(b) applies to `U`, so no such flows exist.

Coupling to an external current, `L ↦ L + Re(J̄ψ)`, gives `E(ψ) = J`. So the forced problem of
Theorem 4 is the standard linear-response setting.

**What follows, and what does not.**

| Statement | Status |
|---|---|
| No `C¹` (even pointwise Fréchet-differentiable) forced solution map near any smooth solution with a regular point, in any Sobolev-comparable spaces | ANALYTICALLY PROVED (Theorem 4) |
| No `C¹` unforced flows from every initial time whose derivative family satisfies (S1′) | ANALYTICALLY PROVED (Theorems 2(b), 4) |
| Linearized dynamics about every static or co-rotating localized state with a nonzero core: no `C₀`-semigroup on Sobolev-comparable spaces; no linear spectral stability in the `C₀` sense | ANALYTICALLY PROVED (Theorem 2(a)) |
| Orbital stability via Grillakis–Shatah–Strauss / Lyapunov / spectral methods: these methods need a `C¹` flow or a `C₀` linearization, which do not exist | ANALYTICALLY PROVED (inapplicability) |
| Orbital stability in a framework with merely continuous, non-differentiable flows | OPEN (not refuted) |
| Stable and centre manifold theorems: they require `C¹` semiflows, so they are inapplicable. Exact finite-dimensional invariant families (symmetry orbits of an exact solution) are not excluded. | ANALYTICALLY PROVED / not excluded |
| Analytic data: Cauchy–Kovalevskaya gives local analytic solutions for analytic coefficients and data, because `t = 0` is non-characteristic. The existence time scales like the analyticity radius, which is `~1/N` for frequency-`N` data. No contradiction. | consistent; analytic-class well-posedness is NOT excluded |
| Gevrey classes `G^s`, `s > 1` (expected ill-posed with `T_N ∝ N^{1/s−1}`) | OPEN (not proved here) |
| Nonlinear norm inflation, Hölder discontinuity of a continuous flow, finite-time blow-up | OPEN. **Not claimed.** |

## 9. Dependencies

The proof of Theorem 1 uses:

* Taylor's formula with integral remainder;
* the Fourier definition of `H^s` and the equivalent derivative norms for integer `m`;
* the Leibniz rule;
* elementary Gaussian integrals and the inequality `ab ≤ a²/2 + b²/2`.

Theorem 2 also uses the product rule for strongly continuous operator families, which is elementary
and whose proof is given inline. Theorem 4 uses the chain rule for Fréchet derivatives.

**No existence theorem, energy estimate, microlocal result or the Lax–Mizohata theorem is used.**
The external Lax–Mizohata citation in earlier reports is replaced by Theorem 1.

The remaining hypotheses are those of the no-go statements themselves: the hypothesized solution
operators or flows. They are not unproved lemmas.

## 10. Adversarial review

| Concern | Resolution |
|---|---|
| Root multiplicities | The system symbol `P·I_n` has each root with multiplicity `n`, but it is scalar times identity. The scalar `P` has four simple roots (Lemma 2), so the transport equation is a vector ODE in `t` with invertible leading coefficient `V₀ ≠ 0`. |
| Sign conventions | Signature flip maps `γ ↦ −γ`. Orientation flip maps `P ↦ −P`. Neither changes admissibility or the root set (§3). |
| Matrix coupling | Off-diagonal couplings have order `≤ 3`. They enter only `𝒞` (order 3, at `N³`) and `Q_k` (`k ≥ 2`). They cannot remove growth, which sits in the exact phase. The amplitude stays `e + O(t + 1/N)`. |
| Derivatives of `F` | `F′, F″, F‴` occur only in coefficients of order `≤ 3` (Lemma 1, I1). |
| Cutoff effects | The commutator lives where the beam is smaller by `e^{−Nρ²/8}N^{C₁ρK}` (Lemma 7(iii)), which is faster than any power. |
| Short-time error accumulation | Nothing is time-stepped. `L v_N` is computed exactly and bounded pointwise (Lemma 6). |
| Formal vs actual evolution | `v_N` is an actual smooth compactly supported function; only finite sums are used. The contradiction is with (4.2) for actual functions. |
| Approximate solutions only | Theorem 1 refutes the inhomogeneous estimate, which every Hadamard well-posedness implies (Corollary 2, Theorem 2). Exact solutions are used in class A (Theorem 3). |
| Analytic spaces | Not excluded (§8). The theorem is a Sobolev (finite-loss) statement. |
| Linear-to-nonlinear transfer | Only through differentiability (Theorem 4) or semigroup/evolution-system hypotheses (Theorem 2). No norm inflation is claimed. |
| Nodes | Irrelevant to the local statement (§7). |
| Background not a solution | Theorem 1 never uses `E(ψ₀) = 0`. Theorem 4 does. |
| Complex `v_N` | §5.10. |
| `T_N → 0` | The estimate must hold for all `T′ ≤ T`. Fixed-time unboundedness is proved in class A (Theorem 3). |

## 11. Verification record

**Symbolic and numerical** (`derivations/s004_illposedness_verify.py`; receipt
`audit/s004_closure/illposedness_verification.json`; tests `derivations/test_s004_illposedness.py`):

| Check | Content | Label |
|---|---|---|
| I1 | linearization structure, generic background, 1+1 and 1+2: order 4, total background order of the `∂^α` coefficient `≤ 4−|α|`, `F‴`, `V″`, real | exact |
| I2 | factorization, simple root, `∂_λP = 2iw√(4κθ−γ²)`, four sampled triples | exact |
| I3 | `P(∂)e^{Nφ} = 0` exactly in 1+1 (frozen part); closed-form `L²` norm | exact |
| I4 | exact beam phase, d = 1, 2, 3: `∇ₓφ·∇ₓφ = −(S+wt)²`, `∂ₜφ = w(S+wt)`, `P(∇φ) = 0`; `|S−1| ≤ (9/4)|x|` | exact (+ sampled bound) |
| I5 | conjugation: `N⁴` and `N³` coefficients, generic `φ`, `a`, 1+1 and 1+2 | exact |
| I6 | class A co-rotating, off-shell: constant co-rotating coefficients; `λ/k → w`, error `0.70/k` | numerical |
| I7 | end-to-end on three backgrounds, refinement study, four negative controls | numerical |
| I8 | exponent bookkeeping for a grid of `(s, σ)` | exact rationals |

**I7 backgrounds.** Each uses the exact Euler–Lagrange linearization, with the principal part
checked symbolically:

* B1: class C, `ψ₀ = cos y + 0.3i cos y sin(y+t)`, nodes at `y = ±π/2`, `(κ,θ,γ) = (1,1,0)`;
* B2: class B, nodeless, `1 + 0.4 sin(y − 0.7t) + 0.5i cos(2y + t)`, `(1, 4, −1)`, `ε = 4/5`,
  `α = 1/2`, `V = u²/2 − u³/3`;
* B3: a seeded random smooth background with a random admissible triple, `ε`, `α` and cubic `V`.

**I7 measurements.** Amplitudes are Taylor jets with exact polynomial algebra. FFT
differentiation is **not** used, because the phase derivatives are not periodic; this was a failed
first candidate, see below. The residual `L v_N` is evaluated with the **exact** coefficients.
Measured (receipt `audit/s004_closure/illposedness_verification.json`):

* coefficient-jet error `≤ 2×10⁻¹⁵`;
* transport consistency `≤ 1.2×10⁻¹³`;
* exact-eikonal `N⁴` coefficient `≤ 5×10⁻¹⁴`;
* hierarchy jets (analytically zero) `≤ 6.2×10⁻⁶`;
* `|u(T_N)|` / Gaussian prediction within `T_N` of 1, approaching 1 as `N` grows (B2: `0.967 → 0.994`);
* residual/growth decaying (slopes `−1.58`, `−6.7`, `−3.4`);
* contradiction ratio increasing on every background:

| `N` | B1 (class C) | B2 (class B) | B3 (sampled: `κ=13/8, θ=8/5, γ=−48/25`) |
|---|---|---|---|
| 400 | 146 | 0.0038 | 25 |
| 800 | 363 | 1.17 | 460 |
| 1600 | 1164 | 386 | 1688 |
| 3200 | 3774 | 2738 | 38720 |

On B2, `Re w = 0.866` gives fast growth, but at `N ≤ 800` (`T_N ≥ 0.08`) the `t`-truncation still
dominates the residual. The contradiction appears once `N³T_N^{K′}` has decayed, exactly as §5.9
predicts.

**Refinement and error separation** (on B1).

* **Analytic error.** The residual slope responds to the number `J` of WKB corrections:
  `+0.09` (J = 2), `−0.80` (J = 3), `−1.58` (J = 4). The predicted asymptotic slopes `2 − J` are
  approached from above as `N` grows.
* **Jet truncation.** Changing the jet degree 64 → 48 leaves the residual unchanged at the reported
  precision.
* **Quadrature.** Changing the grid 4001 → 8001 changes `|u(T_N)|` by `0` and the residual by `5×10⁻⁶`
  relative.
* **Rounding, part 1.** `P(∇φ) ≡ 0` analytically (I3, I4, Lean `beam_eikonal`), but its float64
  value is `~10⁻¹⁶ N⁴` relative (`2.8×10⁻²` at `N = 3200`). It is excluded from the residual and
  reported separately per row. At `N ≥ 1600` it would otherwise exceed the analytic residual. This is
  a documented rounding floor, not an approximation error.
* **Rounding, part 2.** The rounding floor of the retained terms is reported per row as an
  **upper estimate**: the analytically zero hierarchy jets, evaluated at the beam scale and
  multiplied by `N^p`. It stays below the measured residual for `N ≤ 1600` on all three
  backgrounds. At `N = 3200` it is comparable (B1, B2) or larger (B3), which is where the study
  stops.

**Negative controls** (each must fail to produce a contradiction, and does). Ratios for
`N = 400 → 3200`:

* C1: wrong root `1.05w`, so `P(∇φ) ≠ 0`: `9.7×10⁻¹¹ → 3.0×10⁻¹⁴`;
* C2: decaying root `−w̄`: `3.4×10⁻²⁰ → 2.5×10⁻²⁷`;
* C3: transport hierarchy not solved: `8.6×10⁻⁸ → 1.3×10⁻⁹`;
* C4: strictly hyperbolic principal part `(1, 1, 3)`, with purely imaginary roots and no growth:
  `9.5×10⁻⁹ → 1.9×10⁻¹¹`.

**Failed candidate constructions** (preserved):

1. The FFT-differentiated amplitude recursion on a periodic grid gave transport inconsistency
   `6×10⁴` and hierarchy residuals `10⁴¹`. Spectral differentiation of non-periodic
   phase-derivative products produces Gibbs errors. It was replaced by Taylor-jet algebra.
2. Jets in the unscaled variable `y` lost the hierarchy to cancellation (`0.15` at level `N⁰`). The
   scaled variable `Y = y/0.25` restores `10⁻⁹`.
3. The class-A check with the explicit weight ran out of memory. It was replaced by the exact cubic
   Taylor polynomial of `F` at `ρ₀`, which is exact since only `F…F‴` enter (I1); it runs in 4 s.
4. The original resonance candidate `V″ = −1/2` is REFUTED; see the Phase XXXI record.

**Lean.** The pinned `lake build` passes. The strict audit covers **136 roots**: the 100 earlier
roots and 36 Phase XXXIII roots. All depend only on `propext`, `Classical.choice` and
`Quot.sound`. The checker enumerates the root names; see the report §8.

* `S004CharacteristicCone`:
  * the Laplace symbol (`laplaceSymbol`, `laplaceSymbol_real_frequency`);
  * the factorization (`hadamardRatio_add_conj`, `hadamardRatio_mul_conj`, `laplaceSymbol_factor`);
  * the derivative (`laplaceSymbol_hasDerivAt`, `laplaceSymbolDeriv_homogeneous`,
    `laplaceSymbolDeriv_at_hadamard_root`);
  * the root and its simplicity (`laplaceSymbol_at_hadamard_root`, `hadamard_root_simple`,
    `exists_growing_simple_root`).
* `S004LocalizedHadamard`:
  * the 1+1 beam phase and its derivatives (`beamPhase_hasDerivAt_t/_y`, `beam_transport`,
    `beam_eikonal`, `laplaceSymbol_first_order_factor`, `beam_first_order_annihilated`);
  * exact `L²` norm and growth (`beam_sq_integrable`, `beam_norm_sq_integral`,
    `beam_growth_factor`, `beam_growth_on_log_scale`, `log_scale_exponential`);
  * phase inequalities (`gaussian_completing_square`, `annulus_phase_bound`);
  * the contradiction step (`ratio_tendsto_atTop`, `no_uniform_estimate_of_ratio`,
    `localized_estimate_fails`, `parameter_choice`);
  * the package `localized_hadamard_beam`.
* `S004ScopedNoGo.s004_scoped_no_go_localization` and
  `S004Issue53Interface.s004_issue53_interface_with_localization` integrate them.

**Not formalized:** Sobolev spaces, the variable-coefficient transport hierarchy (Lemmas 4–6), the
norm estimates of Lemma 7 in `d` dimensions, and the passage to (4.2). These are ANALYTICALLY PROVED
above, not LEAN VERIFIED. No Lean theorem about the principal symbol is presented as a Lean proof of
the PDE theorem.

## 12. Scope and limitations

* **Model assumptions:**
  * the corrected S004 density with `F(u) = u/(u + ε²e^{−2αu})²`, `ε ≠ 0`;
  * any smooth `V`;
  * admissible `(κ, θ, γ)`;
  * a flat fixed background metric;
  * the fixed inertial foliation (`Δ_h = Δ`);
  * two real components.
* **Not examined:**
  * a dynamical metric or a curved background, including covariance of the spatial operator
    `Δ_h^†`;
  * a dynamical foliation `n^μ`;
  * nonlocal or auxiliary-field completions;
  * parameter regions outside admissibility.

  These change the principal symbol or its interpretation. The proof extends verbatim to any smooth
  principal part `a(t,x)p(∂)` whose roots are simple and non-real near the point; we do not claim
  more.
* **Not proved:**
  * nonlinear instability without differentiability hypotheses;
  * norm inflation;
  * Gevrey-class ill-posedness;
  * the homogeneous problem for time-dependent backgrounds when solvability is assumed only from a
    single initial time (a non-standard notion).
* **Not claimed:** that Wave Confinement Theory as a whole is refuted. The result concerns the
  analyzed corrected S004 action.
