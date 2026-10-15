# Foundational stage 2 — nine open issues: checkable deductions and explicit non-closures

**Living research extension, October 2026.** The source of truth remains [the corrected action](../WCT_CORRECTED_VARIATIONAL_CLOSURE_DRAFT.md), the [derivation audit](../DERIVATION_CLOSURE.md), and the exact [Q1–Q7](cross_paper/QUADRATIC_ACTION_AND_DISPERSION.md)/[F1–F2](cross_paper/FINITE_SCALE_SELECTION.md) calculations. All nine issues remain open unless their *original* acceptance criteria are independently met. This document adds a limited theorem, formal counterexamples, executable identities, and a strict holdout gate. It neither backdates a paper nor turns a conditional source into an unconditional physical assertion.

Reproduce (Python 3.11, NumPy 2.3.5, SciPy 1.16.2, SymPy 1.14.0):

    python -m unittest -v derivations.test_gr_constants_closure derivations.test_foundational_stage2 derivations.test_spectral_circle_numeric derivations.test_ostrogradsky_witness
    python derivations/foundational_stage2.py
    python derivations/spectral_circle_numeric.py
    python derivations/ostrogradsky_witness.py
    python SM_Spectrum/audit_v18_identifiability.py --output audit/v18_identifiability.json
    python audit/prospective/freeze_gate.py --self-test

The JSON output of the full-coefficient audit is an **in-sample diagnostic**. These commands neither perform new observations nor inspect a held-out data set.

## #11 — tensor/tetrad emergence: minimal rank budget, not a GR derivation

Write the historical complex scalar as \(\psi=a+ib\). Its induced symmetric anisotropic bilinear is \(T_{\mu\nu}=\Re(\partial_\mu\bar\psi\partial_\nu\psi)=a_{,\mu}a_{,\nu}+b_{,\mu}b_{,\nu}\). Consequently \(\operatorname{rank}T\leq 2\) in a 4-dimensional tangent space. All 3-by-3 minors vanish, as explicitly checked in \`collective_rank_budget\`. A proposed multiplet of \(N\) independent **real** gradient directions \(\phi^A\), \(T_{\mu\nu}=\sum_A w_A \partial_\mu\phi^A\partial_\nu\phi^A\), has rank at most \(\min(N,4)\). Thus at least four real directions are necessary for a generic rank-4 bilinear of this particular positive-semidefinite form; taking four coordinate gradients gives an algebraic full-rank example. This is **not sufficient** to obtain an arbitrary Lorentzian metric, a dynamical tetrad, diffeomorphism gauge structure, a massless spin-2 pole, or Einstein self-interactions. Signature, weights, kinetic action and generic-source response are genuinely absent. Independent metric \(g_{\mu\nu}\) in the conditional low-energy action cannot be counted as microscopic emergence.

## #12 — a precise spectral result for a *specified* confinement operator

**Theorem (periodic magnetic circle + bounded potential).** Let \(L>0\), \(a\in\mathbb R\), and let \(W\in L^\infty(S^1_L,\mathbb R)\) with essential infimum/supremum \(w_-\leq W\leq w_+\). On the periodic domain \(H^2_{\rm per}([0,L])\) in \(L^2(S^1_L)\), define the self-adjoint, lower-bounded operator
\[
H_{a,W}=(-i\partial_s+a)^2+W(s).
\]
Let \(\lambda_j(a,W)\) denote its ordered real eigenvalues, with multiplicity, and let \(\mu_j(a)\) denote the ordered multiset \(\{(2\pi n/L+a)^2:n\in\mathbb Z\}\). Then for every \(j\geq0\),
\[
\boxed{\mu_j(a)+w_-\leq\lambda_j(a,W)\leq\mu_j(a)+w_+.}\tag{S1}
\]
For constant \(W=w_0\) both inequalities coincide and \(\lambda_n=(2\pi n/L+a)^2+w_0\) exactly (labels \(n\) are Fourier labels; ordered \(j\) must account for multiplicity). With nonnegative spectrum this implies the *conditional* separated-field mass bounds
\[
\frac{\hbar}{c}\sqrt{\max(0,\mu_j+w_-)}
\leq m_j=\frac{\hbar}{c}\sqrt{\lambda_j}
\leq\frac{\hbar}{c}\sqrt{\mu_j+w_+}.
\]
When the lower bound is negative it is only the trivial zero lower mass bound provided \(H_{a,W}\geq0\) is separately known; \(m_j\) cannot be declared real otherwise.

**Proof.** For normalized \(f\in H^1_{\rm per}\), the two quadratic forms satisfy \(q_{a,0}[f]+w_-\le q_{a,W}[f]\le q_{a,0}[f]+w_+\). Both associated periodic operators have compact resolvent. Apply Courant–Fischer min–max to the ordered eigenvalues, then the monotonicity of the square root on nonnegative inputs. For \(W\equiv w_0\), Fourier waves \(L^{-1/2}e^{2\pi ins/L}\) diagonalize the operator and saturate both bounds. QED. This is operator-specific and does **not** prove a general curvature/torsion law.

A declared geometric *model choice* such as \(W(s)=c_\kappa\kappa(s)^2+c_\tau\tau(s)^2+W_{\rm boundary}(s)\) can be inserted into S1 **only after** its self-adjoint reduction, coefficients, cross section, and geometry/connection map are derived. It would yield bounds from the essential extrema of that resulting W; curvature/torsion do not automatically equal a square-root eigenvalue. A flat connection \(a\) carries holonomy \(aL\bmod2\pi\), but equating it with curve torsion is likewise an added assumption. On a spatial circle of radius \(R\), \(\kappa=1/R>0\), while for the perfectly legitimate operator \(H_{0,0}=-\partial_s^2\) with periodic data, \(\lambda_0=0\). Thus any universal claim \(\sqrt{\lambda_0}=\langle\sqrt{\kappa^2+\tau^2}\rangle\) **fails even on the circle** unless additional mode restrictions/operator terms are declared. Changing from periodic to twisted/Dirichlet data changes the spectrum at unchanged local geometry.

\`spectral_circle_numeric.py\` checks the nonconstant Fourier-Galerkin model \(W(s)=v_0+v_1\cos(2\pi s/L)\), whose finite matrix has free diagonal \((2\pi n/L+a)^2+v_0\) and neighboring off-diagonal \(v_1/2\); its sorted Ritz eigenvalues obey the pointwise finite-matrix analogue of S1 for two lengths and nonzero holonomy. Finite-truncation errors are **not certified** and the potential is an imposed example, not an independently derived WCT confiner. This constitutes a real theorem and an explicit limitation; a general geometry theorem and nontrivial self-consistent WCT backgrounds remain #12 work.

## #13 — vacuum/length: exact scale ambiguity

Choosing \(V(u)=\frac{\lambda}{2}(u-u_\star)^2\) with \(\lambda>0\) has a minimizing amplitude \(u_\star\), but that input parameter has not been dynamically determined; varying \(u_\star\) yields a different candidate vacuum. If a dimensionless stationary eigenproblem has \(\hat\lambda_n\) and an unspecified physical conversion \(x=L_0\hat x\), then
\[
\lambda_n^{\rm phys}=\hat\lambda_n/L_0^2,\qquad m_n^{\rm phys}=(\hbar/c)\sqrt{\hat\lambda_n}/L_0.
\]
At fixed dimensionless solution, choosing \(L_0\) twice as large divides the eigenvalues by four and masses by two. The executable scale check verifies this identity symbolically. No intrinsic physical confinement length, action scale or uncalibrated numerical \(G_{\rm eff}\) follows until the dimensional input is selected by dynamics and normalization; a stable isolated vacuum and boundary/interface problem are further obligations.

## #14 — controlled quantum sector and unfixed action scale

On a **separately declared healthy reduced** oscillator with dimensionless classical action \(S_0\), consider \(S=C S_0\), \(C>0\). The classical Euler–Lagrange equations are invariant under this constant rescaling; in a path-integral expression \(e^{iCS_0/\hbar}\), the dimensionless quantum parameter is \(\hbar/C\). For a positive oscillator frequency \(\omega_n\), a canonical oscillator quantization in \(S_0\) coordinates yields \(E_{n,N}^{(0)}=(\hbar/C)\omega_n(N+\frac12)\), \(N\geq0\). This is an explicit *conditional oscillator spectrum*, not a derivation of \(\hbar\), of a mode of the untruncated corrected curvature action, or of its physical Hamiltonian. The higher-time ghost shown below blocks promoting an unconstrained naive quantization to a healthy full theory. Selecting \(C\), quantizing constrained physical modes and recovering the semiclassical/physical WCT sector remain open.

## #15 — local U(1): gauge-covariant candidate; coupling free

In natural units choose a real gauge field \(A_\mu\), representation charge \(Q\in\mathbb R\) (distinct from \(u=|\psi|^2\)), and
\[
D_\mu=\nabla_\mu-iQA_\mu,\quad
\psi\mapsto e^{iQ\chi}\psi,\quad A_\mu\mapsto A_\mu+\partial_\mu\chi,\quad
F_{\mu\nu}=\partial_\mu A_\nu-\partial_\nu A_\mu .
\]
Both \(D_\mu\psi\) and \(D_\mu D^\mu\psi\) transform covariantly. Let
\[
\Theta_A=-\frac{(D_\mu D^\mu\psi)\bar\psi}{u+\varepsilon^2e^{-2\alpha u}} .
\]
This is gauge invariant, as is the explicitly **candidate** local action
\[
S_{\rm cand}=\int\sqrt{-g}\left[
  -Z(u)g^{\mu\nu}\overline{D_\mu\psi}D_\nu\psi-V(u)
  -\frac{\beta_\Theta}{2}|\Theta_A|^2
  -\frac{1}{4g_A^2}F_{\mu\nu}F^{\mu\nu}
\right]d^4x,\qquad g_A>0.
\]
A spatial higher-derivative rail, if included, must be constructed from its actual gauge-covariant projector derivatives and variational adjoints; replacing a partial derivative symbol only in one term is insufficient. \`local_u1_covariance\` symbolically expands one- and two-covariant derivatives for an arbitrary test field, arbitrary local \(\chi(x)\), and arbitrary \(A(x)\), giving zero difference. The \(F^2\) term is gauge invariant by commuting partial derivatives. Set \(A_c=A/g_A\) so the kinetic term is \(-F_c^2/4\) and \(D=\nabla-i(Qg_A)A_c\); for a unit representation, \(\alpha_{\rm EM}=g_A^2/(4\pi)\) **only as a definition once conventions/units are fixed**. The arbitrary \(g_A\), charge representations and quantum-normalization mechanism remain unresolved. Global winding does not determine them. No numerical fine-structure constant is predicted here.

## #16 and #18 — higher-time no-go witness vs finite-frequency minimum

For the actual nonzero static background see [Q6–Q7](cross_paper/QUADRATIC_ACTION_AND_DISPERSION.md). In each decoupled real sector, put \(q=|\mathbf k|^2\), \(z=\omega^2\), \(K=F(u_0)\kappa>0\), \(G=F(u_0)\gamma\), \(T=F(u_0)\theta>0\), \(m^2=-2u_0V''(u_0)\) for amplitude and \(m^2=0\) for phase. The exact inverse quadratic kernel is
\[
\mathcal D(z,q)=y-m^2-Ky^2+Gqy-Tq^2,\quad y=z-q .
\]
The curvature-form positivity \(G^2<4KT\) **does not guarantee healthy Lorentzian poles**. Where the discriminant \(\Delta=(1+Gq)^2-4K(m^2+Tq^2)\) is positive, the two pole slopes \(\partial_z\mathcal D(z_\pm,q)=\mp\sqrt{\Delta}\) have *opposite signs*. At \(\Delta=0\) the poles collide; at \(\Delta<0\) they are not real. The nonzero background's acceleration Hessian in the full Q5 blocks is nondegenerate when \(K>0\), so simply retaining an auxiliary field without a proved new constraint cannot delete the upper mode.

An independent exact elementary canonical witness makes the obstruction explicit: \(L=\dot q^2/2-m^2q^2/2-K\ddot q^2/2\). Its acceleration Hessian is \(-K\ne0\); Ostrogradsky coordinates \(Q_1=q,Q_2=\dot q\), momenta \(p_2=-K\ddot q,\ p_1=\dot q+Kq^{(3)}\) produce
\[
H=p_1Q_2-\frac{p_2^2}{2K}-\frac{Q_2^2}{2}+\frac{m^2Q_1^2}{2}.
\]
The linear \(p_1Q_2\) term is unbounded in both directions at any \(Q_2\ne0\). This is a nondegenerate sector no-go diagnostic, not a complete Hamiltonian proof for every constrained/covariant field configuration. An admissible physical completion must change the dynamics or prove an actual degeneracy/constraint/UV prescription, and subsequently test localized stability.

For \(0<4Km^2<1\), \(B=4KT-G^2>0\), \(s=\sqrt{1-4Km^2}\), if \(G>s\), [F1–F2](cross_paper/FINITE_SCALE_SELECTION.md) derives a unique interior minimum of the *lower real amplitude oscillatory-frequency branch*:
\[
q_{\min}=\frac{G-(2K+G)\sqrt{(G^2+Bs^2)/((2K+G)^2+B)}}{B}>0.
\]
The executable rational witness \(K=10^{-3},G=2,T=2000,m^2=1\) checks \(q_{\min}<q_{\rm collision}\), the exact pole polynomial, \(z_- >0\), and opposite pole residues. With \(m^2>0,G>0\), \(\mathcal D(0,q)=-m^2-q-(K+G+T)q^2<0\) and the sum of both real \(z\) roots is positive: wherever the two distinct poles are real, both have **positive squared frequency**. Therefore this witness's finite minimum is not a real finite *growth-instability band*. At larger \(q\) where \(\Delta<0\), the full theory develops complex poles, not a selectively healthy Swift–Hohenberg growth band. The zero-vacuum curvature weight still satisfies \(F(0)=0\), so no zero-vacuum linear \(k^4\) term arises from that literal quotient-square action.

**Live model decision:** treat the corrected action as its own untruncated (currently ghost-obstructed) candidate; classify the effective Swift–Hohenberg/PFF selector as **phenomenological**, not a derived reduction. The \(G_{\rm alternative}(0)>0\) option is a *different action*, and a nonzero fourth-time Hessian would not cure ghosts by itself. No unique physical wavelength, stable parameter region, full constraint analysis, or source-rail equivalence is claimed.

## #17 — target-blind prospective execution requires external evidence

[The machine-checkable template and fail-closed gate](../audit/prospective/) require a theory-derived dimensionless interval, equation/derivation/forward-map references, immutable source-manifest digest and analysis commit, mutually disjoint opaque train/calibration/holdout IDs, all transforms/exclusions/null/alternative/multiplicity/stats/threshold/injection references, independent review and explicit prior-access attestation. The gate never reads outcomes or executes a holdout; filling its fields yields at most \`METADATA_READY_FOR_EXTERNAL_VERIFICATION_NOT_FROZEN\`, not approval. The committed template intentionally fails with status BLOCKED. Existing historical data are not repackaged as a virgin holdout. No prediction, freeze, calibration or prospective evaluation has occurred in this stage.

## #27 — full historical coefficient identifiability obstruction

\`SM_Spectrum/audit_v18_identifiability.py\` reruns the unchanged v18 source, verifies a **separate explicit branch-dispatch prediction** agrees with its original outputs, then differentiates all 29 stored coefficients plus the two historically live optimized parameters around the original result, excluding the artificial zero-mass target. The live epsilon derivative is backward one-sided at the exact upper optimizer bound. The stored coefficients are **not refit**; no new particle labels, learned indices, or use of observed mass in assigning modes is permitted. The script logs source/CSV SHA256, versions, all 11 nonzero baseline outputs, the scaled 11×31 sensitivity matrix, SVD with a declared numerical relative threshold, one-sided/central schemes, and its unavoidable **structural** local nullity at least \(31-11=20\) (regardless of numerical conditioning; global identifiability and degrees-of-freedom counting need more care). Any inferred rank is expressly local and dependent on scaling/finite differences. No 12th target is called held-out.

The checked repository tree at \`dced5cf017564c116942fc42c6b000494af328bb\` contains \`SM_Spectrum/SM_final.py\`, \`_map.py\`, \`_terms_perfection_v18.py\`, \`calculate_masses.py\`, \`resonance_map.py\` and two Wavefield_Folding CSVs, but **no v7–v9 source/CSV/XML artifacts under those names**. This is a tree-local inventory, not proof those versions never existed in Git history, external archives or another repo. The six-column Helmholtz CSV header does not state physical units, operator/domain/boundary data, eigenfunction overlaps, or an independent map to MeV. Jumps/reorderings in printed columns are diagnostics, not proven genuine branch crossings. Consequently the previously measured v18 in-sample error does not entail a WCT spectral origin, an independently fixed \(n\leftrightarrow\) particle map, or a target-blind prediction. Classification stays \`EMPIRICAL_FIT_ONLY\`; the issue may be considered for a formal *negative audit/explicit obstruction* close-out only after reviewer acceptance, never as \`SPECTRAL_BRIDGE_DERIVED\`.

## Issue status boundary

| Issue | Delivered in this stage | Remaining original acceptance |
|---|---|---|
| #11 | Symbolic minimal anisotropy rank obstruction | Derived full dynamic tensor/tetrad and weak-field GR |
| #12 | S1 circle/holonomy/ordered-spectrum theorem, equality case and nonconstant numerical test | Self-consistent general geometry/curvature/torsion operator theorem |
| #13 | Symbolic physical-scale degeneracy | Theory-native amplitude, length/action selection |
| #14 | Quantized reduced oscillator and action normalization obstruction | Quantum completion of constrained physical WCT sector; theory-native action scale |
| #15 | Full candidate gauge invariant curvature quotient/action and symbolic D,D² identity | Independently normalized/predicted charge and \(g_A\), symbolic full-action/matching audit |
| #16 | Explicit Hessian/pole/Ostrogradsky unbounded canonical witness | Actual healthy constrained/UV completion and localized stability |
| #17 | Fail-closed executable metadata gate | Independently derived, registered, calibrated, untouched executed holdout |
| #18 | Exact model classification and real-pole test, distinct phenomenological rail | Healthy physical mechanism and parameter/cutoff/rail derivation |
| #27 | All-31 local sensitivity, structural nullity, CSV header and missing-asset inventory | Independent operator/index/MeV mapping or approved explicit obstruction close-out |

**No original acceptance criteria or historical frozen claims have been silently checked off.** A passing self-test measures implementation consistency, not scientific completion.
