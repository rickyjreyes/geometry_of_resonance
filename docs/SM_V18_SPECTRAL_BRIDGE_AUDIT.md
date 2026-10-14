# SM v18 spectral-bridge audit: baseline and obstruction

**Status as of 2026-10-14: `EMPIRICAL_FIT_ONLY` for the *current evidence*, not a theorem that no spectral bridge can exist.** Implementation handoff for [#27](https://github.com/rickyjreyes/geometry_of_resonance/issues/27). The executable [audit_v18.py](../SM_Spectrum/audit_v18.py) loads the unchanged v18 program, records its baseline and optimizer, preserves zero-target behavior, and computes a local **two-free-parameter** sensitivity diagnostic. Run in an environment with NumPy and SciPy; its JSON records versions and the exact source SHA256. It does not refit the 29 stored coefficients or produce a holdout.

## What is already established, and what needs deriving

The source `SM_Spectrum/_terms_perfection_v18.py` stores **9 light + 7 middle + 13 heavy = 29** hand-selected coefficients. The execution additionally optimizes `alpha_decay` and `epsilon_cutoff` on **the same 12 targets**, including (n=0) with denominator (10^{-6}). There are **11 nonzero** target masses. The ratio of 31 nominal fitted/stored scalar coefficients to 11 nonzero target masses is 31/11 ≈ 2.82; this is a parameter-count warning, **not** a proven full-rank model dimension, because coefficients are stored, routing is piecewise and active sensitivities may vanish. A Jacobian only for the two current free variables cannot establish identifiability of all historical coefficients.

The issue's prior 2026-10-07 reported reproduction found alpha about 5.172385 and epsilon=5 (upper bound), mean target-relative loss about 1.853%, with bottom ≈17.02% relative error. These are prior *in-sample* reported values, not new outputs manufactured by this handoff. The script prints exact current results on execution; retain result.success, message, gradients/evaluation counts, full per-target residuals, and bound hits. Its output states whether an exponential or mass floor clips, and excludes the artificial zero target from nonzero-only summary metrics.

| v18 term / operation | What the code actually defines | Status relative to #12 |
|---|---|---|
| Harmonic n/(n+1) | index interpolation, not `sqrt(lambda_n)` of a specified operator | Empirical |
| Modulo-three sign, log(1+n), n², n³ | chosen index functions without prescribed domain, wave operator or boundary conditions | Motivated labels / empirical coefficients |
| Exponential hierarchy and n⁻² chirality | stored coefficients, explicit exponent clipping and n=0 branch | Empirical |
| Logistic suppression with optimized alpha, epsilon | two parameters trained on listed target masses; epsilon has upper bound 5 | Empirical, in sample |
| n=3 neutrino multiplier | special target-specific branch | Empirical |
| Middle sine/harmonic/offset/quadratic | seven stored coefficients and clipping at 1e−12 | Empirical |
| Heavy quark/boson power laws, W/Z sine break and Higgs correction | thirteen stored coefficients and mass/index-dependent routing | Empirical, target-specific |
| Index n, n/k; MeV conversion | neither a geometric eigenmode labeling map nor a calibrated operator-to-MeV normalization is supplied | Missing bridge |

**No target-blind particle assignment exists here.** `SM_Spectrum/_map.py` uses different heavy coefficients and nearest-observed-mass labeling; it is **not v18** and cannot independently validate v18. `SM_Spectrum/SM_final.py` anchors electron/up/neutrino directly. `SM_Spectrum/calculate_masses.py` references a local `photon/Photon/results/wct_harmonics.csv` that is not in the inspected repository tree. `Wavefield_Folding/cone_helmholtz_spectrum.csv` contains `k2_m0` through `k2_m5`, but units, boundary conditions, branch tracking across crossings and map from k² to physical MeV must be established before using it as a spectrum derivation.

The formal separated-field relation in [DERIVATION_CLOSURE.md](../DERIVATION_CLOSURE.md), (m_n=(\hbar/c)\sqrt{\lambda_n}), holds under its own explicitly separated eigenproblem. This identity does **not** make a fitted assignment of integers to the electron, quarks or bosons a consequence of that eigenproblem. An eigenmode test must freeze the operator, physical units and boundaries, continue mode identities through crossings by overlap/symmetry rather than nearest mass, and predict held-out labels *before* examining their measurements.

## Auditable next acceptance gates

1. Execute `python SM_Spectrum/audit_v18.py --output audit/v18_baseline.json` and attach the generated artifact and dependency/source hashes, optimizer/clipping/boundary notes. Do not check in a claimed numeric reproduction without running the script.
2. Recover and hash any original v7–v9/CSV/XML assets, comparing formulas and parameter counts without silently substituting v18. Missing assets remain missing.
3. Specify confinement operator, its domain, boundary conditions, eigenvalue units, branch identity and a *preobservational* index assignment. Derive a mapping to MeV including independently sourced (\hbar,c,L) or state external calibration.
4. Perform full active-coefficient rank/SVD and condition analysis, sensitivity under branch switches and alternative baseline models of comparable flexibility; then freeze train/test split, indices, hyperparameters and decision threshold before a truly held-out evaluation.

**Decision:** This is an identified obstruction, not negative proof against all possible WCT eigenmode interpretations. The small in-sample fit error cannot be described as independent physical prediction. Keep #27 open until an auditable independent spectral mapping or an explicitly accepted obstruction is reviewed.
