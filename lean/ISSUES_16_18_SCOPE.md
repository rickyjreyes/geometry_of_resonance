# Geometry issues #16 and #18: Lean proof scope

**Status:** kernel-proof candidates in this PR. CI must pass before treating the files as Lean-checked. The physical closure questions remain **OPEN**.

## #18: finite wavelength
- `R18_FiniteScale.lean` formalizes the static S004 Q7 discriminant identity, residue-slope algebra, and a sufficient large-`q` condition for negative discriminant (complex poles).
- The independent spatial-only rail `omega^2(q) = gap + bending*(q-qStar)^2`, for `gap>0, bending>0, qStar>0`, has an exact, unique finite-`q` minimum and positive squared frequency.
- A **separate** dissipative growth rail has a positive finite-wave-number band center. This is not derived from the S004 variational action.

## #16: higher-derivative health
- `R16_GhostObstruction.lean` proves positivity of the corrected nonzero-background weight and positive acceleration coefficient for `kappa>0`.
- The explicit fourth-time-order Ostrogradsky Hamiltonian is unbounded above/below in its canonical affine momentum direction when `q2 != 0`.
- A mixed auxiliary kinetic form is also indefinite on an explicit one-dimensional section.
- For an **independently defined spatial-only** quadratic rail, Lean proves acceleration-independence and a nonnegative linear-mode Hamiltonian when `gap>0` and `bending>=0`.

## What is *not* proved
1. The spatial-only rail is **not** mathematically equivalent to corrected S004.
2. There is no independently selected S004 background, coefficient set, length or physical cutoff.
3. No full S004 nonlinear positive-energy theory is established, and no localized spectrally or orbitally stable S004 solution is proved.
4. The full field-theoretic Dirac count, complex-frequency growth-rate asymptotics and Sobolev ill-posedness remain documented analytical results, **not** in these Lean files.
5. Finite oscillatory frequency minimum, finite growth band and stable Hamiltonian are different statements; these proof modules do not identify them.

Grounding: `docs/cross_paper/QUADRATIC_ACTION_AND_DISPERSION.md` (Q6–Q7), `docs/cross_paper/FINITE_SCALE_SELECTION.md` (F1–F2), and `docs/foundational/GHOST_AND_CONSTRAINT_ANALYSIS.md` (H2/H4).

Do **not** close issues #16 or #18 based only on these Lean modules.
