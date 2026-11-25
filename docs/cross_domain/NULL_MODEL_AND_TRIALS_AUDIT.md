# Null models and selection freedom

The primary GWTC/CMS/Fe/a panel is **ineligible for a four-frequency null** because a is a regression slope. No fourth frequency was fabricated from a. The following calculations apply only to the separately identified source-existing LHCb-k1 reference panel.

These are manufacture-of-agreement diagnostics, **not scientific p-values**. The distributions log-uniform[1,100] and log-uniform[5,50] were fixed in the protocol as arbitrary support choices. They do not model merger populations, dimuon production, atomic transitions or LHCb selection. Historical experiment, peak, pipeline and publication choices are not jointly calibrated.

## Frozen procedure

Seed 20261007, NumPy PCG64, 10000 independent four-tuples per support. Fit a shared log scale after the same allowed diagnostic selection on each observed and synthetic tuple:

\[
\nu_i/(|p_i| n_i^d)=\theta,\qquad d\in\{+1,-1\}.
\]

The harmonic/subharmonic direction is one whole-model choice; per-domain n ranges from 1 to M, with M=6,12,24. The n=1 case is retained as the no-integer-search baseline. Exponent menus are {1}, {1/2,1,2}, and {1/2,1,2,-1}. Both native and source-required common mass/energy panels are retained. Reciprocal exponents change orientation and duplicate magnitude fits. None of the freely selected exponents or integers is allowed as physical evidence.

The score is RMS log residual after fitting the shared scale. The omnibus statistic selects the minimum over every cap, menu, direction and coordinate panel, and repeats that complete selection on every null tuple. A global one-dimensional boundary-sweep algorithm finds the same optimum as full enumeration; verification compares it against exhaustive searches at small caps. No empirical event data or peak-search script is executed.

## Results

The selected observed score is **0.009996201745556049** in RMS log units, obtained in the native M=24 power-menu harmonic family. This apparent roughly one-percent RMS alignment is an optimized negative control, not a derived prediction.

| Synthetic support | Null quartets at least as close after the same full selection | Add-one diagnostic fraction | Monte Carlo standard error |
|---|---:|---:|---:|
| [1, 100] | 7319 / 10000 | 0.731926807319 | 0.004430 |
| [5, 50] | 9918 / 10000 | 0.991800819918 | 0.000902 |


The strong support dependence is part of the result. Even the broad support makes equally good fitted agreement common. A small optimized residual by itself does not discriminate a common physical origin.

| Procedure on native reference frequencies | Observed RMS log residual | Diagnostic fraction [1,100] | Diagnostic fraction [5,50] |
|---|---:|---:|---:|
| No integer or exponent search | 0.601712208584 | 0.113488651135 | 0.591140885911 |
| Harmonic n=1..12 only | 0.0430771180182 | 0.240375962404 | 0.757224277572 |
| n=1..12 plus three exponents | 0.0143179769775 | 0.198380161984 | 0.465453454655 |


These rows are conditional family diagnostics. Choosing the most attractive row afterward requires the omnibus selection, not its isolated fraction. The full 96 family/support rows are preserved in `null_diagnostics.json`, including fitted labels explicitly marked negative-control outputs.

## Parameter and trials accounting

| Freedom | Accounting |
|---|---|
| U0/U1 shared scale | One continuous parameter; no fitted dataset multipliers or labels |
| Primary physical integer search | Empty admissible range; zero trials, because independent assignments absent |
| A negative-control family with M integers and J exponents | (M J)^4 nominal label tuples, one fitted continuous scale |
| M=12, J=3, one direction | 1,679,616 nominal tuples |
| Full negative-control cap/menu/direction/panel inventory | 478,349,768 nominal model-label entries, **not independent tests** |
| Reciprocal choice | Duplicate frequency-magnitude fit; phase orientation still distinct |
| Common units or normalization | Unit change shifts log phase; one common frequency multiplier is absorbed by fitted theta |
| Four independent continuous multipliers/widths | Saturated output rank 4; exact fit for every positive quartet, analytically; no fit performed |
| Seven geometric norms assigned to four domains | Up to 2401 assignments before coordinate/harmonic choices; not a calibrated physical null |
| Alternative empirical pipelines/modes | 18 reference panels all disclosed; selection among them is not calibrated by the four-number null |

The nominal count overstates independent trials because menus and caps are nested, choices duplicate products p n, reciprocal signs coincide in magnitudes, and common rescalings are absorbed by theta. An exact independent-trials count is not available or needed for the stated diagnostic: Monte Carlo repeats the whole search. It is not converted into a Bonferroni factor or combined with source p-values.

The source experiments have separate conditional nulls and dependencies. Rebinnings of Fe II, observed/Ritz variants, CMS files sharing detector systematics, and LHCb sidebands or reused events cannot be multiplied into independent confirmation. The source LHCb triplet bootstrap incorporates reselecting wells only within its declared procedure; it is not a four-domain look-elsewhere correction. No cross-domain discovery significance or Bayes factor is claimed.
