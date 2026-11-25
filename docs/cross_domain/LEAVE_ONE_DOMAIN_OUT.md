# Leave-one-domain-out prediction

Every usable low-parameter stress model was evaluated in all four held-out directions. These are **retrospective** tests: the source targets were historically known. They do not constitute prospective validation or action-derived predictions.

## Primary historical-anchor panel

| Family | GWTC held out | CMS held out | NIST held out | LHCb-a held out |
|---|---|---|---|---|
| U0 | No valid three-domain frequency training set | Same | Same | No output map to a |
| U1 | No map converting training a to frequency | Same | Same | No output map to a |
| U2 | Independent width/sector missing | Same | Same | Same plus slope map missing |
| U3 | Physical S004-to-observable map missing | Same | Same | Same |
| U4 | Full physical chain incomplete | Same | Same | Same |

These 20 explicit entries are `null` predictions and residuals in `cross_validation_results.json`, with reasons. They are not zero residuals or skipped failures. No physical four-domain mechanism passed the eligibility requirements. Manufacturing a frequency from a would invalidate the test.

## Supplementary source-existing reference-frequency panel

The fourth entry here is **LHCb k1=7.61054**, a historical low-frequency reference fixed in the later two-mode model, not the primary slope a and not an independent newly measured frequency. This panel supplies a mathematically valid stress test of U0/U1 while preserving that limitation. CMS-imported 3.5129129129129133 is excluded as a fourth independent target.

For each fold, convert only by the declared factors c_i, train on the other three, freeze

\[
\hat\theta_{-i}=\exp\left[\frac13\sum_{j\ne i}\ln(c_j y_j)\right],
\qquad \hat y_i=\hat\theta_{-i}/c_i,
\]

then compute signed, absolute and fractional errors. This minimizes squared log-ratio error without inventing observational variances. U0 uses c=(1,1,1,1); U1 uses c=(1,1,1,2). The map, frequency source and held-out label are never reselected within a fold. The arithmetic starts from exact source decimal strings at 80-digit precision. There is no legitimate common sigma model, so no sigma-normalized frequency residual is reported.

### U0 — equal native dimensionless log-frequency
| Held-out reference | Observed native frequency | Prediction in native coordinate | Absolute error | Fractional error |
|---|---:|---:|---:|---:|
| GWTC_V1 | 9.602325620315224 | 11.8761329462888 | 2.27380732597358 | +23.679757% |
| CMS | 7.025825825825827 | 13.1795748400888 | 6.153749014263 | +87.587554% |
| NIST_FE160 | 31.3265306122449 | 8.0074904069054 | 23.3190402053395 | -74.438630% |
| LHCB_K1 | 7.61054 | 12.8330150230192 | 5.22247502301919 | +68.621609% |


U0 is a hypothesis about source-native numbers, not the physically required common coordinate. It is included because requested, not used to obscure the q2-versus-mass distinction.

### U1 — one scale in mass/energy-log coordinates
| Held-out reference | Observed native frequency | Prediction in native coordinate | Absolute error | Fractional error |
|---|---:|---:|---:|---:|
| GWTC_V1 | 9.602325620315224 | 14.9629898903793 | 5.36066427006407 | +55.826729% |
| CMS | 7.025825825825827 | 16.6052237696928 | 9.57939794386694 | +136.345508% |
| NIST_FE160 | 31.3265306122449 | 10.0888057204914 | 21.2377248917535 | -67.794692% |
| LHCB_K1 | 7.61054 | 6.41650751150959 | 1.1940324884904 | -15.689195% |


No one of these folds is an accurate equality prediction at the precision of the frozen references. NIST breaks numerical equality in both panels; CMS has the largest U1 fractional error. Without calibrated frequency errors, these are literal predictive mismatches, not a Gaussian-significance exclusion of an unknown true frequency.

## All source-choice sensitivities, without selecting a winner

Three GWTC values × Fe/Co 160 × three LHCb mode references/optima = 18 target panels. Both U0 and U1 have all four folds: **36 fits and 144 held-out predictions**. The choices are correlated pipelines/species/modes, not 18 replications. The high-k LHCb results describe a different component, so they cannot silently replace k1 or a. Full 80-digit predictions and training IDs are in the JSON.

| GWTC variant | Atomic target | LHCb target | Model | GWTC error | CMS error | Atomic error | LHCb error |
|---|---|---|---|---:|---:|---:|---:|
| GWTC_V1 | NIST_FE160 | LHCB_K1 | U0 | +23.679757% | +87.587554% | -74.438630% | +68.621609% |
| GWTC_V1 | NIST_FE160 | LHCB_K1 | U1 | +55.826729% | +136.345508% | -67.794692% | -15.689195% |
| GWTC_V1 | NIST_FE160 | LHCB_K2REF | U0 | +69.326363% | +156.820673% | -65.004671% | -34.289412% |
| GWTC_V1 | NIST_FE160 | LHCB_K2REF | U1 | +113.337849% | +223.573772% | -55.908648% | -67.144706% |
| GWTC_V1 | NIST_FE160 | LHCB_K2BEST | U0 | +79.021585% | +171.525610% | -63.000922% | -44.397682% |
| GWTC_V1 | NIST_FE160 | LHCB_K2BEST | U1 | +125.553064% | +242.100832% | -53.384083% | -72.198841% |
| GWTC_V1 | NIST_CO160 | LHCB_K1 | U0 | +22.239593% | +85.403230% | -73.524492% | +66.658130% |
| GWTC_V1 | NIST_CO160 | LHCB_K1 | U1 | +54.012237% | +133.593432% | -66.642950% | -16.670935% |
| GWTC_V1 | NIST_CO160 | LHCB_K2REF | U0 | +67.354677% | +153.830178% | -63.753151% | -35.054565% |
| GWTC_V1 | NIST_CO160 | LHCB_K2REF | U1 | +110.853681% | +219.805984% | -54.331832% | -67.527283% |
| GWTC_V1 | NIST_CO160 | LHCB_K2BEST | U0 | +76.937006% | +168.363887% | -61.677743% | -45.045132% |
| GWTC_V1 | NIST_CO160 | LHCB_K2BEST | U1 | +122.926658% | +238.117310% | -51.716982% | -72.522566% |
| GWTC_V3 | NIST_FE160 | LHCB_K1 | U0 | +21.257814% | +88.828257% | -74.269567% | +69.736871% |
| GWTC_V3 | NIST_FE160 | LHCB_K1 | U1 | +52.775272% | +137.908696% | -67.581686% | -15.131564% |
| GWTC_V3 | NIST_FE160 | LHCB_K2REF | U0 | +66.010552% | +158.519283% | -64.773212% | -33.854803% |
| GWTC_V3 | NIST_FE160 | LHCB_K2REF | U1 | +109.160188% | +225.713886% | -55.617028% | -66.927401% |
| GWTC_V3 | NIST_FE160 | LHCB_K2BEST | U0 | +75.515919% | +173.321478% | -62.756210% | -44.029929% |
| GWTC_V3 | NIST_FE160 | LHCB_K2BEST | U1 | +121.136200% | +244.363484% | -53.075765% | -72.014964% |
| GWTC_V3 | NIST_CO160 | LHCB_K1 | U0 | +19.845852% | +86.629485% | -73.349383% | +67.760405% |
| GWTC_V3 | NIST_CO160 | LHCB_K1 | U1 | +50.996312% | +135.138417% | -66.422327% | -16.119797% |
| GWTC_V3 | NIST_CO160 | LHCB_K2REF | U0 | +64.077476% | +155.509009% | -63.513414% | -34.625017% |
| GWTC_V3 | NIST_CO160 | LHCB_K2REF | U1 | +106.724666% | +221.921178% | -54.029783% | -67.312508% |
| GWTC_V3 | NIST_CO160 | LHCB_K2BEST | U0 | +73.472160% | +170.138843% | -61.424280% | -44.681661% |
| GWTC_V3 | NIST_CO160 | LHCB_K2BEST | U1 | +118.561226% | +240.353615% | -51.397638% | -72.340830% |
| GWTC_V4 | NIST_FE160 | LHCB_K1 | U0 | +22.434360% | +88.221454% | -74.352253% | +69.191418% |
| GWTC_V4 | NIST_FE160 | LHCB_K1 | U1 | +54.257628% | +137.144171% | -67.685863% | -15.404291% |
| GWTC_V4 | NIST_FE160 | LHCB_K2REF | U0 | +67.621327% | +157.688526% | -64.886414% | -34.067361% |
| GWTC_V4 | NIST_FE160 | LHCB_K2REF | U1 | +111.189638% | +224.667198% | -55.759654% | -67.033681% |
| GWTC_V4 | NIST_FE160 | LHCB_K2BEST | U0 | +77.218923% | +172.443155% | -62.875894% | -44.209790% |
| GWTC_V4 | NIST_FE160 | LHCB_K2BEST | U1 | +123.281852% | +243.256865% | -53.226557% | -72.104895% |
| GWTC_V4 | NIST_CO160 | LHCB_K1 | U0 | +21.008699% | +86.029748% | -73.435025% | +67.221304% |
| GWTC_V4 | NIST_CO160 | LHCB_K1 | U1 | +52.461407% | +134.382795% | -66.530229% | -16.389348% |
| GWTC_V4 | NIST_CO160 | LHCB_K2REF | U0 | +65.669495% | +154.687926% | -63.630665% | -34.835100% |
| GWTC_V4 | NIST_CO160 | LHCB_K2REF | U1 | +108.730485% | +220.886679% | -54.177509% | -67.417550% |
| GWTC_V4 | NIST_CO160 | LHCB_K2BEST | U0 | +75.155334% | +169.270747% | -61.548244% | -44.859427% |
| GWTC_V4 | NIST_CO160 | LHCB_K2BEST | U1 | +120.681893% | +239.259882% | -51.553823% | -72.429714% |


## Why integers and the partial geometric analogy do not get artificial LOO passes

Unknown held-out sectors cannot be selected using the fourth target after fitting three. The negative-control harmonic searches therefore have **no eligible physical LOO result**. Similarly, lambda_CMS=2 lambda_Fe has only two specified frequency outputs, and the LHCb geometric comparator lacks the DSI-to-regional-slope bridge. No GWTC assignment was supplied. Their partial residuals are reported in `derived_quantities.json`, not called four-domain predictions.

The already executed source CMS→LHCb test is retained as a separate two-domain retrospective transfer. Its unchanged phase/sign template failed (q=0, A_hat=0, p=1). A subsequent free-phase diagnostic was positive at the imported frequency, but was post-unblinding and boundary-limited. That prevents conflating failure of the exact waveform with proof that the frequency is absent.
