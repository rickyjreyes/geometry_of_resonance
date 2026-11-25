# Empirical target ledger

This synthesis freezes **source estimands, not four interchangeable numbers**. “Primary” means the priority chosen in the published audit protocol. It does not make an exploratory source result prospective. Exact decimal tokens are stored as strings in [cross_domain_targets.json](../../audit/cross_domain/cross_domain_targets.json); calculations use those tokens at 80-digit precision. Displayed derived values may be shortened. Source precision is not a statistical uncertainty.

| Domain | Exact primary source value | Mathematical object and coordinate | Actual observable | Original status |
|---|---|---|---|---|
| GWTC | **9.602325620315224** | Angular log-frequency, dimension 1; ell=ln(M_chirp/M_sun) | Source-frame chirp-mass population residual | Training-selected, declared fixed for catalog holdout; public artifact chronology caveat below |
| CMS | **7.025825825825827** | Angular log-frequency, dimension 1; ell=ln(m_mumu/GeV) | Inclusive opposite-sign dimuon mass residual | Exploratory H-file selection followed by fixed-frequency file and period tests |
| NIST Fe II | **31.3265306122449** | Angular log-frequency, dimension 1; ell=ln(spectroscopic wavenumber/(cm^-1)) | Density of unique transition lines, not photon flux | Canonical exploratory scan; same maximum at 120/160/200 bins |
| LHCb | **1.2282873602374818** | Dimensionless zero-intercept regression slope a | Selected signal-region winding triplet versus B-low sideband triplet | Exploratory triplet selection and scale fit; **not a DSI period** |

These three frequencies are coefficients of an angular phase, not cycles per unit log coordinate; dividing by 2pi gives cycles per unit log coordinate. NIST's fitted k is conjugate to the log of a spectroscopic wavenumber. It is not the spectroscopic wavenumber itself and not the spatial k_star of S004. None of the four primary values is an observed particle mass, root mass, radial eigenvalue, or Goldstone phase gradient.

## Exact source pins

- **GWTC_V1:** [E008: tables/gwtc_frozen_mode.json](https://github.com/rickyjreyes/GWTC/blob/bf95826c93ba11001f3461b653e3f279b4799573/tables/gwtc_frozen_mode.json); commit `bf95826c93ba11001f3461b653e3f279b4799573`.

- **CMS:** [E061: docs/CMS_RUN2016G_FILE2_PHASE_LOCK_RESULT_2026-08-31.json](https://github.com/rickyjreyes/wct-cms/blob/82a007c4f81691b522cad7611c0b711a0ff94163/docs/CMS_RUN2016G_FILE2_PHASE_LOCK_RESULT_2026-08-31.json); commit `82a007c4f81691b522cad7611c0b711a0ff94163`.

- **NIST_FE160:** [E040: outputs/fe_ion2_160/nist_summary.json](https://github.com/rickyjreyes/NIST/blob/4ddc3948c1e119f463e5d3d49bda1814ed195a84/outputs/fe_ion2_160/nist_summary.json); commit `4ddc3948c1e119f463e5d3d49bda1814ed195a84`.

- **LHCB_A:** [E022: lhcb_wct_analysis_kit/outputs_wct_cross_region_scaling/cross_region_scaling_summary.json](https://github.com/rickyjreyes/LHC/blob/5fc36d638e0ba1b03e236d6a11547b29966bbe8b/lhcb_wct_analysis_kit/outputs_wct_cross_region_scaling/cross_region_scaling_summary.json); commit `5fc36d638e0ba1b03e236d6a11547b29966bbe8b`.


Every repository file used is listed with its Git blob SHA1 and SHA256 in [SOURCE_LEDGER.json](../../audit/cross_domain/SOURCE_LEDGER.json). The code/result takes precedence over a rounded paper table. Source repositories were read, not edited.

## GWTC variants and freeze chronology

The original executable frozen-mode artifact uses `GWTC_STRICT_FROZEN_MODE_V1`; the result is `GWTC_STRICT_FROZEN_HOLDOUT_V1`. Later narrative stage labels do not change those IDs. Its 173 training objects select k=9.602325620315224; the 104-object retained catalog evaluation reports delta D=10.035425115602617 and add-one p=0.0012998700129987 under its declared null. These are a test statistic and a tail fraction, not frequency error bars.

Public Git chronology is weaker than a pre-result artifact timestamp: the training-only code was committed at 2026-08-26 21:06:07Z, the holdout result at 21:19:01Z, and the frozen-mode JSON at 21:19:39Z. The manifest says the holdout was unevaluated at freezing, but its published bytes were committed after the result. This does not prove leakage; it prevents a stronger chronology claim based on Git alone.

| Separate pipeline | Exact frequency | Meaning |
|---|---:|---|
| V3 Gaussian KDE | 9.794117647058822 | Training-selected mode; same holdout had already been inspected; source labels this robustness, not independent replication |
| V4 structured null / published future target | 9.7 | Exactly imposed target; not a rounding replacement for V1 or V3 |

V3's bandwidth is 0.19675997038948742, selected on training data. V4 explicitly calls its exercise `ROBUSTNESS_CHALLENGE_NOT_INDEPENDENT_REPLICATION`. Its future acceptance band [9.5,10.0] is not a measured confidence interval. No untouched future-catalog prediction is evaluated here.

The source training support has ell_min=0.17058630057553367, ell_max=4.61512051684126 and width 4.444534216265726. Its reported n=6.792393139583111 is a cycle count computed from k and that support, not an independently selected topological integer.

## CMS observable, phase and support

The frozen candidate uses 2–120 GeV, 350 log bins, pT>=4 GeV, |eta|<=2.4, tight muons, degree-7 robust Chebyshev background, and exclusions 2.9–3.3, 3.55–3.85, 8.5–11.5, 80–100 GeV. The August 28 freeze precedes later file tests. The Run2016G file-2 phase/sign freeze uses phi=-0.2313916852932179 in A cos(omega ell-phi), A>0; this phase came from the preceding G file. Its result gives delta chi2=126.28323995422409, positive amplitude 0.9708617746009612, refit-bootstrap p=0.001996007984031936 (500 draws), and permutation p=0.000999000999000999 (1000 draws). G2 is a file holdout within G, not another detector or period. Background flexibility remains an explicit limitation.

The discovery phase used for the separate CMS→LHCb test is -0.1889538223 in the freeze convention, not the G2 phase. The coordinate conversion is k_q2=omega_m/2. Exact decimal division gives **3.5129129129129135**; the LHCb implementation serializes **3.5129129129129133**. Both are preserved. Their 2e-16 difference is a floating-point representation issue, not a physical discrepancy.

The reported discovery cycle count 4.565194460725239 uses the span of retained bin centers (`src/cms_wct/analysis.py`, x=log(centers[mask]/m0), then max(x)-min(x)). The complete endpoint span ln(120/2) instead gives 4.57827524714566 cycles. The nominal sum of the continuously retained masked intervals gives 3.75555026019519. These are different analysis conventions; none is a derived physical closure domain.

## NIST Fe II and Co II

Fe II has 9447 unique wavenumbers from 20615 raw rows. The 160-bin record runs from physical wavenumbers 5721.4978 to 49990.73 cm^-1. The stored ell endpoints 8.658759675005578 and 10.812819095459146 are **first/last bin centers**; the scanner's width 2.1540594204535672 consequently changes with bin count. The 120/160/200 results are rebinnings of the same lines, not independent experiments. All three have zero exceedances in 5000 canonical Poisson scan-null draws, whose add-one floor is 1/5001.

| Within-domain control | Exact k | Interpretation |
|---|---:|---|
| Co II, 120 bins | 30.24489795918367 | Exploratory neighbor scan, 2450 unique lines |
| Co II, 160 bins | 30.24489795918367 | Primary Co comparison within the atomic domain |
| Co II, 200 bins | 30.213085234093636 | One scan-grid step different |

Co II at 160 bins differs from Fe II by about -3.453%; they are qualitatively similar, not identical estimates of a demonstrated invariant. Fe's n≈10.74 and Co's n≈10.97 do not establish a common independently selected n=11.

The later R audit matters: broader bin-grid stability is inconclusive, observed/Ritz line sets share underlying transitions, and a separate training/held-out model comparison has held-out log likelihood -560.677192339407 for the harmonic model versus -523.841554349368 for the smooth model. The gain is **negative**, about -36.836. That split selected/locked k=31.9627851140456, not the exact canonical k, so it is a transfer limitation of the audited pipeline, not a new test performed here or a direct falsification of the canonical freeze.

No calibrated sigma_k or confidence interval for these primary frequency estimands is supplied in the cited records. Stability across a selected bin ladder, a grid step, a p-value, or an injection recovery rate cannot be substituted for sigma_k.

## LHCb: preserve distinct quantities

The primary a is exactly (n_A dot n_B)/(n_A dot n_A) for the ordered triplets:

- A: (9.814120755431825, 14.56141637666396, 19.01961386711593), from k=(12.9,19.14,25.0).
- B: (12.142121492766814, 17.634985977589896, 23.508242739755296), from k=(15.96,23.18,30.9).

The fit RMSE 0.17511381200126633 is in winding units; **it is not sigma_a**. The original May 13 paper §15.2 does report a separate 2000-draw well-resampling bootstrap, reselecting best-Koide triplets: approximate median 1.228, standard deviation 0.36 and reported one-sigma band [0.82,1.45], with a heavy-tailed distribution. That is a conditional well-selection uncertainty, not a Gaussian experimental precision or a frequency interval. The scale-fit null fraction 0.00585997656009376 does not establish phase locking: p_phase=0.45298618805524776 and the reported joint affine/phase fraction is 0.06791572833708664.

| Other LHCb source quantity | Exact value | Role retained in this audit |
|---|---:|---|
| Dominant low-frequency reference k1 | 7.61054 | Documented in May paper; fixed input to later two-mode analysis; supplementary frequency stress panel only |
| Historical high-frequency reference k2 | 19.5296 | Different reference mode; sensitivity only |
| Later bounded two-mode scan optimum k2 | 23.08 | Exploratory local scan optimum; amplitude cap active |
| Imported CMS test frequency | 3.5129129129129133 | Imported input, **not an independent LHCb frequency measurement** |

The paper and two-mode manifest preserve the first two literals, but do not furnish an independent interval or additional digits for their original estimation. They cannot replace the primary a result. All source-choice panels are reported rather than choosing the most favorable mode.

The exclusive candidate spectrum is B0→K*0 mu+mu-, with ell=ln(q2/GeV^2), q2=m_mumu^2. Retained intervals are (0.1,8), (11,12.5), (14.5,19) GeV^2; their total log measure is 4.780150335923678. This disconnected analysis support is not automatically one physically glued loop.

The August 31 fixed CMS frequency/phase/positive-sign test gave **A_hat=0, q_locked=0, p=1** (10000 null draws). The post-unblinding two-quadrature diagnostic at the same frequency gave score 398.52848531, zero exceedances in 10000 draws and fitted phase -1.9757308469, displaced by -2.1646846692 rad from the frozen phase. Its coefficient safety bound was active. Thus the unchanged waveform failed in the specified pipeline; the frequency is not proven absent. The followup is exploratory and boundary-limited.

## Paper/source discrepancies

The October 6 2–3–6 paper prints a=1.228287431; the exact pinned LHCb output gives 1.2282873602374818. This audit uses the latter without treating their difference as uncertainty. The NIST paper's minus-phase cosine and atan2(-b,a) convention are internally inconsistent as written; no NIST phase transfer is attempted, and its quoted frequency is unaffected. Unreadable later LHCb Library copies were not assumed identical to the readable May 13 paper. Repository result artifacts remain primary.
