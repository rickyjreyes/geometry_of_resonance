# Source provenance and scope

The audit uses immutable repository commits and identified original paper versions. All 62 repository sources were verified against their Git blob hashes and SHA256. The preceding 22-paper/48-repository-file theory audit remains available at commit caba0a5d034b1a347e60ffcd36dd7302f2c3e10d; its full source inventory is referenced rather than duplicated. The new search covered scale-winding, DSI, S004/PFF, radial shells, root-mass and dispersion/rest-energy maps, 2–3–6 geometry, winding/harmonic assignments, and explicit empirical cross-domain comparisons.

The coverage claim is the indexed available WCT corpus and retrieved governing sources, not every unpublished or unindexed document. Later unreadable LHCb copies are recorded in the machine ledger. Their contents were not assumed; the readable original May 13 paper and pinned executable/result files carry the claims used here.

## Repository pins

| Repository | Exact commit | Source head timestamp |
|---|---|---|
| GWTC | `bf95826c93ba11001f3461b653e3f279b4799573` | 2026-09-02T09:27:36Z |
| wct-cms | `82a007c4f81691b522cad7611c0b711a0ff94163` | 2026-09-07T21:42:33Z |
| NIST | `4ddc3948c1e119f463e5d3d49bda1814ed195a84` | 2026-09-20T02:00:07Z |
| LHC | `5fc36d638e0ba1b03e236d6a11547b29966bbe8b` | 2026-09-01T01:19:42Z |

## Paper retrievals

| Source | Version identifier | Extracted coverage used |
|---|---|---|
| GWTC_Prediction_with_Figures.tex | libfile_214f6d5a814c8191ad0555ed8a2b26e7 | All 1833 extracted lines |
| WCT_CMS_Cross_Period_Replication_2026-08-28(2).tex | libfile_00d6dee010548191b3f436d06b6e810b | All 1739 extracted lines |
| Bin-Stable Log-Periodic Structure in Public NIST Atomic Line List.pdf | libfile_ae9e6c2fff108191871b401d5ab1e583 | All 462 extracted lines; internal May 28 date |
| LHC_paper.pdf | libfile_432b515f34e48191bf2e71ecfd9fd71c | All 881 extracted lines; internal May 13 date, original release May 9 |
| 2–3–6 phase/window-covariance upgrade, P05 | libfile_7081ef2d175c8191a48967bca82d5674 | Prior full 926-line retrieval plus fresh governing excerpt 519–663; October 6 revision |

Retrieval-window hashes and Library identifiers are in SOURCE_LEDGER.json. An extracted-text hash does not claim the original binary hash. Internal manuscript dates and upload dates are distinct; no later revision is silently backdated. Repository histories supporting the relevant freeze/result ordering are in CHRONOLOGY.json. GWTC's public freeze artifact postdates its result commit; the corresponding limitation is retained prominently.

## Repository source files

Each source link is immutable. Full hashes and byte counts are in [SOURCE_LEDGER.json](../../audit/cross_domain/SOURCE_LEDGER.json).

| ID | Repository / file | Git blob SHA1 |
|---|---|---|
| E001 | [rickyjreyes/GWTC / GWTC_V4_STRUCTURED_NULL_RESULT.md](https://github.com/rickyjreyes/GWTC/blob/bf95826c93ba11001f3461b653e3f279b4799573/GWTC_V4_STRUCTURED_NULL_RESULT.md) | `1038dfbb413df10b1a6024a1a268f21afb0b400a` |
| E002 | [rickyjreyes/GWTC / PROVENANCE.md](https://github.com/rickyjreyes/GWTC/blob/bf95826c93ba11001f3461b653e3f279b4799573/PROVENANCE.md) | `15f619e483ecac66347e807b5d4b8aa883c98f2f` |
| E003 | [rickyjreyes/GWTC / README.md](https://github.com/rickyjreyes/GWTC/blob/bf95826c93ba11001f3461b653e3f279b4799573/README.md) | `47531aaeaf4cf5d39eb013013f8b6ef8713e9245` |
| E004 | [rickyjreyes/GWTC / RESULTS.md](https://github.com/rickyjreyes/GWTC/blob/bf95826c93ba11001f3461b653e3f279b4799573/RESULTS.md) | `b807704aa66ad47a7a6e935c60b3fd6472bde3e6` |
| E005 | [rickyjreyes/GWTC / scripts/evaluate_frozen_gwtc_holdout.py](https://github.com/rickyjreyes/GWTC/blob/bf95826c93ba11001f3461b653e3f279b4799573/scripts/evaluate_frozen_gwtc_holdout.py) | `846556ae9d3099b1020717fd01ee2c4972ea08e6` |
| E006 | [rickyjreyes/GWTC / scripts/fit_frozen_gwtc_mode.py](https://github.com/rickyjreyes/GWTC/blob/bf95826c93ba11001f3461b653e3f279b4799573/scripts/fit_frozen_gwtc_mode.py) | `8a23fbb2e482351ac16262993d3e01becdf94987` |
| E007 | [rickyjreyes/GWTC / tables/gwtc_frozen_holdout_result.csv](https://github.com/rickyjreyes/GWTC/blob/bf95826c93ba11001f3461b653e3f279b4799573/tables/gwtc_frozen_holdout_result.csv) | `9af91e6af42d4a8a3b7ef2c6dcd1792fc6641c48` |
| E008 | [rickyjreyes/GWTC / tables/gwtc_frozen_mode.json](https://github.com/rickyjreyes/GWTC/blob/bf95826c93ba11001f3461b653e3f279b4799573/tables/gwtc_frozen_mode.json) | `d788fa7d3343dad1f428dae7559b97ff91a95263` |
| E009 | [rickyjreyes/GWTC / tables/gwtc_v3_frozen_unbinned_kde_mode.json](https://github.com/rickyjreyes/GWTC/blob/bf95826c93ba11001f3461b653e3f279b4799573/tables/gwtc_v3_frozen_unbinned_kde_mode.json) | `f8e612baffd66b0c188f1531997e9db1a72ab306` |
| E010 | [rickyjreyes/GWTC / tables/gwtc_v3_unbinned_kde_holdout_result.csv](https://github.com/rickyjreyes/GWTC/blob/bf95826c93ba11001f3461b653e3f279b4799573/tables/gwtc_v3_unbinned_kde_holdout_result.csv) | `d468dc0d2e858bd040820a6ddd430c93c8439607` |
| E011 | [rickyjreyes/GWTC / tables/gwtc_v4_frozen_structured_null.json](https://github.com/rickyjreyes/GWTC/blob/bf95826c93ba11001f3461b653e3f279b4799573/tables/gwtc_v4_frozen_structured_null.json) | `83ef8554005a3dbad32076069da8fdef0169ba13` |
| E012 | [rickyjreyes/GWTC / tables/gwtc_v4_structured_null_result.csv](https://github.com/rickyjreyes/GWTC/blob/bf95826c93ba11001f3461b653e3f279b4799573/tables/gwtc_v4_structured_null_result.csv) | `797acfc37212941dd908f76cdf5947edf7e08022` |
| E013 | [rickyjreyes/LHC / README.md](https://github.com/rickyjreyes/LHC/blob/5fc36d638e0ba1b03e236d6a11547b29966bbe8b/README.md) | `04d37fab79085e6b37c5e2938ad8c29eed534718` |
| E014 | [rickyjreyes/LHC / lhcb_wct_analysis_kit/21_cross_region_scaling_phase_test.py](https://github.com/rickyjreyes/LHC/blob/5fc36d638e0ba1b03e236d6a11547b29966bbe8b/lhcb_wct_analysis_kit/21_cross_region_scaling_phase_test.py) | `723d694b4d0d5374686b0a3696f3b2af635a756b` |
| E015 | [rickyjreyes/LHC / lhcb_wct_analysis_kit/CMS_LOCKED_LHCB_FREEZE_2026-08-31.md](https://github.com/rickyjreyes/LHC/blob/5fc36d638e0ba1b03e236d6a11547b29966bbe8b/lhcb_wct_analysis_kit/CMS_LOCKED_LHCB_FREEZE_2026-08-31.md) | `384544d55e38d4056e866e983ca095d30d6012ea` |
| E016 | [rickyjreyes/LHC / lhcb_wct_analysis_kit/CMS_TO_LHCB_FIXED_FREQUENCY_RESULT_2026-08-31.md](https://github.com/rickyjreyes/LHC/blob/5fc36d638e0ba1b03e236d6a11547b29966bbe8b/lhcb_wct_analysis_kit/CMS_TO_LHCB_FIXED_FREQUENCY_RESULT_2026-08-31.md) | `2a519afb6c7b514f457fbd4845e9f55d499d5f28` |
| E017 | [rickyjreyes/LHC / lhcb_wct_analysis_kit/R/audit/AUDIT_NOTE.md](https://github.com/rickyjreyes/LHC/blob/5fc36d638e0ba1b03e236d6a11547b29966bbe8b/lhcb_wct_analysis_kit/R/audit/AUDIT_NOTE.md) | `ebf8e49e6139b03925162666715cb62ba205ee7c` |
| E018 | [rickyjreyes/LHC / lhcb_wct_analysis_kit/R/audit/run_veto_invariance_analysis.R](https://github.com/rickyjreyes/LHC/blob/5fc36d638e0ba1b03e236d6a11547b29966bbe8b/lhcb_wct_analysis_kit/R/audit/run_veto_invariance_analysis.R) | `0fe239f9bb61ce6742e849b5ff6fcff328a7eb03` |
| E019 | [rickyjreyes/LHC / lhcb_wct_analysis_kit/README.md](https://github.com/rickyjreyes/LHC/blob/5fc36d638e0ba1b03e236d6a11547b29966bbe8b/lhcb_wct_analysis_kit/README.md) | `ee44de89f5aa60ad8fca28827c5c7252cea3ad5e` |
| E020 | [rickyjreyes/LHC / lhcb_wct_analysis_kit/outputs/analysis_summary.json](https://github.com/rickyjreyes/LHC/blob/5fc36d638e0ba1b03e236d6a11547b29966bbe8b/lhcb_wct_analysis_kit/outputs/analysis_summary.json) | `3097287d8da929ec8790f8a056d185e8e8dceffd` |
| E021 | [rickyjreyes/LHC / lhcb_wct_analysis_kit/outputs_logcos_poisson_twomode_kde_polar/two_mode_summary.json](https://github.com/rickyjreyes/LHC/blob/5fc36d638e0ba1b03e236d6a11547b29966bbe8b/lhcb_wct_analysis_kit/outputs_logcos_poisson_twomode_kde_polar/two_mode_summary.json) | `ef6d37b135dd5699b77b911bda88eb263b2b1435` |
| E022 | [rickyjreyes/LHC / lhcb_wct_analysis_kit/outputs_wct_cross_region_scaling/cross_region_scaling_summary.json](https://github.com/rickyjreyes/LHC/blob/5fc36d638e0ba1b03e236d6a11547b29966bbe8b/lhcb_wct_analysis_kit/outputs_wct_cross_region_scaling/cross_region_scaling_summary.json) | `3264411221a9de00938b959cdf7c73ecb8e794c7` |
| E023 | [rickyjreyes/LHC / lhcb_wct_analysis_kit/outputs_wct_koide_comb/koide_comb_summary.json](https://github.com/rickyjreyes/LHC/blob/5fc36d638e0ba1b03e236d6a11547b29966bbe8b/lhcb_wct_analysis_kit/outputs_wct_koide_comb/koide_comb_summary.json) | `cfede81f7f657b8d232d1ae70acb84422fc194a8` |
| E024 | [rickyjreyes/LHC / lhcb_wct_analysis_kit/outputs_wct_locked_winding_cross_region/locked_winding_summary.json](https://github.com/rickyjreyes/LHC/blob/5fc36d638e0ba1b03e236d6a11547b29966bbe8b/lhcb_wct_analysis_kit/outputs_wct_locked_winding_cross_region/locked_winding_summary.json) | `72d1dd394a57b007df9ef4eacf6835b643c4ff86` |
| E025 | [rickyjreyes/LHC / lhcb_wct_analysis_kit/outputs_wct_veto_covariance/veto_covariance_best_triplets.csv](https://github.com/rickyjreyes/LHC/blob/5fc36d638e0ba1b03e236d6a11547b29966bbe8b/lhcb_wct_analysis_kit/outputs_wct_veto_covariance/veto_covariance_best_triplets.csv) | `51d576576cf1087c647eaee8fff3ce1279bb34a8` |
| E026 | [rickyjreyes/LHC / lhcb_wct_analysis_kit/outputs_wct_veto_covariance/veto_covariance_scan_summary.csv](https://github.com/rickyjreyes/LHC/blob/5fc36d638e0ba1b03e236d6a11547b29966bbe8b/lhcb_wct_analysis_kit/outputs_wct_veto_covariance/veto_covariance_scan_summary.csv) | `25ac94d5fdea77a8defdba5857040293c46609b5` |
| E027 | [rickyjreyes/LHC / lhcb_wct_analysis_kit/outputs_wct_veto_covariance/veto_covariance_summary.json](https://github.com/rickyjreyes/LHC/blob/5fc36d638e0ba1b03e236d6a11547b29966bbe8b/lhcb_wct_analysis_kit/outputs_wct_veto_covariance/veto_covariance_summary.json) | `ed98a8b4d7e81d5903837ad10e1b5632888f23cd` |
| E028 | [rickyjreyes/LHC / lhcb_wct_analysis_kit/outputs_wct_well_first_koide/well_first_summary.json](https://github.com/rickyjreyes/LHC/blob/5fc36d638e0ba1b03e236d6a11547b29966bbe8b/lhcb_wct_analysis_kit/outputs_wct_well_first_koide/well_first_summary.json) | `bfbfdf8fbb5fb321886f5d1ccf868282f41bda06` |
| E029 | [rickyjreyes/LHC / lhcb_wct_analysis_kit/tables_r/statistical_audit/branch_provenance.csv](https://github.com/rickyjreyes/LHC/blob/5fc36d638e0ba1b03e236d6a11547b29966bbe8b/lhcb_wct_analysis_kit/tables_r/statistical_audit/branch_provenance.csv) | `1c0db8c752355705e0f3165c317274aa6b8fdb3f` |
| E030 | [rickyjreyes/LHC / lhcb_wct_analysis_kit/tables_r/statistical_audit/final_claim_matrix.csv](https://github.com/rickyjreyes/LHC/blob/5fc36d638e0ba1b03e236d6a11547b29966bbe8b/lhcb_wct_analysis_kit/tables_r/statistical_audit/final_claim_matrix.csv) | `61a3c03aafc6d31528ab6d2d8d1e3d1232529a14` |
| E031 | [rickyjreyes/LHC / lhcb_wct_analysis_kit/tables_r/statistical_audit/stage_registry.csv](https://github.com/rickyjreyes/LHC/blob/5fc36d638e0ba1b03e236d6a11547b29966bbe8b/lhcb_wct_analysis_kit/tables_r/statistical_audit/stage_registry.csv) | `737739aa04d99735ab036ac99c17db8accb6fcdf` |
| E032 | [rickyjreyes/LHC / lhcb_wct_analysis_kit/tables_r/statistical_audit/veto_invariance_models.csv](https://github.com/rickyjreyes/LHC/blob/5fc36d638e0ba1b03e236d6a11547b29966bbe8b/lhcb_wct_analysis_kit/tables_r/statistical_audit/veto_invariance_models.csv) | `2e77a9c0af3f98425c302faa2b5a4131ae96f67c` |
| E033 | [rickyjreyes/NIST / R/README.md](https://github.com/rickyjreyes/NIST/blob/4ddc3948c1e119f463e5d3d49bda1814ed195a84/R/README.md) | `0149a33451e3198ba0c44e50c2233ef83409fcdd` |
| E034 | [rickyjreyes/NIST / README.md](https://github.com/rickyjreyes/NIST/blob/4ddc3948c1e119f463e5d3d49bda1814ed195a84/README.md) | `8881ad7df20410ba2efa7af36ceda187e6efaa80` |
| E035 | [rickyjreyes/NIST / RESULTS.md](https://github.com/rickyjreyes/NIST/blob/4ddc3948c1e119f463e5d3d49bda1814ed195a84/RESULTS.md) | `19bb2c33a18a98ef130045da28ce532fff03398b` |
| E036 | [rickyjreyes/NIST / outputs/batch_neighbors/co_ion2_120_full_bin120/nist_summary.json](https://github.com/rickyjreyes/NIST/blob/4ddc3948c1e119f463e5d3d49bda1814ed195a84/outputs/batch_neighbors/co_ion2_120_full_bin120/nist_summary.json) | `4b8636ed236211ffdeff0102b416a65573f19ee9` |
| E037 | [rickyjreyes/NIST / outputs/batch_neighbors/co_ion2_160_full/nist_summary.json](https://github.com/rickyjreyes/NIST/blob/4ddc3948c1e119f463e5d3d49bda1814ed195a84/outputs/batch_neighbors/co_ion2_160_full/nist_summary.json) | `f2e8d9aede5e823ff4af4ac424c937b0fe82fb57` |
| E038 | [rickyjreyes/NIST / outputs/batch_neighbors/co_ion2_200_full_bin200/nist_summary.json](https://github.com/rickyjreyes/NIST/blob/4ddc3948c1e119f463e5d3d49bda1814ed195a84/outputs/batch_neighbors/co_ion2_200_full_bin200/nist_summary.json) | `db91c2e7bf4974d87c6bb9aa4d47b725a996ebb1` |
| E039 | [rickyjreyes/NIST / outputs/fe_ion2_120/nist_summary.json](https://github.com/rickyjreyes/NIST/blob/4ddc3948c1e119f463e5d3d49bda1814ed195a84/outputs/fe_ion2_120/nist_summary.json) | `1ae172a78f868ab48bf46bfbd4adeb31cd320383` |
| E040 | [rickyjreyes/NIST / outputs/fe_ion2_160/nist_summary.json](https://github.com/rickyjreyes/NIST/blob/4ddc3948c1e119f463e5d3d49bda1814ed195a84/outputs/fe_ion2_160/nist_summary.json) | `6fc35b3669f8d33290f8f7a181696c518217e7b7` |
| E041 | [rickyjreyes/NIST / outputs/fe_ion2_200/nist_summary.json](https://github.com/rickyjreyes/NIST/blob/4ddc3948c1e119f463e5d3d49bda1814ed195a84/outputs/fe_ion2_200/nist_summary.json) | `1d61b691a464066e57ea924f597f8519435867cb` |
| E042 | [rickyjreyes/NIST / scripts/nist_wct_log_spectral_scan_FIXED.py](https://github.com/rickyjreyes/NIST/blob/4ddc3948c1e119f463e5d3d49bda1814ed195a84/scripts/nist_wct_log_spectral_scan_FIXED.py) | `152f8db887337eb05a261f2ae6fc993a239fc759` |
| E043 | [rickyjreyes/NIST / tables_r/statistical_audit/analysis_registry.csv](https://github.com/rickyjreyes/NIST/blob/4ddc3948c1e119f463e5d3d49bda1814ed195a84/tables_r/statistical_audit/analysis_registry.csv) | `21dd177e60be7b36194a9bfbbf0a3912c28c2063` |
| E044 | [rickyjreyes/NIST / tables_r/statistical_audit/bin_grid_results.csv](https://github.com/rickyjreyes/NIST/blob/4ddc3948c1e119f463e5d3d49bda1814ed195a84/tables_r/statistical_audit/bin_grid_results.csv) | `450bfa6138466a74f8b8d13368508fad719f8c99` |
| E045 | [rickyjreyes/NIST / tables_r/statistical_audit/final_claim_matrix.csv](https://github.com/rickyjreyes/NIST/blob/4ddc3948c1e119f463e5d3d49bda1814ed195a84/tables_r/statistical_audit/final_claim_matrix.csv) | `733c934bbc79db38af8d5d4291e123d3a35e8708` |
| E046 | [rickyjreyes/NIST / tables_r/statistical_audit/holdout_results.csv](https://github.com/rickyjreyes/NIST/blob/4ddc3948c1e119f463e5d3d49bda1814ed195a84/tables_r/statistical_audit/holdout_results.csv) | `5c5b9ad54feb061d7fbc7eb6dab9f6cd0bec6d99` |
| E047 | [rickyjreyes/NIST / tables_r/statistical_audit/model_comparison.csv](https://github.com/rickyjreyes/NIST/blob/4ddc3948c1e119f463e5d3d49bda1814ed195a84/tables_r/statistical_audit/model_comparison.csv) | `116702500a58e9a32099a3d374e776503eaf2636` |
| E048 | [rickyjreyes/NIST / tables_r/statistical_audit/null_calibration.csv](https://github.com/rickyjreyes/NIST/blob/4ddc3948c1e119f463e5d3d49bda1814ed195a84/tables_r/statistical_audit/null_calibration.csv) | `9ad878b6c9538dfb79f19ff4272361cbc7d497b3` |
| E049 | [rickyjreyes/wct-cms / README.md](https://github.com/rickyjreyes/wct-cms/blob/82a007c4f81691b522cad7611c0b711a0ff94163/README.md) | `452e227f2c8e6e67617ed8e505e2df76281666b9` |
| E050 | [rickyjreyes/wct-cms / audit/2026-09-06/baseline/G1/summary.json](https://github.com/rickyjreyes/wct-cms/blob/82a007c4f81691b522cad7611c0b711a0ff94163/audit/2026-09-06/baseline/G1/summary.json) | `166645ef8bb35ece7e65112891ba44dbf16c9562` |
| E051 | [rickyjreyes/wct-cms / audit/2026-09-06/baseline/G2_locked/summary.json](https://github.com/rickyjreyes/wct-cms/blob/82a007c4f81691b522cad7611c0b711a0ff94163/audit/2026-09-06/baseline/G2_locked/summary.json) | `4435f3985606ad937f2743d7b3dab27321de1e49` |
| E052 | [rickyjreyes/wct-cms / audit/2026-09-06/baseline/H1/config.json](https://github.com/rickyjreyes/wct-cms/blob/82a007c4f81691b522cad7611c0b711a0ff94163/audit/2026-09-06/baseline/H1/config.json) | `babf65e7469820df1933f8fcb69df34f1db1d7d3` |
| E053 | [rickyjreyes/wct-cms / audit/2026-09-06/baseline/H1/summary.json](https://github.com/rickyjreyes/wct-cms/blob/82a007c4f81691b522cad7611c0b711a0ff94163/audit/2026-09-06/baseline/H1/summary.json) | `1f0d1252bed6725110ad26a8258d551fee8c0253` |
| E054 | [rickyjreyes/wct-cms / audit/2026-09-06/baseline/H2/summary.json](https://github.com/rickyjreyes/wct-cms/blob/82a007c4f81691b522cad7611c0b711a0ff94163/audit/2026-09-06/baseline/H2/summary.json) | `58d5986b0c42389c4fb2db98a546b94fb4a69e31` |
| E055 | [rickyjreyes/wct-cms / audit/2026-09-06/certificate/summary.json](https://github.com/rickyjreyes/wct-cms/blob/82a007c4f81691b522cad7611c0b711a0ff94163/audit/2026-09-06/certificate/summary.json) | `e271bb7b819b5111379f6170396babe422478d97` |
| E056 | [rickyjreyes/wct-cms / docs/ANALYSIS_PROTOCOL.md](https://github.com/rickyjreyes/wct-cms/blob/82a007c4f81691b522cad7611c0b711a0ff94163/docs/ANALYSIS_PROTOCOL.md) | `e5dbb24076d5cdec7ed4c8e0881f28be03266a35` |
| E057 | [rickyjreyes/wct-cms / docs/CMS_CANDIDATE_FREEZE_2026-08-28.md](https://github.com/rickyjreyes/wct-cms/blob/82a007c4f81691b522cad7611c0b711a0ff94163/docs/CMS_CANDIDATE_FREEZE_2026-08-28.md) | `779bb9015797edbc47e91adf49cdba42af72c9fb` |
| E058 | [rickyjreyes/wct-cms / docs/CMS_INDEPENDENT_ADVERSARIAL_AUDIT.md](https://github.com/rickyjreyes/wct-cms/blob/82a007c4f81691b522cad7611c0b711a0ff94163/docs/CMS_INDEPENDENT_ADVERSARIAL_AUDIT.md) | `8c03c17746bec5f40a435904e05c13990117a55d` |
| E059 | [rickyjreyes/wct-cms / docs/CMS_REPLICATION_FILE2_RESULT_2026-08-28.md](https://github.com/rickyjreyes/wct-cms/blob/82a007c4f81691b522cad7611c0b711a0ff94163/docs/CMS_REPLICATION_FILE2_RESULT_2026-08-28.md) | `da7ec0f02d00b14f164e140e0baf3d722c33c6e0` |
| E060 | [rickyjreyes/wct-cms / docs/CMS_RUN2016G_FILE2_PHASE_LOCK_FREEZE_2026-08-31.md](https://github.com/rickyjreyes/wct-cms/blob/82a007c4f81691b522cad7611c0b711a0ff94163/docs/CMS_RUN2016G_FILE2_PHASE_LOCK_FREEZE_2026-08-31.md) | `e9104305e7c0d41908cde1d24fd158195013dd48` |
| E061 | [rickyjreyes/wct-cms / docs/CMS_RUN2016G_FILE2_PHASE_LOCK_RESULT_2026-08-31.json](https://github.com/rickyjreyes/wct-cms/blob/82a007c4f81691b522cad7611c0b711a0ff94163/docs/CMS_RUN2016G_FILE2_PHASE_LOCK_RESULT_2026-08-31.json) | `6c3e90c12a3a4ddabffa769d0edaaa2f9375471a` |
| E062 | [rickyjreyes/wct-cms / src/cms_wct/analysis.py](https://github.com/rickyjreyes/wct-cms/blob/82a007c4f81691b522cad7611c0b711a0ff94163/src/cms_wct/analysis.py) | `69be6e0098101b1f2e194e031e80b64cb9e2fcc4` |

The local task attachment supplies instructions only; it is not empirical evidence. No remembered anchor replaced a retrieved source value. Frozen source precision, competing pipelines, negative results and uncertainty limitations are all preserved.
