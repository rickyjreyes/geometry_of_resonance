# Universal invariant search

No independently defined invariant was recovered that sends **all four primary objects** to the same physical quantity. This is a failure to obtain a predictive map, not a theorem that no common mechanism can exist.

## Candidate invariants and identifiability

| Candidate I_i | What is established | What is missing | Result |
|---|---|---|---|
| Native nu_i | Three primary log-frequency estimands | LHCb a is not a frequency; equality is not implied by WCT | Primary panel ineligible; reference-frequency U0 stress test disagrees |
| Common mass/energy nu_i | Fixed q2→mass factor 2; wavenumber/energy and mass/energy constants preserve nu | Shared dynamical observable, and a→period map | U1 reference stress test disagrees |
| nu_i L_i/(2pi) | A dimensionless cycle-count identity | Physical L_i, phase closure and independent integer n_i | Fitted cycle count is not a prediction |
| nu_i L_i/(2pi n_i)=1 | Exact under the conditional theorem | Both L_i and n_i must be independent of nu_i | Tautological if defined from targets |
| k_star,i R_i | Dimensionless | Derived physical radius and phase-to-observable map | Free R_i would fit arbitrary frequencies |
| S004 branch ratios | K m², G/K, T/K are dimensionless combinations in a fixed convention | Independently fixed action, state/background and observable dependence | No four-domain measurable invariant identified |
| Geometric norm sqrt6 or sqrt(3/2) | Exact algebra in the 2–3–6 construction | Operator linking each norm to physical observable scaling | Partial numerical analogies only |
| lambda_CMS/lambda_NIST | Ratio can be calculated | Why it should equal 2 from the same physical branch; no GWTC or LHCb-a map | Retrospective analogy, not a four-domain invariant |

With nu_i=2pi n_i/L_i, allowing one independent continuous L_i per domain gives L_i=2pi n_i/nu_i for any positive data. Four fitted widths explain zero residual constraints. Likewise nu_i=c_i theta has five nominal continuous symbols but a one-dimensional scale redundancy; its output rank is four. Every positive quartet is fit exactly, and c_heldout is unconstrained. No such multiplier fit was performed.

Unknown discrete labels do not repair this: fitting n_i from observed nu_i consumes target information. With three training systems, an unassigned held-out n has multiple possible predictions. Selecting whichever one matches the held-out value is not cross-validation.

## Reproduce the published proximities without upgrading them

| Object | Exact-source-derived period or slope | Published comparator | Meaning |
|---|---:|---:|---|
| GWTC V1 | 1.923872254519363 | No independently assigned common-mechanism comparator | Not silently discarded |
| CMS | 2.445619519256875 | sqrt6 | Physical dilation assignment remains a hypothesis |
| Fe II | 1.222100055884532 | sqrt(3/2) | Physical dilation assignment remains a hypothesis |
| LHCb a | 1.2282873602374818 | sqrt(3/2) | Regional triplet slope, not the same type of period |

The exact-source CMS/Fe period ratio is **2.0011614494909615**. Under the partial hypothesis lambda_CMS=2 lambda_Fe, Fe predicts CMS frequency 7.030389778945075 (fractional error +0.064960%); CMS predicts Fe 31.23611701834412 (-0.288617%). This preserves the attractive numerical relation. It supplies neither a GWTC prediction nor a physical LHCb-a identification.

| Published fixed comparator | Frequency/slope prediction | Fractional prediction error |
|---|---:|---:|
| CMS | 7.01342497705518 | -0.176504% |
| NIST_FE160 | 30.99248335569948 | -1.066340% |
| LHCB_A | 1.224744871391589 | -0.288409% |


The LHCb difference from sqrt(3/2) is about -0.00354249 in a, or -0.010 of the paper's approximate conditional bootstrap standard deviation. The distribution is heavy-tailed and broad enough to include many simple alternatives. This is not a Gaussian z-test, nor evidence uniquely favoring the geometric expression.

P05's admissible mathematical scale set is {sqrt2,sqrt3,sqrt(3/2),sqrt6,2,3,6}. Four independently selected members create up to 7^4=2401 assignments before harmonics, variable choices or normalizations. The set's exact geometry does not select which member acts on which measured coordinate.

The genuine root-mass rule matters: r→sqrt6 r would give m→6m and mass-log frequency 2pi/ln6≈3.50671249. It does **not** give the CMS mass-log frequency near 7.026. Assigning sqrt6 directly to mass is a different physical hypothesis. Dividing lambda_CMS by 2 cannot be justified by m=r²; that rule squares lambda.

## Equality and fixed-scale stress tests

For a physically required coordinate re-expression, the available reference panel becomes (9.602325620315224, 7.025825825825827, 31.3265306122449, 15.22108) in mass/energy-log units. A single shared value cannot equal these four frozen references. The full U1 best common value is about 13.3923840242613; that fitted mean is not a WCT prediction. All held-out errors appear in [LEAVE_ONE_DOMAIN_OUT.md](LEAVE_ONE_DOMAIN_OUT.md).

Zero-parameter trial assignments lambda_mass=sqrt6 and lambda_mass=6 are also preserved in `derived_quantities.json`, with predictions for every reference domain. They do not rescue the discrepant systems. The first nearly matches CMS only after making an unproved mass-dilation assignment; the second follows the conditional root-mass interpretation but misses CMS substantially. The primary LHCb slope remains unmapped in either case.

## Model ranking

If completed, explanatory strength would be U4 (full physical chain), U3 (shared action plus independently specified systems), U2 (independent winding/domain inputs), U1 (shared scale plus required maps), then U0 (numerical equality). Presently U3/U4 lack numeric forward maps, U2 lacks independent sectors and physical widths, and U0/U1 fail the available reference stress test. Missing premises are not fitted away. The current finite-k theorem is retained as a conditional component, not confused with completed cross-domain support.
