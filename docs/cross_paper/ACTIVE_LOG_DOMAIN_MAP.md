# Physical domain to log coordinate and scale winding

P03 (the scale–winding/bijection paper) already proves the conditional relation

\[
\nu\Delta\ell=2\pi n,\qquad\lambda=e^{\Delta\ell/n},\qquad
\nu=2\pi/\log\lambda
\tag{D1}
\]

for the nontrivial oriented recurrence with a stable confined log-eigenmode and specified log domain. Use n≠0 and a positive orientation convention for λ>1; n=0 is a separate nonoscillatory case. This is an existing downstream theorem. Its operator-existence burden is stated in that paper and is not reclassified as a missing theorem of elementary algebra.

## Coordinates and units cannot be identified silently

| Quantity | Definition | Units | Current derivation |
|---|---|---|---|
| k_min | Minimum of amplitude dispersion | L⁻¹ | Derived in F2, parameter dependent |
| k_r | Local radial oscillation momentum | L⁻¹ | Conditional on a radial operator/eigenstate |
| ∇ϑ | Goldstone/spatial phase gradient | L⁻¹ | Distinct branch from amplitude carrier |
| Λ_n | Second-order radial eigenvalue | L⁻² | Conditional discrete spectral problem |
| ν | Frequency conjugate to ℓ=log(x/x_ref) | Dimensionless | D1 requires physical log-eigenmode |
| Δℓ | log(x_+/x_−) | Dimensionless | No independently selected endpoints yet |

For x=r, ∂ℓϑ=r∂rϑ. A constant spatial carrier ϑ=k_min r becomes k_min r_ref e^ℓ, not νℓ. Constant log frequency instead requires a phase proportional to log r, i.e. radial gradient ν/r. Multiplying k_min by a guessed radius only creates a dimensionless number; it does not prove this phase law.

For a general positive monotone observable x=X(s), a putative phase ϑ(s) has local log frequency

\[
\nu(\ell)=\frac{d\vartheta/ds}{d\log X/ds}.
\tag{D2}
\]

It is constant only if those derivatives obey the required relation. Turning points or nonmonotone X require separate branches. D2 is the explicit coordinate bridge to test once a field and observable have been constructed.

## Existing candidate physical maps

| Candidate | Existing support | Necessary unresolved premise |
|---|---|---|
| Mass/rest energy | P07 effective dispersion, P08/P22 loop-energy identities | Identify stress-energy, loop, sector, and physical normalization of the same S004 branch |
| Curvature σ or Θ | Filament P04 and loop P22 constructions | Positive operational X, selected branch and endpoints; S004 curvature has units L⁻², while loop σ has L⁻¹ |
| Spectral wavenumber/momentum | PFF Fourier band; p=ℏk conditional identification | PFF unstable band and physical support need derivation; low-mode cutoff is not a domain endpoint |
| Frequency/energy | Derived amplitude pole; E=ℏω if quantized | State, detector and spectral support; a minimum alone supplies no interval |
| Shell or confinement radius | Closure L2–L3, PFF radial operator | Actual global solution, operational edge definition, and X map |
| Coherence length | Geometry/PFF phenomenology | Independently fixed coherence observable and endpoint criterion |

Even the PFF unstable interval q_±=[a±√(a²+4br)]/(2b), when both are positive, is a Fourier **growth** interval. Its width is not automatically a physical log domain. The mapped stable EFT witness has no such growing interval. An arbitrary EFT cutoff is an approximation boundary, not a predicted recurrence boundary. For m_n∝n/R at fixed R, adjacent ratios (n+1)/n vary; this sequence is not constant-ratio DSI.

A diffuse localized field also needs an operational endpoint definition. Thresholding at, for example, a chosen mass fraction creates a convention unless the detector or dynamics independently supplies that threshold. No such threshold is selected here.

Consequently no Δℓ, ν or λ is output as a parameter-free physical prediction. Gap class `CROSS_PAPER_REDUCTION_REQUIRED` because candidate endpoints and the theorem exist; physical preparation/measurement choices remain inputs. `PHYSICAL_ACTIVE_DOMAIN_NOT_DERIVED`, `NOT_IDENTIFIABLE`, and the previous closure-frequency/estimator counterexample remain valid. They do not negate the newly completed S004-to-EFT bridge.

Falsifier of an asserted D1 application: an endpoint changes with a numerical box, the phase is not linear in log X, X is nonpositive/multivalued without a branch declaration, or ν is equated to spatial k despite incompatible units.
