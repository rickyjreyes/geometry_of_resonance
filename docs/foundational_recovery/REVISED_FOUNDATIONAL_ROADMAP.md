# Revised foundational roadmap

Date: 2026-10-20. The next major mathematical allocation should address **strict
subadditivity for the independently defined cubic–quintic Hamiltonian on R^2**,
at fixed k_*=1 and the exact existing beta=gamma and D4 norm. This choice is
based on its complete governing equation, global H^2 flow, conserved energy
and norm, surviving periodic proof, and newly excluded vanishing alternative.
It is not based on attractive plots or fitted particle data.

## The next single highest-value task

Prove or refute, with the original coefficients and no projection,

    I_2(m)<I_2(mu)+I_2(m-mu) for every 0<mu<m,
    I_2(m)=inf_(||u||_2^2=m) [||(Delta+1)u||_2^2
                              -beta||u||_4^4/2+gamma||u||_6^6/3],
    m=0x1.45b8a0bad2714p+1,
    beta=gamma=0x1.05f1d178c75ebp+2.

This is a precise whole-space mathematical question about the independent
model, not an assignment of physical mass. B3 already proves -infinity<I_2(m)<0
and rules out vanishing. A strict all-split inequality would discharge B4's
remaining compactness hypothesis and give a whole-space minimizer set with
orbital H^2 stability. It would not prove that the minimizer is the periodic
D4 branch, unique, real, D4-symmetric, a physical particle, or derived from S004.

Acceptance requires an analytic inequality or rigorous enclosures addressing
**all** 0<mu<m, including endpoint regimes. Sampled mass grids and two trial
upper bounds do not compare the true infima. Conversely, a refutation must
show equality/failure with justified infimum bounds or a precise minimization
obstruction; one expensive unsuccessful search is insufficient. Keep complex
and anisotropic competitors admissible. Do not impose radial or D4 symmetry
to manufacture compactness. Record whether any conclusion concerns the whole
minimizer set or one orbit.

## Dependency order after that task

1. If strict binding succeeds, establish any desired frequency/decay and full
   amplitude/phase spectral claims for the resulting whole-space states.
   Prove a box-limit theorem separately if identifying them with the D4 branch.
2. Select a foundational physical program only when an actual healthy action,
   field content, units and admissible data have independent justification.
   A1 shows that merely adding the already published dynamical metric fails
   the ghost gate. Future degeneracy/gauge operators require an explicit new
   theory proposal and its own constraint and nonlinear analysis.
3. A nonlinear parent reduction needs the full object map in
   `NONLINEAR_MODEL_BRIDGE.md`, including the gapless phase and excluded modes.
   Do not restart this route by selecting a quadratic branch and calling it
   the complete complex Hamiltonian.
4. Physical operator, length, action normalization, particle assignments, rest
   energy and inertia remain downstream. No empirical mass fitting is an
   acceptance path for these missing inputs.
5. Only after a common model and units are established should matched
   dimension/lifetime questions be reopened. H^2 algebra in d<=3 is an estimate,
   not physical dimensional selection or impossibility of 4D confinement.

## Existing work retained

- PR #33 stays draft and unmerged. Its H2/H4/M1 and all original receipts are
  immutable dependencies of this stacked Phase III PR.
- Eigen's positive scalar and corrected positive spatial dilation obstructions,
  all nine negative attempts, and the radial repeated-Laplacian defect remain.
  B3 is for a different energy and does not repair the eigen pipeline.
- D4 #24/#25/#27 retain their scoped positive results. Numerical integrator,
  envelope and sharp gain obligations #21–23 remain independent open tasks.
- DS-01 has its own equation and unresolved nonlinear refinement/denominator
  obligations; no broad scan or lifetime comparison was executed.
- No provisional maturity score is increased for improved negative knowledge,
  documentation, a test count or this model-separation decision.

## Issue handling

Geometry #16 receives the exact scalar–tensor ghost/constraint result, #18 the
nonlinear finite-window obstruction and distinct-model decision, and #12 the
verified M1/dependency boundary. Supporting issue comments must identify the
independent Hamiltonian when discussing B2/B3; they must not imply the eigen
positive functional now binds. Acceptance criteria and issue states remain
unchanged. Publication is a new draft PR stacked on the still-draft #33;
neither PR is merged by this work.
