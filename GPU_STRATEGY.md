# GPU Strategy & Research Roadmap for WCT

## Current State Assessment

### What you have (strengths)
- 7 master equations + 82 canonical families — solid theoretical scaffold
- Working GPU code (CuPy) for 2D (1024x1024) and 3D (128x128x128) evolution
- SM spectrum fitting with differential evolution (29 free params across 3 regimes)
- Ensemble averaging (10 runs) with entropy/coherence/resonance tracking
- 10 Zenodo-published papers with explicit scope control (SCOPE.md)
- Physical constant derivation pipeline (vacuum energy, Lambda_eff)

### What's missing (gaps that block credibility)
1. **No convergence studies** — you run 1024x1024 but never show that 512x512 and 2048x2048 bracket the same answer
2. **No error bars on physical predictions** — vacuum energy comes out as a single number, not a confidence interval
3. **SM spectrum has 29 free parameters for 12 target masses** — this is underdetermined; any smooth function can fit 12 points with 29 knobs
4. **3D runs are too small** — 128^3 with 100k steps is a few minutes on a modern GPU; the interesting regime is 512^3+ for 10^6+ steps
5. **No comparison to null models** — you never show that a *generic* nonlinear wave equation fails where WCT succeeds
6. **No reproducible benchmarks** — no CI, no requirements.txt, no containerized runs

---

## The Three Calculations That Would Change Everything

### Calculation 1: Convergence-Verified Mass Emergence (HIGH PRIORITY)

**The question:** Does the curvature operator Theta[psi] produce *quantized* effective mass values that are independent of grid resolution, domain size, and initial conditions?

**What to run:**
- 3D WCT evolution (Master Eq 7) on grids: 64^3, 128^3, 256^3, 512^3, 1024^3
- For each grid: 50+ ensemble runs with random ICs
- Track: number of stable localized structures, their effective mass (integrated |psi|^2 in connected regions), their lifetime
- Run until t_final is large enough that masses *stop changing* (steady state)
- Plot mass values vs grid resolution — they should converge

**Why this matters:** Right now you claim mass emerges from confinement, but you've never shown the emergent mass is *resolution-independent*. If it is, that's a genuine prediction. If it drifts with grid size, it's a numerical artifact.

**GPU requirement:** 1024^3 x float64 x 3 fields = ~24 GB VRAM. Needs an A100 (80GB) or H100. Multi-day run for the full sweep.

**Estimated scope:** ~1 week on a single A100 for the full convergence ladder.

### Calculation 2: Parameter-Free SM Mass Ratios (CRITICAL PRIORITY)

**The problem with SM_Spectrum/:** You have 29 parameters fitting 12 masses. This proves nothing — it's curve fitting. The theory needs to produce mass *ratios* from the dynamics alone, with zero or minimal free parameters.

**What to run instead:**
- Solve the full UWCT equation (Master Eq 7) in a spherical/toroidal domain
- Find ALL stable eigenmodes (not just the ground state)
- Catalog eigenmode energies E_0, E_1, E_2, ...
- Compute ratios: E_1/E_0, E_2/E_0, etc.
- Compare to known ratios: m_mu/m_e = 206.768, m_tau/m_e = 3477.23, m_p/m_e = 1836.15

**Why this matters:** If WCT *naturally* produces a spectrum with ratios close to the muon/electron or proton/electron mass ratio from curvature feedback alone — with no fitting — that would be a landmark result. Even getting the right *order of magnitude* of these ratios from a parameter-free simulation would be publishable in a top journal.

**GPU requirement:** Spectral eigenvalue problems on 3D grids. Use iterative solvers (LOBPCG via CuPy/SciPy). The nonlinear eigenvalue problem requires self-consistent iteration. A100 or better, running for days to converge the first 20-50 eigenmodes at high resolution.

### Calculation 3: Curvature-Sourced CMB Power Spectrum (HIGH IMPACT)

**The question:** Does the UCM cosmology module (Master Eq 8) produce an acoustic peak structure qualitatively matching the CMB?

**What to run:**
- Evolve the coupled (psi, Phi, delta_gamma, v_gamma, delta_b, v_b) system
- Large 1D k-space grid (10,000+ modes) evolved for 10^5+ timesteps
- Extract P(k) at "recombination" (when tight-coupling breaks)
- Compare peak positions and ratios to Planck CMB data

**Why this matters:** The CMB power spectrum is the most precisely measured thing in cosmology. If WCT's curvature-sourced potential produces acoustic peaks at approximately the right k-ratios, that's a falsifiable prediction. If it doesn't, you know exactly where the theory needs modification.

**GPU requirement:** This is actually tractable on modest hardware (1D k-space). A single V100 running for a day could sweep the parameter space. The GPU advantage is running 10,000+ parameter combinations in parallel.

---

## Where to Run Multi-Day GPU Calculations

### Tier 1: Free / Low-Cost (start here)
| Platform | GPU | VRAM | Cost | Best For |
|----------|-----|------|------|----------|
| Google Colab Pro+ | A100 | 40GB | $50/mo | Prototyping, short runs (<12h) |
| Kaggle Notebooks | T4/P100 | 16GB | Free (30h/week) | Small 2D/3D convergence tests |
| Lightning.ai | A10G | 24GB | Free tier available | Development, debugging |

### Tier 2: Research Cloud (for multi-day runs)
| Platform | GPU | VRAM | Cost | Best For |
|----------|-----|------|------|----------|
| Lambda Cloud | A100/H100 | 80GB | $1.10-$2.49/hr | Production convergence runs |
| RunPod | A100/H100 | 80GB | $1.04-$2.39/hr | Flexible spot instances |
| Vast.ai | A100 | 80GB | $0.70-$1.50/hr | Cheapest for long runs |
| CoreWeave | H100 | 80GB | $2.06/hr | Maximum performance |

### Tier 3: Academic / Grant-Funded
| Platform | GPU | Access | Best For |
|----------|-----|--------|----------|
| XSEDE/ACCESS | A100/V100 clusters | Free (proposal) | Large-scale sweeps |
| NSF CloudBank | AWS/GCP/Azure | Startup allocation | Quick access |
| University HPC | Varies | Department access | If affiliated |
| NVIDIA Academic | DGX/A100 | Application | Free hardware grants |

### Recommendation
Start with **Vast.ai** or **RunPod** for spot A100 instances. Budget ~$200-500 for the convergence study (Calc 1). That buys roughly 200-400 A100-hours, enough for the full grid-resolution ladder with ensemble averaging.

For the eigenmode calculation (Calc 2), you want sustained access — **Lambda Cloud** reserved instances or an **ACCESS allocation** (free, requires a 2-page proposal).

---

## What to Fix in Your Code First

### 1. Add convergence infrastructure
Your current code hardcodes grid sizes. You need a parameter sweep driver:
- Accept grid size, timesteps, ensemble count as CLI arguments
- Output structured JSON/HDF5 (not just print statements)
- Track wall-clock time per step for scaling analysis

### 2. Replace CuPy-only with CuPy+NumPy fallback
Right now your code crashes if there's no GPU. Add a simple backend switch so you can develop on CPU and deploy on GPU without code changes.

### 3. Add a nonlinear eigenvalue solver
For Calculation 2, you need to find stationary states of the UWCT equation. This means solving:
```
E * psi = -Theta[psi] + g|psi|^2 psi + L_WCT psi
```
self-consistently. Implement an imaginary-time evolution method or a Newton-Krylov solver. This is the single most important missing piece of infrastructure.

### 4. Implement proper energy conservation checks
Your time-stepper doesn't verify that the Lyapunov energy E_WCT is actually decreasing (or conserved, depending on the regime). Add this diagnostic — it catches numerical instabilities immediately.

### 5. Add null-model comparisons
Run the same simulations with Theta[psi] replaced by a simple -nabla^2 psi (no curvature feedback). Show that the standard wave equation does NOT produce the same mass emergence, convergence, or spectral structure. This is the control experiment your work is missing.

---

## The Big Picture: How to Get This Published in a Top Journal

### Current gap
The theory has many equations but no "killer number" — a single quantitative prediction that either matches observation or doesn't. Without this, the work reads as a framework rather than a falsifiable theory.

### Path to a killer number

**Option A (fastest):** Compute the muon/electron mass ratio from the eigenmode spectrum.
- If E_1/E_0 from the nonlinear eigenvalue problem comes out near 206.768 (or even 150-300), write it up immediately.
- Target: Physical Review Letters or Physical Review D.

**Option B (most impactful):** Compute the CMB acoustic peak ratio from the UCM cosmology.
- The ratio of the first-to-second peak position is ~1:2:3 in the standard model. If WCT reproduces this from curvature dynamics alone, it's a major result.
- Target: Journal of Cosmology and Astroparticle Physics.

**Option C (most novel):** Predict the curvature-coherence decay profile in optical cavities.
- You already have water cavity experiments. Derive a precise functional form for the decay curve from WCT and compare to your data.
- Target: Physical Review A or Optics Letters.

### What NOT to do
- Don't add more equations. You have 82+ already. More notation without more numbers won't help.
- Don't optimize the 29-parameter SM fit further. It's underdetermined and will be dismissed by referees.
- Don't write more papers until you have a convergence-verified numerical result.

---

## Prioritized Action Plan

| Priority | Action | Hardware | Outcome |
|----------|--------|----------|---------|
| 1 | Add convergence testing (grid refinement study) | Any GPU | Proves numerics are trustworthy |
| 2 | Implement nonlinear eigenvalue solver | A100 | Enables parameter-free mass ratios |
| 3 | Run 3D convergence ladder (64^3 to 1024^3) | A100 80GB | First resolution-independent mass values |
| 4 | Compute eigenmode spectrum and mass ratios | A100 multi-day | The "killer number" |
| 5 | Add null-model controls | Any GPU | Proves WCT is special vs generic NL waves |
| 6 | Run UCM cosmology to extract CMB peaks | V100 or better | Second independent prediction |
| 7 | Package everything (Docker, CI, HDF5 output) | CPU | Reproducibility for referees |

---

## Summary

You have a rich theoretical framework with working GPU code. The gap is between "interesting simulations" and "falsifiable predictions." The bridge is:

1. **Convergence** — prove your numbers don't depend on the grid
2. **Eigenvalue spectrum** — find the natural mass ratios with zero free parameters
3. **Null controls** — show generic wave equations can't do what WCT does

These three things, done properly on A100-class hardware over 1-2 weeks of GPU time, would transform this from a framework into a testable theory. Budget roughly $300-700 in cloud GPU time, or apply for a free ACCESS/XSEDE allocation.
