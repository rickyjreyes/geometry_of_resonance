import cupy as cp
import cupyx.scipy.fft as fft
import matplotlib.pyplot as plt
import time
import datetime

# -------------------------
# Simulation Parameters
# -------------------------
nx, ny = 1024, 1024
Lx, Ly = 10.0, 10.0
dx, dy = Lx / nx, Ly / ny
dt = cp.float64(0.0002)
nt = 7500
c = cp.float64(1.0)
alpha = cp.float64(1.5)
theta = cp.float64(0.0026)
damping = cp.float64(0.00002)
noise_level = cp.float64(0.00001)
ensemble_runs = 50

# Physical constants
hbar = 1.054571817e-34  # J·s
c_light = 2.99792458e8  # m/s
G = 6.67430e-11         # m^3/kg/s^2
rho_vac_obs = 5.96e-10  # J/m^3

# -------------------------
# Helper Functions
# -------------------------
def laplacian_2d_4th(f, dx, dy):
    return (
        (-f + 16*(cp.roll(f, -1, axis=0) + cp.roll(f, 1, axis=0))
         - (cp.roll(f, -2, axis=0) + cp.roll(f, 2, axis=0))) / (12 * dy**2) +
        (-f + 16*(cp.roll(f, -1, axis=1) + cp.roll(f, 1, axis=1))
         - (cp.roll(f, -2, axis=1) + cp.roll(f, 2, axis=1))) / (12 * dx**2)
    )

def compute_Wpsi(psi, dx, dy):
    lap = laplacian_2d_4th(psi, dx, dy).astype(cp.float64)
    W = -lap / (psi + cp.float64(1e-8))
    return cp.clip(W, -1e5, 1e5)

def entropy(psi):
    p = cp.abs(psi)**2
    p /= cp.sum(p) + 1e-12
    return -cp.sum(p * cp.log(p + 1e-12))

def coherence_length(psi, dx):
    corr = fft.ifft2(cp.abs(fft.fft2(psi))**2).real
    corr /= corr[0, 0] + 1e-12
    corr1d = corr[corr.shape[0] // 2, :]
    try:
        coherence_idx = cp.where(corr1d < 0.5)[0][0]
        coherence_length = coherence_idx * dx
    except IndexError:
        print("⚠️ Fallback coherence length used")
        coherence_length = Lx
    return coherence_length

def clip_field(psi, limit=100):
    return cp.clip(psi, -limit, limit)

def compute_resonance_strength(psi):
    p = cp.abs(psi)**2
    peak = cp.max(p)
    avg = cp.mean(p)
    return peak / (avg + 1e-12)

# -------------------------
# Main Simulation Loop
# -------------------------
x = cp.linspace(-Lx/2, Lx/2, nx, dtype=cp.float64)
y = cp.linspace(-Ly/2, Ly/2, ny, dtype=cp.float64)
X, Y = cp.meshgrid(x, y)

all_entropies = []
all_coherences = []
all_resonance_strengths = []

start_time = time.time()
global_start_time = time.time()
global_completed_steps = 0
total_steps = nt * ensemble_runs

for run in range(ensemble_runs):
    print(f"Starting run {run+1}/{ensemble_runs}")
    psi = cp.random.rand(ny, nx).astype(cp.float64) * 0.07
    psi_old = cp.copy(psi)

    entropy_log = []
    coherence_log = []
    resonance_log = []

    for t in range(nt):
        Wpsi = compute_Wpsi(psi, dx, dy)
        lap = laplacian_2d_4th(psi, dx, dy)

        psi_new = (2 - damping * dt) * psi - psi_old + (c * dt)**2 * (
            lap - alpha * Wpsi * psi - theta * cp.tanh(Wpsi)**2 * psi
        )

        psi_new = 0.9995 * psi_new + 0.00025 * (
            cp.roll(psi_new, 1, axis=0) + cp.roll(psi_new, -1, axis=0) +
            cp.roll(psi_new, 1, axis=1) + cp.roll(psi_new, -1, axis=1)
        )
        psi_new += noise_level * cp.random.randn(ny, nx)
        psi_new = clip_field(psi_new, limit=100)

        psi_old = cp.copy(psi)
        psi = cp.copy(psi_new)

        if t % 1000 == 0 or t == nt - 1:
            entropy_log.append(entropy(psi))
            coherence_log.append(coherence_length(psi, dx))
            resonance_log.append(compute_resonance_strength(psi))

        global_completed_steps += 1
        if global_completed_steps % 100 == 0:
            elapsed = time.time() - global_start_time
            avg_time = elapsed / global_completed_steps
            remaining = total_steps - global_completed_steps
            eta = datetime.datetime.now() + datetime.timedelta(seconds=remaining * avg_time)
            print(f"\U0001f501 Progress: {global_completed_steps}/{total_steps} steps")
            print(f"\u23f1 Elapsed: {elapsed/60:.2f} min | Avg per step: {avg_time:.4f}s")
            print(f"\u23f3 Remaining: {remaining*avg_time/60:.2f} min")
            print(f"\U0001f552 ETA Finish: {eta.strftime('%Y-%m-%d %H:%M:%S')}")

    all_entropies.append(cp.array(entropy_log))
    all_coherences.append(cp.array(coherence_log))
    all_resonance_strengths.append(cp.array(resonance_log))

# -------------------------
# Post-Processing with Physical Scale Fit
# -------------------------
all_entropies = cp.array(all_entropies)
all_coherences = cp.array(all_coherences)
all_resonance_strengths = cp.array(all_resonance_strengths)

mean_entropy = cp.mean(all_entropies)
std_entropy = cp.std(all_entropies)
mean_coherence_sim = cp.mean(all_coherences)
std_coherence_sim = cp.std(all_coherences)
mean_resonance = cp.mean(all_resonance_strengths)
std_resonance = cp.std(all_resonance_strengths)

# Derive unit scale
xi_phys_m = (hbar * c_light / rho_vac_obs) ** 0.25
unit_scale_m = xi_phys_m / mean_coherence_sim
unit_scale_um = unit_scale_m * 1e6

# Real coherence in µm
mean_coherence_um = mean_coherence_sim * unit_scale_um
std_coherence_um = std_coherence_sim * unit_scale_um

# Energy and cosmological constant
rho_sim = hbar * c_light / xi_phys_m**4
Lambda_eff = (8 * cp.pi * G / c_light**4) * rho_sim

print("\n=== Ensemble Averaged Results ===")
print(f"Average Coherence Length ⟨ξ⟩: {mean_coherence_sim:.5f} units ≈ {mean_coherence_um:.2f} µm")
print(f"Coherence Std Dev: {std_coherence_sim:.5f} units ≈ {std_coherence_um:.2f} µm")
print(f"Average Entropy ⟨S⟩: {mean_entropy:.5f}")
print(f"Entropy Std Dev: {std_entropy:.5f}")
print(f"Average Resonance Strength ⟨ρ⟩: {mean_resonance:.5f}")
print(f"Resonance Strength Std Dev: {std_resonance:.5f}")
print("\n--- Derived Physical Quantities ---")
print(f"Physical scale per unit: {unit_scale_um:.2f} µm/unit")
print(f"Vacuum energy density: {rho_sim:.3e} J/m³")
print(f"Effective Λ: {Lambda_eff:.3e} 1/m²")
print("===========================")

# Save results
cp.save("entropy_ensemble.npy", all_entropies)
cp.save("coherence_ensemble.npy", all_coherences)
cp.save("resonance_ensemble.npy", all_resonance_strengths)

end_time = time.time()
print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")
