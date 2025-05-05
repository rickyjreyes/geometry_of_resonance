import cupy as cp
import cupyx.scipy.fft as fft
import itertools
import random
import time
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import datetime

# -------------------------
# Simulation Parameters
# -------------------------
nx, ny = 1024, 1024
Lx, Ly = 10.0, 10.0
dx, dy = Lx / nx, Ly / ny
dt = cp.float64(0.0002)
nt = 20000
plot_interval = 1000
c = cp.float64(1.0)

alpha_vals = cp.linspace(1.45, 1.55, 21)
theta_vals = cp.linspace(0.0024, 0.0028, 9)
damping_vals = cp.array([0.000015, 0.00002, 0.000025])
noise_vals = cp.array([1e-5, 5e-6, 2e-6])

x = cp.linspace(-Lx / 2, Lx / 2, nx, dtype=cp.float64)
y = cp.linspace(-Ly / 2, Ly / 2, ny, dtype=cp.float64)
X, Y = cp.meshgrid(x, y)

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
        return coherence_idx * dx
    except IndexError:
        return Lx

def clip_field(psi, limit=100):
    return cp.clip(psi, -limit, limit)

def compute_resonance_strength(psi):
    p = cp.abs(psi)**2
    return cp.max(p) / (cp.mean(p) + 1e-12)

param_grid = list(itertools.product(alpha_vals, theta_vals, damping_vals, noise_vals))
random.shuffle(param_grid)

total_configs = len(param_grid)
total_steps = total_configs * nt
global_completed_steps = 0
global_start_time = time.time()

best_config = None
best_coherence = -cp.inf

for alpha, theta, damping, noise_level in param_grid:
    alpha = cp.float64(alpha)
    theta = cp.float64(theta)
    damping = cp.float64(damping)
    noise_level = cp.float64(noise_level)

    psi = cp.random.rand(ny, nx).astype(cp.float64) * 0.07
    psi_old = cp.copy(psi)
    coherence_history = []

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
        psi_new = clip_field(psi_new)

        psi_old = cp.copy(psi)
        psi = cp.copy(psi_new)

        if t % plot_interval == 0 or t == nt - 1:
            ξ = coherence_length(psi, dx)
            S = entropy(psi)
            ρ = compute_resonance_strength(psi)
            coherence_history.append(float(ξ))

            print(f"Step {t}, ⟨ξ⟩: {ξ:.5f}, ⟨S⟩: {S:.5f}, ⟨ρ⟩: {ρ:.5f}")

            if ξ > best_coherence:
                best_coherence = ξ
                best_config = (float(alpha), float(theta), float(damping), float(noise_level))

            if ξ > 0.1:
                with open("viable_universes.txt", "a") as f:
                    f.write(f"{alpha:.4f}, {theta:.4f}, {damping:.5f}, {noise_level:.1e}, {ξ:.5f}, {S:.5f}, {ρ:.5f}\n")

            if len(coherence_history) >= 5 and all(abs(val - 10.0) < 1e-3 for val in coherence_history[-5:]):
                print("✅ Early exit: coherence plateaued")
                break

        global_completed_steps += 1
        if global_completed_steps % 100 == 0:
            elapsed = time.time() - global_start_time
            avg_time = elapsed / global_completed_steps
            remaining = total_steps - global_completed_steps
            eta = datetime.datetime.now() + datetime.timedelta(seconds=remaining * avg_time)
            print(f"🔁 Progress: {global_completed_steps}/{total_steps} steps")
            print(f"⏱ Elapsed: {elapsed/60:.2f} min | Avg per step: {avg_time:.4f}s")
            print(f"⏳ Remaining: {remaining*avg_time/60:.2f} min")
            print(f"🕒 ETA Finish: {eta.strftime('%Y-%m-%d %H:%M:%S')}")

print("\n=== Best Stable Configuration ===")
print(f"α = {best_config[0]:.4f}, θ = {best_config[1]:.4f}, γ = {best_config[2]:.5f}, noise = {best_config[3]:.1e}")
print(f"Max Coherence Length ⟨ξ⟩ = {best_coherence:.5f}")
print(f"Total runtime: {time.time() - global_start_time:.2f} seconds")