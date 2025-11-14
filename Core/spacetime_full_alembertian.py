import cupy as cp
import cupyx.scipy.fft as fft
import matplotlib.pyplot as plt
import os

# -------------------------
# Simulation Parameters
# -------------------------
nx, ny = 1024, 1024
Lx, Ly = 10.0, 10.0
dx, dy = Lx / nx, Ly / ny
dt = 0.0001
nt = 1000000
c = 1.0
alpha = 2.0
theta = 0.002
damping = 0.00001
epsilon = 1e-6

save_interval = 10000
output_dir = "spacetime_emergence_outputs_v2"
os.makedirs(output_dir, exist_ok=True)

# -------------------------
# Helper Functions
# -------------------------
def laplacian_2d_4th(f, dx, dy):
    return (
        (-f + 16 * (cp.roll(f, -1, axis=0) + cp.roll(f, 1, axis=0))
         - (cp.roll(f, -2, axis=0) + cp.roll(f, 2, axis=0))) / (12 * dy**2) +
        (-f + 16 * (cp.roll(f, -1, axis=1) + cp.roll(f, 1, axis=1))
         - (cp.roll(f, -2, axis=1) + cp.roll(f, 2, axis=1))) / (12 * dx**2)
    )

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
        coherence_length = Lx
    return coherence_length

def clip_field(psi, limit=100):
    return cp.clip(psi, -limit, limit)

# -------------------------
# Initialize Field
# -------------------------
x = cp.linspace(-Lx/2, Lx/2, nx)
y = cp.linspace(-Ly/2, Ly/2, ny)
X, Y = cp.meshgrid(x, y)

sigma = 0.2
amplitude = 0.01
psi = amplitude * cp.exp(-(X**2 + Y**2) / (2 * sigma**2))
psi_old = cp.copy(psi)
psi_prev = cp.copy(psi)

entropy_log = []
coherence_log = []
rho_log = []

# -------------------------
# Main Evolution Loop
# -------------------------
for t in range(nt):
    lap = laplacian_2d_4th(psi, dx, dy)
    d2psi_dt2 = (psi - 2 * psi_old + psi_prev) / dt**2
    box_psi = d2psi_dt2 - lap
    denom = psi + epsilon * cp.exp(-alpha * cp.abs(psi)**2)
    Wpsi = box_psi / denom
    Wpsi = cp.clip(Wpsi, -1e5, 1e5)
    feedback = -alpha * Wpsi * psi - theta * (cp.tanh(Wpsi)**2) * psi

    psi_new = (2 - damping * dt) * psi - psi_old + (c * dt)**2 * (lap + feedback)

    psi_new = 0.9995 * psi_new + 0.00025 * (
        cp.roll(psi_new, 1, axis=0) + cp.roll(psi_new, -1, axis=0) +
        cp.roll(psi_new, 1, axis=1) + cp.roll(psi_new, -1, axis=1)
    )

    psi_new = clip_field(psi_new, limit=5)

    psi_prev = cp.copy(psi_old)
    psi_old = cp.copy(psi)
    psi = cp.copy(psi_new)

    if t % 1000 == 0:
        entropy_log.append(float(entropy(psi)))
        coherence_log.append(float(coherence_length(psi, dx)))
        rho_log.append(float(cp.mean(cp.abs(psi)**2)))

    if t % save_interval == 0:
        snapshot = psi.get()
        filename = os.path.join(output_dir, f"wave_snapshot_{t:07d}.npy")
        cp.save(filename, snapshot)
        print(f"Saved snapshot at t = {t}")

# -------------------------
# Post-Processing
# -------------------------
cp.save(os.path.join(output_dir, "entropy_log.npy"), cp.array(entropy_log))
cp.save(os.path.join(output_dir, "coherence_log.npy"), cp.array(coherence_log))
cp.save(os.path.join(output_dir, "rho_log.npy"), cp.array(rho_log))


# Convert logs to arrays
entropy_array = cp.array(entropy_log)
coherence_array = cp.array(coherence_log)

# Estimate resonance strength ⟨ρ⟩ as average amplitude squared (you can redefine this if needed)
rho_array = cp.array(rho_log)
# Compute ensemble statistics
avg_S = float(cp.mean(entropy_array))
std_S = float(cp.std(entropy_array))
avg_xi = float(cp.mean(coherence_array))
std_xi = float(cp.std(coherence_array))
avg_rho = float(cp.mean(rho_array))
std_rho = float(cp.std(rho_array))

# Print results
print("\n=== Ensemble-Averaged Results ===")
print(f"Average Entropy ⟨S⟩: {avg_S:.5f}")
print(f"Entropy Std Dev: {std_S:.5f}")
print(f"Average Coherence Length ⟨ξ⟩: {avg_xi:.5f}")
print(f"Coherence Std Dev: {std_xi:.5f}")
print(f"Average Resonance Strength ⟨ρ⟩: {avg_rho:.5f}")
print(f"Resonance Std Dev: {std_rho:.5f}")

print("\n=== Simulation Complete ===")
print(f"Snapshots saved to {output_dir}/")
print("Entropy and coherence logs saved.")