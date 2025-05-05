import cupy as cp
import cupyx.scipy.fft as fft
import matplotlib.pyplot as plt
import os

# -------------------------
# Simulation Parameters
# -------------------------
nx, ny, nz = 128, 128, 128
Lx, Ly, Lz = 10.0, 10.0, 10.0
dx, dy, dz = Lx / nx, Ly / ny, Lz / nz
dt = 0.0001
nt = 100000
c = 1.0
alpha = 2.0
theta = 0.002
damping = 0.00001

save_interval = 1000
output_dir = "spacetime_emergence_outputs_3d"
os.makedirs(output_dir, exist_ok=True)

# -------------------------
# Helper Functions
# -------------------------
def laplacian_3d_4th(f, dx, dy, dz):
    return (
        (-f + 16 * (cp.roll(f, -1, axis=0) + cp.roll(f, 1, axis=0)) -
         (cp.roll(f, -2, axis=0) + cp.roll(f, 2, axis=0))) / (12 * dx**2) +
        (-f + 16 * (cp.roll(f, -1, axis=1) + cp.roll(f, 1, axis=1)) -
         (cp.roll(f, -2, axis=1) + cp.roll(f, 2, axis=1))) / (12 * dy**2) +
        (-f + 16 * (cp.roll(f, -1, axis=2) + cp.roll(f, 1, axis=2)) -
         (cp.roll(f, -2, axis=2) + cp.roll(f, 2, axis=2))) / (12 * dz**2)
    )

def compute_Wpsi(psi, dx, dy, dz):
    lap = laplacian_3d_4th(psi, dx, dy, dz)
    W = -lap / (psi + 1e-8)
    return cp.clip(W, -1e5, 1e5)

def entropy(psi):
    p = cp.abs(psi)**2
    p /= cp.sum(p) + 1e-12
    return -cp.sum(p * cp.log(p + 1e-12))

def clip_field(psi, limit=100):
    return cp.clip(psi, -limit, limit)

# -------------------------
# Initialize Field
# -------------------------
x = cp.linspace(-Lx/2, Lx/2, nx)
y = cp.linspace(-Ly/2, Ly/2, ny)
z = cp.linspace(-Lz/2, Lz/2, nz)
X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')

sigma = 0.2
amplitude = 0.01
psi = amplitude * cp.exp(-(X**2 + Y**2 + Z**2) / (2 * sigma**2))
psi_old = cp.copy(psi)

entropy_log = []

# -------------------------
# Main Evolution Loop
# -------------------------
for t in range(nt):
    Wpsi = compute_Wpsi(psi, dx, dy, dz)
    lap = laplacian_3d_4th(psi, dx, dy, dz)

    psi_new = (2 - damping * dt) * psi - psi_old + (c * dt)**2 * (
        lap - alpha * Wpsi * psi - theta * (cp.tanh(Wpsi)**2) * psi
    )

    # Simple smoothing step
    psi_new = 0.9995 * psi_new + 0.00025 * (
        cp.roll(psi_new, 1, axis=0) + cp.roll(psi_new, -1, axis=0) +
        cp.roll(psi_new, 1, axis=1) + cp.roll(psi_new, -1, axis=1) +
        cp.roll(psi_new, 1, axis=2) + cp.roll(psi_new, -1, axis=2)
    ) / 3

    # psi_new = clip_field(psi_new)
    psi_new = clip_field(psi_new, limit=5)

    psi_old = cp.copy(psi)
    psi = cp.copy(psi_new)

    if t % 1000 == 0:
        entropy_log.append(float(entropy(psi)))

    if t % save_interval == 0:
        filename = os.path.join(output_dir, f"wave_snapshot_{t:07d}.npy")
        cp.save(filename, psi.get())
        print(f"Saved snapshot at t = {t}")

# -------------------------
# Post-Processing
# -------------------------
cp.save(os.path.join(output_dir, "entropy_log.npy"), cp.array(entropy_log))
print("\n=== Simulation Complete ===")
print(f"Snapshots saved to {output_dir}/")