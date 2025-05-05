import numpy as np
import matplotlib.pyplot as plt

# Grid setup
nx, ny = 256, 256
Lx, Ly = 10.0, 10.0
dx, dy = Lx / nx, Ly / ny
dt = 0.0002
nt = 1000

alpha = 2.0
theta = 0.0026
damping = 0.00005
noise_level = 1e-5

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

psi = 0.07 * np.random.rand(ny, nx)
psi += 0.01 * np.sin(2 * np.pi * X / Lx)
psi_old = np.copy(psi)

# First Pass Entropy
# def laplacian_2d(f):
#     return (
#         -f + 16*(np.roll(f, -1, axis=0) + np.roll(f, 1, axis=0)) - (np.roll(f, -2, axis=0) + np.roll(f, 2, axis=0))
#         + -f + 16*(np.roll(f, -1, axis=1) + np.roll(f, 1, axis=1)) - (np.roll(f, -2, axis=1) + np.roll(f, 2, axis=1))
#     ) / (12 * dx**2)

# Chaotic Entropy
def laplacian_2d(f):
    return (
        (-30*f + 16*(np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0)) - (np.roll(f, 2, axis=0) + np.roll(f, -2, axis=0)))
        + (-30*f + 16*(np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1)) - (np.roll(f, 2, axis=1) + np.roll(f, -2, axis=1)))
    ) / (12 * dx**2)


def compute_entropy(psi):
    prob_density = np.abs(psi)**2
    prob_density /= np.sum(prob_density)
    entropy = -np.sum(prob_density * np.log(prob_density + 1e-12))
    return entropy

entropy_log = []
curvature_energy_log = []
snapshots = []

for step in range(nt):
    lap = laplacian_2d(psi)
    W_reg = -lap / (psi + 1e-8 + np.exp(-alpha * np.abs(psi)**2))
    energy_density = np.abs(psi)**2
    curvature_energy = W_reg * energy_density

    psi_new = (
        2 * psi - psi_old +
        dt**2 * (lap - theta * W_reg + noise_level * np.random.randn(ny, nx))
        - damping * (psi - psi_old)
    )

    psi_old = psi
    psi = psi_new

    if step % 50 == 0:
        entropy_log.append(compute_entropy(psi))
        curvature_energy_log.append(np.mean(curvature_energy))
        snapshots.append(psi.copy())

# Plot
fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

axs[0].plot(range(0, nt, 50), entropy_log, label="Entropy S(t)")
axs[0].set_ylabel("Entropy")
axs[0].grid(True)
axs[0].legend()

axs[1].plot(range(0, nt, 50), curvature_energy_log, color='orange', label="⟨Wψ · |ψ|²⟩")
axs[1].set_xlabel("Time step")
axs[1].set_ylabel("Curvature Energy")
axs[1].grid(True)
axs[1].legend()

plt.suptitle("WCT Diagnostics: Entropy and Curvature Energy")
plt.tight_layout()
plt.show()
