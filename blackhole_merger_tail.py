import cupy as cp
import matplotlib.pyplot as plt

# Grid setup
nx, nt = 512, 1000
Lx = 100.0
x = cp.linspace(0, Lx, nx)
dx = x[1] - x[0]
dt = 0.02
c = 1.0
theta = 0.0026  # WCT curvature feedback strength
epsilon = 1e-6

# Initialize wavefield: a localized initial ringdown profile
psi = cp.exp(-((x - Lx/2)**2) / 10.0) * cp.cos(5 * cp.pi * x / Lx)
psi_old = cp.copy(psi)
psi_new = cp.zeros_like(psi)

# Laplacian (1D second derivative)
def laplacian(f):
    return (cp.roll(f, -1) + cp.roll(f, 1) - 2 * f) / dx**2

# Record wave amplitude at midpoint over time
mid_index = nx // 2
tail_history = []

# Main loop
for t in range(nt):
    lap_psi = laplacian(psi)
    feedback = (theta * lap_psi / (psi**2 + epsilon)) ** 2
    psi_new = (
        2 * psi - psi_old + (c**2 * dt**2) * (lap_psi - feedback)
    )
    psi_old = cp.copy(psi)
    psi = cp.copy(psi_new)
    tail_history.append(psi[mid_index].item())

# Plot tail decay curve
plt.figure(figsize=(8, 4))
plt.plot(cp.asnumpy(cp.arange(nt) * dt), tail_history, label='WCT Tail')
plt.xlabel("Time (t)")
plt.ylabel("Amplitude at Center")
plt.title("Gravitational Wave Tail with WCT Curvature Feedback")
plt.grid(True)
plt.tight_layout()
plt.savefig("blackhole_tail_feedback.png", dpi=300)
plt.show()
