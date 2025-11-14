import numpy as np
import matplotlib.pyplot as plt

# Grid and time settings
Nx, Ny = 100, 100
dx = dy = 0.1
dt = 0.01
steps = 100

# Wavefield initialization
psi = np.exp(-((np.linspace(-5, 5, Nx)[:, None])**2 + (np.linspace(-5, 5, Ny)[None, :])**2))

# Regularization parameters
epsilon = 1e-3
alpha = 0.1
kappa = 1.0
damping = 0.01

# Store mass accumulation
mass_over_time = []

# Laplacian operator
def laplacian(f):
    return (
        -4*f + np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0)
        + np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1)
    ) / dx**2

# Time evolution
for t in range(steps):
    lap = laplacian(psi)
    curvature = lap / (psi + epsilon * np.exp(-alpha * psi**2))
    C = np.gradient(curvature, dx, axis=0)**2 + np.gradient(curvature, dy, axis=1)**2
    C_magnitude_sq = C
    m_eff = np.sum(C_magnitude_sq) * dx * dy
    mass_over_time.append(m_eff)

    # Add damping
    psi -= dt * damping * psi

    # Curvature feedback
    psi += dt * kappa * curvature

# Plot the result
plt.figure(figsize=(8, 5))
plt.plot(np.linspace(0, 1, steps), mass_over_time)
plt.title("Stabilized Curvature-Confinement Mass Accumulation")
plt.xlabel("Time")
plt.ylabel("Effective Mass $m_{\\text{eff}}$")
plt.grid(True)
plt.tight_layout()
plt.show()
