import numpy as np
import matplotlib.pyplot as plt

# Grid setup
N = 128
L = 10.0
x = np.linspace(-L / 2, L / 2, N)
y = np.linspace(-L / 2, L / 2, N)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)

# Initial wavefunction: Gaussian with phase twist (like a vortex ring projection)
r = np.sqrt(X**2 + Y**2)
theta = np.arctan2(Y, X)
psi = np.exp(-r**2) * np.exp(1j * 3 * theta)

# Parameters
epsilon = 1e-3
alpha = 0.8
kappa = 1.0
dt = 0.01
steps = 100

# Storage for tracking effective mass
mass_history = []

# Time evolution loop
for t in range(steps):
    laplacian_psi = (
        np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0) +
        np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1) -
        4 * psi
    ) / dx**2

    # Curvature vector field C
    denom = psi + epsilon * np.exp(-alpha * np.abs(psi)**2)
    curvature_potential = laplacian_psi / denom
    grad_Cx = (np.roll(curvature_potential, -1, axis=1) - np.roll(curvature_potential, 1, axis=1)) / (2 * dx)
    grad_Cy = (np.roll(curvature_potential, -1, axis=0) - np.roll(curvature_potential, 1, axis=0)) / (2 * dy)
    Cx, Cy = grad_Cx, grad_Cy

    # Curl of curvature field (pseudo-z component in 2D)
    curl_C = (np.roll(Cy, -1, axis=1) - np.roll(Cy, 1, axis=1)) / (2 * dx) - \
             (np.roll(Cx, -1, axis=0) - np.roll(Cx, 1, axis=0)) / (2 * dy)

    # Evolve psi using simplified curvature-induced term
    dpsi_dt = curl_C
    psi += 1j * dt * dpsi_dt  # using Schrödinger-like update

    # Calculate curvature energy density and accumulate effective mass
    curvature_magnitude_sq = Cx**2 + Cy**2
    m_eff = np.sum(curvature_magnitude_sq) * dx * dy
    mass_history.append(m_eff)

# Plot mass accumulation over time
plt.figure(figsize=(6, 4))
plt.plot(np.arange(steps) * dt, mass_history)
plt.xlabel('Time')
plt.ylabel('Effective Mass $m_{eff}$')
plt.title('Curvature-Confinement Mass Accumulation Over Time')
plt.grid(True)
plt.tight_layout()
plt.show()
