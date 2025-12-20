import numpy as np
import matplotlib.pyplot as plt
import os

# Create output folder
output_dir = "mass_outputs"
os.makedirs(output_dir, exist_ok=True)

# Grid setup
nx, ny = 200, 200
x = np.linspace(-10, 10, nx)
y = np.linspace(-10, 10, ny)
dx = x[1] - x[0]
X, Y = np.meshgrid(x, y)

# Time parameters
dt = 0.01
num_steps = 300
c = 1.0
lambda_factor = 0.05

# Phase initialization
phi1, phi2, phi3 = 0, np.pi/2, np.pi
wave_info_labels = ['001', '110', '011']  # symbolic bit labels

# Wave motion parameters
r = 3.0
omega = 0.2

# Wavefields
psi_old = np.zeros_like(X)
psi = np.zeros_like(X)
psi_new = np.zeros_like(X)

# Localized confinement wells
localized_mass = np.exp(-(X**2 + Y**2) / 0.5)
nested_mass = np.exp(-(X**2 + Y**2) / 0.2)

# Track metrics
energy_log, curvature_log, entropy_log = [], [], []
mass_accum = np.zeros_like(X)

# Constants
gamma, delta = 0.05, 0.01  # mass accumulation
info_map = np.random.uniform(0.5, 1.5, size=psi.shape)  # symbolic info feedback

# Plot setup
plt.ion()
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
im_psi = axs[0, 0].imshow(psi, cmap='viridis', origin='lower', extent=[-10, 10, -10, 10])
axs[0, 0].set_title("Wavefunction ψ(x, y, t)")
im_energy = axs[0, 1].imshow(psi**2, cmap='hot', origin='lower', extent=[-10, 10, -10, 10])
axs[0, 1].set_title("Energy Density ε = ψ²")
im_force = axs[0, 2].imshow(np.zeros_like(psi), cmap='coolwarm', origin='lower')
axs[0, 2].set_title("Emergent Force |F_eff|")
im_curvature = axs[1, 0].imshow(np.zeros_like(psi), cmap='plasma', origin='lower')
axs[1, 0].set_title("Ricci-like Curvature Rψ")
quiver_ax = axs[1, 1]
im_mass = axs[1, 2].imshow(mass_accum, cmap='cividis', origin='lower')
axs[1, 2].set_title("Accumulated Mass")

# Simulation loop
for t in range(num_steps):
    # Wave motion
    cx1 = -r * np.cos(omega * t)
    cx2 = r * np.cos(omega * t)
    cy3 = -r * np.sin(omega * t)

    wave1 = np.exp(-((X - cx1)**2 + Y**2) / 1.0) * np.cos(3 * (X - cx1) + phi1)
    wave2 = np.exp(-((X - cx2)**2 + Y**2) / 1.0) * np.cos(3 * (X - cx2) + phi2)
    wave3 = np.exp(-(X**2 + (Y - cy3)**2) / 1.0) * np.cos(3 * (Y - cy3) + phi3)

    psi = wave1 + wave2 + wave3
    psi += 0.5 * localized_mass * np.cos(3 * X)
    psi += 0.3 * nested_mass * np.sin(5 * Y)  # nested confinement

    # Laplacian
    laplacian = (
        np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0) +
        np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1) - 4 * psi
    ) / dx**2

    # Curvature
    with np.errstate(divide='ignore', invalid='ignore'):
        curvature = -laplacian / psi
        curvature[~np.isfinite(curvature)] = 0
        curvature = np.clip(curvature, -10, 10)

    # Info feedback
    curvature += 0.01 * info_map  # info-induced curvature tension

    # Field update
    psi_new = 2 * psi - psi_old + c**2 * dt**2 * (
        laplacian + lambda_factor * psi**3 + 0.01 * curvature)

    # Radiation dampening
    threshold = 5.0
    psi_new[np.abs(psi_new) > threshold] *= 0.5

    # Mass accumulation
    energy_density = psi**2
    mass_accum += gamma * energy_density * dt - delta * mass_accum * dt

    # Force
    grad_ex, grad_ey = np.gradient(energy_density, dx)
    force_magnitude = np.sqrt(grad_ex**2 + grad_ey**2)

    # Entropy (symbolic)
    prob_density = energy_density / (np.sum(energy_density) + 1e-10)
    entropy = -np.sum(prob_density * np.log(prob_density + 1e-12))

    # Log
    energy_log.append(np.sum(energy_density))
    curvature_log.append(np.sum(np.abs(curvature)))
    entropy_log.append(entropy)

    # Update visualizations
    im_psi.set_data(psi)
    im_energy.set_data(energy_density)
    im_force.set_data(force_magnitude)
    im_curvature.set_data(curvature)
    im_mass.set_data(mass_accum)

    im_psi.set_clim(psi.min(), psi.max())
    im_energy.set_clim(energy_density.min(), energy_density.max())
    im_force.set_clim(force_magnitude.min(), force_magnitude.max())
    im_curvature.set_clim(curvature.min(), curvature.max())
    im_mass.set_clim(mass_accum.min(), mass_accum.max())

    quiver_ax.clear()
    norm = force_magnitude + 1e-10
    quiver_ax.quiver(X[::5, ::5], Y[::5, ::5],
                     -grad_ex[::5, ::5] / norm[::5, ::5],
                     -grad_ey[::5, ::5] / norm[::5, ::5],
                     color='white', scale=2, width=0.003)
    quiver_ax.set_title("Vector Field F_eff")
    quiver_ax.set_xlim(-10, 10)
    quiver_ax.set_ylim(-10, 10)

    fig.suptitle(f"Wave Simulation – Step {t} | Entropy: {entropy:.3f}", fontsize=14)
    plt.pause(0.01)

    # Save frame
    if t % 20 == 0:
        frame_filename = os.path.join(output_dir, f"frame_{t:03d}.png")
        plt.savefig(frame_filename)

    # Shift states
    psi_old = psi.copy()
    psi = psi_new.copy()

plt.ioff()
plt.show()
