
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import label

output_dir = "mass_outputs_longrun"
os.makedirs(output_dir, exist_ok=True)

nx, ny = 200, 200
x = np.linspace(-10, 10, nx)
y = np.linspace(-10, 10, ny)
dx = x[1] - x[0]
X, Y = np.meshgrid(x, y)

dt = 0.01
num_steps = 864000  # approx. 1 day @ 0.01 timestep
c = 1.0
lambda_factor = 0.2

phi = [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]
omega = 0.2
r = 3.0

psi_old = np.zeros_like(X)
psi = np.zeros_like(X)
psi_new = np.zeros_like(X)

localized_mass = np.exp(-(X**2 + Y**2) / 0.5)
nested_mass = np.exp(-(X**2 + Y**2) / 0.2)

energy_log, curvature_log, entropy_log, stable_mass_log = [], [], [], []
mass_accum = np.zeros_like(X)

gamma, delta = 0.05, 0.01
alpha_confinement = 0.6
info_map = np.random.uniform(0.5, 1.5, size=psi.shape)
stabilization_threshold = 0.8

for t in range(num_steps):
    psi = np.zeros_like(X)
    for i, phase in enumerate(phi):
        angle = 2 * np.pi * i / len(phi)
        cx = r * np.cos(omega * t + angle)
        cy = r * np.sin(omega * t + angle)
        wave = np.exp(-((X - cx)**2 + (Y - cy)**2)) * np.cos(3 * ((X - cx)*np.cos(angle) + (Y - cy)*np.sin(angle)) + phase)
        psi += wave

    psi += 0.5 * localized_mass * np.cos(3 * X)
    psi += 0.3 * nested_mass * np.sin(5 * Y)

    laplacian = (
        np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0) +
        np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1) - 4 * psi
    ) / dx**2

    with np.errstate(divide='ignore', invalid='ignore'):
        curvature = -laplacian / psi
        curvature[~np.isfinite(curvature)] = 0
        curvature = np.clip(curvature, -10, 10)

    curvature += 0.01 * info_map

    V = alpha_confinement * mass_accum
    grad_Vx, grad_Vy = np.gradient(V, dx)
    restoring_potential = grad_Vx + grad_Vy

    psi_new = 2 * psi - psi_old + c**2 * dt**2 * (
        laplacian + lambda_factor * psi**3 + 0.01 * curvature - restoring_potential)

    energy_density = psi**2
    mass_accum += gamma * energy_density * dt - delta * mass_accum * dt

    labeled, num_masses = label(mass_accum > stabilization_threshold)

    grad_ex, grad_ey = np.gradient(energy_density, dx)
    force_magnitude = np.sqrt(grad_ex**2 + grad_ey**2)

    prob_density = energy_density / (np.sum(energy_density) + 1e-10)
    entropy = -np.sum(prob_density * np.log(prob_density + 1e-12))

    if t % 100 == 0:
        with open(os.path.join(output_dir, "log.txt"), "a") as f:
            f.write(f"Step {t} | Energy: {np.sum(energy_density):.2f} | Curvature: {np.sum(np.abs(curvature)):.2f} | Entropy: {entropy:.4f} | Stable Masses: {num_masses}\n")
        np.savez_compressed(os.path.join(output_dir, f"state_{t:06d}.npz"),
                            psi=psi, mass=mass_accum, energy=energy_density)

    psi_old = psi.copy()
    psi = psi_new.copy()
