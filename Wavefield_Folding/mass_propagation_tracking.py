
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import label, center_of_mass, gaussian_filter
from matplotlib.animation import FuncAnimation

# Create output folder
output_dir = "mass_outputs_propagation"
os.makedirs(output_dir, exist_ok=True)

# Grid setup
nx, ny = 200, 200
x = np.linspace(-10, 10, nx)
y = np.linspace(-10, 10, ny)
dx = x[1] - x[0]
X, Y = np.meshgrid(x, y)

# Time parameters
dt = 0.01
num_steps = 5000  # shortened for quicker processing
c = 1.0
lambda_factor = 0.2

# Phase setup
phi_list = np.linspace(0, 2 * np.pi, 7)
r = 3.0
omega = 0.2

# Fields
psi_old = np.zeros_like(X)
psi = np.zeros_like(X)
psi_new = np.zeros_like(X)
localized_mass = np.exp(-(X**2 + Y**2) / 0.5)
nested_mass = np.exp(-(X**2 + Y**2) / 0.2)
mass_accum = np.zeros_like(X)

# Constants
gamma, delta = 0.05, 0.001
alpha_confinement = 0.6
info_map = np.random.uniform(0.5, 1.5, size=psi_old.shape)
stabilization_threshold = 0.6

# Logs
centroid_log = open(os.path.join(output_dir, "centroids.txt"), "w")

# Create video frames folder
video_frames_dir = os.path.join(output_dir, "frames")
os.makedirs(video_frames_dir, exist_ok=True)

for t in range(num_steps):
    psi = np.zeros_like(X)
    for i, phi in enumerate(phi_list):
        cx = r * np.cos(omega * t + phi)
        cy = r * np.sin(omega * t + phi)
        wave = np.exp(-((X - cx)**2 + (Y - cy)**2)) * np.cos(3 * (X - cx) + phi)
        psi += wave

    # Add multiple traveling waves
    for k in range(3):
        offset = k * 2
        travel_wave = np.exp(-((X + 4 - 0.01 * t - offset)**2 + (Y - 2 + k)**2)) * np.cos(6 * X - 0.1 * t)
        psi += 0.5 * travel_wave

    # Energy density and Laplacian
    energy_density = psi**2
    laplacian = (
        np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0) +
        np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1) - 4 * psi
    ) / dx**2
    with np.errstate(divide='ignore', invalid='ignore'):
        curvature = -laplacian / psi
        curvature[~np.isfinite(curvature)] = 0
    curvature += 0.01 * info_map

    nucleation_zone = energy_density > 1.5 * np.mean(energy_density)
    mass_accum += 0.01 * nucleation_zone.astype(float)

    # Centroid tracking
    labeled, num_masses = label(mass_accum > stabilization_threshold)
    if num_masses > 0:
        centroids = center_of_mass(mass_accum, labeled, range(1, num_masses + 1))
        for i, (cy, cx) in enumerate(centroids):
            centroid_log.write(f"{t},{i},{x[int(cx)]:.2f},{y[int(cy)]:.2f}\n")

    # Save frame image
    if t % 100 == 0:
        plt.figure(figsize=(5, 5))
        plt.imshow(mass_accum, cmap="cividis", origin="lower", extent=[-10, 10, -10, 10])
        plt.title(f"Mass Accum (Step {t})")
        plt.colorbar()
        plt.savefig(os.path.join(video_frames_dir, f"frame_{t:04d}.png"))
        plt.close()

    # Time update
    psi_new = 2 * psi - psi_old + c**2 * dt**2 * laplacian
    psi_old = psi.copy()
    psi = psi_new.copy()

centroid_log.close()
