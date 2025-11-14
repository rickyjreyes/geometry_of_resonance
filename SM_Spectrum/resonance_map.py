import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter

# Physical constants
c = 3e8  # Speed of light (m/s)
hbar = 1.0545718e-34  # Reduced Planck constant (J·s)
lambda_0 = 532e-9  # Wavelength (m)
k0 = 2 * np.pi / lambda_0  # Wave number (1/m)

# Wave Confinement Theory constants
alpha_curvature = 1e-7  # Linear curvature feedback strength
theta = 1e-120  # Nonlinear curvature feedback coefficient
xi = 112e-6  # Vacuum coherence length scale (m)

# Cavity size
L = 336e-6  # Length of cavity (m)

# Simulation parameters
Nx = 1000  # Number of spatial points
Nt = 2000  # Number of time steps
dx = L / Nx
dt = 1e-16  # Time step size (s)

# Spatial grid
x = np.linspace(0, L, Nx)

# Fields
psi = np.exp(1j * k0 * x)  # Initial 650 nm standing wave
psi_old = psi.copy()
psi_new = np.zeros_like(psi, dtype=complex)

# Output folder
output_dir = "wct_standing_wave"
os.makedirs(output_dir, exist_ok=True)

# Helper function: curvature feedback term
def curvature_feedback(psi, dx):
    laplacian = (np.roll(psi, -1) - 2 * psi + np.roll(psi, 1)) / dx**2
    curvature = -laplacian / (psi + 1e-12)
    return curvature

# Time evolution
for n in range(Nt):
    # Curvature feedback term
    W_psi = curvature_feedback(psi, dx)

    # Apply vacuum coherence smoothing
    xi_smoothing = xi  # coherence scale in meters
    W_psi = gaussian_filter(W_psi.real, sigma=xi_smoothing / dx) + 1j * gaussian_filter(W_psi.imag, sigma=xi_smoothing / dx)

    # Nonlinear feedback term
    nonlinear_feedback = theta * (W_psi)**2

    # Update using wave equation + curvature feedback + nonlinear term
    psi_new = (2 * psi - psi_old
               + c**2 * dt**2 * (np.roll(psi, -1) - 2 * psi + np.roll(psi, 1)) / dx**2
               - alpha_curvature * dt**2 * (W_psi + nonlinear_feedback) * psi)

    # Rotate fields
    psi_old = psi.copy()
    psi = psi_new.copy()

    # Save snapshots
    if n % 100 == 0:
        plt.figure(figsize=(10, 4))
        plt.plot(x * 1e6, np.real(psi), label='Re(psi)', alpha=0.7)
        plt.plot(x * 1e6, np.imag(psi), label='Im(psi)', alpha=0.7)
        plt.title(f"Time step {n}")
        plt.xlabel("x (microns)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/wave_{n:05d}.png")
        plt.close()

print("Simulation complete!")
