# Retry using a more memory-safe method and limit data points
import numpy as np
import matplotlib.pyplot as plt

# Constants
c = 2.998e8         # Speed of light (m/s)
hbar = 1.055e-34    # Reduced Planck constant (J·s)

# Parameters
xi_0 = 1e-5         # Initial coherence scale (m)
xi_max = 1.0        # Max coherence scale (m)
k = 1e-17           # Growth rate (1/s)
t_max = 5e17        # Total time (s)
dt = 1e15           # Time step (s)

# Create time array with fewer points for plotting
t = np.arange(0, t_max + dt, dt)
n = len(t)

xi = np.zeros(n)
xi[0] = xi_0

# Compute logistic growth of xi(t)
for i in range(1, n):
    xi[i] = xi[i-1] + dt * k * xi[i-1] * (1 - xi[i-1] / xi_max)

# Compute derived quantities
rho_vac = hbar * c / xi**4
Lambda = 1 / xi**2
H = np.sqrt((8 * np.pi / 3) * (c**4 / xi**3))  # Simplified form from earlier

# Plotting results
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axs[0].plot(t / 1e17, xi, label="ξ(t)")
axs[0].set_ylabel("Coherence ξ (m)")
axs[0].set_title("WCT Coherence Growth and Cosmic Expansion")
axs[0].grid(True)

axs[1].plot(t / 1e17, rho_vac, label="ρ_vac(t)", color="darkred")
axs[1].set_ylabel("ρ_vac (J/m³)")
axs[1].set_yscale("log")
axs[1].grid(True)

axs[2].plot(t / 1e17, H, label="H(t)", color="darkgreen")
axs[2].set_ylabel("H (1/s)")
axs[2].set_xlabel("Time (×10¹⁷ s)")
axs[2].set_yscale("log")
axs[2].grid(True)

plt.tight_layout()
plt.show()
