
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------
# Simulation Parameters
# -----------------------
nx, ny = 128, 128
Lx, Ly = 10.0, 10.0
dx, dy = Lx / nx, Ly / ny
alpha = 1.5
theta = 0.0026
kappa = 0.1
damping = 0.00002
epsilon = 1e-6
noise_level = 1e-5

# -----------------------
# Initial Field ψ₀ Setup
# -----------------------
cp.random.seed(42)
X, Y = cp.meshgrid(cp.linspace(0, 2 * cp.pi, nx), cp.linspace(0, 2 * cp.pi, ny))
noise = cp.random.randn(nx, ny)
psi0 = cp.sin(X) * cp.cos(Y) + noise_level * noise

# -----------------------
# Laplacian Operator
# -----------------------
def laplacian(f):
    return (
        cp.roll(f, 1, axis=0) + cp.roll(f, -1, axis=0) +
        cp.roll(f, 1, axis=1) + cp.roll(f, -1, axis=1) -
        4 * f
    ) / dx**2

lap_psi0 = laplacian(psi0)

# -----------------------
# Feedback Construction
# -----------------------
denom = psi0**2 + epsilon * cp.exp(-alpha * psi0**2)
denom = cp.maximum(denom, epsilon)
lap_term = laplacian(psi0)
safe_psi0 = psi0 + 1e-6 * cp.sign(psi0)  # Smooth zero-avoidance

feedback_term = (2 * kappa * lap_psi0 / denom) * (
    (lap_term / safe_psi0) - (2 * psi0 * lap_psi0 / denom)
)
geo_term = (theta * lap_psi0 / denom) ** 2
feedback_term = cp.clip(feedback_term, -1e2, 1e2)

# -----------------------
# Diagonal Proxy Spectrum
# -----------------------
diagonal = (-lap_psi0 + feedback_term + geo_term +
            2 * kappa**2 * (lap_psi0**2) / (denom**2)).flatten()
eigvals = cp.asnumpy(cp.sort(diagonal))

# -----------------------
# Histogram, Entropy, CSV
# -----------------------
hist, bins = np.histogram(eigvals, bins=60)
bin_centers = 0.5 * (bins[1:] + bins[:-1])
prob_density = hist / np.sum(hist) if np.sum(hist) > 0 else hist
spec_entropy = -np.sum(prob_density * np.log(prob_density + 1e-12))

# Save to CSV
pd.DataFrame({'eigenvalue': eigvals}).to_csv("approx_eigenvalues.csv", index=False)
print("Saved: approx_eigenvalues.csv")

# Plot histogram with peak highlighted
peak_bin = np.argmax(hist)
colors = ['crimson' if i == peak_bin else 'navy' for i in range(len(hist))]

plt.figure(figsize=(8, 4))
plt.bar(bin_centers, hist, width=np.diff(bins)[0], color=colors, alpha=0.8)
plt.yscale('log')
plt.xlabel("Eigenvalue")
plt.ylabel("Frequency (log scale)")
plt.title(f"Wave Confinement Eigenmode Spectrum (Approx)\nSpectral Entropy: {spec_entropy:.4f}")
plt.grid(True)
plt.tight_layout()
plt.savefig("approx_spectrum_gpu.png")
print("Saved: approx_spectrum_gpu.png")

# -----------------------
# Visualize Fields
# -----------------------
plt.figure()
plt.imshow(cp.asnumpy(feedback_term), cmap='viridis')
plt.title("Feedback Term Field")
plt.colorbar()
plt.tight_layout()
plt.savefig("feedback_field.png")
print("Saved: feedback_field.png")

plt.figure()
plt.imshow(cp.asnumpy(geo_term), cmap='magma')
plt.title("Geometric Feedback Field")
plt.colorbar()
plt.tight_layout()
plt.savefig("geo_field.png")
print("Saved: geo_field.png")
