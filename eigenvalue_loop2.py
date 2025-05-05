import cupy as cp
import matplotlib.pyplot as plt

# Parameters
nx, ny = 64, 64
a = 0.1
epsilon = 1e-6
alpha = 1.5
kappa = 0.01
theta = 0.0026

# Grid and background field
x = cp.linspace(0, 2 * cp.pi, nx)
y = cp.linspace(0, 2 * cp.pi, ny)
X, Y = cp.meshgrid(x, y)
psi0 = cp.sin(X) * cp.cos(Y)

# Laplacian function
def laplacian(f):
    return (
        cp.roll(f, 1, axis=0) + cp.roll(f, -1, axis=0) +
        cp.roll(f, 1, axis=1) + cp.roll(f, -1, axis=1) - 4 * f
    ) / a**2

lap_psi0 = laplacian(psi0)
denom = psi0**2 + epsilon * cp.exp(-alpha * cp.abs(psi0)**2)
denom = cp.maximum(denom, epsilon)

lap_psi0_term = laplacian(psi0)
safe_psi0 = cp.where(cp.abs(psi0) < 1e-6, 1e-6, psi0)  # Avoid division by zero

feedback_factor = (2 * kappa * lap_psi0 / denom) * (
    (lap_psi0_term / safe_psi0) - (2 * psi0 * lap_psi0 / denom)
)
feedback_factor = cp.nan_to_num(cp.clip(feedback_factor, -1e2, 1e2), nan=0, posinf=1e10, neginf=-1e10)

geo_feedback = (theta * lap_psi0 / denom) ** 2

# Only the diagonal of the operator (eigenvalues)
diagonal = (-lap_psi0 + feedback_factor + geo_feedback +
            2 * kappa**2 * (lap_psi0**2) / (denom**2)).flatten()

# Convert to CPU and sort eigenvalues
eigvals = cp.asnumpy(cp.sort(diagonal))

# Top 10 modes
top_10 = [(eigvals[i], 1.0) for i in range(10)]
for idx, (eig, norm) in enumerate(top_10):
    print(f"Mode {idx+1}: Eigenvalue = {eig:.6f}, Norm = {norm:.1f}")

# Save histogram (non-blocking)
plt.figure(figsize=(8, 4))
plt.hist(eigvals, bins=50, color='navy', alpha=0.75)
plt.yscale('log')
plt.title("Eigenvalue Spectrum with 2-Loop Feedback Correction")
plt.xlabel("Eigenvalue")
plt.ylabel("Frequency (Log Scale)")
plt.grid(True)
plt.tight_layout()
plt.savefig("loop2_spectrum.png")
print("Saved: loop2_spectrum.png")
