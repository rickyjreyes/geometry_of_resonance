import cupy as cp
import matplotlib.pyplot as plt
from cupy.linalg import eigvalsh

# -----------------------
# Simulation Parameters
# -----------------------
nx, ny = 128, 128
Lx, Ly = 10.0, 10.0
dx, dy = Lx / nx, Ly / ny
alpha = 1.5            # Curvature confinement strength
theta = 0.0026         # Geometric feedback scale
kappa = 0.1            # Resonance amplification (confinement coupling)
damping = 0.00002      # Field dissipation
epsilon = 1e-6         # Regularization (prevents divergence)
noise_level = 1e-5     # Random perturbation level

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
denom = psi0**2 + epsilon * cp.exp(-alpha * cp.abs(psi0)**2)
denom = cp.maximum(denom, epsilon)

lap_term = laplacian(psi0)

# Wave Confinement Feedback Term (nonlinear + curvature-regulated)
feedback_term = (2 * kappa * lap_psi0 / denom) * (
    (lap_term / psi0) - (2 * psi0 * lap_psi0 / denom)
)

# Geometric Feedback Term (Resonance Stabilizer)
geo_term = (theta * lap_psi0 / denom) ** 2

# Damping and NaN Stabilization
feedback_term = cp.nan_to_num(cp.clip(feedback_term, -1e2, 1e2), nan=0.0, posinf=1e10, neginf=-1e10)

# -----------------------
# Construct Operator Matrix
# -----------------------
O_operator = cp.zeros((nx * ny, nx * ny))

for i in range(nx * ny):
    row, col = divmod(i, ny)
    O_operator[i, i] = -lap_psi0[row, col] + feedback_term[row, col] + geo_term[row, col]

# -----------------------
# Eigenvalue Spectrum
# -----------------------
try:
    eigenvalues = eigvalsh(O_operator)
    ev_cpu = eigenvalues.get()

    # Visualization: Eigenvalue Distribution (log frequency)
    plt.figure(figsize=(8, 4))
    plt.hist(ev_cpu, bins=60, color='darkblue', alpha=0.8)
    plt.yscale('log')
    plt.xlabel("Eigenvalue")
    plt.ylabel("Frequency (log scale)")
    plt.title("Wave Confinement Eigenmode Spectrum")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

except cp.linalg.LinAlgError:
    print("Eigenvalue computation failed to converge.")
