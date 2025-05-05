import cupy as cp
import matplotlib.pyplot as plt
from cupy.linalg import eigvalsh

# Parameters
nx, ny = 128, 128  # Larger grid for more eigenmodes
Lx, Ly = 10.0, 10.0
dx, dy = Lx / nx, Ly / ny
dt = 0.0002
nt = 750000
c = 1.0
alpha = 1.5
theta = 0.0026
damping_factor = 0.00002  # Damping to limit feedback growth
epsilon = 1e-6  # Regularization to avoid division by zero
kappa = 0.1  # Increased feedback strength to excite higher eigenmodes
noise_level = 1e-5  # Noise level

# Generate a smooth background field ψ₀ with randomness
cp.random.seed(42)  # Optional: Set seed for reproducibility
random_noise = cp.random.randn(nx, ny)  # Gaussian random field
X, Y = cp.meshgrid(cp.linspace(0, 2 * cp.pi, nx), cp.linspace(0, 2 * cp.pi, ny))
psi0 = cp.sin(X) * cp.cos(Y) + noise_level * random_noise  # Add randomness to the smooth background

# Discrete Laplacian (2nd-order central difference)
def laplacian(f):
    if f.ndim == 1:
        f = cp.expand_dims(f, axis=0)  # Convert to 2D if 1D
    return (
        cp.roll(f, 1, axis=0) + cp.roll(f, -1, axis=0) +
        cp.roll(f, 1, axis=1) + cp.roll(f, -1, axis=1) - 
        4 * f
    ) / dx**2  # Use dx/dy as the spatial step size

# Compute laplacian of psi₀
lap_psi0 = laplacian(psi0)

# Regularized denominator including exponential term
denom = psi0**2 + epsilon * cp.exp(-alpha * cp.abs(psi0)**2)

# Prevent division by zero by applying a small positive offset to denom
denom = cp.maximum(denom, epsilon)

# Modify the feedback term with regularization and damping
lap_psi0_term = laplacian(psi0)
feedback_factor = (2 * kappa * lap_psi0 / denom) * (
    (lap_psi0_term / psi0) - (2 * psi0 * lap_psi0 / denom)
)

# Apply a hard cutoff to limit large feedback values
feedback_factor = cp.clip(feedback_factor, -1e2, 1e2)  # Clamp feedback to a range

# Handle NaN or Inf values in feedback_factor by replacing them with small numbers
feedback_factor = cp.nan_to_num(feedback_factor, nan=0, posinf=1e10, neginf=-1e10)

# Geometric feedback term (second part of the Lagrangian)
geo_feedback = (theta * lap_psi0 / denom) ** 2

# Create the operator matrix O (2D grid flattened to 1D)
O_operator = cp.zeros((nx * ny, nx * ny))

# Fill the operator matrix with Laplacian-like terms and feedback
for i in range(nx * ny):
    row, col = divmod(i, ny)  # Convert 1D index to 2D grid indices
    O_operator[i, i] = -lap_psi0[row, col] + feedback_factor[row, col] + geo_feedback[row, col]  # Applying feedback to diagonal

# Try to compute eigenvalues of the operator O
try:
    eigenvalues = eigvalsh(O_operator)
    # Plot histogram of eigenvalues with log scale
    plt.figure(figsize=(8, 4))
    plt.hist(eigenvalues.get(), bins=50, color='blue', alpha=0.7)  # Use .get() to move data to CPU for plotting
    plt.yscale('log')  # Logarithmic scale for eigenvalues
    plt.title("Histogram of Eigenvalues (With Regularization and Damping)")
    plt.xlabel("Eigenvalue")
    plt.ylabel("Frequency (Log scale)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
except cp.linalg.LinAlgError:
    print("Eigenvalue computation did not converge. There may still be instability.")
