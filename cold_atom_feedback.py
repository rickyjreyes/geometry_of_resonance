import cupy as cp
import matplotlib.pyplot as plt

# Parameters
nx, ny = 200, 200
Lx, Ly = 10.0, 10.0
x = cp.linspace(-Lx / 2, Lx / 2, nx)
y = cp.linspace(-Ly / 2, Ly / 2, ny)
X, Y = cp.meshgrid(x, y)
dx = x[1] - x[0]

# Trap curvature parameters
k_trap = 1.0           # Harmonic trap strength
theta = 0.0026         # WCT curvature coupling
alpha = 1.5            # Regularization factor
epsilon = 1e-6         # To avoid division by zero

# Initial atom density (localized)
density = cp.exp(-(X**2 + Y**2))

# Laplacian using 2D central difference
def laplacian(f):
    return (
        cp.roll(f, 1, axis=0) + cp.roll(f, -1, axis=0) +
        cp.roll(f, 1, axis=1) + cp.roll(f, -1, axis=1) - 4 * f
    ) / dx**2

# Compute curvature feedback term
lap_dens = laplacian(density)
feedback = (theta * lap_dens / (density + epsilon)) ** 2

# Harmonic potential
V_harm = 0.5 * k_trap * (X**2 + Y**2)

# Total effective potential
V_eff = V_harm + feedback

# Plotting
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(cp.asnumpy(density), extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], cmap='viridis')
plt.title("Atom Density")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(cp.asnumpy(V_harm), extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], cmap='magma')
plt.title("Harmonic Trap Potential")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(cp.asnumpy(V_eff), extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], cmap='inferno')
plt.title("Effective Potential (WCT + Harmonic)")
plt.colorbar()

plt.suptitle("Cold Atom Trap with Curvature Feedback")
plt.tight_layout()
plt.savefig("cold_atom_feedback.png", dpi=300)
plt.show()
