import cupy as cp
import matplotlib.pyplot as plt

# -----------------------------
# 1D Wave Decay Under Curvature
# -----------------------------
# Decay model: dψ/dt ∝ −σ ∇²ψ · ψ
# Represents curvature-driven dissipation (e.g., entropy stress in confined systems)

# Grid
nx = 512
x = cp.linspace(-10, 10, nx)
dx = x[1] - x[0]

# Initial localized wavefunction (Gaussian)
psi = cp.exp(-x**2)
entropy_stress = 0.01  # σ value (entropy-induced curvature coupling)

# Normalize initial energy
initial_norm = cp.sum(cp.abs(psi)**2).get()
decay_log = [1.0]  # Relative to initial norm

# Time evolution (Euler method)
timesteps = 100
for _ in range(timesteps):
    curvature = -cp.gradient(cp.gradient(psi, dx), dx)  # ∇²ψ
    decay = entropy_stress * curvature * psi            # −σ ∇²ψ · ψ
    psi += -0.01 * decay                                # Euler update
    current_norm = cp.sum(cp.abs(psi)**2).get()
    decay_log.append(current_norm / initial_norm)       # Relative decay

# Plot relative norm over time
plt.figure(figsize=(8, 4))
plt.plot(decay_log, label="Relative Wavefunction Norm")
plt.xlabel("Time Step")
plt.ylabel("Norm (Relative to Initial)")
plt.title("Entropy-Induced Decay of Confined Wave Packet")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()
