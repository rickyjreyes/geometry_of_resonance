import cupy as cp
import matplotlib.pyplot as plt

# Grid
nx = 512
x = cp.linspace(-10, 10, nx)
dx = x[1] - x[0]

# Initial localized wavefunction
psi = cp.exp(-x**2)
entropy_stress = 0.01  # sigma value

# Simulate decay over time due to entropy curvature
timesteps = 100
decay_log = []

for _ in range(timesteps):
    curvature = -cp.gradient(cp.gradient(psi, dx), dx)
    decay = entropy_stress * curvature * psi
    psi += -0.01 * decay  # simple Euler update
    decay_log.append(cp.sum(cp.abs(psi)**2).get())

# Plot
plt.plot(decay_log)
plt.xlabel("Time Step")
plt.ylabel("Wavefunction Norm")
plt.title("Entropy-Induced Decay of Confined Wave Packet")
plt.grid()
plt.tight_layout()
plt.show()
