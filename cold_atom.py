import cupy as cp
import matplotlib.pyplot as plt

# Parameters
nx = 256
x = cp.linspace(-5, 5, nx)
dx = x[1] - x[0]
psi = cp.exp(-x**2)  # Ground state wavefunction

# Feedback curvature potential: parabolic + nonlinear
kappa = 0.1
curv_feedback = kappa * (x**2 + 0.1 * cp.sin(2 * cp.pi * x))

# Effective potential
V_eff = 0.5 * x**2 + curv_feedback

# Plot
plt.plot(x.get(), V_eff.get(), label='Trap + Feedback')
plt.title("Cold Atom Trap with Curvature Feedback")
plt.xlabel("x")
plt.ylabel("V(x)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
