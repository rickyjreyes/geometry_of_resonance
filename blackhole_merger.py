import cupy as cp
import matplotlib.pyplot as plt

# Time evolution of a scalar waveform with nonlinear decay
nt = 5000
t = cp.linspace(0, 10, nt)
theta = 1e-12
initial_amplitude = 1.0

# Simulate tail with theta-dependent nonlinear damping
waveform = initial_amplitude * cp.exp(-t) / (1 + theta * t**2)

# Plot
plt.plot(t.get(), waveform.get())
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Nonlinear Gravitational Wave Tail")
plt.grid()
plt.tight_layout()
plt.show()
