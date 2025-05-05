import cupy as cp
import matplotlib.pyplot as plt

# Grid
nx, ny = 200, 200
Lx, Ly = 10.0, 10.0
x = cp.linspace(-Lx / 2, Lx / 2, nx)
y = cp.linspace(-Ly / 2, Ly / 2, ny)
X, Y = cp.meshgrid(x, y)
dx = x[1] - x[0]

# WCT background field psi0 (confined mode)
psi0 = cp.exp(-(X**2 + Y**2)) * cp.cos(2 * cp.pi * X / Lx)

# Effective birefringence from gradient energy density
grad_psi_x = cp.gradient(psi0, dx, axis=0)
grad_psi_y = cp.gradient(psi0, dx, axis=1)
index_shift = (grad_psi_x**2 + grad_psi_y**2) / (psi0**2 + 1e-6)

# Simulated light wave with polarization splitting
wavelength = 0.5
kx = 2 * cp.pi / wavelength
wave_unperturbed = cp.cos(kx * X)

# Apply birefringent index shift (WCT effective medium)
wave_shifted_x = wave_unperturbed * cp.exp(-0.5 * index_shift)
wave_shifted_y = wave_unperturbed * cp.exp(0.5 * index_shift)

# Interference pattern (e.g. in a polarimeter)
interference = wave_shifted_x - wave_shifted_y

# Move to CPU for plotting
wave_shifted_x_cpu = cp.asnumpy(wave_shifted_x)
wave_shifted_y_cpu = cp.asnumpy(wave_shifted_y)
interference_cpu = cp.asnumpy(interference)

# Plot the polarization-dependent phase shift
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.imshow(wave_shifted_x_cpu, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], cmap='inferno')
plt.title('X-Polarized Wave')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(wave_shifted_y_cpu, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], cmap='inferno')
plt.title('Y-Polarized Wave')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(interference_cpu, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], cmap='coolwarm')
plt.title('Birefringence Signal (Δφ)')
plt.colorbar()

plt.tight_layout()
plt.show()
