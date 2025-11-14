import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

# Grid and time setup
nx, ny = 256, 256
Lx, Ly = 5e-6, 5e-6
dx, dy = Lx / nx, Ly / ny
dt = 1e-17
nt = 200
t = cp.linspace(0, nt * dt, nt)

# Wave properties
wavelength = 532e-9
c = 3e8
n_water = 1.33
v_light = c / n_water
omega = 2 * cp.pi * v_light / wavelength
cooling_rate = 1e13
damping = cp.exp(-cooling_rate * t)

# Field and curvature
psi = cp.random.rand(ny, nx) * 0.07
delta_n = 1.49e-5
shift_speed = 0.1
base_random = cp.random.normal(0, 1, size=(ny, nx))
x_cp = cp.linspace(0, Lx, nx)

# Disruption point
disrupt_frame = 120

com_over_time = []
curv_over_time = []
energy_over_time = []

for ti in range(nt):
    shift = int(shift_speed * ti)

    # Disruption event: invert curvature mid-run
    if ti == disrupt_frame:
        base_random *= -1  # simulate rapid curvature flip

    shifted = cp.roll(base_random, shift, axis=1)
    n_drag = n_water + delta_n * shifted
    k_drag = 2 * cp.pi * n_drag / wavelength
    wave = cp.sin(k_drag * x_cp[None, :]) * cp.sin(omega * t[ti]) * damping[ti]
    psi_t = psi * wave

    # Center of mass
    intensity = cp.abs(psi_t)**2
    total_intensity = cp.sum(intensity)
    com_x = cp.sum(intensity * x_cp[None, :]) / total_intensity
    com_over_time.append(cp.asnumpy(com_x))

    # Curvature COM
    curvature_profile = cp.sum(shifted, axis=0)
    curvature_com = cp.sum(curvature_profile * x_cp) / cp.sum(curvature_profile)
    curv_over_time.append(cp.asnumpy(curvature_com))

    # Energy
    epsilon_0 = 8.854e-12
    epsilon_r = n_water**2
    epsilon = epsilon_0 * epsilon_r
    energy_density = 0.5 * epsilon * cp.abs(psi_t)**2
    total_energy = cp.sum(energy_density) * dx * dy
    energy_over_time.append(cp.asnumpy(total_energy))

# Convert to arrays
com_array = np.array(com_over_time)
curv_array = np.array(curv_over_time)
energy_array = np.array(energy_over_time)
time_fs = np.linspace(0, nt * dt * 1e15, nt)

# Plot COM vs curvature
plt.figure(figsize=(10, 4))
plt.plot(time_fs, curv_array * 1e6, '--', label='Curvature Center (Dragged)')
plt.plot(time_fs, com_array * 1e6, label='Wave Center of Mass')
plt.axvline(time_fs[disrupt_frame], color='r', linestyle=':', label='Curvature Disruption')
plt.xlabel('Time (fs)')
plt.ylabel('Position (microns)')
plt.title('Wave COM vs Curvature Center with Disruption')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Plot energy over time
plt.figure(figsize=(8, 4))
plt.plot(time_fs, energy_array * 1e15)
plt.axvline(time_fs[disrupt_frame], color='r', linestyle=':', label='Curvature Disruption')
plt.xlabel("Time (fs)")
plt.ylabel("Total Field Energy (fJ)")
plt.title("Energy Evolution During Curvature Disruption")
plt.grid()
plt.tight_layout()
plt.legend()
plt.show()
