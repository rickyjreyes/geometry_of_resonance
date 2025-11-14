import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import label, gaussian_filter

# Create output folder
output_dir = "mass_outputs_propagation"
os.makedirs(output_dir, exist_ok=True)

# Grid setup
nx, ny = 200, 200
x = np.linspace(-10, 10, nx)
y = np.linspace(-10, 10, ny)
dx = x[1] - x[0]
X, Y = np.meshgrid(x, y)

# Time parameters
dt = 0.01
num_steps = 30000
c = 1.0
lambda_factor = 0.2

# Phase setup
phi_list = np.linspace(0, 2 * np.pi, 7)
r = 3.0
omega = 0.2

# Fields
psi_old = np.zeros_like(X)
psi = np.zeros_like(X)
psi_new = np.zeros_like(X)

localized_mass = np.exp(-(X**2 + Y**2) / 0.5)
nested_mass = np.exp(-(X**2 + Y**2) / 0.2)

energy_log, curvature_log, entropy_log, stable_mass_log = [], [], [], []
mass_accum = np.zeros_like(X)
V_memory = np.zeros_like(X)
mass_peak_memory = np.zeros_like(X)

# Constants
gamma, delta = 0.05, 0.001
alpha_confinement = 0.6
info_map = np.random.uniform(0.5, 1.5, size=psi.shape)
info_map += 0.05 * np.random.normal(size=psi.shape)
stabilization_threshold = 0.6

# Plot setup
# plt.ion()
# fig, axs = plt.subplots(2, 3, figsize=(15, 10))
# im_psi = axs[0, 0].imshow(psi, cmap='viridis', origin='lower', extent=[-10, 10, -10, 10])
# axs[0, 0].set_title("Wavefunction ψ(x, y, t)")
# im_energy = axs[0, 1].imshow(psi**2, cmap='hot', origin='lower')
# axs[0, 1].set_title("Energy Density ε = ψ²")
# im_force = axs[0, 2].imshow(np.zeros_like(psi), cmap='coolwarm', origin='lower')
# axs[0, 2].set_title("Emergent Force |F_eff|")
# im_curvature = axs[1, 0].imshow(np.zeros_like(psi), cmap='plasma', origin='lower')
# axs[1, 0].set_title("Ricci-like Curvature Rψ")
# quiver_ax = axs[1, 1]
# im_mass = axs[1, 2].imshow(mass_accum, cmap='cividis', origin='lower')
# axs[1, 2].set_title("Accumulated Mass")

for t in range(num_steps):
    if t > 10000:
        r = 1.0

    psi = np.zeros_like(X)
    for i, phi in enumerate(phi_list):
        cx = r * np.cos(omega * t + phi)
        cy = r * np.sin(omega * t + phi)
        wave = np.exp(-((X - cx)**2 + (Y - cy)**2)) * np.cos(3 * (X - cx) + phi)
        psi += wave

    wave_packet = np.exp(-((X - 3 - 0.02 * t)**2 + Y**2)) * np.cos(5 * (X - 3 - 0.02 * t))
    psi += wave_packet
    # Traveling high-curvature packet to promote mass propagation
    travel_wave = np.exp(-((X + 4 - 0.01 * t)**2 + (Y - 2)**2)) * np.cos(6 * X - 0.1 * t)
    psi += 0.8 * travel_wave


    temperature = np.exp(-t / 10000)
    psi += 0.5 * localized_mass * np.cos(3 * X)
    psi += 0.3 * nested_mass * np.sin(5 * Y)
    psi += 0.02 * temperature * np.random.normal(size=psi.shape) * np.sin(10 * X) * np.cos(10 * Y)

    slow_wave = 0.1 * np.cos(0.5 * X + 0.3 * Y + 0.01 * t)
    psi += slow_wave
    compression_wave = 0.01 * np.sin(0.2 * X + 0.2 * Y + 0.001 * t)
    psi += compression_wave

    # Introduce a slight directional curvature gradient
    directional_bias = 0.005 * (X / 10) * psi
    psi += directional_bias


    laplacian = (
        np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0) +
        np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1) - 4 * psi
    ) / dx**2

    with np.errstate(divide='ignore', invalid='ignore'):
        curvature = -laplacian / psi
        curvature[~np.isfinite(curvature)] = 0
        curvature = np.clip(curvature, -10, 10)

    curvature += 0.01 * info_map

    G = 0.05
    grad_mass_x, grad_mass_y = np.gradient(mass_accum, dx)
    gravity_force = G * (grad_mass_x + grad_mass_y)

    charge_density = gaussian_filter(mass_accum, sigma=3)
    grad_charge_x, grad_charge_y = np.gradient(charge_density, dx)
    em_force = 0.05 * (grad_charge_x - grad_charge_y)

    strong_conf = 0.05 * psi**5
    tension_field = -0.01 * gaussian_filter(psi, sigma=4)
    psi += tension_field
    weak_force = 0.01 * np.gradient(np.tanh(psi), dx)[0] * np.gradient(np.tanh(psi), dx)[1]

    energy_density = psi**2
    nucleation_zone = energy_density > 1.5 * np.mean(energy_density)
    mass_accum += 0.01 * nucleation_zone.astype(float)

    curvature_stability = np.exp(-np.abs(curvature - np.mean(curvature)))
    local_decay = delta * (1 - curvature_stability)

    entropy_barrier = np.exp(-0.5 * np.mean(entropy_log[-100:]) / np.max(entropy_log + [1e-8])) if entropy_log else 1.0
    
    local_decay = 0.0  # fallback for propagation test
    mass_accum += entropy_barrier * (gamma * energy_density * dt) - local_decay * mass_accum * dt

    div_mass = 0.005 * (np.roll(mass_accum, 1, axis=0) - np.roll(mass_accum, -1, axis=0) +
                        np.roll(mass_accum, 1, axis=1) - np.roll(mass_accum, -1, axis=1))
    mass_accum += div_mass * dt
    mass_drift = 0.002 * (np.roll(mass_accum, -1, axis=1) - mass_accum)
    mass_accum += mass_drift * dt
    mass_accum = gaussian_filter(mass_accum, sigma=0.5)


    excitation_field = 0.001 * np.sin(5 * X + 0.05 * t) * np.sin(5 * Y + 0.05 * t)
    psi += excitation_field

    radiation_pressure = gaussian_filter(energy_density, sigma=5) - energy_density
    # psi += 0.01 * radiation_pressure

    psi_var = gaussian_filter(psi**2, sigma=3) - psi**2
    instability_seed = 0.005 * np.tanh(psi_var)
    psi += instability_seed

    grad_energy_x, grad_energy_y = np.gradient(energy_density, dx)
    energy_flux = 0.01 * (grad_energy_x + grad_energy_y)
    psi += energy_flux
    # === Entropy Force from δS/δψ ===
    with np.errstate(divide='ignore', invalid='ignore'):
        p_x = energy_density / (np.sum(energy_density) + 1e-10)
        log_p = np.log(p_x + 1e-12)
        entropy_force = -2 * psi * (log_p - np.mean(log_p)) / (np.sum(p_x) + 1e-10)
        entropy_force = np.nan_to_num(entropy_force)
    psi += 0.002 * entropy_force  # Small entropy feedback


    mass_peak_memory = 0.995 * mass_peak_memory + 0.005 * (mass_accum > 1.0).astype(float)
    psi += 0.02 * gaussian_filter(mass_peak_memory, sigma=3)
    psi += 0.01 * np.tanh(curvature * mass_peak_memory)

    V = alpha_confinement * np.log1p(mass_accum * 10)
    beta_dynamic = 0.98 - 0.5 * np.exp(-t / 10000)
    V_memory = beta_dynamic * V_memory + (1 - beta_dynamic) * V
    grad_Vx, grad_Vy = np.gradient(V_memory, dx)
    restoring_potential = grad_Vx + grad_Vy + gravity_force - em_force

    repulsion = -0.01 * (mass_accum > 2.0).astype(float)
    mass_kernel = gaussian_filter(mass_accum, sigma=1)
    local_pressure = 0.02 * (mass_kernel - mass_accum)
    restoring_potential += repulsion + local_pressure

    curv_mag = np.abs(curvature)
    energy_damp = 0.005 * (1 - np.tanh(curv_mag)) * np.gradient(psi**2, dx)[0]**2

    psi_new = 2 * psi - psi_old + c**2 * dt**2 * (
        laplacian + lambda_factor * psi**3 + strong_conf + 0.02 * curvature - restoring_potential + weak_force - energy_damp)

    labeled, num_masses = label(mass_accum > stabilization_threshold)
    prob_density = energy_density / (np.sum(energy_density) + 1e-10)
    entropy = -np.sum(prob_density * np.log(prob_density + 1e-12))

    energy_log.append(np.sum(energy_density))
    curvature_log.append(np.sum(np.abs(curvature)))
    entropy_log.append(entropy)
    stable_mass_log.append(num_masses)

    if t % 100 == 0:
        # im_psi.set_data(psi)
        # im_energy.set_data(energy_density)
        # im_force.set_data(np.sqrt(grad_energy_x**2 + grad_energy_y**2))
        # im_curvature.set_data(curvature)
        # im_mass.set_data(mass_accum)

        # im_psi.set_clim(psi.min(), psi.max())
        # im_energy.set_clim(energy_density.min(), energy_density.max())
        # im_force.set_clim(0, 1)
        # im_curvature.set_clim(curvature.min(), curvature.max())
        # im_mass.set_clim(mass_accum.min(), mass_accum.max())

        # quiver_ax.clear()
        # norm = np.sqrt(grad_energy_x**2 + grad_energy_y**2) + 1e-10
        # # quiver_ax.quiver(X[::5, ::5], Y[::5, ::5],
        #                  -grad_energy_x[::5, ::5] / norm[::5, ::5],
        #                  -grad_energy_y[::5, ::5] / norm[::5, ::5],
        #                  color='white', scale=2, width=0.003)
        # quiver_ax.set_title("Vector Field F_eff")
        # quiver_ax.set_xlim(-10, 10)
        # quiver_ax.set_ylim(-10, 10)

        # fig.suptitle(f"Step {t} | Entropy: {entropy:.4f} | Stable Masses: {num_masses}", fontsize=14)
        # plt.pause(0.001)

        with open(os.path.join(output_dir, "log.txt"), "a") as f:
            weak_energy = np.sum(np.abs(weak_force))
            strong_energy = np.sum(np.abs(strong_conf))
            max_mass = np.max(mass_accum)
            f.write(f"Step {t} | Energy: {np.sum(energy_density):.2f} | Curvature: {np.sum(np.abs(curvature)):.2f} | Entropy: {entropy:.4f} | Stable Masses: {num_masses} | Weak-Energy: {weak_energy:.2f} | Strong-Energy: {strong_energy:.2f} | Max Mass: {max_mass:.2f}\n")
        np.savez_compressed(os.path.join(output_dir, f"state_{t:06d}.npz"),
                            psi=psi, mass=mass_accum, energy=energy_density)

    if t % 1000 == 0:
        frame_filename = os.path.join(output_dir, f"frame_{t:03d}.png")
        plt.savefig(frame_filename)

    
    info_map += 0.0002 * np.random.normal(size=info_map.shape)
    info_map = np.clip(info_map, 0.3, 2.0)

    psi_old = psi.copy()
    psi = psi_new.copy()

# plt.ioff()
# plt.show()
