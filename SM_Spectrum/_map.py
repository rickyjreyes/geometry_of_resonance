import numpy as np
import matplotlib.pyplot as plt

# WCT mass functions from your model
def wct_mass_light(n, params):
    n = np.longdouble(n)
    m0, beta, sigma, theta, gamma, A_l, lambda_l, chi, epsilon_nu = map(np.longdouble, params)
    ratio = n / (n + 1) if n > 0 else np.longdouble(0)
    m_harmonic = ratio * m0
    delta_topo = beta * ((n % 3 - 1) * (-1)**int(n))
    delta_entropy = sigma * np.log1p(n)
    delta_curv = theta * n**2
    delta_inertia = gamma * n**3
    safe_exp = np.exp(np.clip(lambda_l * n, -50, 50))
    delta_hierarchy = A_l * safe_exp
    delta_chiral = chi / (n**2 if n != 0 else np.longdouble(1))
    mass = m_harmonic + delta_topo + delta_entropy + delta_curv + delta_inertia + delta_hierarchy + delta_chiral
    if n == 3:
        mass *= epsilon_nu
    return np.maximum(mass, np.longdouble(1e-6))

def wct_mass_middle(n, params):
    amp, scale1, scale2, phase, harmonic, offset, quad = map(np.longdouble, params)
    base = amp * np.sin(scale1 * n + phase)
    harmonic_term = harmonic * np.sin(scale2 * n)
    constant_offset = offset
    quad_term = quad * n**2
    return np.maximum(base + harmonic_term + constant_offset + quad_term, np.longdouble(1e-6))

def wct_mass_heavy(n, heavy_params, light_params):
    base_mass = wct_mass_light(n, light_params)
    (
        qcd_quark, qcd_boson, quark_scale, boson_scale,
        boson_nonlin1_scale, boson_nonlin1_power,
        boson_nonlin2_scale, boson_nonlin2_power,
        wz_break_scale, wz_break_angle, wz_break_phase,
        higgs_harmonic_scale, higgs_offset
    ) = map(np.longdouble, heavy_params)
    safe_quark = np.power(np.clip(n / 50, 1e-6, 1e6), np.clip(quark_scale, -10, 10))
    safe_boson = np.power(np.clip(n / 100, 1e-6, 1e6), np.clip(boson_scale, -10, 10))
    safe_boson_nonlin1 = boson_nonlin1_scale * np.power(np.clip(n / 100, 1e-6, 1e6), boson_nonlin1_power)
    safe_boson_nonlin2 = boson_nonlin2_scale * np.power(np.clip(n / 100, 1e-6, 1e6), boson_nonlin2_power)

    correction = np.longdouble(0)
    if 19 <= n <= 73:
        correction = qcd_quark * safe_quark
    elif 91 <= n <= 123:
        correction = qcd_boson * safe_boson + safe_boson_nonlin1 + safe_boson_nonlin2
        if n in [95, 96]:
            correction += wz_break_scale * np.sin(wz_break_angle * n + wz_break_phase)
        if n == 123:
            correction += higgs_harmonic_scale * (n / (n + 1)) + higgs_offset
    return np.maximum(base_mass + correction, np.longdouble(1e-6))

# Optimized parameters
light_params = np.array([
    0.3877, -2.3676, -8.2963, 6.8830, -1.7948, -8.6415, -6.1744, 0.9981, 0.0048
])
middle_params = np.array([
    2241.1, 5.0632, 9.4062, 1.6469, 0.2092, 998.7, 0.2427
])
heavy_params = np.array([
    316518.3, 101539.5, -8.5335, 7.8058, 12539.5, 0.1547,
    4533.3, -8.9307, 7378.5, 7.9130, -5.8920, 463275.3, -859100.2
])

# Standard Model particle reference masses in MeV
sm_particles = {
    "Electron": 0.511,
    "Up Quark": 2.2,
    "Down Quark": 4.7,
    "Strange Quark": 95,
    "Charm Quark": 1275,
    "Bottom Quark": 4180,
    "Top Quark": 173000,
    "Muon": 105.7,
    "Tau": 1777,
    "W Boson": 80379,
    "Z Boson": 91188,
    "Higgs": 125100
}

# Evaluate mass at fractional harmonics up to a subharmonic depth of 6
harmonic_range = range(1, 124)
subharmonic_depth = 6

harmonics = []
subharmonics = []
masses = []

for n in harmonic_range:
    for k in range(1, subharmonic_depth + 1):
        nk = n / k
        if nk < 1:
            continue
        if nk in [0, 1, 2, 3]:
            mass = wct_mass_light(nk, light_params)
        elif nk in [12, 19, 53, 73, 91]:
            mass = wct_mass_middle(nk, middle_params)
        else:
            mass = wct_mass_heavy(nk, heavy_params, light_params)

        harmonics.append(n)
        subharmonics.append(k)
        masses.append(float(mass))

# Plotting
plt.figure(figsize=(12, 8))
sc = plt.scatter(harmonics, subharmonics, c=np.log10(masses), cmap='viridis', s=50)
plt.colorbar(sc, label='log10(Predicted Mass in MeV)')
plt.xlabel('Harmonic Index (n)')
plt.ylabel('Subharmonic Level (k)')
plt.title('Harmonic–Subharmonic Map with Mass Predictions')

# Overlay SM particles
for label, sm_mass in sm_particles.items():
    closest_idx = np.argmin(np.abs(np.array(masses) - sm_mass))
    x = harmonics[closest_idx]
    y = subharmonics[closest_idx]
    plt.plot(x, y, 'r*', markersize=10)
    plt.text(x + 0.5, y + 0.1, label, fontsize=9, color='red')

plt.grid(True)
plt.tight_layout()
plt.show()
