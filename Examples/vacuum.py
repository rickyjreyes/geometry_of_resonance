import numpy as np

# Constants
c = 3e8  # Speed of light in vacuum (m/s)
hbar = 1.0545718e-34  # Reduced Planck constant (J·s)
eV_to_J = 1.60218e-19  # 1 eV in Joules

# Assume the lowest stable vacuum mode corresponds to an energy scale (guess)
E0_eV = 1e-12  # Energy scale ~ pico-eV
E0_J = E0_eV * eV_to_J

# Calculate corresponding frequency
f0 = E0_J / hbar  # Frequency in Hz

# Wavelength
lambda0 = c / f0  # in meters

# Estimate vacuum toroid "circumference" from fundamental wavelength
# Assume it's the full loop (1st harmonic)
R0 = lambda0 / (2 * np.pi)  # Radius assuming circular wave path

f0, lambda0, R0


import matplotlib.pyplot as plt

# Base resonance from WCT (user's value)
base_resonance = 85.4e-6  # 85.4 micrometers in meters

# Fix: convert base to float before negative exponentiation
n = np.arange(-6, 7, 1)  # from -6 to 6

scaling_type_1 = base_resonance / (2.0 ** n)  # binary scaling
scaling_type_2 = base_resonance / (3.0 ** n)  # ternary scaling
scaling_type_3 = base_resonance * np.exp(-n * 1.25)  # exponential/log scaling

# Known physical phenomena reference scales (in meters)
reference_points = {
    "Casimir (avg)": 100e-9,
    "Visible Light": 500e-9,
    "IR Peak": 10e-6,
    "WCT Base": base_resonance,
    "CMB Peak": 2e-3,
}

# Plotting
plt.figure(figsize=(12, 6))
plt.semilogy(n, scaling_type_1, 'o-', label="Binary Shells (×2)")
plt.semilogy(n, scaling_type_2, 's-', label="Ternary Shells (×3)")
plt.semilogy(n, scaling_type_3, 'x-', label="Exponential (e^1.25n)")

# Add reference lines
for label, y in reference_points.items():
    plt.axhline(y, linestyle='--', label=f"{label} ({y*1e6:.1f} µm)")

plt.xlabel("Harmonic Shell Index (n)")
plt.ylabel("Resonant Wavelength (m)")
plt.title("WCT Harmonic Shell Scaling vs Physical Phenomena")
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.show()
