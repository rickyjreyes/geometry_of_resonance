import cupy as cp
import matplotlib.pyplot as plt

# Physical constants
pi = cp.pi

# Plate separation (in micrometers)
a = cp.linspace(0.5, 30, 500)  # in micrometers
a_m = a * 1e-6  # convert to meters

# Parameters from Wave Confinement Theory
xi = 10e-6          # coherence length (10 μm)
sigma = 0.0806      # entropy-curvature ratio
theta = 1e-12       # curvature feedback constant
epsilon = 1e-9      # small regularization to avoid divide-by-zero

# Standard Casimir pressure (QED)
P_qed = - (pi**2 / (240 * a_m**4))

# WCT-corrected Casimir pressure with feedback and entropy effects
feedback_term = 1 + (theta / (a_m + epsilon)**2)
P_wct = P_qed * feedback_term * (1 + sigma * cp.exp(-a_m / xi))

# Percent deviation
percent_deviation = 100 * (P_wct - P_qed) / cp.abs(P_qed)

# Convert to NumPy for plotting
a_cpu = cp.asnumpy(a)
P_qed_cpu = cp.asnumpy(cp.abs(P_qed))
P_wct_cpu = cp.asnumpy(cp.abs(P_wct))
percent_deviation_cpu = cp.asnumpy(percent_deviation)

# Plot Casimir pressure comparison
plt.figure(figsize=(10, 5))
plt.plot(a_cpu, P_qed_cpu, label='Standard QED (|P|)', color='orange')
plt.plot(a_cpu, P_wct_cpu, '--', label='WCT-corrected (|P|)', color='red')
plt.xlabel("Plate Separation a (μm)")
plt.ylabel("Casimir Pressure (N/m²)")
plt.title("Casimir Force: Standard QED vs. WCT Prediction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot percent deviation
plt.figure(figsize=(10, 4))
plt.plot(a_cpu, percent_deviation_cpu, color='purple')
plt.xlabel("Plate Separation a (μm)")
plt.ylabel("Percent Deviation from QED (%)")
plt.title("WCT vs. QED: Percent Deviation of Casimir Pressure")
plt.grid(True)
plt.tight_layout()
plt.show()
