
import numpy as np
import matplotlib.pyplot as plt
import re

# Path to your log file
log_file = "mass_outputs_propagation/log.txt"

# Initialize lists
steps, energies, curvatures, entropies, masses, weak_energies, strong_energies, max_masses = [], [], [], [], [], [], [], []

# Read and parse the log
with open(log_file, "r") as f:
    for line in f:
        match = re.findall(r"[-+]?[0-9]*\.?[0-9]+", line)
        if len(match) >= 8:
            steps.append(int(match[0]))
            energies.append(float(match[1]))
            curvatures.append(float(match[2]))
            entropies.append(float(match[3]))
            masses.append(int(match[4]))
            weak_energies.append(float(match[5]))
            strong_energies.append(float(match[6]))
            max_masses.append(float(match[7]))

steps = np.array(steps)
energies = np.array(energies)
curvatures = np.array(curvatures)
entropies = np.array(entropies)
masses = np.array(masses)
weak_energies = np.array(weak_energies)
strong_energies = np.array(strong_energies)
max_masses = np.array(max_masses)

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(steps, energies, label="Energy")
plt.xlabel("Step")
plt.ylabel("Total Energy")
plt.title("Total Energy vs Step")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(steps, curvatures, label="Curvature", color="purple")
plt.xlabel("Step")
plt.ylabel("Total Curvature")
plt.title("Curvature vs Step")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(steps, entropies, label="Entropy", color="green")
plt.xlabel("Step")
plt.ylabel("Entropy")
plt.title("Entropy vs Step")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(steps, masses, label="Stable Masses", color="red")
plt.xlabel("Step")
plt.ylabel("Stable Masses")
plt.title("Stable Masses vs Step")
plt.grid(True)

plt.tight_layout()
plt.savefig("mass_outputs_propagation/summary_plots.png")
plt.show()

# Print basic stats
print(f"Average Entropy: {np.mean(entropies):.4f}")
print(f"Average Stable Masses: {np.mean(masses):.2f}")
print(f"Peak Max Mass Observed: {np.max(max_masses):.2f}")
