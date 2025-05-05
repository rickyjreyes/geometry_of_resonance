import cupy as cp
import matplotlib.pyplot as plt
import os

# -------------------------
# Load Data
# -------------------------

entropy_data = cp.load('entropy_ensemble.npy')  # Shape: (ensemble_runs, timepoints)
coherence_data = cp.load('coherence_ensemble.npy')  # Shape: (ensemble_runs, timepoints)

# -------------------------
# Basic Statistics
# -------------------------

mean_entropy = cp.mean(entropy_data, axis=0)
std_entropy = cp.std(entropy_data, axis=0)

mean_coherence = cp.mean(coherence_data, axis=0)
std_coherence = cp.std(coherence_data, axis=0)

# -------------------------
# Plotting Functions
# -------------------------
def plot_entropy():
    plt.figure(figsize=(8,6))
    mean = mean_entropy.get()
    std = std_entropy.get()
    plt.plot(mean, label='Mean Entropy')
    plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.3)
    plt.xlabel('Time (×1000 steps)')
    plt.ylabel('Entropy')
    plt.title('Entropy Evolution')
    plt.legend()
    plt.grid()
    plt.savefig('entropy_evolution.png')
    plt.close()

def plot_coherence():
    plt.figure(figsize=(8,6))
    mean = mean_coherence.get()
    std = std_coherence.get()
    plt.plot(mean, label='Mean Coherence Length')
    plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.3)
    plt.xlabel('Time (×1000 steps)')
    plt.ylabel('Coherence Length (ξ)')
    plt.title('Coherence Length Evolution')
    plt.legend()
    plt.grid()
    plt.savefig('coherence_evolution.png')
    plt.close()
def plot_entropy_vs_coherence():
    plt.figure(figsize=(8,6))
    x = mean_coherence.get()
    y = mean_entropy.get()
    plt.scatter(x, y, alpha=0.7)
    plt.xlabel('Coherence Length (ξ)')
    plt.ylabel('Entropy (S)')
    plt.title('Entropy vs Coherence Length')
    plt.grid()
    plt.savefig('entropy_vs_coherence.png')
    plt.close()
# -------------------------
# Run All Plots
# -------------------------

plot_entropy()
plot_coherence()
plot_entropy_vs_coherence()

print("\n=== Analysis Complete ===")
print("Saved plots: entropy_evolution.png, coherence_evolution.png, entropy_vs_coherence.png")