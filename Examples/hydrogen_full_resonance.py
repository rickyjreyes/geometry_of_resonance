import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# Grid and coordinates
x = np.linspace(-8, 8, 300)
y = np.linspace(-8, 8, 300)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
Θ = np.arctan2(Y, X)

# Time steps
times = np.linspace(0, 1.5, 40)

# Setup figure
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(-1.2, 1.2)
ax.set_title("Photon → Neutrino → Electron → Quark (WCT Resonance Formation)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("ψ(x, y)")
surf = [None]
label = ax.text2D(0.05, 0.9, "", transform=ax.transAxes, fontsize=12, color='black')

# Update function with overlays, labels, interference, and heatmap option
def update(frame):
    t = times[frame]

    # Base resonance components
    photon = np.exp(-R**2 / (6 - 4 * (t / times[-1]))) * np.cos(2 * np.pi * (3 + t) * R)
    neutrino_web = 0.2 * np.sin(8 * Θ + 5 * t) * np.exp(-R**2 / 6)
    electron_core = 0.5 * np.exp(-R**2 / 1.5**2) * np.cos(10 * R - t * 2)
    quark_shell = 0.3 * np.cos(3 * R - t * 4) * np.exp(-((R - 4)**2) / 1.5)

    # Interference: second particle forming slightly offset
    interference = 0.3 * np.exp(-((R - 3)**2) / 2) * np.sin(5 * R - t * 3)

    # Total field with chirality
    ψ = photon * np.exp(-t * 2.5) + neutrino_web + electron_core + quark_shell + interference
    ψ *= np.cos(Θ + t * 3)

    # Remove old surface and update
    if surf[0] is not None:
        surf[0].remove()
    surf[0] = ax.plot_surface(X, Y, ψ, cmap='coolwarm', edgecolor='none')

    # Set label overlays
    if t < 0.4:
        label.set_text("Photon Field")
    elif t < 0.8:
        label.set_text("Neutrino Web Formation")
    elif t < 1.1:
        label.set_text("Electron Core Emergence")
    else:
        label.set_text("Quark Shell + Chiral Locking")

    return surf[0], label

# Animate and save with overlays and enhanced evolution
ani = animation.FuncAnimation(fig, update, frames=len(times), interval=100)
output_path = "./wct_full_labeled_resonance.mp4"
ani.save(output_path, writer='ffmpeg', fps=10)

output_path
