import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simulation parameters
size = 100
c = 1.0
dx = 1.0
dt = 0.1
steps = 500
noise_level = 0.01

assert c * dt / dx < 1, "Unstable parameters."

u = np.zeros((size, size))
u_prev = np.zeros((size, size))
u_next = np.zeros((size, size))

u[size//2, size//2] = 1.0

fig, ax = plt.subplots()
im = ax.imshow(u, cmap='viridis', vmin=-1, vmax=1)
plt.colorbar(im)

energy_over_time = []

def update(frame):
    global u, u_prev, u_next
    
    for i in range(1, size - 1):
        for j in range(1, size - 1):
            u_next[i, j] = (2 * u[i, j] - u_prev[i, j] +
                            (c * dt / dx)**2 *
                            (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - 4 * u[i, j]))
            u_next[i, j] += noise_level * (np.random.rand() - 0.5)

    u_next[0, :] = 0
    u_next[-1, :] = 0
    u_next[:, 0] = 0
    u_next[:, -1] = 0

    total_energy = np.sum(u_next**2)
    energy_over_time.append(total_energy)

    im.set_data(u_next)
    ax.set_title(f"Step {frame}, Energy {total_energy:.4f}")

    u_prev, u, u_next = u, u_next, u_prev

    return [im]

ani = animation.FuncAnimation(fig, update, frames=steps, interval=30, blit=True)

# Save as MP4
ani.save('wave_decay_simulation.mp4', writer='ffmpeg', fps=30)

plt.show()

# Plot energy decay
plt.figure()
plt.plot(energy_over_time)
plt.xlabel('Time Step')
plt.ylabel('Total Energy')
plt.title('Wave Energy Decay Over Time')
plt.savefig('decay_time.png')
plt.show()
