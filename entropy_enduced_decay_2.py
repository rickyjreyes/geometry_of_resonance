import cupy as cp
import matplotlib.pyplot as plt

# Grid and time setup
nx, ny = 200, 200
Lx, Ly = 10.0, 10.0
dx, dy = Lx / nx, Ly / ny
dt = 0.001
steps = 1000

# Physical parameters
sigma = 0.0806
epsilon = 1e-5
decay_threshold = 1e3

# Coordinate grid
x = cp.linspace(-Lx / 2, Lx / 2, nx)
y = cp.linspace(-Ly / 2, Ly / 2, ny)
X, Y = cp.meshgrid(x, y)

# Initial wave packet (asymmetric)
psi = cp.exp(-(X**2 + Y**2)) * (1 + 0.1 * X)
psi_prev = psi.copy()

# Laplacian operator
def laplacian(f):
    return (
        cp.roll(f, 1, axis=0) + cp.roll(f, -1, axis=0) +
        cp.roll(f, 1, axis=1) + cp.roll(f, -1, axis=1) -
        4 * f
    ) / dx**2

# Evolution loop
collapse_detected = False
frames = []
for step in range(steps):
    grad_x = (cp.roll(psi, -1, axis=0) - cp.roll(psi, 1, axis=0)) / (2 * dx)
    grad_y = (cp.roll(psi, -1, axis=1) - cp.roll(psi, 1, axis=1)) / (2 * dy)
    grad_energy = grad_x**2 + grad_y**2

    lap = laplacian(psi)
    entropy_feedback = -sigma * grad_energy / (psi + epsilon)

    psi_new = 2 * psi - psi_prev + dt**2 * (lap + entropy_feedback)
    psi_prev = psi
    psi = psi_new

    if cp.max(cp.abs(psi)) > decay_threshold and not collapse_detected:
        print(f"Collapse triggered at step {step}")
        collapse_detected = True

    if step % 100 == 0:
        frames.append(cp.asnumpy(psi.copy()))

# Plot final collapse frame
plt.imshow(frames[-1], cmap='inferno', extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
plt.title("Entropy-Induced Decay of Confined Waveform")
plt.colorbar(label='Field Amplitude')
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.savefig("entropy_enduced_decay.png", dpi=300)
plt.show()
