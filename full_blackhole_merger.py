import cupy as cp
import matplotlib.pyplot as plt

# Time grid
nt = 3000
dt = 0.02
t = cp.linspace(0, nt*dt, nt)

# Parameters
f0 = 0.05         # Initial frequency for inspiral
tau = 30          # Chirp timescale
t_merge = 35      # Merger time
sigma = 2.5       # Merger width
omega = 2 * cp.pi * 0.5
omega_qnm = 2 * cp.pi * 0.4
alpha = 0.1       # Ringdown damping rate
theta = 0.0026    # WCT feedback strength
A = 1.0

# 1. Inspiral (chirp signal)
f_chirp = f0 * (1 + t / tau)**1.5
chirp = (1 - cp.exp(-t / tau)) * cp.sin(2 * cp.pi * f_chirp * t)

# 2. Merger burst
merger = cp.exp(-((t - t_merge)**2) / sigma**2) * cp.cos(omega * t)

# 3. Ringdown (classical)
ringdown = A * cp.exp(-alpha * (t - t_merge)) * cp.cos(omega_qnm * (t - t_merge))
ringdown *= (t > t_merge)  # Zero before merger

# 4. Tail (WCT-corrected ringdown)
tail = ringdown / (1 + theta * (t - t_merge)**2)

# Total waveform
waveform = chirp + merger + tail

# Plot
plt.figure(figsize=(10, 5))
plt.plot(cp.asnumpy(t), cp.asnumpy(waveform), label="WCT Full Merger Signal", color='black')
plt.axvline(t_merge, color='gray', linestyle='--', label="Merger Time")
plt.xlabel("Time")
plt.ylabel("Strain Amplitude (arb. units)")
plt.title("Full Gravitational Waveform with WCT Tail")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("full_merger_wct.png", dpi=300)
plt.show()
