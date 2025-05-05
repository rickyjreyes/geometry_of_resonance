import cupy as cp
import matplotlib.pyplot as plt

# Parameters
nx = 2048
L = 10.0
x = cp.linspace(0, L, nx)
dx = x[1] - x[0]
k0 = 2 * cp.pi / 0.5  # light wavelength

# Wave confinement phase fluctuation model
noise_level = 1e-4
psi0 = cp.sin(2 * cp.pi * x / L) + noise_level * cp.random.randn(nx)

# Simulated optical wave
wave_x = cp.cos(k0 * x)
grad_psi = cp.gradient(psi0, dx)
delta_phi = grad_psi / (psi0**2 + 1e-6)

# Interferometer phase shift (WCT correction)
wave_shifted = wave_x * cp.exp(-1j * delta_phi)

# FFT in CuPy
fft_result = cp.fft.fft(cp.real(wave_shifted))
freqs = cp.fft.fftfreq(nx, d=dx)
psd = cp.abs(fft_result)**2

# Convert for plotting
plt.figure(figsize=(10, 4))
plt.plot(cp.asnumpy(freqs[:nx//2]), cp.asnumpy(psd[:nx//2]))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density")
plt.title("Michelson Interferometer Phase Noise PSD (WCT Model)")
plt.grid(True)
plt.tight_layout()
plt.savefig("michaelson_psd.png", dpi=300)
plt.show()
