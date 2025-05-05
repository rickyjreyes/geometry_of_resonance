import cupy as cp
import matplotlib.pyplot as plt

# Grid and synthetic coherence fluctuation field
nx = 1024
x = cp.linspace(0, 1, nx)
wavelength = 500e-9  # 500 nm
k = 2 * cp.pi / wavelength

# Coherence noise: simulate spatial phase fluctuations
noise_level = 1e-3
phase_noise = noise_level * cp.random.randn(nx)
field = cp.cos(k * x + phase_noise)

# Compute power spectral density (PSD)
fft_data = cp.fft.fft(field - cp.mean(field))
psd = cp.abs(fft_data)**2
freq = cp.fft.fftfreq(nx, d=(x[1] - x[0]))

# Plot
plt.plot(freq.get(), psd.get())
plt.xlabel("Frequency")
plt.ylabel("PSD")
plt.title("Michelson Phase Noise Spectrum")
plt.grid()
plt.xlim(0, 1e7)
plt.tight_layout()
plt.show()
