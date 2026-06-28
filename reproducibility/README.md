# Frozen finite-band reference run

This directory provides the deterministic simulation and figure-regeneration component of the WCT canonical release.

It evolves a fixed seeded field under the exact linear Fourier growth law

\[
\sigma(k)=r+a k^2-b k^4
\]

and checks that the observed radial power peak falls within one discrete radial bin of

\[
k_\star=\sqrt{a/(2b)}.
\]

This is a reproducibility fixture for the finite-band spectral rail. It is **not** a proof of nonlinear pattern selection and is **not** empirical validation.

```bash
python -m pip install -r reproducibility/requirements.lock
make reproduce
make verify
```

The expected SHA-256 values cover the radial spectrum table, regenerated figure, rounded summary, and the numerical field encoded as explicit little-endian signed integers at a scale of `1e10`. The quantized field encoding removes `.npy` header drift and sub-ULP FFT differences across supported Python versions while preserving the declared numerical precision. Any material dependency, code, parameter, or renderer drift changes at least one hash and fails CI.
