#!/usr/bin/env python3
"""Deterministic finite-band reference simulation for WCT release verification.

This is a bounded linear spectral-rail demonstration. It verifies that exact
Fourier evolution amplifies modes near the analytic maximum of
sigma(k) = r + a k^2 - b k^4 for a fixed seeded initial field. It does not
establish nonlinear pattern selection or empirical validity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 20260628
GRID = 64
BOX_LENGTH = 20.0
R = 0.20
A = 1.00
B = 0.25
TIME = 4.0
BINS = 48
FIELD_SCALE = 10_000_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_float(value: float, digits: int = 10) -> float:
    return float(f"{value:.{digits}f}")


def radial_spectrum(power: np.ndarray, kmag: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, float(kmag.max()), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    indices = np.digitize(kmag.ravel(), edges, right=False) - 1
    values = power.ravel()
    sums = np.bincount(indices, weights=values, minlength=bins + 1)[:bins]
    counts = np.bincount(indices, minlength=bins + 1)[:bins]
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return centers, means


def run(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    field0 = rng.normal(0.0, 1.0, size=(GRID, GRID))

    dx = BOX_LENGTH / GRID
    kaxis = 2.0 * np.pi * np.fft.fftfreq(GRID, d=dx)
    kx, ky = np.meshgrid(kaxis, kaxis, indexing="ij")
    k2 = kx**2 + ky**2
    kmag = np.sqrt(k2)
    sigma = R + A * k2 - B * k2**2

    field_hat0 = np.fft.fft2(field0)
    field_hat_t = field_hat0 * np.exp(sigma * TIME)
    field_t = np.fft.ifft2(field_hat_t).real
    power = np.abs(field_hat_t) ** 2

    centers, radial_power = radial_spectrum(power, kmag, BINS)
    nonzero = centers > 0
    observed_index = int(np.argmax(np.where(nonzero, radial_power, -np.inf)))
    observed_k = float(centers[observed_index])
    analytic_k = float(np.sqrt(A / (2.0 * B)))
    bin_width = float(centers[1] - centers[0])

    csv_path = output_dir / "finite_band_spectrum.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["k_center", "mean_power"])
        for center, value in zip(centers, radial_power, strict=True):
            writer.writerow([f"{center:.12e}", f"{value:.12e}"])

    # Store the field as explicit little-endian signed integers at 1e-10 scale.
    # This removes Python/NumPy .npy header and sub-ULP FFT drift from the hash.
    field_quantized = np.rint(field_t * FIELD_SCALE).astype("<i8")
    field_path = output_dir / "finite_band_field_i10.bin"
    field_path.write_bytes(field_quantized.tobytes(order="C"))
    field_stable = field_quantized.astype(np.float64) / FIELD_SCALE

    fig_path = output_dir / "finite_band_spectrum.png"
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=120)
    ax.plot(centers, radial_power)
    ax.axvline(analytic_k, linestyle="--", label=r"analytic $k_\star$")
    ax.set_xlabel("radial wavenumber k")
    ax.set_ylabel("mean Fourier power")
    ax.set_title("Deterministic finite-band spectral rail")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, metadata={"Software": "WCT frozen release"})
    plt.close(fig)

    summary_path = output_dir / "finite_band_summary.json"
    summary = {
        "schema_version": "1.1.0",
        "scientific_boundary": "Deterministic linear spectral-rail reference; not nonlinear or empirical validation.",
        "seed": SEED,
        "grid": GRID,
        "box_length": BOX_LENGTH,
        "parameters": {"r": R, "a": A, "b": B, "time": TIME},
        "field_encoding": {
            "file": field_path.name,
            "dtype": "little-endian-int64",
            "scale": FIELD_SCALE,
            "shape": [GRID, GRID],
        },
        "analytic_k_star": stable_float(analytic_k, 12),
        "observed_peak_bin_center": stable_float(observed_k, 12),
        "radial_bin_width": stable_float(bin_width, 12),
        "peak_within_one_bin": abs(observed_k - analytic_k) <= bin_width,
        "field_l2": stable_float(float(np.linalg.norm(field_stable))),
        "field_mean": stable_float(float(field_stable.mean())),
        "field_std": stable_float(float(field_stable.std())),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    outputs = [csv_path, field_path, fig_path, summary_path]
    return {path.name: sha256(path) for path in outputs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("reproducibility/artifacts"))
    parser.add_argument("--hashes-out", type=Path, default=None)
    args = parser.parse_args()
    hashes = run(args.output_dir)
    if args.hashes_out:
        args.hashes_out.parent.mkdir(parents=True, exist_ok=True)
        args.hashes_out.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(hashes, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
