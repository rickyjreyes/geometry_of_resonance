#!/usr/bin/env python3
"""
Nonlinear axisymmetric nonspherical relaxation test for the corrected WCT
fixed-mass energy.

The real field is expanded in normalized m=0 spherical harmonics,

    psi(r,theta) = sum_l a_l(r) Y_l0(theta),

and optimized at fixed L2 mass for

    E = integral [ |grad psi|^2
                   + (delta + F(psi^2)) |Delta psi|^2
                   - (g/2) psi^4 ] d^3x,

    F(u) = u / (u + eps^2 exp(-2 alpha u))^2.

The angular nonlinear terms are evaluated with Gauss-Legendre quadrature.
This is a nondimensional numerical deformation test, not a continuum proof.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.polynomial.legendre import leggauss
from numpy.polynomial.legendre import Legendre

from wct_radial_confinement_test import RadialWCT

torch.set_default_dtype(torch.float64)


class AxisymmetricWCT:
    def __init__(
        self,
        n=80,
        radius=25.0,
        lmax=6,
        nang=48,
        mass=1.0,
        eps=0.5,
        alpha=1.0,
        delta=0.1,
        g=120.0,
    ):
        self.n = n
        self.radius = radius
        self.lmax = lmax
        self.mass = mass
        self.eps = eps
        self.alpha = alpha
        self.delta = delta
        self.g = g

        self.dr = radius / n
        self.r = (torch.arange(n) + 0.5) * self.dr
        self.faces = torch.arange(n + 1) * self.dr
        self.wr = self.r**2 * self.dr

        x, wx = leggauss(nang)
        basis = []
        for l in range(lmax + 1):
            P = Legendre.basis(l)(x)
            basis.append(math.sqrt((2 * l + 1) / (4.0 * math.pi)) * P)
        self.Y = torch.tensor(np.asarray(basis), dtype=torch.float64)  # (L, A)
        self.wang = torch.tensor(2.0 * math.pi * wx, dtype=torch.float64)

    def radial_laplacian(self, a, l):
        """L_l a = r^-2 (r^2 a')' - l(l+1) a/r^2."""
        flux = torch.zeros(self.n + 1, dtype=a.dtype, device=a.device)
        flux[1:self.n] = (
            self.faces[1:self.n] ** 2 * (a[1:] - a[:-1]) / self.dr
        )
        flux[self.n] = (
            self.faces[self.n] ** 2 * (0.0 - a[-1]) / (0.5 * self.dr)
        )
        radial = (flux[1:] - flux[:-1]) / (self.r**2 * self.dr)
        return radial - l * (l + 1) * a / self.r**2

    def radial_gradient_energy(self, a, l):
        interior = (
            self.faces[1:self.n] ** 2
            * ((a[1:] - a[:-1]) / self.dr) ** 2
            * self.dr
        )
        outer = (
            self.faces[self.n] ** 2
            * ((0.0 - a[-1]) / (0.5 * self.dr)) ** 2
            * (0.5 * self.dr)
        )
        angular = l * (l + 1) * torch.sum(a**2) * self.dr
        return torch.sum(interior) + outer + angular

    def F(self, u):
        q = self.eps**2 * torch.exp(-2.0 * self.alpha * u)
        return u / (u + q) ** 2

    def normalize_coeffs(self, coeffs):
        mass = torch.sum(self.wr[None, :] * coeffs**2)
        return coeffs * torch.sqrt(torch.tensor(self.mass) / (mass + 1e-300))

    def energy_parts(self, coeffs):
        laps = torch.stack(
            [self.radial_laplacian(coeffs[l], l) for l in range(self.lmax + 1)]
        )

        kinetic = sum(
            self.radial_gradient_energy(coeffs[l], l)
            for l in range(self.lmax + 1)
        )
        biharmonic = torch.sum(self.wr[None, :] * laps**2)

        # psi and Delta psi on (r, angular quadrature) grid.
        psi = coeffs.T @ self.Y
        lap_psi = laps.T @ self.Y
        u = psi**2

        curvature_ang = torch.sum(
            self.wang[None, :] * self.F(u) * lap_psi**2,
            dim=1,
        )
        quartic_ang = torch.sum(self.wang[None, :] * psi**4, dim=1)
        curvature = torch.sum(self.wr * curvature_ang)
        quartic = torch.sum(self.wr * quartic_ang)
        mass = torch.sum(self.wr[None, :] * coeffs**2)

        energy = (
            kinetic
            + self.delta * biharmonic
            + curvature
            - 0.5 * self.g * quartic
        )
        return energy, mass, kinetic, biharmonic, curvature, quartic

    def nonspherical_mass_fraction(self, coeffs):
        total = torch.sum(self.wr[None, :] * coeffs**2)
        nonsph = torch.sum(self.wr[None, :] * coeffs[1:]**2)
        return nonsph / total

    def radial_seed(self, width=1.4, max_iter=700):
        if hasattr(self, "_radial_seed_cache"):
            return self._radial_seed_cache.clone()

        radial = RadialWCT(
            n=self.n,
            radius=self.radius,
            mass=self.mass,
            eps=self.eps,
            alpha=self.alpha,
            delta=self.delta,
            g=self.g,
        )
        sqrtw = torch.sqrt(radial.w)
        phi0 = radial.normalize(torch.exp(-0.5 * (radial.r / width) ** 2))
        y = torch.nn.Parameter((sqrtw * phi0).clone())

        def field():
            x = y / (torch.linalg.vector_norm(y) + 1e-300)
            return x / sqrtw

        optimizer = torch.optim.LBFGS(
            [y], lr=0.5, max_iter=max_iter, max_eval=2 * max_iter,
            tolerance_grad=1e-12, tolerance_change=1e-15, history_size=100,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            phi = field()
            energy = radial.energy_parts(phi)[0]
            energy.backward()
            return energy

        optimizer.step(closure)
        with torch.no_grad():
            phi = field().clone()

        coeffs = torch.zeros((self.lmax + 1, self.n), dtype=torch.float64)
        coeffs[0] = math.sqrt(4.0 * math.pi) * phi
        coeffs = self.normalize_coeffs(coeffs)
        self._radial_seed_cache = coeffs.clone()
        return coeffs

    def deform(self, coeffs, mode_weights):
        """Add radial-shape-matched angular components, then renormalize."""
        base = coeffs[0].clone()
        for l, amp in mode_weights.items():
            if 1 <= l <= self.lmax:
                coeffs[l] = amp * base
        return self.normalize_coeffs(coeffs)

    def solve(self, coeffs0, max_iter=900):
        sqrtw = torch.sqrt(self.wr)
        y0 = coeffs0 * sqrtw[None, :]
        y = torch.nn.Parameter(y0.clone())

        def coeffs_from_y():
            x = y / (torch.linalg.vector_norm(y) + 1e-300)
            return x / sqrtw[None, :]

        optimizer = torch.optim.LBFGS(
            [y],
            lr=0.35,
            max_iter=max_iter,
            max_eval=2 * max_iter,
            tolerance_grad=1e-11,
            tolerance_change=1e-14,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            coeffs = coeffs_from_y()
            energy = self.energy_parts(coeffs)[0]
            energy.backward()
            return energy

        optimizer.step(closure)
        with torch.no_grad():
            coeffs = coeffs_from_y().clone()
            parts = self.energy_parts(coeffs)
            frac = self.nonspherical_mass_fraction(coeffs)
        return coeffs, parts, frac


def amplitude_for_fraction(fraction):
    # If one added mode has the same radial shape as l=0, amplitude a gives
    # fraction a^2/(1+a^2).
    return math.sqrt(fraction / (1.0 - fraction))


def default_tests():
    return [
        ("quadrupole", {2: amplitude_for_fraction(0.0826)}),
        ("quadrupole", {2: amplitude_for_fraction(0.2647)}),
        ("octupole", {3: amplitude_for_fraction(0.20)}),
        ("mixed", {2: 0.55, 4: 0.25}),
        ("strong quadrupole", {2: amplitude_for_fraction(0.50)}),
        ("strong quadrupole", {2: amplitude_for_fraction(0.6923)}),
        ("strong quadrupole", {2: amplitude_for_fraction(0.80)}),
        ("random even modes", {2: 0.35, 4: -0.28, 6: 0.22}),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=80)
    ap.add_argument("--R", type=float, default=25.0)
    ap.add_argument("--lmax", type=int, default=6)
    ap.add_argument("--nang", type=int, default=48)
    ap.add_argument("--g", type=float, default=120.0)
    ap.add_argument("--max-iter", type=int, default=900)
    ap.add_argument("--out", default="nonlinear_nonspherical_results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    model = AxisymmetricWCT(
        n=args.N,
        radius=args.R,
        lmax=args.lmax,
        nang=args.nang,
        mass=1.0,
        eps=0.5,
        alpha=1.0,
        delta=0.1,
        g=args.g,
    )

    rows = []
    for name, weights in default_tests():
        seed = model.radial_seed()
        seed = model.deform(seed, weights)
        initial_frac = float(model.nonspherical_mass_fraction(seed))
        coeffs, parts, final_frac = model.solve(seed, max_iter=args.max_iter)
        energy = float(parts[0])
        rows.append(
            {
                "test": name,
                "mode_weights": json.dumps(weights, sort_keys=True),
                "initial_nonspherical_mass_fraction": initial_frac,
                "final_energy": energy,
                "final_nonspherical_mass_fraction": float(final_frac),
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "nonlinear_nonspherical_tests.csv", index=False)
    summary = {
        "N": args.N,
        "R": args.R,
        "lmax": args.lmax,
        "nang": args.nang,
        "g": args.g,
        "max_initial_nonspherical_mass_fraction": float(
            frame["initial_nonspherical_mass_fraction"].max()
        ),
        "max_final_nonspherical_mass_fraction": float(
            frame["final_nonspherical_mass_fraction"].max()
        ),
        "min_final_energy": float(frame["final_energy"].min()),
        "scope": (
            "Axisymmetric real-field nonlinear deformation test in a truncated "
            "m=0 spherical-harmonic basis; not full 3D complex orbital stability."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
