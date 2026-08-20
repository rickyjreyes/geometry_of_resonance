#!/usr/bin/env python3
"""
Radial 3D confinement test for the corrected WCT curvature operator.

Tests the fixed-mass functional

    E[phi] = ∫ [ |∇phi|^2
                 + (delta + F(phi^2)) |Δphi|^2
                 - (g/2) phi^4 ] d^3x

with

    F(u) = u / (u + eps^2 exp(-2 alpha u))^2.

The radial field is optimized at fixed L2 mass using a mass-weighted sphere
parameterization and PyTorch L-BFGS. A 3D Pohozaev/Derrick residual is reported
as an independent stationarity diagnostic.

This is a nondimensional numerical experiment, not an empirical validation.
"""

import argparse
import json
import math

import torch

torch.set_default_dtype(torch.float64)


class RadialWCT:
    def __init__(
        self,
        n=128,
        radius=20.0,
        mass=1.0,
        eps=0.5,
        alpha=1.0,
        delta=0.1,
        g=120.0,
    ):
        self.n = n
        self.radius = radius
        self.mass = mass
        self.eps = eps
        self.alpha = alpha
        self.delta = delta
        self.g = g

        self.dr = radius / n
        self.r = (torch.arange(n) + 0.5) * self.dr
        self.faces = torch.arange(n + 1) * self.dr
        self.w = 4.0 * math.pi * self.r**2 * self.dr

    def normalize(self, phi):
        m = torch.sum(self.w * phi**2)
        return phi * torch.sqrt(torch.tensor(self.mass) / (m + 1e-300))

    def laplacian(self, phi):
        """Finite-volume radial Laplacian in 3D."""
        flux = torch.zeros(self.n + 1, dtype=phi.dtype, device=phi.device)

        # Interior faces: r^2 dphi/dr
        flux[1:self.n] = (
            self.faces[1:self.n] ** 2
            * (phi[1:] - phi[:-1])
            / self.dr
        )

        # r=0 symmetry: dphi/dr = 0, so flux[0] = 0.

        # Outer Dirichlet boundary phi(R)=0.
        flux[self.n] = (
            self.faces[self.n] ** 2
            * (0.0 - phi[-1])
            / (0.5 * self.dr)
        )

        return (
            (flux[1:] - flux[:-1])
            / (self.r**2 * self.dr)
        )

    def gradient_energy(self, phi):
        interior = (
            self.faces[1:self.n] ** 2
            * ((phi[1:] - phi[:-1]) / self.dr) ** 2
            * self.dr
        )
        outer = (
            self.faces[self.n] ** 2
            * ((0.0 - phi[-1]) / (0.5 * self.dr)) ** 2
            * (0.5 * self.dr)
        )
        return 4.0 * math.pi * (torch.sum(interior) + outer)

    def F(self, u):
        q = self.eps**2 * torch.exp(-2.0 * self.alpha * u)
        return u / (u + q) ** 2

    def energy_parts(self, phi):
        lap = self.laplacian(phi)
        u = phi**2

        kinetic = self.gradient_energy(phi)
        biharmonic = torch.sum(self.w * lap**2)
        curvature = torch.sum(self.w * self.F(u) * lap**2)
        quartic = torch.sum(self.w * phi**4)

        energy = (
            kinetic
            + self.delta * biharmonic
            + curvature
            - 0.5 * self.g * quartic
        )
        mass = torch.sum(self.w * phi**2)

        return energy, mass, kinetic, biharmonic, curvature, quartic, lap

    def pohozaev(self, phi):
        energy, mass, kinetic, biharmonic, curvature, quartic, lap = (
            self.energy_parts(phi)
        )

        u = phi**2
        q = self.eps**2 * torch.exp(-2.0 * self.alpha * u)

        # 4 F(u) + 3 u F'(u)
        coeff = (
            u * (u + 7.0 * q + 12.0 * self.alpha * u * q)
            / (u + q) ** 3
        )
        nonlinear_scaling = torch.sum(self.w * coeff * lap**2)

        residual = (
            2.0 * kinetic
            + 4.0 * self.delta * biharmonic
            + nonlinear_scaling
            - 1.5 * self.g * quartic
        )

        scale = max(
            abs(float(
                2.0 * kinetic
                + 4.0 * self.delta * biharmonic
                + nonlinear_scaling
            )),
            abs(float(1.5 * self.g * quartic)),
            1e-30,
        )

        return float(residual), float(residual) / scale

    def solve(self, start_width=1.4, max_iter=800):
        sqrtw = torch.sqrt(self.w)

        phi0 = torch.exp(-0.5 * (self.r / start_width) ** 2)
        phi0 = self.normalize(phi0)

        # x_i = sqrt(w_i) phi_i makes fixed mass a Euclidean sphere.
        x0 = sqrtw * phi0
        y = torch.nn.Parameter(x0.clone())

        def field_from_y():
            x = y / (torch.linalg.vector_norm(y) + 1e-300)
            return x / sqrtw

        optimizer = torch.optim.LBFGS(
            [y],
            lr=0.5,
            max_iter=max_iter,
            max_eval=2 * max_iter,
            tolerance_grad=1e-11,
            tolerance_change=1e-14,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        calls = 0

        def closure():
            nonlocal calls
            calls += 1
            optimizer.zero_grad()
            phi = field_from_y()
            energy = self.energy_parts(phi)[0]
            energy.backward()
            return energy

        optimizer.step(closure)

        with torch.no_grad():
            phi = field_from_y()
            (
                energy,
                mass,
                kinetic,
                biharmonic,
                curvature,
                quartic,
                _lap,
            ) = self.energy_parts(phi)

            rms = torch.sqrt(
                torch.sum(self.w * self.r**2 * phi**2) / mass
            )
            inside_r5 = (
                torch.sum(self.w[self.r < 5.0] * phi[self.r < 5.0] ** 2)
                / mass
            )

        p_abs, p_rel = self.pohozaev(phi)

        return {
            "N": self.n,
            "R": self.radius,
            "M": self.mass,
            "eps": self.eps,
            "alpha": self.alpha,
            "delta": self.delta,
            "g": self.g,
            "energy": float(energy),
            "kinetic": float(kinetic),
            "delta_biharmonic": float(self.delta * biharmonic),
            "curvature_term": float(curvature),
            "quartic_norm": float(quartic),
            "mass": float(mass),
            "rms_radius": float(rms),
            "mass_inside_r5": float(inside_r5),
            "peak_amplitude": float(torch.max(torch.abs(phi))),
            "boundary_amplitude": float(torch.abs(phi[-1])),
            "pohozaev_residual": p_abs,
            "pohozaev_relative": p_rel,
            "optimizer_calls": calls,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--g", type=float, default=120.0)
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--R", type=float, default=20.0)
    parser.add_argument("--M", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--width", type=float, default=1.4)
    parser.add_argument("--max-iter", type=int, default=800)
    args = parser.parse_args()

    model = RadialWCT(
        n=args.N,
        radius=args.R,
        mass=args.M,
        eps=args.eps,
        alpha=args.alpha,
        delta=args.delta,
        g=args.g,
    )
    result = model.solve(
        start_width=args.width,
        max_iter=args.max_iter,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
