#!/usr/bin/env python3
"""
Reproducibility suite for the corrected radial WCT confinement test.

Requires:
    numpy
    pandas
    scipy
    torch

This script imports RadialWCT from wct_radial_confinement_test.py and runs:
  1. direct critical-coupling minimization from multiple starting widths,
  2. a polished g=120 stationary-state calculation,
  3. delta=0 versus delta>0 narrow-probe UV diagnostics.

The archived summary files in this directory also record separate domain, grid,
radial-Hessian, and parameter-sensitivity sweeps performed with the same
RadialWCT discretization.

No physical-unit calibration is performed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize as opt
import torch

from wct_radial_confinement_test import RadialWCT

torch.set_default_dtype(torch.float64)


def solve_return(model: RadialWCT, start_width=1.4, max_iter=800):
    sqrtw = torch.sqrt(model.w)
    phi0 = torch.exp(-0.5 * (model.r / start_width) ** 2)
    phi0 = model.normalize(phi0)
    y = torch.nn.Parameter((sqrtw * phi0).clone())

    def field():
        x = y / (torch.linalg.vector_norm(y) + 1e-300)
        return x / sqrtw

    optimizer = torch.optim.LBFGS(
        [y],
        lr=0.5,
        max_iter=max_iter,
        max_eval=2 * max_iter,
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        phi = field()
        energy = model.energy_parts(phi)[0]
        energy.backward()
        return energy

    optimizer.step(closure)
    with torch.no_grad():
        return field().clone()


def scipy_polish(model: RadialWCT, phi0, maxiter=1000):
    sqrtw = torch.sqrt(model.w)
    y0 = (sqrtw * phi0).detach().cpu().numpy().copy()

    def fg(y_np):
        y = torch.tensor(y_np, dtype=torch.float64, requires_grad=True)
        x = y / (torch.linalg.vector_norm(y) + 1e-300)
        phi = x / sqrtw
        energy = model.energy_parts(phi)[0]
        grad, = torch.autograd.grad(energy, y)
        return float(energy.detach()), grad.detach().cpu().numpy()

    result = opt.minimize(
        fg,
        y0,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": maxiter,
            "ftol": 1e-15,
            "gtol": 1e-12,
            "maxls": 50,
            "maxcor": 100,
        },
    )
    y = torch.tensor(result.x, dtype=torch.float64)
    x = y / torch.linalg.vector_norm(y)
    return x / sqrtw


def projected_gradient_residual(model: RadialWCT, phi):
    sqrtw = torch.sqrt(model.w)
    x = (sqrtw * phi).detach().clone().requires_grad_(True)
    energy = model.energy_parts(x / sqrtw)[0]
    grad, = torch.autograd.grad(energy, x)
    x0 = x.detach()
    lam = (grad.detach() @ x0) / (x0 @ x0)
    projected = grad.detach() - lam * x0
    return float(
        torch.linalg.vector_norm(projected)
        / (torch.linalg.vector_norm(grad.detach()) + 1e-30)
    )


def critical_coupling(model: RadialWCT, start_width=1.4, max_iter=700):
    sqrtw = torch.sqrt(model.w)
    phi0 = model.normalize(torch.exp(-0.5 * (model.r / start_width) ** 2))
    y = torch.nn.Parameter((sqrtw * phi0).clone())

    def field():
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

    def closure():
        optimizer.zero_grad()
        phi = field()
        _, _, kinetic, bih, curvature, quartic, _ = model.energy_parts(phi)
        positive = kinetic + model.delta * bih + curvature
        quotient = 2.0 * positive / quartic
        quotient.backward()
        return quotient

    optimizer.step(closure)
    with torch.no_grad():
        phi = field()
        _, mass, kinetic, bih, curvature, quartic, _ = model.energy_parts(phi)
        gc = 2.0 * (kinetic + model.delta * bih + curvature) / quartic
        rms = torch.sqrt(torch.sum(model.w * model.r**2 * phi**2) / mass)
    return float(gc), float(rms)


def gaussian_probe_energy(N, R, width, delta, g):
    model = RadialWCT(
        n=N, radius=R, mass=1.0, eps=0.5, alpha=1.0, delta=delta, g=g
    )
    phi = model.normalize(torch.exp(-0.5 * (model.r / width) ** 2))
    return float(model.energy_parts(phi)[0])


def main():
    out = Path("results_radial_wct")
    out.mkdir(exist_ok=True)

    # Direct gc calculation.
    gc_rows = []
    for width in (0.8, 1.4, 2.5):
        model = RadialWCT(
            n=128, radius=20, mass=1, eps=0.5, alpha=1, delta=0.1, g=0
        )
        gc, rms = critical_coupling(model, width)
        gc_rows.append({"start_width": width, "gc": gc, "rms": rms})
    pd.DataFrame(gc_rows).to_csv(out / "critical_coupling.csv", index=False)

    # g=120 stationary candidate.
    model = RadialWCT(
        n=160, radius=25, mass=1, eps=0.5, alpha=1, delta=0.1, g=120
    )
    phi = solve_return(model, 1.4)
    phi = scipy_polish(model, phi)
    E, mass, kinetic, bih, curvature, quartic, _ = model.energy_parts(phi)
    pohoz_abs, pohoz_rel = model.pohozaev(phi)
    residual = projected_gradient_residual(model, phi)
    rms = torch.sqrt(torch.sum(model.w * model.r**2 * phi**2) / mass)

    stationary = {
        "energy": float(E),
        "mass": float(mass),
        "rms_radius": float(rms),
        "kinetic": float(kinetic),
        "delta_biharmonic": float(model.delta * bih),
        "curvature": float(curvature),
        "attraction": float(-0.5 * model.g * quartic),
        "pohozaev_absolute": pohoz_abs,
        "pohozaev_relative": pohoz_rel,
        "projected_gradient_relative": residual,
    }
    (out / "stationary_g120.json").write_text(
        json.dumps(stationary, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {"r": model.r.detach().cpu().numpy(), "phi": phi.detach().cpu().numpy()}
    ).to_csv(out / "profile_g120.csv", index=False)

    # UV control.
    uv_rows = []
    for width in (0.30, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05):
        for delta in (0.0, 0.1):
            uv_rows.append(
                {
                    "width": width,
                    "delta": delta,
                    "energy": gaussian_probe_energy(
                        2048, 5.0, width, delta, 120.0
                    ),
                }
            )
    pd.DataFrame(uv_rows).to_csv(out / "uv_control.csv", index=False)

    print(json.dumps(
        {
            "gc_mean": float(np.mean([x["gc"] for x in gc_rows])),
            "gc_spread": float(np.ptp([x["gc"] for x in gc_rows])),
            "stationary": stationary,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
