#!/usr/bin/env python3
"""
Nonspherical stability audit for the corrected 3D radial WCT candidate.

Requires wct_radial_confinement_test.py from the previous radial experiment.

The script:
  1. solves/refines the radial fixed-mass candidate,
  2. constructs the constrained second variation in spherical-harmonic sectors l>=1,
  3. checks the l=1 translation/Goldstone mode against dphi/dr,
  4. reports the smallest eigenvalue for each l.

The second variation is for real perturbations
    psi = phi(r) + t v(r) Y_lm(Ω)
of
    E = ∫[|∇psi|² + (delta + F(psi²))|Δpsi|² - (g/2)psi⁴] d³x,
where
    F(u)=u/(u+eps² exp(-2 alpha u))².

This is a nondimensional numerical stability test, not a continuum proof.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize as opt
import torch

from wct_radial_confinement_test import RadialWCT

torch.set_default_dtype(torch.float64)


def solve_return(model: RadialWCT, start_width=1.4, max_iter=800):
    sqrtw = torch.sqrt(model.w)
    phi0 = model.normalize(torch.exp(-0.5 * (model.r / start_width) ** 2))
    y = torch.nn.Parameter((sqrtw * phi0).clone())

    def field():
        x = y / (torch.linalg.vector_norm(y) + 1e-300)
        return x / sqrtw

    optimizer = torch.optim.LBFGS(
        [y], lr=0.5, max_iter=max_iter, max_eval=2 * max_iter,
        tolerance_grad=1e-12, tolerance_change=1e-15,
        history_size=100, line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        phi = field()
        energy = model.energy_parts(phi)[0]
        energy.backward()
        return energy

    optimizer.step(closure)
    return field().detach().clone()


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
        fg, y0, jac=True, method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-12,
                 "maxls": 50, "maxcor": 100},
    )
    y = torch.tensor(result.x, dtype=torch.float64)
    x = y / torch.linalg.vector_norm(y)
    return (x / sqrtw).detach()


def radial_matrix(model: RadialWCT):
    N = model.n
    dr = float(model.dr)
    r = model.r.detach().cpu().numpy()
    faces = model.faces.detach().cpu().numpy()

    L0 = np.zeros((N, N))
    for i in range(N):
        denom = r[i] ** 2 * dr

        if i + 1 < N:
            a = faces[i + 1] ** 2 / dr
            L0[i, i] -= a / denom
            L0[i, i + 1] += a / denom
        else:
            a = faces[N] ** 2 / (0.5 * dr)
            L0[i, i] -= a / denom

        if i > 0:
            a = faces[i] ** 2 / dr
            L0[i, i] -= a / denom
            L0[i, i - 1] += a / denom

    return L0


def constraint_lambda(model: RadialWCT, phi: torch.Tensor):
    sqrtw = torch.sqrt(model.w)
    x = (sqrtw * phi).detach().clone().requires_grad_(True)
    energy = model.energy_parts(x / sqrtw)[0]
    grad, = torch.autograd.grad(energy, x)
    x0 = x.detach()
    return float((grad.detach() @ x0) / (2.0 * (x0 @ x0)))


def spectrum(model: RadialWCT, phi: torch.Tensor, lmax=20):
    phi_np = phi.detach().cpu().numpy()
    N = model.n
    dr = float(model.dr)
    r = model.r.detach().cpu().numpy()
    wr = r ** 2 * dr

    L0 = radial_matrix(model)
    z0 = L0 @ phi_np
    lam = constraint_lambda(model, phi)

    u = phi_np ** 2
    q = model.eps ** 2 * np.exp(-2.0 * model.alpha * u)
    D = u + q
    Dp = 1.0 - 2.0 * model.alpha * q
    Dpp = 4.0 * model.alpha ** 2 * q

    F = u / D ** 2
    Fp = D ** -2 - 2.0 * u * Dp * D ** -3
    Fpp = (
        -4.0 * Dp * D ** -3
        -2.0 * u * Dpp * D ** -3
        +6.0 * u * Dp ** 2 * D ** -4
    )

    Hp = 2.0 * phi_np * Fp
    Hpp = 2.0 * Fp + 4.0 * u * Fpp

    W = np.diag(wr)
    invsqrt = 1.0 / np.sqrt(wr)

    rows = []
    eigvecs = {}

    for l in range(1, lmax + 1):
        Ll = L0 - np.diag(l * (l + 1) / r ** 2)

        q_grad = -(W @ Ll + Ll.T @ W)
        q_delta = 2.0 * model.delta * (Ll.T @ (wr[:, None] * Ll))
        q_local = np.diag(wr * Hpp * z0 ** 2)

        cdiag = wr * (4.0 * Hp * z0)
        q_mix = (cdiag[:, None] * Ll + Ll.T * cdiag[None, :]) / 2.0

        q_curv_lap = 2.0 * (Ll.T @ ((wr * F)[:, None] * Ll))
        q_quartic = np.diag(wr * (-6.0 * model.g * u))
        q_mass = np.diag(wr * (-2.0 * lam))

        Q = (
            q_grad + q_delta + q_local + q_mix
            + q_curv_lap + q_quartic + q_mass
        )
        Q = (Q + Q.T) / 2.0

        Qx = (invsqrt[:, None] * Q) * invsqrt[None, :]
        evals, evecs = np.linalg.eigh(Qx)
        eigvecs[l] = evecs[:, 0]
        rows.append({
            "l": l,
            "eigmin": float(evals[0]),
            "eig2": float(evals[1]),
        })

    # Translation-mode overlap for l=1.
    dp = np.gradient(phi_np, dr, edge_order=2)
    translation = np.sqrt(wr) * dp
    translation /= np.linalg.norm(translation)
    overlap = abs(float(eigvecs[1] @ translation))

    return pd.DataFrame(rows), lam, overlap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=160)
    ap.add_argument("--R", type=float, default=25.0)
    ap.add_argument("--g", type=float, default=120.0)
    ap.add_argument("--lmax", type=int, default=20)
    ap.add_argument("--out", default="nonspherical_results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    model = RadialWCT(
        n=args.N, radius=args.R, mass=1.0,
        eps=0.5, alpha=1.0, delta=0.1, g=args.g,
    )
    phi = solve_return(model)
    phi = scipy_polish(model, phi)

    table, lam, overlap = spectrum(model, phi, args.lmax)
    table.to_csv(out / "spectrum.csv", index=False)

    energy = float(model.energy_parts(phi)[0])
    summary = {
        "N": args.N,
        "R": args.R,
        "g": args.g,
        "energy": energy,
        "constraint_lambda": lam,
        "translation_overlap_l1": overlap,
        "smallest_shape_eigenvalue_l_ge_2": float(
            table.loc[table["l"] >= 2, "eigmin"].min()
        ),
        "interpretation": (
            "l=1 should converge to the neutral translation mode. "
            "Strictly positive l>=2 eigenvalues indicate no detected "
            "real nonspherical shape instability in the tested sectors."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
