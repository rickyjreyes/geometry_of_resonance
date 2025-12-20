import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.optimize import brentq
import pandas as pd

def find_first_bessel_zero(nu):
    """
    Find the first zero of the Bessel function J_nu(x) for real order nu >= 0.
    This corresponds to the lowest radial eigenvalue of the Helmholtz equation
    on a conical manifold with fractional angular order nu.
    """
    lower = max(0.1, nu)
    upper = max(5.0, nu + 20)

    val_lower = jv(nu, lower)
    val_upper = jv(nu, upper)

    if val_lower * val_upper > 0:
        x_scan = np.linspace(lower, upper, 200)
        vals = jv(nu, x_scan)
        idx = np.where(np.diff(np.sign(vals)))[0]
        if len(idx) > 0:
            i = idx[0]
            lower, upper = x_scan[i], x_scan[i+1]
        else:
            return np.nan

    try:
        return brentq(lambda x: jv(nu, x), lower, upper)
    except ValueError:
        return np.nan

def simulate_cone_spectrum():
    # Curvature parameter: alpha = 1 (flat), alpha < 1 (cone)
    alpha_values = np.linspace(1.0, 0.4, 50)

    # Angular winding numbers
    m_values = [0, 1, 2, 3, 4, 5]

    # Store lowest radial eigenvalues k^2
    results = {m: [] for m in m_values}

    for alpha in alpha_values:
        for m in m_values:
            nu = m / alpha
            k = find_first_bessel_zero(nu)
            results[m].append(k**2 if not np.isnan(k) else np.nan)

    # Plot
    plt.figure(figsize=(10, 6))
    for m in m_values:
        plt.plot(1 - alpha_values, results[m], linewidth=2, label=f"m = {m}")

    plt.title("Mode-Dependent Spectral Lifting on a Conical Manifold")
    plt.xlabel("Curvature Deficit (1 − α)")
    plt.ylabel("Lowest Radial Eigenvalue $k_{m,1}^2$")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("cone_helmholtz_spectrum.png")
    plt.show()

    # Save data
    data = {"Curvature_Deficit": 1 - alpha_values}
    for m in m_values:
        data[f"k2_m{m}"] = results[m]

    pd.DataFrame(data).to_csv("cone_helmholtz_spectrum.csv", index=False)
    print("Simulation complete.")

if __name__ == "__main__":
    simulate_cone_spectrum()
