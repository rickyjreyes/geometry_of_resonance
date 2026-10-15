"""Reproduce unchanged v18 in-sample baseline and audit only the two live optimizer parameters.

Usage:
    python SM_Spectrum/audit_v18.py --output audit/v18_baseline.json

Does NOT reoptimize stored coefficients, assign modes, or claim prospective holdout.
"""
import argparse
import contextlib
import hashlib
import io
import json
import runpy
from pathlib import Path

import numpy as np
import scipy

SOURCE = Path(__file__).with_name("_terms_perfection_v18.py")

def run_audit():
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        namespace = runpy.run_path(str(SOURCE), run_name="__wct_v18_audit__")
    result = namespace["result"]
    targets = namespace["all_targets"]
    predict = namespace["total_mass"]
    x = np.asarray(result.x, dtype=float)
    rows = []
    for n, observed in targets.items():
        value = float(predict(n, *x))
        relative = abs(value - observed) / (observed + 1e-6)
        rows.append(dict(index=int(n), target_mev=float(observed), fitted_mev=value,
                         residual_mev=value - float(observed), target_relative_loss=float(relative),
                         absolute_relative_error_nonzero=(float(abs(value-observed)/observed) if observed else None),
                         at_mass_floor=bool(value <= 1.000001e-12),
                         index_branch=("light" if n in namespace["light_targets"] else
                                       "middle" if n in namespace["middle_targets"] else "heavy")))
    nonzero = [r for r in rows if r["target_mev"] > 0]
    # Sensitivity is for the TWO free suppression variables only, not the other 29 stored values.
    indices = [r["index"] for r in nonzero]
    baseline = np.array([float(predict(n, *x)) for n in indices])
    jac = np.zeros((len(indices), 2))
    bounds = [(0.0,50.0),(0.0,5.0)]
    scheme = []
    for j in range(2):
        h = 1e-5 * max(1.0, abs(x[j]))
        hi, lo = x.copy(), x.copy()
        if x[j] + h <= bounds[j][1] and x[j] - h >= bounds[j][0]:
            hi[j] += h
            lo[j] -= h
            jac[:, j] = (np.array([float(predict(n,*hi)) for n in indices]) -
                         np.array([float(predict(n,*lo)) for n in indices]))/(2*h)
            scheme.append("central")
        elif x[j]+h <= bounds[j][1]:
            hi[j] += h
            jac[:,j] = (np.array([float(predict(n,*hi)) for n in indices])-baseline)/h
            scheme.append("forward")
        elif x[j]-h >= bounds[j][0]:
            lo[j] -= h
            jac[:,j] = (baseline-np.array([float(predict(n,*lo)) for n in indices]))/h
            scheme.append("backward")
        else:
            scheme.append("no feasible finite difference")
    singular = np.linalg.svd(jac, full_matrices=False, compute_uv=False)
    rank = int(np.linalg.matrix_rank(jac))
    cond = (float(singular[0]/singular[-1]) if len(singular) > 1 and singular[-1] > 0 else None)
    data = {
        "classification":"EMPIRICAL_FIT_ONLY", "no_holdout":True,
        "source":"SM_Spectrum/_terms_perfection_v18.py",
        "source_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "versions":{"numpy":np.__version__,"scipy":scipy.__version__},
        "stored_coefficients":29, "live_optimized_coefficients":2,
        "optimizer":{"method":"L-BFGS-B","bounds":bounds,"x":x.tolist(),
                     "success":bool(result.success),"status":int(result.status),
                     "message":str(result.message),"nit":int(result.nit),
                     "nfev":int(result.nfev),"fun":float(result.fun),
                     "at_bounds":[bool(abs(x[j]-l)<1e-8 or abs(x[j]-u)<1e-8) for j,(l,u) in enumerate(bounds)]},
        "objective_includes_zero_target":True,
        "zero_target_denominator":1e-6,
        "all_target_mean_loss":float(np.mean([r["target_relative_loss"] for r in rows])),
        "nonzero_mean_abs_relative_error":float(np.mean([r["absolute_relative_error_nonzero"] for r in nonzero])),
        "nonzero_target_count":len(nonzero),
        "nominal_total_coefficient_to_nonzero_target_ratio":31/len(nonzero),
        "clipping":{"light_exponential": "np.clip(lambda_l*n,-50,50)",
                    "heavy_powers": "np.clip(n/50 or n/100,1e-6,1e6); selected exponents clipped [-10,10]",
                    "mass_floor":"max(prediction,1e-12)"},
        "two_free_parameter_jacobian":{"indices":indices,"columns":["alpha_decay","epsilon_cutoff"],
          "finite_difference_scheme":scheme,"matrix":jac.tolist(),"singular_values":singular.tolist(),
          "numerical_rank":rank,"condition_number":cond,
          "warning":"NOT the rank/identifiability of the 29 stored historical coefficients"},
        "per_target":rows,"original_program_stdout":capture.getvalue()
    }
    assert len(rows)==12 and len(nonzero)==11
    assert np.isfinite(data["all_target_mean_loss"])
    return data

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output", type=Path)
    args=p.parse_args()
    data=run_audit()
    out=json.dumps(data,indent=2,allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(out,encoding="utf-8")
        print("Wrote",args.output)
    else:
        print(out)
