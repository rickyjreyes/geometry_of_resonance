"""Full *local* 31-coefficient sensitivity of unchanged, in-sample v18.

29 stored coefficients + 2 historically live optimized variables. Never fit them
again, never reassign labels by target mass, never call this a holdout/eigenmode
derivation. Unlike audit_v18.py, this audit examines sensitivity of all stored
coefficients at the current frozen source values.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import runpy
from pathlib import Path

import numpy as np
import scipy

HERE=Path(__file__).resolve().parent
SOURCE=HERE/"_terms_perfection_v18.py"
CSV=HERE.parent/"Wavefield_Folding"/"cone_helmholtz_spectrum.csv"


def execute():
    with contextlib.redirect_stdout(io.StringIO()):
        ns=runpy.run_path(str(SOURCE),run_name="__wct_v18_full_jacobian__")
    opt=ns["result"]
    assert bool(opt.success), "Historical two-live-parameter optimizer did not converge"
    light=np.array(ns["light_params_best"],dtype=float)
    middle=np.array(ns["middle_params_best"],dtype=float)
    heavy=np.array(ns["heavy_params_best"],dtype=float)
    x=np.r_[light,middle,heavy,np.asarray(opt.x,dtype=float)]
    assert len(x)==31 and len(light)==9 and len(middle)==7 and len(heavy)==13
    keys=[int(n) for n,v in ns["all_targets"].items() if v>0]
    observations=np.array([ns["all_targets"][n] for n in keys],dtype=float)
    assert len(keys)==11
    def predictions(coeff):
        l,m,h=coeff[:9],coeff[9:16],coeff[16:29]
        alpha,eps=coeff[29:31]
        values=[]
        for n in keys:
            if n in ns["light_targets"]:
                pred=ns["wct_mass_light_fixed"](n,l,alpha,eps)
            elif n in ns["middle_targets"]:
                pred=ns["wct_mass_middle"](n,m)
            else:
                pred=ns["wct_mass_heavy"](n,h,l,alpha,eps)
            values.append(float(pred))
        return np.asarray(values,dtype=float)
    base=predictions(x)
    previous=np.array([float(ns["total_mass"](n,*opt.x)) for n in keys])
    if not np.allclose(base,previous,atol=1e-10,rtol=1e-10):
        raise AssertionError("Full-coefficient evaluator differs from unchanged original v18")
    J=np.zeros((len(keys),len(x)))
    schemes=[]
    for j in range(len(x)):
        step=1e-5*max(1.,abs(x[j]))
        hi=x.copy()
        lo=x.copy()
        hi[j]+=step
        lo[j]-=step
        # Only the original *live* suppression parameters have optimizer bounds;
        # epsilon is precisely at its upper bound.
        if j==30 and hi[j]>5.:
            J[:,j]=(base-predictions(lo))/step
            schemes.append("backward, active live upper bound")
        elif j==29 and (lo[j]<0 or hi[j]>50):
            J[:,j]=(predictions(hi)-base)/step if lo[j]<0 else (base-predictions(lo))/step
            schemes.append("feasible one-sided, live bound")
        else:
            J[:,j]=(predictions(hi)-predictions(lo))/(2*step)
            schemes.append("central local perturbation, historical coefficient held otherwise")
    # Scaled sensitivity is solely a numerical diagnostic, not a statistical
    # information matrix or globally identifiable parameter count.
    scales=np.maximum(1.,np.abs(x))
    Jscaled=(J*scales[None,:])/observations[:,None]
    singular=np.linalg.svd(Jscaled,full_matrices=False,compute_uv=False)
    relative_tolerance=1e-8
    numerical_rank=int(np.count_nonzero(singular>relative_tolerance*singular[0]))
    with CSV.open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        header=reader.fieldnames
        spectra=list(reader)
    wanted=["Curvature_Deficit"]+[f"k2_m{i}" for i in range(6)]
    assert header==wanted, "Input CSV layout changed; re-audit rather than reinterpret"
    vals=np.array([[float(row[k]) for k in wanted] for row in spectra])
    jumps=np.abs(np.diff(vals[:,1:],axis=0))
    # Large jumps diagnose insufficient mode identity information. They do NOT
    # establish a genuine spectral crossing without eigenfunctions/overlaps.
    jump_per_branch=np.max(jumps,axis=0).tolist()
    report={
      "classification":"EMPIRICAL_FIT_ONLY",
      "no_holdout":True,"no_new_fit":True,"no_particle_reassignment":True,
      "source_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
      "csv_sha256":hashlib.sha256(CSV.read_bytes()).hexdigest(),
      "versions":{"numpy":np.__version__,"scipy":scipy.__version__},
      "nonzero_target_count":len(keys),"indices":keys,"observed_mev":observations.tolist(),
      "baseline_predicted_mev":base.tolist(),
      "coefficient_sections":{"light":[0,9],"middle":[9,16],"heavy":[16,29],"live":[29,31]},
      "stored_count":29,"live_count":2,"total_nominal_coefficients":31,
      "sensitivity":{"raw_shape":list(J.shape),"scaled_singular_values":singular.tolist(),
         "relative_singular_threshold":relative_tolerance,"local_scaled_rank":numerical_rank,
         "structural_rank_upper_bound":len(keys),
         "structural_nullity_lower_bound":len(x)-len(keys),
         "local_scaled_nullity_lower_bound":len(x)-numerical_rank,
         "column_schemes":schemes,
         "scaled_column_norms":np.linalg.norm(Jscaled,axis=0).tolist(),
         "normalized_residual_jacobian":Jscaled.tolist(),
         "warning":"31 coefficients on only 11 nonzero outputs: rank <=11 and nullity >=20 "
                   "for any differentiable local model, regardless of finite-difference conditioning. "
                   "Stored coefficients were not reoptimized; n=0 is excluded here. "
                   "No global identifiability, inference interval or true holdout is implied."},
      "cone_csv":{"header":header,"rows":len(spectra),
         "max_adjacent_absolute_k2_change_by_column":jump_per_branch,
         "units_in_header":False,"operator_domain_and_boundary_supplied_by_csv":False,
         "eigenfunction_overlaps_or_mode_identity_supplied":False,
         "physical_mev_conversion_supplied":False,
         "warning":"Printed k2_m labels and sorted numerical columns do not establish "
                   "persistent eigenmode identity through crossing/reordering."}
    }
    if not (len(spectra)>1 and np.isfinite(Jscaled).all() and 0<=numerical_rank<=11):
        raise AssertionError("Invalid diagnostic result")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    result=execute()
    payload=json.dumps(result,indent=2,allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(payload,encoding="utf-8")
        print("Wrote",args.output,"; nominal nullity >=",result["sensitivity"]["structural_nullity_lower_bound"])
    else:
        print(payload)
