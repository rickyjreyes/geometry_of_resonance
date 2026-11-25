#!/usr/bin/env python3
"""Run the actual Phase IV checks, with command/output provenance.

Use a fresh output directory. The output is a new receipt set; frozen receipts
are never overwritten. Full replay requires authorized local D4 inputs.
Public CI reruns the public checks and validates the recorded D4 evidence;
it does not rerun the private-source D4 verifier.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--d4-source-root",type=Path,
                      help="Run all 13 groups with authorized local pinned D4 inputs")
    mode.add_argument("--public-ci",action="store_true",
                      help="Run public checks and validate recorded D4 evidence without replaying private inputs")
    args=parser.parse_args()
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/"execution.json").exists():raise ValueError("Use a fresh output directory")
    py=sys.executable
    cross=out/"cross_paper"
    cross.mkdir()
    shutil.copyfile(ROOT/"audit/cross_paper/verify_derivations.py",cross/"verify.py")
    d4root=out/"d4_source"
    prepare=[py,"audit/strict_binding/prepare_d4_replay.py","--destination",str(d4root)]
    if args.d4_source_root:prepare += ["--from-root",str(args.d4_source_root.resolve())]
    commands=[
        ("independent-binding-identities",[py,"derivations/strict_binding.py","--output",str(out/"verification.json")]),
        ("adversarial-diagnostics",[py,"audit/strict_binding/run_diagnostics.py","--output",str(out/"diagnostics.json")]),
        ("all-derivation-regressions",[py,"-m","unittest","discover","-v","-s","derivations","-p","test_*.py"]),
        ("pr34-reconstruction",[py,"derivations/foundational_recovery.py","--output",str(out/"pr34_verification.json")]),
        ("pr33-reconstruction",[py,"derivations/foundational_closure.py","--output",str(out/"pr33_verification.json")]),
        ("cross-paper-replay",[py,str(cross/"verify.py")]),
        ("pr33-integrity",[py,"audit/foundational/validate_records.py"]),
        ("pr34-integrity",[py,"audit/foundational_recovery/validate_records.py"]),
        ("historical-d4-scope",[py,"audit/d4_handoff/validate_claims.py"]),
        ("prospective-gate-self-test",[py,"audit/prospective/freeze_gate.py","--self-test"]),
        ("prospective-regressions",[py,"-m","unittest","discover","-v","-s","audit/prospective","-p","test_*.py"]),
    ]
    if args.public_ci:
        commands.append(("recorded-evidence-integrity",[py,"audit/strict_binding/validate_records.py"]))
    else:
        commands.extend([
            ("prepare-pinned-d4-replay",prepare),
            ("d4-rational-replay",[py,str(d4root/"tools/closure_d4_basin_rational_replay.py"),"--output",str(out/"d4_basin_replay.json")]),
        ])
    receipt={"schema":"wct-binding-execution-v1","date_local":"2026-10-22 America/Los_Angeles",
             "frozen_base":"dff2463e4dae6c1ee4fe0c4ebc6125bf90c9f49e",
             "verification_mode":"public-ci" if args.public_ci else "full-local",
             "implementation_sha256":{name:hashlib.sha256((ROOT/"audit/strict_binding"/name).read_bytes()).hexdigest()
                                      for name in ["reproduce.py","prepare_d4_replay.py","source_ledger.json"]},
             "python":platform.python_version(),"dependencies":{name:importlib.metadata.version(name) for name in ["sympy","mpmath","numpy","scipy"]},
             "commands":[],"all_commands_passed":False,
             "scope":{"new_analytic_proofs":True,"finite_dimensional_gaussian_trial_search":True,
                      "PDE_propagation":False,"stationary_root_search":False,
                      "D4_private_inputs_hash_checked":False,
                      "D4_rational_replay_executed":False,
                      "recorded_D4_evidence_validated":False,
                      "original_D4_interval_campaign_rerun":False,"physical_spectrum":False,
                      "external_independent_review":False}}
    for name,command in commands:
        started=datetime.now(timezone.utc).isoformat();tick=time.monotonic()
        result=subprocess.run(command,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        log=out/(name+".log");log.write_text(result.stdout)
        receipt["commands"].append({"name":name,"command":command,"cwd":str(ROOT),
          "started_at_utc":started,"seconds":round(time.monotonic()-tick,3),
          "returncode":result.returncode,"log":log.name,
          "log_sha256":hashlib.sha256(log.read_bytes()).hexdigest()})
        if result.returncode==0:
            if name=="prepare-pinned-d4-replay":receipt["scope"]["D4_private_inputs_hash_checked"]=True
            if name=="d4-rational-replay":receipt["scope"]["D4_rational_replay_executed"]=True
            if name=="recorded-evidence-integrity":receipt["scope"]["recorded_D4_evidence_validated"]=True
        (out/"execution.json").write_text(json.dumps(receipt,indent=2)+"\n")
        print(f"{name}: {'PASS' if result.returncode==0 else 'FAIL'}",flush=True)
        if result.returncode:
            print(result.stdout)
            raise SystemExit(result.returncode)
    receipt["all_commands_passed"]=True
    (out/"execution.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(f"All {len(commands)} command groups passed; original D4 interval campaigns were not rerun.")
    if args.public_ci:
        print("Public CI validated the recorded D4 evidence; it did NOT execute the private-source D4 replay.")


if __name__=="__main__":main()
