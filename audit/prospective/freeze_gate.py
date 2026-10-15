"""Fail-closed administrative preregistration checker for issue #17.

NEVER inspects outcomes, executes the holdout, approves its own independent
signoff, or upgrades a validated form into a scientifically derived prediction.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

HERE=Path(__file__).resolve().parent
TEMPLATE=HERE/"template.json"
HASH=re.compile(r"^[0-9a-f]{64}$")
SHA=re.compile(r"^[0-9a-f]{40}$")
REQUIRED_THEORY=("equation_ref","derivation_ref","observable","dimensionless_target",
                 "predeclared_uncertainty","detector_forward_map_ref")
REQUIRED_PROTOCOL=("source_manifest_sha256","analysis_commit_sha",
   "train_ids","calibration_ids","holdout_ids","transforms_ref","exclusions_ref",
   "null_model_ref","background_alternatives_ref","multiplicity_ref",
   "statistic_ref","decision_thresholds_ref","injection_ref","calibration_ref",
   "deviation_policy_ref")


def check(spec):
    blockers=[]
    if spec.get("schema")!="wct.prospective.v1":
        blockers.append("unrecognized schema")
    theory=spec.get("theory") or {}
    protocol=spec.get("protocol") or {}
    attest=spec.get("attestation") or {}
    execution=spec.get("execution") or {}
    for key in REQUIRED_THEORY:
        if theory.get(key) is None or theory.get(key)=="":
            blockers.append("theory."+key+" missing")
    for key in REQUIRED_PROTOCOL:
        if protocol.get(key) is None or protocol.get(key)=="":
            blockers.append("protocol."+key+" missing")
    if not isinstance(theory.get("dimensionless_target"),dict) or not all(
       x in theory.get("dimensionless_target",{}) for x in ("lower","upper")):
        blockers.append("numerical target interval not specified")
    if not isinstance(protocol.get("source_manifest_sha256"),str) or not HASH.fullmatch(protocol["source_manifest_sha256"]):
        blockers.append("source content digest invalid")
    if not isinstance(protocol.get("analysis_commit_sha"),str) or not SHA.fullmatch(protocol["analysis_commit_sha"]):
        blockers.append("analysis freeze commit invalid")
    populations=[]
    for key in ("train_ids","calibration_ids","holdout_ids"):
        ids=protocol.get(key)
        if not isinstance(ids,list) or not ids or not all(isinstance(x,str) and x for x in ids):
            blockers.append(key+" must be nonempty array of opaque unique IDs")
            continue
        if len(set(ids))!=len(ids):
            blockers.append(key+" has duplicates")
        populations.append((key,set(ids)))
    if len(populations)==3:
        for i in range(3):
            for j in range(i+1,3):
                if populations[i][1]&populations[j][1]:
                    blockers.append(populations[i][0]+" overlaps "+populations[j][0])
    if attest.get("holdout_uninspected") is not True:
        blockers.append("prior access to heldout data not independently attested")
    if not attest.get("independent_review_ref"):
        blockers.append("independent review not documented")
    if execution.get("unblinded") is not False or execution.get("result") is not None:
        blockers.append("this checker cannot validate an executed/unblinded experiment")
    # Passing only means that the metadata fields were populated and the opaque
    # split IDs are disjoint. It cannot verify truth, scientific derivation,
    # immutable storage, or absence of historical target exposure.
    return {"schema":"wct.prospective.gate.v1",
            "status":"BLOCKED" if blockers else "METADATA_READY_FOR_EXTERNAL_VERIFICATION_NOT_FROZEN",
            "blockers":blockers,"no_holdout_executed_by_checker":True,
            "no_claim_of_prospective_prediction":True}


def self_test():
    spec=json.loads(TEMPLATE.read_text(encoding="utf-8"))
    blocked=check(spec)
    assert blocked["status"]=="BLOCKED" and len(blocked["blockers"])>5
    # An invented/partial specification must not be treated as real evidence.
    assert blocked["no_holdout_executed_by_checker"] is True
    assert check({})["status"]=="BLOCKED"
    print("Prospective gate fail-closed self-test: PASS")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--spec",type=Path,default=TEMPLATE)
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:
        self_test()
    else:
        report=check(json.loads(args.spec.read_text(encoding="utf-8")))
        print(json.dumps(report,indent=2))
        if report["status"]=="BLOCKED":
            raise SystemExit(2)
