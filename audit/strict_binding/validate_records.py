#!/usr/bin/env python3
"""Fail closed on altered evidence or an overstated universal/D4 conclusion."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def main():
    manifest=json.loads((HERE/"manifest.json").read_text())
    for row in manifest["files"]:
        assert hashlib.sha256((ROOT/row["path"]).read_bytes()).hexdigest()==row["sha256"],row["path"]
    source=json.loads((HERE/"source_ledger.json").read_text())
    for row in source["local_inherited_files"]:
        assert hashlib.sha256((ROOT/row["path"]).read_bytes()).hexdigest()==row["sha256"],row["path"]
    matrix=json.loads((HERE/"claim_matrix.json").read_text())
    assert matrix["primary"]=="STRICT_BINDING_RESTRICTED"
    assert matrix["all_positive_masses"]=="OPEN"
    assert matrix["D4_all_splits"]=="OPEN"
    assert matrix["D4_whole_space_existence"]=="OPEN"
    assert matrix["D4_whole_space_orbital_stability"]=="OPEN"
    rows={r["id"]:r for r in matrix["claims"]}
    assert len(rows)==len(matrix["claims"])
    for row in rows.values():
        assert row["scope"] and (ROOT/row["proof"]).is_file()
    v=json.loads((HERE/"verification.json").read_text())
    assert v["status"]=="PASS" and len(v["symbolic_checks"])==14
    assert all(x["passed"] for x in v["symbolic_checks"].values())
    assert not v["D4_all_splits_proved"] and not v["all_positive_masses_proved"]
    beta=F(float.fromhex(v["parameters"]["beta_hex"]))
    m=F(float.fromhex(v["parameters"]["D4_mass_hex"]))
    ms=4/(1+3*beta/16)
    assert F(v["exact_constants"]["m_star"])==ms<m
    assert list(map(F,v["D4_certified_split_interval_exact"]))==[2*m-2*ms,2*ms-m]
    diag=json.loads((HERE/"diagnostics.json").read_text())
    assert diag["status"]=="PASS" and not diag["full_functional_counterexample_found"]
    assert all(not x["certifies_infimum_comparison"] for x in diag["D4_trial_split_comparisons"])
    exe=json.loads((HERE/"execution.json").read_text())
    assert exe["all_commands_passed"] and len(exe["commands"])==13
    assert exe["verification_mode"]=="full-local"
    assert exe["scope"]["D4_private_inputs_hash_checked"]
    assert exe["scope"]["D4_rational_replay_executed"]
    assert not exe["scope"]["recorded_D4_evidence_validated"]
    for name,digest in exe["implementation_sha256"].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,name
    for row in exe["commands"]:
        assert row["returncode"]==0
        assert hashlib.sha256((HERE/row["log"]).read_bytes()).hexdigest()==row["log_sha256"]
    d4=json.loads((HERE/"d4_basin_replay.json").read_text())
    assert d4["status"]=="PASS" and len(d4["checks"])==5 and all(d4["checks"].values())
    assert not exe["scope"]["original_D4_interval_campaign_rerun"]
    assert not exe["scope"]["physical_spectrum"]
    print(f"PASS: {len(manifest['files'])} Phase IV hashes, {len(source['local_inherited_files'])} inherited pins, {len(rows)} scoped claims, 13 execution groups")
    print("Recorded full-local D4 replay evidence validated; this integrity check does not execute the private-source replay.")


if __name__=="__main__":main()
