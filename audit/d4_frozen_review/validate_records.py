#!/usr/bin/env python3
"""Validate additive review evidence without rewriting historical manifests."""

import hashlib
import json
from pathlib import Path

from certify_gap import require, verify_record


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    manifest = json.loads((HERE/"manifest.json").read_text())
    for row in manifest["files"] + manifest["inherited_pins"]:
        require(hashlib.sha256((ROOT/row["path"]).read_bytes()).hexdigest() == row["sha256"], row["path"])
    verify_record(json.loads((HERE/"certificate.json").read_text()))
    evidence = HERE/"evidence"
    receipt = json.loads((evidence/"execution.json").read_text())
    require(receipt["all_commands_passed"] and len(receipt["commands"]) == 8, "expected complete local review")
    require(receipt["scope"]["private_rational_replay_executed"], "missing local periodic replay")
    for key in ("numerical_infimum_comparisons", "original_interval_campaign_rerun", "proof_assistant_formalization", "external_peer_review"):
        require(receipt["scope"][key] is False, "overstated scope: " + key)
    for path, sha in receipt["input_sha256"].items():
        require(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == sha, path)
    for row in receipt["commands"]:
        require(row["returncode"] == 0, row["name"])
        require(hashlib.sha256((evidence/row["log"]).read_bytes()).hexdigest() == row["log_sha256"], row["log"])
    inherited = json.loads((evidence/"baseline_execution.json").read_text())
    require(inherited["all_commands_passed"] and len(inherited["commands"]) == 15, "inherited reproduction incomplete")
    require(not inherited["scope"]["D4_rational_replay_executed"], "public baseline scope altered")
    source = json.loads((evidence/"source_verification.json").read_text())
    require(source["status"] == "PASS" and len(source["checked_files"]) == 7, "source reconstruction incomplete")
    basin = json.loads((evidence/"d4_basin_replay.json").read_text())
    require(basin["status"] == "PASS" and len(basin["checks"]) == 5 and all(basin["checks"].values()), "periodic replay incomplete")
    symbols = json.loads((evidence/"identities.json").read_text())
    require(symbols["status"] == "PASS" and len(symbols["identities"]) == 10 and all(v == "0" for v in symbols["identities"].values()), "symbolic reconstruction incomplete")
    print(f"PASS: {len(manifest['files'])} new files, {len(manifest['inherited_pins'])} inherited pins, exact certificate and recorded local checks")
    print("Recorded private-source checks validated; this validation does not execute them.")


if __name__ == "__main__":
    main()
