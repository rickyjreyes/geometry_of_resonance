#!/usr/bin/env python3
"""Reproduce the new certificate and the preserved mathematical baseline."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--d4-source-root", type=Path,
                        help="Optional authorized pinned wct-pde sources; never downloaded by this runner")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    py = sys.executable
    commands = [
        ("fresh-rational-certificate", [py, str(HERE/"certify_gap.py"), "--output", str(out/"certificate.json")]),
        ("committed-certificate-check", [py, str(HERE/"certify_gap.py"), "--check", str(HERE/"certificate.json")]),
        ("optimized-python-check", [py, "-O", str(HERE/"certify_gap.py"), "--check", str(HERE/"certificate.json")]),
        ("independent-exact-tests", [py, "-m", "unittest", "discover", "-v", "-s", str(HERE), "-p", "test_*.py"]),
        ("independent-symbolic-identities", [py, str(HERE/"check_identities.py"), "--output", str(out/"identities.json")]),
        ("inherited-phase-v-public", [py, "audit/d4_binding/reproduce.py", "--public-ci", "--output-dir", str(out/"baseline")]),
    ]
    if args.d4_source_root:
        source = args.d4_source_root.resolve()
        commands.extend([
            ("fresh-pinned-source-conventions", [py, "audit/d4_binding/verify_sources.py", "--source-root", str(source), "--output", str(out/"source_verification.json")]),
            ("fresh-periodic-rational-replay", [py, str(source/"tools/closure_d4_basin_rational_replay.py"), "--output", str(out/"d4_basin_replay.json")]),
        ])
    tracked_inputs = sorted(HERE.glob("*.py")) + [HERE/"certificate.json", ROOT/"docs/d4_frozen_review/PROOF.md"]
    receipt = {
        "schema": "d4-frozen-review-execution-v1",
        "reviewed_base": "7264a760bee05fea89a1e42249b49f39bd078b78",
        "mode": "with-private-rational-replay" if args.d4_source_root else "public",
        "python": platform.python_version(),
        "dependencies": {name: importlib.metadata.version(name) for name in ("sympy", "mpmath", "numpy", "scipy")},
        "input_sha256": {str(p.relative_to(ROOT)): digest(p) for p in tracked_inputs},
        "commands": [], "all_commands_passed": False,
        "scope": {"numerical_infimum_comparisons": False, "original_interval_campaign_rerun": False,
                  "proof_assistant_formalization": False, "external_peer_review": False,
                  "private_rational_replay_executed": False},
    }
    for name, command in commands:
        started = datetime.now(timezone.utc).isoformat()
        tick = time.monotonic()
        result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log = out/(name+".log")
        log.write_text(result.stdout)
        receipt["commands"].append({"name": name, "command": command, "started_at_utc": started,
                                    "seconds": round(time.monotonic()-tick, 3), "returncode": result.returncode,
                                    "log": log.name, "log_sha256": digest(log)})
        if name == "fresh-periodic-rational-replay" and result.returncode == 0:
            receipt["scope"]["private_rational_replay_executed"] = True
        (out/"execution.json").write_text(json.dumps(receipt, indent=2)+"\n")
        print(name + (": PASS" if result.returncode == 0 else ": FAIL"), flush=True)
        if result.returncode:
            print(result.stdout)
            raise SystemExit(result.returncode)
    receipt["all_commands_passed"] = True
    (out/"execution.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(f"PASS: {len(commands)} review groups, including the inherited 15-group public reproduction")


if __name__ == "__main__":
    main()
