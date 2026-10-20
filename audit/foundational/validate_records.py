"""Read-only integrity/scope gate; a metadata PASS is not a physical theorem."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def validate():
    manifest = json.loads((HERE / "manifest.json").read_text())
    for path, expected in manifest["sha256"].items():
        data = (ROOT / path).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected, path
    claims = json.loads((HERE / "claim_matrix.json").read_text())
    assert claims["primary_classification"] == "FOUNDATIONAL_OBSTRUCTION_CONFIRMED"
    ids = {x["id"]: x for x in claims["claims"]}
    assert len(ids) == len(claims["claims"])
    for claim in ids.values():
        assert all(claim[k] for k in ("model", "scope", "status", "evidence", "limitations"))
    assert ids["S004_UV"]["status"] == "PROVED_ANALYTICALLY"
    assert ids["D4_root"]["model"] == "CQ_HAMILTONIAN_2D_TORUS"
    assert ids["S004_to_CQ"]["status"] == "OPEN"
    assert ids["physical_mass"]["status"] == "MASS_BRIDGE_OBSTRUCTED"
    verification = json.loads((HERE / "verification.json").read_text())
    assert verification["status"] == "PASS"
    assert verification["PDE_propagations"] == 0
    assert not verification["physical_mass_predictions"]
    assert verification["script_sha256"] == hashlib.sha256(
        (ROOT / "derivations/foundational_closure.py").read_bytes()).hexdigest()
    replay = json.loads((HERE / "d4_basin_replay.json").read_text())
    assert replay["status"] == "PASS" and all(replay["checks"].values())
    assert replay["delta_physical"] == "21/5000000000000"
    assert replay["epsilon_physical"] == "21/5000000000"
    print(f"FOUNDATIONAL INTEGRITY/SCOPE: PASS ({len(manifest['sha256'])} files; {len(ids)} scoped claims)")


if __name__ == "__main__":
    validate()
