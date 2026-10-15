"""Fail closed on accidental D4 handoff status promotion; stdlib only."""
import hashlib
import json
from pathlib import Path

PATH = Path(__file__).with_name("claims.json")
def validate(data):
    assert data["schema"] == "wct-d4-handoff-v1"
    assert data["model"]["dimension"] == 2
    assert "NOT four spatial dimensions" in data["model"]["symmetry"]
    assert len(data["archive"]["sha256"]) == 64
    assert data["archive"]["sha256"] == "db9cfbbbb4a4b9d696bf4d0af6c2eb125e8795acfb8505491628eb0238f47b64"
    assert data["archive"]["member_count"] == 200
    assert data["report"]["commit"] == "26296d5c40857c37adca6a1aae47930b3598c4ff"
    by_id = {v["id"]: v for v in data["claims"]}
    assert len(by_id) == len(data["claims"]), "duplicate claim ID"
    expected = {
        "production_provenance":"PASS", "durable_release":"PASS",
        "historical_aggregate":"FAIL", "pade_finite_protocol":"PASS",
        "optimized_integrator_equivalence":"OPEN", "refined_envelope":"OPEN",
        "leading_gain_certificate":"OPEN", "continuum_application":"CONDITIONAL",
        "all_time_orbital_application":"CONDITIONAL", "independent_d4_fem":"PENDING",
        "formal_d4_application":"PENDING", "canonical_action_equivalence":"NOT_ESTABLISHED",
        "lorentzian_ghost_freedom":"NOT_ESTABLISHED",
    }
    assert set(by_id) == set(expected), "missing or unexpected status rows require manual review"
    for key, value in expected.items():
        assert by_id[key]["status"] == value, f"unaudited promotion or alteration: {key}"
        assert by_id[key]["scope"] and by_id[key]["source_issue"]
    assert len(data["papers"]) == 2
    return hashlib.sha256(PATH.read_bytes()).hexdigest()

if __name__ == "__main__":
    digest = validate(json.loads(PATH.read_text(encoding="utf-8")))
    print("D4 HANDOFF SCOPED STATUS VALIDATION: PASS", digest)
