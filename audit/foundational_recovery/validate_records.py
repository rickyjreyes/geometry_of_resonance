#!/usr/bin/env python3
"""Verify additive Phase III artifact integrity and explicit scope boundaries."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def main():
    manifest=json.loads((HERE/'manifest.json').read_text())
    for row in manifest['files']:
        path=ROOT/row['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('artifact changed: '+row['path'])
    matrix=json.loads((HERE/'claim_matrix.json').read_text())
    assert matrix['primary']=='INDEPENDENT_EFFECTIVE_MODEL_SUPPORTED'
    assert matrix['route_a']=='FOUNDATIONAL_RECOVERY_NOT_ESTABLISHED'
    rows={c['id']:c for c in matrix['claims']}
    assert len(rows)==len(matrix['claims'])
    for key in ['WHOLE_SPACE_BINDING','NONLINEAR_PARENT']:
        assert rows[key]['classification']=='OPEN'
    assert rows['B4']['classification']=='CONDITIONAL'
    assert rows['PHYSICAL_MASS']['classification']=='OBSTRUCTED'
    assert not rows['D4']['new']
    for row in rows.values():
        assert row['scope'] and (ROOT/row['proof']).is_file()
    for name in ['verification.json','pr33_verification.json']:
        result=json.loads((HERE/name).read_text())
        if name=='verification.json':
            assert result['status']=='PASS'
            assert result['symbolic_check_count']==38
            assert not result['gaussian_certificate']['whole_space_minimizer_proved']
    executions=json.loads((HERE/'execution.json').read_text())
    assert executions['all_commands_passed']
    assert all(r['returncode']==0 for r in executions['commands'])
    assert all(matrix[k] for k in ['no_empirical_mass_fitting','no_pde_campaign','no_new_interval_root_campaign','no_maturity_rerating'])
    print(f"PASS: {len(manifest['files'])} artifact hashes; {len(rows)} scoped claims; execution receipts")


if __name__=='__main__':main()
