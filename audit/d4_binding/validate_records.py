#!/usr/bin/env python3
"""Fail closed on altered evidence, source pins or circular proof wiring."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent


def main():
    read=lambda name:json.loads((HERE/name).read_text())
    manifest=read('manifest.json');ledger=read('source_ledger.json')
    for row in manifest['files']+ledger['local_inherited_files']:
        assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
    v=read('verification.json');assert v['status']=='PASS' and v['primary']=='D4_STRICT_BINDING_PROVED'
    assert len(v['symbolic_checks'])==20 and all(r['passed'] for r in v['symbolic_checks'].values())
    m=F(float.fromhex(v['parameters']['mass_hex']));beta=F(float.fromhex(v['parameters']['beta_hex']));rho=1+3*beta/32
    assert F(v['exact']['rho'])==rho and F(v['exact']['D4_gate_9_minus_rho_m_squared'])==9-rho*m*m>0
    mb_lo,mb_hi=map(F,v['enclosures']['new_m_B']['exact'])
    assert m<mb_lo<=mb_hi and rho*mb_lo**2<=9<=rho*mb_hi**2
    assert F(v['enclosures']['left_slope_upper_enclosure']['exact'][1])<0
    assert F(v['enclosures']['endpoint_F_linear_coefficient']['exact'][0])>0
    assert all(v['whole_interval_gates'].values()) and not v['universal_all_mass_theorem']
    matrix=read('claim_matrix.json');assert matrix['primary']=='D4_STRICT_BINDING_PROVED'
    claims={row['id']:row for row in matrix['claims']};assert len(claims)==len(matrix['claims'])==27
    graph=read('proof_dependencies.json')
    assert graph['nodes']==[{'id':r['id'],'status':r['status'],'depends_on':r['depends_on']} for r in matrix['claims']]
    allowed={'PROVED','CERTIFIED','NUMERICALLY_SUPPORTED','CONDITIONAL','OPEN','REFUTED'}
    assert {r['status'] for r in claims.values()}==allowed
    for row in claims.values():
        assert row['scope'] and (ROOT/row['proof']).is_file()
        assert all(parent in claims for parent in row['depends_on'])
    def ancestors(node,chain=()):
        assert node not in chain,('cycle',chain,node)
        parents=set(claims[node]['depends_on'])
        for parent in list(parents):parents|=ancestors(parent,chain+(node,))
        return parents
    for node in claims:ancestors(node)
    forbidden={'S2','C1','C2','C3','O1','O3'}
    for node in ['V1','V2','V3','E1','E2','E3','F']:
        assert not ancestors(node)&forbidden,(node,ancestors(node)&forbidden)
    for node in ['V3','S2','E3','F','C1','C2','C3','O1','O3']:
        assert claims[node]['status']=='PROVED'
        assert all(claims[a]['status'] in {'PROVED','CERTIFIED'} for a in ancestors(node))
    exe=read('execution.json');assert exe['mode']=='full-local' and exe['all_commands_passed'] and len(exe['commands'])==17
    for key in ['D4_private_inputs_hash_checked','D4_rational_replay_executed','private_source_conventions_checked','fresh_authenticated_source_extraction_receipt_included']:assert exe['scope'][key]
    for key in ['recorded_evidence_validated','original_D4_interval_campaign_rerun','PDE_propagation','stationary_root_search','external_mathematical_review','physical_spectrum']:assert not exe['scope'][key]
    for path,digest in exe['implementation_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    for row in exe['commands']:
        assert row['returncode']==0 and hashlib.sha256((HERE/row['log']).read_bytes()).hexdigest()==row['log_sha256']
    origin=read('source_extraction.json');sources=read('source_verification.json')
    assert hashlib.sha256((HERE/'source_extraction.json').read_bytes()).hexdigest()==exe['source_extraction_sha256']
    pins={r['path']:r['sha256'] for r in ledger['d4_replay_inputs']}
    allpins={r['path']:r['sha256'] for r in ledger['d4_replay_inputs']+ledger['reviewed_supporting_sources']}
    assert {r['path']:r['sha256'] for r in origin['files']}==pins
    assert {r['path']:r['sha256'] for r in origin['files']+origin['supporting_review_files']}==allpins
    assert sources['status']=='PASS' and {r['path']:r['sha256'] for r in sources['checked_files']}==allpins
    d4=read('d4_basin_replay.json');assert d4['status']=='PASS' and len(d4['checks'])==5 and all(d4['checks'].values())
    assert all(pins[p]==digest for p,digest in d4['sources'].items())
    diag=read('diagnostics.json');assert diag['status']=='PASS'
    assert len(diag['gaussian_dilation_probes'])==240 and len(diag['two_profile_trials'])==59 and len(diag['endpoint_certified_scalar_probes'])==20
    assert all(not r['certifies_infimum_comparison'] for r in diag['two_profile_trials'])
    assert all(F(r['F']['exact'][0])>0 for r in diag['endpoint_certified_scalar_probes'])
    print(f"PASS: {len(manifest['files'])} new artifact hashes, {len(ledger['local_inherited_files'])} inherited pins, {len(claims)} scoped claims, acyclic dependencies, 17 successful execution groups")
    print('Recorded full-local D4 evidence validated; this validator does not execute the private-source replay.')


if __name__=='__main__':main()
