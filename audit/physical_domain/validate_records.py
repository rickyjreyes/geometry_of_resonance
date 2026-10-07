"""Validate publication consistency; does not evaluate or promote physical claims."""
import argparse
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(name):
    return json.loads((HERE/name).read_bytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrap', action='store_true',
                        help='First authoring pass only, before the separate publication manifest is sealed.')
    args = parser.parse_args()
    checks = []

    def check(name, condition):
        assert condition, name
        checks.append(name)

    if not args.bootstrap:
        manifest = read('ARTIFACT_SHA256.json')
        for item in manifest['artifacts']:
            data = (ROOT/item['path']).read_bytes()
            assert len(data)==item['bytes'], item['path']
            assert hashlib.sha256(data).hexdigest()==item['sha256'], item['path']
        assert manifest['manifest_self_excluded'] is True

    expected = 'WCT_ACTIVE_DOMAIN_UNDERDETERMINED'
    decision, status, pred = (read(n) for n in ['decision.json','derivation_status.json','prediction.json'])
    graph, assumptions = (read(n) for n in ['dependency_graph.json','assumptions.json'])
    ledger, symbolic = (read(n) for n in ['SOURCE_LEDGER.json','symbolic_results.json'])
    check('decision_consistency', decision['decision']==status['decision']==expected)
    check('physical_classifications_unresolved',
          status['physical_active_domain']=='PHYSICAL_ACTIVE_DOMAIN_NOT_DERIVED'
          and status['winding_selection']=='NOT_IDENTIFIABLE'
          and status['spectral_mapping']=='SPECTRAL_MEASUREMENT_MAP_NOT_DERIVED'
          and not status['combined_prediction_derived'] and not status['scientific_closure_completed'])
    check('prediction_gate_and_nulls', not pred['quantitative_prediction_issued']
          and all(pred[k] is None for k in ['x','x0','W','L_W','n_W','mean_k_W','observable_mapping',
                                           'physical_system','allowed_physical_branches','uncertainty']))
    check('no_empirical_frequency_inputs', all(not d['empirical_frequency_inputs']
          for d in [decision,pred,assumptions,symbolic]))
    check('no_simulations_or_holdout', decision['new_PDE_simulations']==decision['new_collider_analyses']==0
          and not decision['ATLAS_final_holdout_accessed']
          and symbolic['new_PDE_simulations']==symbolic['collider_evaluations']==0)
    check('frozen_ATLAS_and_scores_preserved',
          decision['atlas_preservation']['frozen_commit']=='af6da1f5cfdc0a6a685c79309db342dd2338aae9'
          and not decision['atlas_preservation']['repo_written']
          and not decision['research_completion_scores_changed']
          and decision['unresolved_issues_closed']==[])
    check('exactly_one_next_task', len(decision['next_tasks'])==1
          and not decision['next_tasks'][0]['large_PDE_run_justified']
          and not decision['next_tasks'][0]['collider_run_justified'])
    check('symbolic_checks_complete', symbolic['checks_run']==symbolic['checks_passed']==30
          and len(symbolic['checks'])==30 and all(c['passed'] for c in symbolic['checks']))
    source_ids = {s['id'] for s in ledger['repository_sources']+ledger['paper_sources']}
    assumption_ids = {a['id'] for a in assumptions['assumptions']}
    node_ids = {n['id'] for n in graph['nodes']}
    check('source_counts_and_pins', len(source_ids)==56
          and len(ledger['repository_sources'])==48 and len(ledger['paper_sources'])==8
          and all(re.fullmatch(r'[0-9a-f]{40}',s['revision'])
                  and re.fullmatch(r'[0-9a-f]{40}',s['git_blob_sha1'])
                  and re.fullmatch(r'[0-9a-f]{64}',s['sha256'])
                  and s['exact_git_blob_verified'] for s in ledger['repository_sources']))
    check('manuscript_read_coverage_explicit',
          sum(p['full_extracted_text_read'] for p in ledger['paper_sources'])==7
          and next(p for p in ledger['paper_sources'] if p['id']=='P01')['full_extracted_text_read'] is False
          and all(p['original_binary_hash'] is None for p in ledger['paper_sources']))
    check('graph_integrity', not graph['complete'] and all(
          e['source'] in node_ids and e['target'] in node_ids
          and e['status'] in graph['allowed_edge_statuses']
          and set(e['assumptions'])<=assumption_ids and set(e['sources'])<=source_ids
          for e in graph['edges']))
    pairs = {(e['source'],e['target']) for e in graph['edges']}
    chain = graph['required_chain']
    check('required_dependency_chain_present', all(pair in pairs for pair in zip(chain,chain[1:])))
    check('assumption_source_integrity', all(set(a['sources'])<=source_ids for a in assumptions['assumptions'])
          and not assumptions['arbitrary_closure_axioms_adopted'])
    origins = read('prior_vs_new.json')
    check('prior_and_new_separated', bool(origins['prior']) and bool(origins['audit_results'])
          and all(set(r['sources'])<=source_ids for r in origins['prior']+origins['audit_results'])
          and all((ROOT/r['report']).is_file() for r in origins['audit_results']))
    baseline = read('BASELINE_BLOBS.json')
    base_paths = {b['path'] for b in baseline['blobs']}
    additions = sorted([*ROOT.glob('docs/*.md'),*HERE.glob('*')])
    addition_files = [p for p in additions if p.is_file()]
    # In a full checkout, pre-existing docs may be present; only the sealed audit paths are additions.
    if not args.bootstrap:
        audit_paths = {a['path'] for a in read('ARTIFACT_SHA256.json')['artifacts']}
        audit_paths.add('audit/physical_domain/ARTIFACT_SHA256.json')
    else:
        audit_paths = {str(p.relative_to(ROOT)) for p in addition_files}
    check('no_baseline_path_replaced', not (audit_paths & base_paths)
          and baseline['base_commit']=='a0048ef751a8876a9494f6de24b952ebc1db7213'
          and len(baseline['blobs'])==122)
    broken = []
    for relative in audit_paths:
        if not relative.endswith('.md'):
            continue
        path = ROOT/relative
        prose = re.sub(r'```[\s\S]*?```|`[^`]*`', '', path.read_text())
        for target in re.findall(r'\]\(([^)]+)\)',prose):
            if '://' in target or target.startswith('#'):
                continue
            if not (path.parent/target.split('#')[0]).exists():
                broken.append((relative,target))
    check('report_links_resolve', not broken)
    check('required_reports_present', all((ROOT/'docs'/name).is_file() for name in [
          'PHYSICAL_ACTIVE_DOMAIN_DERIVATION.md','WINDING_SELECTION_ANALYSIS.md',
          'SPECTRAL_OBSERVABLE_MAPPING.md','GEOMETRIC_SCALE_OPERATOR_AUDIT.md',
          'FOUNDATIONAL_CLOSURE_VERDICT.md','DYNAMICS_AND_SOURCE_INVENTORY.md']))
    result = dict(schema='WCT_PHYSICAL_DOMAIN_VALIDATION_V1',status='PASS',checks_passed=len(checks),
                  checks=checks,symbolic_checks_passed=30,
                  source_git_blob_hashes_verified_during_inventory=48,
                  scientific_prediction_derived=False,
                  scope='Publication consistency, provenance references, report links and reference symbolic output. Not physical closure or independent rerun of preserved solvers.')
    (HERE/'VALIDATION.json').write_bytes((json.dumps(result,indent=2,allow_nan=False)+'\n').encode())
    print(json.dumps(dict(status='PASS',record_checks=len(checks),symbolic_checks=30,
                          reference_hashes_verified=not args.bootstrap)))


if __name__=='__main__':
    main()
