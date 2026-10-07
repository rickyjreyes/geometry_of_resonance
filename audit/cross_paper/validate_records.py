"""Check cross-paper audit structure, provenance and frozen-scope consistency.

This does not prove physical assumptions or nonlinear existence. Run alongside
verify_derivations.py for the independently checked algebra.
"""
import hashlib
import json
import re
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
DOC=REPO/'docs/cross_paper'
checks=[]
def check(name,condition):
    assert condition,name
    checks.append(name)
def read(name):
    return json.loads((HERE/name).read_text())
required_docs=['CROSS_PAPER_DERIVATION_LEDGER.md','CORRECTED_ACTION_TO_PFF_REDUCTION.md',
    'CORRECTED_ACTION_TO_EFT_KERNEL.md','QUADRATIC_ACTION_AND_DISPERSION.md',
    'FINITE_SCALE_SELECTION.md','LOCALIZATION_AND_DOMAIN_SELECTION.md',
    'RADIAL_OPERATOR_AND_SHELL_SPECTRUM.md','SECTOR_ENERGY_AND_WINDING.md',
    'ACTIVE_LOG_DOMAIN_MAP.md','OBSERVABLE_FORWARD_MAP.md','DECISION.md']
check('all_eleven_requested_reports',all((DOC/n).is_file() for n in required_docs))
g=read('DERIVATION_GRAPH.json');p=read('FREE_PARAMETER_LEDGER.json')
pred=read('candidate_predictions.json');prior=read('prior_vs_new_derivation.json')
src=read('SOURCE_LEDGER.json');algebra=read('symbolic_verification.json')
check('decision_agreement',g['decision']==pred['decision']==prior['new_decision']=='FINITE_SCALE_DERIVED_BUT_CLOSURE_INCOMPLETE')
check('new_isolated_branch',g['branch']=='agent/wct-cross-paper-derivation')
check('baseline_agreement',g['baseline_commit']==src['baseline_commit']==prior['baseline_commit']=='a84a4f0745d8e440961797ab8da809ae8598a836')
check('all_five_routes',sorted(x['id'] for x in g['routes'])==list('ABCDE'))
nodes={n['id'] for n in g['nodes']};ids={e['id'] for e in g['edges']}
check('unique_graph_records',len(nodes)==len(g['nodes']) and len(ids)==len(g['edges']))
check('graph_endpoints_exist',all(e['from_node'] in nodes and e['to_node'] in nodes for e in g['edges']))
check('route_edges_exist',all(set(r['edges'])<=ids for r in g['routes']))
fields=['source_equations','source_papers','assumptions','units','field_definitions',
    'transformation_or_reduction','asymptotic_parameter','error_or_order',
    'free_parameters','physical_interpretation','status','falsifier','prior_or_new']
check('all_thirteen_research_fields',all(all(f in e for f in fields) for e in g['edges']))
check('nonempty_bridge_explanations',all(all(bool(e[f]) for f in fields if f!='free_parameters') for e in g['edges']))
check('allowed_arrow_statuses',all(e['status'] in g['arrow_status_vocabulary'] for e in g['edges']))
check('allowed_gap_classifications',all(e['gap_classification'] in g['gap_vocabulary'] for e in g['edges']))
parids={x['id'] for x in p['parameters']}
check('parameter_references_resolve',all(set(e['free_parameters'])<=parids for e in g['edges']))
check('prediction_parameter_references_resolve',all(set(c['free_parameters'])<=parids for c in pred['candidates']))
sourceids={x['id'] for x in src['paper_sources']+src['repository_sources']}
check('source_references_resolve',all(set(e['source_papers'])<=sourceids for e in g['edges']))
check('all_bridge_reports_exist',all((DOC/e['report']).is_file() for e in g['edges']))
check('prior_new_covers_every_bridge',{e['bridge_id'] for e in prior['results']}==ids)
check('old_labels_preserved',set(prior['preserved_labels'])=={'PHYSICAL_ACTIVE_DOMAIN_NOT_DERIVED','NOT_IDENTIFIABLE','SPECTRAL_MEASUREMENT_MAP_NOT_DERIVED','WCT_ACTIVE_DOMAIN_UNDERDETERMINED'})
check('unselected_predictions_are_null',all(pred[k] is None for k in ['selected_winding','active_log_domain','observable_log_frequency','DSI_lambda','physical_236_scale']))
check('no_parameter_free_numeric_prediction',pred['parameter_free_numeric_predictions']==[] and not p['independent_numeric_prediction'])
check('no_empirical_inputs',pred['empirical_frequency_inputs']==[] and all(c['empirical_inputs']==[] for c in pred['candidates']) and all(not x['empirically_fitted'] for x in p['parameters']))
check('holdout_and_collider_frozen',not pred['holdout_opened'] and pred['collider_searches']==0 and src['empirical_scope']['collider_searches']==0)
check('no_PDE_runs',pred['expensive_PDE_runs']==0 and algebra['PDE_runs']==0)
check('all_symbolic_checks_reported',algebra['checks_passed']==len(algebra['checks'])==35)
check('paper_coverage_consistent',all(p['unique_lines_read']==p['total_extracted_lines'] and p['full_extracted_text_read'] for p in src['paper_sources']))
check('source_counts_consistent',src['counts']==dict(repository_sources=len(src['repository_sources']),paper_sources=len(src['paper_sources']),full_extracted_papers=sum(p['full_extracted_text_read'] for p in src['paper_sources'])))
check('source_text_hashes_present',all(re.fullmatch('[0-9a-f]{64}',w['sha256_of_content_array']) for p in src['paper_sources'] for w in p['read_windows']))
decision=(DOC/'DECISION.md').read_text()
sections=re.findall(r'^## ([A-N])\. ',decision,re.M)
check('final_report_A_through_N',sections==list('ABCDEFGHIJKLMN'))
next_section=decision.split('## N. Exactly one highest-value next task\n',1)[1].split('\n## ',1)[0]
check('exactly_one_next_task',next_section.count('**')==2 and not re.search(r'^\s*[-*]\s|^\s*\d+\.',next_section,re.M))
manifest=read('BASELINE_MANIFEST.json')
check('baseline_manifest_144_files',len(manifest['files'])==144 and manifest['commit']==g['baseline_commit'])
present=[]
for f in manifest['files']:
    path=REPO/f['path']
    if path.is_file():
        raw=path.read_bytes();sha=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        assert sha==f['sha'],'Modified local baseline: '+f['path']
        present.append(f['path'])
check('local_baseline_snapshot_unchanged',len(present)>0)
baseline_paths={f['path'] for f in manifest['files']}
newpaths={str(f.relative_to(REPO)) for folder in [HERE,DOC] for f in folder.iterdir() if f.is_file()}
check('new_paths_do_not_replace_baseline',not newpaths.intersection(baseline_paths))
result=dict(schema='WCT_CROSS_PAPER_RECORD_VERIFICATION_V1',checks_passed=len(checks),checks=checks,
    local_baseline_files_checked=len(present),remote_baseline_files_to_preserve=144,
    limitation='Remote preservation is additionally verified against the complete published Git tree. Structural checks do not establish nonlinear existence.',PDE_runs=0,empirical_frequency_inputs=[])
(HERE/'record_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
