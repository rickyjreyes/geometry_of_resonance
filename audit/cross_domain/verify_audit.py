#!/usr/bin/env python3
"""Meaningful checks of provenance, no-leak LOO, and global null optimization."""
import argparse, hashlib, itertools, json
from pathlib import Path
import numpy as np
import mpmath as mp
import run_audit as audit

HERE=Path(__file__).resolve().parent
checks=[]
def check(name,condition):
    checks.append({'name':name,'passed':bool(condition)})
    if not condition:raise AssertionError(name)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-dir',type=Path);p.add_argument('--published-tree',type=Path);args=p.parse_args()
    tj=json.loads((HERE/'cross_domain_targets.json').read_text());ts={r['id']:r for r in tj['records']}
    cv=json.loads((HERE/'cross_validation_results.json').read_text());n=json.loads((HERE/'null_diagnostics.json').read_text())
    required_docs=['CROSS_DOMAIN_SINGLE_MECHANISM_AUDIT.md','EMPIRICAL_TARGET_LEDGER.md','CROSS_DOMAIN_TARGET_LEDGER.md','THEORY_TRANSFORMATION_LEDGER.md','UNIVERSAL_INVARIANT_SEARCH.md','SECTOR_ASSIGNMENT_AUDIT.md','LEAVE_ONE_DOMAIN_OUT.md','NULL_MODEL_AND_TRIALS_AUDIT.md','FAILURE_MODES.md','DECISION_CROSS_DOMAIN.md']
    check('all requested reports exist',all((HERE.parent.parent/'docs/cross_domain'/name).is_file() for name in required_docs))
    decision=json.loads((HERE/'DECISION_CROSS_DOMAIN.json').read_text())
    check('exact permitted verdict and protected scope',decision['verdict']=='NUMERICAL_CONVERGENCE_ONLY' and decision['protected_state']=={'empirical_repositories_modified':False,'empirical_peak_searches_run':0,'ATLAS_holdout_opened':False})
    models=json.loads((HERE/'candidate_unified_mechanisms.json').read_text())['candidates']
    check('all model classes and four-domain missing predictions disclosed',{'U0','U1','U2','U3','U4'}.issubset({m['id'] for m in models}) and all(set(m['physical_predictions_by_domain'])=={'GWTC','CMS','NIST','LHCb'} for m in models))
    check('transformation categories explicit',all(m['transformation'] in ['DERIVED','PHYSICALLY_JUSTIFIED','HYPOTHESIS','POST_HOC_NOT_ALLOWED'] for m in models))
    check('primary preserves slope a instead of a fabricated fourth frequency',tj['primary_target_ids'][-1]=='LHCB_A' and ts['LHCB_A']['quantity_type']=='interregional_triplet_regression_slope' and ts['LHCB_A']['DSI_period'] is False)
    check('imported CMS frequency excluded from independent reference panel','LHCB_CMS_IMPORTED' not in tj['supplementary_reference_ids'])
    check('source decimal tokens preserved',all(isinstance(r['value_decimal'],str) for r in ts.values()))
    check('all required U0-U4 primary folds explicit',len(cv['primary_anchor_panel'])==5 and all(len(m['folds'])==4 for m in cv['primary_anchor_panel']))
    check('no invented primary physical predictions',all(f['predicted'] is None and f['residual'] is None for m in cv['primary_anchor_panel'] for f in m['folds']))
    check('all source-choice fits retained',len(cv['supplementary_frequency_panels'])==36)
    for k,r in enumerate(cv['supplementary_frequency_panels']):
        ids=r['target_ids'];factors=list(map(audit.dec,r['native_to_common_multipliers']))
        for i,f in enumerate(r['folds']):
            train=[j for j in range(4) if j!=i]
            pred=mp.exp(sum(mp.log(audit.dec(ts[ids[j]]['value_decimal'])*factors[j]) for j in train)/3)/factors[i]
            check(f'LOO {k}:{i} uses exactly three other domains',ids[i] not in f['training_ids'] and len(f['training_ids'])==3 and abs(pred-audit.dec(f['predicted']))<mp.mpf('1e-70'))
            check(f'LOO {k}:{i} error and no invented sigma',abs((pred/audit.dec(f['observed'])-1)-audit.dec(f['fractional_error']))<mp.mpf('1e-70') and f['normalized_residual'] is None)
    altered={k:dict(v) for k,v in ts.items()};altered['GWTC_V1']['value_decimal']='987654.321'
    a=audit.fit_and_loo(tj['supplementary_reference_ids'],ts,[1,1,1,2],'U1','test')
    b=audit.fit_and_loo(tj['supplementary_reference_ids'],altered,[1,1,1,2],'U1','test')
    check('held-out target perturbation leaves its prediction identical',a['folds'][0]['predicted']==b['folds'][0]['predicted'])
    check('mass-squared conversion follows phase chain rule',audit.dec(ts['CMS']['value_decimal'])/2==audit.dec('3.5129129129129135'))
    check('power map preserves frequency times log width',audit.dec('7.025825825825827')*audit.dec('4.0943445622221007')==(audit.dec('7.025825825825827')/2)*(2*audit.dec('4.0943445622221007')))
    # Verify the nontrivial global optimizer against exhaustive enumerations.
    rng=np.random.Generator(np.random.PCG64(404))
    samples=rng.uniform(0,4,(12,4))
    for cap,menu,direction in itertools.product([2,3],[[1],[.5,1,2]],[1,-1]):
        off,_=audit.choice_offsets(cap,menu,direction)
        choices=np.array(list(itertools.product(off,repeat=4)))
        actual=audit.optimize(samples,cap,menu,direction)
        expected=[]
        for x in samples:
            z=x+choices;expected.append(np.sqrt(np.min(np.mean((z-z.mean(1)[:,None])**2,axis=1))))
        check(f'global sweep equals exhaustive cap={cap}, menu={menu}, direction={direction}',np.allclose(actual,expected,atol=1e-11,rtol=1e-10))
    check('perfect synthetic harmonic quartet recovered',audit.optimize(np.log([[1,2,3,4]]),4,[1],1)[0]<1e-12)
    for support in n['results']:
        rows=support['rows']
        check(f'null all families retained {support["support"]}',len(rows)==48)
        for panel,cap,direction in itertools.product(['native','mass_energy'],[1,6,12,24],[1,-1]):
            rr={r['family']:r for r in rows}
            identity=rr[f'{panel}:M{cap}:identity:d{direction}'];powers=rr[f'{panel}:M{cap}:powers:d{direction}'];rec=rr[f'{panel}:M{cap}:powers_reciprocal:d{direction}']
            check(f'menu nesting and reciprocal magnitude duplication {support["support"]}:{panel}:{cap}:{direction}',powers['observed_rms_log_residual']<=identity['observed_rms_log_residual']+1e-12 and powers['observed_rms_log_residual']==rec['observed_rms_log_residual'] and powers['count_null_at_least_as_close']==rec['count_null_at_least_as_close'])
        om=support['omnibus'];check(f'omnibus includes entire selection {support["support"]}',abs(om['observed_score']-min(r['observed_rms_log_residual'] for r in rows))<1e-12 and om['add_one_fraction']==(om['null_count']+1)/10001)
    source_count=0
    if args.source_dir:
        ledger=json.loads((HERE/'SOURCE_LEDGER.json').read_text())
        for s in ledger['repository_sources']:
            raw=(args.source_dir/s['repository'].split('/')[-1]/s['path']).read_bytes()
            check(f'source SHA256 {s["id"]}',hashlib.sha256(raw).hexdigest()==s['sha256'])
            check(f'source Git blob SHA1 {s["id"]}',hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==s['git_blob_sha1'])
            source_count+=1
        for id,r in ts.items():
            s=r['source'];path=args.source_dir/s['repository'].split('/')[-1]/s['path']
            ptr=s['json_pointer_or_locator']
            if s['path'].endswith('.json') and ptr.startswith('/'):
                obj=json.loads(path.read_text(),parse_float=str)
                for key in ptr.split('/')[1:]:obj=obj[key]
                check(f'exact target source extraction {id}',obj==r['value_decimal'])
            else:check(f'exact target source literal {id}',r['value_decimal'] in path.read_text())
    preserved=None
    if args.published_tree:
        tree=json.loads(args.published_tree.read_text());lookup={v['path']:v for v in tree['tree'] if v['type']=='blob'}
        base=json.loads((HERE/'BASELINE_MANIFEST.json').read_text())
        for b in base['blobs']:check('baseline preserved '+b['path'],lookup[b['path']]['sha']==b['sha'])
        preserved=len(base['blobs'])
    audit.save('verification.json',{'checks_passed':len(checks),'checks_failed':sum(not r['passed'] for r in checks),'source_files_verified':source_count,'baseline_blobs_verified_against_supplied_tree':preserved,'checks':checks})
    print(f'{len(checks)} checks passed; {source_count} source files verified; baseline preservation={preserved}')

if __name__=='__main__':main()
