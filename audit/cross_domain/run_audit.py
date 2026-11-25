#!/usr/bin/env python3
"""Summary-only retrospective audit. Never loads or searches empirical event data.

Run: python audit/cross_domain/run_audit.py
Requires numpy and mpmath; reads exact decimal tokens in cross_domain_targets.json.
"""
import argparse
import itertools
import json
from pathlib import Path
import platform

import mpmath as mp
import numpy as np

HERE=Path(__file__).resolve().parent
mp.mp.dps=80
def dec(x): return mp.mpf(str(x))
def string(x): return mp.nstr(x,80)
def save(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def residual(pred,obs):
    p,o=dec(pred),dec(obs)
    return {'predicted':string(p),'observed':string(o),'signed_error':string(p-o),
            'absolute_error':string(abs(p-o)),'fractional_error':string(p/o-1),
            'normalized_residual':None,'normalized_residual_reason':'No calibrated sigma for this target frequency; no error bar invented.'}
def fit_and_loo(ids,targets,multipliers,model,panel):
    vals=[dec(targets[x]['value_decimal']) for x in ids]
    factors=list(map(dec,multipliers))
    xs=[mp.log(v*f) for v,f in zip(vals,factors)]
    mu=sum(xs)/4;theta=mp.exp(mu)
    folds=[]
    for i,id in enumerate(ids):
        train=[j for j in range(4) if j!=i]
        theta_train=mp.exp(sum(xs[j] for j in train)/3)
        pred=theta_train/factors[i]
        folds.append({'held_out':id,'held_out_domain':targets[id]['domain'],
                      'training_ids':[ids[j] for j in train],
                      'fit_uses_held_out':False,'shared_scale_frozen':string(theta_train),
                      'held_out_multiplier':string(factors[i]),**residual(pred,vals[i])})
    return {'panel':panel,'model':model,'target_ids':ids,'native_to_common_multipliers':list(map(str,multipliers)),
            'shared_scale_full_fit':string(theta),
            'full_fit_rms_log_residual':string(mp.sqrt(sum((x-mu)**2 for x in xs)/4)),
            'full_fit_predictions':[dict(target_id=id,**residual(theta/f,v)) for id,v,f in zip(ids,vals,factors)],
            'folds':folds,'status':'RETROSPECTIVE_DESCRIPTIVE_STRESS_TEST_NOT_ACTION_PREDICTION'}

def choice_offsets(cap,menu,direction):
    """Unique possible log(theta) offsets; labels retain original integer/exponent."""
    choices={}
    # Fraction arithmetic removes exact n*p degeneracies before taking logarithms.
    from fractions import Fraction
    for n in range(1,cap+1):
        for p in menu:
            factor=abs(Fraction(str(p)))*(Fraction(n)**direction)
            choices.setdefault(factor,{'n':n,'exponent':p,'orientation':1 if p>0 else -1})
    factors=sorted(choices,reverse=True)
    return -np.log(np.array([float(v) for v in factors])),[choices[v] for v in factors]

def optimize(log_values,cap,menu,direction,return_labels=False):
    """Global fit over ALL per-domain choices and one continuous log scale.

    For fixed scale each domain chooses the closest candidate log(theta).
    Sweep the union of their Voronoi boundaries. Every globally optimal
    assignment occurs in this sweep; each swept assignment's best scale is its
    mean. O(d*m log(d*m)) instead of enumerating m**d assignments.
    """
    x=np.atleast_2d(np.asarray(log_values,float));d=x.shape[1]
    off,labels=choice_offsets(cap,menu,direction)
    if len(off)==1:
        mu=x.mean(1)+off[0];score=np.sqrt(np.mean((x-x.mean(1)[:,None])**2,axis=1))
        lab=[[labels[0] for _ in range(d)] for _ in range(len(x))]
        return (score,mu,lab) if return_labels else score
    scores=[];mus=[];chosen=[]
    for start in range(0,len(x),1000):
        z=x[start:start+1000];N=len(z)
        boundaries=(z[:,:,None]+(off[:-1]+off[1:])/2).reshape(N,-1)
        order=np.argsort(boundaries,axis=1,kind='stable')
        ds=np.broadcast_to(np.diff(off),(N,d,len(off)-1)).reshape(N,-1)
        dss=(2*z[:,:,None]*np.diff(off)+np.diff(off**2)).reshape(N,-1)
        ds=np.take_along_axis(ds,order,axis=1)
        dss=np.take_along_axis(dss,order,axis=1)
        initial=z+off[0]
        sums=np.concatenate([initial.sum(1)[:,None],initial.sum(1)[:,None]+np.cumsum(ds,axis=1)],axis=1)
        squares=np.concatenate([(initial**2).sum(1)[:,None],(initial**2).sum(1)[:,None]+np.cumsum(dss,axis=1)],axis=1)
        variances=np.maximum(0,squares/d-(sums/d)**2)
        best=np.argmin(variances,axis=1)
        mu=sums[np.arange(N),best]/d
        # Re-evaluate directly at nearest assignments to avoid cancellation.
        index=np.argmin(abs(z[:,:,None]+off-mu[:,None,None]),axis=2)
        selected=z+off[index]
        mu=selected.mean(1)
        score=np.sqrt(((selected-mu[:,None])**2).mean(1))
        scores.extend(score);mus.extend(mu)
        if return_labels:chosen.extend([[labels[j] for j in row] for row in index])
    return (np.array(scores),np.array(mus),chosen) if return_labels else np.array(scores)

def derived(targets):
    out={'precision_digits':80,'DSI_periods':{},'source_window_cycles':{},'published_geometric_analogies':[],
         'warning':'DSI conversion re-expresses a sinusoid period; it does not derive physical discrete scale invariance.'}
    for id,r in targets.items():
        if r['quantity_type']=='log_angular_frequency':
            nu=dec(r['value_decimal']);out['DSI_periods'][id]=string(mp.exp(2*mp.pi/nu))
            if r.get('source_window',{}).get('width'):
                out['source_window_cycles'][id]=string(nu*dec(r['source_window']['width'])/(2*mp.pi))
    cms=targets['CMS'];nu=dec(cms['value_decimal'])
    L=mp.log(60)
    masks=sum(mp.log(dec(b)/dec(a)) for a,b in cms['mask_intervals'])
    out['CMS_analysis_crop_diagnostics']={'full_log_width':string(L),'nominal_masked_log_width':string(L-masks),
        'full_crop_cycles':string(nu*L/(2*mp.pi)),'nominal_masked_cycles':string(nu*(L-masks)/(2*mp.pi)),
        'physical_closure_domain':False,'note':'Continuous nominal interval measure, not the effective 287-bin weighted fitting support.'}
    for id,lam in [('CMS',mp.sqrt(6)),('NIST_FE160',mp.sqrt(dec('1.5')))]:
        pred=2*mp.pi/mp.log(lam)
        out['published_geometric_analogies'].append({'id':id,'category':'NUMERICAL_ANALOGY','assigned_lambda':string(lam),
            'physical_assignment_derived':False,**residual(pred,targets[id]['value_decimal']),
            'lambda_log_residual':string(mp.log(mp.exp(2*mp.pi/dec(targets[id]['value_decimal']))/lam))})
    ap=mp.sqrt(dec('1.5'));ao=dec(targets['LHCB_A']['value_decimal'])
    out['published_geometric_analogies'].append({'id':'LHCB_A','category':'NUMERICAL_ANALOGY','physical_assignment_derived':False,
        **residual(ap,ao),'bootstrap_sd_units_approx':string((ap-ao)/dec('.36')),
        'gaussian_z_valid':False,'note':'Approximate paper bootstrap scale .36 is conditional and heavy-tailed; not a significance.'})
    out['unassigned_GWTC_geometric_prediction']=None
    out['CMS_to_NIST_period_ratio']=string(mp.exp(2*mp.pi/dec(targets['CMS']['value_decimal']))/mp.exp(2*mp.pi/dec(targets['NIST_FE160']['value_decimal'])))
    out['period_ratio_two_partial_hypothesis']={
        'status':'PUBLISHED_NUMERICAL_ANALOGY_WITHOUT_PHYSICAL_ASSIGNMENT',
        'equation':'lambda_CMS=2 lambda_NIST; this is not a squared-coordinate map',
        'CMS_from_NIST':residual(2*mp.pi/(2*mp.pi/dec(targets['NIST_FE160']['value_decimal'])+mp.log(2)),targets['CMS']['value_decimal']),
        'NIST_from_CMS':residual(2*mp.pi/(2*mp.pi/dec(targets['CMS']['value_decimal'])-mp.log(2)),targets['NIST_FE160']['value_decimal']),
        'GWTC_prediction':None,'LHCb_a_prediction':None,
        'four_domain_LOO_eligible':False}
    out['Co160_vs_Fe_fractional_frequency_difference']=string(dec(targets['NIST_CO160']['value_decimal'])/dec(targets['NIST_FE160']['value_decimal'])-1)
    out['fixed_geometric_scale_stress_tests']=[]
    ids=['GWTC_V1','CMS','NIST_FE160','LHCB_K1']
    for name,lam in [('mass_energy_dilation_sqrt6_HYPOTHESIS',mp.sqrt(6)),('root_mass_sqrt6_implies_mass_dilation6_HYPOTHESIS',dec(6))]:
        scale=2*mp.pi/mp.log(lam)
        out['fixed_geometric_scale_stress_tests'].append({'hypothesis':name,'shared_parameters_fitted':0,
            'predictions':[{'id':id,**residual(scale/(2 if id=='LHCB_K1' else 1),targets[id]['value_decimal'])} for id in ids]})
    return out

def nulls(targets):
    ids=['GWTC_V1','CMS','NIST_FE160','LHCB_K1']
    observed=np.log([float(targets[id]['value_decimal']) for id in ids])
    rng=np.random.Generator(np.random.PCG64(20261007))
    supports=[[1,100],[5,50]];N=10000;results=[]
    menus={'identity':[1],'powers':[.5,1,2],'powers_reciprocal':[.5,1,2,-1]}
    for low,high in supports:
        draws=rng.uniform(np.log(low),np.log(high),(N,4))
        omnibus_obs=np.inf;omnibus_null=np.full(N,np.inf);omnibus_name=None
        cache={};rows=[]
        for panel,factor in [('native',[1,1,1,1]),('mass_energy',[1,1,1,2])]:
            shift=np.log(factor);ox=observed+shift;nx=draws+shift
            for cap,menu_name,direction in itertools.product([1,6,12,24],menus,[1,-1]):
                menu=menus[menu_name]
                cachekey=(panel,cap,tuple(sorted(set(abs(p) for p in menu))),direction if cap>1 else 1)
                if cachekey not in cache:
                    score,mu,labels=optimize(ox,cap,menu,direction,True)
                    ns=optimize(nx,cap,menu,direction)
                    cache[cachekey]=(float(score[0]),float(mu[0]),labels[0],ns)
                obs,mu,labels,ns=cache[cachekey]
                count=int(np.count_nonzero(ns<=obs+1e-13))
                name=f'{panel}:M{cap}:{menu_name}:d{direction}'
                row={'family':name,'integer_cap':cap,'exponent_menu':menu,'direction':direction,
                     'nominal_label_tuples':int((cap*len(menu))**4),
                     'unique_offset_tuples_upper_bound':int(len(choice_offsets(cap,menu,direction)[0])**4),
                     'observed_rms_log_residual':obs,'fitted_shared_scale':float(np.exp(mu)),
                     'negative_control_selected_labels':labels,'count_null_at_least_as_close':count,
                     'draws':N,'add_one_fraction':(count+1)/(N+1),
                     'null_score_quantiles':dict(zip(['q01','q05','q50','q95'],map(float,np.quantile(ns,[.01,.05,.5,.95]))))}
                rows.append(row)
                omnibus_null=np.minimum(omnibus_null,ns)
                if obs<omnibus_obs-1e-13:omnibus_obs=obs;omnibus_name=name
        count=int(np.count_nonzero(omnibus_null<=omnibus_obs+1e-13))
        results.append({'support':[low,high],'rows':rows,'omnibus':{'selection':'min over both normalization panels, every cap/menu, and harmonic/subharmonic direction','observed_score':omnibus_obs,'selected_family':omnibus_name,'null_count':count,'draws':N,'add_one_fraction':(count+1)/(N+1),'monte_carlo_standard_error':float(np.sqrt((count/N)*(1-count/N)/N)),
            'nominal_model_labels_upper_bound':sum(r['nominal_label_tuples'] for r in rows),
            'independent_trials_count':None,'independent_trials_reason':'Nested caps, reciprocal duplicates, common-scale degeneracy and correlated choices; Monte Carlo repeats the whole selection instead.'}})
    return {'schema':'WCT_COINCIDENCE_DIAGNOSTIC_V1','seed':20261007,'generator':'numpy PCG64','draws_per_support':N,
        'target_ids':ids,'primary_anchor_panel_status':'INELIGIBLE_TYPE_MISMATCH','status':'NEGATIVE_CONTROL_ONLY_NOT_SCIENTIFIC_P_VALUE',
        'score':'RMS of log ratios to fitted shared scale after selected integer and coordinate maps',
        'normalization_note':'Changing physical units shifts log phase; a shared frequency multiplier cancels in the fitted log residual. Independently fitted domain multipliers would fit every quartet exactly and are not fitted here.',
        'unmodeled_historical_selection':'Experiment, pipeline, peak/mode and source choices; support is arbitrary, not a physical data-generating null. No aggregate significance claimed.',
        'results':results}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--skip-nulls',action='store_true');args=parser.parse_args()
    j=json.loads((HERE/'cross_domain_targets.json').read_text());targets={r['id']:r for r in j['records']}
    cv=[]
    for gi,ni,li in itertools.product(['GWTC_V1','GWTC_V3','GWTC_V4'],['NIST_FE160','NIST_CO160'],['LHCB_K1','LHCB_K2REF','LHCB_K2BEST']):
        ids=[gi,'CMS',ni,li]
        panel='PRIMARY_REFERENCE' if ids==j['supplementary_reference_ids'] else 'SOURCE_CHOICE_SENSITIVITY'
        cv.extend(fit_and_loo(ids,targets,factors,model,panel) for model,factors in [('U0',[1,1,1,1]),('U1',[1,1,1,2])])
    reasons={'U0':'LHCb a is not a log frequency; comparing it to frequencies has no typed map.',
             'U1':'No source-derived map sends LHCb interregional slope a into a mass/energy DSI period.',
             'U2':'No independently fixed physical widths and integer sectors; held-out sector cannot be inferred from its target.',
             'U3':'S004 spatial branch does not supply system backgrounds, physical domain or observable response map.',
             'U4':'Nonlinear domain solution, sector selection and all observable maps remain incomplete.'}
    blocked=[{'model':m,'status':'INELIGIBLE_OR_NOT_IDENTIFIABLE','folds':[{'held_out':id,'training_ids':[x for x in j['primary_target_ids'] if x!=id],'predicted':None,'residual':None,'reason':reason} for id in j['primary_target_ids']]} for m,reason in reasons.items()]
    save('cross_validation_results.json',{'schema':'WCT_CROSS_VALIDATION_V1','precision_digits':80,'objective':'Unweighted sum of squared log ratios; shared scale is geometric mean of three training values in each fold.',
         'retrospective':True,'no_prospective_claim':True,'primary_anchor_panel':blocked,'supplementary_frequency_panels':cv,
         'fitted_harmonic_models_LOO':'INELIGIBLE: labels chosen from all four values would leak held-out information; no such LOO computed.',
         'all_frequency_normalized_residuals_unavailable':True})
    save('derived_quantities.json',derived(targets))
    if not args.skip_nulls:save('null_diagnostics.json',nulls(targets))
    save('execution_environment.json',{'python':platform.python_version(),'numpy':np.__version__,'mpmath':mp.__version__,'precision_digits':80,'summary_only':True,'empirical_peak_searches_run':0,'ATLAS_holdout_access':False})
    print(f'Wrote {len(cv)} complete four-fold fits ({len(cv)*4} predictions), 20 blocked primary folds, and derived quantities.')
    if not args.skip_nulls:print('Wrote all null diagnostics with both prespecified supports.')

if __name__=='__main__':main()
