#!/usr/bin/env python3
"""Verify exact source bytes and independently reconstruct frozen conventions."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'derivations'))
from d4_binding import BETA_HEX,MASS_HEX


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    ledger=json.loads((HERE/'source_ledger.json').read_text());checked=[]
    for row in ledger['d4_replay_inputs']+ledger['reviewed_supporting_sources']:
        data=(args.source_root/row['path']).read_bytes()
        digest=hashlib.sha256(data).hexdigest();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        assert digest==row['sha256'] and blob==row['git_blob_sha1'],row['path']
        checked.append({'path':row['path'],'sha256':digest,'git_blob_sha1':blob})
    source=(args.source_root/'src/wct_pde/projection_free.py').read_text()
    module=ast.parse(source);names={'mass','rescale_initial_mass','hamiltonian'}
    functions=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(functions)==3
    env={'np':np,'ComplexArray':np.ndarray,'ProjectionFreeConfig':object,'finite_band_action':lambda f,c:2*f}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<pinned-source-functions>','exec'),env)
    config=SimpleNamespace(domain=SimpleNamespace(cell_area=.125),model=SimpleNamespace(beta=float.fromhex(BETA_HEX),gamma=float.fromhex(BETA_HEX)))
    field=np.array([[1+2j,3-1j],[.5j,-1+0j]])
    expected=float(np.sum(abs(field)**2)*.125)
    assert env['mass'](field,config)==expected
    normalized=env['rescale_initial_mass'](field,float.fromhex(MASS_HEX),config)
    assert abs(env['mass'](normalized,config)-float.fromhex(MASS_HEX))<1e-14
    density=abs(field)**2;expected_h=float(np.sum(4*density-config.model.beta*density**2/2+config.model.gamma*density**3/3)*.125)
    assert abs(env['hamiltonian'](field,config)-expected_h)<1e-11
    existence=args.source_root/'docs/evidence/d4_independent_closure/d4_independent_existence_v2/certificate.json'
    root=json.loads(existence.read_text());pars=root['parameters']
    assert pars['beta']==pars['gamma']==BETA_HEX and pars['mass']==MASS_HEX
    validator=(args.source_root/'tools/closure_d4_independent_existence.py').read_text()
    diagonal=next(n for n in ast.walk(ast.parse(validator)) if isinstance(n,ast.FunctionDef) and n.name=='diagonal')
    norm=ast.dump(diagonal.body[0].value,include_attributes=False)
    target=ast.dump(ast.parse('(scale*(i*i+j*j)-1)**2-omega',mode='eval').body,include_attributes=False)
    assert norm==target
    assert '(1764*sum((v*v for v in candidate.values()), arb(0))-mass)/2' in validator
    root_validator_hash=next(r['sha256'] for r in checked if r['path']=='tools/closure_d4_independent_existence.py')
    assert root_validator_hash==root['input_sha256']['tools/closure_d4_independent_existence.py']
    audit=(args.source_root/'docs/D4_INDEPENDENT_EXISTENCE_AUDIT.md').read_text()
    assert 'i u_t=(Delta+1)^2u-beta|u|^2u+gamma|u|^4u' in audit
    result={'schema':'d4-source-conventions-v1','status':'PASS','checked_files':checked,
      'squared_integral_mass_verified':True,'square_root_amplitude_verified':True,'energy_coefficients_verified':True,
      'root_parameters':pars,'k_squared_from_actual_root_diagonal':1,'torus_integral_factor':1764,
      'half_mass_residual_is_equation_scaling_not_norm_redefinition':True,
      'supporting_document_original_hash':root['input_sha256']['docs/D4_INDEPENDENT_EXISTENCE_AUDIT.md'],
      'supporting_document_pinned_revision_hash':next(r['sha256'] for r in checked if r['path']=='docs/D4_INDEPENDENT_EXISTENCE_AUDIT.md'),
      'original_interval_campaign_rerun':False}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: 5 replay inputs and 2 supporting sources; exact squared integral norm, coefficients and actual k=1 operator')


if __name__=='__main__':main()
